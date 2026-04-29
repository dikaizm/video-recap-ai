"""
narrate.py — Two-pass narration: rank scene importance, then write calibrated voiceover.

Pass 1 (batch): Score all scenes 1–5 for narrative importance in one API call.
Pass 2 (per-scene): Write narration with word count scaled to importance.

Usage:
    python narrate.py \
        --storyboard output/storyboard.json \
        --analysis output/analysis.json \
        --output output/storyboard_narrated.json \
        --api-key sk-...
"""
import argparse
import json
import os
import re as _re
import sys
import time

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# Word count targets per importance score.
# Score 5 (climax) gets room for dramatic tension and pay-off.
# Lower scores stay tight but add hook-like consequence.
SCORE_WORD_TARGETS = {
    5: (25, 40),   # climax / key revelation — full dramatic narration
    4: (18, 28),   # significant story beat — build tension
    3: (14, 20),   # moderate — advances story, adds consequence
    2: (10, 16),   # transition — brief but with a hook
    1: (10, 14),   # atmospheric filler — one expanded sentence
}

NARRATE_BATCH_SIZE = 12  # scenes per API call in batch mode

RANK_SYSTEM_PROMPT = """\
Score each movie scene 1–5 for narrative importance in a recap voiceover.

5 = Climax, revelation, confrontation, or major character decision
4 = Significant — clearly advances plot or reveals key character information
3 = Moderate — sets up later events or shows character state
2 = Transition — low new information, bridges other scenes
1 = Atmospheric filler — long establishing shot, little story content

Factor in the scene's position in the film (earlier scenes score lower than the same
content near the climax). Discovery of a crash site, alien object, or mysterious voice
should score 4–5 regardless of description quality.

Output ONLY a JSON array of integers, one score per scene in the order given.
Example for 4 scenes: [3,5,2,4]\
"""

NARRATE_SYSTEM_PROMPT = """\
You are writing spoken voiceover lines for a movie recap video. Your goal is to keep \
viewers hooked so they do not skip to the next video.

Each line will be read aloud by a text-to-speech voice. Write the way a narrator speaks — \
not the way a film critic writes.

You will receive:
- Scene position (% through film) to indicate where this falls in the story arc
- Scene description (from an AI vision model — may be incomplete)
- Optional: dialogue snippets (from speech recognition — may be fragmented)
- Optional: scene emotion (dominant feeling in the scene)
- Importance score (1–5) and a target word count to guide narration density
- Previous scenes with their visual descriptions and what was said, for continuity

NARRATIVE ARC — let the scene position shape your tone:
- 0–25% (Setup): Build mystery and curiosity. End each line with unanswered questions \
or a sense that something is off. Do not reveal too much.
- 25–60% (Rising Action): Escalate stakes. Each scene should make the situation worse. \
End with "but", "only to discover", "unaware that" — create hooks that pull to the next scene.
- 60–85% (Climax): Highest tension. Deliver the payoff. Focus on confrontation, \
revelation, and irreversible change. Short, punchy sentences.
- 85–100% (Resolution): Show cost and consequence. What did the protagonist lose or gain? \
Make the ending feel earned.

Every line must do THREE things:
1. Advance the story — show consequence, not action
2. Reveal character state — how the protagonist feels or how their situation has changed
3. End with narrative propulsion — either a hook ("but...", "only to find...") or a \
stakes escalation ("now there was no turning back")

HARD RULES — no exceptions:
- Hit the target word count (±3 words)
- NEVER repeat or closely paraphrase the visual description — always reframe in terms of story consequence, character state, or what changes
- When consecutive scenes show the same action (walking, exiting, credits rolling), each line must advance the story or shift perspective — never restate the same beat
- No metaphors, poetic comparisons, or symbolic language
- No abstract nouns as emotion carriers: "weight of", "ghost of", "cold witness", "thin comfort"
- No "as [clause]" constructions that stack two ideas into one sentence
- No "we watch / we see / we follow / we witness"
- No filler phrases: "in this scene", "we now see", "the camera shows"
- Vary the sentence opening every line — never the same grammatical structure twice in a row
- If the visual description is vague, write the most likely plain story beat from context
- The word "but" or "only to" must appear at least once every 3 lines to keep tension alive
- Every sentence must be understandable when heard — no pronouns whose referent is ambiguous

Return ONLY the narration text. No labels, no quotes.\
"""

NARRATE_BATCH_SYSTEM = """\
You are writing spoken voiceover lines for a movie recap video. Your goal is to keep \
viewers hooked so they do not skip to the next video.

You will receive a BATCH of scenes to narrate. Write one narration line per scene, in order.
Each line will be read aloud by a text-to-speech voice.

NARRATIVE ARC — let the scene position (% through film) shape your tone:
- 0–25% (Setup): Build mystery and curiosity. End with unanswered questions.
- 25–60% (Rising Action): Escalate stakes. End with "but", "only to discover", "unaware that".
- 60–85% (Climax): Highest tension. Short, punchy sentences. Irreversible change.
- 85–100% (Resolution): Cost and consequence. Earned ending.

Each line must: (1) advance the story showing consequence, (2) reveal character state, \
(3) end with a hook or stakes escalation.

HARD RULES:
- Hit the target word count for each scene (±3 words)
- NEVER repeat the visual description — reframe as story consequence or character state
- No metaphors, no abstract nouns as emotion carriers ("weight of", "echoes of")
- No "we watch / we see / we follow / we witness"
- No filler phrases: "in this scene", "the camera shows"
- Vary the sentence opening — never the same structure twice in a row
- The word "but" or "only to" must appear at least once every 3 lines

OUTPUT a JSON object with a single key "narration" containing an array of strings:
{"narration": ["scene 1 narration", "scene 2 narration", ...]}
One string per scene, in the exact order of the scenes provided. No numbering.\
"""


def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(len(wa | wb), 1)


def _parse_batch_response(raw: str, expected_count: int) -> list[str]:
    """Parse JSON batch narration response into a list of strings."""
    cleaned = _re.sub(r"<thinking>.*?</thinking>", "", raw, flags=_re.DOTALL).strip()

    # Try parsing as JSON object with "narration" key first
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "narration" in data:
            return data["narration"]
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: extract JSON array from response
    match = _re.search(r"\[.*\]", cleaned, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: split by numbered lines "1. 2. 3." or newline-separated
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    lines = [_re.sub(r"^\d+[. )]+", "", l).strip() for l in lines]
    narrations = [l for l in lines if len(l.split()) >= 3]
    if narrations:
        return narrations

    raise ValueError(f"Could not parse {expected_count} narrations from: {raw[:200]}")


def _http_post(api_key: str, messages: list[dict], max_tokens: int = 120,
               temperature: float = 0.4, retries: int = 3, timeout: int = 60,
               thinking_enabled: bool = True, json_mode: bool = False) -> str:
    import urllib.request
    import urllib.error

    body: dict = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not thinking_enabled:
        body["thinking"] = {"type": "disabled"}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    payload = json.dumps(body).encode()

    req = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"].strip()
                if not content:
                    usage = result.get("usage", {})
                    raise RuntimeError(
                        f"DeepSeek returned empty content — reasoning tokens likely exhausted max_tokens budget. "
                        f"Usage: {usage}"
                    )
                return content
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429 and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    rate-limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"DeepSeek API error {e.code}: {body}") from e

    raise RuntimeError("DeepSeek API: max retries exceeded")


def rank_scenes(scenes: list[dict], api_key: str, total_duration_sec: float) -> dict[int, int]:
    """Batch-rank all scenes in one API call. Returns {window: score}."""
    lines = []
    for s in scenes:
        pct = int(round(s["startSec"] / total_duration_sec * 100)) if total_duration_sec else 0
        dur = round(s["endSec"] - s["startSec"], 1)
        pov = s.get("povText", "")[:80]
        dialogue = s.get("dialogue", "")[:60]
        parts = [f"{s['window']}|{pct}%|{dur}s"]
        if pov:
            parts.append(pov)
        if dialogue:
            parts.append(f'"{dialogue}"')
        lines.append(" ".join(parts))

    user_content = "\n".join(lines)
    print(f"[narrate] ranking {len(scenes)} scenes...", flush=True)

    raw = _http_post(
        api_key,
        messages=[
            {"role": "system", "content": RANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max(256, len(scenes) * 8),
        temperature=0.1,
        timeout=120,
        thinking_enabled=False,
        json_mode=True,
    )

    print(f"  [rank] response ({len(raw)} chars): {raw[:120]}", flush=True)
    try:
        # Extract all integers from response — robust to wrapping, newlines, markdown, thinking blocks
        numbers = [int(x) for x in _re.findall(r"\d+", raw)]
        # If there are way more numbers than scenes (e.g. scene IDs), take the last N
        if len(numbers) >= len(scenes):
            scores_list = numbers[-len(scenes):] if len(numbers) > len(scenes) else numbers
        else:
            raise ValueError(f"got {len(numbers)} integers, need {len(scenes)}")

        if len(scores_list) != len(scenes):
            raise ValueError(f"got {len(scores_list)} scores for {len(scenes)} scenes")

        return {
            scenes[i]["window"]: max(1, min(5, int(scores_list[i])))
            for i in range(len(scenes))
        }
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"  [warn] ranking parse failed ({e}), defaulting all to score 3", flush=True)
        return {s["window"]: 3 for s in scenes}


def _word_target(score: int) -> tuple[int, int]:
    return SCORE_WORD_TARGETS.get(score, (12, 18))


def build_scene_prompt(
    scene: dict,
    analysis_scene: dict | None,
    score: int,
    consecutive_hint: str = "",
    total_duration: float = 0.0,
) -> str:
    pov = scene.get("povText", "").strip()
    dialogue = scene.get("dialogue", "").strip()
    emotion = scene.get("emotion", "").strip()

    if analysis_scene:
        full_dialogue = analysis_scene.get("dialogue", "").strip()
        if full_dialogue and len(full_dialogue) > len(dialogue):
            dialogue = full_dialogue

    min_w, max_w = _word_target(score)
    timestamp = scene.get("startFmt", "")
    pct = int(round(scene["startSec"] / total_duration * 100)) if total_duration else 0

    parts = [
        f"[Scene {scene['window']}" + (f" at {timestamp}" if timestamp else "") +
        (f" — {pct}% through film" if total_duration else "") + "]",
        f"Importance: {score}/5 — Target: {min_w}–{max_w} words",
    ]
    if pov:
        parts.append(f"Visual: {pov}")
    if dialogue:
        parts.append(f"Dialogue: {dialogue}")
    if emotion:
        parts.append(f"Emotion: {emotion}")
    if consecutive_hint:
        parts.append(consecutive_hint)

    return "\n".join(parts)


def narrate(
    storyboard_path: str,
    output_path: str,
    api_key: str,
    analysis_path: str | None = None,
    story_context: str | None = None,
    force: bool = False,
) -> dict:
    if not force and os.path.exists(output_path):
        with open(output_path) as f:
            storyboard = json.load(f)
        has_narration = all(s.get("narratedText") for s in storyboard.get("scenes", []))
        if has_narration:
            print("[narrate] existing narration found, skipping (use --narrate-force to regenerate)")
            return storyboard
    else:
        with open(storyboard_path) as f:
            storyboard = json.load(f)

    analysis_by_window: dict[int, dict] = {}
    characters_observed: str = ""
    if analysis_path and os.path.exists(analysis_path):
        with open(analysis_path) as f:
            analysis = json.load(f)
        for s in analysis.get("scenes", []):
            analysis_by_window[int(s["window"])] = s
        characters_observed = analysis.get("characters_observed", "") or analysis.get("context", "")

    scenes = storyboard["scenes"]
    total = len(scenes)
    narrate_start = time.time()

    total_duration = max((s["endSec"] for s in scenes), default=1.0)

    # Pass 1: rank all scenes in one batch call
    scores = rank_scenes(scenes, api_key, total_duration)
    score_dist = {i: sum(1 for v in scores.values() if v == i) for i in range(1, 6)}
    print(f"[narrate] score distribution: {score_dist}", flush=True)

    for scene in scenes:
        scene["importanceScore"] = scores.get(scene["window"], 3)

    # If story context is provided it is authoritative — sanitize character roster
    # by dropping VLM observations that contain gendered terms likely from dark/ambiguous frames
    if story_context and characters_observed:
        obs_list = [o for o in characters_observed.split("; ")
                    if not any(w in o.lower() for w in ("female", "woman"))]
        characters_observed = "; ".join(obs_list)

    # Build system prompt — inject character constraint and story context
    narrate_system = NARRATE_SYSTEM_PROMPT
    if characters_observed:
        narrate_system += (
            f"\n\nCHARACTER CONSTRAINT: Only refer to characters actually seen in the film."
            f" Observed characters: {characters_observed}."
            f" Do NOT invent genders, names, or people not supported by this list."
        )
        print(f"[narrate] character constraint: {characters_observed[:120]}", flush=True)
    if story_context:
        narrate_system += (
            "\n\nSTORY CONTEXT — use this to write accurate story beats when visual "
            "descriptions are vague, dark, or show only abstract patterns:\n"
            + story_context
            + "\n\nWhen the VLM description is unclear, infer the correct story beat from "
            "the context above. Never contradict it — if it names characters as men, use he/him."
        )
        print(f"[narrate] story context injected ({len(story_context)} chars)", flush=True)

    # Pass 2: batch narrate — 12 scenes per API call
    narration_history: list[tuple[str, str]] = [
        (s.get("povText", ""), s["narratedText"])
        for s in scenes if s.get("narratedText")
    ]

    pending = [
        (i, scene) for i, scene in enumerate(scenes)
        if not scene.get("narratedText") or force
    ]

    batch_size = NARRATE_BATCH_SIZE
    total_todo = len(pending)
    batch_num = 0

    # Batch system prompt with story context and character constraint
    batch_system = NARRATE_BATCH_SYSTEM
    if characters_observed:
        batch_system += (
            f"\n\nCHARACTER CONSTRAINT: Only refer to characters actually seen in the film."
            f" Observed characters: {characters_observed}."
            f" Do NOT invent genders, names, or people not supported by this list."
        )
    if story_context:
        batch_system += (
            "\n\nSTORY CONTEXT — use this to write accurate story beats when visual "
            "descriptions are vague, dark, or show only abstract patterns:\n"
            + story_context
            + "\n\nWhen the VLM description is unclear, infer the correct story beat from "
            "the context above. Never contradict it — if it names characters as men, use he/him."
        )

    for chunk_start in range(0, len(pending), batch_size):
        chunk = pending[chunk_start:chunk_start + batch_size]
        if not chunk:
            break
        batch_num += 1
        batch_scenes = [(idx, scenes[idx]) for idx, _ in chunk]

        chunk_windows = f"{chunk[0][1]['window']:02d}–{chunk[-1][1]['window']:02d}"
        print(f"\n  batch {batch_num}: scenes {chunk_windows} ({len(batch_scenes)} scenes)...")

        # Build scene list for the prompt
        scene_lines: list[str] = []
        for idx, scene in batch_scenes:
            w = scene["window"]
            sc = scene["importanceScore"]
            min_w, max_w = _word_target(sc)
            pct = int(round(scene["startSec"] / total_duration * 100)) if total_duration else 0
            pov = scene.get("povText", "").strip()
            dialogue = scene.get("dialogue", "").strip()
            emotion = scene.get("emotion", "").strip()

            line = f"[{w}] importance={sc}/5 target={min_w}–{max_w}w pos={pct}%"
            if pov:
                line += f" visual: {pov}"
            if dialogue:
                line += f" dialogue: {dialogue}"
            if emotion:
                line += f" emotion: {emotion}"
            scene_lines.append(line)

        # Build history context from last 6 narrated scenes
        history_block = ""
        if narration_history:
            recent = narration_history[-6:]
            history_block = "Previous scenes already narrated:\n"
            history_block += "\n".join(
                f"- [{pov[:55]}] → {nar}" for pov, nar in recent
            )
            history_block += "\n\n"

        user_prompt = history_block + "Scenes to narrate:\n" + "\n".join(scene_lines)
        # Each scene ~40 words → ~60 tokens; JSON overhead ~300 tokens; pad 2x for safety
        batch_max_tokens = max(512, len(batch_scenes) * 80 + 400)

        result = _http_post(
            api_key,
            messages=[
                {"role": "system", "content": batch_system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=batch_max_tokens,
            temperature=0.5,
            timeout=180,
            thinking_enabled=False,
            json_mode=True,
        )

        # Parse the JSON response
        narrations: list[str] = _parse_batch_response(result, len(batch_scenes))

        for j, (idx, scene) in enumerate(batch_scenes):
            if j < len(narrations):
                narration = narrations[j].strip()
                wc = len(narration.split())
                # Echo detection + retry for individual scenes
                pov_text = scene.get("povText", "")
                if pov_text and _word_overlap(narration, pov_text) >= 0.75:
                    print(f"    [retry] scene {scene['window']:02d} echo ({_word_overlap(narration, pov_text):.0%})...")
                    solo_prompt = build_scene_prompt(
                        scene, analysis_by_window.get(scene["window"]),
                        scene["importanceScore"],
                        total_duration=total_duration,
                    )
                    narration = _http_post(
                        api_key,
                        messages=[
                            {"role": "system", "content": narrate_system},
                            {"role": "user", "content": "Rewrite this narration to focus on story consequence, not visual description:\n" + solo_prompt},
                        ],
                        max_tokens=80,
                        temperature=0.65,
                        timeout=120,
                        thinking_enabled=False,
                    )
                    wc = len(narration.split())

                scene["narratedText"] = narration
                narration_history.append((pov_text, narration))
                print(f"    scene {scene['window']:02d} score={scene['importanceScore']} words={wc} → {narration[:80]}{'...' if len(narration) > 80 else ''}")

        # Incremental save after each batch
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(storyboard, f, indent=2)
        batch_done = min(chunk_start + batch_size, total_todo)
        print(f"  saved {batch_done}/{total_todo} scenes")

    narration_processing_sec = round(time.time() - narrate_start, 1)
    meta = storyboard.setdefault("metadata", {})
    meta["llm_model"] = DEEPSEEK_MODEL
    meta["narration_processing_sec"] = narration_processing_sec

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(storyboard, f, indent=2)

    print(f"[narrate] wrote {total} scenes → {output_path}")
    return storyboard


def main():
    parser = argparse.ArgumentParser(description="Rank + narrate scenes via DeepSeek")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--story-context", default=None, help="Story synopsis text or path to .txt/.md file")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("[error] DeepSeek API key required (--api-key or DEEPSEEK_API_KEY env var)", file=sys.stderr)
        sys.exit(1)

    story_context = args.story_context
    if story_context and os.path.isfile(story_context):
        with open(story_context) as f:
            story_context = f.read().strip()

    narrate(
        storyboard_path=args.storyboard,
        output_path=args.output,
        api_key=args.api_key,
        analysis_path=args.analysis,
        story_context=story_context,
        force=args.force,
    )


if __name__ == "__main__":
    main()
