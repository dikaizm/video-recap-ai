"""
narrate.py — Two-pass narration: rank scene importance, then write calibrated voiceover.

Pass 1 (batch): Score all scenes 1–5 for narrative importance in one API call.
Pass 2 (per-scene): Write narration with word count scaled to importance + display duration.

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
import sys
import time

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Word count targets per importance score.
# Always narrate — no silence. Lower scores get fewer words so visuals can breathe.
SCORE_WORD_TARGETS = {
    5: (20, 30),   # climax / key revelation — full narration
    4: (15, 22),   # significant story beat
    3: (12, 18),   # moderate — advances story but not pivotal
    2: (8, 12),    # transition / establishing — brief observation
    1: (6, 10),    # long atmospheric filler — one plain sentence
}

RANK_SYSTEM_PROMPT = """\
Score each movie scene 1–5 for narrative importance in a recap voiceover.

5 = Climax, revelation, confrontation, or major character decision
4 = Significant — clearly advances plot or reveals key character information
3 = Moderate — sets up later events or shows character state
2 = Transition — low new information, bridges other scenes
1 = Atmospheric filler — long establishing shot, little story content

Factor in the scene's position in the film (earlier scenes score lower than the same
content near the climax).

Return ONLY a JSON array of integers, one score per scene in the order given, nothing else.
Example for 4 scenes: [3,5,2,4]\
"""

NARRATE_SYSTEM_PROMPT = """\
You are writing spoken voiceover lines for a movie recap video.

Each line will be read aloud by a text-to-speech voice. Write the way a narrator speaks — \
not the way a film critic writes.

You will receive:
- Scene description (from an AI vision model — may be incomplete)
- Optional: dialogue snippets (from speech recognition — may be fragmented)
- Optional: scene emotion (dominant feeling in the scene)
- Importance score (1–5) and a target word count to guide narration density
- Previous narration lines for continuity

Your job: write narration that hits the target word count as closely as possible.
For low-importance scenes, keep it brief — one plain sentence stating what is happening.
For high-importance scenes, go deeper — what is at stake, what changes, what is felt.

HARD RULES — no exceptions:
- Hit the target word count (±3 words)
- No metaphors, poetic comparisons, or symbolic language
- No abstract nouns as emotion carriers: "weight of", "ghost of", "cold witness", "thin comfort"
- No "as [clause]" constructions that stack two ideas into one sentence
- No "we watch / we see / we follow / we witness"
- No filler phrases: "in this scene", "we now see", "the camera shows"
- Vary the sentence opening every line — never the same grammatical structure twice in a row
- If the visual description is vague, write the most likely plain story beat from context

Return ONLY the narration text. No labels, no quotes.\
"""


def _http_post(api_key: str, messages: list[dict], max_tokens: int = 120,
               temperature: float = 0.4, retries: int = 3) -> str:
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

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
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"].strip()
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
        max_tokens=max(128, len(scenes) * 4),  # compact array: ~3 chars per score
        temperature=0.1,
    )

    print(f"  [rank] response: {raw[:120]}", flush=True)
    try:
        # Strip markdown fences and any surrounding text; find the JSON array
        cleaned = raw.strip()
        start = cleaned.index("[")
        end = cleaned.rindex("]") + 1
        scores_list = json.loads(cleaned[start:end])

        if len(scores_list) != len(scenes):
            raise ValueError(f"got {len(scores_list)} scores for {len(scenes)} scenes")

        return {
            scenes[i]["window"]: max(1, min(5, int(scores_list[i])))
            for i in range(len(scenes))
        }
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"  [warn] ranking parse failed ({e}), defaulting all to score 3", flush=True)
        return {s["window"]: 3 for s in scenes}


def _word_target(score: int, display_secs: float) -> tuple[int, int]:
    """Return (min_words, max_words) based on importance score and available display time."""
    base_min, base_max = SCORE_WORD_TARGETS.get(score, (12, 18))
    # Cap max by speaking pace (~2.3 words/sec) so narration never overruns the clip
    time_cap = max(base_min, int(display_secs * 2.3))
    return base_min, min(base_max, time_cap)


def build_scene_prompt(
    scene: dict,
    analysis_scene: dict | None,
    score: int,
    display_secs: float,
) -> str:
    pov = scene.get("povText", "").strip()
    dialogue = scene.get("dialogue", "").strip()
    emotion = scene.get("emotion", "").strip()

    if analysis_scene:
        full_dialogue = analysis_scene.get("dialogue", "").strip()
        if full_dialogue and len(full_dialogue) > len(dialogue):
            dialogue = full_dialogue

    min_w, max_w = _word_target(score, display_secs)
    timestamp = scene.get("startFmt", "")

    parts = [
        f"[Scene {scene['window']}" + (f" at {timestamp}" if timestamp else "") + "]",
        f"Importance: {score}/5 — Target: {min_w}–{max_w} words",
    ]
    if pov:
        parts.append(f"Visual: {pov}")
    if dialogue:
        parts.append(f"Dialogue: {dialogue}")
    if emotion:
        parts.append(f"Emotion: {emotion}")

    return "\n".join(parts)


def narrate(
    storyboard_path: str,
    output_path: str,
    api_key: str,
    analysis_path: str | None = None,
    force: bool = False,
) -> dict:
    # If narrated output already exists and is complete, skip
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
    if analysis_path and os.path.exists(analysis_path):
        with open(analysis_path) as f:
            analysis = json.load(f)
        for s in analysis.get("scenes", []):
            analysis_by_window[int(s["window"])] = s

    scenes = storyboard["scenes"]
    total = len(scenes)
    narrate_start = time.time()

    # Compute total film duration from storyboard
    total_duration = max((s["endSec"] for s in scenes), default=1.0)

    # Pass 1: rank all scenes in one batch call
    scores = rank_scenes(scenes, api_key, total_duration)
    score_dist = {i: sum(1 for v in scores.values() if v == i) for i in range(1, 6)}
    print(f"[narrate] score distribution: {score_dist}", flush=True)

    # Store scores on scenes
    for scene in scenes:
        scene["importanceScore"] = scores.get(scene["window"], 3)

    # Pass 2: write narration per scene
    narration_history: list[str] = [s["narratedText"] for s in scenes if s.get("narratedText")]

    for scene in scenes:
        window = scene["window"]
        existing = scene.get("narratedText", "")
        if existing and not force:
            print(f"  skip scene {window:02d} (narration exists, score={scene['importanceScore']})")
            if existing not in narration_history:
                narration_history.append(existing)
            continue

        score = scene["importanceScore"]
        display_secs = scene.get("displayFrames", 45) / storyboard.get("fps", 30)
        prompt = build_scene_prompt(scene, analysis_by_window.get(window), score, display_secs)

        context = narration_history[-3:] if narration_history else None
        context_block = ""
        if context:
            context_block = "Previous narration (vary structure, do not echo):\n"
            context_block += "\n".join(f"- {line}" for line in context)
            context_block += "\n\n"

        print(f"  narrate scene {window:02d}/{total} (score={score})...")
        narration = _http_post(
            api_key,
            messages=[
                {"role": "system", "content": NARRATE_SYSTEM_PROMPT},
                {"role": "user", "content": context_block + prompt},
            ],
            max_tokens=80,
            temperature=0.5,
        )

        scene["narratedText"] = narration
        narration_history.append(narration)
        wc = len(narration.split())
        print(f"    score={score} words={wc} → {narration[:80]}{'...' if len(narration) > 80 else ''}")

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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("[error] DeepSeek API key required (--api-key or DEEPSEEK_API_KEY env var)", file=sys.stderr)
        sys.exit(1)

    narrate(
        storyboard_path=args.storyboard,
        output_path=args.output,
        api_key=args.api_key,
        analysis_path=args.analysis,
        force=args.force,
    )


if __name__ == "__main__":
    main()
