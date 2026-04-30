"""
intro.py — Build a short intro beat that precedes the recap.

The intro:
  - Speaks ONE archetypal sentence (~15-20 words) — sets tone, no spoilers.
  - Shows 2-3 establishing visuals picked from the first 20% of source scenes.
  - Is prepended to the clustered beat list before TTS so it goes through the
    normal narrate/TTS/render pipeline as beat 1.
"""
from __future__ import annotations

import json
import re
from narrate import _http_post  # type: ignore


_INTRO_NARRATION_SYSTEM = """\
You write the OPENING narration line for a movie recap video — exactly ONE sentence.

HARD REQUIREMENTS (failure on any = rejection)
- Length: 15-20 words. Count before responding. Do NOT exceed 20 words.
- ZERO character names. Use only archetypal nouns: "survivors", "the colony", \
"the wasteland", "the outsider", "darkness". Names like Briggs, Sam, Mason are \
forbidden.
- ZERO spoilers: never reveal the climax, deaths, twists, betrayals, who turns on \
whom, or who survives. Speak only about the SETUP and the CENTRAL THREAT.
- Tone: trailer voiceover — concrete, declarative, present tense.
- NO metaphors, NO "we follow / we see", NO question marks, NO "as" clauses.

Return ONLY the sentence text. No quotes, no labels, no preamble.
"""


_INTRO_PICK_SYSTEM = """\
Pick scenes for the OPENING montage of a movie recap. Goal: introduce the world's setting \
and atmosphere without revealing plot.

CRITERIA — prefer scenes that:
- Show environment, location, weather, architecture, or wide establishing shots.
- Convey atmosphere or mood.

AVOID scenes that:
- Reveal central conflicts, deaths, climaxes, romantic moments, or major character decisions.
- Show only one character's face in close-up with strong emotion.
- Contain credits, titles, logos, or text-only frames.

Return ONLY JSON: {"indices": [<integers>]}
"""


def _word_count(text: str) -> int:
    return len(text.split())


def generate_intro_narration(story_context: str, api_key: str) -> str:
    """Generate a single archetypal opening sentence from the story synopsis."""
    if not story_context:
        story_context = "A group of people face danger in a hostile world and must decide how far they will go to survive."
    user = (
        "Story context (do NOT copy phrasing — paraphrase archetypally):\n"
        f"{story_context.strip()[:4000]}\n\n"
        "Write the opening narration sentence."
    )
    raw = _http_post(
        api_key,
        messages=[
            {"role": "system", "content": _INTRO_NARRATION_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=120,
        temperature=0.6,
        timeout=60,
        thinking_enabled=False,
    )
    line = raw.strip().strip('"').strip("'").strip()
    # Take only the first sentence if model emitted more.
    line = re.split(r"(?<=[.!?])\s+", line)[0].strip()

    # Soft length guard: if way too long, truncate at last word boundary <= 25 words.
    words = line.split()
    if len(words) > 25:
        line = " ".join(words[:25]).rstrip(",;:") + "."
    if not line.endswith(("." , "!", "?")):
        line += "."
    return line


def pick_intro_scenes(
    source_scenes: list[dict], api_key: str, count: int = 3, pool_pct: float = 0.20,
) -> list[dict]:
    """Ask the LLM to pick `count` establishing scenes from the first ``pool_pct`` of source scenes."""
    if not source_scenes:
        return []
    pool_size = max(8, min(len(source_scenes), int(len(source_scenes) * pool_pct)))
    pool = source_scenes[:pool_size]

    lines = []
    for i, s in enumerate(pool):
        desc = (s.get("povText") or s.get("description") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] window={s.get('window', '?')} {desc[:280]}")

    user = (
        f"Pick {count} scenes from this pool that work as opening establishing visuals:\n\n"
        + "\n".join(lines)
        + f"\n\nReturn JSON of the form {{\"indices\": [<{count} integers>]}}."
    )

    try:
        raw = _http_post(
            api_key,
            messages=[
                {"role": "system", "content": _INTRO_PICK_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=120,
            temperature=0.3,
            timeout=60,
            thinking_enabled=False,
            json_mode=True,
        )
        data = json.loads(raw)
        idxs = data.get("indices", [])
        chosen = []
        seen = set()
        for x in idxs:
            xi = int(x)
            if 0 <= xi < len(pool) and xi not in seen:
                seen.add(xi)
                chosen.append(pool[xi])
            if len(chosen) >= count:
                break
        if not chosen:
            raise ValueError("no valid indices")
        return chosen
    except Exception as e:  # noqa: BLE001
        print(f"  [intro] scene pick failed ({e}); falling back to first {count} pool scenes")
        return pool[:count]


def build_intro_beat(
    source_scenes: list[dict],
    story_context: str,
    api_key: str,
    fps: int,
    scene_count: int = 3,
) -> dict:
    """Generate narration + pick scenes + assemble a beat dict ready to prepend.

    Display frames here are an estimate (3 wps + 0.4s padding). The post-TTS
    sync step recomputes them from the actual audio file, and align.py replaces
    the `segments` array with STT-aligned timing.
    """
    narration = generate_intro_narration(story_context, api_key)
    intro_scenes = pick_intro_scenes(source_scenes, api_key, count=scene_count)

    if not intro_scenes:
        # Without source scenes there is nothing to render — caller should drop the intro.
        return {}

    word_count = _word_count(narration)
    estimated_secs = max(4.0, word_count / 2.7) + 0.4
    display_frames = max(int(estimated_secs * fps), len(intro_scenes) * fps)

    per_scene_frames = max(1, display_frames // len(intro_scenes))
    remainder = display_frames - per_scene_frames * len(intro_scenes)

    embedded_scenes: list[dict] = []
    segments: list[dict] = []
    for i, s in enumerate(intro_scenes):
        scene_copy = dict(s)
        frames = per_scene_frames + (remainder if i == 0 else 0)
        scene_copy["displayFrames"] = frames
        embedded_scenes.append(scene_copy)
        segments.append({
            "startSec": float(s.get("startSec", 0.0)),
            "displayFrames": frames,
            "sourceWindow": int(s.get("window", -1)),
        })

    first_start = float(intro_scenes[0].get("startSec", 0.0))
    return {
        "isIntro": True,
        "startSec": first_start,
        "endSec": first_start + display_frames / fps,
        "durationInFrames": display_frames,
        "displayFrames": display_frames,
        "scenes": embedded_scenes,
        "segments": segments,
        "povText": "[INTRO] " + narration,
        "narratedText": narration,
        "ttsText": narration,
        "dialogue": "",
        "emotion": "",
        "startFmt": "00:00:00",
    }
