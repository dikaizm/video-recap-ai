"""
eval_agent.py — Phase 3 of agent loop: cross-evaluate narrated text vs recap visuals.

Takes the narrated storyboard and the VLM review of the recap, then uses DeepSeek
to evaluate each beat on:
  - Visual match: does narration match what's shown?
  - Audio timing: is audio in sync with cuts?
  - Scene selection: was the best source scene chosen?

Outputs eval_results.json with per-beat scores and fix instructions.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narrate import _http_post  # type: ignore

EVAL_SYSTEM = """\
You are a quality reviewer for a movie recap video. For each beat, compare:
  1. The narration (what the voiceover says)
  2. The recap visual description (what a VLM saw on screen during this beat)
  3. The available source scenes (description of original footage that could be shown)

Evaluate these dimensions:
  - VISUAL MATCH: Does the narration describe what's actually on screen?
  - SCENE CHOICE: Among the source scenes, was the best one picked for this narration?
  - TIMING: Does the narration length fit the beat's video duration?

Score each beat 1-5 (5 = perfect, flawless match between audio and visual).
Score 5: narration perfectly describes what's shown, best source scene chosen, timing fits
Score 4: good match, minor room for improvement
Score 3: acceptable but noticeable mismatch (narration mentions something not shown)
Score 2: clear mismatch, either narration or scene choice wrong
Score 1: completely wrong — different scene shown than what's being described

For each beat respond with JSON:
{
  "beat_index": 0,
  "score": 3,
  "visual_match": "yes" | "partial" | "no",
  "issues": ["narration mentions gunfight but scene shows people talking in a room"],
  "fix_type": "ok" | "rewrite_narration" | "swap_scene" | "adjust_timing" | "both",
  "fix_suggestion": "specific instruction for how to fix",
  "alternative_scene_window": null | 15
}

fix_type rules:
  - "ok": score 4+, no fix needed
  - "rewrite_narration": narration is wrong for what's shown
  - "swap_scene": source scene doesn't match narration but another candidate would
  - "adjust_timing": narration word count doesn't fit beat duration
  - "both": both narration and scene choice need changing
"""


def _describe_source_scene(scene: dict, max_chars: int = 300) -> str:
    """Build a compact description of a source scene for the LLM prompt."""
    parts = []
    w = scene.get("window", "?")
    desc = (scene.get("povText") or scene.get("description") or "").strip()
    dialogue = (scene.get("dialogue") or "").strip()
    parts.append(f"Scene window={w}: {desc[:max_chars]}")
    if dialogue:
        parts.append(f"  dialogue: {dialogue[:150]}")
    return "\n".join(parts)


def _build_beat_prompt(
    beat: dict,
    review_entry: dict,
    source_scenes: list[dict],
    beat_idx: int,
    total_beats: int,
) -> str:
    """Build the evaluation prompt for a single beat."""
    narrated = beat.get("narratedText", "")
    vlm_d = review_entry.get("vlm_description", {})

    vlm_text = vlm_d.get("description", "")
    if not vlm_text and isinstance(vlm_d, dict):
        # Build from structured VLM output
        fields = []
        for k in ("location", "action", "characters", "key_objects", "mood", "emotion"):
            if vlm_d.get(k):
                fields.append(f"{k}: {vlm_d[k]}")
        vlm_text = "; ".join(fields)
    if not vlm_text:
        vlm_text = "(no VLM description available)"

    candidate_text = ""
    if source_scenes:
        candidate_text = "Source scenes (the original footage this beat can draw from):\n"
        candidate_text += "\n".join(_describe_source_scene(s) for s in source_scenes[:5])
    else:
        candidate_text = "No source scene information available."

    prompt = f"""Beat {beat_idx + 1}/{total_beats}:

Narration (what the voiceover says):
"{narrated}"

Recap visual (what VLM sees on screen during this beat):
{vlm_text}

{candidate_text}

Beat duration: {beat.get('displayFrames', 0) / 30:.1f}s
Narration word count: {len(narrated.split())}

Evaluate this beat and return ONLY the JSON object.\
"""
    return prompt


def evaluate_beats(
    storyboard_path: str,
    review_path: str,
    output_path: str,
    api_key: str,
    analysis_path: str | None = None,
    batch_size: int = 5,
) -> dict:
    """Cross-evaluate every beat: narration vs recap visual vs source scenes.

    Returns eval_results dict with per-beat scores, fix_type, and fix_suggestion.
    """
    with open(storyboard_path) as f:
        storyboard = json.load(f)
    with open(review_path) as f:
        review = json.load(f)

    beats = storyboard.get("scenes", [])
    review_beats = review.get("beats", [])

    # Load full source scene pool for scene-swap suggestions
    source_by_window: dict[int, dict] = {}
    if analysis_path and os.path.exists(analysis_path):
        with open(analysis_path) as f:
            analysis = json.load(f)
        for s in analysis.get("scenes", []):
            source_by_window[s.get("window")] = s

    total = len(beats)
    print(f"[eval] evaluating {total} beats against review descriptions...")
    eval_start = time.time()

    all_results: list[dict] = []
    parse_failures = 0

    for chunk_start in range(0, total, batch_size):
        chunk_end = min(chunk_start + batch_size, total)
        chunk_beats = beats[chunk_start:chunk_end]
        chunk_reviews = review_beats[chunk_start:chunk_end]

        # Build batched messages
        messages: list[dict] = [{"role": "system", "content": EVAL_SYSTEM}]
        user_parts: list[str] = []

        for i, (beat, rv) in enumerate(zip(chunk_beats, chunk_reviews)):
            beat_idx = chunk_start + i
            # Build source scene candidates for this beat
            source_windows = set(rv.get("source_windows", []))
            source_windows.update(
                s.get("window") for s in beat.get("scenes", []) if "window" in s
            )
            source_candidates = [
                source_by_window[w]
                for w in sorted(source_windows)
                if w in source_by_window
            ]

            prompt = _build_beat_prompt(
                beat, rv, source_candidates, beat_idx, total,
            )
            user_parts.append(prompt)

        user_content = "\n---\n".join(user_parts)
        user_content += f"\n\nReturn a JSON array with {len(chunk_beats)} evaluation objects:\n"
        user_content += '{"evaluations": [{"beat_index": 0, "score": 3, ...}, ...]}'
        messages.append({"role": "user", "content": user_content})

        print(f"  [eval] beats {chunk_start + 1}–{chunk_end}/{total}...")
        try:
            raw = _http_post(
                api_key,
                messages=messages,
                max_tokens=max(1024, len(chunk_beats) * 400),
                temperature=0.2,
                timeout=120,
                thinking_enabled=False,
                json_mode=True,
            )
            data = json.loads(raw)
            chunk_results = data.get("evaluations", [])
            if not isinstance(chunk_results, list):
                raise ValueError(f"bad evaluations: {chunk_results}")
        except Exception as e:
            print(f"  [eval] parse failed for chunk {chunk_start + 1}–{chunk_end}: {e}")
            parse_failures += 1
            # Fill with unknown results — score 0 so they don't inflate ok_ratio
            chunk_results = []
            for i in range(len(chunk_beats)):
                chunk_results.append({
                    "beat_index": chunk_start + i,
                    "score": 0,
                    "visual_match": "unknown",
                    "issues": [f"evaluation parse error: {e}"],
                    "fix_type": "eval_failed",
                    "fix_suggestion": "could not evaluate",
                })

        # Re-index beat_index to absolute positions
        for r in chunk_results:
            r["beat_index"] = chunk_start + r.get("beat_index", 0)
        all_results.extend(chunk_results)

    # Sort by beat_index
    all_results.sort(key=lambda r: r.get("beat_index", 0))

    # Compute aggregate stats
    scores = [r.get("score", 0) for r in all_results]
    avg_score = sum(scores) / max(1, len(scores))
    ok_count = sum(1 for r in all_results if r.get("fix_type") == "ok" and r.get("score", 0) > 0)
    fix_counts = {
        "rewrite_narration": sum(1 for r in all_results if r.get("fix_type") == "rewrite_narration"),
        "swap_scene": sum(1 for r in all_results if r.get("fix_type") == "swap_scene"),
        "adjust_timing": sum(1 for r in all_results if r.get("fix_type") == "adjust_timing"),
        "both": sum(1 for r in all_results if r.get("fix_type") == "both"),
    }

    total_chunks = max(1, -(-total // batch_size))  # ceil div
    # If >50% of chunks failed to parse, evaluation is unreliable — don't pass
    eval_reliable = parse_failures <= total_chunks // 2
    eval_output = {
        "total_beats": total,
        "parse_failures": parse_failures,
        "eval_reliable": eval_reliable,
        "average_score": round(avg_score, 2),
        "ok_count": ok_count,
        "ok_ratio": round(ok_count / max(1, total), 2),
        "fix_counts": fix_counts,
        "passed": eval_reliable and ok_count / max(1, total) >= 0.85,
        "evaluations": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=2)

    elapsed = time.time() - eval_start
    reliability_note = "" if eval_reliable else f" ⚠ {parse_failures}/{total_chunks} chunks failed — eval unreliable"
    print(
        f"[eval] done — avg_score={avg_score:.2f}, ok={ok_count}/{total} "
        f"({ok_count / max(1, total):.0%}), parse_failures={parse_failures}/{total_chunks}"
        f"{reliability_note}, elapsed={elapsed:.1f}s"
    )
    return eval_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--storyboard", required=True, help="Path to storyboard_narrated.json")
    parser.add_argument("--review", required=True, help="Path to review_analysis.json")
    parser.add_argument("--output", required=True, help="Path for eval_results.json")
    parser.add_argument("--deepseek-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--analysis", default=None, help="Original analysis.json for source scene lookup")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    evaluate_beats(
        storyboard_path=args.storyboard,
        review_path=args.review,
        output_path=args.output,
        api_key=args.deepseek_key,
        analysis_path=args.analysis,
        batch_size=args.batch_size,
    )
