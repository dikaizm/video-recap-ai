"""
transform.py — Convert analyze.py JSON output to Remotion storyboard format.

Usage:
    python transform.py \
        --input test_analysis.json \
        --output output/storyboard.json \
        --video-filename video.mp4 \
        --fps 30 \
        --recap-ratio 0.15
"""
import argparse
import json
import os
import re

HEADER_PATTERN = re.compile(
    r"^\s*(LOCATION|CHARACTERS|ACTION|DIALOGUE|OBJECTS|MOOD|TONE|STORY|KEY|SETTING|EMOTION)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)
EMOTION_PATTERN = re.compile(r"^\s*EMOTION\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
TIMESTAMP_PREFIX = re.compile(r"^\[[\d:]+\]\s*")

MIN_DISPLAY_SECS = 1.5  # floor so no clip is too short to read


def clean_description(text: str, max_chars: int = 300) -> str:
    cleaned = HEADER_PATTERN.sub("", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    result = ""
    for sentence in sentences:
        if len(result) + len(sentence) + 1 > max_chars:
            break
        result = (result + " " + sentence).strip()
    return result or text[:max_chars].strip()


def extract_dialogue_preview(dialogue: str, max_chars: int = 120) -> str:
    if not dialogue:
        return ""
    first_line = dialogue.strip().splitlines()[0]
    stripped = TIMESTAMP_PREFIX.sub("", first_line).strip()
    return stripped[:max_chars]


def compute_display_secs(scenes: list[dict], recap_ratio: float) -> list[float]:
    """
    Distribute recap time proportionally to each scene's original duration.
    Total recap = original_duration * recap_ratio.
    Each scene gets: (scene_duration / total_duration) * total_recap_secs,
    subject to a MIN_DISPLAY_SECS floor.
    """
    durations = [max(0.0, float(s["end_sec"]) - float(s["start_sec"])) for s in scenes]
    total_duration = sum(durations) or 1.0
    total_recap_secs = total_duration * recap_ratio

    raw = [(d / total_duration) * total_recap_secs for d in durations]

    # Apply floor: scenes below MIN_DISPLAY_SECS are clamped up, others scaled down
    floored = [max(MIN_DISPLAY_SECS, t) for t in raw]
    floor_excess = sum(f - r for f, r in zip(floored, raw) if f > r)

    # Scale down scenes that are above the floor to absorb the excess
    above = [(i, t) for i, (t, r) in enumerate(zip(floored, raw)) if t == r and t > MIN_DISPLAY_SECS]
    if above and floor_excess > 0:
        above_total = sum(t for _, t in above)
        for i, t in above:
            floored[i] = t - (t / above_total) * floor_excess

    return floored


def build_scene(
    scene: dict,
    fps: int,
    display_secs: float,
    voiceover_dir: str | None = None,
) -> dict:
    start_sec = float(scene["start_sec"])
    end_sec = float(scene["end_sec"])
    window = int(scene["window"])

    source_frames = max(1, round((end_sec - start_sec) * fps))
    display_frames = max(1, round(display_secs * fps))

    voiceover_path = None
    if voiceover_dir:
        candidate = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")
        if os.path.exists(candidate):
            voiceover_path = f"voiceover/scene_{window:02d}.mp3"

    description = scene.get("description", "")
    emotion_match = EMOTION_PATTERN.search(description)
    emotion = emotion_match.group(1).strip() if emotion_match else ""

    result: dict = {
        "window": window,
        "startSec": start_sec,
        "endSec": end_sec,
        "durationInFrames": source_frames,
        "displayFrames": display_frames,
        "povText": clean_description(description),
        "dialogue": extract_dialogue_preview(scene.get("dialogue", "")),
        "startFmt": scene.get("start_fmt", ""),
    }
    if emotion:
        result["emotion"] = emotion
    if voiceover_path:
        result["voiceoverPath"] = voiceover_path
    return result


def load_analysis(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "scenes" not in data:
        raise ValueError(f"Expected 'scenes' key in {path}")
    return data


def transform(
    analysis_path: str,
    output_path: str,
    video_path: str,
    fps: int = 30,
    recap_ratio: float = 0.15,
    voiceover_dir: str | None = None,
) -> dict:
    analysis = load_analysis(analysis_path)
    all_scenes = analysis["scenes"]
    scenes_raw = [s for s in all_scenes if not s.get("is_credits", False)]
    if len(scenes_raw) < len(all_scenes):
        print(f"[transform] filtered {len(all_scenes) - len(scenes_raw)} credit/title scene(s)")

    display_secs_list = compute_display_secs(scenes_raw, recap_ratio)

    total_original = sum(
        max(0.0, float(s["end_sec"]) - float(s["start_sec"])) for s in scenes_raw
    )
    total_recap = sum(display_secs_list)
    print(
        f"[transform] original={total_original:.1f}s  recap={total_recap:.1f}s  "
        f"ratio={total_recap/total_original:.1%}"
    )

    storyboard = {
        "videoPath": "video.mp4",
        "fps": fps,
        "recapRatio": recap_ratio,
        "metadata": {
            "vlm_model": analysis.get("vlm_model"),
            "device": analysis.get("device"),
            "transcriber": analysis.get("transcriber"),
            "whisper_model": analysis.get("whisper_model"),
            "analysis_processing_sec": analysis.get("processing_sec"),
        },
        "scenes": [
            build_scene(s, fps, d, voiceover_dir)
            for s, d in zip(scenes_raw, display_secs_list)
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(storyboard, f, indent=2)

    print(f"[transform] wrote {len(storyboard['scenes'])} scenes → {output_path}")
    return storyboard


def main():
    parser = argparse.ArgumentParser(description="Transform analyze.py output to Remotion storyboard")
    parser.add_argument("--input", required=True, help="Path to analyze.py JSON output")
    parser.add_argument("--output", default="output/storyboard.json", help="Output storyboard path")
    parser.add_argument("--video", required=True, help="Absolute path to the source video file")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--recap-ratio", type=float, default=0.15,
        help="Recap duration as fraction of original (default: 0.15 = 15%%)",
    )
    parser.add_argument("--voiceover-dir", default=None, help="Directory containing voiceover MP3 files")
    args = parser.parse_args()

    transform(
        analysis_path=args.input,
        output_path=args.output,
        video_path=args.video,
        fps=args.fps,
        recap_ratio=args.recap_ratio,
        voiceover_dir=args.voiceover_dir,
    )


if __name__ == "__main__":
    main()
