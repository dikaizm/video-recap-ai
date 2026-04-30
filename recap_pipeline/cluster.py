"""
cluster.py — Group storyboard scenes into narrative beats for recap pacing.

Each beat combines 2-5 consecutive scenes into one unit with enough video time
(8-14s) to accommodate a single spoken narration line.
"""

import json
import os
import re


def _scene_text(scene: dict) -> str:
    return scene.get("povText") or scene.get("narratedText") or scene.get("description", "")


def _word_count(text: str) -> int:
    return len(text.split())


def cluster_scenes(scenes: list[dict], target_duration: float, fps: int = 30,
                   min_beat_sec: float = 4.0, max_beat_sec: float = 8.0) -> list[dict]:
    """
    Group consecutive scenes into beats of 4-8s each.
    Each beat combines a small number of scenes (typically 2-3) to leave room
    for a single narration line to cover them without compressing the story.
    """
    if not scenes:
        return []
    
    total_frames = sum(s.get("displayFrames", 1) for s in scenes)
    
    beats: list[dict] = []
    current: list[dict] = []
    current_frames = 0
    
    for scene in scenes:
        display_frames = scene.get("displayFrames", 1)
        
        # Flush current beat if adding this scene would exceed max_beat_sec
        if current and (current_frames + display_frames) / fps > max_beat_sec:
            beats.append(_build_beat(current, fps))
            current = []
            current_frames = 0
        
        current.append(scene)
        current_frames += display_frames
    
    # Flush remaining
    if current:
        beats.append(_build_beat(current, fps))
    
    _print_stats(beats, fps)
    return beats


def _build_beat(scenes: list[dict], fps: int) -> dict:
    first = scenes[0]
    last = scenes[-1]
    
    # Combine POV text with scene markers
    combined_pov = "\n".join(
        f"[Scene {s['window']:02d}] {_scene_text(s)}"
        for s in scenes
    )
    
    # Combine dialogue
    dialogues = []
    for s in scenes:
        d = s.get("dialogue", "").strip()
        if d:
            dialogues.append(d)
    combined_dialogue = " ".join(dialogues) if dialogues else ""
    
    # Dominant emotion from scenes
    emotions = [s.get("emotion", "") for s in scenes if s.get("emotion")]
    dominant_emotion = max(set(emotions), key=emotions.count) if emotions else ""
    
    return {
        "startSec": first["startSec"],
        "endSec": last["endSec"],
        "durationInFrames": sum(s.get("durationInFrames", 1) for s in scenes),
        "displayFrames": sum(s.get("displayFrames", 1) for s in scenes),
        "scenes": scenes,
        "povText": combined_pov,
        "dialogue": combined_dialogue,
        "emotion": dominant_emotion,
    }


def _merge_short_beats(beats: list[dict], fps: int, min_sec: float) -> list[dict]:
    """Merge beats shorter than min_sec with their neighbors."""
    if len(beats) <= 1:
        return beats
    
    i = 0
    while i < len(beats):
        beat = beats[i]
        dur = beat["displayFrames"] / fps
        if dur < min_sec / 2 and i > 0:
            # Merge with previous beat
            prev = beats[i - 1]
            merged = _merge_two_beats(prev, beat, fps)
            beats[i - 1:i + 1] = [merged]
            continue
        elif dur < min_sec / 2 and i < len(beats) - 1:
            # Merge with next beat
            nxt = beats[i + 1]
            merged = _merge_two_beats(beat, nxt, fps)
            beats[i:i + 2] = [merged]
            continue
        i += 1
    
    return beats


def _merge_two_beats(a: dict, b: dict, fps: int) -> dict:
    all_scenes = a["scenes"] + b["scenes"]
    return _build_beat(all_scenes, fps)


def _print_stats(beats: list[dict], fps: int):
    total_frames = sum(b["displayFrames"] for b in beats)
    total_secs = total_frames / fps
    total_scenes = sum(len(b["scenes"]) for b in beats)
    avg_dur = total_secs / len(beats) if beats else 0
    scenes_per_beat = total_scenes / len(beats) if beats else 0
    
    print(f"[cluster] → {len(beats)} beats, {total_scenes} scenes, {total_secs:.1f}s recap")
    print(f"[cluster]   avg {avg_dur:.1f}s/beat, {scenes_per_beat:.1f} scenes/beat")


# ── CLI for standalone testing ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster storyboard scenes into beats")
    parser.add_argument("--storyboard", required=True, help="Path to storyboard.json")
    parser.add_argument("--output", default="storyboard_clustered.json")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-beat", type=float, default=8.0)
    parser.add_argument("--max-beat", type=float, default=14.0)
    args = parser.parse_args()
    
    with open(args.storyboard) as f:
        data = json.load(f)
    
    clustered = cluster_scenes(
        data["scenes"],
        target_duration=sum(s["displayFrames"] for s in data["scenes"]) / args.fps,
        fps=args.fps,
        min_beat_sec=args.min_beat,
        max_beat_sec=args.max_beat,
    )
    
    output = {**data, "scenes": clustered}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"[cluster] wrote {len(clustered)} beats → {args.output}")
