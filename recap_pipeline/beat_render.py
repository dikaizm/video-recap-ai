"""
beat_render.py — Convert beat-based storyboard back to per-scene format for Remotion.
"""

import json
import os


def unroll_beats(beat_storyboard: dict, voiceover_dir: str | None = None) -> dict:
    """
    Unroll clustered beats back into individual scenes for Remotion rendering.
    
    Each scene within a beat gets:
    - Proportional displayFrames within the beat
    - The beat's voiceover for narration (all scenes in beat share the same audio)
    """
    beats = beat_storyboard["scenes"]
    unrolled_scenes: list[dict] = []
    voiceover_map: dict[int, str] = {}  # beat index -> voiceover path
    
    for beat_idx, beat in enumerate(beats):
        beat_frames = beat["displayFrames"]
        member_scenes = beat.get("scenes", [beat])  # fallback to self if no children
        total_source_frames = sum(s.get("durationInFrames", 1) for s in member_scenes)
        voiceover_path = None
        
        if voiceover_dir:
            candidate = os.path.join(voiceover_dir, f"scene_{beat_idx + 1:02d}.mp3")
            if os.path.exists(candidate):
                voiceover_path = f"voiceover/scene_{beat_idx + 1:02d}.mp3"
        
        for scene in member_scenes:
            scene_source = scene.get("durationInFrames", 1)
            proportion = scene_source / total_source_frames if total_source_frames > 0 else 0
            scene_display = max(1, round(beat_frames * proportion))
            
            unrolled = {
                "window": scene.get("window", beat_idx + 1),
                "startSec": scene.get("startSec", beat["startSec"]),
                "endSec": scene.get("endSec", beat["endSec"]),
                "durationInFrames": scene_source,
                "displayFrames": scene_display,
                "povText": scene.get("povText", ""),
                "dialogue": scene.get("dialogue", ""),
                "startFmt": scene.get("startFmt", ""),
                "narratedText": beat.get("narratedText", ""),
                "emotion": beat.get("emotion", ""),
            }
            # Only first scene in beat gets voiceover (avoids duplicate audio)
            if voiceover_path and scene is member_scenes[0]:
                unrolled["voiceoverPath"] = voiceover_path
            unrolled_scenes.append(unrolled)
    
    return {
        **{k: v for k, v in beat_storyboard.items() if k != "scenes"},
        "scenes": unrolled_scenes,
    }


# ── CLI for standalone testing ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unroll beats to per-scene storyboard")
    parser.add_argument("--input", required=True, help="Path to clustered storyboard JSON")
    parser.add_argument("--output", required=True, help="Output path for unrolled storyboard")
    parser.add_argument("--voiceover-dir", default=None, help="Voiceover directory")
    args = parser.parse_args()
    
    with open(args.input) as f:
        data = json.load(f)
    
    unrolled = unroll_beats(data, voiceover_dir=args.voiceover_dir)
    
    with open(args.output, "w") as f:
        json.dump(unrolled, f, indent=2)
    
    total_display = sum(s["displayFrames"] for s in unrolled["scenes"])
    print(f"[unroll] {len(data['scenes'])} beats → {len(unrolled['scenes'])} scenes, "
          f"{total_display/30:.1f}s total display")
