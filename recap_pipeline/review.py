"""
review.py — Phase 2 of agent loop: re-analyze recap video with VLM.

Extracts one frame per beat from the recap video timeline, sends each to Ollama
VLM for a visual description, then writes review_analysis.json.

Each entry in review_analysis corresponds to one beat in the recap and contains
a VLM description of what the recap *actually shows* at that beat.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time

import av
from PIL import Image

# Reuse the Ollama client from analyze.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ollama_generate  # type: ignore

REVIEW_SYSTEM_PROMPT = """\
You are a video analyst. Look at this frame from a movie recap video.
Describe ONLY what you can directly observe. Do NOT invent details.

Output a JSON object with this exact structure:
{
  "location": "interior or exterior, describe the setting",
  "characters": "who is visible — clothing, build, visible features. Use 'person'/'figure' when gender is uncertain.",
  "action": "what is literally happening in the frame",
  "key_objects": "specific objects visible",
  "mood": "visual atmosphere — lighting, color, composition",
  "emotion": "dominant human emotion — fear, tension, relief, anger, etc."
}

Respond ONLY with the JSON object. No markdown, no explanations.\
"""


def _encode_frame(frame: av.VideoFrame) -> str:
    """Convert AV frame to base64 PNG string."""
    img = frame.to_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _timeline_of_beats(beats: list[dict], fps: int) -> list[tuple[float, float]]:
    """Compute (start_sec, end_sec) of each beat in the recap video timeline."""
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for b in beats:
        dur = b.get("displayFrames", 30) / fps
        spans.append((cursor, cursor + dur))
        cursor += dur
    return spans


def review_recap(
    recap_path: str,
    storyboard_path: str,
    output_path: str,
    ollama_model: str = "gemma4:e2b",
    ollama_host: str = "http://localhost:11434",
    decode_height: int = 480,
    parallel: int = 4,
) -> dict:
    """Extract one frame per beat from the recap video, describe each with VLM.

    Returns a dict with key `beats` — one entry per beat containing the VLM
    description.
    """
    with open(storyboard_path) as f:
        storyboard = json.load(f)

    beats = storyboard.get("scenes", [])
    if not beats:
        raise ValueError("Storyboard has no scenes/beats")

    fps = storyboard.get("fps", 30)
    spans = _timeline_of_beats(beats, fps)
    total_beats = len(beats)

    print(f"[review] {total_beats} beats, extracting frames from {recap_path}")
    started = time.time()

    # Extract one frame per beat (at midpoint of beat in recap timeline)
    container = av.open(recap_path)
    video_stream = container.streams.video[0]
    stream_fps = float(video_stream.average_rate)

    frames_b64: list[str | None] = [None] * total_beats

    # Build a map of target PTS → beat index (pts = presentation timestamp)
    time_base = float(video_stream.time_base)
    target_pts_map: dict[int, int] = {}
    for i, (start, end) in enumerate(spans):
        mid = (start + end) / 2.0
        target_pts = int(mid / time_base)
        target_pts_map[target_pts] = i

    # Single pass through the video
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            pts = frame.pts
            # Find the closest target PTS
            for tgt, idx in list(target_pts_map.items()):
                if abs(pts - tgt) <= int(0.5 / time_base):  # within 0.5s
                    if decode_height and decode_height > 0:
                        img = frame.to_image()
                        w, h = img.size
                        new_h = decode_height
                        new_w = int(w * new_h / h)
                        img = img.resize((new_w, new_h), Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        frames_b64[idx] = base64.b64encode(buf.getvalue()).decode()
                    else:
                        frames_b64[idx] = _encode_frame(frame)
                    del target_pts_map[tgt]
                    break
            if not target_pts_map:
                break

    container.close()

    missing = sum(1 for f in frames_b64 if f is None)
    if missing > 0:
        print(f"[review] {missing}/{total_beats} frames could not be extracted — will use nearest fallback")

    # For any missing frames, use nearest available
    for i in range(total_beats):
        if frames_b64[i] is not None:
            continue
        for j in range(1, max(i + 1, total_beats - i)):
            if i - j >= 0 and frames_b64[i - j] is not None:
                frames_b64[i] = frames_b64[i - j]
                break
            if i + j < total_beats and frames_b64[i + j] is not None:
                frames_b64[i] = frames_b64[i + j]
                break

    extract_elapsed = time.time() - started
    print(f"[review] extracted {total_beats - sum(1 for f in frames_b64 if f is None)}/{total_beats} frames ({extract_elapsed:.1f}s)")

    # Send each frame to VLM (parallel with ThreadPoolExecutor)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict | None] = [None] * total_beats

    def _describe_beat(idx: int) -> tuple[int, dict]:
        b64 = frames_b64[idx]
        if b64 is None:
            return idx, {"error": "no frame extracted"}
        try:
            raw = ollama_generate(
                prompt="Describe this frame from a movie recap video.",
                images=[b64],
                model=ollama_model,
                host=ollama_host,
            )
            data = json.loads(raw)
            return idx, data
        except Exception as e:
            return idx, {"error": str(e), "raw": raw if 'raw' in dir() else ""}

    print(f"[review] sending {total_beats} frames to VLM ({ollama_model})...")
    vlm_start = time.time()

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(_describe_beat, i) for i in range(total_beats)]
        for future in as_completed(futures):
            idx, data = future.result()
            results[idx] = data
            if (idx + 1) % 20 == 0 or (idx + 1) == total_beats:
                elapsed = time.time() - vlm_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  [review] {idx + 1}/{total_beats} beats ({elapsed:.0f}s, {rate:.1f} bps)")

    vlm_elapsed = time.time() - vlm_start
    print(f"[review] VLM done ({vlm_elapsed:.1f}s, {total_beats / vlm_elapsed:.1f} bps)")

    # Build review_analysis output
    review = {
        "video": recap_path,
        "fps": fps,
        "source_storyboard": storyboard_path,
        "beats": [],
    }

    for i, (beat, desc) in enumerate(zip(beats, results)):
        entry = {
            "beat_index": i,
            "window": beat.get("window", i + 1),
            "recap_start_sec": spans[i][0],
            "recap_end_sec": spans[i][1],
            "narratedText": beat.get("narratedText", ""),
            "vlm_description": desc or {},
        }
        # Include original source scene windows for reference
        entry["source_windows"] = [
            s.get("window") for s in beat.get("scenes", []) if "window" in s
        ]
        review["beats"].append(entry)

    with open(output_path, "w") as f:
        json.dump(review, f, indent=2)

    total_elapsed = time.time() - started
    print(f"[review] done → {output_path} ({total_elapsed:.1f}s total)")
    return review


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recap", required=True, help="Path to recap.mp4")
    parser.add_argument("--storyboard", required=True, help="Path to storyboard_narrated.json")
    parser.add_argument("--output", required=True, help="Path to output review_analysis.json")
    parser.add_argument("--ollama-model", default="gemma4:e2b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--decode-height", type=int, default=480)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    review_recap(
        recap_path=args.recap,
        storyboard_path=args.storyboard,
        output_path=args.output,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        decode_height=args.decode_height,
        parallel=args.parallel,
    )
