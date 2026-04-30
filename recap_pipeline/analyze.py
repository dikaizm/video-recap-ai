"""
Video scene analyzer using Ollama VLM + speech transcription.

Pipeline:
  1. Extract audio from video → transcribe with faster-whisper or whisperx
  2. Extract video frames at a fixed interval
  3. For each analysis window, send frames + dialogue to Ollama VLM
  4. Output per-scene descriptions with timestamps and spoken dialogue

Usage:
    python analyze.py --video /path/to/video.mp4
    python analyze.py --video /path/to/video.mp4 --ollama-model gemma4:e2b
    python analyze.py --video /path/to/video.mp4 --transcriber whisperx
    python analyze.py --video /path/to/video.mp4 --no-audio
    python analyze.py --video /path/to/video.mp4 --context "Sci-fi short film..."
    python analyze.py --video /path/to/video.mp4 --interval 5 --window 8 --overlap 2
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

from dotenv import load_dotenv
load_dotenv()

import av
from PIL import Image

OLLAMA_DEFAULT_MODEL = "smolvlm:500m"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"

SYSTEM_PROMPT = """\
You are a video analyst. Look carefully at the frames provided and answer only what you can directly observe.
Do NOT invent details that are not visible.

IMPORTANT — CREDITS / TITLES OVERRIDE EVERYTHING ELSE:
The frames are END CREDITS, OPENING TITLES, or PRODUCTION LOGOS whenever ANY of the following is true:
  - rolling, scrolling, or sliding text against a dark or solid-colored background
  - lists of names (cast, crew, actors, directors, producers, "starring", "directed by")
  - production-company or distributor logos shown alone (e.g. studio idents)
  - a title card (the movie's title centered on screen with no characters acting)
  - long static text overlays where no person is performing an action
  - a static frame of text on black/dark background with no live action visible
When ANY of those is true, respond with EXACTLY:
{"is_credits": true}
Do NOT add any other field. Do NOT describe the scene further. Do NOT output the normal scene JSON.
The credits flag overrides the normal scene description below.

Otherwise (the frames show actual story content with characters or action), output a JSON object with this exact structure:
{
  "location": "interior or exterior, describe the setting from the frames",
  "characters": "who is visible — describe clothing, build, and visible features ONLY. NEVER state or infer gender unless a face is clearly visible and unambiguous. Use 'person' or 'figure' when gender is uncertain. NEVER write 'male', 'female', 'man', 'woman', 'he', 'she' unless you are certain from clear facial features.",
  "action": "what is literally happening in the frames",
  "dialogue": "only if dialogue is provided — what is said and by whom",
  "key_objects": "specific objects visible in the frames",
  "mood": "the visual atmosphere — lighting, color, composition",
  "emotion": "the dominant human emotion in this scene — fear, grief, tension, relief, anger, confusion, determination, isolation, etc. — infer from body language, expressions, and context even if faces are not clearly visible"
}

Respond ONLY with the JSON object. No markdown, no explanations."""


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def ollama_generate(
    prompt: str,
    images: list[str],  # base64-encoded PNG strings
    model: str = OLLAMA_DEFAULT_MODEL,
    host: str = OLLAMA_DEFAULT_HOST,
    retries: int = 3,
) -> str:
    # gemma4 (and most Ollama vision models) accept images as a sibling field
    # on the user message. The content-array format returns 400 for gemma4 in Ollama.
    # num_predict is omitted — gemma4 returns empty responses when it is set.
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": images},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }).encode()

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result["message"]["content"].strip()
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [ollama] connection error, retrying in {wait}s: {e}")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Ollama unreachable at {host}: {e}") from e


def check_ollama(model: str, host: str) -> None:
    """Verify Ollama is running and the model is available."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        # Match with or without tag suffix
        base = model.split(":")[0]
        if not any(n == model or n.startswith(base + ":") for n in names):
            print(f"  [warn] model '{model}' not found in Ollama. Available: {names}")
            print(f"  Run: ollama pull {model}")
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at {host} — is it running? ({e})") from e


def image_to_base64(img: Image.Image, max_size: int = 672, quality: int = 60) -> str:
    """Resize image to max_size on longest edge and encode as base64 JPEG."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, out_wav: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-vn", out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------

def transcribe_faster_whisper(
    audio_path: str, model_size: str, language: str | None
) -> list[dict]:
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

    kwargs: dict = {"beam_size": 5, "word_timestamps": True}
    if language:
        kwargs["language"] = language

    segments_iter, info = whisper.transcribe(audio_path, **kwargs)
    print(f"  Detected language : {info.language} ({info.language_probability:.2f})")

    segments = []
    for seg in segments_iter:
        words = [{"start": w.start, "end": w.end, "word": w.word}
                 for w in (seg.words or [])]
        segments.append({"start": seg.start, "end": seg.end,
                         "text": seg.text.strip(), "words": words})
    return segments


def transcribe_whisperx(
    audio_path: str, model_size: str, language: str | None
) -> list[dict]:
    import whisperx

    device = "cpu"
    compute_type = "int8"

    print(f"  Loading WhisperX model ({model_size})...")
    wx_model = whisperx.load_model(model_size, device, compute_type=compute_type,
                                   language=language)

    audio = whisperx.load_audio(audio_path)
    result = wx_model.transcribe(audio, batch_size=8)
    detected_lang = result.get("language", language or "unknown")
    print(f"  Detected language : {detected_lang}")

    print("  Running forced alignment...")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    segments = []
    for seg in result["segments"]:
        words = [{"start": w["start"], "end": w["end"], "word": w["word"]}
                 for w in seg.get("words", [])]
        segments.append({"start": seg["start"], "end": seg["end"],
                         "text": seg["text"].strip(), "words": words})
    return segments


def transcribe(
    audio_path: str, backend: str, model_size: str, language: str | None,
) -> list[dict]:
    if backend == "whisperx":
        return transcribe_whisperx(audio_path, model_size, language)
    return transcribe_faster_whisper(audio_path, model_size, language)


# ---------------------------------------------------------------------------
# Dialogue slicing
# ---------------------------------------------------------------------------

def get_dialogue_for_window(
    segments: list[dict], start_ts: float, end_ts: float
) -> str:
    lines = []
    for seg in segments:
        if seg["end"] >= start_ts and seg["start"] <= end_ts:
            ts = format_timestamp(seg["start"])
            lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def iter_frame_windows(
    video_path: str,
    interval_sec: float,
    window_size: int,
    overlap: int,
    decode_height: int | None = None,
):
    """Stream analysis windows one at a time — only window_size frames in memory at once."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    interval_frames = max(1, int(fps * interval_sec))
    step = max(1, window_size - overlap)

    # Compute reformat target — only downscale, never upscale
    target_w: int | None = None
    target_h: int | None = None
    if decode_height:
        orig_w = stream.codec_context.width or 0
        orig_h = stream.codec_context.height or 0
        if orig_h and orig_h > decode_height:
            scale = decode_height / orig_h
            target_w = int(orig_w * scale) & ~1  # keep even for yuv420p
            target_h = decode_height

    buffer: list[tuple[float, Image.Image]] = []
    window_idx = 0

    try:
        for i, frame in enumerate(container.decode(stream)):
            if i % interval_frames != 0:
                continue
            ts = float(frame.pts * stream.time_base)
            if target_w and target_h:
                frame = frame.reformat(width=target_w, height=target_h)
            buffer.append((ts, frame.to_image()))

            while len(buffer) >= window_size:
                chunk = buffer[:window_size]
                yield window_idx, chunk[0][0], chunk[-1][0], [img for _, img in chunk]
                window_idx += 1
                buffer = buffer[step:]
    finally:
        container.close()

    # Yield any remaining frames as a partial window
    if buffer:
        yield window_idx, buffer[0][0], buffer[-1][0], [img for _, img in buffer]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_user_prompt(
    start_fmt: str,
    end_fmt: str,
    n_frames: int,
    dialogue: str,
    context: str | None,
    prev_scene: str | None,
) -> str:
    parts = []
    if prev_scene:
        parts.append(f"Story context from preceding scenes:\n{prev_scene}\n")
    if context:
        parts.append(f"Video context: {context}\n")
    parts.append(f"These {n_frames} frames are from {start_fmt} to {end_fmt}.")
    if dialogue:
        parts.append(f"Dialogue spoken:\n{dialogue}")
    else:
        parts.append("No dialogue.")
    parts.append("Describe only what you can see in the frames using the structure above.")
    return "\n".join(parts)


def _is_credits_response(text: str) -> bool:
    """Check if VLM response indicates credits/titles."""
    if not text:
        return False
    try:
        data = json.loads(text)
        return data.get("is_credits") is True
    except json.JSONDecodeError:
        # Fallback: check for old format
        return "CREDITS:TRUE" in text.upper().replace(" ", "")


def _parse_vlm_response(text: str) -> dict:
    """Parse VLM JSON response into structured data."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to parse old text format
        result = {}
        for line in text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                result[key] = val.strip()
        return result


def _format_description(data: dict) -> str:
    """Format parsed VLM data into human-readable description."""
    if data.get("is_credits"):
        return "CREDITS: true"
    parts = []
    for key in ["location", "characters", "action", "dialogue", "key_objects", "mood", "emotion"]:
        val = data.get(key, "")
        if val:
            parts.append(f"{key.upper().replace('_', ' ')}: {val}")
    return "\n".join(parts)


def extract_prev_scene_summary(description: str, max_chars: int = 300) -> str:
    """Extract a summary from the previous scene for context."""
    # Handle both JSON and old text format
    data = _parse_vlm_response(description)
    if data.get("is_credits"):
        return ""
    # Build summary from action and characters
    parts = []
    for key in ["action", "characters", "emotion"]:
        val = data.get(key, "")
        if val:
            parts.append(val)
    summary = ". ".join(parts)
    return summary[:max_chars] if summary else description[:max_chars]


def extract_character_roster(results: list[dict]) -> str:
    """Aggregate unique CHARACTERS observations across all VLM descriptions."""
    observations = []
    for entry in results:
        desc = entry.get("description", "")
        if not desc or _is_credits_response(desc):
            continue
        data = _parse_vlm_response(desc)
        chars = data.get("characters", "")
        if chars and chars.lower() not in ("none visible", "none", ""):
            observations.append(chars)
    if not observations:
        return ""
    seen: set[str] = set()
    unique: list[str] = []
    for obs in observations:
        key = obs.lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(obs)
    return "; ".join(unique[:10])


# ---------------------------------------------------------------------------
# VLM inference via Ollama
# ---------------------------------------------------------------------------

def analyze_window(
    frames: list[Image.Image],
    start_ts: float,
    end_ts: float,
    model: str = OLLAMA_DEFAULT_MODEL,
    host: str = OLLAMA_DEFAULT_HOST,
    dialogue: str = "",
    context: str | None = None,
    prev_scene: str | None = None,
) -> str:
    start_fmt = format_timestamp(start_ts)
    end_fmt = format_timestamp(end_ts)
    prompt = build_user_prompt(start_fmt, end_fmt, len(frames), dialogue, context, prev_scene)
    images_b64 = [image_to_base64(img) for img in frames]
    return ollama_generate(prompt, images_b64, model=model, host=host)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a video with Ollama VLM + speech transcription"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")

    # Ollama args
    parser.add_argument("--ollama-model", default=OLLAMA_DEFAULT_MODEL,
                        help=f"Ollama model name (default: {OLLAMA_DEFAULT_MODEL})")
    parser.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST,
                        help=f"Ollama host URL (default: {OLLAMA_DEFAULT_HOST})")

    # Vision args
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Frame sampling interval in seconds (default: 5)")
    parser.add_argument("--window", type=int, default=4,
                        help="Frames per analysis window (default: 4)")
    parser.add_argument("--overlap", type=int, default=1,
                        help="Frame overlap between windows (default: 1)")
    parser.add_argument("--context", default=None,
                        help="Background context about the video (characters, genre, etc.)")
    parser.add_argument("--no-continuity", action="store_true",
                        help="Disable passing previous scene summary to next window")

    # Audio args
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio transcription (vision-only mode)")
    parser.add_argument("--transcriber", default="faster-whisper",
                        choices=["faster-whisper", "whisperx"],
                        help="Speech-to-text backend (default: faster-whisper)")
    parser.add_argument("--whisper-model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", default=None,
                        help="Force transcript language e.g. 'en' (auto-detected if omitted)")

    parser.add_argument("--decode-height", type=int, default=None,
                        help="Decode frames at this height before VLM (e.g. 640) — faster, same quality. Default: full resolution.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <video_stem>_analysis.json)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output) if args.output
        else Path(__file__).parent / (video_path.stem + "_analysis.json")
    )
    transcript_path = output_path.with_name(output_path.stem + "_transcript.json")

    pipeline_start = time.time()
    transcriber_label = "none" if args.no_audio else f"{args.transcriber} ({args.whisper_model})"
    print(f"VLM:          {args.ollama_model} (Ollama)")
    print(f"Ollama host:  {args.ollama_host}")
    print(f"Transcriber:  {transcriber_label}")
    print(f"Video:        {video_path.name}")
    print(f"Params:       interval={args.interval}s  window={args.window}  overlap={args.overlap}")
    if args.decode_height:
        print(f"Decode res:   height={args.decode_height}px")
    if args.context:
        print(f"Context:      {args.context[:80]}{'...' if len(args.context) > 80 else ''}")
    print()

    # Verify Ollama is reachable before doing the heavy work
    print("Checking Ollama...")
    check_ollama(args.ollama_model, args.ollama_host)
    print("  → Ollama ready\n")

    # ------------------------------------------------------------------
    # Step 1: Extract audio (quick) + Start transcription in parallel
    # ------------------------------------------------------------------
    segments: list[dict] = []
    transcript_future = None

    if not args.no_audio:
        print("Step 1/3 — Extracting audio and starting transcription...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        extract_audio(str(video_path), tmp_wav)
        
        # Start transcription in background thread
        def transcribe_worker():
            try:
                segs = transcribe(tmp_wav, args.transcriber, args.whisper_model, args.language)
                transcript_path.write_text(json.dumps(segs, indent=2, ensure_ascii=False))
                print(f"\n  → Transcription complete: {len(segs)} segments")
                Path(tmp_wav).unlink(missing_ok=True)
                return segs
            except Exception as e:
                print(f"\n  → Transcription error: {e}")
                Path(tmp_wav).unlink(missing_ok=True)
                return []
        
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        transcript_future = executor.submit(transcribe_worker)
        print(f"  → Transcription running in background\n")
    else:
        print("Step 1/3 — Audio transcription skipped (--no-audio)\n")

    # ------------------------------------------------------------------
    # Step 2+3: Stream frames → VLM (runs in parallel with transcription)
    # ------------------------------------------------------------------
    decode_label = f" at {args.decode_height}p" if args.decode_height else ""
    print(f"Step 2/3 — Streaming frames{decode_label} → VLM ({args.ollama_model})...")

    results = []
    prev_scene_history: list[str] = []

    for idx, start_ts, end_ts, window_frames in iter_frame_windows(
        str(video_path), args.interval, args.window, args.overlap,
        decode_height=args.decode_height,
    ):
        start_fmt = format_timestamp(start_ts)
        end_fmt = format_timestamp(end_ts)
        
        # Get dialogue from transcription if available
        dialogue = ""
        if transcript_future:
            if transcript_future.done():
                segments = transcript_future.result()
                dialogue = get_dialogue_for_window(segments, start_ts, end_ts)
            # If not done yet, proceed without dialogue (will be added in post-processing if needed)

        print(f"[{idx+1:02d}] {start_fmt} → {end_fmt}  ({len(window_frames)} frames)", flush=True)
        if dialogue:
            preview = dialogue.replace("\n", " ")
            print(f"  Dialogue: {preview[:120]}{'...' if len(preview) > 120 else ''}")

        prev_context: str | None = None
        if not args.no_continuity and prev_scene_history:
            prev_context = "\n".join(prev_scene_history[-3:])

        raw_response = analyze_window(
            frames=window_frames,
            start_ts=start_ts,
            end_ts=end_ts,
            model=args.ollama_model,
            host=args.ollama_host,
            dialogue=dialogue,
            context=args.context,
            prev_scene=prev_context,
        )

        # Parse VLM response (JSON or fallback to text)
        parsed_data = _parse_vlm_response(raw_response)
        is_credits = parsed_data.get("is_credits") is True

        # Format description for storage (human-readable or JSON string)
        description = _format_description(parsed_data) if parsed_data else raw_response

        if is_credits:
            print(f"  → credits/title detected, skipping\n")
        else:
            if not args.no_continuity:
                prev_scene_history.append(extract_prev_scene_summary(description))
                if len(prev_scene_history) > 3:
                    prev_scene_history.pop(0)
            print(f"  {description[:200]}{'...' if len(description) > 200 else ''}\n")

        entry = {
            "window": idx + 1,
            "start_sec": round(start_ts, 2),
            "end_sec": round(end_ts, 2),
            "start_fmt": start_fmt,
            "end_fmt": end_fmt,
            "frame_count": len(window_frames),
            "is_credits": is_credits,
            "dialogue": dialogue,
            "description": description,
            "vlm_data": parsed_data if parsed_data else None,
        }
        results.append(entry)
    
    # Wait for transcription to complete if still running
    if transcript_future:
        print("\nWaiting for transcription to complete...")
        segments = transcript_future.result()
        executor.shutdown()
        print(f"  → Transcript saved to: {transcript_path}")
        
        # Backfill dialogue for scenes that were processed before transcription finished
        for entry in results:
            if not entry["dialogue"]:
                entry["dialogue"] = get_dialogue_for_window(segments, entry["start_sec"], entry["end_sec"])

    processing_sec = round(time.time() - pipeline_start, 1)
    characters_observed = extract_character_roster(results)
    if characters_observed:
        print(f"  → Characters observed: {characters_observed}")
    output = {
        "video": str(video_path),
        "vlm_model": args.ollama_model,
        "vlm_backend": "ollama",
        "ollama_host": args.ollama_host,
        "transcriber": None if args.no_audio else args.transcriber,
        "whisper_model": None if args.no_audio else args.whisper_model,
        "interval_sec": args.interval,
        "window_size": args.window,
        "overlap": args.overlap,
        "context": args.context,
        "decode_height": args.decode_height,
        "characters_observed": characters_observed,
        "total_windows": len(results),
        "processing_sec": processing_sec,
        "scenes": results,
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
