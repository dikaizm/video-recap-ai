"""
Video scene analyzer using SmolVLM2-500M-Video-Instruct + speech transcription.

Pipeline:
  1. Extract audio from video → transcribe with faster-whisper or whisperx
  2. Extract video frames at a fixed interval
  3. For each analysis window, inject matching dialogue + frames into SmolVLM2
  4. Output per-scene descriptions with timestamps and spoken dialogue

Usage:
    python analyze.py --video /path/to/video.mp4
    python analyze.py --video /path/to/video.mp4 --transcriber whisperx
    python analyze.py --video /path/to/video.mp4 --transcriber faster-whisper --whisper-model medium
    python analyze.py --video /path/to/video.mp4 --no-audio
    python analyze.py --video /path/to/video.mp4 --context "Sci-fi short film..."
    python analyze.py --video /path/to/video.mp4 --interval 5 --window 8 --overlap 2
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import av
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

SYSTEM_PROMPT = """\
You are a professional film analyst. You will be shown frames from a scene in a movie or video, \
along with any dialogue spoken during that scene.
Your task is to write a detailed scene description covering:
- LOCATION: Where the scene takes place (interior/exterior, environment details)
- CHARACTERS: Who is present, their appearance, expressions, and body language
- ACTION: What is happening moment by moment
- DIALOGUE: What is being said and what it reveals about the characters or plot
- KEY OBJECTS: Any notable props, items, or elements that stand out
- MOOD & TONE: The emotional atmosphere of the scene
- STORY SIGNIFICANCE: How this scene advances the plot or reveals character

Be specific and cinematic. Write in present tense."""


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract audio track from video to a mono 16 kHz WAV (Whisper's required format)."""
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
# Normalised output: list of {start, end, text, words: [{start, end, word}]}
# ---------------------------------------------------------------------------

def transcribe_faster_whisper(
    audio_path: str, model_size: str, language: str | None
) -> list[dict]:
    from faster_whisper import WhisperModel

    # CTranslate2 has no MPS support — CPU int8 is fast enough for one-shot transcription
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

    # WhisperX also has no MPS support — use CPU
    device = "cpu"
    compute_type = "int8"

    print(f"  Loading WhisperX model ({model_size})...")
    wx_model = whisperx.load_model(model_size, device, compute_type=compute_type,
                                   language=language)

    audio = whisperx.load_audio(audio_path)
    result = wx_model.transcribe(audio, batch_size=8)
    detected_lang = result.get("language", language or "unknown")
    print(f"  Detected language : {detected_lang}")

    # Forced alignment for word-level timestamps
    print("  Running forced alignment...")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    # Normalise to the same structure as faster-whisper output
    segments = []
    for seg in result["segments"]:
        words = [{"start": w["start"], "end": w["end"], "word": w["word"]}
                 for w in seg.get("words", [])]
        segments.append({"start": seg["start"], "end": seg["end"],
                         "text": seg["text"].strip(), "words": words})
    return segments


def transcribe(
    audio_path: str,
    backend: str,
    model_size: str,
    language: str | None,
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

def extract_frames(video_path: str, interval_sec: float) -> list[tuple[float, Image.Image]]:
    frames = []
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    interval_frames = max(1, int(fps * interval_sec))

    for i, frame in enumerate(container.decode(stream)):
        if i % interval_frames == 0:
            ts = float(frame.pts * stream.time_base)
            frames.append((ts, frame.to_image()))

    container.close()
    return frames


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
    parts = [
        f"The following {n_frames} frames span from {start_fmt} to {end_fmt} in the video.",
    ]
    if context:
        parts.append(f"Background context: {context.strip()}")
    if prev_scene:
        parts.append(
            f"Previous scene summary: {prev_scene.strip()}\n"
            "Use this to maintain narrative continuity."
        )
    if dialogue:
        parts.append(f"Dialogue spoken during this scene:\n{dialogue}")
    else:
        parts.append("No dialogue was detected during this scene.")
    parts.append(
        "Now analyze the frames and write a detailed scene description using the structure: "
        "LOCATION, CHARACTERS, ACTION, DIALOGUE, KEY OBJECTS, MOOD & TONE, STORY SIGNIFICANCE."
    )
    return "\n\n".join(parts)


def extract_prev_scene_summary(description: str, max_chars: int = 300) -> str:
    sentences = [s.strip() for s in description.replace("\n", " ").split(".") if s.strip()]
    summary = ""
    for sentence in reversed(sentences):
        candidate = sentence + ". " + summary
        if len(candidate) > max_chars:
            break
        summary = candidate
    return summary.strip() or description[-max_chars:]


# ---------------------------------------------------------------------------
# VLM inference
# ---------------------------------------------------------------------------

def analyze_window(
    processor,
    model,
    device: torch.device,
    frames: list[Image.Image],
    start_ts: float,
    end_ts: float,
    dialogue: str = "",
    context: str | None = None,
    prev_scene: str | None = None,
    max_new_tokens: int = 400,
) -> str:
    start_fmt = format_timestamp(start_ts)
    end_fmt = format_timestamp(end_ts)
    user_text = build_user_prompt(
        start_fmt, end_fmt, len(frames), dialogue, context, prev_scene
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                *[{"type": "image"} for _ in frames],
                {"type": "text", "text": user_text},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=frames, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a video with SmolVLM2 + speech transcription"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")

    # Vision args
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Frame sampling interval in seconds (default: 5)")
    parser.add_argument("--window", type=int, default=8,
                        help="Frames per analysis window (default: 8)")
    parser.add_argument("--overlap", type=int, default=2,
                        help="Frame overlap between windows (default: 2)")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Max new tokens per scene description (default: 400)")
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

    device = get_device()
    transcriber_label = "none" if args.no_audio else f"{args.transcriber} ({args.whisper_model})"
    print(f"Device:       {device}")
    print(f"VLM:          {VLM_MODEL_ID}")
    print(f"Transcriber:  {transcriber_label}")
    print(f"Video:        {video_path.name}")
    print(f"Params:       interval={args.interval}s  window={args.window}  "
          f"overlap={args.overlap}  max_tokens={args.max_tokens}")
    if args.context:
        print(f"Context:      {args.context[:80]}{'...' if len(args.context) > 80 else ''}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Transcribe audio
    # ------------------------------------------------------------------
    segments: list[dict] = []

    if not args.no_audio:
        print("Step 1/3 — Transcribing audio...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            extract_audio(str(video_path), tmp_wav)
            segments = transcribe(tmp_wav, args.transcriber, args.whisper_model, args.language)
        finally:
            Path(tmp_wav).unlink(missing_ok=True)

        print(f"  → {len(segments)} transcript segments")
        transcript_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        print(f"  → Transcript saved to: {transcript_path}")
        print()
    else:
        print("Step 1/3 — Audio transcription skipped (--no-audio)\n")

    # ------------------------------------------------------------------
    # Step 2: Extract frames
    # ------------------------------------------------------------------
    print("Step 2/3 — Extracting frames...")
    all_frames = extract_frames(str(video_path), args.interval)
    print(f"  → {len(all_frames)} frames extracted")

    step = max(1, args.window - args.overlap)
    windows: list[tuple[float, float, list[Image.Image]]] = []
    for i in range(0, len(all_frames), step):
        chunk = all_frames[i : i + args.window]
        if not chunk:
            break
        windows.append((chunk[0][0], chunk[-1][0], [f for _, f in chunk]))

    print(f"  → {len(windows)} analysis windows\n")

    # ------------------------------------------------------------------
    # Step 3: VLM analysis
    # ------------------------------------------------------------------
    print("Step 3/3 — Loading VLM...")
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL_ID,
        dtype=torch.float16 if device.type != "cpu" else torch.float32,
    ).to(device)
    model.eval()
    print("  → Model ready\n")

    results = []
    prev_scene: str | None = None

    for idx, (start_ts, end_ts, window_frames) in enumerate(windows):
        start_fmt = format_timestamp(start_ts)
        end_fmt = format_timestamp(end_ts)
        dialogue = get_dialogue_for_window(segments, start_ts, end_ts) if segments else ""

        print(f"[{idx+1:02d}/{len(windows)}] {start_fmt} → {end_fmt}  ({len(window_frames)} frames)")
        if dialogue:
            preview = dialogue.replace("\n", " ")
            print(f"  Dialogue: {preview[:120]}{'...' if len(preview) > 120 else ''}")

        description = analyze_window(
            processor=processor,
            model=model,
            device=device,
            frames=window_frames,
            start_ts=start_ts,
            end_ts=end_ts,
            dialogue=dialogue,
            context=args.context,
            prev_scene=None if args.no_continuity else prev_scene,
            max_new_tokens=args.max_tokens,
        )

        if not args.no_continuity:
            prev_scene = extract_prev_scene_summary(description)

        entry = {
            "window": idx + 1,
            "start_sec": round(start_ts, 2),
            "end_sec": round(end_ts, 2),
            "start_fmt": start_fmt,
            "end_fmt": end_fmt,
            "frame_count": len(window_frames),
            "dialogue": dialogue,
            "description": description,
        }
        results.append(entry)
        print(f"  {description[:200]}{'...' if len(description) > 200 else ''}\n")

    output = {
        "video": str(video_path),
        "vlm_model": VLM_MODEL_ID,
        "transcriber": None if args.no_audio else args.transcriber,
        "whisper_model": None if args.no_audio else args.whisper_model,
        "interval_sec": args.interval,
        "window_size": args.window,
        "overlap": args.overlap,
        "context": args.context,
        "total_windows": len(results),
        "scenes": results,
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
