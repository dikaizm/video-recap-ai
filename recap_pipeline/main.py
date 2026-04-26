"""
main.py — Full pipeline: analyze → transform → (narrate) → (TTS) → Remotion render.

Usage:
    # Full pipeline:
    python recap_pipeline/main.py --video /path/to/movie.mp4

    # Skip analysis if JSON already exists:
    python recap_pipeline/main.py --video /path/to/movie.mp4 \
        --skip-analysis --analysis-json output/<run>/analysis.json

    # With macOS TTS + DeepSeek narration:
    python recap_pipeline/main.py --video /path/to/movie.mp4 \
        --tts macos --narrate

    # With ElevenLabs:
    python recap_pipeline/main.py --video /path/to/movie.mp4 \
        --tts elevenlabs --elevenlabs-key sk_...
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Project root is one level up from this file (recap_pipeline/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTION_DIR = os.path.join(PROJECT_ROOT, "remotion")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")

# Ensure sibling modules (analyze, transform, narrate, tts) are importable
sys.path.insert(0, PACKAGE_DIR)

# Load .env if present
_env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_log_file = None  # set after run_dir is created


class _Tee:
    """Write to multiple sinks simultaneously (terminal + log file)."""
    def __init__(self, *sinks):
        self._sinks = sinks

    def write(self, data):
        for s in self._sinks:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._sinks:
            s.flush()

    def fileno(self):
        return self._sinks[0].fileno()


def setup_run_logging(run_dir: str) -> None:
    global _log_file
    log_path = os.path.join(run_dir, "pipeline.log")
    _log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    print(f"[log] pipeline log → {log_path}", flush=True)


def _stream_subprocess(cmd: list[str], **kwargs) -> None:
    """Run subprocess and stream its stdout+stderr through Python (captures into log)."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
    )
    assert proc.stdout
    for raw in proc.stdout:
        print(raw.decode(errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def make_run_dir(video_path: str) -> str:
    stem = Path(video_path).stem
    safe_stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)[:40]
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_BASE, f"{safe_stem}_{suffix}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_analysis(video_path: str, output_json: str, extra_args: list[str] | None = None) -> str:
    analyze_script = os.path.join(PACKAGE_DIR, "analyze.py")
    cmd = [sys.executable, "-u", analyze_script, "--video", video_path, "--output", output_json]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[analyze] {' '.join(cmd)}")
    _stream_subprocess(cmd, cwd=PROJECT_ROOT)
    return output_json


def _link_or_copy(src: str, dest: str) -> None:
    if os.path.exists(dest) or os.path.islink(dest):
        os.unlink(dest)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def setup_remotion_public(video_path: str, voiceover_dir: str | None = None) -> None:
    """Place the video (and optional voiceovers) in remotion/public/ for Remotion's static server."""
    public_dir = os.path.join(REMOTION_DIR, "public")
    os.makedirs(public_dir, exist_ok=True)

    abs_video = os.path.abspath(video_path)
    _link_or_copy(abs_video, os.path.join(public_dir, "video.mp4"))
    print(f"[setup] video → remotion/public/video.mp4")

    if voiceover_dir and os.path.isdir(voiceover_dir):
        public_vo = os.path.join(public_dir, "voiceover")
        os.makedirs(public_vo, exist_ok=True)
        count = 0
        for mp3 in sorted(os.listdir(voiceover_dir)):
            if mp3.endswith(".mp3"):
                _link_or_copy(
                    os.path.join(voiceover_dir, mp3),
                    os.path.join(public_vo, mp3),
                )
                count += 1
        print(f"[setup] {count} voiceover(s) → remotion/public/voiceover/")


def get_audio_duration(mp3_path: str) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", mp3_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def adjust_display_frames_to_audio(storyboard: dict, voiceover_dir: str, fps: int) -> dict:
    """Expand each scene's displayFrames to fit the actual voiceover audio duration."""
    padding_frames = round(0.4 * fps)
    adjusted = 0

    for scene in storyboard["scenes"]:
        window = scene["window"]
        mp3_path = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")
        if not os.path.exists(mp3_path):
            continue

        duration = get_audio_duration(mp3_path)
        if duration is None:
            continue

        needed_frames = round(duration * fps) + padding_frames
        if needed_frames > scene["displayFrames"]:
            scene["displayFrames"] = needed_frames
            adjusted += 1

    if adjusted:
        print(f"[sync] expanded displayFrames for {adjusted} scene(s) to fit narration audio")

    return storyboard


def write_run_log(run_dir: str, log: dict) -> None:
    log_path = os.path.join(run_dir, "run_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[log] run log → {log_path}")


def install_remotion_deps() -> None:
    node_modules = os.path.join(REMOTION_DIR, "node_modules")
    if os.path.isdir(node_modules):
        print("[deps] node_modules exists, skipping npm install")
        return

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm not found on PATH — install Node.js first")

    print("[deps] running npm install...")
    subprocess.run([npm, "install"], check=True, cwd=REMOTION_DIR)


def run_render(storyboard: dict, output_mp4: str, concurrency: int = 1) -> None:
    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError("npx not found on PATH — install Node.js first")

    abs_output = os.path.abspath(output_mp4)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    props_json = json.dumps({"storyboard": storyboard})

    if len(props_json) > 4_000:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, dir=REMOTION_DIR
        ) as tf:
            tf.write(props_json)
            props_arg = ["--props", tf.name]
            cleanup_props = tf.name
    else:
        props_arg = ["--props", props_json]
        cleanup_props = None

    cmd = [
        npx, "remotion", "render", "MovieRecap", abs_output,
        *props_arg,
        "--concurrency", str(concurrency),
        "--codec", "h264",
        "--image-format", "jpeg",
        "--log", "verbose",
    ]

    print(f"[render] starting Remotion render → {abs_output}")
    try:
        subprocess.run(cmd, check=True, cwd=REMOTION_DIR)
    finally:
        if cleanup_props and os.path.exists(cleanup_props):
            os.unlink(cleanup_props)

    print(f"[render] done → {abs_output}")


def main():
    parser = argparse.ArgumentParser(description="Movie recap pipeline")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--ollama-model", default="gemma4:e2b", help="Ollama model for VLM analysis")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis step")
    parser.add_argument("--analysis-json", default=None, help="Existing analysis JSON (implies --skip-analysis)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--recap-ratio", type=float, default=0.15,
        help="Recap duration as fraction of original movie (default: 0.15 = 15%%)",
    )
    parser.add_argument("--tts", choices=["macos", "gtts", "elevenlabs"], default=None)
    parser.add_argument("--elevenlabs-key", default=None)
    parser.add_argument("--tts-voice", default="Samantha", help="macOS TTS voice name")
    parser.add_argument("--narrate", action="store_true", help="Synthesize narration via DeepSeek before TTS")
    parser.add_argument("--deepseek-key", default=os.environ.get("DEEPSEEK_API_KEY"), help="DeepSeek API key (env: DEEPSEEK_API_KEY)")
    parser.add_argument("--narrate-force", action="store_true", help="Re-generate narration even if it exists")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip story coherence evaluation after narration")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", default=None, help="Output MP4 path (default: output/<video>_<timestamp>/recap.mp4)")
    args = parser.parse_args()

    pipeline_start = time.time()
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    run_dir = make_run_dir(args.video)
    setup_run_logging(run_dir)
    print(f"[run] output directory: {run_dir}")

    output_mp4 = args.output or os.path.join(run_dir, "recap.mp4")
    voiceover_dir = os.path.join(run_dir, "voiceover") if args.tts else None
    storyboard_path = os.path.join(run_dir, "storyboard.json")

    # Step 1: Analysis
    if args.analysis_json:
        analysis_json = args.analysis_json
        print(f"[analyze] using existing JSON: {analysis_json}")
    elif args.skip_analysis:
        candidates = [
            os.path.join(OUTPUT_BASE, "analysis.json"),
            os.path.join(OUTPUT_BASE, "shared", "analysis.json"),
        ]
        analysis_json = next((p for p in candidates if os.path.exists(p)), None)
        if not analysis_json:
            print("[error] --skip-analysis set but no analysis.json found", file=sys.stderr)
            sys.exit(1)
        print(f"[analyze] using existing JSON: {analysis_json}")
    else:
        analysis_json = os.path.join(run_dir, "analysis.json")
        run_analysis(args.video, analysis_json, extra_args=[
            "--ollama-model", args.ollama_model,
            "--ollama-host", args.ollama_host,
        ])

    # Step 2: Transform
    from transform import transform as do_transform

    storyboard = do_transform(
        analysis_path=analysis_json,
        output_path=storyboard_path,
        video_path=args.video,
        fps=args.fps,
        recap_ratio=args.recap_ratio,
        voiceover_dir=voiceover_dir,
    )

    # Step 3: Narration (optional)
    narrated_storyboard_path = os.path.join(run_dir, "storyboard_narrated.json")
    narration_meta: dict[int, dict] = {}

    if args.narrate:
        if not args.deepseek_key:
            print("[error] --deepseek-key required for narration (or set DEEPSEEK_API_KEY)", file=sys.stderr)
            sys.exit(1)

        from narrate import narrate as do_narrate

        print("[narrate] synthesizing narration via DeepSeek...")
        storyboard = do_narrate(
            storyboard_path=storyboard_path,
            output_path=narrated_storyboard_path,
            api_key=args.deepseek_key,
            analysis_path=analysis_json,
            force=args.narrate_force,
        )
        storyboard_path = narrated_storyboard_path

        for scene in storyboard["scenes"]:
            if scene.get("narratedText"):
                scene["povText"] = scene["narratedText"]
            narration_meta[scene["window"]] = {
                "narratedText": scene.get("narratedText", ""),
                "importanceScore": scene.get("importanceScore"),
            }

    # Step 3b: Story coherence evaluation (optional, runs when --narrate is active)
    if args.narrate and not args.no_evaluate:
        from evaluate import evaluate as do_evaluate

        evaluated_storyboard_path = os.path.join(run_dir, "storyboard_evaluated.json")
        storyboard = do_evaluate(
            storyboard_path=storyboard_path,
            output_path=evaluated_storyboard_path,
            api_key=args.deepseek_key,
            force=args.narrate_force,
        )
        storyboard_path = evaluated_storyboard_path

        for scene in storyboard["scenes"]:
            if scene.get("narratedText"):
                scene["povText"] = scene["narratedText"]
            narration_meta[scene["window"]] = {
                "narratedText": scene.get("narratedText", ""),
                "importanceScore": scene.get("importanceScore"),
            }

    # Step 4: TTS (optional)
    if args.tts:
        from tts import generate_batch

        tts_kwargs: dict = {}
        if args.tts == "macos":
            tts_kwargs["voice"] = args.tts_voice
        elif args.tts == "elevenlabs":
            if not args.elevenlabs_key:
                print("[error] --elevenlabs-key required for ElevenLabs TTS", file=sys.stderr)
                sys.exit(1)
            tts_kwargs["api_key"] = args.elevenlabs_key

        print(f"[tts] generating voiceovers with backend={args.tts}...")
        generate_batch(storyboard["scenes"], voiceover_dir, args.tts, **tts_kwargs)

        storyboard = do_transform(
            analysis_path=analysis_json,
            output_path=storyboard_path,
            video_path=args.video,
            fps=args.fps,
            recap_ratio=args.recap_ratio,
            voiceover_dir=voiceover_dir,
        )

        for scene in storyboard["scenes"]:
            meta = narration_meta.get(scene["window"])
            if meta:
                if meta.get("narratedText"):
                    scene["povText"] = meta["narratedText"]
                    scene["narratedText"] = meta["narratedText"]
                if meta.get("importanceScore") is not None:
                    scene["importanceScore"] = meta["importanceScore"]

        adjust_display_frames_to_audio(storyboard, voiceover_dir, args.fps)

        with open(storyboard_path, "w") as f:
            json.dump(storyboard, f, indent=2)

    # Step 5: Place video + voiceovers in remotion/public/
    setup_remotion_public(args.video, voiceover_dir=voiceover_dir)

    # Step 6: Install Node deps
    install_remotion_deps()

    # Step 7: Render
    render_start = time.time()
    run_render(storyboard, output_mp4, concurrency=args.concurrency)
    render_sec = round(time.time() - render_start, 1)

    total_sec = round(time.time() - pipeline_start, 1)

    sb_meta = storyboard.get("metadata", {})
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "video": args.video,
        "output_mp4": output_mp4,
        "run_dir": run_dir,
        "fps": args.fps,
        "recap_ratio": args.recap_ratio,
        "tts": args.tts,
        "narrate": args.narrate,
        "scene_count": len(storyboard.get("scenes", [])),
        "metadata": sb_meta,
        "timing": {
            "analysis_processing_sec": sb_meta.get("analysis_processing_sec"),
            "narration_processing_sec": sb_meta.get("narration_processing_sec"),
            "render_sec": render_sec,
            "total_sec": total_sec,
        },
    }
    write_run_log(run_dir, run_log)
    print(f"[done] total pipeline time: {total_sec:.1f}s")


if __name__ == "__main__":
    main()
