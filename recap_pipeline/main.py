"""
main.py — Full pipeline: analyze → transform → (narrate) → TTS → Remotion render.

Usage:
    # Folder-based input (recommended):
    python recap_pipeline/main.py --input input/crash_site/
    #   expects: input/crash_site/video.{mp4,mkv,...}  (required)
    #            input/crash_site/story.{md,txt}        (optional)

    # Legacy: explicit video path:
    python recap_pipeline/main.py --video /path/to/movie.mp4

    # Skip analysis if JSON already exists:
    python recap_pipeline/main.py --input input/crash_site/ \\
        --skip-analysis --analysis-json output/<run>/analysis.json

    # With DeepSeek narration:
    python recap_pipeline/main.py --input input/crash_site/ \\
        --narrate

    # Skip TTS (render without voiceover):
    python recap_pipeline/main.py --input input/crash_site/ \\
        --no-tts
"""
import argparse
import json
import os
import re
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


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".flv"}


def resolve_input_folder(folder: str) -> tuple[str, str | None]:
    """Return (video_path, story_path) discovered inside a movie input folder.

    Looks for:
      video.<ext>   — any extension in VIDEO_EXTENSIONS
      story.md / story.txt

    Raises FileNotFoundError when no video file is found.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")

    video_path: str | None = None
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if stem.lower() == "video" and ext.lower() in VIDEO_EXTENSIONS:
            video_path = os.path.join(folder, name)
            break

    if video_path is None:
        raise FileNotFoundError(
            f"No video file found in {folder}. "
            f"Expected a file named 'video' with one of: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    story_path: str | None = None
    for candidate in ("story.md", "story.txt"):
        p = os.path.join(folder, candidate)
        if os.path.isfile(p):
            story_path = p
            break

    return video_path, story_path


def make_run_dir(name: str) -> str:
    """Create and return an output run directory named after `name` (movie folder or video stem)."""
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:40]
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_BASE, f"{safe_name}_{suffix}")
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


# Minimum fraction of non-credits scenes that must have a non-empty VLM description
# to allow the pipeline to continue to narration.
VLM_MIN_DESCRIPTION_RATIO = 0.25


def check_vlm_quality(analysis_json: str, min_ratio: float = VLM_MIN_DESCRIPTION_RATIO) -> None:
    """Abort the pipeline if the VLM produced too few scene descriptions.

    Empty descriptions (blank or credits-only) mean the narration LLM has
    nothing to ground character identity on, leading to hallucinations.
    """
    with open(analysis_json) as f:
        data = json.load(f)
    scenes = data.get("scenes", [])
    non_credits = [s for s in scenes if not s.get("is_credits", False)]
    if not non_credits:
        return
    described = sum(
        1 for s in non_credits
        if s.get("description", "").strip() and "CREDITS: true" not in s.get("description", "").upper()
    )
    ratio = described / len(non_credits)
    print(
        f"[analyze] VLM coverage: {described}/{len(non_credits)} non-credits scenes "
        f"have descriptions ({ratio:.0%})",
        flush=True,
    )
    if ratio < min_ratio:
        print(
            f"\n[error] VLM description coverage too low ({ratio:.0%} < {min_ratio:.0%}).\n"
            f"  Most scenes have no visual description — narration would hallucinate characters and events.\n"
            f"  Check that Ollama is running the correct model and that frames are not all black.\n"
            f"  You can lower the threshold with --vlm-min-coverage or provide --context to guide narration.",
            file=sys.stderr,
        )
        sys.exit(1)


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
    """Set each scene's displayFrames to exactly match its voiceover audio duration."""
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
        if needed_frames != scene["displayFrames"]:
            scene["displayFrames"] = needed_frames
            adjusted += 1

    if adjusted:
        print(f"[sync] set displayFrames for {adjusted} scene(s) to match audio duration")

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


def _default_concurrency() -> int:
    """Half of logical CPU count, capped at 8, minimum 2."""
    try:
        import multiprocessing
        return max(2, min(8, multiprocessing.cpu_count() // 2))
    except Exception:
        return 4


def run_render(storyboard: dict, output_mp4: str, concurrency: int | None = None, gl: str = "angle") -> None:
    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError("npx not found on PATH — install Node.js first")

    if concurrency is None:
        concurrency = _default_concurrency()

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
        "--gl", gl,
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", metavar="FOLDER",
        help="Movie input folder containing video.{mp4,mkv,...} and optional story.{md,txt}",
    )
    input_group.add_argument("--video", help="Input video file path (legacy — use --input instead)")
    parser.add_argument("--ollama-model", default="gemma4:e2b", help="Ollama model for VLM analysis")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--decode-height", type=int, default=640,
                        help="Decode video frames at this height before VLM (default: 640, set 0 to disable)")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis step")
    parser.add_argument("--analysis-json", default=None, help="Existing analysis JSON (implies --skip-analysis)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--recap-ratio", type=float, default=0.15,
        help="Recap duration as fraction of original movie (default: 0.15 = 15%%)",
    )
    parser.add_argument("--no-tts", action="store_true", help="Skip Qwen3 voiceover generation")
    parser.add_argument("--qwen3-speed", type=float, default=1.3, help="Qwen3 speech speed (default: 1.3)")
    parser.add_argument("--narrate", action="store_true", help="Synthesize narration via DeepSeek before TTS")
    parser.add_argument("--deepseek-key", default=os.environ.get("DEEPSEEK_API_KEY"), help="DeepSeek API key (env: DEEPSEEK_API_KEY)")
    parser.add_argument("--narrate-force", action="store_true", help="Re-generate narration even if it exists")
    parser.add_argument("--story-context", default=None, help="Story synopsis text or path to a .txt/.md file to guide narration")
    parser.add_argument(
        "--vlm-min-coverage", type=float, default=VLM_MIN_DESCRIPTION_RATIO,
        help=f"Minimum fraction of scenes that must have VLM descriptions (default: {VLM_MIN_DESCRIPTION_RATIO}). Set to 0 to disable.",
    )
    parser.add_argument("--no-evaluate", action="store_true", help="Skip story coherence evaluation after narration")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Remotion render concurrency (default: half of CPU cores, max 8)")
    parser.add_argument("--gl", default="angle",
                        help="Remotion WebGL backend: angle (default, Metal on macOS), swiftshader, egl")
    parser.add_argument("--output", default=None, help="Output MP4 path (default: output/<video>_<timestamp>/recap.mp4)")
    parser.add_argument("--render-height", type=int, default=None, help="Downscale output to this height in pixels (e.g. 480). Width is scaled proportionally.")
    args = parser.parse_args()

    pipeline_start = time.time()

    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(start: float) -> str:
        secs = round(time.time() - start, 1)
        if secs < 60:
            return f"{secs}s"
        return f"{int(secs // 60)}m {int(secs % 60)}s"

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Resolve video path and optional story context from --input folder or --video
    auto_story_path: str | None = None
    if args.input:
        try:
            video_path, auto_story_path = resolve_input_folder(args.input)
        except FileNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)
        run_name = Path(args.input).resolve().name
        print(f"[input] folder: {args.input}")
        print(f"[input] video:  {video_path}")
        if auto_story_path:
            print(f"[input] story:  {auto_story_path}")
    else:
        video_path = args.video
        run_name = Path(video_path).stem

    run_dir = make_run_dir(run_name)
    setup_run_logging(run_dir)
    print(f"[run] output directory: {run_dir}")

    output_mp4 = args.output or os.path.join(run_dir, "recap.mp4")
    voiceover_dir = os.path.join(run_dir, "voiceover") if not args.no_tts else None
    storyboard_path = os.path.join(run_dir, "storyboard.json")

    # Step 1: Analysis
    step_start = time.time()
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
        extra_args = ["--ollama-model", args.ollama_model, "--ollama-host", args.ollama_host]
        if args.decode_height:
            extra_args += ["--decode-height", str(args.decode_height)]
        print(f"[analyze] start at {_ts()} (VLM={args.ollama_model})...")
        run_analysis(video_path, analysis_json, extra_args=extra_args)
        print(f"[analyze] done at {_ts()} — elapsed {_elapsed(step_start)}")

    if args.vlm_min_coverage > 0:
        check_vlm_quality(analysis_json, min_ratio=args.vlm_min_coverage)

    # Step 2: Transform
    step_start = time.time()
    print(f"[transform] start at {_ts()}...")
    from transform import transform as do_transform

    storyboard = do_transform(
        analysis_path=analysis_json,
        output_path=storyboard_path,
        video_path=video_path,
        fps=args.fps,
        recap_ratio=args.recap_ratio,
        voiceover_dir=voiceover_dir,
    )
    print(f"[transform] done at {_ts()} — elapsed {_elapsed(step_start)}")

    # Step 3: Narration (optional)
    narrated_storyboard_path = os.path.join(run_dir, "storyboard_narrated.json")
    narration_meta: dict[int, dict] = {}

    if args.narrate:
        if not args.deepseek_key:
            print("[error] --deepseek-key required for narration (or set DEEPSEEK_API_KEY)", file=sys.stderr)
            sys.exit(1)

        # Resolve story context: --story-context takes precedence, then auto-detected story file
        story_context: str | None = args.story_context
        if story_context and os.path.isfile(story_context):
            with open(story_context) as f:
                story_context = f.read().strip()
            print(f"[narrate] loaded story context from --story-context ({len(story_context)} chars)")
        elif not story_context and auto_story_path:
            with open(auto_story_path) as f:
                story_context = f.read().strip()
            print(f"[narrate] loaded story context from {auto_story_path} ({len(story_context)} chars)")

        from narrate import narrate as do_narrate

        step_start = time.time()
        print(f"[narrate] start at {_ts()} ({len(storyboard.get('scenes', []))} scenes)...")
        storyboard = do_narrate(
            storyboard_path=storyboard_path,
            output_path=narrated_storyboard_path,
            api_key=args.deepseek_key,
            analysis_path=analysis_json,
            story_context=story_context,
            force=args.narrate_force,
        )
        print(f"[narrate] done at {_ts()} — elapsed {_elapsed(step_start)}")
        storyboard_path = narrated_storyboard_path

        for scene in storyboard["scenes"]:
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
            narration_meta[scene["window"]] = {
                "narratedText": scene.get("narratedText", ""),
                "importanceScore": scene.get("importanceScore"),
            }

    # Step 4: TTS (enabled by default, use --no-tts to skip)
    if not args.no_tts:
        from tts import generate_batch

        _qwen3_model_path = os.path.join(PROJECT_ROOT, "models", "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
        if not os.path.exists(_qwen3_model_path):
            print(f"[error] Qwen3 model not found at {_qwen3_model_path!r}. "
                  f"Download it to the models/ directory.", file=sys.stderr)
            sys.exit(1)

        # Default voice design for cinematic thriller narration
        # Guidelines: specific, multidimensional, objective, original, concise
        # Dimensions: gender, age, pitch, pace, emotion, characteristics, purpose
        # Speed: 1.3x = brisk, efficient delivery for recap pacing
        _qwen3_instruct = (
            "A middle-aged male voice with a deep, low-pitched tone and magnetic quality. "
            "Brisk, efficient pace at 1.3x speed for recap narration. "
            "Rich vocal texture, serious yet calm emotional register. "
            "Ideal for documentary narration and thriller storytelling."
        )

        tts_kwargs: dict = {
            "model_path": _qwen3_model_path,
            "instruct": _qwen3_instruct,
            "speed": args.qwen3_speed,
        }
        if args.deepseek_key:
            tts_kwargs["api_key"] = args.deepseek_key

        step_start = time.time()
        print(f"[tts] start at {_ts()}...")
        generate_batch(storyboard["scenes"], voiceover_dir, **tts_kwargs)
        print(f"[tts] done at {_ts()} — elapsed {_elapsed(step_start)}")

        # Persist ttsText (stamped in-place by generate_batch) into the evaluated storyboard
        tts_storyboard_path = os.path.join(run_dir, "storyboard_tts.json")
        with open(tts_storyboard_path, "w") as f:
            json.dump(storyboard, f, indent=2)
        print(f"[tts] storyboard with ttsText → {tts_storyboard_path}")

        storyboard = do_transform(
            analysis_path=analysis_json,
            output_path=storyboard_path,
            video_path=video_path,
            fps=args.fps,
            recap_ratio=args.recap_ratio,
            voiceover_dir=voiceover_dir,
        )

        for scene in storyboard["scenes"]:
            meta = narration_meta.get(scene["window"])
            if meta:
                if meta.get("narratedText"):
                    scene["narratedText"] = meta["narratedText"]
                if meta.get("importanceScore") is not None:
                    scene["importanceScore"] = meta["importanceScore"]

        adjust_display_frames_to_audio(storyboard, voiceover_dir, args.fps)

        with open(storyboard_path, "w") as f:
            json.dump(storyboard, f, indent=2)

    # Step 5: Place video + voiceovers in remotion/public/
    setup_remotion_public(video_path, voiceover_dir=voiceover_dir)

    # Step 6: Install Node deps
    install_remotion_deps()

    # Step 7: Render
    step_start = time.time()
    print(f"[render] start at {_ts()}...")
    run_render(storyboard, output_mp4, concurrency=args.concurrency, gl=args.gl)
    print(f"[render] done at {_ts()} — elapsed {_elapsed(step_start)}")

    pipeline_elapsed = _elapsed(pipeline_start)
    print(f"\n{'='*60}")
    print(f"[pipeline] ALL DONE at {_ts()} — total elapsed {pipeline_elapsed}")
    print(f"[pipeline] output → {output_mp4}")
    print(f"{'='*60}")

    # Optional: downscale to target height
    if args.render_height:
        stem = re.sub(r"_\d+p$", "", Path(output_mp4).stem)
        scaled_mp4 = str(Path(output_mp4).with_name(f"{stem}_{args.render_height}p.mp4"))
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("[warn] ffmpeg not found — skipping downscale", file=sys.stderr)
        else:
            print(f"[render] downscaling to {args.render_height}p → {scaled_mp4}")
            subprocess.run(
                [
                    ffmpeg, "-y", "-i", output_mp4,
                    "-vf", f"scale=-2:{args.render_height}",
                    "-c:v", "libx264", "-crf", "20", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "128k",
                    scaled_mp4,
                ],
                check=True,
            )
            output_mp4 = scaled_mp4
            print(f"[render] downscaled → {output_mp4}")

    total_sec = round(time.time() - pipeline_start, 1)

    sb_meta = storyboard.get("metadata", {})
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "video": video_path,
        "output_mp4": output_mp4,
        "run_dir": run_dir,
        "fps": args.fps,
        "recap_ratio": args.recap_ratio,
        "tts": "qwen3" if not args.no_tts else None,
        "narrate": args.narrate,
        "scene_count": len(storyboard.get("scenes", [])),
        "metadata": sb_meta,
        "timing": {
            "analysis_processing_sec": sb_meta.get("analysis_processing_sec"),
            "narration_processing_sec": sb_meta.get("narration_processing_sec"),
            "total_sec": total_sec,
        },
    }
    write_run_log(run_dir, run_log)
    print(f"[done] total pipeline time: {_elapsed(pipeline_start)}")


if __name__ == "__main__":
    main()
