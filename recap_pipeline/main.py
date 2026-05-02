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

from state import PipelineState, atomic_write_json, probe_audio

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

# Import premiere export
from premiere_export import generate_premiere_xml, generate_premiere_edl

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
    """Set each scene's displayFrames to exactly match its voiceover audio duration.

    If the scene already has a `segments` list, those are scaled proportionally and
    a final-segment correction makes their sum exactly equal the new total.
    """
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
        old_frames = scene["displayFrames"]
        if needed_frames != old_frames:
            scene["displayFrames"] = needed_frames
            adjusted += 1

            segs = scene.get("segments")
            if isinstance(segs, list) and segs and old_frames > 0:
                ratio = needed_frames / old_frames
                running = 0
                for seg in segs:
                    seg_new = max(1, round(seg.get("displayFrames", 1) * ratio))
                    seg["displayFrames"] = seg_new
                    running += seg_new
                # Reconcile rounding so the sum equals needed_frames exactly.
                diff = needed_frames - running
                if diff != 0 and segs:
                    segs[-1]["displayFrames"] = max(1, segs[-1]["displayFrames"] + diff)

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


PIPELINE_STEPS = [
    "analysis", "transform", "cluster", "intro",
    "narrate", "tts", "align", "greeting",
    "render", "premiere", "metadata",
]

DOWNSTREAM: dict[str, list[str]] = {
    "analysis":  ["transform", "cluster", "intro", "narrate", "tts", "align", "greeting", "render", "premiere", "metadata"],
    "transform": ["cluster", "intro", "narrate", "tts", "align", "greeting", "render", "premiere", "metadata"],
    "cluster":   ["intro", "narrate", "tts", "align", "greeting", "render", "premiere", "metadata"],
    "intro":     ["narrate", "tts", "align", "greeting", "render", "premiere", "metadata"],
    "narrate":   ["tts", "align", "greeting", "render", "premiere", "metadata"],
    "tts":       ["align", "greeting", "render", "premiere", "metadata"],
    "align":     ["greeting", "render", "premiere", "metadata"],
    "greeting":  ["render", "premiere", "metadata"],
    "render":    ["premiere", "metadata"],
    "premiere":  ["metadata"],
    "metadata":  [],
}


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
    parser.add_argument("--resume", action="store_true", help="Resume from last successful step (auto-detects what needs to be run)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--recap-ratio", type=float, default=0.15,
        help="Recap duration as fraction of original movie (default: 0.15 = 15%%)",
    )
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS voiceover generation")
    parser.add_argument("--tts-model", default=None,
                        help="TTS model path or HuggingFace repo ID (default: local Qwen3-TTS-12Hz model)")
    parser.add_argument("--qwen3-speed", type=float, default=1.0, help="TTS speech speed (default: 1.0)")
    parser.add_argument(
        "--genre", default="auto",
        help=(
            "Voice preset genre: auto (detect from story context), "
            "thriller, sci-fi, horror, action, drama, comedy, romance, fantasy, documentary "
            "(default: auto)"
        ),
    )
    parser.add_argument("--no-greeting", action="store_true",
                        help="Skip branded channel greeting beat at the start")
    parser.add_argument("--greeting-text", default=None,
                        help="Override greeting text (default: built-in Premiere Roll signature)")
    parser.add_argument("--max-beat-sec", type=float, default=8.0,
                        help="Maximum beat duration in seconds (default: 8.0)")
    parser.add_argument("--min-beat-sec", type=float, default=4.0,
                        help="Minimum beat duration in seconds (default: 4.0)")
    parser.add_argument("--no-intro", action="store_true",
                        help="Skip prepending an intro beat to the recap")
    parser.add_argument("--intro-scenes", type=int, default=3,
                        help="Number of establishing source scenes to use in the intro (default: 3)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip post-TTS STT+LLM scene alignment (segments)")
    parser.add_argument("--align-whisper-model", default="tiny.en",
                        help="faster-whisper model size for align STT (default: tiny.en)")
    parser.add_argument("--align-expand", type=int, default=2,
                        help="Neighbor expansion when picking candidate scenes per phrase (default: 2)")
    parser.add_argument("--narrate", action="store_true", help="Synthesize narration via DeepSeek before TTS")
    parser.add_argument("--deepseek-key", default=os.environ.get("DEEPSEEK_API_KEY"), help="DeepSeek API key (env: DEEPSEEK_API_KEY)")
    parser.add_argument("--narrate-force", action="store_true", help="Re-generate narration even if it exists (alias for --force-narrate)")
    for _step in PIPELINE_STEPS:
        parser.add_argument(
            f"--force-{_step}",
            action="store_true",
            help=f"Force re-run of {_step} step and all downstream steps",
        )
    parser.add_argument(
        "--force-from", metavar="STEP", choices=PIPELINE_STEPS,
        help="Re-run from this step and all downstream steps",
    )
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
    parser.add_argument("--no-agent", action="store_true", help="Disable agent self-review loop (runs by default)")
    parser.add_argument("--agent-max-iter", type=int, default=3, help="Maximum self-improvement iterations (default: 3)")
    parser.add_argument("--agent-threshold", type=float, default=0.85, help="Stop when this fraction of beats score 4+ (default: 0.85)")
    parser.add_argument("--agent-no-render", action="store_true", help="Skip re-render during agent loop (eval only)")
    args = parser.parse_args()

    # Agent loop: delegate to agent.py and exit early (default behaviour; skip with --no-agent)
    if not args.no_agent:
        # Agent always enables narration and TTS
        args.narrate = True
        args.no_tts = False
        from agent import run_agent_loop
        sys.exit(run_agent_loop(args))

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

    # Find existing run directory for --resume
    def _find_latest_run(name: str) -> str | None:
        """Find the most recent run directory for this video."""
        matching = []
        for d in os.listdir(OUTPUT_BASE):
            if d.startswith(name + "_") and os.path.isdir(os.path.join(OUTPUT_BASE, d)):
                matching.append(d)
        if not matching:
            return None
        # Sort by timestamp (newest last)
        matching.sort()
        return os.path.join(OUTPUT_BASE, matching[-1])

    if args.resume:
        existing_run = _find_latest_run(run_name)
        if existing_run:
            run_dir = existing_run
            print(f"[resume] using existing run: {run_dir}")
        else:
            print(f"[resume] no existing run found, creating new")
            run_dir = make_run_dir(run_name)
    else:
        run_dir = make_run_dir(run_name)

    setup_run_logging(run_dir)
    print(f"[run] output directory: {run_dir}")

    output_mp4 = args.output or os.path.join(run_dir, "recap.mp4")
    voiceover_dir = os.path.join(run_dir, "voiceover")
    storyboard_path = os.path.join(run_dir, "storyboard.json")
    narrated_storyboard_path = os.path.join(run_dir, "storyboard_narrated.json")
    evaluated_storyboard_path = os.path.join(run_dir, "storyboard_evaluated.json")
    tts_storyboard_path = os.path.join(run_dir, "storyboard_tts.json")
    analysis_json = os.path.join(run_dir, "analysis.json")

    # --- Pipeline state manifest ---
    ps = PipelineState(run_dir)

    # Resolve force flags: --force-<step> and --force-from clear state for that step + downstream.
    _forced: set[str] = set()
    for _step in PIPELINE_STEPS:
        flag = f"force_{_step.replace('-', '_')}"
        if getattr(args, flag, False):
            _forced.update([_step] + DOWNSTREAM.get(_step, []))
    # --narrate-force is a legacy alias for --force-narrate
    if args.narrate_force:
        _forced.update(["narrate"] + DOWNSTREAM.get("narrate", []))
    if getattr(args, "force_from", None):
        _forced.update([args.force_from] + DOWNSTREAM.get(args.force_from, []))
    if _forced:
        print(f"[state] forcing re-run of: {', '.join(s for s in PIPELINE_STEPS if s in _forced)}")
        ps.clear_steps(list(_forced))

    if args.resume:
        print(f"[resume] pipeline state: {ps.summary()}")

    # Step 1: Analysis
    step_start = time.time()
    if args.analysis_json:
        analysis_json = args.analysis_json
        print(f"[analyze] using external JSON: {analysis_json}")
        ps.mark_complete("analysis", note="external")
    elif args.skip_analysis or (args.resume and ps.is_complete("analysis")):
        if args.resume and ps.is_complete("analysis"):
            print("[analyze] resuming — already complete")
        if not os.path.exists(analysis_json):
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
        ps.mark_running("analysis")
        extra_args = ["--ollama-model", args.ollama_model, "--ollama-host", args.ollama_host]
        if args.decode_height:
            extra_args += ["--decode-height", str(args.decode_height)]
        print(f"[analyze] start at {_ts()} (VLM={args.ollama_model})...")
        run_analysis(video_path, analysis_json, extra_args=extra_args)
        ps.mark_complete("analysis")
        print(f"[analyze] done at {_ts()} — elapsed {_elapsed(step_start)}")

    if args.vlm_min_coverage > 0:
        check_vlm_quality(analysis_json, min_ratio=args.vlm_min_coverage)

    # Step 2: Transform (scene selection)
    step_start = time.time()
    from transform import transform as do_transform

    if args.resume and ps.is_complete("transform") and os.path.exists(storyboard_path):
        print("[transform] resuming — already complete")
        with open(storyboard_path) as f:
            storyboard = json.load(f)
    else:
        ps.mark_running("transform")
        print(f"[transform] start at {_ts()}...")
        storyboard = do_transform(
            analysis_path=analysis_json,
            output_path=storyboard_path,
            video_path=video_path,
            fps=args.fps,
            recap_ratio=args.recap_ratio,
            voiceover_dir=voiceover_dir,
        )
        ps.mark_complete("transform")
        print(f"[transform] done at {_ts()} — elapsed {_elapsed(step_start)}")

    # Step 2b: Cluster scenes into beats (for proper pacing)
    clustered_storyboard_path = os.path.join(run_dir, "storyboard_clustered.json")
    from cluster import cluster_scenes

    # source_scenes_pool comes from pre-cluster storyboard (individual scenes).
    # Always populated from storyboard.json so align.py can expand to neighbors.
    source_scenes_pool = list(storyboard["scenes"])

    step_start = time.time()
    if args.resume and ps.is_complete("cluster") and os.path.exists(clustered_storyboard_path):
        print("[cluster] resuming — already complete")
        with open(clustered_storyboard_path) as f:
            clustered = json.load(f)
        clustered_scenes = clustered["scenes"]
    else:
        ps.mark_running("cluster")
        print(f"[cluster] start at {_ts()}...")
        target_duration = sum(s["displayFrames"] for s in storyboard["scenes"]) / args.fps
        clustered_scenes = cluster_scenes(
            storyboard["scenes"],
            target_duration=target_duration,
            fps=args.fps,
            min_beat_sec=args.min_beat_sec,
            max_beat_sec=args.max_beat_sec,
        )
        clustered = {**storyboard, "scenes": clustered_scenes}
        atomic_write_json(clustered_storyboard_path, clustered)
        ps.mark_complete("cluster")
        print(f"[cluster] done at {_ts()} — elapsed {_elapsed(step_start)}")

    # Step 2c: Build + prepend the intro beat (skipped with --no-intro or when
    # narration is disabled, since the intro narration is generated via DeepSeek).
    if not args.no_intro and args.narrate:
        if not args.deepseek_key:
            ps.mark_skipped("intro", "no deepseek key")
            print("[intro] skipping — DeepSeek key required (or pass --no-intro)")
        elif args.resume and ps.is_complete("intro"):
            print("[intro] resuming — already complete")
            # Reload clustered so the intro beat is present in memory.
            with open(clustered_storyboard_path) as f:
                clustered = json.load(f)
            clustered_scenes = clustered["scenes"]
        else:
            # Resolve story context for the intro narration.
            _intro_context: str | None = args.story_context
            if _intro_context and os.path.isfile(_intro_context):
                with open(_intro_context) as f:
                    _intro_context = f.read().strip()
            elif not _intro_context and auto_story_path:
                with open(auto_story_path) as f:
                    _intro_context = f.read().strip()

            from intro import build_intro_beat

            ps.mark_running("intro")
            step_start = time.time()
            print(f"[intro] start at {_ts()}...")
            intro_beat = build_intro_beat(
                source_scenes=source_scenes_pool,
                story_context=_intro_context or "",
                api_key=args.deepseek_key,
                fps=args.fps,
                scene_count=args.intro_scenes,
            )
            if intro_beat:
                clustered_scenes = [intro_beat] + clustered_scenes
                clustered["scenes"] = clustered_scenes
                atomic_write_json(clustered_storyboard_path, clustered)
                print(
                    f"[intro] prepended intro beat ({len(intro_beat['scenes'])} scenes, "
                    f"{intro_beat['displayFrames']} frames) — \"{intro_beat['narratedText']}\""
                )
            else:
                print("[intro] skipped — no source scenes available")
            ps.mark_complete("intro")
            print(f"[intro] done at {_ts()} — elapsed {_elapsed(step_start)}")
    else:
        ps.mark_skipped("intro", "--no-intro or narrate disabled")

    # Step 3: Narration (optional)
    narration_meta: dict[int, dict] = {}

    # Resolve story context once — used by narration, genre detection, and metadata gen
    story_context: str | None = args.story_context
    if story_context and os.path.isfile(story_context):
        with open(story_context) as f:
            story_context = f.read().strip()
    elif not story_context and auto_story_path:
        with open(auto_story_path) as f:
            story_context = f.read().strip()

    _narrate_ran = False  # controls whether the validation block runs below
    if args.narrate:
        if not args.deepseek_key:
            print("[error] --deepseek-key required for narration (or set DEEPSEEK_API_KEY)", file=sys.stderr)
            sys.exit(1)

        if args.resume and ps.is_complete("narrate"):
            print("[narrate] resuming — already complete")
            with open(narrated_storyboard_path) as f:
                clustered = json.load(f)
            storyboard = clustered
            storyboard_path = narrated_storyboard_path
        else:
            if story_context:
                print(f"[narrate] story context: {len(story_context)} chars")

            from narrate import narrate_beats as do_narrate

            ps.mark_running("narrate")
            step_start = time.time()
            print(f"[narrate_beats] start at {_ts()} ({len(clustered['scenes'])} beats)...")
            clustered = do_narrate(
                storyboard_path=clustered_storyboard_path,
                output_path=narrated_storyboard_path,
                api_key=args.deepseek_key,
                analysis_path=analysis_json,
                story_context=story_context,
                force=False,  # narrate_beats skips beats that already have narratedText
            )
            print(f"[narrate_beats] done at {_ts()} — elapsed {_elapsed(step_start)}")
            storyboard = clustered
            storyboard_path = narrated_storyboard_path
            _narrate_ran = True
    else:
        ps.mark_skipped("narrate", "--narrate not requested")

    # Step 3b: Validate narration — every beat must have narratedText before TTS.
    # Runs only when narration was executed this pass (not on resume-skip).
    if _narrate_ran:
        from narrate import narrate_beats as do_narrate

        for retry in range(2):
            missing = [i for i, b in enumerate(clustered["scenes"]) if not b.get("narratedText")]
            if not missing:
                break

            if retry == 0:
                # First retry: fill only missing beats (force=False skips already-narrated)
                print(f"[validate] {len(missing)}/{len(clustered['scenes'])} beats missing narration → filling gaps")
                clustered = do_narrate(
                    storyboard_path=narrated_storyboard_path,
                    output_path=narrated_storyboard_path,
                    api_key=args.deepseek_key,
                    analysis_path=analysis_json,
                    story_context=story_context,
                    force=False,
                )
            else:
                # Second retry: narrate only missing beats individually via LLM
                print(f"[validate] {len(missing)} beats still missing → individual retry: {missing}")
                for idx in missing:
                    from narrate import _http_post, NARRATE_BATCH_SYSTEM, _parse_batch_response
                    beat = clustered["scenes"][idx]
                    pov = beat.get("povText", "").strip()
                    dialogue = beat.get("dialogue", "").strip()

                    pct = int(round(beat.get("startSec", 0) / max(b.get("endSec", 1) for b in clustered["scenes"]) * 100))
                    user_prompt = f"Beat to narrate:\n[Beat] pos={pct}% visual: {pov[:300]}"
                    if dialogue:
                        user_prompt += f" dialogue: {dialogue[:200]}"

                    result = _http_post(args.deepseek_key, [
                        {"role": "system", "content": NARRATE_BATCH_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ], max_tokens=100, temperature=0.5, timeout=60, thinking_enabled=False, json_mode=True)

                    narrations = _parse_batch_response(result, 1)
                    if narrations:
                        clustered["scenes"][idx]["narratedText"] = narrations[0].strip()
                        print(f"    beat {idx+1:02d}: {narrations[0][:60]}...")

        missing = [i for i, b in enumerate(clustered["scenes"]) if not b.get("narratedText")]
        if missing:
            print(f"[error] {len(missing)} beats still without narration. Cannot proceed with TTS.", file=sys.stderr)
            print(f"  Beat indices: {missing}", file=sys.stderr)
            sys.exit(1)

        storyboard = clustered
        storyboard_path = narrated_storyboard_path
        ps.mark_complete("narrate", scene_count=len(clustered["scenes"]))

    # Step 4: TTS (enabled by default, use --no-tts to skip)
    _default_model_path = os.path.join(PROJECT_ROOT, "models", "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
    _qwen3_model_path = args.tts_model if args.tts_model else _default_model_path
    if not args.no_tts:
        _is_local = os.path.exists(_qwen3_model_path) or _qwen3_model_path.startswith(("/", "."))
        if _is_local and not os.path.exists(_qwen3_model_path):
            print(f"[error] Qwen3 model not found at {_qwen3_model_path!r}. "
                  f"Download it to the models/ directory.", file=sys.stderr)
            sys.exit(1)

        if args.resume and ps.is_complete("tts") and os.path.exists(tts_storyboard_path):
            print("[tts] resuming — already complete")
            with open(tts_storyboard_path) as f:
                storyboard = json.load(f)
            storyboard_path = tts_storyboard_path
            # Ensure window fields are set (they're in the checkpoint but re-confirm)
            for i, scene in enumerate(storyboard["scenes"]):
                scene.setdefault("window", i + 1)
        else:
            from tts import generate_batch
            from voices import resolve_voice_instruct, get_voice_instruct
            _genre_arg = args.genre
            if _genre_arg == "auto":
                if story_context and args.deepseek_key:
                    print(f"[voice] auto-detecting genres from story context...")
                    _instruct, _genres = resolve_voice_instruct(story_context, args.deepseek_key)
                    print(f"[voice] detected genres: {_genres}")
                    if len(_genres) > 1:
                        print(f"[voice] blending {len(_genres)} genre presets → custom instruct")
                else:
                    _genres = ["documentary"]
                    _instruct = get_voice_instruct("documentary")
            else:
                _genres = [_genre_arg]
                _instruct = get_voice_instruct(_genre_arg)
                print(f"[voice] using genre preset: {_genre_arg}")
            storyboard.setdefault("metadata", {})["voice_genres"] = _genres
            storyboard["metadata"]["voice_instruct"] = _instruct

            tts_kwargs: dict = {
                "model_path": _qwen3_model_path,
                "speed": args.qwen3_speed,
                "instruct": _instruct,
            }
            if args.deepseek_key:
                tts_kwargs["api_key"] = args.deepseek_key

            ps.mark_running("tts")
            step_start = time.time()
            print(f"[tts] start at {_ts()}...")
            generate_batch(storyboard["scenes"], voiceover_dir, **tts_kwargs)
            print(f"[tts] done at {_ts()} — elapsed {_elapsed(step_start)}")

            # Beats produced by cluster.py have no `window` field. Assign window=i+1
            # so `adjust_display_frames_to_audio` (and downstream Zod validation) can
            # locate each beat's `scene_NN.mp3` file.
            for i, scene in enumerate(storyboard["scenes"]):
                scene.setdefault("window", i + 1)

            # Final sync guard: clamp each beat's displayFrames to its actual TTS
            # audio duration + 0.4s padding so visuals end with the narration.
            adjust_display_frames_to_audio(storyboard, voiceover_dir, args.fps)

            # Persist checkpoint before align so a crash in align can resume from here.
            atomic_write_json(tts_storyboard_path, storyboard)
            atomic_write_json(storyboard_path, storyboard)
            storyboard_path = tts_storyboard_path
            ps.mark_complete("tts")
            print(f"[tts] storyboard with ttsText → {tts_storyboard_path}")

        # Step 4b: Align scene cuts to narration phrases via STT (per beat).
        # Replaces the placeholder/cluster-based segments with phrase-aligned
        # ones whose timing comes from word-level Whisper timestamps.
        if not args.no_align:
            if not args.deepseek_key:
                ps.mark_skipped("align", "no deepseek key")
                print("[align] skipping — DeepSeek key required for phrase matching (or pass --no-align)")
            elif args.resume and ps.is_complete("align"):
                print("[align] resuming — already complete")
                # segments are already in storyboard loaded from tts_storyboard_path
            else:
                from align import align_beats

                ps.mark_running("align")
                step_start = time.time()
                print(f"[align] start at {_ts()}...")

                def _align_progress(i: int, beat: dict) -> None:
                    atomic_write_json(tts_storyboard_path, storyboard)

                align_beats(
                    storyboard["scenes"],
                    source_scenes=source_scenes_pool,
                    voiceover_dir=voiceover_dir,
                    api_key=args.deepseek_key,
                    fps=args.fps,
                    expand=args.align_expand,
                    whisper_model_size=args.align_whisper_model,
                    on_progress=_align_progress,
                )
                # Final persist with complete alignment data
                atomic_write_json(tts_storyboard_path, storyboard)
                atomic_write_json(storyboard_path, storyboard)
                ps.mark_complete("align")
                print(f"[align] done at {_ts()} — elapsed {_elapsed(step_start)}")
        else:
            ps.mark_skipped("align", "--no-align")
    else:
        ps.mark_skipped("tts", "--no-tts")
        ps.mark_skipped("align", "--no-tts")

    # Step 4d: Build channel greeting beat (prepended before render)
    if not args.no_greeting:
        from greeting import build_greeting_beat, DEFAULT_GREETING

        _greeting_text = args.greeting_text or DEFAULT_GREETING
        greeting_mp3 = os.path.join(voiceover_dir, "scene_00.mp3")

        if args.resume and ps.is_complete("greeting"):
            print("[greeting] resuming — already complete")
            # greeting beat is already at index 0 in storyboard (in the tts_storyboard_path checkpoint)
        else:
            ps.mark_running("greeting")
            greeting_beat = None

            # Reuse existing scene_00.mp3 only when it was generated for the same greeting text.
            _greeting_state = ps._state.get("greeting", {})
            _stored_text = _greeting_state.get("greeting_text")
            if os.path.exists(greeting_mp3) and _stored_text == _greeting_text:
                _dur = probe_audio(greeting_mp3)
                if _dur and _dur > 0.3:
                    print(f"[greeting] scene_00.mp3 exists ({_dur:.1f}s) — reusing")
                    _padding_frames = 15
                    _display_frames = max(int((_dur + _padding_frames / args.fps) * args.fps), args.fps * 3)
                    from greeting import CHANNEL_NAME
                    greeting_beat = {
                        "isGreeting": True, "window": 0,
                        "startSec": 0.0, "endSec": round(_display_frames / args.fps, 3),
                        "durationInFrames": _display_frames, "displayFrames": _display_frames,
                        "povText": f"[GREETING] {_greeting_text}",
                        "narratedText": _greeting_text,
                        "dialogue": "", "startFmt": "00:00:00",
                        "segments": [], "channelName": CHANNEL_NAME,
                    }

            if greeting_beat is None:
                step_start = time.time()
                greeting_beat = build_greeting_beat(
                    fps=args.fps,
                    voiceover_dir=voiceover_dir,
                    model_path=_qwen3_model_path,
                    greeting_text=_greeting_text,
                    tts_speed=args.qwen3_speed,
                )
                print(f"[greeting] done at {_ts()} — elapsed {_elapsed(step_start)}")

            storyboard["scenes"].insert(0, greeting_beat)
            atomic_write_json(storyboard_path, storyboard)
            ps.mark_complete("greeting", greeting_text=_greeting_text)
            print(f"[greeting] beat prepended → window=0, {greeting_beat['displayFrames']} frames")
    else:
        ps.mark_skipped("greeting", "--no-greeting")

    # Step 5: Place video + voiceovers in remotion/public/
    setup_remotion_public(video_path, voiceover_dir=voiceover_dir if not args.no_tts else None)

    # Also copy latest storyboard for Remotion Studio preview (auto-load in Root.tsx)
    studio_sb_path = os.path.join(REMOTION_DIR, "src", "latest_storyboard.json")
    atomic_write_json(studio_sb_path, storyboard)
    print(f"[setup] storyboard → remotion/src/latest_storyboard.json")

    # Step 5b: Prepare beats for Remotion render (add window field for Zod validation).
    # Always re-derives render_storyboard from the current storyboard (cheap).
    with open(storyboard_path) as f:
        render_storyboard = json.load(f)
    for i, beat in enumerate(render_storyboard["scenes"]):
        # Greeting beat has window=0; regular beats have window already set in Step 4.
        w = beat.get("window", 0 if beat.get("isGreeting") else i)
        vo_path = f"voiceover/scene_{w:02d}.mp3"
        real_path = os.path.join(run_dir, vo_path)
        if os.path.exists(real_path):
            beat["voiceoverPath"] = vo_path
        beat["window"] = w
        beat["endSec"] = beat.get("endSec", beat.get("startSec", 0) + beat.get("displayFrames", 30) / args.fps)
        beat["durationInFrames"] = beat.get("durationInFrames", beat.get("displayFrames", 30))
        beat["startFmt"] = beat.get("startFmt", "")
        beat["dialog"] = beat.get("dialogue", "")
    print(f"[render] {len(render_storyboard['scenes'])} beats prepared")

    # Step 6: Install Node deps
    install_remotion_deps()

    # Step 7: Render
    # Skip if complete AND output is newer than the storyboard checkpoint (not stale).
    _render_stale = (
        os.path.exists(output_mp4)
        and os.path.exists(storyboard_path)
        and os.path.getmtime(storyboard_path) > os.path.getmtime(output_mp4)
    )
    if args.resume and ps.is_complete("render") and not _render_stale:
        print(f"[render] resuming — already complete: {output_mp4}")
    else:
        if _render_stale:
            print("[render] storyboard is newer than output — re-rendering")
        ps.mark_running("render")
        step_start = time.time()
        print(f"[render] start at {_ts()}...")
        run_render(render_storyboard, output_mp4, concurrency=args.concurrency, gl=args.gl)
        ps.mark_complete("render")
        print(f"[render] done at {_ts()} — elapsed {_elapsed(step_start)}")

    pipeline_elapsed = _elapsed(pipeline_start)
    print(f"\n{'='*60}")
    print(f"[pipeline] render done at {_ts()} — total elapsed {pipeline_elapsed}")
    print(f"[pipeline] output → {output_mp4}")
    print(f"{'='*60}")

    # Step 8: Generate Premiere Pro XML/EDL for editing
    xml_path = os.path.join(run_dir, f"{run_name}_premiere.xml")
    edl_path = os.path.join(run_dir, f"{run_name}_premiere.edl")
    if args.resume and ps.is_complete("premiere") and os.path.exists(xml_path):
        print("[premiere] resuming — already complete")
    else:
        ps.mark_running("premiere")
        step_start = time.time()
        print(f"[premiere] generating XML/EDL...")
        generate_premiere_xml(render_storyboard, xml_path, video_path, project_name=run_name)
        generate_premiere_edl(render_storyboard, edl_path, project_name=run_name)
        ps.mark_complete("premiere")
        print(f"[premiere] XML → {xml_path}")
        print(f"[premiere] EDL → {edl_path}")
        print(f"[premiere] done at {_ts()} — elapsed {_elapsed(step_start)}")

    # Step 9: Generate YouTube metadata (title, description, hashtags)
    meta_path = os.path.join(run_dir, f"{run_name}_metadata.md")
    if args.resume and ps.is_complete("metadata") and os.path.exists(meta_path):
        print("[metadata] resuming — already complete")
    elif args.deepseek_key:
        from metadata_gen import generate_video_metadata
        ps.mark_running("metadata")
        try:
            generate_video_metadata(
                storyboard=render_storyboard,
                movie_title=run_name.replace("-", " ").replace("_", " "),
                api_key=args.deepseek_key,
                story_context=story_context or "",
                output_path=meta_path,
            )
            ps.mark_complete("metadata")
        except Exception as e:
            print(f"[metadata] failed: {e}", file=sys.stderr)
    else:
        ps.mark_skipped("metadata", "no deepseek key")
        print("[metadata] skipped — no DeepSeek key (pass --deepseek-key to generate)")

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
        "tts": "qwen3" if ps.is_complete("tts") else None,
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
