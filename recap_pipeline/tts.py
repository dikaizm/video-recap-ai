"""
tts.py — Qwen3-TTS voiceover generation via mlx_audio (Apple Silicon).

Generates per-scene MP3 files from a single continuous TTS stream:
  1. Concatenate all scene narrations into one text.
  2. Generate one TTS call for the entire duration (consistent voice).
  3. Chop into individual scene_{N}.mp3 via ffmpeg
     using word-count-proportional timestamps.
"""
import os
import shutil
import subprocess
import tempfile


def _resolve_qwen3_model_path(model_path: str) -> str:
    """Resolve snapshots sub-directory if present (HuggingFace cache layout)."""
    snapshots = os.path.join(model_path, "snapshots")
    if os.path.isdir(snapshots):
        subs = [f for f in os.listdir(snapshots) if not f.startswith(".")]
        if subs:
            return os.path.join(snapshots, subs[0])
    return model_path


def generate_qwen3_tts(
    text: str,
    output_mp3_path: str,
    model,                          # pre-loaded mlx_audio model
    speaker: str | None = "Ryan",
    instruct: str = "warm",
    speed: float = 1.0,
    mode: str = "custom",           # "custom" | "design"
    sample_rate: int = 24000,
) -> None:
    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError:
        raise ImportError("Install mlx_audio: pip install mlx_audio")

    tmp_dir = tempfile.mkdtemp(prefix="qwen3tts_")
    try:
        if mode == "design":
            generate_audio(model=model, text=text, instruct=instruct,
                           speed=speed, output_path=tmp_dir)
        else:
            generate_audio(model=model, text=text, voice=speaker,
                           instruct=instruct, speed=speed, output_path=tmp_dir)

        wav_path = os.path.join(tmp_dir, "audio_000.wav")
        if not os.path.exists(wav_path):
            raise RuntimeError(f"Qwen3 TTS produced no output in {tmp_dir}")

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-ar", str(sample_rate), "-q:a", "4", output_mp3_path],
            check=True, capture_output=True,
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_qwen3_model(model_path: str):
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError:
        raise ImportError("Install mlx_audio: pip install mlx_audio")
    resolved = _resolve_qwen3_model_path(model_path)
    print(f"[tts] loading Qwen3 model from {resolved}")
    return load_model(resolved)




def _count_words(text: str) -> int:
    return len(text.split())


def _get_mp3_duration(mp3_path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", mp3_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _chop_audio(
    scenes: list[dict],
    audio_path: str,
    audio_duration: float,
    voiceover_dir: str,
) -> None:
    total_words = sum(
        _count_words(s.get("povText") or s.get("description", ""))
        for s in scenes
    )
    if total_words == 0:
        return

    cumulative_words = 0
    for scene in scenes:
        text = scene.get("povText") or scene.get("description", "")
        scene_words = _count_words(text)
        window = scene["window"]
        out_path = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")

        if os.path.exists(out_path):
            print(f"  skip scene {window:02d} (exists)")
            cumulative_words += scene_words
            continue

        start_sec = (cumulative_words / total_words) * audio_duration
        end_sec = ((cumulative_words + scene_words) / total_words) * audio_duration
        duration = end_sec - start_sec

        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", f"{start_sec:.3f}",
             "-t", f"{duration:.3f}",
             "-q:a", "4",
             out_path],
            check=True, capture_output=True,
        )

        print(f"  chopped scene {window:02d}.mp3 ({start_sec:.1f}s–{end_sec:.1f}s)")
        cumulative_words += scene_words


# ── main entry point ─────────────────────────────────────────────────


def generate_batch(
    scenes: list[dict],
    voiceover_dir: str,
    model_path: str,
    **kwargs,
) -> list[str]:
    """Generate per-scene MP3 voiceovers from a single continuous TTS stream.

    Concatenates all scene narrations, generates one TTS call for
    consistent voice timbre across the entire duration, then chops into
    individual scene_{N}.mp3 files using word-count-proportional timestamps.
    """
    os.makedirs(voiceover_dir, exist_ok=True)

    qwen3_model = load_qwen3_model(model_path)

    scenes_with_text: list[dict] = []
    for scene in scenes:
        text = scene.get("povText") or scene.get("description", "")
        if text.strip():
            scenes_with_text.append(scene)

    if not scenes_with_text:
        return [""] * len(scenes)

    concatenated_text = " ".join(
        s.get("povText") or s.get("description", "") for s in scenes_with_text
    )
    word_count = _count_words(concatenated_text)
    print(f"[tts] {len(scenes_with_text)} scenes, {word_count} words — generating full audio…")

    tmp_path = os.path.join(voiceover_dir, "_full.mp3")
    generate_qwen3_tts(concatenated_text, tmp_path, model=qwen3_model, **kwargs)

    audio_duration = _get_mp3_duration(tmp_path)
    print(f"[tts] {audio_duration:.1f}s audio — chopping into {len(scenes_with_text)} scene(s)")
    _chop_audio(scenes_with_text, tmp_path, audio_duration, voiceover_dir)
    os.unlink(tmp_path)

    window_to_path: dict[int, str] = {}
    for scene in scenes_with_text:
        w = scene["window"]
        p = os.path.join(voiceover_dir, f"scene_{w:02d}.mp3")
        if os.path.exists(p):
            window_to_path[w] = p

    return [window_to_path.get(s["window"], "") for s in scenes]


