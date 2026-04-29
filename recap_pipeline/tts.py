"""
tts.py — Qwen3-TTS voiceover generation via mlx_audio (Apple Silicon).

Generates per-scene MP3 files from a continuous TTS stream:
  1. Concatenate all scene narrations into one text.
  2. Split into ~2-minute chunks (memory constraint).
  3. Generate one continuous TTS audio per chunk (consistent voice).
  4. Chop each chunk into individual scene_{N}.mp3 via ffmpeg
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


# ── continuous-mode helpers ───────────────────────────────────────────

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


def _split_into_chunks(
    scenes: list[dict],
    max_chunk_sec: int = 120,
    wpm: int = 200,
) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_words = 0
    words_per_sec = wpm / 60

    for scene in scenes:
        text = scene.get("povText") or scene.get("description", "")
        if not text.strip():
            continue
        scene_words = _count_words(text)
        scene_sec = scene_words / words_per_sec

        if current_chunk and current_words / words_per_sec + scene_sec > max_chunk_sec:
            chunks.append(current_chunk)
            current_chunk = []
            current_words = 0

        current_chunk.append(scene)
        current_words += scene_words

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _chop_group(
    group: list[dict],
    chunk_audio_path: str,
    chunk_duration: float,
    voiceover_dir: str,
    wpm: int = 200,
) -> None:
    total_words = sum(
        _count_words(s.get("povText") or s.get("description", ""))
        for s in group
    )
    if total_words == 0:
        return

    cumulative_words = 0
    for scene in group:
        text = scene.get("povText") or scene.get("description", "")
        scene_words = _count_words(text)
        window = scene["window"]
        out_path = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")

        if os.path.exists(out_path):
            print(f"  skip scene {window:02d} (exists)")
            cumulative_words += scene_words
            continue

        start_sec = (cumulative_words / total_words) * chunk_duration
        end_sec = ((cumulative_words + scene_words) / total_words) * chunk_duration
        duration = end_sec - start_sec

        subprocess.run(
            ["ffmpeg", "-y", "-i", chunk_audio_path,
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
    """Generate per-scene MP3 voiceovers from a continuous TTS stream.

    Concatenates all scene narrations, splits into ~2-minute chunks
    (memory constraint), generates one TTS call per chunk for consistent
    voice timbre, then chops each chunk into individual scene_{N}.mp3
    files using word-count-proportional timestamps.
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

    chunk_groups = _split_into_chunks(scenes_with_text, max_chunk_sec=120, wpm=200)
    print(f"[tts] {len(scenes_with_text)} scenes → {len(chunk_groups)} chunk(s)")

    for chunk_idx, group in enumerate(chunk_groups):
        tag = f"chunk {chunk_idx + 1}/{len(chunk_groups)}"
        concatenated_text = " ".join(
            s.get("povText") or s.get("description", "") for s in group
        )
        print(f"[tts] {tag}: generating {_count_words(concatenated_text)} words continuously…")

        chunk_path = os.path.join(voiceover_dir, f"_chunk_{chunk_idx:02d}.mp3")
        generate_qwen3_tts(concatenated_text, chunk_path, model=qwen3_model, **kwargs)

        chunk_duration = _get_mp3_duration(chunk_path)
        print(f"[tts] {tag}: {chunk_duration:.1f}s audio — chopping into {len(group)} scene(s)")
        _chop_group(group, chunk_path, chunk_duration, voiceover_dir, wpm=200)

        os.unlink(chunk_path)

    window_to_path: dict[int, str] = {}
    for scene in scenes_with_text:
        w = scene["window"]
        p = os.path.join(voiceover_dir, f"scene_{w:02d}.mp3")
        if os.path.exists(p):
            window_to_path[w] = p

    return [window_to_path.get(s["window"], "") for s in scenes]


