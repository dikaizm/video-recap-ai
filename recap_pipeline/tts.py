"""
tts.py — Text-to-speech backends for voiceover generation.

Backends:
  macos      — macOS 'say' command (no API key, best quality on Apple Silicon)
  gtts       — Google TTS via gTTS library (no API key, requires pip install gtts)
  elevenlabs — ElevenLabs REST API (requires --elevenlabs-key)
  qwen3      — Qwen3-TTS via mlx_audio (local, Apple Silicon, requires mlx_audio)
"""
import json
import os
import subprocess
import tempfile


def generate_macos_tts(
    text: str,
    output_mp3_path: str,
    voice: str = "Samantha",
    rate: int = 175,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tf:
        tf.write(text)
        txt_path = tf.name

    aiff_path = txt_path.replace(".txt", ".aiff")
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-f", txt_path, "-o", aiff_path],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path, "-q:a", "4", "-ar", "44100", output_mp3_path],
            check=True,
            capture_output=True,
        )
    finally:
        for p in (txt_path, aiff_path):
            if os.path.exists(p):
                os.unlink(p)


def generate_gtts(
    text: str,
    output_mp3_path: str,
    lang: str = "en",
    slow: bool = False,
) -> None:
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("Install gTTS: pip install gtts")

    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(output_mp3_path)


def generate_elevenlabs(
    text: str,
    output_mp3_path: str,
    api_key: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_monolingual_v1",
) -> None:
    import urllib.request
    import urllib.error

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = json.dumps({
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    with urllib.request.urlopen(req) as resp:
        with open(output_mp3_path, "wb") as f:
            f.write(resp.read())


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
    speaker: str = "Aiden",
    instruct: str = "Documentary narrator, calm and clear",
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


def generate_batch(
    scenes: list[dict],
    voiceover_dir: str,
    backend: str,
    **kwargs,
) -> list[str]:
    os.makedirs(voiceover_dir, exist_ok=True)
    paths = []

    # Load Qwen3 model once before the loop
    qwen3_model = None
    if backend == "qwen3":
        model_path = kwargs.pop("model_path", None)
        if not model_path:
            raise ValueError("qwen3 backend requires model_path kwarg")
        qwen3_model = load_qwen3_model(model_path)

    for scene in scenes:
        window = scene["window"]
        out_path = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")

        if os.path.exists(out_path):
            print(f"  skip scene {window:02d} (exists)")
            paths.append(out_path)
            continue

        text = scene.get("povText") or scene.get("description", "")
        if not text:
            paths.append("")
            continue

        print(f"  TTS scene {window:02d} ({backend})...")
        if backend == "macos":
            generate_macos_tts(text, out_path, **kwargs)
        elif backend == "gtts":
            generate_gtts(text, out_path, **kwargs)
        elif backend == "elevenlabs":
            generate_elevenlabs(text, out_path, **kwargs)
        elif backend == "qwen3":
            generate_qwen3_tts(text, out_path, model=qwen3_model, **kwargs)
        else:
            raise ValueError(f"Unknown TTS backend: {backend}")

        paths.append(out_path)

    return paths


