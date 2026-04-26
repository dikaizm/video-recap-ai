"""
tts.py — Text-to-speech backends for voiceover generation.

Backends:
  macos      — macOS 'say' command (no API key, best quality on Apple Silicon)
  gtts       — Google TTS via gTTS library (no API key, requires pip install gtts)
  elevenlabs — ElevenLabs REST API (requires --elevenlabs-key)
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


def generate_batch(
    scenes: list[dict],
    voiceover_dir: str,
    backend: str,
    **kwargs,
) -> list[str]:
    os.makedirs(voiceover_dir, exist_ok=True)
    paths = []

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
        else:
            raise ValueError(f"Unknown TTS backend: {backend}")

        paths.append(out_path)

    return paths


