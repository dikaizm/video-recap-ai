"""greeting.py — Build a short branded channel greeting beat prepended before the intro.

The greeting:
  - Speaks a short, catchy channel signature line (5-12 words).
  - Displays a branded title card (handled by Remotion's GreetingCard component).
  - Is prepended as beat 0 (window=0, scene_00.mp3) before everything else.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

CHANNEL_NAME = "Premiere Roll"

# Fixed signature lines — same catchphrase across all episodes for brand recognition.
# Index 0 is the default. Can be overridden via --greeting-text.
SIGNATURE_LINES = [
    "Welcome to Premiere Roll — every story, perfectly framed.",
    "Premiere Roll — where every movie gets its close-up.",
    "This is Premiere Roll. Lights, camera, recap.",
    "You're watching Premiere Roll — every story retold.",
]

DEFAULT_GREETING = SIGNATURE_LINES[0]

# TTS text strips the em-dash for more natural speech phrasing
_TTS_CLEANUPS = [(" — ", ". "), (" — ", ", ")]


def _clean_for_tts(text: str) -> str:
    """Make greeting text more natural for TTS by replacing punctuation."""
    return text.replace(" — ", ". ").replace("—", ".")


def build_greeting_beat(
    fps: int,
    voiceover_dir: str,
    model_path: str,    # path to Qwen3 model directory
    greeting_text: str = DEFAULT_GREETING,
    tts_instruct: str = (
        "A young adult male voice, mid-twenties, warm and confident. "
        "Clear, friendly delivery — like a host welcoming the audience. "
        "Slightly upbeat, natural pace, as if speaking directly to the viewer."
    ),
    tts_speed: float = 1.0,
    padding_frames: int = 15,   # extra silent frames after audio ends
) -> dict:
    """Generate TTS for the greeting and return a beat dict with isGreeting=True.

    The beat is assigned window=0, which maps to voiceover/scene_00.mp3.
    Remotion renders this beat as a GreetingCard (title card, no source video).
    """
    from tts import generate_qwen3_tts, load_qwen3_model

    os.makedirs(voiceover_dir, exist_ok=True)
    mp3_path = os.path.join(voiceover_dir, "scene_00.mp3")

    tts_text = _clean_for_tts(greeting_text)
    print(f"[greeting] generating TTS: \"{tts_text}\"")

    tts_model = load_qwen3_model(model_path)
    generate_qwen3_tts(
        text=tts_text,
        output_mp3_path=mp3_path,
        model=tts_model,
        instruct=tts_instruct,
        speed=tts_speed,
    )

    # Measure audio duration
    audio_secs = _mp3_duration(mp3_path)
    display_secs = audio_secs + padding_frames / fps
    display_frames = max(int(display_secs * fps), fps * 3)  # min 3s

    print(f"[greeting] audio={audio_secs:.2f}s, displayFrames={display_frames} ({display_frames/fps:.1f}s)")

    return {
        "isGreeting": True,
        "window": 0,
        "startSec": 0.0,
        "endSec": round(display_frames / fps, 3),
        "durationInFrames": display_frames,
        "displayFrames": display_frames,
        "povText": f"[GREETING] {greeting_text}",
        "narratedText": greeting_text,
        "ttsText": tts_text,
        "dialogue": "",
        "emotion": "",
        "startFmt": "00:00:00",
        "segments": [],
        "channelName": CHANNEL_NAME,
    }


def _mp3_duration(mp3_path: str) -> float:
    """Return duration in seconds of an mp3 file via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                mp3_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        # Fallback: estimate from word count (~3 wps)
        return max(3.0, len(mp3_path.split()) / 3.0)
