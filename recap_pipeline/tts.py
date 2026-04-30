"""
tts.py — Qwen3-TTS voiceover generation via mlx_audio (Apple Silicon).

Generates per-scene MP3 files from a single continuous TTS stream:
  1. Concatenate all scene narrations into one text.
  2. Generate one TTS call for the entire duration (consistent voice).
  3. trim into individual scene_{N}.mp3 via ffmpeg
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
    # Voice Design Guidelines:
    # - Be specific: "deep, low-pitched" not "nice voice"
    # - Be multidimensional: combine gender, age, pitch, pace, emotion, characteristics, purpose
    # - Be objective: describe physical/perceptual features, not preferences
    # - Be concise: under 2048 chars, every word adds meaning
    # - Supported: Chinese and English only
    instruct: str = (
        "A middle-aged male voice with a deep, low-pitched tone and magnetic quality. "
        "Brisk, efficient pace at 1.3x speed for recap narration. "
        "Rich vocal texture, serious yet calm emotional register. "
        "Ideal for documentary narration and thriller storytelling."
    ),
    speed: float = 1.0,
    sample_rate: int = 24000,
    temperature: float = 0.3,       # lower = more consistent voice across scenes
    max_tokens: int = 2048,         # 400 words ~1200 tokens; leave headroom
) -> None:
    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError:
        raise ImportError("Install mlx_audio: pip install mlx_audio")

    tmp_dir = tempfile.mkdtemp(prefix="qwen3tts_")
    try:
        # Always use voice design mode (instruct-only, no speaker)
        generate_audio(
            model=model,
            text=text,
            speed=speed,
            output_path=tmp_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            instruct=instruct
        )

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


def _transcribe_word_timestamps(mp3_path: str) -> list[dict]:
    """Transcribe audio with word-level timestamps using faster-whisper.

    Returns list of {"word": str, "start": float, "end": float}.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("Install faster-whisper: pip install faster-whisper")

    print("  [trim] transcribing full audio for word timestamps…", flush=True)
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(mp3_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"word": w.word, "start": w.start, "end": w.end})
    print(f"  [trim] {len(words)} words transcribed", flush=True)
    return words


def _scene_text(scene: dict) -> str:
    return scene.get("narratedText") or scene.get("povText") or scene.get("description", "")


def _proportional_boundaries(scenes: list[dict], audio_duration: float) -> list[float]:
    """Word-count-proportional fallback when STT or LLM is unavailable."""
    total = sum(_count_words(_scene_text(s)) for s in scenes)
    if not total:
        return []
    cumulative, boundaries = 0, []
    for scene in scenes[:-1]:
        cumulative += _count_words(_scene_text(scene))
        boundaries.append((cumulative / total) * audio_duration)
    return boundaries


def _stt_boundaries(
    scenes: list[dict],
    words: list[dict],
    audio_duration: float,
    padding_sec: float = 0.1,
) -> list[float]:
    """Proportional indexing into STT word list — no api_key required."""
    import re as _re
    def _wc(text: str) -> int:
        return len(_re.sub(r"[^\w\s]", "", text).split())

    scene_wcs = [_wc(_scene_text(s)) for s in scenes]
    total_narration_words = sum(scene_wcs)
    total_stt = len(words)

    if total_narration_words == 0 or total_stt == 0:
        return []

    boundaries: list[float] = []
    cumulative = 0

    for i, scene in enumerate(scenes[:-1]):
        cumulative += scene_wcs[i]
        proportion = cumulative / total_narration_words
        stt_idx = min(int(proportion * total_stt), total_stt - 1)
        cut = words[stt_idx]["end"] + padding_sec
        if boundaries and cut <= boundaries[-1]:
            cut = boundaries[-1] + 0.05
        cut = min(cut, audio_duration - 0.05)
        boundaries.append(round(cut, 3))
        print(f"    scene {scene['window']:02d} boundary @ {cut:.2f}s "
              f"(word[{stt_idx}]: {words[stt_idx]['word'].strip()!r})", flush=True)

    return boundaries


def _llm_align_boundaries(
    scenes: list[dict],
    words: list[dict],
    audio_duration: float,
    api_key: str,
    padding_sec: float = 0.1,
) -> list[float]:
    """Use DeepSeek to align STT words against known narration, return N-1 cut timestamps."""
    import json as _json
    import re as _re
    import urllib.request

    word_lines = "\n".join(f"{i}: {w['word'].strip()!r}" for i, w in enumerate(words))
    scene_lines = "\n".join(
        f"Scene {s['window']:02d}: {_scene_text(s).strip()}"
        for s in scenes
    )
    n_boundaries = len(scenes) - 1
    prompt = (
        f"TASK: Find where each scene ends in the transcript.\n\n"
        f"There are {len(scenes)} scenes and {len(words)} transcript words (indices 0–{len(words)-1}).\n"
        f"You must return EXACTLY {n_boundaries} integers — one per scene boundary.\n\n"
        f"TRANSCRIPT WORDS (index: word):\n{word_lines}\n\n"
        f"SCENE NARRATIONS (spoken in order):\n{scene_lines}\n\n"
        f"For each scene boundary i (where i goes from 1 to {n_boundaries}), "
        f"return the transcript word INDEX of the last word in scene i.\n"
        f"Output ONLY a JSON array of exactly {n_boundaries} integers, nothing else.\n"
        f"Example for 3 scenes (2 boundaries): [12, 28]"
    )

    payload = _json.dumps({
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": f"You find scene boundaries in a speech transcript. Return ONLY a JSON array of exactly {n_boundaries} integers."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max(128, n_boundaries * 6),
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = _json.loads(resp.read())

    raw = result["choices"][0]["message"]["content"].strip()
    match = _re.search(r"\[[\d,\s]+\]", raw)
    if not match:
        raise ValueError(f"LLM returned unparseable response: {raw[:200]}")

    indices = _json.loads(match.group())
    if len(indices) != n_boundaries:
        raise ValueError(f"Expected {n_boundaries} indices, got {len(indices)}")

    boundaries: list[float] = []
    for i, idx in enumerate(indices):
        idx = max(0, min(int(idx), len(words) - 1))
        cut = words[idx]["end"] + padding_sec
        if boundaries and cut <= boundaries[-1]:
            cut = boundaries[-1] + 0.05
        cut = min(cut, audio_duration - 0.05)
        boundaries.append(round(cut, 3))
        scene = scenes[i]
        print(f"    scene {scene['window']:02d} boundary @ {cut:.2f}s "
              f"(word[{idx}]: {words[idx]['word'].strip()!r})", flush=True)

    return boundaries


def _chop_audio(
    scenes: list[dict],
    audio_path: str,
    audio_duration: float,
    voiceover_dir: str,
    api_key: str | None = None,
) -> list[dict]:
    """Chop full audio into per-scene MP3s. Returns list of {window, start, end} dicts."""
    if not scenes:
        return []

    words = _transcribe_word_timestamps(audio_path)

    if not words:
        print("  [trim] STT returned no words — using word-count proportional fallback")
        boundaries = _proportional_boundaries(scenes, audio_duration)
    elif api_key:
        try:
            print("  [trim] aligning with LLM…", flush=True)
            boundaries = _llm_align_boundaries(scenes, words, audio_duration, api_key)
            print("  [trim] LLM alignment done")
        except Exception as e:
            print(f"  [trim] LLM alignment failed ({e}) — falling back to proportional STT")
            boundaries = _stt_boundaries(scenes, words, audio_duration)
    else:
        boundaries = _stt_boundaries(scenes, words, audio_duration)

    starts = [0.0] + boundaries
    ends   = boundaries + [audio_duration]
    segments: list[dict] = []

    for scene, start_sec, end_sec in zip(scenes, starts, ends):
        w = scene["window"]
        out_path = os.path.join(voiceover_dir, f"scene_{w:02d}.mp3")
        segments.append({"window": w, "start": round(start_sec, 3), "end": round(end_sec, 3)})

        if os.path.exists(out_path):
            print(f"  skip scene {w:02d} (exists)")
            continue

        duration = max(0.05, end_sec - start_sec)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", f"{start_sec:.3f}",
             "-t",  f"{duration:.3f}",
             "-q:a", "4",
             out_path],
            check=True, capture_output=True,
        )
        print(f"  trimmed scene {w:02d}.mp3 ({start_sec:.1f}s–{end_sec:.1f}s)")

    return segments


# ── main entry point ─────────────────────────────────────────────────


_TTS_CHUNK_WORDS = 400  # Qwen3-TTS handles 500+ words reliably; larger chunks = fewer seams


def _split_into_chunks(scenes: list[dict], max_words: int) -> list[list[dict]]:
    """Group scenes into chunks that each stay under max_words total."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0
    for scene in scenes:
        wc = _count_words(_scene_text(scene))
        if current and current_words + wc > max_words:
            chunks.append(current)
            current = []
            current_words = 0
        current.append(scene)
        current_words += wc
    if current:
        chunks.append(current)
    return chunks


def _concat_mp3s(input_paths: list[str], output_path: str) -> None:
    """Concatenate multiple MP3 files into one using ffmpeg concat demuxer."""
    list_file = os.path.abspath(output_path) + ".list.txt"
    try:
        with open(list_file, "w") as f:
            for p in input_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_file, "-q:a", "4", output_path],
            check=True, capture_output=True,
        )
    finally:
        if os.path.exists(list_file):
            os.unlink(list_file)


def generate_batch(
    scenes: list[dict],
    voiceover_dir: str,
    model_path: str,
    api_key: str | None = None,
    **kwargs,
) -> list[str]:
    """Generate per-scene MP3 voiceovers — one TTS call per scene.

    No chunking, no chopping, no word-count alignment. Each scene gets its own TTS
    generation call, so timing is exact and the audio for scene N matches the video
    for scene N precisely.
    """
    os.makedirs(voiceover_dir, exist_ok=True)

    qwen3_model = load_qwen3_model(model_path)

    scenes_with_text: list[dict] = []
    for scene in scenes:
        text = _scene_text(scene)
        if text.strip():
            scenes_with_text.append(scene)

    if not scenes_with_text:
        return [""] * len(scenes)

    total_words = sum(_count_words(_scene_text(s)) for s in scenes_with_text)
    total = len(scenes_with_text)
    print(f"[tts] {total} scenes, {total_words} words — per-scene generation")

    window_to_path: dict[int, str] = {}
    for i, scene in enumerate(scenes_with_text):
        w = scene["window"]
        text = _scene_text(scene)
        wc = _count_words(text)
        out_path = os.path.join(voiceover_dir, f"scene_{w:02d}.mp3")

        if os.path.exists(out_path):
            print(f"  skip scene {w:02d}/{total} (exists)")
            window_to_path[w] = out_path
            continue

        print(f"  scene {w:02d}/{total} ({wc} words)...")
        generate_qwen3_tts(text, out_path, model=qwen3_model, **kwargs)
        window_to_path[w] = out_path

        # Save progress incrementally — stamp ttsText
        scene["ttsText"] = text

        if i % 20 == 0:
            import json as _json
            manifest = {
                "totalWords": total_words,
                "scenes": [
                    {"window": s["window"], "ttsText": _scene_text(s),
                     "audioPath": f"scene_{s['window']:02d}.mp3"}
                    for s in scenes_with_text[:i+1]
                ],
            }
            manifest_path = os.path.join(voiceover_dir, "tts_manifest.json")
            with open(manifest_path, "w") as f:
                _json.dump(manifest, f, indent=2)

    # Stamp ttsText on all scenes
    for scene in scenes_with_text:
        scene["ttsText"] = _scene_text(scene)

    # Write final manifest
    import json as _json
    manifest = {
        "totalWords": total_words,
        "scenes": [
            {"window": s["window"], "ttsText": _scene_text(s),
             "audioPath": f"scene_{s['window']:02d}.mp3"}
            for s in scenes_with_text
        ],
    }
    manifest_path = os.path.join(voiceover_dir, "tts_manifest.json")
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    print(f"[tts] manifest → {manifest_path}")

    return [window_to_path.get(s["window"], "") for s in scenes]


