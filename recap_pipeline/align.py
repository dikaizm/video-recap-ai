"""
align.py — Post-TTS scene alignment via STT word timestamps + LLM matching.

For each clustered beat:
  1. Transcribe its TTS mp3 with faster-whisper (word-level timestamps).
  2. Split the narration text into phrases by punctuation.
  3. Map each phrase to a (start_sec, end_sec) window using STT word timing.
  4. Build a candidate pool of source scenes (beat's own scenes ± neighbor expansion).
  5. LLM picks the best source scene for each phrase (batched, one call per beat).
  6. Emit a `segments` array on the beat: [{startSec, displayFrames}, ...] that the
     render uses to play multiple sub-clips of the source video inside one beat.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Iterable

# Reuse DeepSeek client from narrate.py
from narrate import _http_post  # type: ignore


_PHRASE_SPLIT_RE = re.compile(r"(?<=[.,;:!?])\s+|\s+(?=but |only to |unaware |yet |however )", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

def _load_whisper(model_size: str = "tiny.en"):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_words(mp3_path: str, whisper) -> list[dict]:
    """Return [{word, start, end}, ...] for the given mp3 file."""
    segments_iter, _ = whisper.transcribe(
        mp3_path, beam_size=1, word_timestamps=True, language="en"
    )
    out: list[dict] = []
    for seg in segments_iter:
        for w in (seg.words or []):
            out.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    return out


# ---------------------------------------------------------------------------
# Phrase splitting + mapping
# ---------------------------------------------------------------------------

def split_phrases(text: str, min_words: int = 3) -> list[str]:
    """Split narration into 1–3 phrases by punctuation/conjunctions.

    Merges fragments shorter than ``min_words`` into the previous phrase so each
    phrase has enough content for visual matching.
    """
    text = text.strip()
    if not text:
        return []
    raw = [p.strip() for p in _PHRASE_SPLIT_RE.split(text) if p and p.strip()]
    if not raw:
        return [text]

    merged: list[str] = []
    for part in raw:
        wc = len(part.split())
        if merged and wc < min_words:
            merged[-1] = (merged[-1] + " " + part).strip()
        else:
            merged.append(part)
    return merged or [text]


def _word_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def map_phrases_to_times(
    phrases: list[str], stt_words: list[dict], audio_secs: float,
) -> list[tuple[float, float]]:
    """Map each narration phrase to a (start_sec, end_sec) window.

    Strategy: split STT words proportionally by phrase word counts. Robust to
    minor word mismatches between narration text and STT output.
    """
    if not phrases:
        return []
    if not stt_words:
        # Even split across audio
        n = len(phrases)
        return [(audio_secs * i / n, audio_secs * (i + 1) / n) for i in range(n)]

    counts = [max(1, _word_count(p)) for p in phrases]
    total = sum(counts)
    n_stt = len(stt_words)

    out: list[tuple[float, float]] = []
    cum = 0
    for c in counts:
        start_idx = int(round(cum * n_stt / total))
        cum += c
        end_idx = int(round(cum * n_stt / total))
        start_idx = max(0, min(n_stt - 1, start_idx))
        end_idx = max(start_idx + 1, min(n_stt, end_idx))
        start_sec = stt_words[start_idx]["start"]
        end_sec = stt_words[end_idx - 1]["end"]
        out.append((start_sec, end_sec))

    # Normalize so first phrase starts at 0 and last ends at audio_secs.
    if out:
        first_start = out[0][0]
        last_end = out[-1][1]
        if last_end > first_start:
            scale = audio_secs / (last_end - first_start)
        else:
            scale = 1.0
        out = [(max(0.0, (s - first_start) * scale),
                max(0.0, (e - first_start) * scale)) for (s, e) in out]
        # Force monotonic and final endpoint
        for i in range(1, len(out)):
            if out[i][0] < out[i - 1][1]:
                out[i] = (out[i - 1][1], out[i][1])
        out[-1] = (out[-1][0], audio_secs)
    return out


# ---------------------------------------------------------------------------
# Candidate pool + LLM matching
# ---------------------------------------------------------------------------

def _candidate_pool(
    beat: dict,
    source_index: dict[int, dict],
    source_order: list[int],
    expand: int,
) -> list[dict]:
    """Return source scenes belonging to the beat, expanded by `expand` neighbors on each side."""
    own = [s for s in beat.get("scenes", []) if "window" in s]
    if not own:
        return []
    own_windows = {s["window"] for s in own}
    positions = [source_order.index(s["window"]) for s in own if s["window"] in source_index]
    if not positions:
        return list(own)
    lo = max(0, min(positions) - expand)
    hi = min(len(source_order), max(positions) + expand + 1)

    pool: list[dict] = []
    seen: set[int] = set()
    for w in source_order[lo:hi]:
        if w in seen:
            continue
        seen.add(w)
        scene = source_index.get(w)
        if scene:
            pool.append(scene)
    # Make sure beat's own scenes are present even if filtered out of source_order
    for s in own:
        if s["window"] not in seen:
            pool.append(s)
            seen.add(s["window"])
    return pool


_MATCH_SYSTEM = (
    "You match short narration phrases to the visual scene description that best "
    "depicts them. The narration is the spoken voiceover; the candidate scenes are "
    "from a movie. Pick the candidate whose visual content most directly shows what "
    "the phrase describes (action, character state, location, key objects). "
    "Avoid reusing the same candidate for multiple phrases when better matches exist."
)


def llm_match_phrases(
    phrases: list[str],
    candidates: list[dict],
    api_key: str,
    beat_label: str = "",
) -> list[int]:
    """Return list of candidate indices, one per phrase. Falls back to even split on parse error."""
    if not candidates:
        return [0] * len(phrases)
    if len(candidates) == 1:
        return [0] * len(phrases)

    cand_lines = []
    for i, c in enumerate(candidates):
        desc = (c.get("povText") or c.get("description") or "").strip().replace("\n", " ")
        dialogue = (c.get("dialogue") or "").strip().replace("\n", " ")
        line = f"[{i}] window={c.get('window', '?')} {desc[:280]}"
        if dialogue:
            line += f" | dialogue: {dialogue[:120]}"
        cand_lines.append(line)

    phrase_lines = [f"  P{i}: \"{p}\"" for i, p in enumerate(phrases)]

    user = (
        "Candidate scenes:\n"
        + "\n".join(cand_lines)
        + "\n\nNarration phrases (in spoken order):\n"
        + "\n".join(phrase_lines)
        + f"\n\nReturn ONLY JSON of the form {{\"matches\": [<index for P0>, <index for P1>, ...]}}. "
        f"The list MUST contain exactly {len(phrases)} integer indices in [0, {len(candidates) - 1}]."
    )

    try:
        raw = _http_post(
            api_key,
            messages=[
                {"role": "system", "content": _MATCH_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=200,
            temperature=0.2,
            timeout=60,
            thinking_enabled=False,
            json_mode=True,
        )
        data = json.loads(raw)
        idxs = data.get("matches") if isinstance(data, dict) else None
        if not isinstance(idxs, list) or len(idxs) != len(phrases):
            raise ValueError(f"bad matches list: {idxs}")
        clamped = [max(0, min(len(candidates) - 1, int(x))) for x in idxs]
        return clamped
    except Exception as e:  # noqa: BLE001
        print(f"  [align{':' + beat_label if beat_label else ''}] match parse failed ({e}); even split")
        # Fallback: even split — assign phrases to candidates proportionally
        return [
            min(len(candidates) - 1, int(i * len(candidates) / max(1, len(phrases))))
            for i in range(len(phrases))
        ]


# ---------------------------------------------------------------------------
# Per-beat alignment
# ---------------------------------------------------------------------------

def align_beats(
    beats: list[dict],
    source_scenes: list[dict],
    voiceover_dir: str,
    api_key: str,
    fps: int,
    expand: int = 2,
    whisper_model_size: str = "tiny.en",
) -> list[dict]:
    """Mutate `beats` in place, attaching a `segments` list to each beat that has TTS audio.

    Each segment = {startSec, displayFrames} pointing into the source video.
    The render plays segments in sequence inside the beat while a single audio
    track covers the whole beat.
    """
    if not beats:
        return beats

    print(f"[align] loading faster-whisper ({whisper_model_size})...")
    whisper = _load_whisper(whisper_model_size)

    source_index = {s["window"]: s for s in source_scenes if "window" in s}
    source_order = [s["window"] for s in source_scenes if "window" in s]

    aligned = 0
    skipped = 0
    fallback_count = 0
    started = time.time()

    for i, beat in enumerate(beats):
        text = (beat.get("ttsText") or beat.get("narratedText") or "").strip()
        window = beat.get("window") or (i + 1)
        mp3 = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")
        if not text or not os.path.exists(mp3):
            skipped += 1
            continue

        # Intro keeps the 3 establishing visuals it was built with — phrase-aligning
        # would condense the segment count and drop visuals.
        if beat.get("isIntro"):
            skipped += 1
            continue

        try:
            stt_words = transcribe_words(mp3, whisper)
        except Exception as e:  # noqa: BLE001
            print(f"  [align] beat {window:02d} STT failed ({e}); skipping")
            skipped += 1
            continue

        beat_total_frames = int(beat.get("displayFrames") or 0)
        if beat_total_frames <= 0:
            skipped += 1
            continue
        audio_secs = beat_total_frames / fps

        phrases = split_phrases(text)
        if not phrases:
            skipped += 1
            continue

        phrase_times = map_phrases_to_times(phrases, stt_words, audio_secs)

        candidates = _candidate_pool(beat, source_index, source_order, expand)
        if not candidates:
            skipped += 1
            continue

        # Single phrase + single candidate: nothing to disambiguate, leave as one segment.
        if len(phrases) == 1 or len(candidates) == 1:
            chosen_idxs = [0] * len(phrases)
            fallback_count += 1
        else:
            chosen_idxs = llm_match_phrases(
                phrases, candidates, api_key, beat_label=f"{window:02d}"
            )

        # Build segments
        segments: list[dict] = []
        cum_frames = 0
        for k, ((p_start, p_end), idx) in enumerate(zip(phrase_times, chosen_idxs)):
            seg_secs = max(0.5, p_end - p_start)  # floor at 0.5s for visibility
            seg_frames = max(1, round(seg_secs * fps))
            chosen = candidates[idx]
            segments.append({
                "startSec": float(chosen.get("startSec", 0.0)),
                "displayFrames": seg_frames,
                "sourceWindow": int(chosen.get("window", -1)),
                "phrase": phrases[k],
            })
            cum_frames += seg_frames

        # Reconcile rounding so segments sum to beat's displayFrames exactly.
        if segments:
            diff = beat_total_frames - cum_frames
            if diff != 0:
                segments[-1]["displayFrames"] = max(1, segments[-1]["displayFrames"] + diff)

        beat["segments"] = segments
        aligned += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(beats):
            elapsed = time.time() - started
            print(f"  [align] {i + 1}/{len(beats)} beats ({elapsed:.1f}s)")

    print(
        f"[align] aligned {aligned} beats, skipped {skipped}, "
        f"single-phrase fallbacks {fallback_count} — elapsed {time.time() - started:.1f}s"
    )
    return beats


def align_single_beat(
    beat: dict,
    source_scenes: list[dict],
    voiceover_dir: str,
    api_key: str,
    fps: int,
    expand: int = 2,
    whisper=None,           # pre-loaded faster-whisper model
) -> dict:
    """Re-align a single beat after narration or scene swap.

    If ``whisper`` is not provided, loads a new tiny.en model. Returns the
    updated beat dict with fresh ``segments`` attached.
    """
    if whisper is None:
        whisper = _load_whisper("tiny.en")

    text = (beat.get("ttsText") or beat.get("narratedText") or "").strip()
    window = beat.get("window", 0)
    mp3 = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")

    if not text or not os.path.exists(mp3):
        return beat

    if beat.get("isIntro"):
        return beat

    try:
        stt_words = transcribe_words(mp3, whisper)
    except Exception as e:
        print(f"  [align-single] beat {window:02d} STT failed ({e})")
        return beat

    beat_total_frames = int(beat.get("displayFrames") or 0)
    if beat_total_frames <= 0:
        return beat

    audio_secs = beat_total_frames / fps
    phrases = split_phrases(text)
    if not phrases:
        return beat

    phrase_times = map_phrases_to_times(phrases, stt_words, audio_secs)

    # Build candidate pool
    source_index = {s["window"]: s for s in source_scenes if "window" in s}
    source_order = [s["window"] for s in source_scenes if "window" in s]
    candidates = _candidate_pool(beat, source_index, source_order, expand)

    if not candidates:
        return beat

    chosen_idxs: list[int]
    if len(phrases) == 1 or len(candidates) == 1:
        chosen_idxs = [0] * len(phrases)
    else:
        chosen_idxs = llm_match_phrases(
            phrases, candidates, api_key, beat_label=f"{window:02d}"
        )

    # Build segments
    segments: list[dict] = []
    cum_frames = 0
    for k, ((p_start, p_end), idx) in enumerate(zip(phrase_times, chosen_idxs)):
        seg_secs = max(0.5, p_end - p_start)
        seg_frames = max(1, round(seg_secs * fps))
        chosen = candidates[idx]
        segments.append({
            "startSec": float(chosen.get("startSec", 0.0)),
            "displayFrames": seg_frames,
            "sourceWindow": int(chosen.get("window", -1)),
            "phrase": phrases[k],
        })
        cum_frames += seg_frames

    if segments:
        diff = beat_total_frames - cum_frames
        if diff != 0:
            segments[-1]["displayFrames"] = max(1, segments[-1]["displayFrames"] + diff)

    beat["segments"] = segments
    return beat
