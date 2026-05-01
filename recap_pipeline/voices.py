"""voices.py — Genre-based TTS voice presets with multi-genre blending."""
from __future__ import annotations

VOICE_PRESETS: dict[str, str] = {
    "thriller": (
        "A male voice in his late thirties, low-pitched and controlled. "
        "Cold, measured delivery — every word deliberate, tension held just below the surface. "
        "Minimal inflection, sparse pauses, as if danger could arrive mid-sentence. "
        "Gravelly undertone, clipped consonants, authoritative. "
        "Ideal for psychological thrillers and noir narration."
    ),
    "sci-fi": (
        "A male voice in his early thirties, clear and precise with a slightly formal tone. "
        "Even, unhurried pace — detached and analytical, as if reporting from a distant vantage. "
        "Clean diction, neutral pitch, subtle cosmic weight behind each phrase. "
        "No emotional warmth but no coldness — pure clarity. "
        "Ideal for science fiction: space opera, dystopian, or hard sci-fi."
    ),
    "horror": (
        "A male voice, mid-thirties, hushed and deliberately slow. "
        "Pitch drops slightly at the end of each phrase, as if reluctant to finish the thought. "
        "Breathy, intimate, close-mic quality — as if whispering a warning. "
        "Long pauses before key reveals, slight rasp in the lower register. "
        "Ideal for horror, psychological dread, and supernatural suspense."
    ),
    "action": (
        "A male voice, late twenties, energetic and punchy. "
        "Fast clip between phrases, sharp consonants, forward-leaning momentum. "
        "Raised pitch on key story beats, quick pauses for impact. "
        "Confident, no hesitation — drives the listener forward. "
        "Ideal for action-adventure, heist, war, and high-stakes thrillers."
    ),
    "drama": (
        "A male voice in his forties, warm and emotionally present. "
        "Mid-range pitch, unhurried pace — weight behind each sentence. "
        "Natural rises for revelation, soft falls for grief or resignation. "
        "Earnest, unaffected delivery — speaks as if the story genuinely matters. "
        "Ideal for character drama, tragedy, and literary adaptations."
    ),
    "comedy": (
        "A male voice in his late twenties, bright and light with easy energy. "
        "Slightly faster-than-natural pace, lifted pitch on comic beats. "
        "Playful inflection, quick recoveries after pauses. "
        "Warm and relatable — never forced or over-enthusiastic. "
        "Ideal for comedy, romantic comedy, and light adventure."
    ),
    "romance": (
        "A male voice in his early thirties, soft and warm with gentle intimacy. "
        "Slow, tender pacing — lingers on emotional phrases. "
        "Breathy quality, slightly hushed, as if sharing a private moment. "
        "Smooth transitions, no sharp edges in delivery. "
        "Ideal for romance, drama with romantic themes, and period pieces."
    ),
    "fantasy": (
        "A male voice in his forties, rich baritone with epic weight. "
        "Measured, ceremonial pacing — each sentence feels like a proclamation. "
        "Full resonance, confident projection, slight archaic formality. "
        "Rises for heroic moments, falls for tragedy. "
        "Ideal for fantasy, mythology, historical epic, and adventure."
    ),
    "documentary": (
        "A young adult male voice, mid-twenties, with a soft and warm tone. "
        "Light, gentle timbre — smooth and approachable, never harsh or booming. "
        "Measured, calm delivery with natural pauses at commas and periods. "
        "Subtle dynamic range: slightly more intense for tension, softer for resolution. "
        "Intimate, conversational quality — as if speaking directly to one listener. "
        "Ideal for documentary narration: clear, soft, and engaging throughout."
    ),
}

GENRE_ALIASES: dict[str, str] = {
    "science-fiction": "sci-fi",
    "science_fiction": "sci-fi",
    "scifi": "sci-fi",
    "suspense": "thriller",
    "psychological": "thriller",
    "noir": "thriller",
    "action-adventure": "action",
    "adventure": "action",
    "war": "action",
    "heist": "action",
    "supernatural": "horror",
    "romantic": "romance",
    "epic": "fantasy",
    "historical": "fantasy",
    "period": "fantasy",
    "default": "documentary",
}

_DETECT_SYSTEM = """\
Classify the movie's genres from this synopsis.
Choose UP TO 3 from this list (ranked by dominance, most dominant first):
  thriller, sci-fi, horror, action, drama, comedy, romance, fantasy, documentary

Rules:
- List only genres that genuinely apply — do not pad to 3 if fewer fit.
- Order by how strongly each genre defines the movie's overall tone.
- Use exact names from the list above.

Return ONLY a JSON object: {"genres": ["<primary>", "<secondary>", ...]}
No explanation, no preamble.\
"""

_BLEND_SYSTEM = """\
You write voice design prompts for a text-to-speech model.

You are given 2 or 3 genre-specific voice descriptions. Your task is to blend them into
ONE coherent voice description that honours ALL detected genres simultaneously.

Rules:
- The result must be ONE paragraph of 4-6 sentences.
- Describe ONLY objective, perceptual voice qualities: age, pitch, pace, texture, delivery style.
- Weight the PRIMARY genre (listed first) most heavily — it sets the base character.
- Incorporate secondary genres as modifying layers, not replacements.
- Never mention genre names in the output.
- End with: "Ideal for <adjective> narration that balances <quality A> and <quality B>."
- Under 2048 characters total.

Return ONLY the voice description text. No JSON, no labels.\
"""


def _resolve_alias(genre: str) -> str:
    genre = genre.lower().strip()
    return GENRE_ALIASES.get(genre, genre)


def detect_genres(story_context: str, api_key: str) -> list[str]:
    """Return a ranked list of 1–3 genre keys from VOICE_PRESETS for the given synopsis."""
    if not story_context.strip():
        return ["documentary"]
    try:
        import json
        from narrate import _http_post  # type: ignore
        raw = _http_post(
            api_key,
            messages=[
                {"role": "system", "content": _DETECT_SYSTEM},
                {"role": "user", "content": story_context.strip()[:3000]},
            ],
            max_tokens=60,
            temperature=0.1,
            timeout=30,
            thinking_enabled=False,
            json_mode=True,
        )
        data = json.loads(raw)
        raw_genres = data.get("genres", [])
        if not isinstance(raw_genres, list):
            raise ValueError(f"bad genres field: {raw_genres}")
        resolved: list[str] = []
        seen: set[str] = set()
        for g in raw_genres[:3]:
            g = _resolve_alias(str(g))
            if g in VOICE_PRESETS and g not in seen:
                resolved.append(g)
                seen.add(g)
        return resolved if resolved else ["documentary"]
    except Exception as e:
        print(f"  [voices] genre detect failed ({e}); using 'documentary'")
        return ["documentary"]


def blend_voice_instruct(genres: list[str], api_key: str | None = None) -> str:
    """Return a blended TTS instruct string for a list of genre keys.

    Single genre → return preset directly (no LLM call).
    Multiple genres → ask LLM to fuse the preset descriptions into one coherent instruct.
    Falls back to primary genre preset if LLM call fails.
    """
    genres = [_resolve_alias(g) for g in genres]
    genres = [g for g in genres if g in VOICE_PRESETS]
    if not genres:
        return VOICE_PRESETS["documentary"]
    if len(genres) == 1:
        return VOICE_PRESETS[genres[0]]

    # Build the blending prompt
    parts: list[str] = []
    labels = ["PRIMARY", "SECONDARY", "TERTIARY"]
    for label, g in zip(labels, genres):
        parts.append(f"{label} ({g}):\n{VOICE_PRESETS[g]}")
    user_content = "\n\n".join(parts) + "\n\nBlend these into one unified voice description."

    if not api_key:
        # No LLM available — return primary preset
        print(f"  [voices] no api_key for blend; using primary genre '{genres[0]}'")
        return VOICE_PRESETS[genres[0]]

    try:
        from narrate import _http_post  # type: ignore
        blended = _http_post(
            api_key,
            messages=[
                {"role": "system", "content": _BLEND_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=0.3,
            timeout=45,
            thinking_enabled=False,
            json_mode=False,
        )
        blended = blended.strip()
        if len(blended) < 50:
            raise ValueError(f"blend response too short: {blended!r}")
        return blended
    except Exception as e:
        print(f"  [voices] blend failed ({e}); using primary genre '{genres[0]}'")
        return VOICE_PRESETS[genres[0]]


def resolve_voice_instruct(
    story_context: str,
    api_key: str,
) -> tuple[str, list[str]]:
    """Detect genres from story context and return (instruct_str, detected_genres).

    This is the main entry point for auto-detection.
    - Detects 1–3 genres from the synopsis
    - Blends their presets into one instruct string (LLM-fused for multi-genre)
    """
    genres = detect_genres(story_context, api_key)
    instruct = blend_voice_instruct(genres, api_key)
    return instruct, genres


def get_voice_instruct(genre: str) -> str:
    """Return the TTS instruct string for a single genre key (no blending)."""
    genre = _resolve_alias(genre)
    return VOICE_PRESETS.get(genre, VOICE_PRESETS["documentary"])


# Keep old name for backwards compat
def detect_genre(story_context: str, api_key: str) -> str:
    """Detect primary genre only. Prefer resolve_voice_instruct for multi-genre support."""
    return detect_genres(story_context, api_key)[0]
