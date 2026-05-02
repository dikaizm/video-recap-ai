"""
thumbnail.py — Step 10: Generate a high-CTR YouTube thumbnail.

3-Element Rule:
  1. Face — best emotional close-up frame from the climax range
  2. Action — that frame IS the background (no composite needed)
  3. Text — exactly 2-3 ALL CAPS words via DeepSeek (or fallback map)

Squint Test: text must read at postage-stamp size → big font + black stroke.
"""
import json
import os

import av
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Arial Bold.ttf",
]

STRONG_EMOTIONS = frozenset({
    "fear", "shock", "panic", "horror", "terror", "dread",
    "confrontation", "danger", "suspense", "threat", "alarm",
    "distress", "anxiety", "menace",
})

CLOSE_UP_SIGNALS = frozenset({
    "close-up", "close up", "closeup", "face", "portrait",
    "foreground figure", "extreme close",
})

FALLBACK_HOOK_MAP: dict[str, str] = {
    "fear":          "NO ESCAPE",
    "shock":         "HE KNEW",
    "horror":        "STAY QUIET",
    "terror":        "NO ESCAPE",
    "panic":         "TOO LATE",
    "suspense":      "THE TRUTH",
    "danger":        "TOO LATE",
    "dread":         "THEY LIED",
    "confrontation": "FACE OFF",
    "isolation":     "LEFT BEHIND",
    "grief":         "THEY LIED",
    "sadness":       "IT ENDS NOW",
    "determination": "NO TURNING BACK",
    "despair":       "NO WAY OUT",
}

HOOK_SYSTEM_PROMPT = """\
You generate YouTube thumbnail hook text for a movie recap channel.
Output EXACTLY 2 or 3 words in ALL CAPS that create instant tension or mystery.
The words must feel like a label for the most suspenseful scene.
Good: "NO ESCAPE"  "HE KNEW"  "STAY QUIET"  "TOO LATE"  "THEY LIED"
Bad: "VERY SCARY SCENE" (too long)  "it's bad" (not caps)
Return ONLY the words. No punctuation, no quotes, no explanation."""


def _load_font(size_px: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size_px)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()


def score_scenes_for_thumbnail(
    analysis_scenes: list[dict],
    total_duration_sec: float,
) -> dict:
    """Return the analysis scene best suited for thumbnail frame extraction."""
    candidates = [
        s for s in analysis_scenes
        if not s.get("is_credits", False)
        and "no characters" not in s.get("vlm_data", {}).get("characters", "").lower()
    ]

    if not candidates:
        candidates = [s for s in analysis_scenes if not s.get("is_credits", False)]

    if not candidates:
        return analysis_scenes[-1]

    best, best_score = candidates[0], -1

    for scene in candidates:
        score = 0
        pct = scene["start_sec"] / total_duration_sec if total_duration_sec else 0.5

        if 0.60 <= pct <= 0.85:
            score += 40
        elif 0.45 <= pct < 0.60:
            score += 20

        emotion = scene.get("vlm_data", {}).get("emotion", "").lower()
        if any(e in emotion for e in STRONG_EMOTIONS):
            score += 35

        char_action = (
            scene.get("vlm_data", {}).get("characters", "") + " " +
            scene.get("vlm_data", {}).get("action", "")
        ).lower()
        if any(sig in char_action for sig in CLOSE_UP_SIGNALS):
            score += 25
        else:
            score += 10

        if score > best_score:
            best_score = score
            best = scene

    if best_score <= 20:
        target_pct = 0.72
        best = min(candidates, key=lambda s: abs(s["start_sec"] / total_duration_sec - target_pct) if total_duration_sec else 0)

    return best


def extract_frame_at(
    video_path: str,
    target_sec: float,
    decode_height: int = 720,
) -> Image.Image:
    """Extract a single frame near target_sec from video_path."""
    for attempt_sec in [target_sec, max(0.0, target_sec - 5.0)]:
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            seek_pts = int(attempt_sec / time_base)
            container.seek(seek_pts, stream=stream, any_frame=False)
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = frame.to_image()
                    container.close()
                    w, h = img.size
                    new_h = decode_height
                    new_w = int(w * new_h / h)
                    return img.resize((new_w, new_h), Image.LANCZOS)
            container.close()
        except Exception:
            pass
    raise RuntimeError(f"Could not extract frame from {video_path} near {target_sec}s")


def generate_hook_text(
    movie_title: str,
    climax_narrations: list[str],
    api_key: str,
) -> str:
    """Call DeepSeek to produce 2-3 ALL CAPS hook words for the thumbnail."""
    from narrate import _http_post

    user_msg = (
        f"Movie: {movie_title}\n"
        "Climax narration lines:\n" +
        "\n".join(f"- {n}" for n in climax_narrations[:5]) +
        "\n\nGenerate the thumbnail hook text."
    )
    messages = [
        {"role": "system", "content": HOOK_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = _http_post(api_key, messages, max_tokens=20, temperature=0.7, thinking_enabled=False)
    words = raw.strip().upper().split()[:3]
    return " ".join(words) if words else "NO ESCAPE"


def composite_thumbnail(
    frame: Image.Image,
    hook_text: str,
    output_path: str,
    size: tuple[int, int] = (1280, 720),
    jpeg_quality: int = 92,
) -> str:
    """Composite a 1280x720 thumbnail: cover-crop frame + gradient + bold text."""
    target_w, target_h = size

    # Cover-crop to target size
    src_w, src_h = frame.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    frame = frame.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    canvas = frame.crop((left, top, left + target_w, top + target_h)).convert("RGBA")

    # Dark gradient over bottom 35%
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    grad_start = int(target_h * 0.65)
    for y in range(grad_start, target_h):
        alpha = int(200 * (y - grad_start) / (target_h - grad_start))
        overlay_draw.line([(0, y), (target_w - 1, y)], fill=(0, 0, 0, alpha))
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")

    # Render text
    font_size = 110
    font = _load_font(font_size)
    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), hook_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (target_w - text_w) // 2
    y = target_h - text_h - 48

    # Black stroke
    stroke = 6
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), hook_text, font=font, fill=(0, 0, 0))

    # White fill
    draw.text((x, y), hook_text, font=font, fill=(255, 255, 255))

    canvas.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return output_path


def generate_thumbnail(
    video_path: str,
    analysis_path: str,
    narrated_storyboard_path: str,
    run_dir: str,
    movie_title: str,
    api_key: str | None = None,
    output_filename: str | None = None,
) -> str:
    """Full thumbnail pipeline: score → extract frame → hook text → composite."""
    with open(analysis_path) as f:
        analysis = json.load(f)

    scenes = analysis["scenes"]
    non_credits = [s for s in scenes if not s.get("is_credits", False)]
    total_duration_sec = max((s["end_sec"] for s in non_credits), default=1.0)

    best_scene = score_scenes_for_thumbnail(scenes, total_duration_sec)
    target_sec = (best_scene["start_sec"] + best_scene["end_sec"]) / 2.0

    print(f"[thumbnail] best scene window={best_scene['window']} at {target_sec:.1f}s "
          f"(emotion={best_scene.get('vlm_data', {}).get('emotion', '?')})")

    frame = extract_frame_at(video_path, target_sec)

    # Hook text
    hook_text = None
    if api_key and os.path.exists(narrated_storyboard_path):
        try:
            with open(narrated_storyboard_path) as f:
                storyboard = json.load(f)
            climax_beats = [
                s["narratedText"]
                for s in storyboard.get("scenes", [])
                if s.get("narratedText")
                and 0.60 <= s.get("startSec", 0) / total_duration_sec <= 0.85
            ]
            if climax_beats:
                hook_text = generate_hook_text(movie_title, climax_beats, api_key)
        except Exception as e:
            print(f"[thumbnail] hook text API failed ({e}), using fallback")

    if not hook_text:
        emotion = best_scene.get("vlm_data", {}).get("emotion", "").lower()
        matched_key = next((k for k in FALLBACK_HOOK_MAP if k in emotion), None)
        hook_text = FALLBACK_HOOK_MAP.get(matched_key or "", "NO ESCAPE")

    print(f"[thumbnail] hook text: {hook_text!r}")

    os.makedirs(run_dir, exist_ok=True)
    fname = output_filename or f"{os.path.basename(run_dir)}_thumbnail.jpg"
    output_path = os.path.join(run_dir, fname)

    composite_thumbnail(frame, hook_text, output_path)
    print(f"[thumbnail] written → {output_path}")
    return output_path
