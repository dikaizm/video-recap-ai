"""
metadata_gen.py — Generate YouTube title, description, and hashtags for a recap video.

Uses DeepSeek to produce publish-ready metadata from the narrated storyboard
and optional story context. Description follows the 4-zone pattern used by top
recap channels: Hook → Disclosure → Fair Use → CTA.
"""
from __future__ import annotations

import json
import os
import re

# --------------------------------------------------------------------------- #
# Hardcoded description zones (zones 3 & 4 — same across all videos)
# --------------------------------------------------------------------------- #

_FAIR_USE = (
    "This video is for educational and entertainment purposes only. "
    "It falls under Fair Use guidelines (Section 107 of the Copyright Act) "
    "as it provides original commentary, creative editing, and a transformed "
    "narrative that adds value beyond the original work."
)

_CTA = (
    "Subscribe to Premiere Roll and hit the bell for new movie recaps every week. "
    "Drop a comment with the movie you want recapped next!"
)

# --------------------------------------------------------------------------- #
# Leetspeak encoder for the disclosure line (avoids automated copyright crawlers)
# --------------------------------------------------------------------------- #

_LEET: dict[str, str] = {
    "a": "4", "e": "3", "i": "1", "o": "0",
    "A": "4", "E": "3", "I": "1", "O": "0",
}


def _to_leet(text: str) -> str:
    return "".join(_LEET.get(c, c) for c in text)


# --------------------------------------------------------------------------- #
# LLM prompt — only generates zones 1 & 2 (hook + plot summary)
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are a viral YouTube writer for a movie recap channel called "Premiere Roll".
Generate a click-optimized title and the first two zones of a YouTube description.

Return ONLY a JSON object with exactly these keys:
{
  "title": "<YouTube video title>",
  "hook": "<2-sentence curiosity hook for the description — rephrase the title mystery>",
  "plot_summary": "<2-3 paragraphs of plot summary with spoilers — plain text, no markdown>",
  "hashtags": ["<tag1>", "<tag2>", ...]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TITLE — follow top channels (Movie Recaps, Mystery Recapped, Daniel CC):

NEVER use the movie's actual title. Describe the most shocking or intriguing
premise so the viewer is compelled to click without knowing the movie name.

Pick ONE pattern:

  Pattern A — Abstract Plot Summary
    [Subject] + [Unexpected Action/Condition] + [Consequence]
    e.g. "Scientist Unknowingly Revives A Girl Into A Superhuman Psycho"

  Pattern B — Hyperbolic Constraint
    [Time/Condition] + [Extreme Action] + [Unique Stake]
    e.g. "One Person Dies Every 10 Minutes Unless Someone Confesses A Dark Secret"

  Pattern C — Unaware Victim
    [Character Type] + Doesn't Know / Unaware + [Shocking Fact]
    e.g. "Blind Woman Doesn't Know She's Married To An Invisible Man This Whole Time"

Power words: Unknowingly · Accidentally · This Whole Time · No One Realizes ·
             Forced To · Discovers · Trapped · Replaced · Realizes Too Late

Rules: Title Case · Under 90 chars · No movie name · No year · No "Recap" suffix

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOOK (description zone 1 — first 2 visible lines in YouTube search):
- Rephrases the title mystery as a teaser question or statement
- Max 2 sentences, no movie name, pure curiosity
- Must make someone scroll to find out more

PLOT SUMMARY (description zone 2):
- 2-3 tight paragraphs covering the full story arc
- Spoilers are fine — this is a recap channel
- Plain text only, no asterisks or markdown

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASHTAGS:
- 15-20 tags
- Mix: movie title slug (for search), genre, year if known, general recap terms
- No # prefix — just the word
- Always include: MovieRecap, PremiereRoll
"""


def _build_user_message(movie_title: str, narration_text: str, story_context: str) -> str:
    parts = [f"Movie: {movie_title}"]
    if story_context:
        parts.append(f"\nStory context:\n{story_context[:2000]}")
    if narration_text:
        parts.append(f"\nNarration beats (in order):\n{narration_text[:4000]}")
    return "\n".join(parts)


def generate_video_metadata(
    storyboard: dict,
    movie_title: str,
    api_key: str,
    story_context: str = "",
    output_path: str | None = None,
) -> dict:
    """Call DeepSeek to generate title/hook/plot. Assemble full 4-zone description. Write .md."""
    from narrate import _http_post

    scenes = storyboard.get("scenes", [])
    narration_lines = [
        s["narratedText"].strip()
        for s in scenes
        if s.get("narratedText", "").strip() and not s.get("isGreeting")
    ]
    narration_text = "\n".join(narration_lines)

    user_msg = _build_user_message(movie_title, narration_text, story_context)

    print(f"[metadata] calling DeepSeek for title/description/hashtags...")
    raw = _http_post(
        api_key,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=700,
        temperature=0.5,
        thinking_enabled=False,
        json_mode=True,
    )

    data = json.loads(raw)
    title = str(data.get("title", f"{movie_title} — Full Movie Recap")).strip()
    hook = str(data.get("hook", "")).strip()
    plot_summary = str(data.get("plot_summary", "")).strip()
    hashtags: list[str] = [str(t).strip().lstrip("#") for t in data.get("hashtags", [])]

    description = _assemble_description(hook, plot_summary, movie_title)

    result = {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        # raw zones for downstream use
        "hook": hook,
        "plot_summary": plot_summary,
    }

    if output_path:
        _write_md(output_path, result, movie_title)
        print(f"[metadata] written → {output_path}")

    return result


def _assemble_description(hook: str, plot_summary: str, movie_title: str) -> str:
    """Combine the 4 description zones into a single string."""
    disclosure = f"M0v13: {_to_leet(movie_title)}"
    parts = [
        hook,
        "",
        plot_summary,
        "",
        disclosure,
        "",
        _FAIR_USE,
        "",
        _CTA,
    ]
    return "\n".join(parts)


def _write_md(path: str, meta: dict, movie_title: str) -> None:
    tags_inline = "  ".join(f"#{t}" for t in meta["hashtags"])
    description = meta.get("description") or _assemble_description(
        meta.get("hook", ""), meta.get("plot_summary", ""), movie_title
    )
    lines = [
        f"# {meta['title']}",
        "",
        "## Description",
        "",
        description,
        "",
        "## Hashtags",
        "",
        tags_inline,
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
