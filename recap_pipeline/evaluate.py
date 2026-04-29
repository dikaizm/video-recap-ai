"""
evaluate.py — Post-narration story coherence pass.

Sends the full narrated script to DeepSeek in one prompt. Identifies lines that
are repetitive, disconnected, or stall the story arc. Rewrites only those lines.

Usage:
    python evaluate.py \
        --storyboard output/.../storyboard_narrated.json \
        --output output/.../storyboard_evaluated.json \
        --api-key sk-...
"""
import argparse
import json
import os
import re
import time

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

EVAL_SYSTEM_PROMPT = """\
You are a story editor reviewing a narration script for a movie recap video.
Each numbered line is one scene in chronological story order.

IDENTIFY and REWRITE lines that have any of these problems:
- Repeat the same story beat as an adjacent scene (same subject + same action)
- Feel disconnected from the overall story arc
- Use hollow or abstract phrasing ("the weight of", "echoes of", "a lone figure", "silence speaks", "bears witness")
- Stall the story — same emotion or action restated across 3+ consecutive lines
- AI-sounding filler that adds no story information
- No narrative propulsion — a line that describes action without consequence, \
hook, or character state change
- 3+ consecutive lines without "but", "only to", "now", or another tension-carrying word

OUTPUT: Return a JSON array of ONLY the scenes that need fixing:
[{"window": <scene_number>, "revised": "<new narration>"}]

RULES for rewrites:
- Match the approximate word count of the original (±3 words)
- Each revised line must read distinctly differently from its neighbors
- Every revised line must end with a hook, a consequence, or an escalation of stakes
- No poetic language, no metaphors, no observer phrases ("we see", "we watch", "we witness")
- Advance the story — show consequence, character state, or what changes
- If a line is already good, do NOT include it in the output
- Return ONLY valid JSON — no preamble, no explanation, no commentary\
"""


def _http_post(api_key: str, messages: list[dict], max_tokens: int = 800,
               temperature: float = 0.4, retries: int = 3) -> str:
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"].strip()
                if not content:
                    usage = result.get("usage", {})
                    raise RuntimeError(
                        f"DeepSeek returned empty content — reasoning tokens likely exhausted max_tokens budget. "
                        f"Usage: {usage}"
                    )
                return content
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429 and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [eval] rate-limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"DeepSeek API error {e.code}: {body}") from e

    raise RuntimeError("DeepSeek API: max retries exceeded")


SYNOPSIS_SYSTEM_PROMPT = """\
You are given a sequence of scene descriptions from a short film, in chronological order.
Write a single paragraph (3–5 sentences) that summarizes what this movie is about — \
its story, setting, central conflict, and outcome.
Be concrete and specific. No poetic language. Return only the paragraph.\
"""


def generate_movie_context(scenes: list[dict], api_key: str) -> str:
    """Generate a brief movie synopsis from all scene povTexts. Used as context for evaluation."""
    lines = []
    for s in scenes:
        pov = (s.get("povText") or "").strip()
        if pov:
            lines.append(f"[{s['window']:02d}] {pov[:120]}")

    print("[eval] generating movie context paragraph...")
    return _http_post(
        api_key,
        messages=[
            {"role": "system", "content": SYNOPSIS_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        max_tokens=200,
        temperature=0.3,
    )


def build_eval_prompt(scenes: list[dict], movie_context: str = "") -> str:
    lines = []
    if movie_context:
        lines.append(f"MOVIE CONTEXT:\n{movie_context}\n\nNARRATION SCRIPT:")
    for s in scenes:
        text = (s.get("narratedText") or s.get("povText") or "").strip()
        lines.append(f"[{s['window']:02d}] {text}")
    return "\n".join(lines)


def _parse_revisions(raw: str) -> list[dict]:
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)

    # Try full JSON array parse first
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            revisions = json.loads(match.group())
            if isinstance(revisions, list):
                valid = []
                for item in revisions:
                    if isinstance(item, dict) and "window" in item and "revised" in item:
                        valid.append({"window": int(item["window"]), "revised": str(item["revised"]).strip()})
                return valid
        except json.JSONDecodeError:
            pass

    # Fallback: extract individual complete objects from a truncated array
    objects = re.findall(r'\{\s*"window"\s*:\s*(\d+)\s*,\s*"revised"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', raw)
    if objects:
        print(f"    [eval] partial parse: extracted {len(objects)} complete object(s) from truncated response")
        return [{"window": int(w), "revised": r.replace('\\"', '"')} for w, r in objects]

    print(f"    [eval] warning: could not parse any revisions from response, skipping")
    return []


def evaluate(
    storyboard_path: str,
    output_path: str,
    api_key: str,
    force: bool = False,
) -> dict:
    if os.path.exists(output_path) and not force:
        print(f"[eval] using cached evaluation: {output_path}")
        with open(output_path) as f:
            return json.load(f)

    with open(storyboard_path) as f:
        storyboard = json.load(f)

    scenes = storyboard["scenes"]
    narrated = [s for s in scenes if s.get("narratedText")]
    if not narrated:
        print("[eval] no narratedText found — skipping evaluation")
        with open(output_path, "w") as f:
            json.dump(storyboard, f, indent=2)
        return storyboard

    movie_context = generate_movie_context(scenes, api_key)
    print(f"[eval] context: {movie_context[:120]}...")

    prompt = build_eval_prompt(scenes, movie_context=movie_context)
    print(f"[eval] sending {len(scenes)} scenes to DeepSeek for story coherence review...")

    raw = _http_post(
        api_key,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
        temperature=0.35,
    )

    revisions = _parse_revisions(raw)

    # Store context in metadata for reference
    storyboard.setdefault("metadata", {})["movieContext"] = movie_context

    revision_map = {r["window"]: r["revised"] for r in revisions}
    applied = 0
    for scene in scenes:
        revised = revision_map.get(scene["window"])
        if revised:
            scene["narratedTextPre"] = scene.get("narratedText", "")
            scene["narratedText"] = revised
            applied += 1

    print(f"[eval] {applied} revision(s) applied out of {len(scenes)} scenes")

    with open(output_path, "w") as f:
        json.dump(storyboard, f, indent=2)
    print(f"[eval] evaluated storyboard → {output_path}")

    return storyboard


def main():
    parser = argparse.ArgumentParser(description="Story coherence evaluation pass")
    parser.add_argument("--storyboard", required=True, help="Narrated storyboard JSON")
    parser.add_argument("--output", required=True, help="Output path for evaluated storyboard")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"), help="DeepSeek API key")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = parser.parse_args()

    if not args.api_key:
        print("[error] --api-key required (or set DEEPSEEK_API_KEY)", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    evaluate(args.storyboard, args.output, args.api_key, force=args.force)


if __name__ == "__main__":
    main()
