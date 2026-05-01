"""
agent.py — Phase 4: self-correct orchestrator.

Reads eval_results.json, applies fixes (re-narrate, swap scenes, adjust timing),
re-generates TTS for changed beats, re-renders the recap. Then loops back to
Phase 2 (review) until quality threshold is met or max iterations reached.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narrate import _http_post, _parse_batch_response  # type: ignore


FIX_NARRATION_SYSTEM = """\
You are fixing a narration line that was flagged by a quality reviewer.

The original narration had this problem:
{issue}

Original narration: "{old_text}"

Source scenes available (the footage that will play during this beat):
{scenes_text}

Write a NEW narration line that:
- Accurately describes what's visually shown in the available source scenes
- Keeps the same narrative purpose but fixes the mismatch
- Is 12-30 words, matching the beat's 4-8s video duration
- Never repeats visual descriptions word-for-word — tell the story

Return ONLY a JSON object: {"narration": "new narration text"}\
"""


def _fix_single_narration(
    old_text: str,
    issue: str,
    scenes_text: str,
    api_key: str,
) -> str:
    """Re-narrate one beat with critique context. Returns new narration text."""
    system = FIX_NARRATION_SYSTEM.format(
        issue=issue, old_text=old_text, scenes_text=scenes_text,
    )

    try:
        raw = _http_post(
            api_key,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Generate the corrected narration."},
            ],
            max_tokens=150,
            temperature=0.5,
            timeout=60,
            thinking_enabled=False,
            json_mode=True,
        )
        data = json.loads(raw)
        return data.get("narration", "").strip()
    except Exception as e:
        print(f"    [fix] narration fix failed ({e})")
        return ""


def _fix_scene_swap(
    beat: dict,
    alt_window: int,
    source_by_window: dict[int, dict],
) -> dict:
    """Swap a source scene in a beat. Returns updated beat dict."""
    alt_scene = source_by_window.get(alt_window) or source_by_window.get(str(alt_window))
    if not alt_scene:
        print(f"    [fix] scene swap: window {alt_window} not found in source pool")
        return beat

    # Replace the first scene in the beat (or add if none)
    scenes = beat.get("scenes", [])
    if not scenes:
        beat["scenes"] = [alt_scene]
    else:
        # Add as new scene at start (keeping existing ones)
        beat["scenes"] = [alt_scene] + scenes

    # Clear segments so render falls back to beat startSec
    beat["segments"] = []
    # Update beat startSec to the swapped-in scene
    beat["startSec"] = float(alt_scene.get("startSec", 0))

    print(f"    [fix] scene swap: added window {alt_window} → \"{alt_scene.get('povText', '')[:60]}...\"")
    return beat


def _apply_timing_fix(
    beat: dict,
    fps: int,
) -> dict:
    """Adjust displayFrames to better match narration word count at ~3 wps."""
    wc = len(beat.get("narratedText", "").split())
    if wc == 0:
        return beat
    # ~3 words per second speaking rate + 0.5s padding per scene cut
    ideal_secs = wc / 3.0 + 0.5
    ideal_frames = max(30, round(ideal_secs * fps))
    old_frames = beat.get("displayFrames", 0)
    if abs(ideal_frames - old_frames) > fps:  # only adjust if >1s difference
        beat["displayFrames"] = ideal_frames
        print(f"    [fix] timing: {old_frames / fps:.1f}s → {ideal_frames / fps:.1f}s (wc={wc})")
    return beat


def self_correct(
    run_dir: str,
    storyboard_path: str,
    eval_path: str,
    analysis_path: str,
    api_key: str,
    fps: int = 30,
    tts_model_path: str = "",
    tts_speed: float = 1.0,
) -> tuple[dict, set[int]]:
    """Read eval_results.json, apply all fixes, return (updated_storyboard, changed_beats).

    changed_beats is a set of beat indices whose narratedText or scenes changed,
    so the caller knows which TTS files to re-generate.
    """
    with open(storyboard_path) as f:
        storyboard = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)
    with open(analysis_path) as f:
        analysis = json.load(f)

    beats = storyboard.get("scenes", [])
    evaluations = eval_data.get("evaluations", [])

    # Build source scene index for swaps
    source_by_window: dict[int, dict] = {}
    for s in analysis.get("scenes", []):
        source_by_window[s["window"]] = s

    changed_beats: set[int] = set()
    fixes_applied = {
        "rewrite_narration": 0,
        "swap_scene": 0,
        "adjust_timing": 0,
        "both": 0,
    }

    for ev in evaluations:
        idx = ev.get("beat_index", 0)
        fix_type = ev.get("fix_type", "ok")
        if fix_type == "ok" or idx >= len(beats):
            continue

        beat = beats[idx]
        issues = ev.get("issues", [])
        issue_text = "; ".join(issues) if issues else ev.get("fix_suggestion", "unknown")

        # Build source scene text for re-narration context
        source_scenes = beat.get("scenes", [])
        scenes_text = "\n".join(
            f"  - {s.get('povText', s.get('description', ''))[:200]}"
            for s in source_scenes[:5]
        ) or "(no source scenes)"

        if fix_type in ("rewrite_narration", "both"):
            old_text = beat.get("narratedText", "")
            fix_suggestion = ev.get("fix_suggestion", "")
            context = f"{issue_text}. {fix_suggestion}"

            new_text = _fix_single_narration(old_text, context, scenes_text, api_key)
            if new_text and new_text != old_text:
                beat["narratedText"] = new_text
                print(f"  [fix] beat {idx + 1}: re-narrated → \"{new_text[:60]}...\"")
                changed_beats.add(idx)
                fixes_applied["rewrite_narration"] += 1

        if fix_type in ("swap_scene", "both"):
            alt_window = ev.get("alternative_scene_window")
            if alt_window and alt_window not in {
                s.get("window") for s in beat.get("scenes", [])
            }:
                beats[idx] = _fix_scene_swap(beat, alt_window, source_by_window)
                changed_beats.add(idx)
                fixes_applied["swap_scene"] += 1

        if fix_type == "adjust_timing":
            beats[idx] = _apply_timing_fix(beat, fps)
            changed_beats.add(idx)
            fixes_applied["adjust_timing"] += 1

    # Reconcile total display frames after individual adjustments
    total_old = sum(b.get("displayFrames", 0) for b in beats)
    target_ratio = storyboard.get("recap_ratio", 0.15)
    total_original = sum(
        s.get("endSec", s.get("startSec", 0) + 5) - s.get("startSec", 0)
        for s in analysis.get("scenes", [])
    ) if analysis.get("scenes") else total_old / fps / target_ratio

    # Scale all displayFrames proportionally if total drifted >2%
    total_new = sum(b.get("displayFrames", 0) for b in beats)
    target_frames = round(total_original * target_ratio * fps)
    if total_old > 0 and abs(total_new / max(1, total_old) - 1) > 0.02:
        scale = target_frames / max(1, total_new)
        for b in beats:
            b["displayFrames"] = max(30, round(b.get("displayFrames", 30) * scale))
        print(f"  [fix] rebalanced displayFrames ({total_new} → {target_frames}, scale={scale:.3f})")

    # Re-compute endSec and durationInFrames for all beats
    cursor = 0.0
    for i, b in enumerate(beats):
        b["window"] = i + 1
        b["startSec"] = round(cursor, 3)
        dur = b["displayFrames"] / fps
        b["endSec"] = round(cursor + dur, 3)
        b["durationInFrames"] = b["displayFrames"]
        cursor += dur

    storyboard["scenes"] = beats
    storyboard["total_sec"] = round(cursor, 1)
    storyboard["total_frames"] = int(cursor * fps)

    with open(storyboard_path, "w") as f:
        json.dump(storyboard, f, indent=2)

    print(
        f"\n[fix] applied: rewrite_narration={fixes_applied['rewrite_narration']}, "
        f"swap_scene={fixes_applied['swap_scene']}, "
        f"adjust_timing={fixes_applied['adjust_timing']}, "
        f"both={fixes_applied['both']}"
    )
    return storyboard, changed_beats


def _run_pipeline(args, run_dir: str, video_path: str, auto_story: str | None) -> None:
    """Run all pipeline steps directly, producing output in run_dir."""
    from transform import transform as do_transform
    from cluster import cluster_scenes
    from tts import generate_batch
    from main import (
        setup_remotion_public, install_remotion_deps, run_render,
        adjust_display_frames_to_audio, setup_run_logging, _stream_subprocess,
    )

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
    REMOTION_DIR = os.path.join(PROJECT_ROOT, "remotion")

    fps = getattr(args, "fps", 30)
    recap_ratio = getattr(args, "recap_ratio", 0.15)
    ollama_model = getattr(args, "ollama_model", "gemma4:e2b")
    ollama_host = getattr(args, "ollama_host", "http://localhost:11434")
    decode_height = getattr(args, "decode_height", 640)
    story_context: str = ""  # resolved in narrate step below
    deepseek_key = getattr(args, "deepseek_key", "")
    qwen3_speed = getattr(args, "qwen3_speed", 1.0)

    voiceover_dir = os.path.join(run_dir, "voiceover") if not getattr(args, "no_tts", False) else None
    storyboard_path = os.path.join(run_dir, "storyboard.json")
    narrated_path = os.path.join(run_dir, "storyboard_narrated.json")
    clustered_path = os.path.join(run_dir, "storyboard_clustered.json")
    analysis_json = os.path.join(run_dir, "analysis.json")
    output_mp4 = os.path.join(run_dir, "recap.mp4")

    setup_run_logging(run_dir)

    # Step 1: Analysis (via analyze.py subprocess — it has complex frame extraction)
    step_start = time.time()
    print(f"[pipeline] step 1: analysis...")
    cmd = [
        sys.executable, os.path.join(PACKAGE_DIR, "analyze.py"),
        "--video", video_path,
        "--output", analysis_json,
        "--ollama-model", ollama_model,
        "--ollama-host", ollama_host,
    ]
    if decode_height > 0:
        cmd += ["--decode-height", str(decode_height)]
    _stream_subprocess(cmd, cwd=PROJECT_ROOT)
    print(f"[pipeline] analysis done ({time.time() - step_start:.0f}s)")

    # Step 2: Transform
    step_start = time.time()
    print(f"[pipeline] step 2: transform...")
    storyboard = do_transform(
        analysis_path=analysis_json,
        output_path=storyboard_path,
        video_path=video_path,
        fps=fps,
        recap_ratio=recap_ratio,
        voiceover_dir=voiceover_dir,
    )
    print(f"[pipeline] transform done ({time.time() - step_start:.0f}s)")

    # Step 3: Cluster
    step_start = time.time()
    print(f"[pipeline] step 3: cluster...")
    target_duration = sum(s["displayFrames"] for s in storyboard["scenes"]) / fps
    source_scenes_pool = list(storyboard["scenes"])
    clustered_scenes = cluster_scenes(
        storyboard["scenes"],
        target_duration=target_duration,
        fps=fps,
        min_beat_sec=getattr(args, "min_beat_sec", 4.0),
        max_beat_sec=getattr(args, "max_beat_sec", 8.0),
    )
    clustered = {**storyboard, "scenes": clustered_scenes}
    with open(clustered_path, "w") as f:
        json.dump(clustered, f, indent=2)
    print(f"[pipeline] cluster done ({time.time() - step_start:.0f}s)")

    # Step 4: Intro
    if not getattr(args, "no_intro", False) and getattr(args, "narrate", True) and deepseek_key:
        from intro import build_intro_beat
        step_start = time.time()
        print(f"[pipeline] step 4: intro...")
        story_text = ""
        if auto_story and os.path.isfile(auto_story):
            with open(auto_story) as f:
                story_text = f.read().strip()
        intro_beat = build_intro_beat(
            source_scenes=source_scenes_pool,
            story_context=story_text,
            api_key=deepseek_key,
            fps=fps,
            scene_count=getattr(args, "intro_scenes", 3),
        )
        if intro_beat:
            clustered_scenes = [intro_beat] + clustered_scenes
            clustered["scenes"] = clustered_scenes
            with open(clustered_path, "w") as f:
                json.dump(clustered, f, indent=2)
            print(f"[pipeline] intro done ({time.time() - step_start:.0f}s)")

    # Step 5: Narrate
    if getattr(args, "narrate", False) and deepseek_key:
        from narrate import narrate_beats, _http_post, NARRATE_BATCH_SYSTEM, _parse_batch_response
        step_start = time.time()
        print(f"[pipeline] step 5: narrate...")

        story_context = ""
        if auto_story and os.path.isfile(auto_story):
            with open(auto_story) as f:
                story_context = f.read().strip()

        clustered = narrate_beats(
            storyboard_path=clustered_path,
            output_path=narrated_path,
            api_key=deepseek_key,
            analysis_path=analysis_json,
            story_context=story_context,
            force=getattr(args, "narrate_force", False),
        )
        storyboard = clustered

        # Validation retry
        for retry in range(2):
            missing = [i for i, b in enumerate(clustered["scenes"]) if not b.get("narratedText")]
            if not missing:
                break
            if retry == 0:
                print(f"[validate] {len(missing)}/{len(clustered['scenes'])} beats missing → full re-narrate")
                clustered = narrate_beats(
                    storyboard_path=narrated_path,
                    output_path=narrated_path,
                    api_key=deepseek_key,
                    analysis_path=analysis_json,
                    story_context=story_context,
                    force=True,
                )
            else:
                print(f"[validate] {len(missing)} beats still missing → individual retry")
                for idx in missing:
                    beat = clustered["scenes"][idx]
                    pov = beat.get("povText", "").strip()
                    dialogue = beat.get("dialogue", "").strip()
                    total_sec = max(b.get("endSec", 1) for b in clustered["scenes"])
                    pct = int(round(beat.get("startSec", 0) / max(1, total_sec) * 100))
                    user = f"Beat to narrate:\n[Beat] pos={pct}% visual: {pov[:300]}"
                    if dialogue:
                        user += f" dialogue: {dialogue[:200]}"
                    result = _http_post(deepseek_key, [
                        {"role": "system", "content": NARRATE_BATCH_SYSTEM},
                        {"role": "user", "content": user},
                    ], max_tokens=100, temperature=0.5, timeout=60, thinking_enabled=False, json_mode=True)
                    narrations = _parse_batch_response(result, 1)
                    if narrations:
                        clustered["scenes"][idx]["narratedText"] = narrations[0].strip()

        missing = [i for i, b in enumerate(clustered["scenes"]) if not b.get("narratedText")]
        if missing:
            print(f"[error] {len(missing)} beats still without narration", file=sys.stderr)
            sys.exit(1)

        with open(narrated_path, "w") as f:
            json.dump(clustered, f, indent=2)
        storyboard = clustered
        storyboard_path = narrated_path
        print(f"[pipeline] narrate done ({time.time() - step_start:.0f}s)")

    # Step 6: TTS
    if not getattr(args, "no_tts", False):
        from tts import generate_batch
        step_start = time.time()
        print(f"[pipeline] step 6: TTS...")

        _qwen3_model_path = os.path.join(PROJECT_ROOT, "models", "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
        if not os.path.exists(_qwen3_model_path):
            print(f"[error] Qwen3 model not found at {_qwen3_model_path!r}", file=sys.stderr)
            sys.exit(1)

        from voices import resolve_voice_instruct, get_voice_instruct
        _genre_arg = getattr(args, "genre", "auto")
        if _genre_arg == "auto":
            if story_context and deepseek_key:
                _instruct, _genres = resolve_voice_instruct(story_context, deepseek_key)
                print(f"[voice] detected genres: {_genres}")
                if len(_genres) > 1:
                    print(f"[voice] blending {len(_genres)} genre presets → custom instruct")
            else:
                _genres = ["documentary"]
                _instruct = get_voice_instruct("documentary")
        else:
            _genres = [_genre_arg]
            _instruct = get_voice_instruct(_genre_arg)
            print(f"[voice] using genre preset: {_genre_arg}")

        storyboard.setdefault("metadata", {})["voice_genres"] = _genres
        storyboard["metadata"]["voice_instruct"] = _instruct

        tts_kwargs = {
            "model_path": _qwen3_model_path,
            "speed": qwen3_speed,
            "instruct": _instruct,
        }
        generate_batch(storyboard["scenes"], voiceover_dir, **tts_kwargs)
        print(f"[pipeline] TTS done ({time.time() - step_start:.0f}s)")

        for i, scene in enumerate(storyboard["scenes"]):
            scene.setdefault("window", i + 1)

        adjust_display_frames_to_audio(storyboard, voiceover_dir, fps)

        # Step 7: Align
        if not getattr(args, "no_align", False) and deepseek_key:
            from align import align_beats
            step_start = time.time()
            print(f"[pipeline] step 7: align...")
            align_beats(
                storyboard["scenes"],
                source_scenes=source_scenes_pool,
                voiceover_dir=voiceover_dir,
                api_key=deepseek_key,
                fps=fps,
                expand=getattr(args, "align_expand", 2),
                whisper_model_size=getattr(args, "align_whisper_model", "tiny.en"),
            )
            print(f"[pipeline] align done ({time.time() - step_start:.0f}s)")

        with open(storyboard_path, "w") as f:
            json.dump(storyboard, f, indent=2)

    # Step 8: Render
    setup_remotion_public(video_path, voiceover_dir=voiceover_dir)
    beats_list = storyboard["scenes"]
    for i, beat in enumerate(beats_list):
        vo_path = f"voiceover/scene_{i + 1:02d}.mp3"
        if os.path.exists(os.path.join(run_dir, vo_path)):
            beat["voiceoverPath"] = vo_path
        beat["window"] = i + 1
        beat["endSec"] = beat.get("endSec", beat.get("startSec", 0) + beat.get("displayFrames", 30) / fps)
        beat["durationInFrames"] = beat.get("durationInFrames", beat.get("displayFrames", 30))
        beat["startFmt"] = beat.get("startFmt", "")
        beat["dialog"] = beat.get("dialogue", "")

    install_remotion_deps()
    step_start = time.time()
    print(f"[pipeline] step 8: render...")
    run_render(storyboard, output_mp4, concurrency=getattr(args, "concurrency", None), gl=getattr(args, "gl", "angle"))
    print(f"[pipeline] render done ({time.time() - step_start:.0f}s)")


def run_agent_loop(
    args,
) -> int:
    """Run the full agent loop: produce, review, evaluate, correct, repeat.

    Returns 0 on success, 1 on failure.
    """
    max_iter = getattr(args, "agent_max_iter", 3)
    threshold = getattr(args, "agent_threshold", 0.85)
    agent_no_render = getattr(args, "agent_no_render", False)

    # Use the existing pipeline to produce the initial recap
    # (main.py's main() with --narrate flag)
    # Import main's globals
    from main import (
        OUTPUT_BASE, PROJECT_ROOT, REMOTION_DIR, PACKAGE_DIR,
        make_run_dir, setup_run_logging, run_render,
        setup_remotion_public, install_remotion_deps,
        adjust_display_frames_to_audio,
        resolve_input_folder,
    )

    # Determine run dir
    from pathlib import Path
    if args.input:
        video_path, auto_story = resolve_input_folder(args.input)
        run_name = Path(args.input).resolve().name
    else:
        video_path = args.video
        run_name = Path(video_path).stem
        auto_story = None

    run_dir = make_run_dir(run_name)
    setup_run_logging(run_dir)
    print(f"[agent] run directory: {run_dir}")

    output_mp4 = os.path.join(run_dir, "recap.mp4")
    voiceover_dir = os.path.join(run_dir, "voiceover")
    storyboard_path = os.path.join(run_dir, "storyboard_narrated.json")
    clustered_path = os.path.join(run_dir, "storyboard_clustered.json")
    analysis_json = os.path.join(run_dir, "analysis.json")
    review_json = os.path.join(run_dir, "review_analysis.json")
    eval_json = os.path.join(run_dir, "eval_results.json")

    # --- Iteration 0: produce initial recap ---
    print(f"\n{'=' * 60}")
    print(f"[agent] ITERATION 0: producing initial recap")
    print(f"{'=' * 60}")

    # Run pipeline steps directly (avoids subprocess directory mismatch)
    _run_pipeline(args, run_dir, video_path, auto_story)

    # --- Iteration loop ---
    for iteration in range(1, max_iter + 1):
        print(f"\n{'=' * 60}")
        print(f"[agent] ITERATION {iteration}/{max_iter}: review → evaluate → correct")
        print(f"{'=' * 60}")

        it_start = time.time()
        video_to_review = output_mp4

        # Phase 2: re-analyze recap
        print(f"\n[agent] Phase 2: re-analyze recap...")
        from review import review_recap
        review_recap(
            recap_path=video_to_review,
            storyboard_path=storyboard_path,
            output_path=review_json,
            ollama_model=getattr(args, "ollama_model", "gemma4:e2b"),
            ollama_host=getattr(args, "ollama_host", "http://localhost:11434"),
            decode_height=getattr(args, "decode_height", 480),
        )

        # Phase 3: cross-evaluate
        print(f"\n[agent] Phase 3: cross-evaluate...")
        from eval_agent import evaluate_beats
        eval_data = evaluate_beats(
            storyboard_path=storyboard_path,
            review_path=review_json,
            output_path=eval_json,
            api_key=args.deepseek_key,
            analysis_path=analysis_json,
        )

        avg_score = eval_data.get("average_score", 0)
        ok_ratio = eval_data.get("ok_ratio", 0)
        print(f"[agent] evaluation: avg={avg_score:.2f}, ok_ratio={ok_ratio:.1%}")

        if eval_data.get("passed", False):
            print(f"[agent] PASSED — quality threshold {threshold:.0%} met")
            break

        # Phase 4: self-correct
        fix_count = sum(eval_data.get("fix_counts", {}).values())
        if fix_count == 0:
            print("[agent] no fixes generated — cannot improve further, stopping")
            break

        print(f"\n[agent] Phase 4: self-correct ({fix_count} fixes to apply)...")
        _qwen3_model_path = os.path.join(PROJECT_ROOT, "models", "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")

        updated, changed_beats = self_correct(
            run_dir=run_dir,
            storyboard_path=storyboard_path,
            eval_path=eval_json,
            analysis_path=analysis_json,
            api_key=args.deepseek_key,
            fps=getattr(args, "fps", 30),
            tts_model_path=_qwen3_model_path,
            tts_speed=getattr(args, "qwen3_speed", 1.0),
        )

        # Re-TTS only changed beats
        if changed_beats:
            print(f"\n[agent] re-TTS for {len(changed_beats)} changed beats...")
            from tts import generate_batch, load_qwen3_model, _scene_text

            os.makedirs(voiceover_dir, exist_ok=True)
            qwen3_model = load_qwen3_model(_qwen3_model_path)

            beats_list = updated.get("scenes", [])
            for idx in sorted(changed_beats):
                if idx >= len(beats_list):
                    continue
                beat = beats_list[idx]
                window = beat.get("window", idx + 1)
                text = _scene_text(beat)
                if not text.strip():
                    print(f"    beat {window}: no text, skipping TTS")
                    continue

                out_path = os.path.join(voiceover_dir, f"scene_{window:02d}.mp3")
                from tts import generate_qwen3_tts
                # Reuse the blended instruct saved during initial TTS pass
                _retts_instruct = updated.get("metadata", {}).get(
                    "voice_instruct", ""
                ) or updated.get("metadata", {}).get("voice_genres", ["documentary"])[0]
                if not isinstance(_retts_instruct, str) or len(_retts_instruct) < 20:
                    from voices import get_voice_instruct
                    _retts_instruct = get_voice_instruct("documentary")
                generate_qwen3_tts(
                    text=text,
                    output_mp3_path=out_path,
                    model=qwen3_model,
                    speed=getattr(args, "qwen3_speed", 1.0),
                    instruct=_retts_instruct,
                )
                print(f"    beat {window}: re-generated → {out_path}")

            # Re-sync display frames to new audio
            adjust_display_frames_to_audio(updated, voiceover_dir, getattr(args, "fps", 30))
            with open(storyboard_path, "w") as f:
                json.dump(updated, f, indent=2)

        # Re-render
        if not agent_no_render:
            print(f"\n[agent] re-rendering...")
            setup_remotion_public(video_path, voiceover_dir=voiceover_dir)

            # Ensure window/endSec/durationInFrames fields
            beats_list = updated["scenes"]
            for i, beat in enumerate(beats_list):
                vo_path = f"voiceover/scene_{i + 1:02d}.mp3"
                if os.path.exists(os.path.join(run_dir, vo_path)):
                    beat["voiceoverPath"] = vo_path
                beat["window"] = i + 1
                beat["endSec"] = beat.get("endSec", beat.get("startSec", 0) + beat.get("displayFrames", 30) / getattr(args, "fps", 30))
                beat["durationInFrames"] = beat.get("durationInFrames", beat.get("displayFrames", 30))
                beat["startFmt"] = beat.get("startFmt", "")
                beat["dialog"] = beat.get("dialogue", "")

            install_remotion_deps()
            run_render(updated, output_mp4, concurrency=getattr(args, "concurrency", None), gl=getattr(args, "gl", "angle"))

        it_elapsed = time.time() - it_start
        print(f"[agent] iteration {iteration} done ({it_elapsed:.1f}s)")

    else:
        print(f"\n[agent] stopped after {max_iter} iterations (threshold not reached)")
        return 1

    print(f"\n[agent] complete → {output_mp4}")
    return 0
