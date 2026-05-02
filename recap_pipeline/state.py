"""
state.py — Pipeline step state manifest backed by pipeline_state.json.

Provides:
  PipelineState        — tracks which steps completed cleanly
  atomic_write_json    — crash-safe JSON write via .tmp + os.replace
  probe_audio          — ffprobe duration check
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime


def atomic_write_json(path: str, obj: object) -> None:
    """Write obj as JSON to path atomically (safe against crash mid-write)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def probe_audio(path: str) -> float | None:
    """Return audio duration in seconds, or None if file is missing/unreadable."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not os.path.exists(path):
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        val = result.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None


class PipelineState:
    """Persistent step-completion tracker backed by <run_dir>/pipeline_state.json.

    Each step entry has the form:
      {"status": "complete" | "running" | "skipped", "completed_at": "...", ...}

    A step is considered done only when its status is "complete".
    "running" means the process was killed mid-step — needs re-run.
    Absent entry also means needs re-run.
    """

    COMPLETE = "complete"
    RUNNING  = "running"
    SKIPPED  = "skipped"

    def __init__(self, run_dir: str) -> None:
        self._path = os.path.join(run_dir, "pipeline_state.json")
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._state = data.get("steps", {})
            except Exception:
                self._state = {}

    def _save(self) -> None:
        atomic_write_json(self._path, {"steps": self._state})

    def status(self, step: str) -> str | None:
        return self._state.get(step, {}).get("status")

    def is_complete(self, step: str) -> bool:
        return self._state.get(step, {}).get("status") == self.COMPLETE

    def mark_running(self, step: str) -> None:
        self._state[step] = {"status": self.RUNNING, "started_at": datetime.now().isoformat()}
        self._save()

    def mark_complete(self, step: str, **meta: object) -> None:
        self._state[step] = {
            "status": self.COMPLETE,
            "completed_at": datetime.now().isoformat(),
            **meta,
        }
        self._save()

    def mark_skipped(self, step: str, reason: str = "") -> None:
        self._state[step] = {"status": self.SKIPPED, "reason": reason}
        self._save()

    def clear_steps(self, steps: list[str]) -> None:
        """Remove completion state for the given steps so they will re-run."""
        changed = False
        for step in steps:
            if step in self._state:
                del self._state[step]
                changed = True
        if changed:
            self._save()

    def summary(self) -> str:
        parts = []
        for step, info in self._state.items():
            s = info.get("status", "?")
            parts.append(f"{step}={s}")
        return ", ".join(parts) if parts else "none"
