"""Opt-in wall-time budget and graceful-stop handling for ``train_starvla.py``.

Enabled by ``trainer.max_wall_time_hours``.  The budget is per *run directory*:
every launch appends a session to ``<output_dir>/wall_time_ledger.json``, and a
resumed launch only gets what earlier sessions left.  A launch stops cleanly
(checkpoint + resumable state) once ``elapsed + safety margin`` reaches the
remaining budget, or when SIGTERM/SIGUSR1/SIGINT arrives.  All ranks agree on
the decision through one tiny all-reduce per optimizer step.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist

LEDGER_NAME = "wall_time_ledger.json"
_PROCESS_START = time.monotonic()


def read_ledger(output_dir: str | os.PathLike) -> dict:
    path = Path(output_dir) / LEDGER_NAME
    if not path.exists():
        return {"sessions": [], "total_elapsed_seconds": 0.0}
    return json.loads(path.read_text())


class WallTimeBudget:
    def __init__(self, trainer_cfg, output_dir: str | os.PathLike, is_main_process: bool, start_step: int):
        hours = getattr(trainer_cfg, "max_wall_time_hours", None)
        self.enabled = hours is not None and float(hours) > 0
        self.output_dir = Path(output_dir)
        self.is_main_process = is_main_process
        self.budget_seconds = float(hours) * 3600.0 if self.enabled else float("inf")
        self.safety_seconds = float(getattr(trainer_cfg, "wall_time_safety_minutes", 20.0)) * 60.0
        self.previous_seconds = float(read_ledger(self.output_dir)["total_elapsed_seconds"]) if self.enabled else 0.0
        self.start_step = int(start_step)
        self.signal_name = None
        self.stop_reason = None
        self._session_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.enabled:
            for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
                signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, frame):
        # Only record the request; the training loop checkpoints at the next step boundary.
        self.signal_name = signal.Signals(signum).name

    def session_seconds(self) -> float:
        return time.monotonic() - _PROCESS_START

    def remaining_seconds(self) -> float:
        return self.budget_seconds - self.previous_seconds - self.session_seconds()

    def should_stop(self, device) -> bool:
        """Collective: every rank must call this at the same optimizer step."""
        if not self.enabled:
            return False
        local = 0
        if self.signal_name is not None:
            local = 2
        elif self.remaining_seconds() <= self.safety_seconds:
            local = 1
        flag = torch.tensor([local], device=device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        value = int(flag.item())
        if value:
            self.stop_reason = "signal" if value == 2 else "wall_time_budget"
        return bool(value)

    def write_ledger(self, completed_steps: int, end_reason: str | None = None) -> None:
        if not (self.enabled and self.is_main_process):
            return
        ledger = read_ledger(self.output_dir)
        sessions = [s for s in ledger["sessions"] if s.get("started_at") != self._session_started]
        session = {
            "started_at": self._session_started,
            "elapsed_seconds": round(self.session_seconds(), 1),
            "start_step": self.start_step,
            "last_step": int(completed_steps),
            "end_reason": end_reason,
            "signal": self.signal_name,
        }
        sessions.append(session)
        payload = {
            "budget_hours": self.budget_seconds / 3600.0,
            "safety_minutes": self.safety_seconds / 60.0,
            "sessions": sessions,
            "total_elapsed_seconds": round(self.previous_seconds + self.session_seconds(), 1),
        }
        path = self.output_dir / LEDGER_NAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
