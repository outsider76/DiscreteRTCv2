#!/usr/bin/env python3
"""Wait for the raw TrainingRTC run, then validate and upload its final checkpoint."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

from deployment.upload.upload_stacking_cups_checkpoints import upload_and_verify


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results/Checkpoints"
RUN_NAME = (
    "stacking_cups_20260911_qwenpi_raw_h30_training_rtc_"
    "prefix1to15_2gpu_gb32_ft5k"
)
STEP = 5_000
REMOTE_PREFIX = (
    "models/stacking-cups-20260911-qwenpi-v3-raw-h30-2gpu-gb32-50k-"
    "training-rtc-prefix1to15-ft5k-20260914"
)
STATE_PATH = RESULTS_ROOT / "stacking_cups_20260911_raw_training_rtc_state.json"
POLL_SECONDS = 300
TIMEOUT_SECONDS = 6 * 60 * 60
AUTH_TIMEOUT_SECONDS = 24 * 60 * 60


def record(stage: str, **fields: object) -> None:
    payload = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def run_active() -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "train_starvla.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    return any(f"--run_id {RUN_NAME}" in line for line in result.stdout.splitlines())


def final_artifacts_ready() -> bool:
    run_dir = RESULTS_ROOT / RUN_NAME
    checkpoint = run_dir / f"checkpoints/steps_{STEP}_pytorch_model.pt"
    final_model = run_dir / "final_model/pytorch_model.pt"
    summary = run_dir / "summary.jsonl"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (checkpoint, final_model, summary)):
        return False
    rows = [json.loads(line) for line in summary.read_text().splitlines() if line.strip()]
    return any(row.get("steps") == STEP for row in rows)


def wait_for_final_artifacts() -> None:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    record("waiting_for_training", poll_seconds=POLL_SECONDS)
    while True:
        if final_artifacts_ready() and not run_active():
            record("training_complete")
            return
        if not run_active() and not final_artifacts_ready():
            raise RuntimeError("Raw TrainingRTC process exited before final artifacts were complete")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for raw TrainingRTC completion")
        time.sleep(POLL_SECONDS)


def wait_for_huggingface_auth() -> HfApi:
    deadline = time.monotonic() + AUTH_TIMEOUT_SECONDS
    while True:
        api = HfApi()
        try:
            identity = api.whoami()
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Hugging Face authentication") from exc
            record(
                "waiting_for_huggingface_auth",
                poll_seconds=POLL_SECONDS,
                error_type=type(exc).__name__,
            )
            time.sleep(POLL_SECONDS)
            continue
        record("huggingface_authenticated", identity=identity.get("name", "unknown"))
        return api


def main() -> None:
    try:
        wait_for_final_artifacts()
        api = wait_for_huggingface_auth()
        record("uploading", remote_prefix=REMOTE_PREFIX)
        upload_and_verify(
            api,
            run_dir=RESULTS_ROOT / RUN_NAME,
            step=STEP,
            prefix=REMOTE_PREFIX,
            repo_id="DiscreteRTC/dRTC",
            repo_type="dataset",
        )
        record(
            "complete",
            repo_id="DiscreteRTC/dRTC",
            remote_prefix=REMOTE_PREFIX,
        )
    except Exception as exc:
        record("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
