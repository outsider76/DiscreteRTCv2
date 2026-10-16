#!/usr/bin/env python3
"""Wait for base runs, execute TrainingRTC, then validate and upload final artifacts."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results/Checkpoints"
RAW_RUN = "stacking_cups_20260911_qwenpi_raw_h30_2gpu_gb32_50k"
SPLINE_RUN = "stacking_cups_20260911_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_50k"
RTC_RUN = "stacking_cups_20260911_qwenpi_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k"
RTC_LAUNCHER = REPO_ROOT / "examples/realRobots/Piper/train_files/run_qwenpi_stacking_cups_20260911_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k.sh"
UPLOADER = REPO_ROOT / "deployment/upload/upload_stacking_cups_checkpoints.py"
STATE_PATH = RESULTS_ROOT / "stacking_cups_20260911_pipeline_state.json"
POLL_SECONDS = 300


def record(stage: str, **fields: object) -> None:
    payload = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def checkpoint_ready(run_name: str, step: int) -> bool:
    run_dir = RESULTS_ROOT / run_name
    checkpoint = run_dir / f"checkpoints/steps_{step}_pytorch_model.pt"
    summary = run_dir / "summary.jsonl"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not summary.is_file():
        return False
    rows = [json.loads(line) for line in summary.read_text().splitlines() if line.strip()]
    return any(row.get("steps") == step for row in rows)


def run_active(run_name: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "train_starvla.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    return any(f"--run_id {run_name}" in line for line in result.stdout.splitlines())


def wait_for_base_runs() -> None:
    record("waiting_for_base_runs", poll_seconds=POLL_SECONDS)
    while True:
        raw_ready = checkpoint_ready(RAW_RUN, 50_000)
        spline_ready = checkpoint_ready(SPLINE_RUN, 50_000)
        raw_active = run_active(RAW_RUN)
        spline_active = run_active(SPLINE_RUN)
        if raw_ready and spline_ready and not raw_active and not spline_active:
            record("base_runs_complete")
            return
        failed = []
        if not raw_ready and not raw_active:
            failed.append(RAW_RUN)
        if not spline_ready and not spline_active:
            failed.append(SPLINE_RUN)
        if failed:
            record("failed", reason="base run exited before final checkpoint", runs=failed)
            raise RuntimeError("Base run(s) exited before completion: " + ", ".join(failed))
        time.sleep(POLL_SECONDS)


def run_rtc() -> None:
    if checkpoint_ready(RTC_RUN, 5_000):
        record("rtc_complete", reused_existing=True)
        return
    if (RESULTS_ROOT / RTC_RUN).exists():
        raise FileExistsError(f"Refusing to mix a partial RTC run: {RESULTS_ROOT / RTC_RUN}")
    record("rtc_running", launcher=str(RTC_LAUNCHER))
    log_path = RESULTS_ROOT / "stacking_cups_20260911_training_rtc.log"
    with log_path.open("w") as log:
        subprocess.run(
            [str(RTC_LAUNCHER)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not checkpoint_ready(RTC_RUN, 5_000):
        raise RuntimeError("RTC launcher exited without a completed step-5000 checkpoint")
    record("rtc_complete", log=str(log_path))


def upload() -> None:
    record("uploading")
    log_path = RESULTS_ROOT / "stacking_cups_20260911_hf_upload.log"
    with log_path.open("w") as log:
        subprocess.run(
            [str(Path.home() / "miniconda3/envs/starVLA/bin/python"), str(UPLOADER)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    record("complete", upload_log=str(log_path), repo_id="DiscreteRTC/dRTC")


def main() -> None:
    try:
        wait_for_base_runs()
        run_rtc()
        upload()
    except Exception as exc:
        record("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
