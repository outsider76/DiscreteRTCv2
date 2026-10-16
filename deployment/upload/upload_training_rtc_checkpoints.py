#!/usr/bin/env python3
"""Wait for, validate, upload, and verify both 5k B-spline TrainingRTC checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi


RUNS = (
    (
        "piper_pick_white_block_20260818_qwenpi_25hz_uniform_left_bspline_training_rtc_prefix4to9_c13_gb32_5k",
        "models/piper-pick-white-block-25hz-qwenpi-v3-uniform-left-controls13-gb32-50k-training-rtc-prefix4to9-5k-20260905",
    ),
    (
        "piper_pick_white_block_20260818_qwenpi_25hz_adaptive_left_bspline_training_rtc_prefix4to9_c13_gb32_5k",
        "models/piper-pick-white-block-25hz-qwenpi-v3-adaptive-left-exclude-gripper-native-controls13-gb32-50k-training-rtc-prefix4to9-5k-20260905",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_finished_runs(root: Path, timeout_seconds: int, poll_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    required = []
    for run_name, _ in RUNS:
        run_dir = root / run_name
        required.extend(
            (
                run_dir / "checkpoints/steps_5000_pytorch_model.pt",
                run_dir / "final_model/pytorch_model.pt",
                run_dir / "summary.jsonl",
            )
        )
    while True:
        ready = all(path.is_file() and path.stat().st_size > 0 for path in required)
        if ready:
            return
        if time.monotonic() >= deadline:
            missing = [str(path) for path in required if not path.is_file()]
            raise TimeoutError("Timed out waiting for finalized runs:\n" + "\n".join(missing))
        time.sleep(poll_seconds)


def validate_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint is not a non-empty state dict: {path}")
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if len(tensors) != len(state):
        raise ValueError(f"Checkpoint contains non-tensor state values: {path}")
    if any(value.numel() == 0 for value in tensors):
        raise ValueError(f"Checkpoint contains an empty tensor: {path}")
    return {
        "state_keys": len(state),
        "tensor_values": len(tensors),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/Checkpoints"))
    parser.add_argument("--repo-id", default="DiscreteRTC/dRTC")
    parser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    parser.add_argument("--timeout-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    wait_for_finished_runs(root, args.timeout_seconds, args.poll_seconds)
    api = HfApi()

    for run_name, prefix in RUNS:
        run_dir = root / run_name
        checkpoint = run_dir / "checkpoints/steps_5000_pytorch_model.pt"
        summary_rows = [
            json.loads(line)
            for line in (run_dir / "summary.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not any(row.get("steps") == 5000 for row in summary_rows):
            raise ValueError(f"No completed step-5000 save record in {run_dir / 'summary.jsonl'}")

        validation = validate_checkpoint(checkpoint)
        remote_path = f"{prefix}/checkpoints/steps_5000_pytorch_model.pt"
        print(f"Uploading {checkpoint} -> {args.repo_id}/{remote_path}", flush=True)
        commit = api.upload_file(
            path_or_fileobj=str(checkpoint),
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload {run_name} final TrainingRTC checkpoint",
        )
        info = api.get_paths_info(
            args.repo_id,
            [remote_path],
            repo_type=args.repo_type,
            revision=commit.oid,
        )[0]
        remote_lfs = getattr(info, "lfs", None)
        remote_sha256 = getattr(remote_lfs, "sha256", None)
        if info.size != validation["size_bytes"] or remote_sha256 != validation["sha256"]:
            raise RuntimeError(f"Remote size or LFS SHA-256 mismatch for {remote_path}")

        validation.update(
            run_name=run_name,
            remote_repo_id=args.repo_id,
            remote_repo_type=args.repo_type,
            remote_path=remote_path,
            remote_commit=commit.oid,
            remote_size_matches=True,
            remote_lfs_sha256_matches=True,
        )
        (run_dir / "rtc_checkpoint_upload.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        print(json.dumps(validation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
