#!/usr/bin/env python3
"""Wait for, validate, upload, and verify the two cubes-classification checkpoints."""

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
        "cubes_classification_30hz_qwenpi_raw_h30_gb32_30k",
        "models/cubes-classification-30hz-qwenpi-v3-raw-h30-gb32-30k-20260908",
    ),
    (
        "cubes_classification_30hz_qwenpi_uniform_left_s2_h30_c18_gb32_30k",
        "models/cubes-classification-30hz-qwenpi-v3-uniform-left-s2-h30-c18-gb32-30k-20260908",
    ),
)

ARTIFACTS = (
    ("checkpoints/steps_30000_pytorch_model.pt", "checkpoints/steps_30000_pytorch_model.pt"),
    ("config.yaml", "config.yaml"),
    ("config.full.yaml", "config.full.yaml"),
    ("dataset_statistics.json", "dataset_statistics.json"),
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
                run_dir / "checkpoints/steps_30000_pytorch_model.pt",
                run_dir / "summary.jsonl",
            )
        )
    while True:
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
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
    parser.add_argument("--timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    wait_for_finished_runs(root, args.timeout_seconds, args.poll_seconds)
    api = HfApi()

    for run_name, prefix in RUNS:
        run_dir = root / run_name
        checkpoint = run_dir / "checkpoints/steps_30000_pytorch_model.pt"
        summary_rows = [
            json.loads(line)
            for line in (run_dir / "summary.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not any(row.get("steps") == 30000 for row in summary_rows):
            raise ValueError(f"No completed step-30000 save record in {run_dir / 'summary.jsonl'}")

        validation = validate_checkpoint(checkpoint)
        remote_paths = []
        last_commit = None
        for local_relative, remote_relative in ARTIFACTS:
            local_path = run_dir / local_relative
            if not local_path.is_file() or local_path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing finalized artifact: {local_path}")
            remote_path = f"{prefix}/{remote_relative}"
            print(f"Uploading {local_path} -> {args.repo_id}/{remote_path}", flush=True)
            last_commit = api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                commit_message=f"Upload {run_name} {remote_relative}",
            )
            remote_paths.append(remote_path)

        checkpoint_remote_path = f"{prefix}/checkpoints/steps_30000_pytorch_model.pt"
        info = api.get_paths_info(
            args.repo_id,
            remote_paths,
            repo_type=args.repo_type,
            revision=last_commit.oid,
        )
        by_path = {item.path: item for item in info}
        if set(by_path) != set(remote_paths):
            raise RuntimeError(f"Remote artifact set is incomplete for {prefix}")
        checkpoint_info = by_path[checkpoint_remote_path]
        remote_sha256 = getattr(getattr(checkpoint_info, "lfs", None), "sha256", None)
        if (
            checkpoint_info.size != validation["size_bytes"]
            or remote_sha256 != validation["sha256"]
        ):
            raise RuntimeError(
                f"Remote size or LFS SHA-256 mismatch for {checkpoint_remote_path}"
            )

        validation.update(
            run_name=run_name,
            remote_repo_id=args.repo_id,
            remote_repo_type=args.repo_type,
            remote_path=checkpoint_remote_path,
            remote_artifacts=remote_paths,
            remote_commit=last_commit.oid,
            remote_size_matches=True,
            remote_lfs_sha256_matches=True,
        )
        (run_dir / "checkpoint_upload.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        print(json.dumps(validation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
