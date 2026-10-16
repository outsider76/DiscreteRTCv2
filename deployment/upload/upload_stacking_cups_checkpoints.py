#!/usr/bin/env python3
"""Validate, upload, and remotely verify the three final stacking-cups checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi


RUNS = (
    (
        "stacking_cups_20260911_qwenpi_raw_h30_2gpu_gb32_50k",
        50_000,
        "models/stacking-cups-20260911-qwenpi-v3-raw-h30-2gpu-gb32-50k-20260912",
    ),
    (
        "stacking_cups_20260911_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_50k",
        50_000,
        "models/stacking-cups-20260911-qwenpi-v3-uniform-left-s2-h30-c18-2gpu-gb32-50k-20260912",
    ),
    (
        "stacking_cups_20260911_qwenpi_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k",
        5_000,
        "models/stacking-cups-20260911-qwenpi-v3-uniform-left-s2-h30-c18-2gpu-gb32-50k-training-rtc-prefix4to8-ft5k-20260912",
    ),
)

RUN_METADATA = ("config.yaml", "config.full.yaml", "dataset_statistics.json", "summary.jsonl")
REPRESENTATION_METADATA = ("encoder.json", "manifest.json", "validation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> dict[str, int | str]:
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


def completed_step(summary_path: Path, step: int) -> bool:
    rows = [
        json.loads(line)
        for line in summary_path.read_text().splitlines()
        if line.strip()
    ]
    return any(row.get("steps") == step for row in rows)


def upload_and_verify(
    api: HfApi,
    *,
    run_dir: Path,
    step: int,
    prefix: str,
    repo_id: str,
    repo_type: str,
) -> None:
    checkpoint_relative = f"checkpoints/steps_{step}_pytorch_model.pt"
    checkpoint = run_dir / checkpoint_relative
    required = [checkpoint, *(run_dir / name for name in RUN_METADATA)]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Finalized artifacts are incomplete:\n" + "\n".join(missing))
    if not completed_step(run_dir / "summary.jsonl", step):
        raise ValueError(f"No completed step-{step} save record in {run_dir / 'summary.jsonl'}")

    validation = validate_checkpoint(checkpoint)
    artifacts = [(checkpoint, checkpoint_relative)]
    artifacts.extend((run_dir / name, name) for name in RUN_METADATA)
    last_commit = None
    remote_paths: list[str] = []
    for source, relative in artifacts:
        remote_path = f"{prefix}/{relative}"
        print(f"Uploading {source} -> {repo_id}/{remote_path}", flush=True)
        last_commit = api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {run_dir.name} {relative}",
        )
        remote_paths.append(remote_path)

    assert last_commit is not None
    info = api.get_paths_info(
        repo_id,
        remote_paths,
        repo_type=repo_type,
        revision=last_commit.oid,
    )
    by_path = {item.path: item for item in info}
    if set(by_path) != set(remote_paths):
        raise RuntimeError(f"Remote artifact set is incomplete for {prefix}")
    checkpoint_remote = f"{prefix}/{checkpoint_relative}"
    checkpoint_info = by_path[checkpoint_remote]
    remote_sha256 = getattr(getattr(checkpoint_info, "lfs", None), "sha256", None)
    if checkpoint_info.size != validation["size_bytes"] or remote_sha256 != validation["sha256"]:
        raise RuntimeError(f"Remote size or LFS SHA-256 mismatch for {checkpoint_remote}")

    validation.update(
        run_name=run_dir.name,
        completed_steps=step,
        remote_repo_id=repo_id,
        remote_repo_type=repo_type,
        remote_path=checkpoint_remote,
        remote_artifacts=remote_paths,
        remote_commit=last_commit.oid,
        remote_size_matches=True,
        remote_lfs_sha256_matches=True,
    )
    validation_path = run_dir / "checkpoint_upload.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    api.upload_file(
        path_or_fileobj=str(validation_path),
        path_in_repo=f"{prefix}/checkpoint_upload.json",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Record verified upload for {run_dir.name}",
    )
    print(json.dumps(validation, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/Checkpoints"))
    parser.add_argument("--representation-dir", type=Path, default=Path("data/stacking_cups/20260911_UniformBSpline"))
    parser.add_argument("--repo-id", default="DiscreteRTC/dRTC")
    parser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    representation_dir = args.representation_dir.expanduser().resolve()
    api = HfApi()
    for run_name, step, prefix in RUNS:
        upload_and_verify(
            api,
            run_dir=root / run_name,
            step=step,
            prefix=prefix,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
        )

    representation_prefix = RUNS[1][2] + "/representation"
    for name in REPRESENTATION_METADATA:
        source = representation_dir / name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing representation metadata: {source}")
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=f"{representation_prefix}/{name}",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload stacking-cups B-spline representation {name}",
        )
    print("All three checkpoints and representation metadata uploaded and verified.", flush=True)


if __name__ == "__main__":
    main()
