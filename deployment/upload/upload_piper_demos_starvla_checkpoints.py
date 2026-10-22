#!/usr/bin/env python3
"""Wait for, validate, upload, and verify the piper_demos classify_blocks starVLA checkpoints.

Each run is uploaded to ``<repo>/StarVLA/<run_name>/`` with the run-directory layout
``from_pretrained`` expects (``checkpoints/steps_N_pytorch_model.pt`` next to
``config.yaml`` and ``dataset_statistics.json`` two levels up), so the downloaded
folder is directly loadable.  One commit per run; the remote LFS SHA-256 is checked
against the local file and recorded in ``<run_dir>/checkpoint_upload.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi


RUNS = (
    "piper_demos_classify_blocks_30hz_qwenpi_raw_h30_2gpu_gb32_30k",
    "piper_demos_classify_blocks_30hz_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_30k",
    # Qwen3-VL-2B, delta (state-relative) joint targets; stopped on request at 2026-10-19.
    "piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_raw_h30_reljoints_2gpu_gb32_30k",
    "piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_uniform_left_s2_h30_c18_reljoints_2gpu_gb32_30k",
    # Qwen3-VL-2B, per-step delta joints (StateDeltaJoint dataset): 10k runs and their 30k continuations; 2026-10-22.
    "piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_raw_h30_2gpu_gb32_10k",
    "piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_uniform_left_s2_h30_c18_2gpu_gb32_10k",
    "piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_raw_h30_2gpu_gb32_30k",
    "piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_uniform_left_s2_h30_c18_2gpu_gb32_30k",
)

REMOTE_ROOT = "StarVLA"


def artifacts(step: int) -> tuple[tuple[str, bool], ...]:
    """(run-relative path, required)."""
    return (
        (f"checkpoints/steps_{step}_pytorch_model.pt", True),
        ("config.yaml", True),
        ("config.full.yaml", True),
        ("dataset_statistics.json", True),
        ("summary.jsonl", True),
        ("wall_time_ledger.json", False),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saved_steps(run_dir: Path) -> set[int]:
    summary = run_dir / "summary.jsonl"
    if not summary.is_file():
        return set()
    return {
        json.loads(line).get("steps")
        for line in summary.read_text().splitlines()
        if line.strip()
    }


def run_finished(run_dir: Path, step: int, require_final: bool = True) -> bool:
    # final_model/ is written after the last steps_N save, so its presence marks a clean finish.
    # Runs stopped on purpose before max_train_steps have no final_model/ (``--stopped-early``).
    final = run_dir / "final_model"
    return (
        step in saved_steps(run_dir)
        and (run_dir / f"checkpoints/steps_{step}_pytorch_model.pt").is_file()
        and (not require_final or (final.is_dir() and any(final.iterdir())))
    )


def wait_for_finished_runs(
    root: Path, runs: list[str], step: int, timeout_seconds: int, poll_seconds: int, require_final: bool = True
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        pending = [run for run in runs if not run_finished(root / run, step, require_final)]
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for finished runs:\n" + "\n".join(pending))
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} waiting for: {', '.join(pending)}", flush=True)
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
    if not all(torch.isfinite(value).all() for value in tensors if value.is_floating_point()):
        raise ValueError(f"Checkpoint contains non-finite values: {path}")
    return {
        "state_keys": len(state),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/Checkpoints"))
    parser.add_argument("--repo-id", default="DiscreteRTC/dRTC")
    parser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    parser.add_argument("--step", type=int, default=30000)
    parser.add_argument("--timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS[:2]),
                        help="Runs to wait for and upload (default: the two 4B classify_blocks runs).")
    parser.add_argument("--stopped-early", action="store_true",
                        help="Upload --step from a run stopped before max_train_steps (no final_model/ required).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list uploads only.")
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    require_final = not args.stopped_early
    if not args.dry_run:
        wait_for_finished_runs(
            root, args.runs, args.step, args.timeout_seconds, args.poll_seconds, require_final
        )
    api = HfApi()

    for run_name in args.runs:
        run_dir = root / run_name
        if not run_finished(run_dir, args.step, require_final):
            raise RuntimeError(f"Run has no finished step-{args.step} checkpoint: {run_dir}")
        checkpoint_rel = f"checkpoints/steps_{args.step}_pytorch_model.pt"
        validation = validate_checkpoint(run_dir / checkpoint_rel)

        prefix = f"{REMOTE_ROOT}/{run_name}"
        operations = []
        for relative, required in artifacts(args.step):
            local_path = run_dir / relative
            if not local_path.is_file() or local_path.stat().st_size == 0:
                if required:
                    raise FileNotFoundError(f"Missing finalized artifact: {local_path}")
                continue
            operations.append(
                CommitOperationAdd(path_in_repo=f"{prefix}/{relative}", path_or_fileobj=str(local_path))
            )
        remote_paths = [op.path_in_repo for op in operations]
        print(f"{run_name}: {json.dumps(validation)}", flush=True)
        for path in remote_paths:
            print(f"  -> {args.repo_id}/{path}", flush=True)
        if args.dry_run:
            continue

        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            operations=operations,
            commit_message=f"Upload starVLA checkpoint {run_name} step {args.step}",
        )
        info = api.get_paths_info(args.repo_id, remote_paths, repo_type=args.repo_type, revision=commit.oid)
        by_path = {item.path: item for item in info}
        if set(by_path) != set(remote_paths):
            raise RuntimeError(f"Remote artifact set is incomplete for {prefix}")
        checkpoint_info = by_path[f"{prefix}/{checkpoint_rel}"]
        remote_sha256 = getattr(getattr(checkpoint_info, "lfs", None), "sha256", None)
        if checkpoint_info.size != validation["size_bytes"] or remote_sha256 != validation["sha256"]:
            raise RuntimeError(f"Remote size or LFS SHA-256 mismatch for {prefix}/{checkpoint_rel}")

        validation.update(
            run_name=run_name,
            step=args.step,
            remote_repo_id=args.repo_id,
            remote_repo_type=args.repo_type,
            remote_checkpoint=f"{prefix}/{checkpoint_rel}",
            remote_artifacts=remote_paths,
            remote_commit=commit.oid,
            remote_size_matches=True,
            remote_lfs_sha256_matches=True,
        )
        (run_dir / "checkpoint_upload.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
