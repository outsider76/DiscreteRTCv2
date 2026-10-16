#!/usr/bin/env python3
"""Upload and verify the finalized Piper adaptive B-spline model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "DiscreteRTC/dRTC"
DEFAULT_PREFIX = (
    "models/piper-pick-white-block-25hz-qwenpi-v3-adaptive-left-"
    "exclude-gripper-native-controls13-gb32-50k-20260905"
)
REQUIRED_RUN_FILES = (
    "README.md",
    "checkpoint_validation.json",
    "config.full.yaml",
    "config.yaml",
    "dataset_statistics.json",
    "training_summary.json",
    "wandb-summary.json",
    "adaptive_open_loop_evaluation_all.json",
    "adaptive_open_loop_errors_all.npz",
    "open_loop_comparison_vs_uniform_50k.json",
    "adaptive_open_loop_comparison.png",
    "adaptive_open_loop_comparison.svg",
    "adaptive_left_representation.json",
    "adaptive_left_representation.png",
    "adaptive_left_representation.svg",
)
REPRESENTATION_FILES = (
    "encoder.json",
    "manifest.json",
    "validation.json",
    "representation_comparison.json",
    "training_target_validation.json",
)
CODE_FILES = (
    "BSplineEncoder/src/spline_encoder/__init__.py",
    "BSplineEncoder/src/spline_encoder/adaptive.py",
    "BSplineEncoder/src/spline_encoder/config.py",
    "BSplineEncoder/src/spline_encoder/encoder.py",
    "BSplineEncoder/src/spline_encoder/lerobot.py",
    "BSplineEncoder/src/spline_encoder/result.py",
    "deployment/model_server/policy_wrapper.py",
    "starVLA/model/framework/VLM4A/QwenPI_v3_AdaptiveBSpline.py",
    "examples/realRobots/Piper/train_files/data_registry/data_config.py",
    "examples/realRobots/Piper/train_files/validate_adaptive_bspline_training_target.py",
    "examples/realRobots/Piper/eval_files/evaluate_adaptive_bspline_open_loop.py",
    "examples/realRobots/Piper/eval_files/compare_adaptive_uniform_open_loop.py",
    "examples/realRobots/Piper/eval_files/visualize_adaptive_bspline_open_loop.py",
    "examples/realRobots/Piper/eval_files/visualize_adaptive_bspline_representation.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("representation_dir", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--repo-type", default="dataset", choices=("dataset", "model"))
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    representation_dir = args.representation_dir.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint = run_dir / "checkpoints/steps_50000_pytorch_model.pt"
    sources = [(checkpoint, "checkpoints/steps_50000_pytorch_model.pt")]
    sources.extend((run_dir / name, name) for name in REQUIRED_RUN_FILES)
    sources.extend(
        (representation_dir / name, f"representation/{name}")
        for name in REPRESENTATION_FILES
    )
    sources.extend((repo_root / name, f"code/{name}") for name in CODE_FILES)
    missing = [str(source) for source, _ in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError("Finalized artifact is incomplete:\n" + "\n".join(missing))

    api = HfApi()
    for source, relative in sources:
        destination = f"{args.prefix.rstrip('/')}/{relative}"
        print(f"uploading {source} -> {destination}", flush=True)
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=destination,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload Piper adaptive B-spline {relative}",
        )

    expected = {f"{args.prefix.rstrip('/')}/{relative}" for _, relative in sources}
    remote = set(api.list_repo_files(args.repo_id, repo_type=args.repo_type))
    absent = sorted(expected - remote)
    if absent:
        raise RuntimeError("Upload verification failed; missing remote files:\n" + "\n".join(absent))
    info = api.get_paths_info(
        args.repo_id,
        [f"{args.prefix.rstrip('/')}/checkpoints/steps_50000_pytorch_model.pt"],
        repo_type=args.repo_type,
    )[0]
    remote_size = getattr(info, "size", None)
    if remote_size != checkpoint.stat().st_size:
        raise RuntimeError(
            f"Remote checkpoint size {remote_size} differs from local {checkpoint.stat().st_size}"
        )
    local_sha256 = _sha256(checkpoint)
    remote_lfs = getattr(info, "lfs", None)
    remote_sha256 = getattr(remote_lfs, "sha256", None)
    if remote_sha256 != local_sha256:
        raise RuntimeError(
            f"Remote LFS SHA-256 {remote_sha256} differs from local {local_sha256}"
        )

    validation_path = run_dir / "checkpoint_validation.json"
    validation = json.loads(validation_path.read_text())
    if validation.get("checkpoint_sha256") != local_sha256:
        raise RuntimeError("checkpoint_validation.json contains a stale checkpoint SHA-256")
    validation.update(
        remote_lfs_size_matches=True,
        remote_lfs_sha256_matches=True,
        remote_repo_id=args.repo_id,
        remote_repo_type=args.repo_type,
        remote_path=(
            f"{args.prefix.rstrip('/')}/checkpoints/steps_50000_pytorch_model.pt"
        ),
    )
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    validation_destination = f"{args.prefix.rstrip('/')}/checkpoint_validation.json"
    validation_commit = api.upload_file(
        path_or_fileobj=str(validation_path),
        path_in_repo=validation_destination,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message="Record verified Piper adaptive B-spline LFS integrity",
    )
    downloaded_validation = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=validation_destination,
            repo_type=args.repo_type,
            revision=validation_commit.oid,
            force_download=True,
        )
    )
    if downloaded_validation.read_bytes() != validation_path.read_bytes():
        raise RuntimeError("Remote checkpoint_validation.json differs from local file")
    print(
        f"verified {len(expected)} files, checkpoint size, and LFS SHA-256 at "
        f"https://huggingface.co/datasets/{args.repo_id}/tree/main/{args.prefix}"
    )


if __name__ == "__main__":
    main()
