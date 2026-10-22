#!/usr/bin/env python3
"""Build a state-relative or per-step-delta joint uniform B-spline sidecar from an absolute one.

``--action-mode rel`` (state-relative): every step of the window is relative to the
measured joints at ``t``::

    target[k, :6] = action[t + k, :6] - observation.state[t, :6]    (joints, radians)

``--action-mode delta`` (per-step): the first step is relative to the measured joints and
every later step to the previous command::

    target[0, :6] = action[t, :6] - observation.state[t, :6]
    target[k, :6] = action[t + k, :6] - action[t + k - 1, :6]        (k >= 1)

In both modes the gripper (dim 6) stays absolute.  These are exactly what starVLA's
``action_mode: rel`` / ``action_mode: delta`` with ``action_mode_apply_keys: [action.joints]``
produce for the raw 30-step chunk.  The spline fit is a ridge least-squares solve
(``||B C - A||^2 + lambda ||C||^2``), so the controls are re-fitted on the transformed
windows with the same saved encoder (same geometry, tail padding and regularization).
For ``delta`` the verification also reports the error after integrating the decoded deltas
back to absolute commands (``q[t] + cumsum``), since fit errors accumulate over the window.

The output keeps the absolute sidecar's schema (``manifest.json``, ``encoder.json``,
``data/episode_XXXXXX.npz``) plus ``action_mode``/``action_mode_apply_keys`` in the manifest,
which ``UniformBSplineLeRobotDataset`` checks against ``datasets.vla_data.action_mode``.

    python examples/realRobots/Piper/data_files/build_relative_spline_sidecar.py \\
        --dataset data/processed/piper_demos/ActionCommand/State_Joint/classify_blocks_30hz_cleanup \\
        --absolute-sidecar data/processed/piper_demos/ActionCommand/State_Joint/classify_blocks_30hz_cleanup_spline_uniform_left_s2_h30_c18 \\
        [--action-mode delta]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
BSPLINE_SRC = REPO_ROOT / "BSplineEncoder" / "src"

RELATIVE_DIMS = (0, 6)  # joints; the gripper (dim 6) stays absolute
ACTION_MODES = ("rel", "delta")
APPLY_KEYS = ["action.joints"]


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def absolute_windows(actions: np.ndarray, chunk: int) -> np.ndarray:
    """(T, chunk, D) future command windows with final-action tail padding."""
    length = len(actions)
    index = np.minimum(np.arange(length)[:, None] + np.arange(chunk)[None, :], length - 1)
    return actions[index].astype(np.float64)


def relative_windows(actions: np.ndarray, state: np.ndarray, chunk: int, mode: str = "rel") -> np.ndarray:
    """Absolute windows with the joints converted to ``mode`` (see the module docstring)."""
    windows = absolute_windows(actions, chunk)
    start, end = RELATIVE_DIMS
    joints = windows[:, :, start:end].copy()
    q_t = state[:, None, start:end].astype(np.float64)
    if mode == "rel":
        windows[:, :, start:end] = joints - q_t
    elif mode == "delta":
        windows[:, 1:, start:end] = joints[:, 1:] - joints[:, :-1]
        windows[:, :1, start:end] = joints[:, :1] - q_t
    else:
        raise ValueError(f"unsupported action mode {mode!r}; expected one of {ACTION_MODES}")
    return windows


def integrate_windows(windows: np.ndarray, state: np.ndarray, mode: str) -> np.ndarray:
    """Inverse of :func:`relative_windows`: back to absolute commands."""
    out = np.array(windows, dtype=np.float64, copy=True)
    start, end = RELATIVE_DIMS
    q_t = state[:, None, start:end].astype(np.float64)
    if mode == "delta":
        out[:, :, start:end] = np.cumsum(out[:, :, start:end], axis=1)
    out[:, :, start:end] += q_t
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Source LeRobot dataset (absolute commands)")
    parser.add_argument("--absolute-sidecar", type=Path, required=True,
                        help="Existing absolute uniform sidecar; its encoder.json is reused")
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="rel",
                        help="rel: a[t+k]-q[t]; delta: per-step a[t+k]-a[t+k-1] (a[t]-q[t] at k=0)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <absolute-sidecar>_<action-mode>_joints next to it")
    args = parser.parse_args()

    sys.path.insert(0, str(BSPLINE_SRC))
    from spline_encoder.encoding.serialization import load_encoder, save_encoder

    dataset = args.dataset.expanduser().resolve()
    absolute = args.absolute_sidecar.expanduser().resolve()
    output = (args.output or absolute.with_name(f"{absolute.name}_{args.action_mode}_joints")).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists (not overwriting): {output}")

    abs_manifest = json.loads((absolute / "manifest.json").read_text())
    if Path(abs_manifest["source_dataset"]).resolve() != dataset:
        raise ValueError(f"{absolute} was built from {abs_manifest['source_dataset']}, not {dataset}")
    if abs_manifest.get("action_mode", "abs") != "abs":
        raise ValueError(f"{absolute} is not an absolute sidecar")
    encoder = load_encoder(absolute / "encoder.json")
    config = encoder.config
    if config.mode != "uniform_left":
        raise ValueError(f"only uniform_left sidecars are supported, got {config.mode}")
    chunk = int(config.chunk_size)
    basis = np.asarray(encoder.basis, dtype=np.float64)  # (chunk, num_basis)

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    (staging / "data").mkdir(parents=True)
    digest = sha256()
    record_count = 0
    worst = np.zeros(config.action_dim)
    sum_sq = np.zeros(config.action_dim)
    abs_worst = np.zeros(config.action_dim)  # after integrating back to absolute commands
    abs_sum_sq = np.zeros(config.action_dim)
    paths = sorted((dataset / "data").rglob("episode_*.parquet"))
    try:
        for path in paths:
            frame = pd.read_parquet(path, columns=["episode_index", "frame_index", "action", "observation.state"])
            frame = frame.sort_values("frame_index")
            episode_index = int(frame["episode_index"].iloc[0])
            frame_index = frame["frame_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(frame_index, np.arange(len(frame))):
                raise ValueError(f"{path}: frame_index is not 0..T-1")
            actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
            state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
            with np.load(absolute / "data" / f"episode_{episode_index:06d}.npz") as reference:
                knots = reference["knots"]
                executable_steps = reference["executable_steps"]
                if len(reference["frame_index"]) != len(frame):
                    raise ValueError(f"episode {episode_index}: absolute sidecar length mismatch")

            windows = relative_windows(actions, state, chunk, args.action_mode)
            controls = np.stack([encoder.fit_control_points(window) for window in windows])
            decoded = np.einsum("kb,tbd->tkd", basis, controls)
            error = np.abs(decoded - windows)
            worst = np.maximum(worst, error.max(axis=(0, 1)))
            sum_sq += (error ** 2).sum(axis=(0, 1))
            abs_error = np.abs(integrate_windows(decoded, state, args.action_mode) - absolute_windows(actions, chunk))
            abs_worst = np.maximum(abs_worst, abs_error.max(axis=(0, 1)))
            abs_sum_sq += (abs_error ** 2).sum(axis=(0, 1))
            record_count += len(controls)

            digest.update(frame_index.astype("<i8").tobytes())
            digest.update(actions.astype("<f8").tobytes())
            digest.update(state[:, RELATIVE_DIMS[0]:RELATIVE_DIMS[1]].astype("<f8").tobytes())
            np.savez_compressed(
                staging / "data" / f"episode_{episode_index:06d}.npz",
                episode_index=np.asarray(episode_index, dtype=np.int64),
                frame_index=frame_index,
                observation_index=np.arange(len(frame), dtype=np.int64),
                controls=controls,
                knots=knots,
                executable_steps=executable_steps,
            )
        if record_count != int(abs_manifest["record_count"]):
            raise ValueError(f"record count {record_count} != absolute sidecar {abs_manifest['record_count']}")

        save_encoder(encoder, staging / "encoder.json")
        manifest = {
            **abs_manifest,
            "source_dataset": str(dataset),
            "source_action_fingerprint": digest.hexdigest(),
            "record_count": record_count,
            "action_mode": args.action_mode,
            "action_mode_apply_keys": APPLY_KEYS,
            "relative_to": {"state_key": "observation.state", "state_dims": list(RELATIVE_DIMS),
                            "action_dims": list(RELATIVE_DIMS), "state_index": "t (observation frame)"},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        rmse = np.sqrt(sum_sq / (record_count * chunk))
        abs_rmse = np.sqrt(abs_sum_sq / (record_count * chunk))
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "starvla_commit": git_head(REPO_ROOT),
            "bsplineencoder_commit": git_head(BSPLINE_SRC.parent),
            "dataset": str(dataset),
            "absolute_sidecar": str(absolute),
            "target": {
                "rel": "joints: action[t+k] - observation.state[t] (rad); gripper: absolute action[t+k]",
                "delta": "joints: action[t] - observation.state[t] at k=0, action[t+k] - action[t+k-1] for k>=1 "
                         "(rad); gripper: absolute action[t+k]",
            }[args.action_mode],
            "tail_policy": "uniform windows repeat the final action (same as the absolute sidecar)",
            "encoder_config": json.loads(json.dumps(config.__dict__, default=list)),
            "verification": {
                "records": record_count,
                "max_abs_reconstruction_error_per_dimension": worst.tolist(),
                "rmse_per_dimension": rmse.tolist(),
                "integrated_absolute_max_abs_error_per_dimension": abs_worst.tolist(),
                "integrated_absolute_rmse_per_dimension": abs_rmse.tolist(),
            },
        }
        (staging / "preprocessing_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), **provenance["verification"]}, indent=2))


if __name__ == "__main__":
    main()
