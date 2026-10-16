#!/usr/bin/env python3
"""Visualize an actual adaptive-left spline, its basis, and row interleaving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = REPO_ROOT / "data/20260818_piper_pick_white_block_25hz"
DEFAULT_ENCODED = REPO_ROOT / "data/20260818_piper_pick_white_block_25hz_AdaptiveBSpline"
DEFAULT_OUTPUT = REPO_ROOT / (
    "results/Checkpoints/piper_pick_white_block_20260818_qwenpi_25hz_"
    "adaptive_left_bspline_excl_gripper_native_c13_gb32_50k/"
    "adaptive_left_representation.png"
)


def _episode_parquet(root: Path, episode: int) -> Path:
    paths = list(root.glob(f"data/chunk-*/episode_{episode:06d}.parquet"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one parquet for episode {episode}, got {paths}")
    return paths[0]


def _choose_transition(source: Path, encoded: Path) -> tuple[int, int, np.ndarray]:
    best = None
    for sidecar in sorted((encoded / "data").glob("episode_*.npz")):
        episode = int(sidecar.stem.split("_")[-1])
        actions = np.stack(pd.read_parquet(_episode_parquet(source, episode))["action"])
        with np.load(sidecar) as data:
            valid = set(map(int, data["frame_index"]))
        deltas = np.abs(np.diff(actions[:, 6]))
        for transition in np.argsort(deltas)[::-1]:
            frame = max(0, int(transition) + 1 - 4)
            if frame in valid:
                candidate = (float(deltas[transition]), episode, frame, actions)
                if best is None or candidate[0] > best[0]:
                    best = candidate
                break
    if best is None:
        raise RuntimeError("could not find a valid gripper-transition observation")
    return best[1], best[2], best[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--encoded", type=Path, default=DEFAULT_ENCODED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", type=float, default=25.0)
    args = parser.parse_args()
    source, encoded = args.source.resolve(), args.encoded.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    episode, frame, actions = _choose_transition(source, encoded)

    sidecar_path = encoded / "data" / f"episode_{episode:06d}.npz"
    with np.load(sidecar_path) as data:
        matches = np.flatnonzero(data["frame_index"] == frame)
        if len(matches) != 1:
            raise ValueError(f"frame {frame} appears {len(matches)} times in {sidecar_path}")
        row = int(matches[0])
        controls = data["controls"][row].astype(np.float64)
        durations = data["duration_steps"][row].astype(np.int64)
        interleaved = data["control_duration_interleaved"][row].astype(np.float64)
        knots = data["knots"][row].astype(np.float64)
        executable_steps = int(data["executable_steps"][row])

    if interleaved.shape != (13, 8):
        raise ValueError(f"unexpected interleaved shape {interleaved.shape}")
    np.testing.assert_array_equal(interleaved[:, :7], controls)
    np.testing.assert_array_equal(interleaved[:, 7], durations)
    degree = 3
    reconstructed_knots = np.r_[
        np.zeros(degree + 1), np.cumsum(durations) / args.frequency
    ]
    np.testing.assert_allclose(reconstructed_knots, knots, atol=1e-12, rtol=0)

    executable_end = executable_steps / args.frequency
    support_end = float(knots[-1])
    dense_time = np.linspace(0.0, executable_end, 1200, endpoint=False)
    spline = BSpline(knots, controls, degree, extrapolate=False, axis=0)
    dense = spline(dense_time)
    if not np.isfinite(dense).all():
        raise ValueError("non-finite spline values inside executable domain")
    sample_count = min(executable_steps, len(actions) - frame)
    sample_time = np.arange(sample_count) / args.frequency
    sample_actions = actions[frame : frame + sample_count]
    greville = np.array([
        knots[index + 1 : index + degree + 1].mean()
        for index in range(len(controls))
    ])
    arm_dim = int(np.argmax(np.ptp(sample_actions[:, :6], axis=0)))

    basis = BSpline(knots, np.eye(len(controls)), degree, extrapolate=False)(dense_time)
    partition_error = float(np.max(np.abs(basis.sum(axis=1) - 1.0)))
    sampled_fit = spline(sample_time)
    fit_error = sampled_fit - sample_actions

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 0.9))
    axes = [fig.add_subplot(grid[index]) for index in range(3)]
    dims = [(arm_dim, f"Joint {arm_dim + 1}"), (6, "Gripper (excluded from knot insertion)")]
    distinct = np.unique(knots)
    for axis, (dimension, label) in zip(axes[:2], dims):
        axis.plot(dense_time, dense[:, dimension], lw=2.5, color="#0068b5",
                  label="evaluated cubic B-spline")
        axis.scatter(sample_time, sample_actions[:, dimension], s=24, color="black",
                     zorder=4, label="recorded 25 Hz actions")
        axis.plot(greville, controls[:, dimension], "--s", lw=1.2, ms=5,
                  color="#d1495b", alpha=0.85,
                  label="control polygon (not the trajectory)")
        for knot in distinct[1:-1]:
            axis.axvline(knot, color="#6c757d", lw=0.7,
                         ls="-" if knot <= executable_end else ":", alpha=0.45)
        axis.axvspan(executable_end, support_end, color="#f2cc8f", alpha=0.2,
                     label="right-context support")
        axis.axvline(executable_end, color="#e07a5f", lw=1.6, ls="--",
                     label="native execution end")
        axis.set_xlim(0, support_end)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].set_title(
        f"Actual left-clamped adaptive B-spline — episode {episode}, frame {frame}"
    )
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].legend(ncol=2, fontsize=9)

    for index in range(basis.shape[1]):
        axes[2].plot(dense_time, basis[:, index], lw=1.2, label=f"B{index}")
    axes[2].set_xlim(0, executable_end)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Seconds from observation (native 25 Hz grid)")
    axes[2].set_ylabel("Basis weight")
    axes[2].set_title(
        "Cubic B-spline basis: curve = sum(control coefficient × basis weight); "
        "durations are interleaved after each 7-D control row"
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(ncol=7, fontsize=7, loc="upper right")
    fig.savefig(output, dpi=180)
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path)
    plt.close(fig)

    report = {
        "episode_index": episode,
        "frame_index": frame,
        "control_shape": list(controls.shape),
        "interleaved_shape": list(interleaved.shape),
        "interleaved_row_semantics": "[control_dim_0 ... control_dim_6, duration_steps]",
        "controls_round_trip_exact": bool(np.array_equal(interleaved[:, :7], controls)),
        "durations_round_trip_exact": bool(np.array_equal(interleaved[:, 7], durations)),
        "knots_reconstructed_from_durations_max_error": float(
            np.max(np.abs(reconstructed_knots - knots))
        ),
        "degree": degree,
        "left_clamped_zero_multiplicity": int(np.sum(knots == 0)),
        "native_executable_steps": executable_steps,
        "duration_steps": durations.tolist(),
        "basis_partition_of_unity_max_error": partition_error,
        "selected_arm_dimension": arm_dim,
        "sampled_arm_rmse": float(np.sqrt(np.mean(fit_error[:, :6] ** 2))),
        "sampled_gripper_rmse": float(np.sqrt(np.mean(fit_error[:, 6] ** 2))),
        "note": (
            "The dashed control polygon is shown only to locate coefficients. "
            "The solid trajectory is evaluated from the cubic B-spline basis."
        ),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    print(svg_path)
    print(report_path)


if __name__ == "__main__":
    main()
