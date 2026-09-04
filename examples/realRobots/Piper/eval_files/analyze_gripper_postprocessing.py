#!/usr/bin/env python3
"""Analyze Piper gripper targets, spline controls, and post-train projection.

The stored open-loop error arrays are sufficient to reconstruct every decoded
gripper prediction without running the 4B policy again.  This script compares
continuous regression metrics and the binary command semantics used by the
Piper execution client, then creates publication-ready distribution plots.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = REPO_ROOT / "data/20260818_piper_pick_white_block_25hz"
DEFAULT_CONTROLS = REPO_ROOT / "data/20260818_piper_pick_white_block_25hz_UniformBSpline"
DEFAULT_RUN = REPO_ROOT / (
    "results/Checkpoints/"
    "piper_pick_white_block_20260818_qwenpi_25hz_uniform_left_bspline_s2_h20_c13_gb32_50k"
)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(x.size),
        "min": float(x.min()),
        "q001": float(np.quantile(x, 0.001)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "q999": float(np.quantile(x, 0.999)),
        "max": float(x.max()),
    }


def _error_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error": float(absolute.max()),
    }


def _episode_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_actions(dataset: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    episodes: dict[int, np.ndarray] = {}
    gripper = []
    deltas = []
    paths = sorted(dataset.glob("data/*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files under {dataset}")
    for path in paths:
        actions = np.stack(pd.read_parquet(path, columns=["action"])["action"].to_numpy())
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Unexpected action shape in {path}: {actions.shape}")
        episodes[_episode_id(path)] = actions.astype(np.float32)
        gripper.append(actions[:, 6])
        deltas.append(np.diff(actions[:, 6]))
    return episodes, np.concatenate(gripper), np.concatenate(deltas)


def _load_controls(controls_root: Path) -> np.ndarray:
    values = []
    paths = sorted((controls_root / "data").glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No spline-control files under {controls_root}")
    for path in paths:
        with np.load(path) as episode:
            controls = np.asarray(episode["controls"], dtype=np.float64)
        if controls.ndim != 3 or controls.shape[1:] != (13, 7):
            raise ValueError(f"Unexpected controls shape in {path}: {controls.shape}")
        values.append(controls[:, :, 6])
    return np.concatenate(values, axis=0)


def _reconstruct_open_loop(
    errors_path: Path,
    episodes: dict[int, np.ndarray],
) -> tuple[np.lib.npyio.NpzFile, np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(errors_path)
    target, prediction, spline = [], [], []
    for episode_id, frame_id, pred_error, spline_error in zip(
        archive["episode_index"],
        archive["frame_index"],
        archive["prediction_vs_original_error"],
        archive["spline_vs_original_error"],
    ):
        actions = episodes[int(episode_id)]
        indices = np.minimum(np.arange(int(frame_id), int(frame_id) + 20), len(actions) - 1)
        raw = actions[indices, 6]
        target.append(raw)
        prediction.append(raw + pred_error[:, 6])
        spline.append(raw + spline_error[:, 6])
    return archive, np.asarray(target), np.asarray(prediction), np.asarray(spline)


def _two_means(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.quantile(x, [0.2, 0.8])
    for _ in range(100):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        updated = np.asarray([x[labels == index].mean() for index in range(2)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return np.sort(centers)


def _transition_timing(
    archive: np.lib.npyio.NpzFile,
    target: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    offsets = []
    target_count = predicted_count = 0
    for episode_id in np.unique(archive["episode_index"]):
        rows = archive["episode_index"] == episode_id
        order = np.argsort(archive["frame_index"][rows])
        truth = target[rows, 0][order] > threshold
        pred = prediction[rows, 0][order] > threshold
        truth_events = np.flatnonzero(np.diff(truth.astype(np.int8)) != 0) + 1
        pred_events = np.flatnonzero(np.diff(pred.astype(np.int8)) != 0) + 1
        target_count += len(truth_events)
        predicted_count += len(pred_events)
        if len(truth_events) == len(pred_events):
            offsets.extend((pred_events - truth_events).tolist())
    offsets_array = np.asarray(offsets, dtype=np.int64)
    return {
        "target_transition_count": int(target_count),
        "predicted_transition_count": int(predicted_count),
        "matched_by_order_count": int(offsets_array.size),
        "exact_frame_fraction": float(np.mean(offsets_array == 0)),
        "within_one_frame_fraction": float(np.mean(np.abs(offsets_array) <= 1)),
        "mean_signed_offset_frames": float(offsets_array.mean()),
        "max_absolute_offset_frames": int(np.abs(offsets_array).max()),
    }


def _plot_control_distribution(
    output: Path,
    raw: np.ndarray,
    controls: np.ndarray,
) -> None:
    raw_min, raw_max = float(raw.min()), float(raw.max())
    quantiles = np.quantile(controls, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0)
    outside = np.mean((controls < raw_min) | (controls > raw_max), axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    bins = np.linspace(min(controls.min(), raw_min), max(controls.max(), raw_max), 150)
    ax.hist(raw, bins=bins, density=True, alpha=0.55, label="raw gripper actions", color="#377eb8")
    ax.hist(controls.ravel(), bins=bins, density=True, histtype="step", linewidth=1.8,
            label="all 13 control points", color="#e41a1c")
    ax.axvline(raw_min, color="black", linestyle="--", linewidth=1)
    ax.axvline(raw_max, color="black", linestyle="--", linewidth=1, label="raw min/max")
    ax.set(title="Raw actions vs. fitted B-spline controls", xlabel="gripper value", ylabel="density")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    positions = np.arange(13)
    ax.fill_between(positions, quantiles[0], quantiles[4], alpha=0.18, color="#984ea3", label="1–99%")
    ax.fill_between(positions, quantiles[1], quantiles[3], alpha=0.40, color="#984ea3", label="25–75%")
    ax.plot(positions, quantiles[2], "o-", color="#4d1b69", markersize=4, label="median")
    ax.axhline(raw_min, color="black", linestyle="--", linewidth=1)
    ax.axhline(raw_max, color="black", linestyle="--", linewidth=1)
    ax.set(title="Distribution by left-clamped control index", xlabel="control-point index", ylabel="gripper control value")
    ax.set_xticks(positions)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    value_bins = np.linspace(controls.min(), controls.max(), 120)
    heatmap = np.stack([np.histogram(controls[:, index], bins=value_bins)[0] for index in positions], axis=1)
    heatmap = np.log1p(heatmap)
    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[-0.5, 12.5, value_bins[0], value_bins[-1]],
        cmap="magma",
    )
    ax.axhline(raw_min, color="white", linestyle="--", linewidth=0.9)
    ax.axhline(raw_max, color="white", linestyle="--", linewidth=0.9)
    ax.set(title="Control density by index (log count)", xlabel="control-point index", ylabel="gripper control value")
    ax.set_xticks(positions)
    fig.colorbar(image, ax=ax, label="log(1 + count)")

    ax = axes[1, 1]
    ax.bar(positions, 100.0 * outside, color="#ff7f00")
    ax.set(title="Controls outside the raw action range", xlabel="control-point index", ylabel="outside raw min/max (%)")
    ax.set_xticks(positions)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Piper gripper — uniform-left B-spline control-point distribution", fontsize=15)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"gripper_control_point_distribution.{suffix}", dpi=180)
    plt.close(fig)


def _plot_dashboard(
    output: Path,
    archive: np.lib.npyio.NpzFile,
    raw: np.ndarray,
    delta: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    spline: np.ndarray,
    threshold: float,
) -> None:
    binary_target = target > threshold
    binary_prediction = prediction > threshold
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    bins = np.linspace(min(raw.min(), prediction.min()), max(raw.max(), prediction.max()), 150)
    ax.hist(raw, bins=bins, density=True, alpha=0.55, color="#377eb8", label="dataset")
    ax.hist(prediction.ravel(), bins=bins, density=True, histtype="step", linewidth=1.7,
            color="#e41a1c", label="decoded policy")
    ax.axvline(threshold, color="black", linestyle="--", label=f"binary threshold = {threshold:.2f}")
    ax.set(title="Gripper value distribution", xlabel="continuous gripper value", ylabel="density")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    nonzero = np.abs(delta[np.abs(delta) > 0])
    ax.hist(nonzero, bins=np.geomspace(max(nonzero.min(), 1e-4), nonzero.max(), 70), color="#4daf4a")
    ax.set_xscale("log")
    for q, color in ((0.95, "#ff7f00"), (0.99, "#e41a1c")):
        value = np.quantile(np.abs(delta), q)
        ax.axvline(value, color=color, linestyle="--", label=f"all-delta q{int(q*100)}={value:.3f}")
    ax.set(title="Recorded frame-to-frame changes", xlabel="|gripper[t+1] - gripper[t]|", ylabel="count")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    example_episode = 0
    rows = archive["episode_index"] == example_episode
    order = np.argsort(archive["frame_index"][rows])
    frames = archive["frame_index"][rows][order]
    target_first = target[rows, 0][order]
    prediction_first = prediction[rows, 0][order]
    ax.plot(frames, target_first, color="#377eb8", linewidth=1.5, label="recorded continuous")
    ax.plot(frames, prediction_first, color="#e41a1c", linewidth=1.1, alpha=0.8, label="policy continuous")
    ax.step(frames, (prediction_first > threshold).astype(float), where="post", color="#4daf4a",
            linewidth=1.2, label="constrained command (0/1)")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8)
    ax.set(title="Episode 0: post-train projection makes transitions sharp", xlabel="frame", ylabel="gripper command")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    accuracy = 100.0 * np.mean(binary_prediction == binary_target, axis=0)
    ax.plot(np.arange(1, 21), accuracy, "o-", color="#984ea3", markersize=4)
    ax.set_ylim(min(98.5, accuracy.min() - 0.1), 100.05)
    ax.set_xticks(np.arange(1, 21))
    ax.set(title="Binary gripper-state accuracy by horizon", xlabel="prediction horizon step", ylabel="accuracy (%)")
    ax.grid(alpha=0.25)

    fig.suptitle("Piper gripper distribution and decoded-output constraint", fontsize=15)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"gripper_distribution_and_constraint.{suffix}", dpi=180)
    plt.close(fig)

    # Transition-level details explicitly show decoded splines, not connected controls.
    crossing = np.any(np.diff(binary_target.astype(np.int8), axis=1) != 0, axis=1)
    candidate = np.flatnonzero(crossing)
    exact = candidate[np.all(binary_prediction[candidate] == binary_target[candidate], axis=1)]
    transition_steps = np.asarray([
        np.flatnonzero(np.diff(binary_target[row].astype(np.int8)) != 0)[0] + 1
        for row in exact
    ])
    centered = exact[(transition_steps >= 5) & (transition_steps <= 14)]
    representative = int(centered[len(centered) // 2])
    worst = int(candidate[np.argmax(np.max(np.abs(prediction[candidate] - target[candidate]), axis=1))])
    steps = np.arange(20)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0), constrained_layout=True, sharey=True)
    for ax, row, label in (
        (axes[0], representative, "Representative correctly timed transition"),
        (axes[1], worst, "Worst long-horizon timing error"),
    ):
        ax.plot(steps, target[row], "o-", color="#377eb8", markersize=3, label="recorded target")
        ax.plot(steps, spline[row], color="#ff7f00", linewidth=2.0, label="decoded target B-spline")
        ax.plot(steps, prediction[row], color="#e41a1c", linewidth=2.0, label="decoded policy B-spline")
        ax.step(steps, binary_prediction[row].astype(float), where="mid", color="#4daf4a", linewidth=2.0,
                label="post-train constrained command")
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label=f"threshold {threshold:.2f}")
        ax.set(
            title=(f"{label}\nepisode {int(archive['episode_index'][row])}, "
                   f"observation frame {int(archive['frame_index'][row])}"),
            xlabel="horizon step at 25 Hz",
            ylabel="gripper command",
        )
        ax.set_xticks(steps)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Binary projection sharpens transitions but cannot repair a mistimed forecast", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"gripper_transition_bspline_and_constraint.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie strictly between 0 and 1")

    dataset = args.dataset.expanduser().resolve()
    controls_root = args.controls.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    errors_path = run_dir / "open_loop_errors_all.npz"

    episodes, raw, delta = _load_actions(dataset)
    controls = _load_controls(controls_root)
    archive, target, prediction, spline = _reconstruct_open_loop(errors_path, episodes)
    if controls.shape != (raw.size, 13):
        raise ValueError(f"Controls/raw alignment mismatch: {controls.shape} vs {raw.shape}")

    binary_target = target > args.threshold
    binary_prediction = prediction > args.threshold
    clipped = np.clip(prediction, raw.min(), raw.max())
    centers = _two_means(raw)
    mode_projected = np.where(prediction < centers.mean(), centers[0], centers[1])
    control_outside = (controls < raw.min()) | (controls > raw.max())
    report = {
        "dataset": str(dataset),
        "controls_dataset": str(controls_root),
        "open_loop_arrays": str(errors_path),
        "gripper_dimension": 6,
        "observations": int(raw.size),
        "episodes": int(len(episodes)),
        "raw_action_distribution": _summary(raw),
        "raw_frame_delta_distribution": _summary(delta),
        "raw_frame_delta_absolute_quantiles": {
            "q90": float(np.quantile(np.abs(delta), 0.90)),
            "q95": float(np.quantile(np.abs(delta), 0.95)),
            "q99": float(np.quantile(np.abs(delta), 0.99)),
            "q999": float(np.quantile(np.abs(delta), 0.999)),
            "max": float(np.max(np.abs(delta))),
        },
        "bspline_control_distribution": {
            "shape": list(controls.shape),
            "overall": _summary(controls),
            "per_control_point": [_summary(controls[:, index]) for index in range(13)],
            "outside_raw_action_range_count": int(control_outside.sum()),
            "outside_raw_action_range_fraction": float(control_outside.mean()),
            "outside_raw_action_range_fraction_per_control_point": control_outside.mean(axis=0).tolist(),
        },
        "decoded_policy_distribution": _summary(prediction),
        "continuous_open_loop_metrics": {
            "unconstrained": _error_summary(prediction, target),
            "clip_to_demonstrated_minmax": _error_summary(clipped, target),
            "hard_project_to_two_regression_modes": {
                "centers": centers.tolist(),
                **_error_summary(mode_projected, target),
            },
        },
        "recommended_binary_execution_constraint": {
            "threshold": float(args.threshold),
            "open_command": 0.0,
            "closed_command": 1.0,
            "definition": "closed iff decoded physical gripper output > threshold",
            "overall_state_accuracy": float(np.mean(binary_prediction == binary_target)),
            "first_action_state_accuracy": float(np.mean(binary_prediction[:, 0] == binary_target[:, 0])),
            "first_action_errors": int(np.sum(binary_prediction[:, 0] != binary_target[:, 0])),
            "per_horizon_state_accuracy": np.mean(binary_prediction == binary_target, axis=0).tolist(),
            "transition_timing_from_first_action_predictions": _transition_timing(
                archive, target, prediction, args.threshold
            ),
        },
        "interpretation": (
            "Control values outside the raw action range are expected B-spline coefficients, not robot commands. "
            "Do not connect or independently clip the control points. Decode the spline first, then apply the "
            "binary gripper projection at the robot boundary."
        ),
    }
    report_path = run_dir / "gripper_distribution_and_postprocess.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _plot_control_distribution(run_dir, raw, controls)
    _plot_dashboard(run_dir, archive, raw, delta, target, prediction, spline, args.threshold)
    archive.close()
    print(json.dumps({
        "report": str(report_path),
        "control_plot": str(run_dir / "gripper_control_point_distribution.png"),
        "dashboard": str(run_dir / "gripper_distribution_and_constraint.png"),
        "transition_plot": str(run_dir / "gripper_transition_bspline_and_constraint.png"),
        "first_action_state_accuracy": report["recommended_binary_execution_constraint"]["first_action_state_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
