#!/usr/bin/env python3
"""Visualize the Piper UniformBSpline open-loop evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from spline_encoder import UniformLeftBSplineConfig, create_encoder


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results/Checkpoints/piper_pick_white_block_20260818_qwenpi_25hz_uniform_left_bspline_s2_h20_c13"
)
DEFAULT_DATASET = REPO_ROOT / "data/20260818_piper_pick_white_block_25hz"
DEFAULT_CONTROLS_DATASET = (
    REPO_ROOT / "data/20260818_piper_pick_white_block_25hz_UniformBSpline"
)
ACTION_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]
MODEL_COLOR = "#2878B5"
BASELINE_COLOR = "#E07A2D"
SPLINE_COLOR = "#2E9B67"
GRID_COLOR = "#D8DEE9"


def _rmse_series(section: dict, key: str) -> np.ndarray:
    return np.asarray([item[key] for item in section], dtype=np.float64)


def _episode_actions(dataset_root: Path, episode_id: int) -> np.ndarray:
    matches = sorted(dataset_root.glob(f"data/*/episode_{episode_id:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet for episode {episode_id}, found {len(matches)} under {dataset_root}"
        )
    return np.stack(pd.read_parquet(matches[0])["action"].to_numpy()).astype(np.float32)


def _target_chunk(dataset_root: Path, episode_id: int, frame_id: int, horizon: int = 20) -> np.ndarray:
    actions = _episode_actions(dataset_root, episode_id)
    indices = np.minimum(frame_id + np.arange(horizon), len(actions) - 1)
    return actions[indices]


def _episode_controls(controls_root: Path, episode_id: int) -> np.ndarray:
    path = controls_root / "data" / f"episode_{episode_id:06d}.npz"
    with np.load(path) as episode:
        controls = np.asarray(episode["controls"], dtype=np.float64)
    if controls.ndim != 3 or controls.shape[1:] != (13, 7):
        raise ValueError(f"Unexpected control shape in {path}: {controls.shape}")
    return controls


def _load_encoder(controls_root: Path):
    config = json.loads((controls_root / "encoder.json").read_text())["config"]
    encoder = create_encoder(UniformLeftBSplineConfig(**config))
    if encoder.basis.shape != (20, 13):
        raise ValueError(f"Unexpected decode basis: {encoder.basis.shape}")
    return encoder


def _predicted_physical_controls(
    target_controls: np.ndarray,
    normalized_control_error: np.ndarray,
    report: dict,
) -> np.ndarray:
    """Recover predicted coefficients using the training-time affine normalization."""
    low = np.asarray(report["prediction_bounds"]["training_min"], dtype=np.float64)
    high = np.asarray(report["prediction_bounds"]["training_max"], dtype=np.float64)
    scale = high - low
    if np.any(scale == 0):
        raise ValueError("Cannot recover physical controls with a zero action range")
    target_normalized = 2.0 * (target_controls - low) / scale - 1.0
    predicted_normalized = target_normalized + normalized_control_error
    return 0.5 * (predicted_normalized + 1.0) * scale + low


def _style_axis(axis) -> None:
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def make_dashboard(report: dict, arrays: dict[str, np.ndarray], output: Path) -> None:
    model = report["physical_action_prediction_vs_original_target"]
    baseline = report["current_action_persistence_baseline_vs_original_target"]
    spline = report["spline_representation_vs_original_target"]
    horizons = np.arange(1, 21)

    figure = plt.figure(figsize=(16, 10), facecolor="#F7F9FC")
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.05, 1.0],
        left=0.055,
        right=0.985,
        bottom=0.07,
        top=0.87,
        hspace=0.30,
        wspace=0.27,
    )
    figure.suptitle(
        "Piper QwenPI_v3 UniformBSpline — Open-loop evaluation",
        fontsize=20,
        fontweight="bold",
        color="#17223B",
    )
    figure.text(
        0.5,
        0.925,
        f"{report['evaluated_observations']:,} recorded observations • {report['episodes']} episodes • "
        "20 actions at 25 Hz • training-set evaluation",
        ha="center",
        fontsize=10.5,
        color="#536078",
    )

    # Horizon error.
    axis = figure.add_subplot(grid[0, :2])
    model_h = _rmse_series(model["per_horizon"], "rmse")
    baseline_h = _rmse_series(baseline["per_horizon"], "rmse")
    spline_h = _rmse_series(spline["per_horizon"], "rmse")
    axis.plot(horizons, model_h, marker="o", markersize=4, linewidth=2.4, color=MODEL_COLOR, label="Policy")
    axis.plot(
        horizons,
        baseline_h,
        marker="o",
        markersize=3,
        linewidth=2.0,
        color=BASELINE_COLOR,
        label="Repeat-current-action baseline",
    )
    axis.set(title="Error over the prediction horizon", xlabel="Future action step", ylabel="Physical-action RMSE")
    axis.set_xticks([1, 4, 8, 12, 16, 20])
    axis.set_xlim(1, 20)
    _style_axis(axis)
    spline_axis = axis.twinx()
    spline_axis.plot(
        horizons,
        spline_h,
        linestyle="--",
        linewidth=1.8,
        color=SPLINE_COLOR,
        label="Spline reconstruction",
    )
    spline_axis.set_ylabel("Spline-only RMSE", color=SPLINE_COLOR)
    spline_axis.tick_params(axis="y", colors=SPLINE_COLOR)
    spline_axis.spines["top"].set_visible(False)
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = spline_axis.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, loc="upper left")

    # KPI card.
    axis = figure.add_subplot(grid[0, 2])
    axis.axis("off")
    comparison = report["comparison_to_persistence_baseline"]
    bounds = report["prediction_bounds"]
    kpis = [
        ("Policy RMSE", f"{model['overall']['rmse']:.4f}"),
        ("Policy MAE", f"{model['overall']['mae']:.4f}"),
        ("MSE reduction vs baseline", f"{100 * comparison['mse_reduction_fraction']:.1f}%"),
        ("Spline reconstruction RMSE", f"{spline['overall']['rmse']:.5f}"),
        ("Outside training bounds", f"{100 * bounds['outside_training_minmax_fraction']:.2f}%"),
    ]
    y = 0.92
    for label, value in kpis:
        axis.text(0.03, y, label, fontsize=10, color="#647089", transform=axis.transAxes)
        axis.text(
            0.03,
            y - 0.085,
            value,
            fontsize=20,
            fontweight="bold",
            color="#17223B",
            transform=axis.transAxes,
        )
        y -= 0.19
    axis.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=axis.transAxes, fill=False, edgecolor=GRID_COLOR, linewidth=1.2)
    )

    # Per-dimension RMSE.
    axis = figure.add_subplot(grid[1, 0])
    x = np.arange(len(ACTION_NAMES))
    model_dim = np.asarray([model["per_dimension"][name.lower().replace("j", "joint_") if name != "Gripper" else "gripper"]["rmse"] for name in ACTION_NAMES])
    baseline_dim = np.asarray([baseline["per_dimension"][name.lower().replace("j", "joint_") if name != "Gripper" else "gripper"]["rmse"] for name in ACTION_NAMES])
    width = 0.37
    axis.bar(x - width / 2, model_dim, width, color=MODEL_COLOR, label="Policy")
    axis.bar(x + width / 2, baseline_dim, width, color=BASELINE_COLOR, label="Baseline")
    axis.set(title="RMSE by action dimension", ylabel="Physical-action RMSE")
    axis.set_xticks(x, ACTION_NAMES)
    axis.legend(frameon=False)
    _style_axis(axis)

    # Error quantiles.
    axis = figure.add_subplot(grid[1, 1])
    error = np.abs(arrays["prediction_vs_original_error"])
    quantiles = np.percentile(error, [50, 95, 99], axis=(0, 1))
    for row, (label, color, marker) in enumerate(
        [("Median", "#7A89A8", "o"), ("P95", MODEL_COLOR, "s"), ("P99", "#9C5FB5", "^")]
    ):
        axis.plot(x, quantiles[row], marker=marker, linewidth=2, color=color, label=label)
    axis.set(title="Absolute-error quantiles", ylabel="Absolute physical-action error")
    axis.set_xticks(x, ACTION_NAMES)
    axis.set_yscale("log")
    axis.legend(frameon=False)
    _style_axis(axis)

    # Bounds exceedance.
    axis = figure.add_subplot(grid[1, 2])
    bound_fraction = np.asarray(
        [bounds["per_dimension_fraction"][name.lower().replace("j", "joint_") if name != "Gripper" else "gripper"] for name in ACTION_NAMES]
    )
    bars = axis.bar(x, 100 * bound_fraction, color=[MODEL_COLOR] * 6 + ["#C84B55"])
    axis.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in bound_fraction], padding=3, fontsize=8)
    axis.set(title="Predictions outside training range", ylabel="Values outside min/max (%)")
    axis.set_xticks(x, ACTION_NAMES)
    axis.set_ylim(0, max(6.0, 100 * bound_fraction.max() * 1.2))
    _style_axis(axis)

    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def make_examples(
    report: dict,
    arrays: dict[str, np.ndarray],
    dataset_root: Path,
    output: Path,
) -> None:
    error = arrays["prediction_vs_original_error"]
    per_sample_rmse = np.sqrt(np.mean(np.square(error), axis=(1, 2)))
    representative_row = int(np.argmin(np.abs(per_sample_rmse - np.median(per_sample_rmse))))
    worst_row = int(np.argmax(per_sample_rmse))
    rows = [representative_row, worst_row]
    titles = ["Representative observation", "Highest per-observation RMSE"]

    figure, axes = plt.subplots(
        7,
        2,
        figsize=(14, 15),
        sharex=True,
        facecolor="#F7F9FC",
    )
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.045, top=0.88, hspace=0.08, wspace=0.20)
    figure.suptitle(
        "Example open-loop action chunks",
        fontsize=19,
        fontweight="bold",
        color="#17223B",
        y=0.985,
    )
    steps = np.arange(1, 21)
    for column, (row, title) in enumerate(zip(rows, titles)):
        episode_id = int(arrays["episode_index"][row])
        frame_id = int(arrays["frame_index"][row])
        original = _target_chunk(dataset_root, episode_id, frame_id)
        predicted = original + arrays["prediction_vs_original_error"][row]
        spline_target = original + arrays["spline_vs_original_error"][row]
        axes[0, column].set_title(
            f"{title}\nepisode {episode_id}, frame {frame_id}, RMSE {per_sample_rmse[row]:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        for dim, name in enumerate(ACTION_NAMES):
            axis = axes[dim, column]
            axis.plot(steps, original[:, dim], color="#1E2535", linewidth=2.2, label="Original target")
            axis.plot(steps, predicted[:, dim], color=MODEL_COLOR, linewidth=1.9, label="Policy prediction")
            axis.plot(steps, spline_target[:, dim], color=SPLINE_COLOR, linestyle="--", linewidth=1.3, label="Spline target")
            axis.set_ylabel(f"{name}\n{'rad' if dim < 6 else 'data units'}")
            axis.grid(True, color=GRID_COLOR, linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            if dim == 6:
                axis.set_xlabel("Future action step")
                axis.set_xticks([1, 4, 8, 12, 16, 20])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def make_bspline_control_figure(
    report: dict,
    arrays: dict[str, np.ndarray],
    dataset_root: Path,
    controls_root: Path,
    encoder,
    output: Path,
) -> None:
    """Plot coefficients and the actual dense cubic B-spline they define.

    Coefficients are placed at their Greville abscissae for visualization only.
    The solid curves are evaluated with SciPy's B-spline implementation; no
    line interpolation through coefficient markers is used.
    """
    error = arrays["prediction_vs_original_error"]
    per_sample_rmse = np.sqrt(np.mean(np.square(error), axis=(1, 2)))
    representative_row = int(np.argmin(np.abs(per_sample_rmse - np.median(per_sample_rmse))))
    worst_row = int(np.argmax(per_sample_rmse))
    rows = [representative_row, worst_row]
    titles = ["Representative observation", "Highest per-observation RMSE"]

    knots = np.asarray(encoder.knots, dtype=np.float64)
    greville = np.asarray(encoder.geometry().greville_steps, dtype=np.float64)
    degree = int(encoder.config.degree)
    execution_end = float(encoder.config.chunk_size)
    dense_steps = np.linspace(0.0, np.nextafter(execution_end, 0.0), 801)
    sample_steps = np.arange(encoder.config.chunk_size, dtype=np.float64)
    distinct_knots = np.unique(knots)

    figure, axes = plt.subplots(7, 2, figsize=(16, 16), sharex=True, facecolor="#F7F9FC")
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=0.875, hspace=0.10, wspace=0.16)
    figure.suptitle(
        "Uniform-left cubic B-spline coefficients and decoded action curves",
        fontsize=19,
        fontweight="bold",
        color="#17223B",
        y=0.988,
    )
    figure.text(
        0.5,
        0.958,
        "Solid curves are dense BSpline evaluations; diamonds/crosses are coefficients at Greville abscissae, not interpolated waypoints",
        ha="center",
        fontsize=10.5,
        color="#536078",
    )

    controls_cache: dict[int, np.ndarray] = {}
    for column, (row, title) in enumerate(zip(rows, titles)):
        episode_id = int(arrays["episode_index"][row])
        frame_id = int(arrays["frame_index"][row])
        if episode_id not in controls_cache:
            controls_cache[episode_id] = _episode_controls(controls_root, episode_id)
        target_controls = controls_cache[episode_id][frame_id]
        predicted_controls = _predicted_physical_controls(
            target_controls,
            np.asarray(arrays["control_error"][row], dtype=np.float64),
            report,
        )
        original = _target_chunk(dataset_root, episode_id, frame_id)
        target_spline = BSpline(knots, target_controls, degree, extrapolate=False, axis=0)
        predicted_spline = BSpline(knots, predicted_controls, degree, extrapolate=False, axis=0)
        target_dense = target_spline(dense_steps)
        predicted_dense = predicted_spline(dense_steps)

        # This also guards the visualization against accidentally plotting a
        # polyline or using a knot/control convention different from the model.
        target_at_samples = target_spline(sample_steps)
        predicted_at_samples = predicted_spline(sample_steps)
        model_target = original + arrays["spline_vs_original_error"][row]
        model_prediction = original + arrays["prediction_vs_original_error"][row]
        if not np.allclose(target_at_samples, model_target, atol=5e-4, rtol=0):
            raise ValueError("Direct B-spline decode does not match the evaluated spline target")
        if not np.allclose(predicted_at_samples, model_prediction, atol=5e-4, rtol=0):
            raise ValueError("Direct B-spline decode does not match the evaluated policy prediction")

        axes[0, column].set_title(
            f"{title}\nepisode {episode_id}, frame {frame_id}, policy RMSE {per_sample_rmse[row]:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        for dim, name in enumerate(ACTION_NAMES):
            axis = axes[dim, column]
            for knot in distinct_knots:
                axis.axvline(knot, color="#C9D1DF", linewidth=0.55, alpha=0.6, zorder=0)
            axis.axvspan(execution_end, distinct_knots[-1], color="#E8ECF3", alpha=0.75, zorder=0)
            axis.axvline(execution_end, color="#7A89A8", linestyle="--", linewidth=1.0, zorder=1)
            axis.plot(
                dense_steps,
                target_dense[:, dim],
                color=SPLINE_COLOR,
                linewidth=2.2,
                label="Target B-spline",
                zorder=3,
            )
            axis.plot(
                dense_steps,
                predicted_dense[:, dim],
                color=MODEL_COLOR,
                linewidth=2.0,
                label="Policy B-spline",
                zorder=3,
            )
            axis.scatter(
                sample_steps,
                original[:, dim],
                s=13,
                facecolors="none",
                edgecolors="#1E2535",
                linewidths=0.8,
                label="Original action samples",
                zorder=4,
            )
            axis.scatter(
                greville,
                target_controls[:, dim],
                marker="D",
                s=24,
                color=SPLINE_COLOR,
                edgecolors="white",
                linewidths=0.5,
                label="Target coefficients",
                zorder=5,
            )
            axis.scatter(
                greville,
                predicted_controls[:, dim],
                marker="x",
                s=30,
                color=MODEL_COLOR,
                linewidths=1.2,
                label="Policy coefficients",
                zorder=5,
            )
            axis.set_ylabel(f"{name}\n{'rad' if dim < 6 else 'data units'}")
            axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            if dim == 0:
                axis.text(
                    execution_end + 0.35,
                    0.92,
                    "right-open\nsupport",
                    transform=axis.get_xaxis_transform(),
                    fontsize=7.5,
                    color="#647089",
                    va="top",
                )
            if dim == 6:
                axis.set_xlabel("Action-step coordinate (knot span = 2)")
                axis.set_xticks(distinct_knots)
                axis.tick_params(axis="x", labelsize=8)
            axis.set_xlim(-0.25, distinct_knots[-1] + 0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=5, frameon=False)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--controls-dataset", type=Path, default=DEFAULT_CONTROLS_DATASET)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    dataset_root = args.dataset.expanduser().resolve()
    controls_root = args.controls_dataset.expanduser().resolve()
    report = json.loads((run_dir / "open_loop_evaluation_all.json").read_text())
    with np.load(run_dir / "open_loop_errors_all.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}

    dashboard = run_dir / "open_loop_dashboard.png"
    examples = run_dir / "open_loop_examples.png"
    controls = run_dir / "open_loop_bspline_controls.png"
    make_dashboard(report, arrays, dashboard)
    make_examples(report, arrays, dataset_root, examples)
    encoder = _load_encoder(controls_root)
    make_bspline_control_figure(report, arrays, dataset_root, controls_root, encoder, controls)
    print(dashboard)
    print(examples)
    print(controls)


if __name__ == "__main__":
    main()
