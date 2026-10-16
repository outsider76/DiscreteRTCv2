#!/usr/bin/env python3
"""Visualize adaptive-vs-uniform open-loop errors on the common prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACTION_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]


def _rmse(values, axis):
    return np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)), axis=axis))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adaptive_arrays", type=Path)
    parser.add_argument("uniform_arrays", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.adaptive_arrays.expanduser().resolve()) as handle:
        adaptive = {key: handle[key] for key in handle.files}
    with np.load(args.uniform_arrays.expanduser().resolve()) as handle:
        uniform = {key: handle[key] for key in handle.files}
    np.testing.assert_array_equal(adaptive["sample_index"], uniform["sample_index"])
    horizon = 10
    uniform_model = uniform["prediction_vs_original_error"][:, :horizon]
    uniform_representation = uniform["spline_vs_original_error"][:, :horizon]
    adaptive_model = adaptive["prediction_vs_original_error"][:, :horizon]
    adaptive_representation = adaptive["spline_vs_original_error"][:, :horizon]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    steps = np.arange(horizon)
    axes[0, 0].plot(steps, _rmse(adaptive_model[..., :6], axis=(0, 2)), marker="o", label="Adaptive")
    axes[0, 0].plot(steps, _rmse(uniform_model[..., :6], axis=(0, 2)), marker="o", label="Uniform-left")
    axes[0, 0].set(title="Policy arm RMSE by horizon", xlabel="25 Hz action step", ylabel="RMSE")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(steps, _rmse(adaptive_representation[..., :6], axis=(0, 2)), marker="o", label="Adaptive")
    axes[0, 1].plot(steps, _rmse(uniform_representation[..., :6], axis=(0, 2)), marker="o", label="Uniform-left")
    axes[0, 1].set(title="Representation-only arm RMSE", xlabel="25 Hz action step", ylabel="RMSE")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    width = 0.36
    dims = np.arange(7)
    axes[1, 0].bar(dims - width / 2, _rmse(adaptive_model, axis=(0, 1)), width, label="Adaptive")
    axes[1, 0].bar(dims + width / 2, _rmse(uniform_model, axis=(0, 1)), width, label="Uniform-left")
    axes[1, 0].set_xticks(dims, ACTION_NAMES)
    axes[1, 0].set(title="Policy RMSE per action dimension", ylabel="RMSE")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    duration_error = adaptive["duration_error"].reshape(-1)
    bins = np.arange(duration_error.min() - 0.5, duration_error.max() + 1.5)
    axes[1, 1].hist(duration_error, bins=bins, density=True, color="#5b8ff9", alpha=0.85)
    axes[1, 1].set(
        title="Adaptive predicted knot-duration error",
        xlabel="Predicted − target duration (25 Hz steps)",
        ylabel="Fraction",
    )
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Piper QwenPI_v3: adaptive-left vs uniform-left ({len(adaptive_model):,} observations)",
        fontsize=15,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    print(output)
    print(output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
