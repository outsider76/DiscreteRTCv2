#!/usr/bin/env python3
"""Compare adaptive and uniform checkpoints on their common 10-step prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(np.abs(values))),
        "first_action_rmse": float(np.sqrt(np.mean(np.square(values[:, 0])))),
        "arm_rmse": float(np.sqrt(np.mean(np.square(values[..., :6])))),
        "gripper_rmse": float(np.sqrt(np.mean(np.square(values[..., 6])))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adaptive_arrays", type=Path)
    parser.add_argument("uniform_arrays", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.adaptive_arrays.expanduser().resolve()) as adaptive_file:
        adaptive = {key: adaptive_file[key] for key in adaptive_file.files}
    with np.load(args.uniform_arrays.expanduser().resolve()) as uniform_file:
        uniform = {key: uniform_file[key] for key in uniform_file.files}
    np.testing.assert_array_equal(adaptive["sample_index"], uniform["sample_index"])
    np.testing.assert_array_equal(adaptive["episode_index"], uniform["episode_index"])
    np.testing.assert_array_equal(adaptive["frame_index"], uniform["frame_index"])

    horizon = 10  # every positive-duration 13-row cubic plan guarantees ten native steps
    adaptive_model_error = adaptive["prediction_vs_original_error"][:, :horizon]
    adaptive_representation_error = adaptive["spline_vs_original_error"][:, :horizon]
    uniform_model_error = uniform["prediction_vs_original_error"][:, :horizon]
    uniform_representation_error = uniform["spline_vs_original_error"][:, :horizon]
    adaptive_model = _metrics(adaptive_model_error)
    uniform_model = _metrics(uniform_model_error)
    adaptive_representation = _metrics(adaptive_representation_error)
    uniform_representation = _metrics(uniform_representation_error)
    report = {
        "passed": True,
        "comparison": "same recorded observations and common native-rate prefix",
        "observations": int(len(adaptive["sample_index"])),
        "horizon_steps": int(horizon),
        "adaptive": {
            "model_vs_original": adaptive_model,
            "representation_vs_original": adaptive_representation,
        },
        "uniform": {
            "model_vs_original": uniform_model,
            "representation_vs_original": uniform_representation,
        },
        "adaptive_vs_uniform": {
            "model_arm_rmse_reduction_fraction": (
                1.0 - adaptive_model["arm_rmse"] / uniform_model["arm_rmse"]
            ),
            "model_overall_rmse_reduction_fraction": (
                1.0 - adaptive_model["rmse"] / uniform_model["rmse"]
            ),
            "representation_arm_rmse_reduction_fraction": (
                1.0
                - adaptive_representation["arm_rmse"]
                / uniform_representation["arm_rmse"]
            ),
        },
        "gripper_note": (
            "Continuous gripper error is reported separately because gripper dimension 6 is excluded "
            "from knot placement and is projected to binary commands during deployment."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), **report["adaptive_vs_uniform"]}, indent=2))


if __name__ == "__main__":
    main()
