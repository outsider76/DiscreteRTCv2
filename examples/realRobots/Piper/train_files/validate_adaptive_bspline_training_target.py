#!/usr/bin/env python3
"""Validate the exact 13x8 adaptive training target and deployment decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from starVLA.model.framework.VLM4A.QwenPI_v3_AdaptiveBSpline import (
    decode_adaptive_parameters,
)


def validate(source: Path, encoded: Path) -> dict:
    stats_payload = json.loads((source / "meta" / "stats_gr00t.json").read_text())
    action_stats = stats_payload["statistics"]["action"]
    action_min = np.asarray(action_stats["min"], dtype=np.float64)
    action_max = np.asarray(action_stats["max"], dtype=np.float64)
    scale = action_max - action_min
    if action_min.shape != (7,) or np.any(scale <= 0):
        raise ValueError("expected non-degenerate 7-D Piper action statistics")

    manifest = json.loads((encoded / "manifest.json").read_text())
    encoder = json.loads((encoded / "encoder.json").read_text())
    config = encoder["config"]
    if manifest.get("policy_layout") != "control_point_duration_interleaved_v1":
        raise ValueError("unexpected policy layout")
    if config.get("knot_insertion_excluded_dimensions") != [6]:
        raise ValueError("gripper dimension 6 is not excluded")

    max_parameter_round_trip_error = 0.0
    max_decode_error = 0.0
    observations = 0
    native_horizons_all = []
    duration_min, duration_max = 1, 21
    for path in sorted((encoded / "data").glob("episode_*.npz")):
        with np.load(path) as episode:
            interleaved = np.asarray(
                episode["control_duration_interleaved"], dtype=np.float64
            )
            controls = np.asarray(episode["controls"], dtype=np.float64)
            durations = np.asarray(episode["duration_steps"], dtype=np.int64)
            knots = np.asarray(episode["knots"], dtype=np.float64)
        if interleaved.shape[1:] != (13, 8):
            raise ValueError(f"unexpected target shape in {path}: {interleaved.shape}")
        np.testing.assert_array_equal(interleaved[..., :7], controls)
        np.testing.assert_array_equal(interleaved[..., 7], durations)

        normalized_controls = 2.0 * (controls - action_min) / scale - 1.0
        normalized_durations = (
            2.0 * (durations - duration_min) / (duration_max - duration_min) - 1.0
        )
        parameters = np.concatenate(
            (normalized_controls, normalized_durations[..., None]), axis=2
        ).astype(np.float32)
        decoded_normalized, decoded_durations, native_horizons = decode_adaptive_parameters(
            parameters,
            action_dim=7,
            degree=3,
            duration_min=duration_min,
            duration_max=duration_max,
            output_horizon=64,
        )
        np.testing.assert_array_equal(decoded_durations, durations)
        np.testing.assert_array_equal(native_horizons, durations[:, :10].sum(axis=1))
        decoded_physical = (decoded_normalized.astype(np.float64) + 1.0) * 0.5 * scale + action_min
        for row, (knot, control, native_horizon) in enumerate(
            zip(knots, controls, native_horizons)
        ):
            expected = BSpline(knot, control, 3, extrapolate=False)(
                np.arange(native_horizon) / 25.0
            )
            max_decode_error = max(
                max_decode_error,
                float(np.max(np.abs(decoded_physical[row, :native_horizon] - expected))),
            )
        native_horizons_all.append(native_horizons)

        reconstructed_controls = (parameters[..., :7].astype(np.float64) + 1.0) * 0.5 * scale + action_min
        reconstructed_durations = (
            (parameters[..., 7].astype(np.float64) + 1.0)
            * 0.5
            * (duration_max - duration_min)
            + duration_min
        )
        max_parameter_round_trip_error = max(
            max_parameter_round_trip_error,
            float(np.max(np.abs(reconstructed_controls - controls))),
            float(np.max(np.abs(reconstructed_durations - durations))),
        )
        observations += len(controls)

    if observations != int(manifest["record_count"]):
        raise ValueError("observation count differs from manifest")
    if max_decode_error > 2e-6 or max_parameter_round_trip_error > 2e-6:
        raise ValueError("training target or decode round trip exceeded tolerance")
    native_horizons = np.concatenate(native_horizons_all)
    return {
        "passed": True,
        "observations_checked": observations,
        "target_shape_per_observation": [13, 8],
        "row_layout": "[control_dim_0 ... control_dim_6, duration_steps]",
        "duration_normalization_bounds": [duration_min, duration_max],
        "execution_horizon_mode": "sum_first_10_durations",
        "native_horizon_steps": {
            "min": int(native_horizons.min()),
            "median": float(np.median(native_horizons)),
            "mean": float(native_horizons.mean()),
            "max": int(native_horizons.max()),
        },
        "all_duration_round_trips_exact": True,
        "max_parameter_round_trip_absolute_error": max_parameter_round_trip_error,
        "max_normalized_decode_to_physical_absolute_error": max_decode_error,
        "action_statistics_remain_physical_7d": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("encoded", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    encoded = args.encoded.expanduser().resolve()
    report = validate(source, encoded)
    output = args.output or encoded / "training_target_validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
