#!/usr/bin/env python3
"""Native-variable-horizon open-loop evaluation for adaptive-left B-splines."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = REPO_ROOT / (
    "results/Checkpoints/piper_pick_white_block_20260818_qwenpi_25hz_"
    "adaptive_left_bspline_excl_gripper_native_c13_gb32_50k"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints/steps_50000_pytorch_model.pt"
DEFAULT_CONFIG = REPO_ROOT / (
    "examples/realRobots/Piper/train_files/"
    "starvla_qwenpi_piper_20260818_25hz_adaptive_left_bspline_gb32_50k.yaml"
)
ACTION_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
PARAMETER_NAMES = ACTION_NAMES + ["duration"]


def _collate_samples(samples):
    return samples


def _metric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mse": float(np.mean(np.square(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "max_absolute_error": float(np.max(absolute)),
    }


def _trajectory_metrics(error: np.ndarray, valid: np.ndarray) -> dict:
    values = np.asarray(error)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape[:2] or not mask.any():
        raise ValueError(f"invalid trajectory/mask shapes: {values.shape}, {mask.shape}")
    per_horizon = []
    for step in range(values.shape[1]):
        step_values = values[mask[:, step], step]
        per_horizon.append(None if not len(step_values) else _metric_summary(step_values))
    return {
        "overall": _metric_summary(values[mask]),
        "arm": _metric_summary(values[..., :6][mask]),
        "gripper": _metric_summary(values[..., 6][mask]),
        "per_horizon": per_horizon,
        "per_dimension": {
            name: _metric_summary(values[..., dim][mask])
            for dim, name in enumerate(ACTION_NAMES)
        },
        "first_action": _metric_summary(values[:, 0]),
        "valid_plan_steps": int(mask.sum()),
    }


def _parameter_metrics(error: np.ndarray) -> dict:
    return {
        "overall": _metric_summary(error),
        "controls": _metric_summary(error[..., :7]),
        "duration": _metric_summary(error[..., 7]),
        "per_row": [_metric_summary(error[:, row]) for row in range(error.shape[1])],
        "per_dimension": {
            name: _metric_summary(error[..., dim])
            for dim, name in enumerate(PARAMETER_NAMES)
        },
    }


def _load_raw_episode_actions(dataset) -> dict[int, np.ndarray]:
    result = {}
    for trajectory_id, trajectory_length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        episode_id = int(trajectory_id)
        frame = dataset.get_trajectory_data(episode_id)
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        if actions.shape != (int(trajectory_length), 7) or not np.isfinite(actions).all():
            raise ValueError(f"Episode {episode_id} action mismatch: {actions.shape}")
        result[episode_id] = actions
    return result


def _raw_action_chunks(dataset, raw_actions, sample_indices, horizon):
    episode_ids = np.empty(len(sample_indices), dtype=np.int64)
    frame_ids = np.empty(len(sample_indices), dtype=np.int64)
    chunks = np.empty((len(sample_indices), horizon, 7), dtype=np.float32)
    available = np.zeros((len(sample_indices), horizon), dtype=bool)
    for row, sample_index in enumerate(sample_indices):
        episode_id, frame_id = map(int, dataset.all_steps[int(sample_index)])
        trajectory = raw_actions[episode_id]
        offsets = np.arange(horizon)
        available[row] = offsets < len(trajectory) - frame_id
        chunks[row] = trajectory[np.minimum(frame_id + offsets, len(trajectory) - 1)]
        episode_ids[row] = episode_id
        frame_ids[row] = frame_id
    return chunks, available, episode_ids, frame_ids


def _decode(model, parameters, output_horizon):
    from starVLA.model.framework.VLM4A.QwenPI_v3_AdaptiveBSpline import decode_adaptive_parameters

    return decode_adaptive_parameters(
        parameters,
        action_dim=model.spline_action_dim,
        degree=model.spline_degree,
        duration_min=model.duration_min,
        duration_max=model.duration_max,
        output_horizon=output_horizon,
    )


def _horizon_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.int64)
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from deployment.model_server.action_constraints import GripperConstraint
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg = OmegaConf.load(config_path)
    representation = cfg.framework.action_representation
    if representation.execution_horizon_mode != "predicted_duration_sum":
        raise ValueError("evaluation requires native variable-length execution")
    horizon_cap = int(representation.max_native_horizon)
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode="val").datasets[0]
    total_observations = len(dataset)
    if args.max_samples is None or args.max_samples >= total_observations:
        sample_indices = np.arange(total_observations, dtype=np.int64)
    elif args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    else:
        sample_indices = np.unique(
            np.linspace(0, total_observations - 1, args.max_samples, dtype=np.int64)
        )

    raw_actions = _load_raw_episode_actions(dataset)
    raw_targets, available, episode_ids, frame_ids = _raw_action_chunks(
        dataset, raw_actions, sample_indices, horizon_cap
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": _collate_samples,
    }
    if args.num_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(Subset(dataset, sample_indices.tolist()), **loader_kwargs)

    policy = PolicyServerWrapper(
        ckpt_path=str(checkpoint), device="cuda", use_bf16=True, unnorm_key="new_embodiment"
    )
    model = policy._framework
    processor = policy.get_norm_processor()
    constraint = GripperConstraint(
        mode="binary", dimension=6, minimum=0.0, maximum=1.0, threshold=0.3
    )

    collected = {key: [] for key in (
        "predicted_parameters", "target_parameters", "predicted_durations",
        "target_durations", "predicted_horizons", "target_horizons",
        "predicted_actions", "target_actions",
    )}
    inference_seconds = []
    for batch_number, samples in enumerate(loader, start=1):
        requests = [{key: sample[key] for key in ("image", "lang", "state")} for sample in samples]
        target_parameters = np.asarray([sample["action"] for sample in samples], dtype=np.float32)
        start = time.perf_counter()
        predicted_parameters = np.asarray(
            model.predict_action_parameters(examples=requests)["normalized_actions"], dtype=np.float32
        )
        inference_seconds.append(time.perf_counter() - start)
        predicted_normalized, predicted_durations, predicted_horizons = _decode(
            model, predicted_parameters, horizon_cap
        )
        target_normalized, target_durations, target_horizons = _decode(
            model, target_parameters, horizon_cap
        )
        predicted_actions = np.stack([
            processor.unapply_actions(value) for value in predicted_normalized
        ]).astype(np.float32)
        target_actions = np.stack([
            processor.unapply_actions(value) for value in target_normalized
        ]).astype(np.float32)
        for key, value in (
            ("predicted_parameters", predicted_parameters),
            ("target_parameters", target_parameters),
            ("predicted_durations", predicted_durations),
            ("target_durations", target_durations),
            ("predicted_horizons", predicted_horizons),
            ("target_horizons", target_horizons),
            ("predicted_actions", predicted_actions),
            ("target_actions", target_actions),
        ):
            collected[key].append(value)
        if batch_number % 100 == 0 or batch_number == len(loader):
            done = min(batch_number * args.batch_size, len(sample_indices))
            print(f"evaluated {done}/{len(sample_indices)} observations", flush=True)

    values = {key: np.concatenate(items) for key, items in collected.items()}
    predicted_effective = np.minimum(values["predicted_horizons"], horizon_cap)
    target_effective = np.minimum(values["target_horizons"], horizon_cap)
    steps = np.arange(horizon_cap)[None]
    predicted_valid = steps < predicted_effective[:, None]
    target_valid = steps < target_effective[:, None]
    prediction_raw_valid = predicted_valid & available
    target_raw_valid = target_valid & available
    overlap_valid = predicted_valid & target_valid & available

    parameter_error = values["predicted_parameters"] - values["target_parameters"]
    duration_error = values["predicted_durations"] - values["target_durations"]
    prediction_vs_raw = values["predicted_actions"] - raw_targets
    representation_vs_raw = values["target_actions"] - raw_targets
    prediction_vs_representation = values["predicted_actions"] - values["target_actions"]
    persistence = np.repeat(raw_targets[:, :1], horizon_cap, axis=1)
    constrained_prediction = constraint.apply(values["predicted_actions"])
    constrained_target = constraint.apply(raw_targets)
    timing = np.asarray(inference_seconds, dtype=np.float64)

    report = {
        "evaluation": "open_loop_recorded_observations_native_variable_horizon",
        "definition": (
            "Each prediction uses a recorded observation; plans are sampled at native 25 Hz and "
            "masked at sum(first 10 predicted durations), never resampled to a fixed action count."
        ),
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "dataset_observations": total_observations,
        "evaluated_observations": int(len(sample_indices)),
        "episodes": int(len(np.unique(episode_ids))),
        "action_frequency_hz": 25.0,
        "native_horizon_safety_cap": horizon_cap,
        "parameter_shape_per_observation": [13, 8],
        "parameter_layout": "[control_dim_0 ... control_dim_6, duration]",
        "normalized_parameter_prediction": _parameter_metrics(parameter_error),
        "integer_duration_prediction": {
            **_metric_summary(duration_error),
            "exact_accuracy": float(np.mean(duration_error == 0)),
            "predicted_native_horizon": _horizon_summary(values["predicted_horizons"]),
            "target_native_horizon": _horizon_summary(values["target_horizons"]),
            "horizon_mae_steps": float(np.mean(np.abs(
                values["predicted_horizons"] - values["target_horizons"]
            ))),
            "predicted_truncation_fraction": float(np.mean(
                values["predicted_horizons"] > horizon_cap
            )),
            "target_truncation_fraction": float(np.mean(
                values["target_horizons"] > horizon_cap
            )),
        },
        "physical_action_prediction_vs_original_target": _trajectory_metrics(
            prediction_vs_raw, prediction_raw_valid
        ),
        "physical_action_prediction_vs_spline_target": _trajectory_metrics(
            prediction_vs_representation, overlap_valid
        ),
        "spline_representation_vs_original_target": _trajectory_metrics(
            representation_vs_raw, target_raw_valid
        ),
        "current_action_persistence_baseline_vs_original_target": _trajectory_metrics(
            persistence - raw_targets, prediction_raw_valid
        ),
        "binary_gripper_postprocess_vs_binary_target": _trajectory_metrics(
            constrained_prediction - constrained_target, prediction_raw_valid
        ),
        "availability": {
            "unavailable_tail_fraction_of_target_plan": float(np.mean(
                target_valid & ~available
            )),
            "predicted_valid_steps": int(predicted_valid.sum()),
            "target_valid_steps": int(target_valid.sum()),
            "overlap_available_steps": int(overlap_valid.sum()),
        },
        "inference_timing": {
            "batches": int(len(timing)),
            "total_seconds": float(timing.sum()),
            "mean_batch_seconds": float(timing.mean()),
            "mean_observation_seconds_amortized": float(timing.sum() / len(sample_indices)),
        },
    }
    suffix = "all" if len(sample_indices) == total_observations else f"subset_{len(sample_indices)}"
    report_path = output_dir / f"adaptive_open_loop_evaluation_{suffix}.json"
    arrays_path = output_dir / f"adaptive_open_loop_errors_{suffix}.npz"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        arrays_path,
        sample_index=sample_indices,
        episode_index=episode_ids,
        frame_index=frame_ids,
        parameter_error=parameter_error.astype(np.float32),
        predicted_durations=values["predicted_durations"].astype(np.int16),
        target_durations=values["target_durations"].astype(np.int16),
        duration_error=duration_error.astype(np.int16),
        predicted_horizons=values["predicted_horizons"].astype(np.int16),
        target_horizons=values["target_horizons"].astype(np.int16),
        predicted_valid=predicted_valid,
        target_valid=target_valid,
        available=available,
        prediction_vs_spline_error=prediction_vs_representation.astype(np.float32),
        prediction_vs_original_error=prediction_vs_raw.astype(np.float32),
        spline_vs_original_error=representation_vs_raw.astype(np.float32),
        persistence_vs_original_error=(persistence - raw_targets).astype(np.float32),
        constrained_prediction_vs_target_error=(
            constrained_prediction - constrained_target
        ).astype(np.float32),
    )
    print(json.dumps({
        "report": str(report_path),
        "arrays": str(arrays_path),
        "evaluated_observations": int(len(sample_indices)),
        "parameter_rmse": report["normalized_parameter_prediction"]["overall"]["rmse"],
        "decoded_vs_original_rmse": report["physical_action_prediction_vs_original_target"]["overall"]["rmse"],
        "decoded_arm_rmse": report["physical_action_prediction_vs_original_target"]["arm"]["rmse"],
        "representation_arm_rmse": report["spline_representation_vs_original_target"]["arm"]["rmse"],
        "predicted_horizon": report["integer_duration_prediction"]["predicted_native_horizon"],
    }, indent=2))


if __name__ == "__main__":
    main()
