#!/usr/bin/env python3
"""Evaluate a Piper UniformBSpline policy open-loop on recorded observations.

Every observation is evaluated independently from the recorded state and two
camera views. Predicted actions are never fed back as later observations. The
report separates model error from the B-spline representation's reconstruction
error against the original 20-step physical-action target.
"""

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
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results/Checkpoints/piper_pick_white_block_20260818_qwenpi_25hz_uniform_left_bspline_s2_h20_c13"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints/steps_10000_pytorch_model.pt"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "examples/realRobots/Piper/train_files/starvla_qwenpi_piper_20260818_25hz_uniform_left_bspline.yaml"
)
ACTION_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


def _collate_samples(samples):
    return samples


def _metric_summary(error: np.ndarray) -> dict:
    values = np.asarray(error, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "mse": float(np.mean(np.square(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "max_absolute_error": float(np.max(absolute)),
    }


def _trajectory_metrics(error: np.ndarray) -> dict:
    if error.ndim != 3:
        raise ValueError(f"Expected [N, T, D] trajectory error, got {error.shape}")
    return {
        "overall": _metric_summary(error),
        "per_horizon": [_metric_summary(error[:, step, :]) for step in range(error.shape[1])],
        "per_dimension": {
            name: _metric_summary(error[:, :, dim]) for dim, name in enumerate(ACTION_NAMES)
        },
        "first_action": _metric_summary(error[:, 0, :]),
    }


def _control_metrics(error: np.ndarray) -> dict:
    if error.ndim != 3:
        raise ValueError(f"Expected [N, C, D] control error, got {error.shape}")
    return {
        "overall": _metric_summary(error),
        "per_control_point": [_metric_summary(error[:, point, :]) for point in range(error.shape[1])],
        "per_dimension": {
            name: _metric_summary(error[:, :, dim]) for dim, name in enumerate(ACTION_NAMES)
        },
    }


def _load_raw_episode_actions(dataset) -> dict[int, np.ndarray]:
    result = {}
    for trajectory_id, trajectory_length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        episode_id = int(trajectory_id)
        frame = dataset.get_trajectory_data(episode_id)
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        expected = (int(trajectory_length), 7)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Episode {episode_id} action shape/content mismatch: {actions.shape}, expected {expected}")
        result[episode_id] = actions
    return result


def _raw_action_chunks(
    dataset,
    raw_actions: dict[int, np.ndarray],
    sample_indices: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_ids = np.empty(len(sample_indices), dtype=np.int64)
    frame_ids = np.empty(len(sample_indices), dtype=np.int64)
    chunks = np.empty((len(sample_indices), horizon, 7), dtype=np.float32)
    for row, sample_index in enumerate(sample_indices):
        episode_id, frame_id = dataset.all_steps[int(sample_index)]
        episode_id, frame_id = int(episode_id), int(frame_id)
        trajectory = raw_actions[episode_id]
        future = np.minimum(frame_id + np.arange(horizon), len(trajectory) - 1)
        chunks[row] = trajectory[future]
        episode_ids[row] = episode_id
        frame_ids[row] = frame_id
    return chunks, episode_ids, frame_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate an evenly spaced subset; omit to evaluate every observation.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg = OmegaConf.load(config_path)
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
    raw_targets, episode_ids, frame_ids = _raw_action_chunks(
        dataset, raw_actions, sample_indices, horizon=20
    )

    subset = Subset(dataset, sample_indices.tolist())
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": _collate_samples,
    }
    if args.num_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(subset, **loader_kwargs)

    policy = PolicyServerWrapper(
        ckpt_path=str(checkpoint),
        device="cuda",
        use_bf16=True,
        unnorm_key="new_embodiment",
    )
    model = policy._framework
    processor = policy.get_norm_processor()
    action_stats = policy._norm_processors["new_embodiment"]._norm_stats["new_embodiment"]["action"]
    action_low = np.asarray(action_stats["min"], dtype=np.float32)
    action_high = np.asarray(action_stats["max"], dtype=np.float32)

    predicted_controls_all = []
    target_controls_all = []
    predicted_actions_all = []
    spline_target_actions_all = []
    inference_seconds = []

    for batch_number, samples in enumerate(loader, start=1):
        requests = [{key: sample[key] for key in ("image", "lang", "state")} for sample in samples]
        target_controls = np.asarray([sample["action"] for sample in samples], dtype=np.float32)
        start = time.perf_counter()
        predicted_controls = model.predict_action_parameters(examples=requests)["normalized_actions"]
        inference_seconds.append(time.perf_counter() - start)
        predicted_controls = np.asarray(predicted_controls, dtype=np.float32)

        predicted_normalized = model.decode_normalized_controls(predicted_controls)
        target_normalized = model.decode_normalized_controls(target_controls)
        predicted_actions = np.stack([processor.unapply_actions(value) for value in predicted_normalized])
        spline_targets = np.stack([processor.unapply_actions(value) for value in target_normalized])

        predicted_controls_all.append(predicted_controls)
        target_controls_all.append(target_controls)
        predicted_actions_all.append(predicted_actions.astype(np.float32))
        spline_target_actions_all.append(spline_targets.astype(np.float32))
        if batch_number % 100 == 0 or batch_number == len(loader):
            done = min(batch_number * args.batch_size, len(sample_indices))
            print(f"evaluated {done}/{len(sample_indices)} observations", flush=True)

    predicted_controls = np.concatenate(predicted_controls_all)
    target_controls = np.concatenate(target_controls_all)
    predicted_actions = np.concatenate(predicted_actions_all)
    spline_target_actions = np.concatenate(spline_target_actions_all)
    if predicted_actions.shape != raw_targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predicted_actions.shape} vs {raw_targets.shape}")

    control_error = predicted_controls - target_controls
    spline_prediction_error = predicted_actions - spline_target_actions
    raw_prediction_error = predicted_actions - raw_targets
    representation_error = spline_target_actions - raw_targets
    persistence_actions = np.repeat(raw_targets[:, :1, :], raw_targets.shape[1], axis=1)
    persistence_error = persistence_actions - raw_targets
    outside = np.logical_or(
        predicted_actions < action_low[None, None, :],
        predicted_actions > action_high[None, None, :],
    )
    inference_array = np.asarray(inference_seconds, dtype=np.float64)

    model_raw_metrics = _trajectory_metrics(raw_prediction_error)
    persistence_metrics = _trajectory_metrics(persistence_error)
    model_mse = model_raw_metrics["overall"]["mse"]
    persistence_mse = persistence_metrics["overall"]["mse"]
    report = {
        "evaluation": "open_loop_recorded_observations",
        "definition": (
            "Each prediction uses the recorded observation at that frame; predicted actions are not fed back."
        ),
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "dataset_observations": total_observations,
        "evaluated_observations": int(len(sample_indices)),
        "episodes": int(len(np.unique(episode_ids))),
        "batch_size": args.batch_size,
        "action_frequency_hz": 25.0,
        "control_shape_per_observation": [13, 7],
        "decoded_action_shape_per_observation": [20, 7],
        "normalized_control_prediction": _control_metrics(control_error),
        "physical_action_prediction_vs_spline_target": _trajectory_metrics(spline_prediction_error),
        "physical_action_prediction_vs_original_target": model_raw_metrics,
        "spline_representation_vs_original_target": _trajectory_metrics(representation_error),
        "current_action_persistence_baseline_vs_original_target": persistence_metrics,
        "comparison_to_persistence_baseline": {
            "mse_reduction_fraction": float(1.0 - model_mse / persistence_mse),
            "rmse_reduction_fraction": float(
                1.0
                - model_raw_metrics["overall"]["rmse"]
                / persistence_metrics["overall"]["rmse"]
            ),
        },
        "prediction_bounds": {
            "training_min": action_low.tolist(),
            "training_max": action_high.tolist(),
            "outside_training_minmax_count": int(outside.sum()),
            "total_values": int(outside.size),
            "outside_training_minmax_fraction": float(outside.mean()),
            "per_dimension_fraction": {
                name: float(outside[:, :, dim].mean())
                for dim, name in enumerate(ACTION_NAMES)
            },
        },
        "inference_timing": {
            "batches": int(len(inference_array)),
            "total_seconds": float(inference_array.sum()),
            "mean_batch_seconds": float(inference_array.mean()),
            "mean_observation_seconds_amortized": float(inference_array.sum() / len(sample_indices)),
        },
    }

    suffix = "all" if len(sample_indices) == total_observations else f"subset_{len(sample_indices)}"
    report_path = output_dir / f"open_loop_evaluation_{suffix}.json"
    arrays_path = output_dir / f"open_loop_errors_{suffix}.npz"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        arrays_path,
        sample_index=sample_indices,
        episode_index=episode_ids,
        frame_index=frame_ids,
        control_error=control_error.astype(np.float32),
        prediction_vs_spline_error=spline_prediction_error.astype(np.float32),
        prediction_vs_original_error=raw_prediction_error.astype(np.float32),
        spline_vs_original_error=representation_error.astype(np.float32),
        persistence_vs_original_error=persistence_error.astype(np.float32),
    )
    print(json.dumps({
        "report": str(report_path),
        "arrays": str(arrays_path),
        "evaluated_observations": int(len(sample_indices)),
        "control_rmse": report["normalized_control_prediction"]["overall"]["rmse"],
        "decoded_vs_original_rmse": report["physical_action_prediction_vs_original_target"]["overall"]["rmse"],
        "spline_reconstruction_rmse": report["spline_representation_vs_original_target"]["overall"]["rmse"],
    }, indent=2))


if __name__ == "__main__":
    main()
