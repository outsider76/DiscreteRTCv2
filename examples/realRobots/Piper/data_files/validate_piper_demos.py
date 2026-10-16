#!/usr/bin/env python3
"""Validate the matched raw-action / uniform-left B-spline Piper datasets.

For each task (outputs of ``preprocess_piper_demos.py``) this checks:

1. structure: LeRobot v2.1 metadata, train/validation splits, starVLA state
   aliases pointing at the same columns, sidecar manifest/encoder/provenance;
2. raw/spline alignment: identical episodes, one sidecar record per dataset
   row, ``frame_index`` equality, source-frame mapping rows;
3. finite values for state, action, controls, and knots;
4. geometry: cubic uniform-left, 15 executable spans x 2 steps, 18 controls,
   decode basis (30, 18) with partition of unity and left clamping;
5. reconstruction: decode every record with the SAME basis the policy uses and
   compare with the 30 future actions (tail padded with the final action),
   per split and per dimension; a sample of windows is re-fitted to confirm
   the stored controls are reproducible;
6. timing/leakage: signed offsets of the selected robot sample and camera
   frames relative to each observation time t, recomputed from raw data.pkl;
7. loader compatibility (``--loader``): starVLA datasets for both variants with
   ``episode_split=train`` load, share normalization statistics fitted on the
   training split only, exclude validation episodes, return two 224x224 images
   + (1,7) state + (30,7) raw or (18,7) spline targets, and the decoded spline
   targets match the raw targets in normalized space.

A JSON report is written to ``<output-root>/VALIDATION.json``; the exit code is
non-zero if any hard check fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "BSplineEncoder" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_piper_demos import TASKS, check_encoder_geometry, dataset_path, sidecar_path  # noqa: E402

HORIZON = 30
NUM_CONTROLS = 18


class CheckFailed(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def error_stats(errors: np.ndarray) -> dict:
    absolute = np.abs(errors)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(absolute.mean()),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
        "rmse_per_dim": np.sqrt(np.mean(errors.reshape(-1, errors.shape[-1]) ** 2, axis=0)).round(6).tolist(),
        "max_abs_per_dim": absolute.reshape(-1, errors.shape[-1]).max(axis=0).round(6).tolist(),
    }


def future_windows(actions: np.ndarray, horizon: int) -> np.ndarray:
    """``actions[o:o+H]`` for every observation, padded with the final action."""
    index = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon)[None, :], len(actions) - 1)
    return actions[index]


def split_ids(info: dict) -> dict[str, set[int]]:
    from starVLA.dataloader.gr00t_lerobot.datasets import resolve_split_episode_ids

    return {name: resolve_split_episode_ids(info, name) for name in info["splits"]}


def signed_offsets(raw_dir: Path, alignment: pd.DataFrame, cameras: list[str]) -> dict[str, np.ndarray]:
    with open(raw_dir / "data.pkl", "rb") as handle:
        payload = pickle.load(handle)
    target = alignment["source_time"].to_numpy(np.float64)
    robot_time = np.asarray(payload["timestamps"], dtype=np.float64)
    offsets = {"robot": (robot_time[alignment["source_robot_index"].to_numpy()] - target) * 1000.0}
    capture = payload.get("camera_frame_source_timestamps") or payload.get("camera_timestamps")
    for name in cameras:
        times = np.asarray(capture[name], dtype=np.float64)
        offsets[name] = (times[alignment[f"camera.{name}.source_frame_index"].to_numpy()] - target) * 1000.0
    return offsets


def validate_task(task: str, output_root: Path, refit_every: int) -> dict:
    from spline_encoder import load_encoder

    dataset, sidecar = dataset_path(output_root, task), sidecar_path(output_root, task)
    report: dict = {"dataset": str(dataset), "sidecar": str(sidecar)}
    info = json.loads((dataset / "meta/info.json").read_text())
    require(info["codebase_version"] == "v2.1" and float(info["fps"]) == 30.0, "expected LeRobot v2.1 at 30 Hz")
    require(info["features"]["observation.state"]["shape"] == [7], "state must be 7-D")
    require(info["features"]["action"]["shape"] == [7], "action must be 7-D")
    cameras = sorted(k.split(".")[-1] for k, v in info["features"].items() if v.get("dtype") == "video")
    require(cameras == ["global", "hand"], f"cameras {cameras}")
    splits = split_ids(info)
    require(set(splits) == {"train", "validation"} and not (splits["train"] & splits["validation"]),
            f"splits {info['splits']}")
    require(splits["train"] | splits["validation"] == set(range(info["total_episodes"])), "splits must cover all")
    report["splits"] = info["splits"]

    modality = json.loads((dataset / "meta/modality.json").read_text())
    for alias, source in (("joints", "measured_joints"), ("gripper", "measured_gripper")):
        a, s = modality["state"][alias], modality["state"][source]
        require(a.get("alias_of") == source and (a["start"], a["end"]) == (s["start"], s["end"])
                and a["original_key"] == s["original_key"] == "observation.state", f"state alias {alias}")
    require([modality["action"][k]["start"] for k in ("joints", "gripper")] == [0, 6], "action layout")
    require(all(modality["action"][k]["absolute"] for k in ("joints", "gripper")), "actions must be absolute")
    require(list(modality["video"]) == ["global", "hand"], f"video order {list(modality['video'])}")

    report["geometry"] = check_encoder_geometry(sidecar)
    encoder = load_encoder(sidecar / "encoder.json")
    basis = encoder.basis
    require(np.allclose(basis.sum(axis=1), 1.0, atol=1e-12), "basis partition of unity")
    require(np.allclose(basis[0], np.eye(NUM_CONTROLS)[0], atol=1e-12), "left clamping: S(0) = c0")
    manifest = json.loads((sidecar / "manifest.json").read_text())
    provenance = json.loads((sidecar / "preprocessing_provenance.json").read_text())
    require(Path(manifest["source_dataset"]) == dataset, "manifest source_dataset")
    require(manifest["record_count"] == info["total_frames"], "record count")
    require(manifest["episode_count"] == info["total_episodes"], "episode count")
    require(manifest["alignment"]["low"] is None and manifest["alignment"]["action_indices"] is None,
            "sidecar must store unnormalized controls of the full action vector")
    require(set(provenance["train_episodes"]) == splits["train"], "sidecar train episodes == dataset train split")
    require(provenance["quantization_calibration"] == "none", "no quantization calibration expected")

    source_episodes = [json.loads(line) for line in (dataset / "meta/source_episodes.jsonl").read_text().splitlines()]
    raw_by_episode = {int(e["episode_index"]): e for e in source_episodes}
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    errors = {"train": [], "validation": []}
    refit_max = 0.0
    nonfinite = 0
    offsets: dict[str, list] = {}
    gripper_actions = []
    for episode in episodes:
        index, length = int(episode["episode_index"]), int(episode["length"])
        split = "train" if index in splits["train"] else "validation"
        chunk = index // info["chunks_size"]
        frame = pd.read_parquet(dataset / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet")
        state, actions = np.stack(frame["observation.state"]), np.stack(frame["action"]).astype(np.float64)
        require(len(frame) == length and np.array_equal(frame["frame_index"], np.arange(length)),
                f"episode {index}: frame_index")
        require(np.all(frame["episode_index"] == index), f"episode {index}: episode_index column")
        with np.load(sidecar / f"data/episode_{index:06d}.npz") as stored:
            controls, knots = stored["controls"], stored["knots"]
            require(np.array_equal(stored["frame_index"], frame["frame_index"]), f"episode {index}: sidecar frames")
            require(np.array_equal(stored["observation_index"], np.arange(length)), f"episode {index}: observations")
            require(np.all(stored["executable_steps"] == HORIZON), f"episode {index}: executable steps")
        require(controls.shape == (length, NUM_CONTROLS, 7), f"episode {index}: controls {controls.shape}")
        nonfinite += int((~np.isfinite(state)).sum() + (~np.isfinite(actions)).sum()
                         + (~np.isfinite(controls)).sum() + (~np.isfinite(knots)).sum())
        require(np.allclose(knots, knots[0]) and np.allclose(knots[0] * 30.0, encoder.knots),
                f"episode {index}: fixed knots")
        targets = future_windows(actions, HORIZON)
        decoded = np.einsum("tc,ocd->otd", basis, controls)
        errors[split].append(decoded - targets)
        for o in range(0, length, refit_every):
            refit_max = max(refit_max, float(np.abs(encoder.fit_control_points(targets[o]) - controls[o]).max()))
        alignment = pd.read_parquet(dataset / f"meta/source_alignment/chunk-{chunk:03d}/episode_{index:06d}.parquet")
        require(np.array_equal(alignment["frame_index"], np.arange(length)), f"episode {index}: source alignment")
        raw_dir = Path(raw_by_episode[index]["source_episode"])
        for name, values in signed_offsets(raw_dir, alignment, cameras).items():
            offsets.setdefault(name, []).append(values)
        gripper_actions.append(actions[:, 6])
    require(nonfinite == 0, f"{nonfinite} non-finite values")
    require(refit_max < 1e-9, f"stored controls not reproducible by refit (max {refit_max})")
    report["finite"] = True
    report["refit_max_abs_difference"] = refit_max
    report["reconstruction"] = {
        split: error_stats(np.concatenate(values)) for split, values in errors.items() if values
    }
    all_errors = np.concatenate(errors["train"] + errors["validation"])
    report["reconstruction"]["all"] = error_stats(all_errors)
    report["reconstruction"]["arm_max_abs"] = float(np.abs(all_errors[..., :6]).max())
    report["reconstruction"]["first_step_max_abs"] = float(np.abs(all_errors[:, 0]).max())
    grip = np.concatenate(gripper_actions)
    report["gripper_action"] = {
        "min": float(grip.min()), "max": float(grip.max()),
        "fraction_below_0.05": float(np.mean(grip < 0.05)), "fraction_above_0.95": float(np.mean(grip > 0.95)),
        "binary": bool(np.all(np.isin(grip, [0.0, 1.0]))),
    }
    report["signed_offset_ms_vs_observation_time"] = {
        name: {"min": float(np.concatenate(v).min()), "max": float(np.concatenate(v).max()),
               "mean": float(np.concatenate(v).mean()),
               "fraction_after_t": float(np.mean(np.concatenate(v) > 0))}
        for name, v in offsets.items()
    }
    report["episodes"], report["frames"] = info["total_episodes"], info["total_frames"]
    cleanup = json.loads((dataset / "meta/cleanup_summary.json").read_text())
    report["cleanup_summary"] = {k: cleanup[k] for k in cleanup if not isinstance(cleanup[k], (list, dict))}
    return report


def validate_loader(task: str, output_root: Path, train_yaml: Path, samples: int) -> dict:
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    dataset, sidecar = dataset_path(output_root, task), sidecar_path(output_root, task)
    data_root = REPO_ROOT / "data"
    base_cfg = OmegaConf.load(train_yaml).datasets.vla_data
    require(base_cfg.get("episode_split") == "train", f"{train_yaml} must train on episode_split=train")
    cfg = OmegaConf.merge(base_cfg, {"controls_dataset_path": str(sidecar), "data_root_dir": str(data_root)})
    name = str(dataset.relative_to(data_root))
    raw = make_LeRobotSingleDataset(data_root, name, "piper_abs_joints_30step", data_cfg=cfg)
    spline = make_LeRobotSingleDataset(data_root, name, "piper_uniform_left_bspline_30step", data_cfg=cfg)
    info = json.loads((dataset / "meta/info.json").read_text())
    splits = split_ids(info)
    report: dict = {}
    for label, ds in (("raw", raw), ("spline", spline)):
        require(set(map(int, ds.trajectory_ids)) == splits["train"], f"{label}: trajectories != train split")
        require(ds.modality_keys["video"] == ["video.global", "video.hand"], f"{label}: video keys")
        require(ds.modality_keys["state"] == ["state.joints", "state.gripper"], f"{label}: state keys")
        for key in ds.modality_keys["video"] + ds.modality_keys["state"]:
            require(list(ds.delta_indices[key]) == [0], f"{label}: {key} must use only the current step t")
    require(json.dumps(raw.metadata.statistics.model_dump(), sort_keys=True, default=_jsonable)
            == json.dumps(spline.metadata.statistics.model_dump(), sort_keys=True, default=_jsonable),
            "raw and spline variants must share normalization statistics")
    stats_file = dataset / "meta/stats_gr00t_train.json"
    require(stats_file.exists(), "train-split statistics cache missing")
    train_actions = np.concatenate([
        np.stack(pd.read_parquet(dataset / f"data/chunk-000/episode_{i:06d}.parquet")["action"]) for i in sorted(splits["train"])
    ])
    all_actions = np.concatenate([
        np.stack(pd.read_parquet(p)["action"]) for p in sorted((dataset / "data").rglob("*.parquet"))
    ])
    stats = raw.metadata.statistics.action
    loaded_min = np.r_[np.asarray(stats["joints"].min), np.asarray(stats["gripper"].min)]
    loaded_max = np.r_[np.asarray(stats["joints"].max), np.asarray(stats["gripper"].max)]
    require(np.allclose(loaded_min, train_actions.min(0), atol=1e-6)
            and np.allclose(loaded_max, train_actions.max(0), atol=1e-6), "stats must equal train-split min/max")
    report["normalization"] = {
        "action_min_train": loaded_min.round(6).tolist(), "action_max_train": loaded_max.round(6).tolist(),
        "action_min_all": all_actions.min(0).round(6).tolist(), "action_max_all": all_actions.max(0).round(6).tolist(),
        "stats_cache": str(stats_file),
    }

    from spline_encoder import load_encoder

    basis = load_encoder(sidecar / "encoder.json").basis
    rng = np.random.default_rng(0)
    steps = [raw.all_steps[i] for i in rng.choice(len(raw.all_steps), size=samples, replace=False)]
    # Always include episode tails, where padding conventions matter.
    for trajectory_id, length in list(zip(raw.trajectory_ids, raw.trajectory_lengths))[:3]:
        steps += [(trajectory_id, int(length) - 1), (trajectory_id, int(length) - 10)]
    worst = 0.0
    for trajectory_id, base_index in steps:
        a = raw._pack_sample(raw.transforms(raw.get_step_data(trajectory_id, base_index)))
        b = spline._pack_sample(spline.transforms(spline.get_step_data(trajectory_id, base_index)))
        require(len(a["image"]) == 2 and all(im.size == (224, 224) for im in a["image"]), "two 224x224 images")
        require(a["lang"] == b["lang"] and a["lang"] == info_task(dataset), f"language {a['lang']!r}")
        require(np.array_equal(a["state"], b["state"]) and a["state"].shape == (1, 7), "state (1,7) shared")
        require(all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a["image"], b["image"])),
                "images identical across variants")
        require(a["action"].shape == (HORIZON, 7) and b["action"].shape == (NUM_CONTROLS, 7), "target shapes")
        require(np.isfinite(a["action"]).all() and np.isfinite(b["action"]).all(), "finite targets")
        decoded = basis @ b["action"].astype(np.float64)
        worst = max(worst, float(np.abs(decoded - a["action"].astype(np.float64)).max()))
    report["samples_checked"] = len(steps)
    report["max_abs_decoded_spline_minus_raw_normalized"] = worst
    report["note"] = "normalized space ([-1,1] per dim from train-split min/max); includes fp16 packing"
    return report


def _jsonable(value):
    return np.asarray(value).tolist()


def info_task(dataset: Path) -> str:
    return json.loads((dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/processed/piper_demos")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--refit-every", type=int, default=97, help="Re-fit every Nth window")
    parser.add_argument("--loader", action="store_true", help="Also build starVLA datasets (decodes video)")
    parser.add_argument("--loader-samples", type=int, default=24)
    parser.add_argument("--train-yaml", type=Path, default=REPO_ROOT / (
        "examples/realRobots/Piper/train_files/"
        "starvla_qwenpi_piper_demos_classify_blocks_30hz_uniform_left_s2_h30_c18_2gpu_gb32_30k.yaml"))
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    output_root = args.output_root.expanduser().resolve()
    report = {"validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tasks": {}}
    failed = False
    for task in args.tasks:
        try:
            result = validate_task(task, output_root, args.refit_every)
            if args.loader:
                result["loader"] = validate_loader(task, output_root, args.train_yaml, args.loader_samples)
            result["passed"] = True
        except Exception as exc:  # report every failure per task instead of aborting the run
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            failed = True
        report["tasks"][task] = result
        print(f"[{task}] {'PASS' if result['passed'] else 'FAIL: ' + result['error']}", flush=True)
    path = output_root / "VALIDATION.json"
    previous = json.loads(path.read_text()) if path.exists() else {"tasks": {}}
    report["tasks"] = {**previous.get("tasks", {}), **report["tasks"]}
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["tasks"], indent=1)[:6000])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
