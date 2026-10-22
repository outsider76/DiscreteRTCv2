#!/usr/bin/env python3
"""Open-loop evaluation of Piper per-step delta-joint policies (raw or UniformBSpline) on recorded data.

The policies are trained on the BSplineEncoder ``state-delta-joint`` dataset
(``data/processed/piper_demos/StateDeltaJoint/...``):

    action[t, joints]  = measured q[t+1] - measured q[t]   (rad, one 30 Hz step)
    action[t, gripper] = measured gripper[t+1]              (absolute)

so a 30-step chunk predicted at observation t is executed as
``q_cmd[t+k+1] = q[t] + sum_{j<=k} pred[j, :6]`` and ``gripper_cmd[t+k+1] = pred[k, 6]``.

Every observation comes from the recording (predictions are never fed back), on the held-out
``validation`` episodes.  Before any metric the script checks, and aborts on failure, that

  * the run trains on the StateDeltaJoint dataset with ``action_mode: abs`` (no second delta transform);
  * the stored actions equal the per-step deltas of ``observation.state`` (to float32 rounding);
  * the checkpoint's un-normalization stats equal the dataset's train statistics;
  * every evaluated loader sample carries the state normalized from ``observation.state[t]`` and the
    training target (raw: the per-step delta chunk; B-spline: the sidecar controls);
  * (B-spline) the checkpoint's decode basis equals the sidecar encoder basis;
  * integrating the ground-truth chunk through the same code reproduces the measured joints
    (exactly for raw, within the spline fit error for B-spline).

Reported: per-step delta error vs a zero-delta baseline, integrated joint error per horizon vs
hold-position, and a chunked replay that stitches the first ``--execute-steps`` of each chunk over
whole episodes (with plots).

  python examples/realRobots/Piper/eval_files/evaluate_delta_joint_open_loop.py \\
      --run-dir results/Checkpoints/<run_id> --step 10000
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
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DELTA_DATASET_MARK = "StateDeltaJoint"
REPORT_HORIZONS = (1, 5, 10, 15, 20, 25, 30)


def _collate_samples(samples):
    return samples


def _rmse(error: np.ndarray, axis) -> np.ndarray:
    return np.sqrt(np.mean(np.square(error), axis=axis))


def _resolve(path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _stats_array(stats, keys) -> dict:
    """Concatenate per-key dataset statistics (pydantic or dict) into flat min/max arrays."""
    out = {}
    for field in ("min", "max"):
        parts = []
        for key in keys:
            entry = stats[key]
            entry = entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
            parts.append(np.asarray(entry[field], dtype=np.float64).reshape(-1))
        out[field] = np.concatenate(parts)
    return out


def _unnormalize(values: np.ndarray, stats: dict) -> np.ndarray:
    low, high = stats["min"], stats["max"]
    return (np.asarray(values, dtype=np.float64) + 1.0) / 2.0 * (high - low) + low


def _normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    low, high = stats["min"], stats["max"]
    result = np.asarray(values, dtype=np.float64).copy()
    mask = high != low
    result[..., mask] = 2.0 * (result[..., mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    return result


def _check(condition: bool, message: str, checks: dict, name: str, value=None) -> None:
    if not condition:
        raise AssertionError(f"[{name}] {message}")
    checks[name] = {"passed": True, "detail": message, **({} if value is None else {"value": value})}
    print(f"  ok  {name}: {message}")


def _ground_truth_chunks(actions: np.ndarray, frame: int, horizon: int) -> np.ndarray:
    """Training target of the raw loader: stored actions, joint deltas 0 past the episode end."""
    steps = frame + np.arange(horizon)
    chunk = actions[np.minimum(steps, len(actions) - 1)].copy()
    chunk[steps > len(actions) - 1, :6] = 0.0
    return chunk


def _plot_episode(path: Path, episode: int, states: np.ndarray, replay: np.ndarray, hold: np.ndarray,
                  execute: int, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(states)
    t = np.arange(T) / 30.0
    fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(t, states[:, dim], color="black", lw=1.2, label="measured")
        ax.plot(t, replay[:, dim], color="tab:red", lw=1.0, label="open-loop chunks (q[t0]+cumsum)")
        if dim < 6:
            ax.plot(t, hold[:, dim], color="tab:gray", lw=0.8, ls=":", label="hold q[t0]")
        for start in range(0, T, execute):
            ax.axvline(start / 30.0, color="tab:blue", lw=0.3, alpha=0.3)
        ax.set_ylabel(JOINT_NAMES[dim] if dim < 6 else "gripper")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{title}\nvalidation episode {episode}: replan every {execute} steps from the recorded observation")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", default="10000", help="checkpoint step, or 'final' for final_model")
    parser.add_argument("--stride", type=int, default=1, help="evaluate frames with frame_index %% stride == 0")
    parser.add_argument("--execute-steps", type=int, default=30, help="replay: steps executed per chunk")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="default <run-dir>/open_loop_delta_joint/steps_<step>")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.execute_steps % args.stride:
        raise ValueError("--execute-steps must be a multiple of --stride")

    run_dir = _resolve(args.run_dir)
    checkpoint = (run_dir / "final_model/pytorch_model.pt" if args.step == "final"
                  else run_dir / f"checkpoints/steps_{int(args.step)}_pytorch_model.pt")
    output_dir = _resolve(args.output_dir) if args.output_dir else run_dir / "open_loop_delta_joint" / f"steps_{args.step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg = OmegaConf.load(run_dir / "config.full.yaml")
    vd = cfg.datasets.vla_data
    spline = cfg.framework.name == "QwenPI_v3_UniformBSpline"
    horizon = 30
    checks: dict = {}
    print(f"{run_dir.name} @ {checkpoint.name} ({'B-spline' if spline else 'raw'})")

    # ---- configuration --------------------------------------------------------------------------
    mode = str(vd.get("action_mode", "abs")).lower()
    _check(mode == "abs" and not vd.get("action_mode_apply_keys"),
           f"loader action_mode={mode}, no apply keys (deltas come from the dataset, not a transform)",
           checks, "config_action_mode")
    vd.episode_split = "validation"
    dataset = get_vla_dataset(vd, mode="val").datasets[0]
    _check(DELTA_DATASET_MARK in str(dataset.dataset_path) and dataset._action_mode == "abs",
           f"dataset {dataset.dataset_path}", checks, "config_dataset")
    if spline:
        controls_root = _resolve(vd.controls_dataset_path)
        manifest = json.loads((controls_root / "manifest.json").read_text())
        _check(DELTA_DATASET_MARK in str(controls_root)
               and Path(manifest["source_dataset"]).name == Path(str(dataset.dataset_path)).name,
               f"sidecar {controls_root.name} built from {manifest['source_dataset']}", checks, "config_sidecar")
        rep = cfg.framework.action_representation
        horizon = int(rep.executable_horizon)
    else:
        horizon = int(cfg.framework.action_model.action_horizon)

    # ---- stored targets are per-step deltas of the measured state ---------------------------------
    episodes = {}
    worst = 0.0
    for trajectory_id in dataset.trajectory_ids:
        frame = dataset.get_trajectory_data(int(trajectory_id))
        actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        worst = max(worst, float(np.abs(actions[:-1, :6] - np.diff(states[:, :6], axis=0)).max()),
                    float(np.abs(actions[:-1, 6] - states[1:, 6]).max()))
        episodes[int(trajectory_id)] = (actions, states)
    _check(worst < 1e-6, f"action[t] == (q[t+1]-q[t], g[t+1]) on all {len(episodes)} validation episodes "
           f"({sum(len(a) for a, _ in episodes.values())} rows; max {worst:.1e}, float32 rounding)",
           checks, "data_per_step_delta", worst)

    # ---- policy + normalization ------------------------------------------------------------------
    policy = PolicyServerWrapper(ckpt_path=str(checkpoint), device="cuda", use_bf16=False, unnorm_key="new_embodiment")
    model = policy._framework
    model.eval()
    norm_stats = policy._norm_processors["new_embodiment"]._norm_stats["new_embodiment"]
    ckpt_action = {k: np.asarray(norm_stats["action"][k], np.float64) for k in ("min", "max")}
    ckpt_state = {k: np.asarray(norm_stats["state"][k], np.float64) for k in ("min", "max")}
    # The validation loader normalizes with *validation-split* statistics (meta/stats_gr00t_validation.json),
    # which differ from the train statistics the policy was trained and is deployed with.  Its samples are
    # only used for images/language and for checking targets; the model state is normalized here with the
    # checkpoint statistics, exactly as the live client does.
    loader_action = _stats_array(dataset.metadata.statistics.action, ["joints", "gripper"])
    loader_state = _stats_array(dataset.metadata.statistics.state, ["joints", "gripper"])
    train_stats = json.loads((_resolve(dataset.dataset_path) / "meta/stats_gr00t_train.json").read_text())["statistics"]
    diff = max(np.abs(ckpt_action[k] - np.asarray(train_stats["action"][k])).max() for k in ("min", "max"))
    diff = max(diff, max(np.abs(ckpt_state[k] - np.asarray(train_stats["observation.state"][k])).max() for k in ("min", "max")))
    _check(diff < 1e-6 and np.abs(ckpt_action["max"][:6]).max() < 0.5,
           f"checkpoint stats == meta/stats_gr00t_train.json; joint action range "
           f"[{ckpt_action['min'][:6].min():.4f}, {ckpt_action['max'][:6].max():.4f}] rad (delta scale)",
           checks, "normalization_stats", float(diff))
    if spline:
        from spline_encoder.encoding.serialization import load_encoder

        sidecar_basis = np.asarray(load_encoder(controls_root / "encoder.json").basis, np.float64)
        model_basis = np.asarray(model._bspline_decode_basis, np.float64)
        basis_diff = float(np.abs(sidecar_basis - model_basis).max()) if sidecar_basis.shape == model_basis.shape else np.inf
        _check(basis_diff < 1e-6 and model_basis.shape == (horizon, int(rep.num_control_points)),
               f"model decode basis {model_basis.shape} == sidecar encoder basis", checks, "bspline_basis", basis_diff)
        sidecar_controls = {}
        for episode in episodes:
            with np.load(controls_root / "data" / f"episode_{episode:06d}.npz") as payload:
                sidecar_controls[episode] = payload["controls"].astype(np.float64)

    # ---- evaluate every stride-th frame of the validation episodes -------------------------------
    steps = np.asarray(dataset.all_steps, dtype=np.int64)
    selected = np.flatnonzero(steps[:, 1] % args.stride == 0)
    loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_collate_samples,
                        persistent_workers=bool(args.num_workers), prefetch_factor=2 if args.num_workers else None)
    preds, gts, oracle, keys, packed = [], [], [], [], []
    worst_state = worst_target = 0.0
    split_shift = 0.0
    started = time.time()
    cursor = 0
    for batch in loader:
        rows = steps[selected[cursor:cursor + len(batch)]]
        cursor += len(batch)
        examples = []
        for sample, (episode, frame) in zip(batch, rows):
            states = episodes[int(episode)][1]
            loader_in = np.asarray(sample["state"], np.float64)
            # loader state is observation.state[t] (normalized with the loader's own stats)
            worst_state = max(worst_state, float(np.abs(loader_in.reshape(-1)[:7] - _normalize(states[frame], loader_state)).max()))
            model_in = _normalize(states[frame], ckpt_state).reshape(loader_in.shape).astype(np.asarray(sample["state"]).dtype)
            split_shift = max(split_shift, float(np.abs(model_in.astype(np.float64) - loader_in).max()))
            examples.append({"image": sample["image"], "lang": sample["lang"], "state": model_in})
        with torch.no_grad():
            normalized = model.predict_action(examples=examples)
        normalized = np.asarray(normalized["normalized_actions"], np.float64)
        if normalized.shape[1:] != (horizon, 7):
            raise AssertionError(f"decoded chunk shape {normalized.shape}, expected (B, {horizon}, 7)")
        for sample, (episode, frame), chunk in zip(batch, rows, normalized):
            actions, states = episodes[int(episode)]
            gt = _ground_truth_chunks(actions, int(frame), horizon)
            target = _unnormalize(np.asarray(sample["action"], np.float32), loader_action)
            if spline:
                controls = sidecar_controls[int(episode)][int(frame)]
                worst_target = max(worst_target, float((np.abs(target - controls) / (loader_action["max"] - loader_action["min"])).max()))
                oracle.append(np.asarray(model_basis @ controls))
            else:
                worst_target = max(worst_target, float(np.abs(target - gt).max()))
                oracle.append(gt)
                packed.append(target)
            preds.append(_unnormalize(chunk, ckpt_action))
            gts.append(gt)
            keys.append((int(episode), int(frame)))
    elapsed = time.time() - started
    _check(worst_state < 2e-3, f"loader sample is observation.state[t] (max {worst_state:.2e}, float16 packing); model fed "
           f"train-stat normalization (differs from the validation loader's by up to {split_shift:.3f})",
           checks, "input_state", worst_state)
    _check(worst_target < 2e-3,
           ("loader target == sidecar controls (max rel %.2e)" if spline else "loader target == per-step delta chunk (max %.2e)")
           % worst_target, checks, "training_target", worst_target)

    P, G, O = np.asarray(preds), np.asarray(gts), np.asarray(oracle)
    E = np.asarray(keys)
    Q0 = np.stack([episodes[e][1][f, :6] for e, f in E])
    future = np.stack([episodes[e][1][np.minimum(f + 1 + np.arange(horizon), len(episodes[e][1]) - 1), :6] for e, f in E])
    valid = np.stack([f + 1 + np.arange(horizon) < len(episodes[e][1]) for e, f in E])

    def integrated(chunks):
        return Q0[:, None] + np.cumsum(chunks[..., :6], axis=1)

    def horizon_rmse(q_hat):
        return np.stack([_rmse((q_hat[valid[:, k], k] - future[valid[:, k], k]), axis=0) for k in range(horizon)])

    model_h = horizon_rmse(integrated(P))
    hold_h = horizon_rmse(np.repeat(Q0[:, None], horizon, axis=1))
    oracle_h = horizon_rmse(integrated(O))
    oracle_limit = 1e-6 if not spline else 0.05
    _check(oracle_h.max() <= oracle_limit,
           f"integrating the ground-truth {'decoded controls' if spline else 'chunk'} reproduces measured q "
           f"(max per-horizon RMSE {oracle_h.max():.2e} rad)", checks, "oracle_integration", float(oracle_h.max()))
    if packed:
        # what the raw model is actually trained on: float16-packed targets, integrated over the chunk
        packed_h = horizon_rmse(integrated(np.asarray(packed)))
        checks["oracle_integration"]["float16_packed_targets_max_rmse"] = float(packed_h.max())
        print(f"      float16-packed training targets integrate to max per-horizon RMSE {packed_h.max():.2e} rad")

    # ---- chunked replay over whole episodes -----------------------------------------------------
    index = {key: row for row, key in enumerate(map(tuple, E))}
    replay_err, hold_err, replay_grip = [], [], []
    plots = []
    for episode, (actions, states) in episodes.items():
        T = len(states)
        replay = states.copy()
        hold = states[:, :6].copy()
        for start in range(0, T - 1, args.execute_steps):
            chunk = P[index[(episode, start)]]
            n = min(args.execute_steps, T - 1 - start)
            replay[start + 1:start + 1 + n, :6] = states[start, :6] + np.cumsum(chunk[:n, :6], axis=0)
            replay[start + 1:start + 1 + n, 6] = chunk[:n, 6]
            hold[start + 1:start + 1 + n] = states[start, :6]
        replay_err.append(replay[1:, :6] - states[1:, :6])
        hold_err.append(hold[1:] - states[1:, :6])
        replay_grip.append(replay[1:, 6] - states[1:, 6])
        if not args.no_plots:
            plot = output_dir / f"replay_episode_{episode:03d}.png"
            _plot_episode(plot, episode, states, replay, hold, args.execute_steps, f"{run_dir.name} @ step {args.step}")
            plots.append(plot.name)
    replay_err, hold_err, replay_grip = map(np.concatenate, (replay_err, hold_err, replay_grip))

    zero_delta = _rmse(G[..., :6], axis=(0, 1))
    delta_rmse = _rmse(P[..., :6] - G[..., :6], axis=(0, 1))
    corr = [float(np.corrcoef(P[..., j].ravel(), G[..., j].ravel())[0, 1]) for j in range(6)]
    report = {
        "run": run_dir.name,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT) if checkpoint.is_relative_to(REPO_ROOT) else checkpoint),
        "representation": "uniform_left_bspline_decoded" if spline else "raw",
        "action_definition": "joints: q[t+k+1]-q[t+k] (measured, rad); gripper: measured g[t+k+1]",
        "episodes": sorted(episodes),
        "observations": int(len(P)),
        "stride": args.stride,
        "seconds": round(elapsed, 1),
        "checks": checks,
        "per_step_delta": {
            "rmse_model": dict(zip(JOINT_NAMES, delta_rmse.round(6).tolist())),
            "rmse_zero_delta": dict(zip(JOINT_NAMES, zero_delta.round(6).tolist())),
            "corr": dict(zip(JOINT_NAMES, np.round(corr, 4).tolist())),
            "pred_abs_max": dict(zip(JOINT_NAMES, np.abs(P[..., :6]).max((0, 1)).round(5).tolist())),
            "gt_abs_max": dict(zip(JOINT_NAMES, np.abs(G[..., :6]).max((0, 1)).round(5).tolist())),
        },
        "integrated_joint_rmse_by_horizon": {
            "horizon": list(range(1, horizon + 1)),
            "model": model_h.round(6).tolist(),
            "hold_position": hold_h.round(6).tolist(),
            "ground_truth_representation": oracle_h.round(6).tolist(),
        },
        "gripper_rmse": float(_rmse(P[..., 6] - G[..., 6], axis=None)),
        "replay": {
            "execute_steps": args.execute_steps,
            "joint_rmse_model": dict(zip(JOINT_NAMES, _rmse(replay_err, 0).round(6).tolist())),
            "joint_rmse_hold": dict(zip(JOINT_NAMES, _rmse(hold_err, 0).round(6).tolist())),
            "gripper_rmse": float(_rmse(replay_grip, None)),
            "plots": plots,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    np.set_printoptions(precision=4, suppress=True)
    print(f"\n{len(P)} observations from {len(episodes)} validation episodes in {elapsed:.0f} s")
    print("per-step delta RMSE  model", delta_rmse, "| zero-delta", zero_delta)
    print("corr(pred, GT)", np.round(corr, 3), "| gripper RMSE", round(report["gripper_rmse"], 4))
    for k in REPORT_HORIZONS:
        print(f"q[t+{k:2d}] RMSE  model {model_h[k - 1]} | hold {hold_h[k - 1]} | GT repr {oracle_h[k - 1]}")
    print(f"replay (execute {args.execute_steps}) joint RMSE model", _rmse(replay_err, 0), "| hold", _rmse(hold_err, 0),
          "| gripper", round(report["replay"]["gripper_rmse"], 4))
    print("wrote", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
