#!/usr/bin/env python3
"""Run a Piper checkpoint on one recorded observation without moving hardware.

The input contract exactly follows training:
  image: [global RGB, hand RGB]
  state: normalized [six measured joints, gripper]
  action: a model-configured chunk of absolute [six joint targets, gripper] commands

Use ``--host`` to test a running WebSocket policy server. Without ``--host``,
the checkpoint is loaded in-process. This program never publishes ROS topics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results/Checkpoints/piper_pick_white_block_20260818_qwenpi_measured_50hz_h50"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints/steps_10000_pytorch_model.pt"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/piper_demos/20260818_Pick_white_block"
DEFAULT_TASK = "Pick up white block and place it in the box."
TRAINING_IMAGE_HW = (224, 224)


def _run_dir_from_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.parent.name in {"checkpoints", "final_model"}:
        return checkpoint.parent.parent
    return checkpoint.parent


def _load_statistics(checkpoint: Path, unnorm_key: str) -> dict:
    path = _run_dir_from_checkpoint(checkpoint) / "dataset_statistics.json"
    with path.open(encoding="utf-8") as file:
        all_stats = json.load(file)
    if unnorm_key not in all_stats:
        raise KeyError(f"{unnorm_key!r} not in {path}; available={list(all_stats)}")
    return all_stats[unnorm_key]


def _normalize_minmax(values: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != low.shape or low.shape != high.shape:
        raise ValueError(f"State/stat shape mismatch: values={values.shape}, min={low.shape}, max={high.shape}")
    result = values.copy()
    mask = high != low
    result[mask] = 2.0 * (values[mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    return result


def _read_rgb_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != TRAINING_IMAGE_HW:
        rgb = cv2.resize(
            rgb,
            (TRAINING_IMAGE_HW[1], TRAINING_IMAGE_HW[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(rgb)


def _nearest_frame_index(timestamps: list[float], target: float) -> int:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Camera timestamps must be non-empty and finite")
    right = int(np.clip(np.searchsorted(values, target, side="left"), 0, len(values) - 1))
    left = max(0, right - 1)
    return right if abs(values[right] - target) < abs(target - values[left]) else left


def _load_recorded_observation(raw_root: Path, episode: Path | None, frame_index: int):
    if episode is None:
        candidates = sorted(path.parent for path in raw_root.rglob("data.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No episode directories under {raw_root}")
        episode = candidates[-1]
    episode = episode.expanduser().resolve()

    with (episode / "data.pkl").open("rb") as file:
        data = pickle.load(file)
    count = len(data["observations"])
    if count == 0:
        raise ValueError(f"Empty episode: {episode}")
    if frame_index < 0:
        frame_index = count // 2
    if not 0 <= frame_index < count:
        raise IndexError(f"frame_index={frame_index}, episode length={count}")

    observation = data["observations"][frame_index]
    command = data["actions"][frame_index]
    raw_state = np.concatenate(
        [
            np.asarray(observation["arm_joint_position"], dtype=np.float32).reshape(-1),
            np.asarray(observation["gripper_pos"], dtype=np.float32).reshape(-1),
        ]
    )
    recorded_action = np.concatenate(
        [
            np.asarray(command["arm_joint_position"], dtype=np.float32).reshape(-1),
            np.asarray(command["gripper_pos"], dtype=np.float32).reshape(-1),
        ]
    )
    if raw_state.shape != (7,) or recorded_action.shape != (7,):
        raise ValueError(f"Expected 7-D state/action, got {raw_state.shape}/{recorded_action.shape}")

    camera_timestamps = data.get("camera_timestamps")
    if camera_timestamps is None:
        global_index = hand_index = frame_index
    else:
        target_timestamp = float(data["timestamps"][frame_index])
        global_index = _nearest_frame_index(camera_timestamps["global"], target_timestamp)
        hand_index = _nearest_frame_index(camera_timestamps["hand"], target_timestamp)
    images = [
        _read_rgb_frame(episode / "global_image.mp4", global_index),
        _read_rgb_frame(episode / "hand_image.mp4", hand_index),
    ]
    return episode, frame_index, images, raw_state, recorded_action


def _extract_actions(response: dict) -> np.ndarray:
    if response.get("status") == "error":
        raise RuntimeError(f"Policy server error: {response.get('error')}")
    data = response.get("data", response)
    if "actions" not in data:
        raise KeyError(f"No 'actions' in response; keys={list(data)}")
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[0] <= 0 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions shape (T, 7) with T > 0, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Predicted actions contain NaN or Inf")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--frame-index", type=int, default=-1, help="Negative selects the middle frame.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--unnorm-key", default="new_embodiment")
    parser.add_argument("--host", help="Use WebSocket server at this host instead of loading locally.")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    stats = _load_statistics(checkpoint, args.unnorm_key)
    episode, frame_index, images, raw_state, recorded_action = _load_recorded_observation(
        args.raw_root.expanduser().resolve(), args.episode, args.frame_index
    )
    norm_state = _normalize_minmax(raw_state, stats["state"])
    example = {
        "image": images,  # Training camera order: global, hand.
        "lang": args.task,
        "state": norm_state[None, :],
    }

    start = time.perf_counter()
    if args.host:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        client = WebsocketClientPolicy(host=args.host, port=args.port)
        metadata = client.get_server_metadata()
        response = client.predict_action(
            {"examples": [example], "unnorm_key": args.unnorm_key}
        )
        client.close()
        actions = _extract_actions(response)
    else:
        from deployment.model_server.policy_wrapper import PolicyServerWrapper

        policy = PolicyServerWrapper(
            ckpt_path=str(checkpoint),
            device="cuda",
            use_bf16=not args.no_bf16,
            unnorm_key=args.unnorm_key,
        )
        metadata = policy.metadata
        actions = np.asarray(
            policy.predict_action([example], unnorm_key=args.unnorm_key)["actions"][0],
            dtype=np.float32,
        )
    latency = time.perf_counter() - start

    action_low = np.asarray(stats["action"]["min"], dtype=np.float32)
    action_high = np.asarray(stats["action"]["max"], dtype=np.float32)
    outside = np.logical_or(actions < action_low[None, :], actions > action_high[None, :])

    np.set_printoptions(precision=4, suppress=True)
    print(f"mode: {'websocket' if args.host else 'local'}")
    print(f"checkpoint: {checkpoint}")
    print(f"episode: {episode.name}, frame: {frame_index}")
    print(f"metadata: {metadata}")
    print(f"latency_s: {latency:.3f}")
    print(f"raw_state [j1..j6, gripper]: {raw_state}")
    print(f"recorded target at frame:    {recorded_action}")
    print(f"predicted first target:      {actions[0]}")
    print(f"predicted actions shape:     {actions.shape}")
    print(f"outside training min/max:    {int(outside.sum())}/{outside.size}")
    print(f"predicted {len(actions)}-step absolute target chunk:")
    print(actions)
    print(
        f"PASS: checkpoint inference is finite and has the Piper {len(actions)}x7 "
        "contract; no robot command was sent."
    )


if __name__ == "__main__":
    main()
