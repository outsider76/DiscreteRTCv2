#!/usr/bin/env python3
"""Convert asynchronous Piper raw episodes to a uniform LeRobot v2.1 dataset.

New raw episodes contain independent robot, Orbbec and global-camera timestamps.
This converter creates a 50 Hz target clock by default and selects the nearest
robot sample and nearest frame from each camera for every target timestamp.
Schema-v3 episodes additionally retain the original camera acquisition time for
every presentation frame, so alignment diagnostics still expose duplicated or
stale images introduced while making the raw MP4 duration wall-time-correct.
By default, both ``observation.state`` and the action trajectory come from
Piper feedback; ``--action-source command`` preserves the older GELLO-command
target convention. Older synchronized 30 Hz episodes are also supported.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = {
    "global": "global_image.mp4",
    "hand": "hand_image.mp4",
}
STATE_DIM = 7
ACTION_DIM = 7
ACTION_SOURCES = ("observation", "command")


@dataclass(frozen=True)
class CameraStream:
    path: Path
    timestamps: np.ndarray
    capture_timestamps: np.ndarray
    width: int
    height: int
    source_fps: float


@dataclass(frozen=True)
class Episode:
    source_dir: Path
    timestamps: np.ndarray
    state: np.ndarray
    action: np.ndarray
    cameras: dict[str, CameraStream]
    camera_indices: dict[str, np.ndarray]
    alignment_ms: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return int(self.timestamps.shape[0])


def _validate_timestamps(values: Any, label: str, expected: int) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64).reshape(-1)
    if timestamps.shape != (expected,):
        raise ValueError(f"{label}: expected {expected} timestamps, got {timestamps.shape}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{label}: timestamps must be finite and non-decreasing")
    return timestamps


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return source indices nearest to each target; ties choose the earlier sample."""
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(source[right] - target) < np.abs(target - source[left])
    return np.where(choose_right, right, left).astype(np.int64)


def _video_properties(path: Path) -> tuple[int, int, int, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    result = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    if result[0] <= 0 or result[1] <= 0 or result[2] <= 0 or result[3] <= 0:
        raise ValueError(f"{path}: invalid video metadata {result}")
    return result


def _load_rows(
    data_path: Path,
    observations: list[Any],
    actions: list[Any],
    action_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for frame_index, (observation, command) in enumerate(zip(observations, actions)):
        try:
            state_row = np.concatenate(
                [
                    np.asarray(observation["arm_joint_position"], dtype=np.float32).reshape(-1),
                    np.asarray(observation["gripper_pos"], dtype=np.float32).reshape(-1),
                ]
            )
            if action_source == "observation":
                # Train on the trajectory actually reached by Piper. For an
                # action chunk beginning at t, this yields measured q[t:t+H].
                action_row = state_row.copy()
            elif action_source == "command":
                action_row = np.concatenate(
                    [
                        np.asarray(command["arm_joint_position"], dtype=np.float32).reshape(-1),
                        np.asarray(command["gripper_pos"], dtype=np.float32).reshape(-1),
                    ]
                )
            else:
                raise ValueError(
                    f"Unsupported action_source={action_source!r}; expected one of {ACTION_SOURCES}"
                )
        except KeyError as error:
            raise ValueError(
                f"{data_path}: frame {frame_index} is missing {error.args[0]!r}"
            ) from error
        if state_row.shape != (STATE_DIM,) or action_row.shape != (ACTION_DIM,):
            raise ValueError(
                f"{data_path}: frame {frame_index}: state={state_row.shape}, "
                f"action={action_row.shape}; both must be (7,)"
            )
        state_rows.append(state_row)
        action_rows.append(action_row)
    state = np.stack(state_rows).astype(np.float32, copy=False)
    action = np.stack(action_rows).astype(np.float32, copy=False)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{data_path}: state/action contains NaN or Inf")
    return state, action


def _load_episode(path: Path, target_fps: float, action_source: str) -> Episode:
    data_path = path / "data.pkl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing {data_path}")
    with data_path.open("rb") as stream:
        payload: dict[str, Any] = pickle.load(stream)

    required = {"timestamps", "observations", "actions"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{data_path}: missing keys {sorted(missing)}")
    observations = payload["observations"]
    actions = payload["actions"]
    count = len(payload["timestamps"])
    if count == 0 or len(observations) != count or len(actions) != count:
        raise ValueError(
            f"{data_path}: lengths timestamps={count}, observations={len(observations)}, "
            f"actions={len(actions)}"
        )
    robot_timestamps = _validate_timestamps(
        payload["timestamps"], f"{data_path}: robot", count
    )
    state, action = _load_rows(data_path, observations, actions, action_source)

    raw_camera_timestamps = payload.get("camera_timestamps")
    raw_frame_source_timestamps = payload.get("camera_frame_source_timestamps")
    cameras: dict[str, CameraStream] = {}
    for camera, filename in CAMERAS.items():
        video_path = path / filename
        width, height, frames, source_fps = _video_properties(video_path)
        if raw_camera_timestamps is None:
            # Schema v1 stored one video frame per robot sample.
            if frames != count:
                raise ValueError(
                    f"{video_path}: legacy episode has {frames} frames but {count} robot samples"
                )
            timestamps = robot_timestamps.copy()
        else:
            if camera not in raw_camera_timestamps:
                raise ValueError(f"{data_path}: camera_timestamps is missing {camera!r}")
            timestamps = _validate_timestamps(
                raw_camera_timestamps[camera], f"{data_path}: {camera}", frames
            )
        if raw_frame_source_timestamps is None:
            capture_timestamps = timestamps
        else:
            if camera not in raw_frame_source_timestamps:
                raise ValueError(
                    f"{data_path}: camera_frame_source_timestamps is missing {camera!r}"
                )
            capture_timestamps = _validate_timestamps(
                raw_frame_source_timestamps[camera],
                f"{data_path}: {camera} frame source",
                frames,
            )
        cameras[camera] = CameraStream(
            path=video_path,
            timestamps=timestamps,
            capture_timestamps=capture_timestamps,
            width=width,
            height=height,
            source_fps=source_fps,
        )

    overlap_start = max(
        robot_timestamps[0], *(stream.timestamps[0] for stream in cameras.values())
    )
    overlap_end = min(
        robot_timestamps[-1], *(stream.timestamps[-1] for stream in cameras.values())
    )
    if overlap_end < overlap_start:
        raise ValueError(
            f"{path}: streams have no shared time range: {overlap_start:.6f}..{overlap_end:.6f}"
        )
    target_count = int(np.floor((overlap_end - overlap_start) * target_fps + 1e-9)) + 1
    if target_count < 2:
        raise ValueError(f"{path}: shared stream duration is too short for {target_fps:g} Hz")
    target_source_time = overlap_start + np.arange(target_count, dtype=np.float64) / target_fps
    output_timestamps = np.arange(target_count, dtype=np.float64) / target_fps

    robot_indices = _nearest_indices(robot_timestamps, target_source_time)
    camera_indices: dict[str, np.ndarray] = {}
    alignment_ms = {
        "robot": np.abs(robot_timestamps[robot_indices] - target_source_time) * 1000.0
    }
    for camera, stream in cameras.items():
        indices = _nearest_indices(stream.timestamps, target_source_time)
        camera_indices[camera] = indices
        alignment_ms[camera] = (
            np.abs(stream.capture_timestamps[indices] - target_source_time) * 1000.0
        )

    return Episode(
        source_dir=path,
        timestamps=output_timestamps.astype(np.float32),
        state=state[robot_indices],
        action=action[robot_indices],
        cameras=cameras,
        camera_indices=camera_indices,
        alignment_ms=alignment_ms,
    )


def _features(cameras: dict[str, CameraStream], fps: float) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for camera, stream in cameras.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [stream.height, stream.width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": stream.height,
                "video.width": stream.width,
                "video.channels": 3,
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": ["state"]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ["action"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def _write_episode_parquet(
    path: Path, episode: Episode, episode_index: int, global_offset: int
) -> None:
    count = episode.length
    table = pa.table(
        {
            "observation.state": pa.array(
                episode.state.tolist(), type=pa.list_(pa.float32(), STATE_DIM)
            ),
            "action": pa.array(
                episode.action.tolist(), type=pa.list_(pa.float32(), ACTION_DIM)
            ),
            "timestamp": pa.array(episode.timestamps, type=pa.float32()),
            "frame_index": pa.array(np.arange(count, dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(
                np.full(count, episode_index, dtype=np.int64), type=pa.int64()
            ),
            "index": pa.array(
                np.arange(global_offset, global_offset + count, dtype=np.int64),
                type=pa.int64(),
            ),
            "task_index": pa.array(np.zeros(count, dtype=np.int64), type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_resampled_video(
    stream: CameraStream, indices: np.ndarray, destination: Path, target_fps: float
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode resampled H.264 videos")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{stream.width}x{stream.height}", "-r", str(target_fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    capture = cv2.VideoCapture(str(stream.path))
    if not capture.isOpened():
        process.kill()
        raise ValueError(f"Could not open {stream.path}")
    assert process.stdin is not None and process.stderr is not None
    source_index = -1
    frame: np.ndarray | None = None
    try:
        for requested in indices:
            requested_index = int(requested)
            while source_index < requested_index:
                ok, frame = capture.read()
                source_index += 1
                if not ok or frame is None:
                    raise ValueError(
                        f"{stream.path}: failed to decode source frame {source_index}"
                    )
            assert frame is not None
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        error_text = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {destination}: {error_text.strip()}")
        width, height, frames, fps = _video_properties(destination)
        if (width, height) != (stream.width, stream.height):
            raise RuntimeError(
                f"{destination}: encoded resolution {(width, height)} does not match "
                f"{(stream.width, stream.height)}"
            )
        if frames != len(indices) or not np.isclose(fps, target_fps, atol=1e-3):
            raise RuntimeError(
                f"{destination}: expected {len(indices)} frames at {target_fps:g} Hz, "
                f"got {frames} frames at {fps:g} Hz"
            )
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        if destination.exists():
            destination.unlink()
        raise
    finally:
        capture.release()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def convert(
    raw_root: Path,
    output_root: Path,
    dataset_name: str,
    task: str,
    target_fps: float,
    action_source: str,
    overwrite: bool,
) -> Path:
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    task = task.strip()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("--dataset-name must be one non-empty directory name")
    if not task:
        raise ValueError("--task must not be empty")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if action_source not in ACTION_SOURCES:
        raise ValueError(
            f"--action-source must be one of {ACTION_SOURCES}, got {action_source!r}"
        )

    # Support both legacy <raw_root>/<episode>/data.pkl and the newer
    # <raw_root>/<YYYYMMDD>/<episode>/data.pkl layout.
    source_dirs = sorted({data_path.parent for data_path in raw_root.rglob("data.pkl")})
    if not source_dirs:
        raise RuntimeError(f"No episode directories containing data.pkl under {raw_root}")
    episodes = [_load_episode(path, target_fps, action_source) for path in source_dirs]
    reference = episodes[0]
    for episode in episodes[1:]:
        for camera in CAMERAS:
            current = episode.cameras[camera]
            expected = reference.cameras[camera]
            if (current.width, current.height) != (expected.width, expected.height):
                raise ValueError(
                    f"{camera} resolution changed: {(expected.width, expected.height)} vs "
                    f"{(current.width, current.height)} in {episode.source_dir}"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / dataset_name
    if dataset_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {dataset_path} (pass --overwrite to replace it)"
        )
    staging_path = Path(tempfile.mkdtemp(prefix=f".{dataset_name}_", dir=output_root))
    try:
        total_frames = 0
        episode_lines: list[str] = []
        for episode_index, episode in enumerate(episodes):
            _write_episode_parquet(
                staging_path / f"data/chunk-000/episode_{episode_index:06d}.parquet",
                episode,
                episode_index,
                total_frames,
            )
            for camera in CAMERAS:
                destination = (
                    staging_path
                    / f"videos/chunk-000/observation.images.{camera}/episode_{episode_index:06d}.mp4"
                )
                _write_resampled_video(
                    episode.cameras[camera],
                    episode.camera_indices[camera],
                    destination,
                    target_fps,
                )
            episode_lines.append(
                json.dumps(
                    {"episode_index": episode_index, "tasks": [task], "length": episode.length}
                )
            )
            total_frames += episode.length
            alignment = ", ".join(
                f"{name} median/max={np.median(error):.2f}/{np.max(error):.2f}ms"
                for name, error in episode.alignment_ms.items()
            )
            print(
                f"[{episode_index + 1}/{len(episodes)}] {episode.source_dir.name}: "
                f"{episode.length} frames at {target_fps:g} Hz; {alignment}"
            )

        info = {
            "codebase_version": "v2.1",
            "robot_type": "piper",
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": len(episodes) * len(CAMERAS),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": target_fps,
            "action_source": action_source,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _features(reference.cameras, target_fps),
        }
        modality = {
            "video": {
                "global": {"original_key": "observation.images.global"},
                "hand": {"original_key": "observation.images.hand"},
            },
            "state": {
                "joints": {"original_key": "observation.state", "start": 0, "end": 6, "absolute": True, "dtype": "float32"},
                "gripper": {"original_key": "observation.state", "start": 6, "end": 7, "absolute": True, "dtype": "float32", "range": [0.0, 1.0]},
            },
            "action": {
                "joints": {"original_key": "action", "start": 0, "end": 6, "absolute": True, "dtype": "float32"},
                "gripper": {"original_key": "action", "start": 6, "end": 7, "absolute": True, "dtype": "float32", "range": [0.0, 1.0]},
            },
            "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        }
        embodiment = {
            "robot_name": "Piper",
            "robot_type": "piper",
            "record_frequency": target_fps,
            "body_controller_frequency": target_fps,
            "hand_controller_frequency": target_fps,
            "embodiment_tag": "new_embodiment",
            "action_source": action_source,
        }
        _write_json(staging_path / "meta/info.json", info)
        _write_json(staging_path / "meta/modality.json", modality)
        _write_json(staging_path / "meta/embodiment.json", embodiment)
        (staging_path / "meta/tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": task}) + "\n", encoding="utf-8"
        )
        (staging_path / "meta/episodes.jsonl").write_text(
            "\n".join(episode_lines) + "\n", encoding="utf-8"
        )
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        staging_path.replace(dataset_path)
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    print(f"Converted {len(episodes)} episodes / {total_frames} frames to {dataset_path}")
    print(f"Uniform output frequency: {target_fps:g} Hz")
    print("Alignment: nearest timestamp for robot state/action and both camera streams")
    if action_source == "observation":
        print("Action source: measured Piper feedback joints (6) + measured normalized gripper (1)")
    else:
        print("Action source: GELLO command target joints (6) + normalized gripper target (1)")
    return dataset_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/piper_demos"))
    parser.add_argument("--output-root", type=Path, default=Path("data/piper_lerobot"))
    parser.add_argument("--dataset-name", default="piper_pick_white_block")
    parser.add_argument("--task", default="Pick up white block and place it in the box.")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument(
        "--action-source",
        choices=ACTION_SOURCES,
        default="observation",
        help=(
            "LeRobot action source: 'observation' uses measured Piper feedback "
            "(default); 'command' uses GELLO /control/joint_states targets."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output after conversion succeeds")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert(
        raw_root=args.raw_root,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        task=args.task,
        target_fps=args.target_fps,
        action_source=args.action_source,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
