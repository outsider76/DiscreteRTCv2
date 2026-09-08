#!/usr/bin/env python3
"""Record GELLO-controlled Piper demonstrations in episode format.

Each saved episode has the same layout as bspline-policy's ``EpisodeWriter``::

    <output_dir>/<YYYYMMDD>/<timestamp>/
      data.pkl
      hand_image.mp4
      global_image.mp4

The raw streams are intentionally captured on independent clocks: robot
state/action at 100 Hz, the Orbbec hand camera at 30 Hz, and the OAK global
camera at 30 Hz by default.  When an episode is saved, each camera video is
resampled onto its configured constant-rate clock over the robot time span.
Missing camera frames are filled with the nearest captured frame so MP4 playback
duration remains faithful to wall time.  ``data.pkl`` stores one presentation
timestamp and one original capture timestamp per encoded frame.  All timestamps
use the first recorded robot sample as their common origin.  The converter later
aligns these streams at its requested training frequency.

This node only observes ROS topics.  It never publishes robot commands.
"""

from __future__ import annotations

import argparse
import pickle
import select
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState

try:
    from agx_arm_msgs.msg import GripperStatus
except ImportError:
    GripperStatus = None


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT_NAME = "gripper"
ALL_JOINT_NAMES = (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME)

# Piper modified-DH parameters: (d, a, alpha, theta_offset).  The action EE
# pose uses link6 / gripper-base as the endpoint and XYZW quaternion ordering.
PIPER_MDH = np.asarray(
    [
        (0.123, 0.0, 0.0, 0.0),
        (0.0, 0.0, -np.pi / 2.0, -3.0058060377846343),
        (0.0, 0.28503, 0.0, -1.793849405199772),
        (0.25075, -0.02198, np.pi / 2.0, 0.0),
        (0.0, 0.0, -np.pi / 2.0, 0.0),
        (0.091, 0.0, np.pi / 2.0, 0.0),
    ],
    dtype=np.float64,
)


def _message_timestamp(message: Any, fallback: float) -> float:
    """Return a ROS header timestamp, falling back when a driver leaves it zero."""
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return fallback
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return value if value > 0.0 and np.isfinite(value) else fallback


def _normalized_gripper(width: float, max_width: float) -> float:
    """Convert AGX gripper width to GELLO convention: 0=open, 1=closed."""
    return float(np.clip(1.0 - float(width) / max_width, 0.0, 1.0))


def _rotation_matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized XYZW quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif axis == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Could not convert FK rotation to a finite quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def piper_forward_kinematics(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return action-target EE position and XYZW quaternion from six joints."""
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError(f"Piper FK expects six finite joints, got {joints}")
    transform = np.eye(4, dtype=np.float64)
    for joint, (d, a, alpha, offset) in zip(joints, PIPER_MDH):
        theta = float(joint + offset)
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        link = np.asarray(
            [
                [ct, -st, 0.0, a],
                [ca * st, ca * ct, -sa, -sa * d],
                [sa * st, sa * ct, ca, ca * d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transform = transform @ link
    return transform[:3, 3].copy(), _rotation_matrix_to_xyzw(transform[:3, :3])


def image_message_to_rgb(message: Image | CompressedImage) -> np.ndarray:
    """Decode common ROS Image encodings without cv_bridge.

    Avoiding cv_bridge is intentional: the local GELLO environment currently
    uses NumPy 2 while the Jazzy cv_bridge extension was built with NumPy 1.
    """
    if isinstance(message, CompressedImage):
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError("Compressed image payload is empty")
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                f"OpenCV could not decode compressed image format {message.format!r}"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    encoding = message.encoding.lower()
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}, step={step}")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_bytes = height * step
    if raw.size < required_bytes:
        raise ValueError(f"Image has {raw.size} bytes, expected at least {required_bytes}")
    rows = raw[:required_bytes].reshape(height, step)

    if encoding in {"rgb8", "bgr8", "rgba8", "bgra8", "8uc3", "8uc4"}:
        channels = 4 if encoding in {"rgba8", "bgra8", "8uc4"} else 3
        active_bytes = width * channels
        if step < active_bytes:
            raise ValueError(f"Image step {step} is smaller than row size {active_bytes}")
        image = rows[:, :active_bytes].reshape(height, width, channels)
        if encoding in {"bgr8", "8uc3"}:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif encoding in {"rgba8"}:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif encoding in {"bgra8", "8uc4"}:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return np.ascontiguousarray(image, dtype=np.uint8)

    if encoding in {"mono8", "8uc1"}:
        if step < width:
            raise ValueError(f"Image step {step} is smaller than row size {width}")
        gray = rows[:, :width]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if encoding in {"yuv422", "yuyv", "yuy2", "yuv422_yuy2"}:
        active_bytes = width * 2
        packed = rows[:, :active_bytes].reshape(height, width, 2)
        return cv2.cvtColor(packed, cv2.COLOR_YUV2RGB_YUY2)

    if encoding in {"uyvy", "yuv422_uyvy"}:
        active_bytes = width * 2
        packed = rows[:, :active_bytes].reshape(height, width, 2)
        return cv2.cvtColor(packed, cv2.COLOR_YUV2RGB_UYVY)

    raise ValueError(f"Unsupported image encoding {message.encoding!r}; configure the camera to publish rgb8 or bgr8")


class StreamingMP4Writer:
    """Write RGB frames directly to MP4 without retaining an episode in RAM."""

    def __init__(self, path: Path, fps: float, first_frame: np.ndarray):
        self.path = path
        self.fps = float(fps)
        self.height, self.width = first_frame.shape[:2]
        self._process: Optional[subprocess.Popen] = None
        self._opencv_writer: Optional[cv2.VideoWriter] = None

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            self._open_with_opencv()

    def _open_with_opencv(self) -> None:
        # ``mp4v`` is available in the OpenCV build on the Piper workstation;
        # try it before H.264 so a missing H.264 encoder does not print errors.
        for codec in ("mp4v", "avc1"):
            if self.path.exists():
                self.path.unlink()
            writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                (self.width, self.height),
            )
            if writer.isOpened():
                self._opencv_writer = writer
                return
            writer.release()
        raise RuntimeError("Could not open an MP4 encoder. Install ffmpeg or an OpenCV build with MP4 support.")

    def write(self, rgb_frame: np.ndarray) -> None:
        frame = np.asarray(rgb_frame)
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ValueError(f"Frame shape changed for {self.path.name}: expected {expected_shape}, got {frame.shape}")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._process is not None:
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError(f"ffmpeg stopped while writing {self.path}") from exc
        else:
            assert self._opencv_writer is not None
            self._opencv_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._opencv_writer is not None:
            self._opencv_writer.release()
            self._opencv_writer = None
            return
        if self._process is None:
            return

        process = self._process
        self._process = None
        assert process.stdin is not None
        assert process.stderr is not None
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} (exit {return_code}): {stderr.strip()}")


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return monotonically increasing nearest-source indices for target times."""
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(source[right] - target) < np.abs(target - source[left])
    return np.where(choose_right, right, left).astype(np.int64)


def _video_properties(path: Path) -> tuple[int, int, int, float]:
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


def _resample_video_timeline(
    path: Path,
    source_timestamps: list[float],
    start_timestamp: float,
    end_timestamp: float,
    target_fps: float,
) -> tuple[list[float], list[float], dict[str, float]]:
    """Atomically replace a sparse CFR video with a wall-time-correct CFR video.

    The first returned list is the encoded frame presentation clock.  The second
    records the actual camera acquisition time selected for every encoded frame.
    Keeping both clocks lets downstream conversion distinguish a fresh image
    from a duplicated image while retaining ordinary constant-rate MP4 files.
    """
    timestamps = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        raise ValueError(f"{path}: cannot resample an empty camera stream")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{path}: source timestamps must be finite and non-decreasing")
    if not np.isfinite(start_timestamp) or not np.isfinite(end_timestamp):
        raise ValueError(f"{path}: robot timeline endpoints must be finite")
    if end_timestamp < start_timestamp:
        raise ValueError(f"{path}: robot timeline ends before it starts")

    width, height, source_frames, _ = _video_properties(path)
    if source_frames != timestamps.size:
        raise ValueError(
            f"{path}: video has {source_frames} frames but there are "
            f"{timestamps.size} camera timestamps"
        )

    target_count = int(
        np.floor((end_timestamp - start_timestamp) * target_fps + 1e-9)
    ) + 1
    target_timestamps = (
        start_timestamp + np.arange(target_count, dtype=np.float64) / target_fps
    )
    indices = _nearest_indices(timestamps, target_timestamps)
    selected_source_timestamps = timestamps[indices]
    alignment_ms = np.abs(selected_source_timestamps - target_timestamps) * 1000.0

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to finalize timestamp-aligned camera videos")
    temporary_path = path.with_name(f".{path.stem}_timestamp_aligned.mp4")
    if temporary_path.exists():
        temporary_path.unlink()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(target_fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        process.kill()
        process.wait()
        raise ValueError(f"Could not open {path} for timestamp resampling")

    assert process.stdin is not None and process.stderr is not None
    source_index = -1
    frame: Optional[np.ndarray] = None
    try:
        for requested in indices:
            requested_index = int(requested)
            while source_index < requested_index:
                ok, frame = capture.read()
                source_index += 1
                if not ok or frame is None:
                    raise ValueError(
                        f"{path}: failed to decode source frame {source_index}"
                    )
            assert frame is not None
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        error_text = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed while finalizing {path} (exit {return_code}): "
                f"{error_text.strip()}"
            )
        output_width, output_height, output_frames, output_fps = _video_properties(
            temporary_path
        )
        if (output_width, output_height) != (width, height):
            raise RuntimeError(
                f"{temporary_path}: resolution changed from {(width, height)} to "
                f"{(output_width, output_height)}"
            )
        if output_frames != target_count or not np.isclose(
            output_fps, target_fps, atol=1e-3
        ):
            raise RuntimeError(
                f"{temporary_path}: expected {target_count} frames at {target_fps:g} Hz, "
                f"got {output_frames} frames at {output_fps:g} Hz"
            )
        temporary_path.replace(path)
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        capture.release()

    stats = {
        "captured_frames": int(source_frames),
        "encoded_frames": int(target_count),
        "median_capture_alignment_ms": float(np.median(alignment_ms)),
        "max_capture_alignment_ms": float(np.max(alignment_ms)),
    }
    return (
        target_timestamps.tolist(),
        selected_source_timestamps.tolist(),
        stats,
    )


class EpisodeRecorder:
    """Stream one episode to a staging directory and atomically publish it on save."""

    CAMERA_KEYS = ("hand", "global")

    def __init__(
        self,
        output_dir: Path,
        joint_fps: float,
        hand_fps: float,
        global_fps: float,
        global_camera_type: str = "oak",
    ):
        self.output_dir = output_dir.expanduser().resolve()
        self.stream_fps = {
            "robot": float(joint_fps),
            "hand": float(hand_fps),
            "global": float(global_fps),
        }
        self.camera_sources = {
            "hand": "orbbec",
            "global": str(global_camera_type),
        }
        created_at = datetime.now()
        self.episode_date = created_at.strftime("%Y%m%d")
        self.episode_name = created_at.strftime("%Y%m%dT%H%M%S%f")
        staging_root = self.output_dir.parent / f".{self.output_dir.name}_recording"
        staging_root.mkdir(parents=True, exist_ok=True)
        self.staging_dir = Path(tempfile.mkdtemp(prefix=f"{self.episode_name}_", dir=staging_root))
        self.timestamps: list[float] = []
        self.observations: list[dict[str, Any]] = []
        self.actions: list[dict[str, np.ndarray]] = []
        # Source timestamps count actual camera callbacks.  camera_timestamps is
        # populated during save and corresponds one-to-one with finalized MP4
        # frames on a regular presentation clock.
        self.camera_source_timestamps: dict[str, list[float]] = {
            camera: [] for camera in self.CAMERA_KEYS
        }
        self.camera_timestamps: dict[str, list[float]] = {
            camera: [] for camera in self.CAMERA_KEYS
        }
        self.camera_frame_source_timestamps: dict[str, list[float]] = {
            camera: [] for camera in self.CAMERA_KEYS
        }
        self.camera_resampling: dict[str, dict[str, float]] = {}
        self.video_writers: dict[str, StreamingMP4Writer] = {}
        self._closed = False

    def __len__(self) -> int:
        return len(self.timestamps)

    def record_robot(
        self,
        timestamp: float,
        observation: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
    ) -> None:
        if self._closed:
            raise RuntimeError("Episode recorder is already closed")
        stored_observation = {
            key: np.asarray(value).copy() for key, value in observation.items()
        }
        # Preserve the old observation shape for readers that expect image keys;
        # frames themselves live in the independently timed MP4 streams.
        stored_observation["hand_image"] = None
        stored_observation["global_image"] = None
        self.timestamps.append(float(timestamp))
        self.observations.append(stored_observation)
        self.actions.append({key: np.asarray(value).copy() for key, value in action.items()})

    def record_camera(self, camera: str, timestamp: float, rgb_frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Episode recorder is already closed")
        if camera not in self.CAMERA_KEYS:
            raise ValueError(f"Unknown camera {camera!r}")
        frame = np.asarray(rgb_frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"{camera} frame must be HxWx3 uint8 RGB, got {frame.shape} {frame.dtype}"
            )
        writer = self.video_writers.get(camera)
        if writer is None:
            writer = StreamingMP4Writer(
                self.staging_dir / f"{camera}_image.mp4",
                self.stream_fps[camera],
                frame,
            )
            self.video_writers[camera] = writer
        writer.write(frame)
        self.camera_source_timestamps[camera].append(float(timestamp))

    def _close_videos(self, suppress_errors: bool = False) -> None:
        errors = []
        for writer in self.video_writers.values():
            try:
                writer.close()
            except Exception as exc:  # preserve all encoder errors until every writer is closed
                errors.append(exc)
        self.video_writers.clear()
        if errors and not suppress_errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def save(self) -> Path:
        if self._closed:
            raise RuntimeError("Episode recorder is already closed")
        if not self.timestamps:
            raise RuntimeError("Cannot save an empty episode")

        missing_cameras = [
            camera
            for camera in self.CAMERA_KEYS
            if not self.camera_source_timestamps[camera]
        ]
        if missing_cameras:
            raise RuntimeError(f"Cannot save episode without camera frames: {missing_cameras}")

        self._close_videos()
        print("\nFinalizing timestamp-aligned camera videos...")
        for camera in self.CAMERA_KEYS:
            print(
                f"  {camera}: {len(self.camera_source_timestamps[camera])} captured "
                f"frames -> {self.stream_fps[camera]:g} Hz MP4",
                flush=True,
            )
            (
                self.camera_timestamps[camera],
                self.camera_frame_source_timestamps[camera],
                self.camera_resampling[camera],
            ) = _resample_video_timeline(
                self.staging_dir / f"{camera}_image.mp4",
                self.camera_source_timestamps[camera],
                self.timestamps[0],
                self.timestamps[-1],
                self.stream_fps[camera],
            )
        origin = self.timestamps[0]
        robot_timestamps = [timestamp - origin for timestamp in self.timestamps]
        camera_timestamps = {
            camera: [timestamp - origin for timestamp in timestamps]
            for camera, timestamps in self.camera_timestamps.items()
        }
        camera_source_timestamps = {
            camera: [timestamp - origin for timestamp in timestamps]
            for camera, timestamps in self.camera_source_timestamps.items()
        }
        camera_frame_source_timestamps = {
            camera: [timestamp - origin for timestamp in timestamps]
            for camera, timestamps in self.camera_frame_source_timestamps.items()
        }
        with (self.staging_dir / "data.pkl").open("wb") as file:
            pickle.dump(
                {
                    "schema_version": 3,
                    "timestamps": robot_timestamps,
                    "observations": self.observations,
                    "actions": self.actions,
                    "camera_timestamps": camera_timestamps,
                    "camera_source_timestamps": camera_source_timestamps,
                    "camera_frame_source_timestamps": camera_frame_source_timestamps,
                    "camera_resampling": self.camera_resampling,
                    "stream_fps": self.stream_fps,
                    "camera_sources": self.camera_sources,
                    "kinematics": {
                        "observation_ee_pose": "measured /feedback/tcp_pose",
                        "action_ee_pose": "Piper MDH FK from action arm_joint_position",
                        "quaternion_order": "xyzw",
                    },
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            file.flush()

        date_dir = self.output_dir / self.episode_date
        date_dir.mkdir(parents=True, exist_ok=True)
        episode_dir = date_dir / self.episode_name
        if episode_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {episode_dir}")
        self.staging_dir.replace(episode_dir)
        self._closed = True
        return episode_dir

    def discard(self) -> None:
        if self._closed:
            return
        self._close_videos(suppress_errors=True)
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self._closed = True


@dataclass(frozen=True)
class CollectorSnapshot:
    timestamp: float
    observation: dict[str, np.ndarray]
    action: dict[str, np.ndarray]


class PiperCollectorNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("piper_data_collector")
        self.args = args
        self._lock = RLock()
        self._joint_positions: Optional[np.ndarray] = None
        self._joint_timestamp: Optional[float] = None
        self._joint_velocities = np.zeros(7, dtype=np.float64)
        self._joint_efforts = np.zeros(7, dtype=np.float64)
        self._tcp_pose: Optional[np.ndarray] = None
        self._command: Optional[np.ndarray] = None
        self._hand_image: Optional[np.ndarray] = None
        self._global_image: Optional[np.ndarray] = None
        self._recorder: Optional[EpisodeRecorder] = None
        self._next_robot_record_timestamp: Optional[float] = None
        self._next_camera_record_timestamp: dict[str, Optional[float]] = {
            "hand": None,
            "global": None,
        }
        self._updated_at: dict[str, float] = {}
        self._image_errors_reported: set[str] = set()

        robot_qos = QoSProfile(depth=10)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(JointState, args.feedback_topic, self._feedback_callback, robot_qos)
        self.create_subscription(JointState, args.command_topic, self._command_callback, robot_qos)
        self.create_subscription(PoseStamped, args.tcp_pose_topic, self._tcp_callback, robot_qos)
        hand_image_type = (
            CompressedImage
            if args.hand_image_topic.rstrip("/").endswith("/compressed")
            else Image
        )
        global_image_type = (
            CompressedImage
            if args.global_image_topic.rstrip("/").endswith("/compressed")
            else Image
        )
        self.create_subscription(
            hand_image_type,
            args.hand_image_topic,
            lambda message: self._image_callback("hand", message),
            image_qos,
        )
        self.create_subscription(
            global_image_type,
            args.global_image_topic,
            lambda message: self._image_callback("global", message),
            image_qos,
        )

        if GripperStatus is not None and args.gripper_feedback_topic:
            self.create_subscription(
                GripperStatus,
                args.gripper_feedback_topic,
                self._gripper_callback,
                robot_qos,
            )
        elif GripperStatus is None:
            self.get_logger().warning("agx_arm_msgs is unavailable; using gripper width from feedback/joint_states only")

    def _feedback_callback(self, message: JointState) -> None:
        positions_by_name = {
            name: message.position[index] for index, name in enumerate(message.name) if index < len(message.position)
        }
        if not all(name in positions_by_name for name in ALL_JOINT_NAMES):
            return

        positions = np.asarray([positions_by_name[name] for name in ALL_JOINT_NAMES], dtype=np.float64)
        positions[-1] = _normalized_gripper(positions[-1], self.args.gripper_max_width)

        velocities_by_name = {
            name: message.velocity[index] for index, name in enumerate(message.name) if index < len(message.velocity)
        }
        efforts_by_name = {
            name: message.effort[index] for index, name in enumerate(message.name) if index < len(message.effort)
        }

        record_payload: Optional[
            tuple[EpisodeRecorder, float, dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = None
        with self._lock:
            self._joint_positions = positions
            self._joint_timestamp = _message_timestamp(
                message, self.get_clock().now().nanoseconds * 1e-9
            )
            for index, name in enumerate(ALL_JOINT_NAMES):
                if name in velocities_by_name:
                    velocity = float(velocities_by_name[name])
                    self._joint_velocities[index] = (
                        -velocity / self.args.gripper_max_width if name == GRIPPER_JOINT_NAME else velocity
                    )
                if name in efforts_by_name:
                    self._joint_efforts[index] = float(efforts_by_name[name])
            self._updated_at["robot feedback"] = time.monotonic()
            recorder = self._recorder
            if recorder is not None and self._tcp_pose is not None and self._command is not None:
                timestamp = self._joint_timestamp
                period = 1.0 / self.args.joint_fps
                if (
                    self._next_robot_record_timestamp is None
                    or timestamp + period * 1e-6 >= self._next_robot_record_timestamp
                ):
                    observation = {
                        "arm_joint_position": positions[:6].copy(),
                        "arm_joint_velocity": self._joint_velocities[:6].copy(),
                        "arm_joint_effort": self._joint_efforts[:6].copy(),
                        "arm_pos": self._tcp_pose[:3].copy(),
                        "arm_quat": self._tcp_pose[3:7].copy(),
                        "gripper_pos": positions[6:7].copy(),
                    }
                    action_ee_pos, action_ee_quat = piper_forward_kinematics(
                        self._command[:6]
                    )
                    action = {
                        "arm_joint_position": self._command[:6].copy(),
                        "gripper_pos": self._command[6:7].copy(),
                        "arm_pos": action_ee_pos,
                        "arm_quat": action_ee_quat,
                    }
                    record_payload = (recorder, timestamp, observation, action)
                    if self._next_robot_record_timestamp is None:
                        self._next_robot_record_timestamp = timestamp + period
                    else:
                        self._next_robot_record_timestamp += period
                        while self._next_robot_record_timestamp <= timestamp:
                            self._next_robot_record_timestamp += period
        if record_payload is not None:
            recorder, timestamp, observation, action = record_payload
            recorder.record_robot(timestamp, observation, action)

    def _command_callback(self, message: JointState) -> None:
        values_by_name = {
            name: message.position[index] for index, name in enumerate(message.name) if index < len(message.position)
        }
        if not all(name in values_by_name for name in ALL_JOINT_NAMES):
            return
        command = np.asarray([values_by_name[name] for name in ALL_JOINT_NAMES], dtype=np.float64)
        command[-1] = _normalized_gripper(command[-1], self.args.gripper_max_width)
        with self._lock:
            self._command = command
            self._updated_at["GELLO command"] = time.monotonic()

    def _tcp_callback(self, message: PoseStamped) -> None:
        pose = message.pose
        tcp_pose = np.asarray(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        with self._lock:
            self._tcp_pose = tcp_pose
            self._updated_at["TCP pose"] = time.monotonic()

    def _gripper_callback(self, message: Any) -> None:
        with self._lock:
            if self._joint_positions is not None:
                self._joint_positions[-1] = _normalized_gripper(message.width, self.args.gripper_max_width)
                self._joint_efforts[-1] = float(message.force)

    def _image_callback(self, camera: str, message: Image | CompressedImage) -> None:
        try:
            image = image_message_to_rgb(message)
            if self.args.image_width > 0 and self.args.image_height > 0:
                image = cv2.resize(
                    image,
                    (self.args.image_width, self.args.image_height),
                    interpolation=cv2.INTER_AREA,
                )
            image = np.ascontiguousarray(image, dtype=np.uint8)
        except Exception as exc:
            if camera not in self._image_errors_reported:
                self.get_logger().error(f"Could not decode {camera} camera image: {exc}")
                self._image_errors_reported.add(camera)
            return

        timestamp = _message_timestamp(
            message, self.get_clock().now().nanoseconds * 1e-9
        )
        with self._lock:
            if camera == "hand":
                self._hand_image = image
                label = "hand camera"
            else:
                self._global_image = image
                label = "global camera"
            self._updated_at[label] = time.monotonic()
            recorder = self._recorder
            camera_fps = (
                self.args.hand_fps if camera == "hand" else self.args.global_fps
            )
            period = 1.0 / camera_fps
            next_timestamp = self._next_camera_record_timestamp[camera]
            should_record = recorder is not None and (
                next_timestamp is None
                or timestamp + period * 1e-6 >= next_timestamp
            )
            if should_record:
                if next_timestamp is None:
                    self._next_camera_record_timestamp[camera] = timestamp + period
                else:
                    self._next_camera_record_timestamp[camera] += period
                    while self._next_camera_record_timestamp[camera] <= timestamp:
                        self._next_camera_record_timestamp[camera] += period
        if recorder is not None and should_record:
            recorder.record_camera(camera, timestamp, image)

    def set_recorder(self, recorder: Optional[EpisodeRecorder]) -> None:
        with self._lock:
            self._recorder = recorder
            self._next_robot_record_timestamp = None
            self._next_camera_record_timestamp = {"hand": None, "global": None}

    def readiness(self, max_age: float) -> list[str]:
        now = time.monotonic()
        with self._lock:
            available = {
                "robot feedback": self._joint_positions is not None,
                "TCP pose": self._tcp_pose is not None,
                "GELLO command": self._command is not None,
                "hand camera": self._hand_image is not None,
                "global camera": self._global_image is not None,
            }
            updated_at = self._updated_at.copy()

        problems = []
        for label, is_available in available.items():
            if not is_available:
                problems.append(f"{label}: missing")
                continue
            age = now - updated_at.get(label, 0.0)
            if age > max_age:
                problems.append(f"{label}: stale ({age:.1f}s)")
        return problems

    def snapshot(self, max_age: float) -> tuple[Optional[CollectorSnapshot], list[str]]:
        problems = self.readiness(max_age)
        if problems:
            return None, problems

        with self._lock:
            assert self._joint_positions is not None
            assert self._joint_timestamp is not None
            assert self._tcp_pose is not None
            assert self._command is not None
            assert self._hand_image is not None
            assert self._global_image is not None
            joint_positions = self._joint_positions.copy()
            tcp_pose = self._tcp_pose.copy()
            command = self._command.copy()
            observation = {
                "hand_image": self._hand_image.copy(),
                "global_image": self._global_image.copy(),
                "arm_joint_position": joint_positions[:6],
                "arm_joint_velocity": self._joint_velocities[:6].copy(),
                "arm_joint_effort": self._joint_efforts[:6].copy(),
                "arm_pos": tcp_pose[:3],
                "arm_quat": tcp_pose[3:7],
                "gripper_pos": joint_positions[6:7],
            }
            action = {
                "arm_joint_position": command[:6],
                "gripper_pos": command[6:7],
            }
            timestamp = self._joint_timestamp
        return CollectorSnapshot(timestamp=timestamp, observation=observation, action=action), []



class Keyboard:
    """Non-blocking single-key input while preserving the terminal afterward."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._settings = None

    def __enter__(self) -> "Keyboard":
        if self.enabled:
            self._settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *_: object) -> None:
        if self.enabled and self._settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)

    def read(self) -> Optional[str]:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1).lower() if readable else None


def _show_preview(
    snapshot: CollectorSnapshot, global_camera_type: str
) -> Optional[str]:
    panels = []
    global_label = f"{global_camera_type.upper()} global"
    for label, key in (
        ("Orbbec hand", "hand_image"),
        (global_label, "global_image"),
    ):
        rgb = snapshot.observation[key]
        height = 360
        width = max(1, round(rgb.shape[1] * height / rgb.shape[0]))
        panel = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.putText(
            panel,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    cv2.imshow("Piper data collection", np.concatenate(panels, axis=1))
    key_code = cv2.waitKey(1) & 0xFF
    return chr(key_code).lower() if key_code != 255 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect GELLO-Piper demonstrations in bspline-policy episode format.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/piper_demos"))
    parser.add_argument("--joint-fps", type=float, default=100.0)
    parser.add_argument("--hand-fps", type=float, default=30.0, help="Orbbec MP4 rate")
    parser.add_argument(
        "--global-fps",
        type=float,
        default=30.0,
        help="Global-camera MP4 rate (default: 30).",
    )
    parser.add_argument(
        "--global-camera-type",
        choices=("oak", "realsense"),
        default="oak",
        help="Global camera identity stored in data.pkl metadata (default: oak).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        dest="joint_fps",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-data-age", type=float, default=1.0)
    parser.add_argument("--gripper-max-width", type=float, default=0.1)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--feedback-topic", default="/feedback/joint_states")
    parser.add_argument("--command-topic", default="/control/joint_states")
    parser.add_argument("--tcp-pose-topic", default="/feedback/tcp_pose")
    parser.add_argument("--gripper-feedback-topic", default="/feedback/gripper_status")
    parser.add_argument(
        "--hand-image-topic",
        default="/camera/color/image_raw/compressed",
        help="Orbbec Image or CompressedImage topic (default: compressed)",
    )
    parser.add_argument(
        "--global-image-topic",
        default="/global_camera/camera/color/image_raw/compressed",
        help="Global-camera Image or CompressedImage topic (default: compressed)",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start once all streams are ready (useful for non-interactive collection).",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=0.0,
        help="Automatically save after this many seconds; 0 disables the limit.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("joint_fps", "hand_fps", "global_fps"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_data_age <= 0:
        raise ValueError("--max-data-age must be positive")
    if args.gripper_max_width <= 0:
        raise ValueError("--gripper-max-width must be positive")
    if (args.image_width > 0) != (args.image_height > 0):
        raise ValueError("Set both --image-width and --image-height, or set both to 0")
    if args.episode_seconds < 0:
        raise ValueError("--episode-seconds cannot be negative")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()

    rclpy.init()
    node = PiperCollectorNode(args)
    recorder: Optional[EpisodeRecorder] = None
    recording_started_at = 0.0
    auto_started = False
    next_preview_time = 0.0
    next_status_time = 0.0
    last_status_length = 0
    should_exit = False

    print("\nPiper data collector started (this script does not publish robot commands)")
    print(f"Output directory: {args.output_dir}")
    print(f"Orbbec:   {args.hand_image_topic}")
    print(f"Global ({args.global_camera_type.upper()}): {args.global_image_topic}")
    print(
        f"Raw stream targets: robot={args.joint_fps:g} Hz, "
        f"Orbbec={args.hand_fps:g} Hz, "
        f"{args.global_camera_type.upper()}={args.global_fps:g} Hz"
    )
    print("Controls: [r] start  [s] save  [d] discard  [q] save and quit\n")
    if not sys.stdin.isatty() and not args.auto_start:
        print("stdin is not interactive; use --auto-start (usually with --episode-seconds).")

    def start_recording() -> Optional[EpisodeRecorder]:
        snapshot, problems = node.snapshot(args.max_data_age)
        if problems:
            print("\nCannot start yet: " + "; ".join(problems))
            return None
        assert snapshot is not None
        result = EpisodeRecorder(
            args.output_dir,
            joint_fps=args.joint_fps,
            hand_fps=args.hand_fps,
            global_fps=args.global_fps,
            global_camera_type=args.global_camera_type,
        )
        node.set_recorder(result)
        print(f"\nRecording started: {result.episode_name}")
        return result

    def save_recording(current: EpisodeRecorder) -> None:
        node.set_recorder(None)
        if len(current) == 0:
            current.discard()
            print("\nEmpty episode discarded")
            return
        duration = current.timestamps[-1] - current.timestamps[0] if len(current) > 1 else 0.0
        stream_counts = {
            "robot": len(current.timestamps),
            **{
                camera: len(timestamps)
                for camera, timestamps in current.camera_source_timestamps.items()
            },
        }
        stream_rates = {}
        for stream, timestamps in {
            "robot": current.timestamps,
            **current.camera_source_timestamps,
        }.items():
            stream_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
            stream_rates[stream] = (
                (len(timestamps) - 1) / stream_duration if stream_duration > 0 else 0.0
            )
        episode_dir = current.save()
        print(f"\nSaved robot {len(current)} samples / {duration:.1f} seconds: {episode_dir}")
        print(
            "Achieved capture rates: "
            + ", ".join(
                f"{stream}={stream_rates[stream]:.2f} Hz ({stream_counts[stream]} samples)"
                for stream in ("robot", "hand", "global")
            )
        )
        print(
            "Final MP4 streams: "
            + ", ".join(
                f"{camera}={len(current.camera_timestamps[camera])} frames "
                f"at {current.stream_fps[camera]:g} Hz "
                f"({current.camera_resampling[camera]['captured_frames']} captured)"
                for camera in current.CAMERA_KEYS
            )
        )

    try:
        with Keyboard() as keyboard:
            while rclpy.ok() and not should_exit:
                rclpy.spin_once(node, timeout_sec=0.005)
                now = time.monotonic()

                preview_key = None
                if args.preview and now >= next_preview_time:
                    preview_snapshot, _ = node.snapshot(args.max_data_age)
                    if preview_snapshot is not None:
                        preview_key = _show_preview(
                            preview_snapshot, args.global_camera_type
                        )
                    next_preview_time = now + 1.0 / 30.0
                key = preview_key or keyboard.read()

                if args.auto_start and not auto_started and recorder is None:
                    if not node.readiness(args.max_data_age):
                        recorder = start_recording()
                        if recorder is not None:
                            auto_started = True
                            recording_started_at = now

                if key == "r":
                    if recorder is None:
                        recorder = start_recording()
                        if recorder is not None:
                            recording_started_at = now
                    else:
                        print("\nAlready recording; press s to save or d to discard first.")
                elif key == "s":
                    if recorder is None:
                        print("\nNo episode is currently being recorded.")
                    else:
                        save_recording(recorder)
                        recorder = None
                elif key == "d":
                    if recorder is None:
                        print("\nNo episode is currently being recorded.")
                    else:
                        node.set_recorder(None)
                        frame_count = len(recorder)
                        recorder.discard()
                        recorder = None
                        print(f"\nDiscarded episode ({frame_count} frames)")
                elif key == "q":
                    if recorder is not None:
                        save_recording(recorder)
                        recorder = None
                    should_exit = True
                    continue

                if (
                    recorder is not None
                    and args.episode_seconds > 0
                    and now - recording_started_at >= args.episode_seconds
                ):
                    save_recording(recorder)
                    recorder = None
                    if args.auto_start:
                        should_exit = True

                if now >= next_status_time:
                    problems = node.readiness(args.max_data_age)
                    state = f"Recording: {len(recorder)} frames" if recorder is not None else "Ready"
                    if problems:
                        state = "Waiting: " + "; ".join(problems)
                    status = f"[{state}]"
                    padding = " " * max(0, last_status_length - len(status))
                    print(f"\r{status}{padding}", end="", flush=True)
                    last_status_length = len(status)
                    next_status_time = now + 1.0
    except KeyboardInterrupt:
        print("\nCtrl+C received.")
        if recorder is not None:
            save_recording(recorder)
            recorder = None
    finally:
        if recorder is not None:
            node.set_recorder(None)
            recorder.discard()
        if args.preview:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("\nCollector exited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
