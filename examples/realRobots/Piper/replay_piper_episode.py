#!/usr/bin/env python3
"""Replay one raw Piper demonstration on the physical robot.

By default this script only validates and summarizes ``data.pkl``.  Passing
``--execute`` publishes the recorded absolute GELLO actions using the original
relative timestamps.  Images and observations are not replayed.

Safety properties:
  * execution requires a live Piper feedback stream and an explicit prompt;
  * another /control/joint_states publisher causes an immediate refusal;
  * the robot must already be near the episode's first observed pose, unless
    the operator explicitly requests a slow ``--move-to-start``;
  * /control_enable is closed again on completion, Ctrl-C, or any exception.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Data inspection should also work in a plain Python environment.  Hardware
# execution is guarded in main() and requires these ROS imports to succeed.
ROS_IMPORT_ERROR: ImportError | None = None
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import SetBool
except ImportError as error:  # pragma: no cover - depends on the active shell
    ROS_IMPORT_ERROR = error
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    JointState = None  # type: ignore[assignment,misc]
    SetBool = None  # type: ignore[assignment,misc]

if ROS_IMPORT_ERROR is None:
    try:
        from agx_arm_msgs.msg import GripperStatus
    except ImportError:
        GripperStatus = None
else:
    GripperStatus = None


DEFAULT_EPISODE = Path(
    "/home/tams/DiscreteRTCv2/data/piper_demos/"
    "Pick_white_block/20260810T195158176835/data.pkl"
)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
GRIPPER_NAME = "gripper"
ALL_JOINTS = (*ARM_JOINTS, GRIPPER_NAME)


@dataclass(frozen=True)
class Episode:
    timestamps: np.ndarray
    observations: np.ndarray
    actions: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])


def _row(payload: dict[str, Any], arm_key: str, gripper_key: str, index: int) -> np.ndarray:
    try:
        arm = np.asarray(payload[arm_key], dtype=np.float64).reshape(-1)
        gripper = np.asarray(payload[gripper_key], dtype=np.float64).reshape(-1)
    except KeyError as error:
        raise ValueError(f"frame {index} is missing {error.args[0]!r}") from error
    row = np.concatenate((arm, gripper))
    if row.shape != (7,):
        raise ValueError(f"frame {index}: expected 6 arm joints + 1 gripper, got {row.shape}")
    if not np.isfinite(row).all():
        raise ValueError(f"frame {index}: joint target contains NaN or Inf")
    return row


def load_episode(path: Path) -> Episode:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: pickle root must be a dictionary")
    required = {"timestamps", "observations", "actions"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")

    timestamps = np.asarray(payload["timestamps"], dtype=np.float64).reshape(-1)
    raw_observations = payload["observations"]
    raw_actions = payload["actions"]
    count = timestamps.size
    if count < 2:
        raise ValueError("episode must contain at least two frames")
    if len(raw_observations) != count or len(raw_actions) != count:
        raise ValueError(
            f"length mismatch: timestamps={count}, observations={len(raw_observations)}, "
            f"actions={len(raw_actions)}"
        )
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be finite and non-decreasing")
    timestamps = timestamps - timestamps[0]
    if timestamps[-1] <= 0:
        raise ValueError("episode duration must be positive")

    observations = np.stack(
        [
            _row(item, "arm_joint_position", "gripper_pos", index)
            for index, item in enumerate(raw_observations)
        ]
    )
    actions = np.stack(
        [
            _row(item, "arm_joint_position", "gripper_pos", index)
            for index, item in enumerate(raw_actions)
        ]
    )
    for label, array in (("observation", observations), ("action", actions)):
        invalid = np.flatnonzero((array[:, 6] < 0.0) | (array[:, 6] > 1.0))
        if invalid.size:
            raise ValueError(
                f"{label} frame {int(invalid[0])}: normalized gripper must be in [0, 1]"
            )
    return Episode(timestamps=timestamps, observations=observations, actions=actions)


def validate_recorded_steps(episode: Episode, max_joint_step: float, max_gripper_step: float) -> None:
    changes = np.abs(np.diff(episode.actions, axis=0))
    arm_index = np.unravel_index(int(np.argmax(changes[:, :6])), changes[:, :6].shape)
    gripper_index = int(np.argmax(changes[:, 6]))
    largest_arm = float(changes[arm_index[0], arm_index[1]])
    largest_gripper = float(changes[gripper_index, 6])
    if largest_arm > max_joint_step:
        raise ValueError(
            f"recorded arm jump {largest_arm:.4f} rad at frame {arm_index[0] + 1} -> "
            f"{arm_index[0] + 2}, joint{arm_index[1] + 1}, exceeds "
            f"--max-recorded-joint-step={max_joint_step:.4f}"
        )
    if largest_gripper > max_gripper_step:
        raise ValueError(
            f"recorded gripper jump {largest_gripper:.4f} at frame {gripper_index + 1} -> "
            f"{gripper_index + 2}, exceeds --max-recorded-gripper-step={max_gripper_step:.4f}"
        )


def print_episode_summary(path: Path, episode: Episode, speed: float) -> None:
    intervals = np.diff(episode.timestamps)
    changes = np.abs(np.diff(episode.actions, axis=0))
    effective_hz = (len(episode.timestamps) - 1) / episode.duration
    print(f"Episode: {path.expanduser().resolve()}")
    print(f"Frames: {len(episode.timestamps)}")
    print(f"Recorded duration: {episode.duration:.6f} s")
    print(f"Effective sample rate: {effective_hz:.6f} Hz")
    print(
        "Recorded dt: "
        f"mean={intervals.mean() * 1000:.3f} ms, "
        f"median={np.median(intervals) * 1000:.3f} ms, "
        f"min={intervals.min() * 1000:.3f} ms, max={intervals.max() * 1000:.3f} ms"
    )
    print(f"Replay duration at --speed {speed:g}: {episode.duration / speed:.6f} s")
    print("First observation [q1..q6, normalized gripper]:", np.round(episode.observations[0], 5))
    print("First action      [q1..q6, normalized gripper]:", np.round(episode.actions[0], 5))
    print("Last action       [q1..q6, normalized gripper]:", np.round(episode.actions[-1], 5))
    print(
        f"Largest recorded arm step: {changes[:, :6].max():.5f} rad; "
        f"gripper step: {changes[:, 6].max():.5f}"
    )


class PiperReplayNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("piper_episode_replayer")
        self.args = args
        self.feedback: np.ndarray | None = None
        self.joint_feedback_time = 0.0
        self.create_subscription(JointState, args.feedback_topic, self._feedback_callback, 10)
        if GripperStatus is not None and args.gripper_feedback_topic:
            self.create_subscription(
                GripperStatus, args.gripper_feedback_topic, self._gripper_callback, 10
            )
        self.command_publisher = self.create_publisher(JointState, args.command_topic, 1)
        self.gate_client = self.create_client(SetBool, args.control_service)

    def _feedback_callback(self, message: JointState) -> None:
        values = {
            name: float(message.position[index])
            for index, name in enumerate(message.name)
            if index < len(message.position) and math.isfinite(message.position[index])
        }
        if not all(name in values for name in ALL_JOINTS):
            return
        state = np.asarray([values[name] for name in ALL_JOINTS], dtype=np.float64)
        width = float(np.clip(state[6], 0.0, self.args.gripper_max_width))
        state[6] = 1.0 - width / self.args.gripper_max_width
        self.feedback = state
        self.joint_feedback_time = time.monotonic()

    def _gripper_callback(self, message: Any) -> None:
        if self.feedback is None:
            return
        width = float(np.clip(message.width, 0.0, self.args.gripper_max_width))
        self.feedback[6] = 1.0 - width / self.args.gripper_max_width

    def wait_for_feedback(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            if self.feedback is not None and self.feedback_is_fresh():
                return
        raise RuntimeError(f"Timed out waiting for complete feedback on {self.args.feedback_topic}")

    def feedback_is_fresh(self) -> bool:
        return time.monotonic() - self.joint_feedback_time <= self.args.feedback_timeout

    def assert_fresh_feedback(self) -> None:
        rclpy.spin_once(self, timeout_sec=0.0)
        if self.feedback is None or not self.feedback_is_fresh():
            age = (
                math.inf
                if self.feedback is None
                else time.monotonic() - self.joint_feedback_time
            )
            raise RuntimeError(f"Piper feedback became stale (age={age:.3f}s)")

    def other_command_publishers(self) -> list[str]:
        return sorted(
            f"{info.node_namespace.rstrip('/')}/{info.node_name}"
            for info in self.get_publishers_info_by_topic(self.args.command_topic)
            if info.node_name != self.get_name()
        )

    def assert_no_other_command_publisher(self) -> None:
        publishers = self.other_command_publishers()
        if publishers:
            raise RuntimeError(
                f"Another publisher is active on {self.args.command_topic}: {publishers}. "
                "Stop GELLO, the VLA client, and every other controller first."
            )

    def set_gate(self, enabled: bool, timeout: float = 5.0) -> None:
        if not self.gate_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"Service unavailable: {self.args.control_service}")
        request = SetBool.Request()
        request.data = enabled
        future = self.gate_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Timed out calling {self.args.control_service}")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"{self.args.control_service} rejected request: {response.message}")
        self.get_logger().warning(f"External control gate {'OPEN' if enabled else 'CLOSED'}")

    def publish(self, normalized_target: np.ndarray) -> None:
        target = np.asarray(normalized_target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(f"invalid target: {target}")
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(ALL_JOINTS)
        gripper_width = (1.0 - float(target[6])) * self.args.gripper_max_width
        message.position = [*target[:6].tolist(), gripper_width]
        message.effort = [0.0] * 6 + [self.args.gripper_effort]
        self.command_publisher.publish(message)


def wait_until(node: PiperReplayNode, deadline: float) -> float:
    """Wait while spinning ROS and return scheduling lateness in seconds."""
    while rclpy.ok():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        rclpy.spin_once(node, timeout_sec=min(0.005, remaining))
        if not node.feedback_is_fresh():
            raise RuntimeError("Piper feedback became stale while waiting for the next replay sample")
    return max(0.0, time.monotonic() - deadline)


def check_start(node: PiperReplayNode, start: np.ndarray, args: argparse.Namespace) -> None:
    node.assert_fresh_feedback()
    assert node.feedback is not None
    joint_error = np.abs(node.feedback[:6] - start[:6])
    gripper_error = abs(float(node.feedback[6] - start[6]))
    if joint_error.max() > args.start_joint_tolerance or gripper_error > args.start_gripper_tolerance:
        raise RuntimeError(
            "Current robot is not close enough to the episode start: "
            f"max joint error={joint_error.max():.4f} rad "
            f"(limit {args.start_joint_tolerance:.4f}), gripper error={gripper_error:.4f} "
            f"(limit {args.start_gripper_tolerance:.4f}). Use --move-to-start only after "
            "checking that the interpolation path is collision-free."
        )


def move_to_start(node: PiperReplayNode, start: np.ndarray, args: argparse.Namespace) -> None:
    node.assert_fresh_feedback()
    assert node.feedback is not None
    current = node.feedback.copy()
    maximum_delta = float(np.max(np.abs(start[:6] - current[:6])))
    if maximum_delta > args.max_start_move:
        raise RuntimeError(
            f"Episode start is {maximum_delta:.3f} rad away, exceeding "
            f"--max-start-move={args.max_start_move:.3f}; move the robot closer manually."
        )
    duration = max(args.start_move_min_duration, maximum_delta / args.start_move_speed)
    steps = max(2, math.ceil(duration * args.start_move_rate))
    print(f"Moving to the first recorded observation over {duration:.2f}s ({steps} steps)...")
    begin = time.monotonic()
    for step in range(1, steps + 1):
        wait_until(node, begin + step / args.start_move_rate)
        ratio = step / steps
        node.publish(current + ratio * (start - current))

    settle_deadline = time.monotonic() + args.start_settle_timeout
    period = 1.0 / args.start_move_rate
    while time.monotonic() < settle_deadline:
        node.publish(start)
        rclpy.spin_once(node, timeout_sec=period)
        assert node.feedback is not None
        joint_error = float(np.max(np.abs(node.feedback[:6] - start[:6])))
        gripper_error = abs(float(node.feedback[6] - start[6]))
        if (
            joint_error <= args.start_joint_tolerance
            and gripper_error <= args.start_gripper_tolerance
        ):
            print(
                f"Episode start pose reached (joint error={joint_error:.4f} rad, "
                f"gripper error={gripper_error:.4f})."
            )
            return
        node.assert_fresh_feedback()
    check_start(node, start, args)


def replay(node: PiperReplayNode, episode: Episode, args: argparse.Namespace) -> None:
    begin = time.monotonic()
    late: list[float] = []
    for index, (timestamp, action) in enumerate(zip(episode.timestamps, episode.actions)):
        deadline = begin + float(timestamp) / args.speed
        late.append(wait_until(node, deadline))
        if late[-1] > args.max_schedule_lateness:
            raise RuntimeError(
                f"Replay scheduler is {late[-1] * 1000:.1f} ms late at frame {index + 1}, "
                f"exceeding --max-schedule-lateness={args.max_schedule_lateness * 1000:.1f} ms"
            )
        node.assert_fresh_feedback()
        node.publish(action)
        if index == 0 or (index + 1) % args.status_every == 0 or index + 1 == len(episode.actions):
            print(
                f"[REPLAY] {index + 1:04d}/{len(episode.actions):04d} "
                f"recorded_t={timestamp:.3f}s wall_t={(time.monotonic() - begin):.3f}s "
                f"late={late[-1] * 1000:.2f}ms"
            )

    hold_period = 1.0 / args.hold_rate
    hold_deadline = time.monotonic() + args.hold_seconds
    while time.monotonic() < hold_deadline:
        node.assert_fresh_feedback()
        node.publish(episode.actions[-1])
        wait_until(node, min(hold_deadline, time.monotonic() + hold_period))
    lateness_ms = np.asarray(late) * 1000.0
    print(
        f"Replay complete. Scheduling lateness: median={np.median(lateness_ms):.2f}ms, "
        f"p95={np.percentile(lateness_ms, 95):.2f}ms, max={lateness_ms.max():.2f}ms"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_EPISODE, help="Raw episode data.pkl")
    parser.add_argument("--execute", action="store_true", help="Actually move the physical Piper")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed multiplier; 1.0 preserves original timestamps",
    )
    parser.add_argument(
        "--move-to-start", action="store_true",
        help="Slowly move to the episode's first observed state before replay",
    )
    parser.add_argument("--start-joint-tolerance", type=float, default=0.08)
    parser.add_argument("--start-gripper-tolerance", type=float, default=0.15)
    parser.add_argument("--max-start-move", type=float, default=0.8)
    parser.add_argument("--start-move-speed", type=float, default=0.12, help="rad/s")
    parser.add_argument("--start-move-rate", type=float, default=20.0)
    parser.add_argument("--start-move-min-duration", type=float, default=3.0)
    parser.add_argument("--start-settle-timeout", type=float, default=6.0)
    parser.add_argument("--max-recorded-joint-step", type=float, default=0.12)
    parser.add_argument("--max-recorded-gripper-step", type=float, default=0.25)
    parser.add_argument("--feedback-topic", default="/feedback/joint_states")
    parser.add_argument("--gripper-feedback-topic", default="/feedback/gripper_status")
    parser.add_argument("--command-topic", default="/control/joint_states")
    parser.add_argument("--control-service", default="/control_enable")
    parser.add_argument("--gripper-max-width", type=float, default=0.1)
    parser.add_argument("--gripper-effort", type=float, default=0.5)
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument(
        "--max-schedule-lateness", type=float, default=0.1,
        help="Abort rather than burst commands after this scheduling delay in seconds",
    )
    parser.add_argument("--hold-seconds", type=float, default=0.25)
    parser.add_argument("--hold-rate", type=float, default=20.0)
    parser.add_argument("--status-every", type=int, default=30)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "speed", "start_joint_tolerance", "start_gripper_tolerance", "max_start_move",
        "start_move_speed", "start_move_rate", "start_move_min_duration",
        "start_settle_timeout", "max_schedule_lateness",
        "max_recorded_joint_step", "max_recorded_gripper_step", "gripper_max_width",
        "feedback_timeout", "startup_timeout", "hold_rate",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        raise ValueError("--hold-seconds must be non-negative")
    if args.status_every <= 0:
        raise ValueError("--status-every must be positive")


def confirm_execution(path: Path, episode: Episode, args: argparse.Namespace) -> None:
    print("\nDANGER: this will replay recorded commands on the physical Piper.")
    print(f"  episode: {path.expanduser().resolve()}")
    print(f"  frames: {len(episode.actions)}, replay duration: {episode.duration / args.speed:.2f}s")
    print(f"  move to recorded start first: {'yes' if args.move_to_start else 'no'}")
    print("Clear the workspace and keep the emergency stop within reach.")
    if not sys.stdin.isatty():
        raise RuntimeError("Execution confirmation requires an interactive terminal")
    if input("Type REPLAY to continue: ").strip() != "REPLAY":
        raise RuntimeError("Execution cancelled")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    episode = load_episode(args.data)
    validate_recorded_steps(
        episode, args.max_recorded_joint_step, args.max_recorded_gripper_step
    )
    print_episode_summary(args.data, episode, args.speed)
    if not args.execute:
        print("\nDRY-RUN only: no ROS node was created and no command was sent.")
        print("Add --execute to enable physical replay.")
        return 0

    if ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS 2 Python packages are not loaded. Run this through "
            "examples/realRobots/Piper/run_replay_piper_episode.sh"
        ) from ROS_IMPORT_ERROR
    confirm_execution(args.data, episode, args)
    assert rclpy is not None
    rclpy.init()
    node = PiperReplayNode(args)
    gate_may_be_open = False
    try:
        node.wait_for_feedback(args.startup_timeout)
        node.set_gate(False)
        # Allow ROS graph discovery to settle before checking for competing controllers.
        discovery_deadline = time.monotonic() + 0.5
        while time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.assert_no_other_command_publisher()
        if not args.move_to_start:
            check_start(node, episode.observations[0], args)
        gate_may_be_open = True
        node.set_gate(True)
        if args.move_to_start:
            move_to_start(node, episode.observations[0], args)
        node.assert_no_other_command_publisher()
        replay(node, episode, args)
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrupted by operator.")
        return 130
    finally:
        if gate_may_be_open and rclpy.ok():
            try:
                node.set_gate(False)
                print("External control gate CLOSED.")
            except Exception as error:
                print(f"CRITICAL: failed to close {args.control_service}: {error}", file=sys.stderr)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
