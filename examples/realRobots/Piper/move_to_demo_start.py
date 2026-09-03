#!/usr/bin/env python3
"""Move Piper slowly to the demonstrated task-start joint configuration."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
GRIPPER_NAME = "gripper"
GRIPPER_MAX_WIDTH_M = 0.1
DEFAULT_STATS = Path(__file__).parent / "eval_files" / "piper_demo_start_statistics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slowly interpolate Piper to the median first-frame pose of the "
            "recorded demonstrations, then close the external control gate."
        )
    )
    parser.add_argument("--stats-file", type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        "--gripper-normalized",
        type=float,
        default=None,
        help="Target gripper state (0=open, 1=closed; default: median from --stats-file).",
    )
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--joint-speed",
        type=float,
        default=0.12,
        help="Maximum interpolated joint speed in rad/s (default: 0.12).",
    )
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.8,
        help="Refuse startup when any joint is farther than this many radians.",
    )
    parser.add_argument("--feedback-timeout", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--settle-timeout", type=float, default=6.0)
    parser.add_argument("--joint-tolerance", type=float, default=0.06)
    parser.add_argument("--gripper-tolerance", type=float, default=0.12)
    return parser.parse_args()


def load_target(stats_path: Path, gripper_normalized: float | None) -> list[float]:
    with stats_path.open("r", encoding="utf-8") as stream:
        stats = json.load(stream)
    median = stats.get("median")
    if not isinstance(median, list) or len(median) != 7:
        raise ValueError(f"{stats_path} must contain a seven-element 'median' list")
    target_gripper = float(median[6]) if gripper_normalized is None else gripper_normalized
    target = [float(value) for value in median[:6]] + [target_gripper]
    if not all(math.isfinite(value) for value in target):
        raise ValueError("Target contains a non-finite value")
    if not 0.0 <= target[6] <= 1.0:
        raise ValueError("--gripper-normalized must be in [0, 1]")
    return target


class DemoStartMover(Node):
    def __init__(self) -> None:
        super().__init__("piper_demo_start_mover")
        self.feedback: dict[str, float] | None = None
        self.feedback_time = 0.0
        self.create_subscription(
            JointState, "/feedback/joint_states", self._feedback_callback, 10
        )
        self.gate_client = self.create_client(SetBool, "/control_enable")
        self.command_publisher = None

    def _feedback_callback(self, msg: JointState) -> None:
        if len(msg.position) < len(msg.name):
            return
        values = dict(zip(msg.name, msg.position))
        required = (*ARM_JOINTS, GRIPPER_NAME)
        if all(name in values and math.isfinite(values[name]) for name in required):
            self.feedback = {name: float(values[name]) for name in required}
            self.feedback_time = time.monotonic()

    def wait_for_feedback(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.feedback is not None:
                return
        raise RuntimeError("Timed out waiting for complete /feedback/joint_states")

    def assert_no_other_command_publisher(self) -> None:
        publishers = [
            info
            for info in self.get_publishers_info_by_topic("/control/joint_states")
            if info.node_name != self.get_name()
            or info.node_namespace != self.get_namespace()
        ]
        if publishers:
            names = ", ".join(
                f"{info.node_namespace.rstrip('/')}/{info.node_name}" for info in publishers
            )
            raise RuntimeError(
                "Another /control/joint_states publisher is active: " + names
            )

    def create_command_publisher(self) -> None:
        self.command_publisher = self.create_publisher(
            JointState, "/control/joint_states", 1
        )

    def set_gate(self, enabled: bool, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not self.gate_client.wait_for_service(timeout_sec=0.2):
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for /control_enable")
        request = SetBool.Request()
        request.data = enabled
        future = self.gate_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError("Timed out calling /control_enable")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"/control_enable rejected request: {response.message}")

    def current_normalized_state(self) -> list[float]:
        if self.feedback is None:
            raise RuntimeError("Piper feedback is unavailable")
        gripper_width = min(
            GRIPPER_MAX_WIDTH_M, max(0.0, self.feedback[GRIPPER_NAME])
        )
        gripper = 1.0 - gripper_width / GRIPPER_MAX_WIDTH_M
        return [self.feedback[name] for name in ARM_JOINTS] + [gripper]

    def feedback_is_fresh(self, timeout: float) -> bool:
        return time.monotonic() - self.feedback_time <= timeout

    def publish_normalized_state(self, state: list[float]) -> None:
        if self.command_publisher is None:
            raise RuntimeError("Command publisher has not been created")
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [*ARM_JOINTS, GRIPPER_NAME]
        gripper_width = (1.0 - state[6]) * GRIPPER_MAX_WIDTH_M
        msg.position = [*state[:6], gripper_width]
        msg.effort = [0.0] * 6 + [0.5]
        self.command_publisher.publish(msg)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "rate_hz",
        "joint_speed",
        "min_duration",
        "max_joint_delta",
        "feedback_timeout",
        "startup_timeout",
        "settle_timeout",
        "joint_tolerance",
        "gripper_tolerance",
    )
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def move_to_target(node: DemoStartMover, target: list[float], args: argparse.Namespace) -> None:
    start = node.current_normalized_state()
    joint_deltas = [abs(target[i] - start[i]) for i in range(6)]
    max_delta = max(joint_deltas)
    if max_delta > args.max_joint_delta:
        raise RuntimeError(
            f"Initial pose is too far from the demonstrated start pose: "
            f"max joint delta={max_delta:.3f} rad > {args.max_joint_delta:.3f} rad. "
            "Move Piper closer manually before retrying."
        )

    duration = max(args.min_duration, max_delta / args.joint_speed)
    steps = max(2, math.ceil(duration * args.rate_hz))
    print("Initial measured state:", " ".join(f"{value:.4f}" for value in start))
    print("Target demo-start state:", " ".join(f"{value:.4f}" for value in target))
    print(f"Moving over {duration:.2f}s at {args.rate_hz:.1f} Hz...")

    period = 1.0 / args.rate_hz
    next_tick = time.monotonic()
    for step in range(1, steps + 1):
        rclpy.spin_once(node, timeout_sec=0.0)
        if not node.feedback_is_fresh(args.feedback_timeout):
            raise RuntimeError("Piper feedback became stale during startup motion")
        ratio = step / steps
        command = [
            start[index] + ratio * (target[index] - start[index])
            for index in range(7)
        ]
        node.publish_normalized_state(command)
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))

    deadline = time.monotonic() + args.settle_timeout
    while rclpy.ok() and time.monotonic() < deadline:
        node.publish_normalized_state(target)
        rclpy.spin_once(node, timeout_sec=period)
        if not node.feedback_is_fresh(args.feedback_timeout):
            raise RuntimeError("Piper feedback became stale while settling")
        current = node.current_normalized_state()
        joint_error = max(abs(current[i] - target[i]) for i in range(6))
        gripper_error = abs(current[6] - target[6])
        if (
            joint_error <= args.joint_tolerance
            and gripper_error <= args.gripper_tolerance
        ):
            print(
                f"Demo-start pose reached (max joint error={joint_error:.4f} rad, "
                f"gripper={current[6]:.3f}, error={gripper_error:.3f}; 0=open)."
            )
            return
    current = node.current_normalized_state()
    joint_error = max(abs(current[i] - target[i]) for i in range(6))
    gripper_error = abs(current[6] - target[6])
    raise RuntimeError(
        f"Piper did not reach the start pose within {args.settle_timeout:.1f}s "
        f"(max joint error={joint_error:.3f} rad, "
        f"gripper error={gripper_error:.3f})"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    target = load_target(args.stats_file, args.gripper_normalized)
    rclpy.init()
    node = DemoStartMover()
    gate_may_be_open = False
    try:
        node.wait_for_feedback(args.startup_timeout)
        node.assert_no_other_command_publisher()
        node.create_command_publisher()
        # Establish a known-safe state before temporarily granting control.
        node.set_gate(False)
        node.assert_no_other_command_publisher()
        # Set this before the service call so a late response after a timeout
        # still triggers a best-effort close in finally.
        gate_may_be_open = True
        node.set_gate(True)
        move_to_target(node, target, args)
    finally:
        if gate_may_be_open and rclpy.ok():
            try:
                node.set_gate(False)
                print("External control gate CLOSED.")
            except Exception as error:  # pragma: no cover - hardware failure path
                print(f"ERROR: failed to close /control_enable: {error}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
