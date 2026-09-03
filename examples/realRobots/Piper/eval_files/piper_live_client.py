#!/usr/bin/env python3
"""Connect live Piper observations to the StarVLA policy server.

Dry-run is the default: predictions are printed but no command is published.
Pass ``--execute`` and complete the interactive confirmation to open the AGX
external-control gate and publish rate-limited absolute joint targets.

The six arm joints remain rate-limited. The gripper is controlled as a binary
open/close command: a normalized policy output above ``--gripper-threshold``
closes it fully; all other outputs open it fully.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
import websocket
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import SetBool

from deployment.model_server.tools import msgpack_numpy
from examples.realRobots.Piper.eval_files.piper_eval_recording import (
    ClientViewerRecorder,
    DEFAULT_VIEWER_DATA_DIR,
    make_session_id,
)
from examples.realRobots.Piper.collect_data import (
    ALL_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    image_message_to_rgb,
    _normalized_gripper,
)

try:
    from agx_arm_msgs.msg import GripperStatus
except ImportError:
    GripperStatus = None


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STATS = (
    REPO_ROOT
    / "results/Checkpoints/piper_pick_white_block_20260818_qwenpi_measured_50hz_h50/dataset_statistics.json"
)
DEFAULT_START_STATS = Path(__file__).resolve().with_name("piper_demo_start_statistics.json")
DEFAULT_TASK = "Pick up white block and place it in the box."
IMAGE_HW = (224, 224)
STATE_LABELS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")


def _binary_gripper_target(predicted: float, threshold: float) -> float:
    """Convert the normalized policy gripper output to open=0 or closed=1."""
    return 1.0 if float(predicted) > threshold else 0.0


class PolicyConnection:
    def __init__(self, host: str, port: int, timeout: float):
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)
        self._socket = websocket.create_connection(
            f"ws://{host}:{port}", timeout=timeout, enable_multithread=True
        )
        hello = self._socket.recv()
        if isinstance(hello, str):
            raise RuntimeError(f"Policy server returned text during handshake: {hello}")
        self.metadata = msgpack_numpy.unpackb(hello)
        self.action_chunk_size = int(self.metadata.get("action_chunk_size", 0))
        if self.action_chunk_size <= 0:
            raise RuntimeError(
                "Policy server metadata has no valid positive action_chunk_size: "
                f"{self.metadata.get('action_chunk_size')!r}"
            )
        self._next_request_id = 1
        self.request_id_prefix = "piper-sync"

    def predict(self, example: dict, unnorm_key: str) -> np.ndarray:
        request = {
            "type": "infer",
            "request_id": (
                f"{self.request_id_prefix}-request-{self._next_request_id}"
            ),
            "payload": {"examples": [example], "unnorm_key": unnorm_key},
        }
        self._next_request_id += 1
        self._socket.send_binary(msgpack_numpy.packb(request))
        response_raw = self._socket.recv()
        if isinstance(response_raw, str):
            raise RuntimeError(f"Policy server traceback:\n{response_raw}")
        response = msgpack_numpy.unpackb(response_raw)
        if response.get("status") != "ok":
            raise RuntimeError(f"Policy server error: {response.get('error', response)}")
        actions = np.asarray(response["data"]["actions"], dtype=np.float32)
        expected_shape = (self.action_chunk_size, 7)
        if actions.shape == (1, *expected_shape):
            actions = actions[0]
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise ValueError(f"Invalid policy action array: shape={actions.shape}, finite={np.isfinite(actions).all()}")
        return actions

    def close(self) -> None:
        self._socket.close()


class PiperLiveNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("piper_starvla_client")
        self.args = args
        self._lock = threading.RLock()
        self._joint_state: Optional[np.ndarray] = None
        self._global_image: Optional[np.ndarray] = None
        self._hand_image: Optional[np.ndarray] = None
        self._updated_at: dict[str, float] = {}
        self._image_errors: set[str] = set()

        robot_qos = QoSProfile(depth=10)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState, args.feedback_topic, self._feedback_callback, robot_qos)
        self.create_subscription(
            Image, args.global_image_topic, lambda msg: self._image_callback("global", msg), image_qos
        )
        self.create_subscription(
            Image, args.hand_image_topic, lambda msg: self._image_callback("hand", msg), image_qos
        )
        if GripperStatus is not None and args.gripper_feedback_topic:
            self.create_subscription(
                GripperStatus, args.gripper_feedback_topic, self._gripper_callback, robot_qos
            )

        self._command_publisher = self.create_publisher(JointState, args.command_topic, 1)
        self._gate_client = self.create_client(SetBool, args.control_service)

    def _feedback_callback(self, message: JointState) -> None:
        by_name = {
            name: message.position[index]
            for index, name in enumerate(message.name)
            if index < len(message.position)
        }
        if not all(name in by_name for name in ALL_JOINT_NAMES):
            return
        state = np.asarray([by_name[name] for name in ALL_JOINT_NAMES], dtype=np.float32)
        state[-1] = _normalized_gripper(state[-1], self.args.gripper_max_width)
        with self._lock:
            self._joint_state = state
            self._updated_at["robot"] = time.monotonic()

    def _gripper_callback(self, message: Any) -> None:
        with self._lock:
            if self._joint_state is not None:
                self._joint_state[-1] = _normalized_gripper(
                    message.width, self.args.gripper_max_width
                )
                self._updated_at["robot"] = time.monotonic()

    def _image_callback(self, name: str, message: Image) -> None:
        try:
            image = image_message_to_rgb(message)
            if image.shape[:2] != IMAGE_HW:
                image = cv2.resize(
                    image, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_AREA
                )
            image = np.ascontiguousarray(image, dtype=np.uint8)
        except Exception as exc:
            if name not in self._image_errors:
                self.get_logger().error(f"Could not decode {name} image: {exc}")
                self._image_errors.add(name)
            return
        with self._lock:
            if name == "global":
                self._global_image = image
            else:
                self._hand_image = image
            self._updated_at[name] = time.monotonic()

    def snapshot(self, max_age: float) -> tuple[Optional[tuple[list[np.ndarray], np.ndarray]], list[str]]:
        now = time.monotonic()
        with self._lock:
            values = {
                "robot": self._joint_state,
                "global": self._global_image,
                "hand": self._hand_image,
            }
            updated = dict(self._updated_at)
            problems = []
            for name, value in values.items():
                if value is None:
                    problems.append(f"{name}: missing")
                elif now - updated.get(name, 0.0) > max_age:
                    problems.append(f"{name}: stale {now - updated.get(name, 0.0):.2f}s")
            if problems:
                return None, problems
            assert self._joint_state is not None
            assert self._global_image is not None
            assert self._hand_image is not None
            return (
                [self._global_image.copy(), self._hand_image.copy()],
                self._joint_state.copy(),
            ), []

    def other_command_publishers(self) -> list[str]:
        publishers = self.get_publishers_info_by_topic(self.args.command_topic)
        own_name = self.get_name()
        return sorted(
            f"{info.node_namespace.rstrip('/')}/{info.node_name}"
            for info in publishers
            if info.node_name != own_name
        )

    def set_control_gate(self, enabled: bool, timeout: float = 3.0) -> None:
        if not self._gate_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"Service unavailable: {self.args.control_service}")
        request = SetBool.Request()
        request.data = enabled
        future = self._gate_client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise TimeoutError(f"Timed out calling {self.args.control_service}")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"Control gate request failed: {getattr(response, 'message', None)}")
        self.get_logger().warning(f"External control gate {'OPEN' if enabled else 'CLOSED'}")

    def publish_command(self, normalized_target: np.ndarray) -> None:
        target = np.asarray(normalized_target, dtype=np.float64)
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(ALL_JOINT_NAMES)
        target[-1] = (1.0 - np.clip(target[-1], 0.0, 1.0)) * self.args.gripper_max_width
        message.position = target.tolist()
        message.velocity = []
        message.effort = [0.0] * 6 + [self.args.gripper_effort]
        self._command_publisher.publish(message)


def _load_stats(path: Path, key: str) -> tuple[dict, dict]:
    with path.expanduser().resolve().open(encoding="utf-8") as file:
        all_stats = json.load(file)
    if key not in all_stats:
        raise KeyError(f"unnorm key {key!r} unavailable; choices={list(all_stats)}")
    return all_stats[key]["state"], all_stats[key]["action"]


def _normalize_state(state: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    result = state.astype(np.float32).copy()
    mask = high != low
    result[mask] = 2.0 * (result[mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    return result


def _state_outside_message(
    state: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    joint_margin: float,
    gripper_margin: float,
    range_name: str = "training",
    check_gripper: bool = True,
) -> Optional[str]:
    margins = np.asarray([joint_margin] * 6 + [gripper_margin], dtype=np.float32)
    mask = np.logical_or(state < low - margins, state > high + margins)
    if not check_gripper:
        mask[-1] = False
    if not mask.any():
        return None
    details = ", ".join(
        f"{STATE_LABELS[index]}={state[index]:.4f} not in "
        f"[{low[index]:.4f}, {high[index]:.4f}] (margin={margins[index]:.4f})"
        for index in np.flatnonzero(mask)
    )
    return f"Current state is outside the executable {range_name} range: " + details


def _action_violation_message(actions: np.ndarray, low: np.ndarray, high: np.ndarray) -> Optional[str]:
    mask = np.logical_or(actions < low[None, :], actions > high[None, :])
    if not mask.any():
        return None
    coordinates = np.argwhere(mask)
    details = []
    for step, dim in coordinates[:8]:
        details.append(
            f"step{int(step)}.{STATE_LABELS[int(dim)]}={actions[step, dim]:.4f} "
            f"not in [{low[dim]:.4f}, {high[dim]:.4f}]"
        )
    if len(coordinates) > len(details):
        details.append(f"and {len(coordinates) - len(details)} more")
    return "Predicted action chunk left the training range: " + "; ".join(details)


def _confirm_execution(args: argparse.Namespace, action_chunk_size: int) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("--execute requires an interactive terminal")
    print("\nDANGER: --execute will move the physical Piper robot.")
    print(f"  task: {args.task}")
    print(f"  chunk step rate: {args.rate_hz} Hz")
    print(
        f"  nominal {action_chunk_size}-step chunk duration: "
        f"{action_chunk_size / args.rate_hz:.2f}s"
    )
    print(f"  max joint change per command: {args.max_joint_step} rad")
    print(
        "  gripper: binary open/close "
        f"(close when policy output > {args.gripper_threshold:.0%})"
    )
    print(f"  automatic stop after: {args.duration}s")
    answer = input("Clear the workspace, hold the emergency stop, then type EXECUTE: ")
    if answer != "EXECUTE":
        raise RuntimeError("Execution cancelled")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--start-stats", type=Path, default=DEFAULT_START_STATS)
    parser.add_argument("--unnorm-key", default="new_embodiment")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=50.0,
        help="Execution rate for the action-chunk steps (default: 50 Hz, matching training).",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-data-age", type=float, default=2.0)
    parser.add_argument("--max-joint-step", type=float, default=0.01)
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help=(
            "Binary gripper threshold in normalized model units: output above the "
            "threshold closes fully; output at or below it opens fully (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--state-joint-margin",
        type=float,
        default=0.05,
        help="Allowed joint feedback offset outside training min/max before execution is refused.",
    )
    parser.add_argument(
        "--state-gripper-margin",
        type=float,
        default=0.04,
        help=(
            "Allowed gripper feedback offset outside the demonstrated start range "
            "before execution is refused."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--feedback-topic", default="/feedback/joint_states")
    parser.add_argument("--gripper-feedback-topic", default="/feedback/gripper_status")
    parser.add_argument("--global-image-topic", default="/global_camera/camera/color/image_raw")
    parser.add_argument("--hand-image-topic", default="/camera/color/image_raw")
    parser.add_argument("--command-topic", default="/control/joint_states")
    parser.add_argument("--control-service", default="/control_enable")
    parser.add_argument("--gripper-max-width", type=float, default=0.1)
    parser.add_argument("--gripper-effort", type=float, default=1.0)
    parser.add_argument("--server-timeout", type=float, default=30.0)
    parser.add_argument(
        "--save-viewer-data",
        action="store_true",
        help="Save NPZ, summary JSON and directly loadable viewer JSON.",
    )
    parser.add_argument(
        "--viewer-data-dir",
        type=Path,
        default=DEFAULT_VIEWER_DATA_DIR,
        help="Directory for viewer artifacts (default: eval_files/viewer_data).",
    )
    parser.add_argument(
        "--viewer-session-id",
        default=None,
        help="Optional filename/request-id prefix shared with server traces.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rate_hz <= 0 or args.duration <= 0:
        raise ValueError("--rate-hz and --duration must be positive")
    if args.max_joint_step <= 0:
        raise ValueError("--max-joint-step must be positive")
    if not 0.0 <= args.gripper_threshold <= 1.0:
        raise ValueError("--gripper-threshold must be in [0, 1]")
    if args.state_joint_margin < 0 or args.state_gripper_margin < 0:
        raise ValueError("state range margins must be non-negative")
    state_stats, action_stats = _load_stats(args.stats, args.unnorm_key)
    with args.start_stats.expanduser().resolve().open(encoding="utf-8") as file:
        start_stats = json.load(file)
    state_low = np.asarray(state_stats["min"], dtype=np.float32)
    state_high = np.asarray(state_stats["max"], dtype=np.float32)
    action_low = np.asarray(action_stats["min"], dtype=np.float32)
    action_high = np.asarray(action_stats["max"], dtype=np.float32)
    start_low = np.asarray(start_stats["min"], dtype=np.float32)
    start_high = np.asarray(start_stats["max"], dtype=np.float32)
    start_median = np.asarray(start_stats["median"], dtype=np.float32)

    rclpy.init()
    node = PiperLiveNode(args)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    connection: Optional[PolicyConnection] = None
    gate_open = False
    execution_confirmed = False
    state_warning_active = False
    action_warning_active = False
    control_overruns = 0
    viewer_session_id = args.viewer_session_id or make_session_id("sync")
    viewer_recorder: Optional[ClientViewerRecorder] = None
    if args.save_viewer_data:
        viewer_recorder = ClientViewerRecorder(
            args.viewer_data_dir,
            session_id=viewer_session_id,
            method="sync",
            rate_hz=args.rate_hz,
            execute=args.execute,
            task=args.task,
            run_metadata={
                "max_joint_step_rad": float(args.max_joint_step),
                "gripper_threshold": float(args.gripper_threshold),
                "unnorm_key": args.unnorm_key,
                "server": f"ws://{args.host}:{args.port}",
                "stats": str(args.stats.expanduser().resolve()),
            },
        )

    try:
        print("Waiting for Piper feedback and both cameras...")
        deadline = time.monotonic() + 30.0
        snapshot = None
        last_problem_text = None
        while time.monotonic() < deadline:
            snapshot, problems = node.snapshot(args.max_data_age)
            if snapshot is not None:
                break
            problem_text = "; ".join(problems)
            if problem_text != last_problem_text:
                print("  " + problem_text)
                last_problem_text = problem_text
            time.sleep(0.2)
        if snapshot is None:
            raise TimeoutError("Live observation streams were not ready within 30 seconds")
        print("Observation streams ready.")

        connection = PolicyConnection(args.host, args.port, args.server_timeout)
        connection.request_id_prefix = viewer_session_id
        print(f"Policy metadata: {connection.metadata}")
        expected = {
            "training_obs_image_size": [224, 224],
            "default_unnorm_key": args.unnorm_key,
        }
        for key, value in expected.items():
            if connection.metadata.get(key) != value:
                raise RuntimeError(
                    f"Policy metadata mismatch for {key}: got {connection.metadata.get(key)!r}, expected {value!r}"
                )

        other_publishers = node.other_command_publishers()
        if args.execute and other_publishers:
            raise RuntimeError(
                f"Refusing execution: other publishers exist on {args.command_topic}: {other_publishers}. "
                "Stop GELLO and every other controller first."
            )

        # For physical execution, the requested duration starts only after the
        # confirmation is complete and the control gate has opened.
        start = None if args.execute else time.monotonic()
        count = 0
        chunk_count = 0
        period = 1.0 / args.rate_hz
        while start is None or time.monotonic() - start < args.duration:
            snapshot, problems = node.snapshot(args.max_data_age)
            if snapshot is None:
                raise RuntimeError("Stale/missing live data: " + "; ".join(problems))
            images, raw_state = snapshot
            state_message = _state_outside_message(
                raw_state,
                state_low,
                state_high,
                args.state_joint_margin,
                args.state_gripper_margin,
                "full-training-state",
                check_gripper=False,
            )
            if state_message is not None:
                if args.execute:
                    raise RuntimeError(state_message)
                if not state_warning_active:
                    print("WARNING: " + state_message)
                state_warning_active = True
            else:
                state_warning_active = False

            # The demonstration-start distribution is a precondition, not a
            # workspace limit. Checking it after motion begins would stop a
            # successful policy merely because it left its initial pose.
            if count == 0:
                start_message = _state_outside_message(
                    raw_state,
                    start_low,
                    start_high,
                    args.state_joint_margin,
                    args.state_gripper_margin,
                    "demonstration-start",
                )
                if start_message is not None:
                    start_message += f"; recommended median={np.round(start_median, 4)}"
                    if args.execute:
                        raise RuntimeError(start_message)
                    print("WARNING: " + start_message)

            # Small encoder/calibration offsets are allowed by the margins
            # above, but the VLA always receives the exact training domain.
            model_state = np.clip(raw_state, state_low, state_high)

            example = {
                "image": images,  # Exact training order: global, hand.
                "lang": args.task,
                "state": _normalize_state(model_state, state_stats)[None, :],
            }
            infer_start = time.monotonic()
            actions = connection.predict(example, args.unnorm_key)
            latency = time.monotonic() - infer_start
            # Every predicted step is executed before the next inference.
            # Training min/max remains an OOD diagnostic, not a robot limit.
            action_message = _action_violation_message(actions, action_low, action_high)
            if action_message is not None and not action_warning_active:
                print("WARNING: " + action_message)
            action_warning_active = action_message is not None

            if args.execute and not gate_open:
                # Open only after the first complete, validated inference.
                if not execution_confirmed:
                    _confirm_execution(args, connection.action_chunk_size)
                    execution_confirmed = True
                node.set_control_gate(True)
                gate_open = True
                start = time.monotonic()

            chunk_count += 1
            if viewer_recorder is not None:
                assert start is not None
                ready_elapsed = time.monotonic() - start
                # For physical execution the first inference finishes before
                # the interactive confirmation and before the control clock is
                # started. Keep that bootstrap inference adjacent to t=0
                # instead of counting human confirmation time as model latency.
                request_elapsed = (
                    ready_elapsed - latency
                    if infer_start < start
                    else infer_start - start
                )
                viewer_recorder.record_chunk(
                    chunk_id=chunk_count,
                    origin_step=count,
                    actions=actions,
                    latency=latency,
                    prefix_steps=0,
                    request_elapsed=request_elapsed,
                    ready_elapsed=ready_elapsed,
                    ready_step=count,
                    request_id=(
                        f"{viewer_session_id}-request-{chunk_count}"
                    ),
                    source="sync",
                )
            print(
                f"[INFER] chunk={chunk_count:04d} latency={latency:.3f}s; "
                f"executing {len(actions)} steps at {args.rate_hz:.1f} Hz"
            )
            chunk_interrupted = False
            for chunk_step, target_view in enumerate(actions):
                # The duration is a hard safety deadline and may truncate the
                # final chunk. Completed chunks are always followed by a fresh
                # observation and a new inference.
                if start is not None and time.monotonic() - start >= args.duration:
                    chunk_interrupted = True
                    break

                step_start = time.monotonic()
                step_snapshot, problems = node.snapshot(args.max_data_age)
                if step_snapshot is None:
                    raise RuntimeError(
                        "Stale/missing live data during chunk execution: "
                        + "; ".join(problems)
                    )
                _, step_state = step_snapshot
                step_state_message = _state_outside_message(
                    step_state,
                    state_low,
                    state_high,
                    args.state_joint_margin,
                    args.state_gripper_margin,
                    "full-training-state",
                    check_gripper=False,
                )
                if step_state_message is not None:
                    if args.execute:
                        raise RuntimeError(step_state_message)
                    if not state_warning_active:
                        print("WARNING: " + step_state_message)
                    state_warning_active = True
                else:
                    state_warning_active = False

                target = target_view.copy()
                safe_target = target.copy()
                safe_target[:6] = np.clip(
                    safe_target[:6],
                    step_state[:6] - args.max_joint_step,
                    step_state[:6] + args.max_joint_step,
                )
                safe_target[6] = _binary_gripper_target(
                    target[6], args.gripper_threshold
                )

                publish_started = time.monotonic()
                if args.execute:
                    node.publish_command(safe_target)
                publish_finished = time.monotonic()

                if viewer_recorder is not None:
                    assert start is not None
                    viewer_recorder.record_sample(
                        elapsed=publish_started - start,
                        scheduled_elapsed=count / args.rate_hz,
                        publish_end_elapsed=publish_finished - start,
                        logical_step=count,
                        chunk_id=chunk_count,
                        action_index=chunk_step,
                        state=step_state,
                        prediction=target,
                        command=safe_target,
                    )

                count += 1
                mode = "EXECUTE" if args.execute else "DRY-RUN"
                print(
                    f"[{mode}] #{count:04d} chunk={chunk_count:04d} "
                    f"step={chunk_step + 1}/{len(actions)} "
                    f"state={np.round(step_state, 3)} "
                    f"predicted={np.round(target, 3)} "
                    f"command={np.round(safe_target, 3)}"
                )
                remaining = period - (time.monotonic() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    control_overruns += 1

            if chunk_interrupted:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if gate_open:
            try:
                node.set_control_gate(False)
            except Exception as exc:
                print(f"CRITICAL: failed to close control gate: {exc}", file=sys.stderr)
        if connection is not None:
            connection.close()
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if viewer_recorder is not None:
            try:
                paths = viewer_recorder.save(control_overruns=control_overruns)
                if paths is not None:
                    print(
                        "[CLIENT VIEWER DATA] "
                        f"npz={paths['data']} summary={paths['summary']} "
                        f"viewer_json={paths['viewer_json']}"
                    )
            except Exception as exc:
                print(
                    f"WARNING: failed to save client viewer data: {exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
