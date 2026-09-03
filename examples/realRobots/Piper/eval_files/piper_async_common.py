#!/usr/bin/env python3
"""Shared asynchronous Piper control runtime.

This module is intentionally separate from ``piper_live_client.py`` so the
existing synchronous controller remains unchanged.  Public entry points are
``piper_async_rtc_client.py`` and
``piper_async_temporal_ensemble_client.py``.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import rclpy
import websocket
from rclpy.executors import MultiThreadedExecutor

from deployment.model_server.tools import msgpack_numpy
from examples.realRobots.Piper.eval_files.piper_eval_recording import (
    ClientViewerRecorder,
    DEFAULT_VIEWER_DATA_DIR,
    make_session_id,
)
from examples.realRobots.Piper.eval_files.piper_live_client import (
    DEFAULT_START_STATS,
    DEFAULT_STATS,
    DEFAULT_TASK,
    PiperLiveNode,
    _action_violation_message,
    _binary_gripper_target,
    _load_stats,
    _normalize_state,
    _state_outside_message,
)


Method = Literal["rtc", "temporal_ensemble"]


@dataclasses.dataclass(frozen=True)
class TimedChunk:
    """An action chunk whose row zero is aligned to ``origin_step``."""

    chunk_id: int
    origin_step: int
    actions: np.ndarray
    latency: float


@dataclasses.dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    origin_step: int
    example: dict
    previous_chunk: Optional[np.ndarray] = None
    inference_delay: int = 0
    suffix_length: int = 0
    submitted_monotonic: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    request: InferenceRequest
    actions: Optional[np.ndarray]
    latency: float
    error: Optional[BaseException] = None


class AsyncPolicyConnection:
    """One WebSocket connection used exclusively by the inference worker."""

    def __init__(self, host: str, port: int, timeout: float):
        for key in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
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
                "Policy server metadata has no positive action_chunk_size: "
                f"{self.metadata.get('action_chunk_size')!r}"
            )
        self._next_request_id = 1

    def _predict(self, message_type: str, payload: dict) -> np.ndarray:
        prefix = getattr(self, "request_id_prefix", "piper-async")
        request_id = f"{prefix}-request-{self._next_request_id}"
        self._next_request_id += 1
        message = {
            "type": message_type,
            "request_id": request_id,
            "payload": payload,
        }
        self._socket.send_binary(msgpack_numpy.packb(message))
        response_raw = self._socket.recv()
        if isinstance(response_raw, str):
            raise RuntimeError(f"Policy server traceback:\n{response_raw}")
        response = msgpack_numpy.unpackb(response_raw)
        if response.get("status") != "ok":
            raise RuntimeError(f"Policy server error: {response.get('error', response)}")
        actions = np.asarray(response["data"]["actions"], dtype=np.float32)
        expected = (self.action_chunk_size, 7)
        if actions.shape == (1, *expected):
            actions = actions[0]
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(
                f"Invalid policy action array: shape={actions.shape}, "
                f"finite={np.isfinite(actions).all()}"
            )
        return actions

    def predict(self, example: dict, unnorm_key: str) -> np.ndarray:
        return self._predict(
            "infer",
            {"examples": [example], "unnorm_key": unnorm_key},
        )

    def predict_rtc(
        self,
        example: dict,
        previous_chunk: np.ndarray,
        inference_delay: int,
        suffix_length: int,
        unnorm_key: str,
        prefix_attention_schedule: str,
        max_guidance_weight: float,
    ) -> np.ndarray:
        return self._predict(
            "infer_realtime",
            {
                "examples": [example],
                "prev_action_chunk": np.asarray(previous_chunk, dtype=np.float32),
                "inference_delay": int(inference_delay),
                "unnorm_key": unnorm_key,
                "mode": "pigdm",
                "suffix_length": int(suffix_length),
                "prefix_attention_schedule": prefix_attention_schedule,
                "max_guidance_weight": float(max_guidance_weight),
            },
        )

    def close(self) -> None:
        self._socket.close()


class InferenceWorker(threading.Thread):
    """Serial policy worker; the controller thread never blocks on inference."""

    def __init__(
        self,
        method: Method,
        connection: AsyncPolicyConnection,
        args: argparse.Namespace,
        requests: queue.Queue,
        results: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"piper-{method}-inference", daemon=True)
        self.method = method
        self.connection = connection
        self.args = args
        self.requests = requests
        self.results = results
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return
            started = time.monotonic()
            try:
                if self.method == "rtc":
                    if request.previous_chunk is None:
                        raise RuntimeError("RTC request has no previous chunk")
                    actions = self.connection.predict_rtc(
                        request.example,
                        request.previous_chunk,
                        request.inference_delay,
                        request.suffix_length,
                        self.args.unnorm_key,
                        self.args.prefix_attention_schedule,
                        self.args.max_guidance_weight,
                    )
                else:
                    actions = self.connection.predict(request.example, self.args.unnorm_key)
                self.results.put(
                    InferenceResult(
                        request=request,
                        actions=actions,
                        latency=time.monotonic() - started,
                    )
                )
            except BaseException as exc:
                self.results.put(
                    InferenceResult(
                        request=request,
                        actions=None,
                        latency=time.monotonic() - started,
                        error=exc,
                    )
                )


def build_async_parser(method: Method, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--start-stats", type=Path, default=DEFAULT_START_STATS)
    parser.add_argument("--unnorm-key", default="new_embodiment")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-data-age", type=float, default=2.0)
    parser.add_argument("--max-joint-step", type=float, default=0.01)
    parser.add_argument("--gripper-threshold", type=float, default=0.3)
    parser.add_argument("--state-joint-margin", type=float, default=0.05)
    parser.add_argument("--state-gripper-margin", type=float, default=0.04)
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
        "--log-every",
        type=int,
        default=1,
        help="Print one controller line every N steps (default: 1).",
    )
    parser.add_argument(
        "--save-viewer-data",
        action="store_true",
        help=(
            "Save NPZ, summary JSON and directly loadable viewer JSON for this run."
        ),
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
        help=(
            "Optional filename/request-id prefix used to correlate client files "
            "with the server trace."
        ),
    )

    if method == "rtc":
        parser.add_argument(
            "--min-execution-horizon",
            type=int,
            default=25,
            help="RTC s_min in controller steps (paper real-world default: 25).",
        )
        parser.add_argument(
            "--delay-buffer-size",
            type=int,
            default=10,
            help="Number of observed inference delays used by conservative max forecast.",
        )
        parser.add_argument(
            "--initial-delay-steps",
            type=int,
            default=0,
            help="Initial RTC delay estimate; 0 derives it from first vanilla inference.",
        )
        parser.add_argument(
            "--prefix-attention-schedule",
            choices=("exp", "linear", "ones", "zeros"),
            default="exp",
        )
        parser.add_argument(
            "--max-guidance-weight",
            type=float,
            default=5.0,
            help="ΠGDM guidance clipping beta (paper default: 5).",
        )
    else:
        parser.add_argument(
            "--inference-interval-steps",
            type=int,
            default=1,
            help="Minimum controller steps between inference starts; 1 gives dense TE.",
        )
        parser.add_argument(
            "--new-chunk-weight",
            type=float,
            default=0.5,
            help="Weight of the newer prediction in a two-chunk overlap (default: 0.5).",
        )
    return parser


def _validate_args(args: argparse.Namespace, method: Method) -> None:
    if args.rate_hz <= 0 or args.duration <= 0:
        raise ValueError("--rate-hz and --duration must be positive")
    if args.max_data_age <= 0 or args.max_joint_step <= 0:
        raise ValueError("--max-data-age and --max-joint-step must be positive")
    if not 0.0 <= args.gripper_threshold <= 1.0:
        raise ValueError("--gripper-threshold must be in [0, 1]")
    if args.state_joint_margin < 0 or args.state_gripper_margin < 0:
        raise ValueError("state margins must be non-negative")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if method == "rtc":
        if args.min_execution_horizon <= 0:
            raise ValueError("--min-execution-horizon must be positive")
        if args.delay_buffer_size <= 0 or args.initial_delay_steps < 0:
            raise ValueError("RTC delay parameters are invalid")
        if args.max_guidance_weight <= 0:
            raise ValueError("--max-guidance-weight must be positive")
    else:
        if args.inference_interval_steps <= 0:
            raise ValueError("--inference-interval-steps must be positive")
        if not 0.0 <= args.new_chunk_weight <= 1.0:
            raise ValueError("--new-chunk-weight must be in [0, 1]")


def _wait_for_observation(node: PiperLiveNode, max_age: float) -> tuple[list[np.ndarray], np.ndarray]:
    print("Waiting for Piper feedback and both cameras...")
    deadline = time.monotonic() + 30.0
    last_problem_text = None
    while time.monotonic() < deadline:
        snapshot, problems = node.snapshot(max_age)
        if snapshot is not None:
            print("Observation streams ready.")
            return snapshot
        problem_text = "; ".join(problems)
        if problem_text != last_problem_text:
            print("  " + problem_text)
            last_problem_text = problem_text
        time.sleep(0.2)
    raise TimeoutError("Live observation streams were not ready within 30 seconds")


def _example_from_snapshot(
    images: list[np.ndarray],
    raw_state: np.ndarray,
    state_stats: dict,
    state_low: np.ndarray,
    state_high: np.ndarray,
    task: str,
) -> dict:
    model_state = np.clip(raw_state, state_low, state_high)
    return {
        "image": images,
        "lang": task,
        "state": _normalize_state(model_state, state_stats)[None, :],
    }


def _runtime_state_error(
    state: np.ndarray,
    state_low: np.ndarray,
    state_high: np.ndarray,
    args: argparse.Namespace,
) -> Optional[str]:
    return _state_outside_message(
        state,
        state_low,
        state_high,
        args.state_joint_margin,
        args.state_gripper_margin,
        "full-training-state",
        check_gripper=False,
    )


def _confirm_execution(
    args: argparse.Namespace,
    method: Method,
    action_horizon: int,
    initial_delay: int,
) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("--execute requires an interactive terminal")
    print("\nDANGER: --execute will move the physical Piper robot.")
    print(f"  controller: asynchronous {method}")
    print(f"  task: {args.task}")
    print(f"  control rate: {args.rate_hz:.1f} Hz")
    print(f"  action horizon: {action_horizon} steps")
    print(f"  initial measured/estimated inference delay: {initial_delay} steps")
    if method == "rtc":
        print(f"  RTC s_min: {args.min_execution_horizon} steps")
        print(f"  RTC beta: {args.max_guidance_weight}")
    else:
        print(f"  newer overlap weight: {args.new_chunk_weight:.3f}")
    print(f"  max arm-joint change per command: {args.max_joint_step} rad")
    print(
        "  gripper: binary open/close "
        f"(close when policy output > {args.gripper_threshold:.0%})"
    )
    print(f"  automatic stop after: {args.duration}s")
    answer = input("Clear the workspace, hold the emergency stop, then type EXECUTE: ")
    if answer != "EXECUTE":
        raise RuntimeError("Execution cancelled")


def _select_temporal_ensemble_action(
    chunks: list[TimedChunk],
    step: int,
    new_chunk_weight: float,
) -> tuple[np.ndarray, str, int, int]:
    valid = [
        chunk
        for chunk in chunks
        if 0 <= step - chunk.origin_step < len(chunk.actions)
    ]
    if not valid:
        raise RuntimeError(f"Action starvation at controller step {step}: no valid TE chunk")
    valid = valid[-2:]
    if len(valid) == 1:
        chunk = valid[0]
        index = step - chunk.origin_step
        return (
            chunk.actions[index].copy(),
            f"chunk={chunk.chunk_id} idx={index} w=1.000",
            chunk.chunk_id,
            index,
        )
    old, new = valid
    old_index = step - old.origin_step
    new_index = step - new.origin_step
    old_weight = 1.0 - new_chunk_weight
    action = (
        old_weight * old.actions[old_index]
        + new_chunk_weight * new.actions[new_index]
    )
    source = (
        f"chunks={old.chunk_id}:{old_index},{new.chunk_id}:{new_index} "
        f"w={old_weight:.3f},{new_chunk_weight:.3f}"
    )
    return np.asarray(action, dtype=np.float32), source, new.chunk_id, new_index


def _rtc_request_prefix(
    current_chunk: TimedChunk,
    global_step: int,
    delay_estimate: int,
) -> tuple[np.ndarray, int]:
    """Return the timestamp-aligned old prefix and RTC suffix length ``s``."""

    horizon = len(current_chunk.actions)
    local_index = global_step - current_chunk.origin_step
    if not 0 <= local_index < horizon:
        raise RuntimeError(
            f"Cannot start RTC from expired/misaligned chunk: index={local_index}, H={horizon}"
        )
    if delay_estimate * 2 > horizon:
        raise RuntimeError(
            f"Observed RTC delay {delay_estimate} exceeds H/2={horizon / 2:.1f}"
        )
    if delay_estimate > local_index:
        raise RuntimeError(
            f"RTC requires d <= s; got d={delay_estimate}, s={local_index}"
        )
    if local_index > horizon - delay_estimate:
        raise RuntimeError(
            "RTC inference started too late to guarantee a continuous action: "
            f"s={local_index}, d={delay_estimate}, H={horizon}"
        )
    return current_chunk.actions[local_index:].copy(), local_index


def run_async_controller(method: Method, args: argparse.Namespace) -> None:
    """Run either asynchronous RTC or two-chunk temporal ensembling."""

    _validate_args(args, method)
    state_stats, action_stats = _load_stats(args.stats, args.unnorm_key)
    with args.start_stats.expanduser().resolve().open(encoding="utf-8") as file:
        import json

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

    connection: Optional[AsyncPolicyConnection] = None
    worker: Optional[InferenceWorker] = None
    stop_event = threading.Event()
    request_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    gate_open = False
    control_overruns = 0
    viewer_recorder: Optional[ClientViewerRecorder] = None
    recording_method = getattr(args, "viewer_method", method)
    viewer_session_id = args.viewer_session_id or make_session_id(recording_method)
    if args.save_viewer_data:
        viewer_recorder = ClientViewerRecorder(
            args.viewer_data_dir,
            session_id=viewer_session_id,
            method=recording_method,
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
        images, raw_state = _wait_for_observation(node, args.max_data_age)
        initial_state_message = _runtime_state_error(
            raw_state, state_low, state_high, args
        )
        if initial_state_message is not None:
            if args.execute:
                raise RuntimeError(initial_state_message)
            print("WARNING: " + initial_state_message)

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

        connection = AsyncPolicyConnection(args.host, args.port, args.server_timeout)
        connection.request_id_prefix = viewer_session_id
        print(f"Policy metadata: {connection.metadata}")
        expected = {
            "training_obs_image_size": [224, 224],
            "default_unnorm_key": args.unnorm_key,
        }
        for key, expected_value in expected.items():
            if connection.metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"Policy metadata mismatch for {key}: "
                    f"got {connection.metadata.get(key)!r}, expected {expected_value!r}"
                )
        if method == "rtc" and not connection.metadata.get(
            "supports_inference_time_rtc", False
        ):
            raise RuntimeError(
                "The connected server does not expose inference-time RTC. "
                "Start run_policy_server_rtc.sh for the QwenPI_v3 checkpoint."
            )

        horizon = connection.action_chunk_size
        if method == "rtc" and not 0 < args.min_execution_horizon < horizon:
            raise ValueError(
                f"--min-execution-horizon must be in [1, {horizon - 1}]"
            )

        other_publishers = node.other_command_publishers()
        if args.execute and other_publishers:
            raise RuntimeError(
                f"Refusing execution: other publishers exist on {args.command_topic}: "
                f"{other_publishers}. Stop GELLO and every other controller first."
            )

        initial_example = _example_from_snapshot(
            images, raw_state, state_stats, state_low, state_high, args.task
        )
        infer_started = time.monotonic()
        initial_actions = connection.predict(initial_example, args.unnorm_key)
        initial_latency = time.monotonic() - infer_started
        initial_delay = (
            args.initial_delay_steps
            if method == "rtc" and args.initial_delay_steps > 0
            else max(1, int(np.ceil(initial_latency * args.rate_hz)))
        )
        if method == "rtc" and initial_delay * 2 > horizon:
            raise RuntimeError(
                f"Initial inference delay {initial_delay} is too large for horizon {horizon}; "
                "RTC requires d <= H/2 so an old action remains available."
            )
        if method == "temporal_ensemble" and initial_delay >= horizon:
            raise RuntimeError(
                f"Initial inference delay {initial_delay} is not shorter than horizon {horizon}; "
                "dense temporal ensembling cannot guarantee a continuous action."
            )
        initial_action_message = _action_violation_message(
            initial_actions, action_low, action_high
        )
        if initial_action_message is not None:
            print("WARNING: " + initial_action_message)
        print(
            f"[INITIAL INFER] latency={initial_latency:.3f}s "
            f"delay_estimate={initial_delay} steps chunk=1"
        )

        if args.execute:
            _confirm_execution(args, method, horizon, initial_delay)
            node.set_control_gate(True)
            gate_open = True

        worker = InferenceWorker(
            method,
            connection,
            args,
            request_queue,
            result_queue,
            stop_event,
        )
        worker.start()

        current_chunk = TimedChunk(1, 0, initial_actions, initial_latency)
        te_chunks = [current_chunk]
        delay_buffer = collections.deque(
            [initial_delay], maxlen=getattr(args, "delay_buffer_size", 1)
        )
        next_chunk_id = 2
        request_busy = False
        last_request_step = 0
        global_step = 0
        control_start = time.monotonic()
        next_tick = control_start
        state_warning_active = False
        action_warning_active = initial_action_message is not None
        if viewer_recorder is not None:
            viewer_recorder.record_chunk(
                chunk_id=current_chunk.chunk_id,
                origin_step=current_chunk.origin_step,
                actions=current_chunk.actions,
                latency=current_chunk.latency,
                prefix_steps=0,
                request_elapsed=-initial_latency,
                ready_elapsed=0.0,
                ready_step=0,
                request_id=f"{viewer_session_id}-request-1",
                source="initial",
            )

        while time.monotonic() - control_start < args.duration:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = None
            if result is not None:
                request_busy = False
                if result.error is not None or result.actions is None:
                    raise RuntimeError(
                        f"Asynchronous inference request {result.request.request_id} failed"
                    ) from result.error
                observed_delay = global_step - result.request.origin_step
                if observed_delay >= horizon:
                    raise RuntimeError(
                        f"Inference result expired: observed delay={observed_delay}, "
                        f"horizon={horizon}"
                    )
                chunk = TimedChunk(
                    next_chunk_id,
                    result.request.origin_step,
                    result.actions,
                    result.latency,
                )
                next_chunk_id += 1
                message = _action_violation_message(
                    chunk.actions, action_low, action_high
                )
                if message is not None and not action_warning_active:
                    print("WARNING: " + message)
                action_warning_active = message is not None
                if method == "rtc":
                    current_chunk = chunk
                    delay_buffer.append(max(1, observed_delay))
                else:
                    te_chunks.append(chunk)
                    te_chunks = te_chunks[-2:]
                if viewer_recorder is not None:
                    request_elapsed = (
                        result.request.submitted_monotonic - control_start
                        if result.request.submitted_monotonic is not None
                        else time.monotonic() - control_start - result.latency
                    )
                    viewer_recorder.record_chunk(
                        chunk_id=chunk.chunk_id,
                        origin_step=chunk.origin_step,
                        actions=chunk.actions,
                        latency=chunk.latency,
                        prefix_steps=(
                            result.request.inference_delay
                            if method == "rtc"
                            else 0
                        ),
                        request_elapsed=request_elapsed,
                        ready_elapsed=time.monotonic() - control_start,
                        ready_step=global_step,
                        request_id=(
                            f"{viewer_session_id}-request-"
                            f"{result.request.request_id}"
                        ),
                        source=("rtc" if method == "rtc" else "temporal-ensemble"),
                    )
                print(
                    f"[ASYNC INFER READY] chunk={chunk.chunk_id} "
                    f"origin={chunk.origin_step} ready_step={global_step} "
                    f"observed_delay={observed_delay} latency={chunk.latency:.3f}s"
                )

            snapshot, problems = node.snapshot(args.max_data_age)
            if snapshot is None:
                raise RuntimeError(
                    "Stale/missing live data during asynchronous control: "
                    + "; ".join(problems)
                )
            images, step_state = snapshot
            state_message = _runtime_state_error(
                step_state, state_low, state_high, args
            )
            if state_message is not None:
                if args.execute:
                    raise RuntimeError(state_message)
                if not state_warning_active:
                    print("WARNING: " + state_message)
                state_warning_active = True
            else:
                state_warning_active = False

            if not request_busy:
                should_request = False
                previous_chunk = None
                delay_estimate = 0
                suffix_length = 0
                if method == "rtc":
                    local_index = global_step - current_chunk.origin_step
                    delay_estimate = max(delay_buffer)
                    should_request = local_index >= max(
                        args.min_execution_horizon, delay_estimate
                    )
                    if should_request:
                        previous_chunk, suffix_length = _rtc_request_prefix(
                            current_chunk, global_step, delay_estimate
                        )
                else:
                    should_request = (
                        global_step - last_request_step
                        >= args.inference_interval_steps
                    )

                if should_request:
                    example = _example_from_snapshot(
                        images,
                        step_state,
                        state_stats,
                        state_low,
                        state_high,
                        args.task,
                    )
                    request = InferenceRequest(
                        request_id=next_chunk_id,
                        origin_step=global_step,
                        example=example,
                        previous_chunk=previous_chunk,
                        inference_delay=delay_estimate,
                        suffix_length=suffix_length,
                        submitted_monotonic=time.monotonic(),
                    )
                    request_queue.put_nowait(request)
                    request_busy = True
                    last_request_step = global_step
                    if method == "rtc":
                        print(
                            f"[ASYNC RTC START] request={request.request_id} "
                            f"origin={global_step} d_est={delay_estimate} "
                            f"s={suffix_length} overlap={len(previous_chunk)}"
                        )
                    else:
                        print(
                            f"[ASYNC TE START] request={request.request_id} "
                            f"origin={global_step}"
                        )

            if method == "rtc":
                action_index = global_step - current_chunk.origin_step
                if not 0 <= action_index < horizon:
                    raise RuntimeError(
                        f"RTC action starvation at step {global_step}: "
                        f"chunk={current_chunk.chunk_id}, index={action_index}, H={horizon}"
                    )
                target = current_chunk.actions[action_index].copy()
                source = f"chunk={current_chunk.chunk_id} idx={action_index}"
                selected_chunk_id = current_chunk.chunk_id
                selected_action_index = action_index
            else:
                te_chunks = [
                    chunk
                    for chunk in te_chunks
                    if global_step - chunk.origin_step < horizon
                ]
                (
                    target,
                    source,
                    selected_chunk_id,
                    selected_action_index,
                ) = _select_temporal_ensemble_action(
                    te_chunks, global_step, args.new_chunk_weight
                )

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
                viewer_recorder.record_sample(
                    elapsed=publish_started - control_start,
                    scheduled_elapsed=next_tick - control_start,
                    publish_end_elapsed=publish_finished - control_start,
                    logical_step=global_step,
                    chunk_id=selected_chunk_id,
                    action_index=selected_action_index,
                    state=step_state,
                    prediction=target,
                    command=safe_target,
                )

            if global_step % args.log_every == 0:
                mode = "EXECUTE" if args.execute else "DRY-RUN"
                print(
                    f"[{mode}] step={global_step:05d} {source} "
                    f"state={np.round(step_state, 3)} "
                    f"predicted={np.round(target, 3)} "
                    f"command={np.round(safe_target, 3)}"
                )

            global_step += 1
            next_tick += 1.0 / args.rate_hz
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # Never publish a burst of catch-up commands after a delayed
                # cycle. Resume the 50 Hz cadence from the current time.
                control_overruns += 1
                if control_overruns == 1 or control_overruns % 50 == 0:
                    print(
                        f"WARNING: control deadline missed by {-remaining * 1000.0:.1f} ms "
                        f"(count={control_overruns})"
                    )
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if gate_open:
            try:
                node.set_control_gate(False)
            except Exception as exc:
                print(
                    f"CRITICAL: failed to close control gate: {exc}",
                    file=sys.stderr,
                )
        stop_event.set()
        try:
            request_queue.put_nowait(None)
        except queue.Full:
            pass
        if connection is not None:
            connection.close()
        if worker is not None:
            worker.join(timeout=2.0)
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
