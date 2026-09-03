#!/usr/bin/env python3
"""Viewer-compatible client artifacts and server-side inference traces."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_VIEWER_DATA_DIR = Path(__file__).resolve().with_name("viewer_data")


def make_session_id(method: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_method = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(method)
    )
    return f"piper-{safe_method}-{stamp}-{os.getpid()}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"byte_length": len(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class ClientViewerRecorder:
    """Collect one async-control run and emit files understood by the viewer."""

    def __init__(
        self,
        output_dir: Path,
        *,
        session_id: str,
        method: str,
        rate_hz: float,
        execute: bool,
        task: str,
        run_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.session_id = str(session_id)
        self.method = str(method)
        self.rate_hz = float(rate_hz)
        self.execute = bool(execute)
        self.task = str(task)
        self.run_metadata = dict(run_metadata or {})
        self.samples: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

    def record_chunk(
        self,
        *,
        chunk_id: int,
        origin_step: int,
        actions: np.ndarray,
        latency: float,
        prefix_steps: int,
        request_elapsed: float,
        ready_elapsed: float,
        ready_step: int,
        request_id: str,
        source: str,
    ) -> None:
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValueError(f"viewer chunk must have shape (H, 7), got {array.shape}")
        self.chunks.append(
            {
                "chunk_id": int(chunk_id),
                "origin_step": int(origin_step),
                "prefix_steps": int(prefix_steps),
                "latency": float(latency),
                "request_elapsed": float(request_elapsed),
                "ready_elapsed": float(ready_elapsed),
                "ready_step": int(ready_step),
                "request_id": str(request_id),
                "source": str(source),
                "actions": array.copy(),
            }
        )

    def record_sample(
        self,
        *,
        elapsed: float,
        scheduled_elapsed: float,
        publish_end_elapsed: float,
        logical_step: int,
        chunk_id: int,
        action_index: int,
        state: np.ndarray,
        prediction: np.ndarray,
        command: np.ndarray,
    ) -> None:
        self.samples.append(
            {
                "elapsed": float(elapsed),
                "scheduled_elapsed": float(scheduled_elapsed),
                "publish_end_elapsed": float(publish_end_elapsed),
                "logical_step": int(logical_step),
                "chunk_id": int(chunk_id),
                "action_index": int(action_index),
                "state": np.asarray(state, dtype=np.float32).copy(),
                "prediction": np.asarray(prediction, dtype=np.float32).copy(),
                "command": np.asarray(command, dtype=np.float32).copy(),
            }
        )

    def save(self, *, control_overruns: int = 0) -> Optional[dict[str, Path]]:
        if not self.samples:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_dir / f"{self.session_id}_client"
        data_path = stem.with_suffix(".npz")
        summary_path = stem.with_name(stem.name + "_summary.json")
        viewer_path = stem.with_name(stem.name + "_viewer.json")

        elapsed = np.asarray([item["elapsed"] for item in self.samples])
        scheduled = np.asarray(
            [item["scheduled_elapsed"] for item in self.samples]
        )
        publish_end = np.asarray(
            [item["publish_end_elapsed"] for item in self.samples]
        )
        logical_steps = np.asarray(
            [item["logical_step"] for item in self.samples], dtype=np.int64
        )
        chunk_ids = np.asarray(
            [item["chunk_id"] for item in self.samples], dtype=np.int64
        )
        action_indices = np.asarray(
            [item["action_index"] for item in self.samples], dtype=np.int64
        )
        states = np.stack([item["state"] for item in self.samples])
        predictions = np.stack([item["prediction"] for item in self.samples])
        commands = np.stack([item["command"] for item in self.samples])
        holds = np.zeros(len(self.samples), dtype=bool)

        max_length = max(len(item["actions"]) for item in self.chunks)
        plan_actions = np.full(
            (len(self.chunks), max_length, 7), np.nan, dtype=np.float32
        )
        for index, chunk in enumerate(self.chunks):
            plan_actions[index, : len(chunk["actions"])] = chunk["actions"]

        np.savez_compressed(
            data_path,
            elapsed_s=elapsed,
            scheduled_elapsed_s=scheduled,
            publish_end_elapsed_s=publish_end,
            logical_steps=logical_steps,
            chunk_ids=chunk_ids,
            action_indices=action_indices,
            states=states,
            predictions=predictions,
            commands=commands,
            holds=holds,
            plan_chunk_ids=np.asarray(
                [item["chunk_id"] for item in self.chunks], dtype=np.int64
            ),
            plan_origins=np.asarray(
                [item["origin_step"] for item in self.chunks], dtype=np.int64
            ),
            plan_lengths=np.asarray(
                [len(item["actions"]) for item in self.chunks], dtype=np.int64
            ),
            plan_prefix_steps=np.asarray(
                [item["prefix_steps"] for item in self.chunks], dtype=np.int64
            ),
            plan_request_elapsed_s=np.asarray(
                [item["request_elapsed"] for item in self.chunks]
            ),
            plan_ready_elapsed_s=np.asarray(
                [item["ready_elapsed"] for item in self.chunks]
            ),
            plan_latencies_s=np.asarray(
                [item["latency"] for item in self.chunks]
            ),
            plan_actions=plan_actions,
            plan_model_actions=plan_actions,
        )

        chunk_metadata = [
            {key: value for key, value in chunk.items() if key != "actions"}
            for chunk in self.chunks
        ]
        summary = {
            **self.run_metadata,
            "schema": "piper_async_eval_summary_v1",
            "session_id": self.session_id,
            "method": self.method,
            "task": self.task,
            "execute": self.execute,
            "rate_hz": self.rate_hz,
            "action_representation": "absolute_joint_targets_v1",
            "control_samples": len(self.samples),
            "returned_chunks": len(self.chunks),
            "control_deadline_miss_count": int(control_overruns),
            "control_clock_rebases": int(control_overruns),
            "chunks": chunk_metadata,
            "hold_intervals": [],
            "data": str(data_path),
            "viewer_json": str(viewer_path),
            "viewer_html": (
                "/home/tams/dRTC/dRTCv2/deployment/realRobots/Piper/"
                "piper_policy_viewer.html"
            ),
        }
        summary_path.write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        viewer = {
            "schema": "piper_rtc_policy_viewer_v1",
            "run": summary,
            "samples": {
                "elapsed": elapsed.tolist(),
                "scheduled_elapsed": scheduled.tolist(),
                "publish_end_elapsed": publish_end.tolist(),
                "logical_steps": logical_steps.tolist(),
                "chunk_ids": chunk_ids.tolist(),
                "states": states.tolist(),
                "predictions": predictions.tolist(),
                "commands": commands.tolist(),
                "holds": holds.tolist(),
            },
            "chunks": [
                {
                    **metadata,
                    "actions": chunk["actions"].tolist(),
                    "model_actions": chunk["actions"].tolist(),
                    "spline_step_offsets": None,
                    "spline_model_actions": None,
                    "spline_bezier_step_offsets": None,
                    "spline_bezier_model_actions": None,
                    "spline_control_points": None,
                    "spline_knot_step_offsets": None,
                    "spline_degree": None,
                }
                for metadata, chunk in zip(chunk_metadata, self.chunks)
            ],
            "hold_intervals": [],
            "analysis": None,
        }
        viewer_path.write_text(
            json.dumps(_jsonable(viewer), ensure_ascii=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        return {
            "data": data_path,
            "summary": summary_path,
            "viewer_json": viewer_path,
        }


class ServerTraceRecorder:
    """Append exact server requests/results without storing camera pixels."""

    def __init__(
        self,
        output_dir: Path,
        *,
        server_name: str,
        metadata: dict[str, Any],
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server_name = str(server_name)
        self.server_id = make_session_id(f"{server_name}-server")
        self.path = self.output_dir / f"{self.server_id}_trace.jsonl"
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._write(
            {
                "schema": "piper_policy_server_trace_v1",
                "event": "server_start",
                "server_id": self.server_id,
                "server_name": self.server_name,
                "utc": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._file.write(
                json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    def record(
        self,
        message: dict[str, Any],
        response: dict[str, Any],
        latency: float,
    ) -> None:
        message_type = message.get("type", "infer")
        if message_type not in {
            "infer", "predict_action", "infer_realtime", "predict_action_realtime"
        }:
            return
        payload = message.get("payload", message)
        examples = payload.get("examples", []) if isinstance(payload, dict) else []
        example_metadata = []
        for example in examples if isinstance(examples, list) else []:
            if not isinstance(example, dict):
                continue
            images = example.get("image", [])
            if not isinstance(images, (list, tuple)):
                images = [images]
            example_metadata.append(
                {
                    "lang": example.get("lang"),
                    "state": example.get("state"),
                    "image_shapes": [list(np.asarray(image).shape) for image in images],
                }
            )
        data = response.get("data", {}) if isinstance(response, dict) else {}
        actions = data.get("actions") if isinstance(data, dict) else None
        self._write(
            {
                "schema": "piper_policy_server_trace_v1",
                "event": "inference",
                "server_id": self.server_id,
                "server_name": self.server_name,
                "utc": datetime.now(timezone.utc).isoformat(),
                "request_id": message.get("request_id", "default"),
                "message_type": message_type,
                "mode": payload.get("mode") if isinstance(payload, dict) else None,
                "inference_delay": (
                    payload.get("inference_delay") if isinstance(payload, dict) else None
                ),
                "suffix_length": (
                    payload.get("suffix_length") if isinstance(payload, dict) else None
                ),
                "unnorm_key": (
                    payload.get("unnorm_key") if isinstance(payload, dict) else None
                ),
                "latency_s": float(latency),
                "status": response.get("status") if isinstance(response, dict) else None,
                "error": response.get("error") if isinstance(response, dict) else None,
                "examples": example_metadata,
                "actions": actions,
            }
        )

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()


def recording_server_class(base_class: type) -> type:
    """Return a WebsocketPolicyServer subclass with incremental JSONL tracing."""

    class RecordingWebsocketPolicyServer(base_class):
        def __init__(
            self,
            *args: Any,
            viewer_data_dir: Path,
            viewer_server_name: str,
            **kwargs: Any,
        ) -> None:
            metadata = dict(kwargs.get("metadata") or {})
            self.viewer_recorder = ServerTraceRecorder(
                viewer_data_dir,
                server_name=viewer_server_name,
                metadata=metadata,
            )
            super().__init__(*args, **kwargs)

        def _route_message(self, msg: dict[str, Any]) -> dict[str, Any]:
            started = time.monotonic()
            response = super()._route_message(msg)
            self.viewer_recorder.record(
                msg, response, latency=time.monotonic() - started
            )
            return response

        def serve_forever(self) -> None:
            try:
                print(f"[SERVER VIEWER TRACE] {self.viewer_recorder.path}")
                super().serve_forever()
            finally:
                self.viewer_recorder.close()

    return RecordingWebsocketPolicyServer


def add_server_recording_args(parser: Any) -> None:
    parser.add_argument(
        "--save-viewer-data",
        action="store_true",
        help="Save an incremental server inference trace for Piper evaluation.",
    )
    parser.add_argument(
        "--viewer-data-dir",
        type=Path,
        default=DEFAULT_VIEWER_DATA_DIR,
        help="Directory for client viewer artifacts and server traces.",
    )


def make_policy_server(
    base_class: type,
    *,
    args: Any,
    server_name: str,
    **kwargs: Any,
) -> Any:
    if not args.save_viewer_data:
        return base_class(**kwargs)
    recording_class = recording_server_class(base_class)
    return recording_class(
        **kwargs,
        viewer_data_dir=args.viewer_data_dir,
        viewer_server_name=server_name,
    )
