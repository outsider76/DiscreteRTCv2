#!/usr/bin/env python3
"""Downsample a local LeRobot v2.1 dataset without modifying its source."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"parquet table is missing required column {name!r}")
    return table.set_column(
        index,
        table.schema.field(index),
        pa.array(values, type=table.schema.field(index).type),
    )


def _downsample_video(
    source: Path,
    destination: Path,
    stride: int,
    target_fps: int,
    expected_source_frames: int,
) -> tuple[Path, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = 0
    decoded = 0
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        with av.open(str(destination), mode="w") as output_container:
            output_stream = output_container.add_stream("libx264", rate=target_fps)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.time_base = Fraction(1, target_fps)
            output_stream.options = {"crf": "18", "preset": "fast"}
            for decoded, frame in enumerate(input_container.decode(input_stream), start=1):
                source_index = decoded - 1
                if source_index % stride:
                    continue
                frame.pts = encoded
                frame.time_base = Fraction(1, target_fps)
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)
                encoded += 1
            for packet in output_stream.encode():
                output_container.mux(packet)
    if decoded != expected_source_frames:
        raise ValueError(
            f"{source}: decoded {decoded} frames, expected {expected_source_frames}"
        )
    expected_output_frames = (expected_source_frames + stride - 1) // stride
    if encoded != expected_output_frames:
        raise ValueError(
            f"{source}: encoded {encoded} frames, expected {expected_output_frames}"
        )
    return destination, encoded


def convert(source: Path, destination: Path, target_fps: int, workers: int) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    info = _read_json(source / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("this converter currently requires LeRobot v2.1")
    source_fps = float(info["fps"])
    ratio = source_fps / target_fps
    stride = int(round(ratio))
    if target_fps <= 0 or stride < 1 or not np.isclose(ratio, stride):
        raise ValueError(
            f"source fps {source_fps:g} must be an integer multiple of target fps "
            f"{target_fps:g}"
        )

    episode_metadata = [
        json.loads(line)
        for line in (source / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(episode_metadata) != int(info["total_episodes"]):
        raise ValueError("episode metadata count disagrees with info.json")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}_staging_", dir=destination.parent)
    )
    try:
        new_episode_metadata: list[dict] = []
        video_jobs: list[tuple[Path, Path, int]] = []
        global_index = 0
        for episode in episode_metadata:
            episode_index = int(episode["episode_index"])
            source_length = int(episode["length"])
            parquet_relative = Path(
                info["data_path"].format(
                    episode_chunk=episode_index // int(info["chunks_size"]),
                    episode_index=episode_index,
                )
            )
            source_parquet = source / parquet_relative
            table = pq.read_table(source_parquet)
            if table.num_rows != source_length:
                raise ValueError(
                    f"{source_parquet}: {table.num_rows} rows, metadata says {source_length}"
                )
            keep = np.arange(0, source_length, stride, dtype=np.int64)
            table = table.take(pa.array(keep))
            output_length = len(keep)
            table = _replace_column(
                table, "timestamp", np.arange(output_length, dtype=np.float32) / target_fps
            )
            table = _replace_column(
                table, "frame_index", np.arange(output_length, dtype=np.int64)
            )
            table = _replace_column(
                table,
                "index",
                np.arange(global_index, global_index + output_length, dtype=np.int64),
            )
            target_parquet = staging / parquet_relative
            target_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target_parquet, compression="snappy")

            updated_episode = dict(episode)
            updated_episode["length"] = output_length
            new_episode_metadata.append(updated_episode)
            global_index += output_length

            for feature_name, feature in info["features"].items():
                if feature.get("dtype") != "video":
                    continue
                video_relative = Path(
                    info["video_path"].format(
                        episode_chunk=episode_index // int(info["chunks_size"]),
                        video_key=feature_name,
                        episode_index=episode_index,
                    )
                )
                video_jobs.append(
                    (source / video_relative, staging / video_relative, source_length)
                )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _downsample_video,
                    source_video,
                    target_video,
                    stride,
                    target_fps,
                    source_length,
                )
                for source_video, target_video, source_length in video_jobs
            ]
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    print(f"encoded videos: {completed}/{len(futures)}", flush=True)

        output_info = json.loads(json.dumps(info))
        output_info["fps"] = target_fps
        output_info["total_frames"] = global_index
        output_info["preprocessing"] = {
            "operation": "integer_stride_downsample",
            "source": str(source),
            "source_fps": source_fps,
            "target_fps": target_fps,
            "stride": stride,
            "selected_source_frames": "0, stride, 2*stride, ...",
        }
        for feature in output_info["features"].values():
            if feature.get("dtype") == "video":
                feature["info"]["video.fps"] = float(target_fps)
        _write_json(staging / "meta/info.json", output_info)

        (staging / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in new_episode_metadata),
            encoding="utf-8",
        )
        shutil.copy2(source / "meta/tasks.jsonl", staging / "meta/tasks.jsonl")
        shutil.copy2(source / "meta/modality.json", staging / "meta/modality.json")
        embodiment = _read_json(source / "meta/embodiment.json")
        for key in (
            "record_frequency",
            "body_controller_frequency",
            "hand_controller_frequency",
        ):
            if key in embodiment:
                embodiment[key] = target_fps
        _write_json(staging / "meta/embodiment.json", embodiment)

        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"created {destination}: {len(episode_metadata)} episodes, "
        f"{global_index} frames at {target_fps} Hz",
        flush=True,
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target-fps", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    convert(args.source, args.destination, args.target_fps, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
