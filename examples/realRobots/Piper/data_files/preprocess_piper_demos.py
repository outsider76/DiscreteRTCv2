#!/usr/bin/env python3
"""Build matched raw-action / uniform-left B-spline Piper datasets for starVLA.

Thin driver around the integrated BSplineEncoder pipeline
(``BSplineEncoder/src/data``; ``spline-preprocess-piper``).  For every task it
writes ONE LeRobot v2.1 dataset and ONE spline sidecar computed from that
dataset's final ``action`` column, so both policy variants see identical
episodes, splits, retained frames, observations, state, and action semantics:

* raw-action variant:  ``<output>/ActionCommand/State_Joint/<task>_30hz_cleanup``
  (loaded with robot type ``piper_abs_joints_30step``: 30 future actions);
* B-spline variant:    the same dataset plus the sidecar
  ``<task>_30hz_cleanup_spline_uniform_left_s2_h30_c18`` (robot type
  ``piper_uniform_left_bspline_30step``: 18 control rows decoding to 30 steps).

Semantics (preset ``joint-command`` from ``scripts/legacy_piper/preprocess_piper.py``,
the convention of the earlier ``stacking_cups_action_30hz`` policies):

* state  = measured joints[6] + measured gripper[1] at observation time t;
* action = absolute commanded (GELLO) joints[6] + gripper[1] at t (offset 0);
* images = ``global`` then ``hand`` frames nearest to t on the 30 Hz clock.

The pipeline names state blocks ``measured_joints``/``measured_gripper``.
starVLA's Piper data configs (and the deployment observation contract derived
from them) read ``state.joints``/``state.gripper``, so this driver adds those
two names to ``meta/modality.json`` as aliases of the SAME columns.

``--preset state-delta-joint`` builds the BSplineEncoder per-step delta variant instead
(``<output>/StateDeltaJoint/<task>_30hz_cleanup`` plus the same-geometry sidecar):

* action = measured joint increment ``q[t+1] - q[t]`` (6, rad) + absolute measured
  gripper[t+1] (offset 1; the last row of each episode, which has no next sample, is dropped);
* the sidecar fits each 30-step window of those per-step increments, so a decoded chunk is
  integrated back to joint targets with ``q[t] + cumsum``;
* its summary goes to ``<output>/StateDeltaJoint/PREPROCESSING.json``.

Language instructions: by default the pipeline's short task names are kept
("Classify the blocks.", "Hang the mug.", "Stack the cups."); the trained policies and
their open-loop evaluations on these datasets use them.  ``--detailed-instructions``
opts into the longer sentences in ``DETAILED_TASK_INSTRUCTIONS`` (written from the
recorded videos): new outputs get them through the pipeline's ``task_description``,
existing outputs are rewritten in place (``meta/tasks.jsonl`` and the ``tasks`` field of
``meta/episodes.jsonl`` only; the sidecar's dataset fingerprint covers ``info.json`` and
the parquet files, which are untouched).  Nothing else in the pipeline output is modified.

Example::

    python examples/realRobots/Piper/data_files/preprocess_piper_demos.py \
        --raw-root data/raw/piper_demos --output-root data/processed/piper_demos
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
BSPLINE_SRC = REPO_ROOT / "BSplineEncoder" / "src"

TASKS = ("classify_blocks", "hanging_mug", "stacking_cup")
PRESET = "joint-command"
# Presets this driver can build; values are the output sub-directories (BSplineEncoder ``PRESETS``).
SUPPORTED_PRESETS = {"joint-command": "ActionCommand/State_Joint", "state-delta-joint": "StateDeltaJoint"}
FPS = 30.0
SPLINE = {
    "mode": "uniform_left",
    "chunk_size": 30,
    "degree": 3,
    "num_basis": 18,
    "span_length_steps": 2,
    "regularization": 1.0e-4,
    "quantize": False,
    "normalization": "none",
}
# Geometry every consumer (data_config.py, QwenPI_v3_UniformBSpline) relies on.
EXPECTED_ENCODER = {
    "mode": "uniform_left",
    "action_dim": 7,
    "chunk_size": 30,
    "frequency_hz": 30.0,
    "degree": 3,
    "num_basis": 18,
    "span_length_steps": 2,
    "regularization": 1.0e-4,
    "end_padding": True,
    "implementation_version": "uniform_left_direct_fit_v1",
}
STATE_ALIASES = {"joints": "measured_joints", "gripper": "measured_gripper"}
# Opt-in (--detailed-instructions) longer instructions; one per task, identical for every episode.
DETAILED_TASK_INSTRUCTIONS = {
    "classify_blocks": (
        "Sort the small cubes on the table by color: pick up the cubes one at a time, "
        "put every blue cube into the blue box and every red cube into the red box, "
        "and stop when no cubes are left on the table."
    ),
    "hanging_mug": (
        "Pick up the dark green mug from the table by its rim and hang it by its handle "
        "on the left peg of the wooden mug tree."
    ),
    "stacking_cup": (
        "Stack the three upside-down cups into one tower: place the teal cup over the red cup, "
        "then place the blue cup over the teal cup, so the red cup is at the bottom and the "
        "blue cup is on top."
    ),
}


def dataset_path(output_root: Path, task: str, preset: str = PRESET) -> Path:
    return output_root / SUPPORTED_PRESETS[preset] / f"{task}_{FPS:g}hz_cleanup"


def summary_path(output_root: Path, preset: str = PRESET) -> Path:
    # The default preset keeps its historical location at the output root.
    return output_root / ("" if preset == PRESET else SUPPORTED_PRESETS[preset]) / "PREPROCESSING.json"


def sidecar_path(output_root: Path, task: str, preset: str = PRESET) -> Path:
    dataset = dataset_path(output_root, task, preset)
    return dataset.parent / f"{dataset.name}_spline_uniform_left_s2_h30_c18"


def check_encoder_geometry(sidecar: Path) -> dict:
    from spline_encoder import load_encoder

    state = json.loads((sidecar / "encoder.json").read_text())
    config = state["config"]
    mismatched = {k: (config.get(k), v) for k, v in EXPECTED_ENCODER.items() if config.get(k) != v}
    if mismatched:
        raise AssertionError(f"{sidecar}: encoder geometry mismatch (actual, expected): {mismatched}")
    encoder = load_encoder(sidecar / "encoder.json")
    cfg = encoder.config
    # 30 executable actions are 15 cubic spans of 2 steps -> 15 + 3 = 18 controls.
    assert cfg.executable_spans == 15 and cfg.right_context_spans == 3, cfg
    assert cfg.num_basis == cfg.executable_spans + cfg.degree == 18, cfg
    assert cfg.executable_spans * cfg.span_length_steps == cfg.chunk_size == 30, cfg
    assert encoder.basis.shape == (30, 18), encoder.basis.shape
    assert len(encoder.knots) == 18 + 3 + 1, encoder.knots
    manifest = json.loads((sidecar / "manifest.json").read_text())
    assert manifest["mode"] == "uniform_left" and manifest["quantized"] is False, manifest
    return {"encoder": config, "executable_spans": cfg.executable_spans,
            "right_context_spans": cfg.right_context_spans, "basis_shape": list(encoder.basis.shape),
            "tokenizer_id": encoder.tokenizer_id}


def add_starvla_state_aliases(dataset: Path) -> None:
    path = dataset / "meta" / "modality.json"
    modality = json.loads(path.read_text())
    state = modality["state"]
    for alias, source in STATE_ALIASES.items():
        if source not in state:
            raise KeyError(f"{path}: state block {source!r} missing")
        entry = {k: v for k, v in state[source].items() if k != "alias_of"}
        if alias in state and {k: v for k, v in state[alias].items() if k != "alias_of"} != entry:
            raise ValueError(f"{path}: existing state.{alias} differs from {source}")
        state[alias] = {**entry, "alias_of": source}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(modality, indent=2) + "\n")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(tmp, path)


def set_task_instruction(dataset: Path, instruction: str) -> str | None:
    """Make ``instruction`` the single task of ``dataset``; return the replaced text (None if unchanged)."""
    tasks_path, episodes_path = dataset / "meta/tasks.jsonl", dataset / "meta/episodes.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    if [row["task_index"] for row in tasks] != [0]:
        raise ValueError(f"{tasks_path}: expected exactly one task with index 0, got {tasks}")
    previous = tasks[0]["task"]
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    if any(row.get("tasks") != [previous] for row in episodes):
        raise ValueError(f"{episodes_path}: episodes do not all use the single task {previous!r}")
    if previous == instruction:
        return None
    _write_jsonl_atomic(episodes_path, [{**row, "tasks": [instruction]} for row in episodes])
    _write_jsonl_atomic(tasks_path, [{"task_index": 0, "task": instruction}])
    return previous


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data/raw/piper_demos")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/processed/piper_demos")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--preset", default=PRESET, choices=sorted(SUPPORTED_PRESETS),
                        help="joint-command: absolute GELLO commands at t (default); "
                             "state-delta-joint: per-step measured joint deltas q[t+1]-q[t]")
    parser.add_argument("--validation-episodes", type=int, default=5,
                        help="Last K episodes (recording order) form the held-out 'validation' split")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing outputs (pipeline swaps them in atomically)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Only re-check geometry/aliases for tasks whose outputs already exist")
    parser.add_argument("--detailed-instructions", action="store_true",
                        help="Use DETAILED_TASK_INSTRUCTIONS instead of the short task names (rewrites existing "
                             "meta/tasks.jsonl + episodes.jsonl; changes what open-loop evals of older policies see)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(BSPLINE_SRC))
    import data as bspline_data
    from data.config import PRESETS, config_from_mapping
    from data.pipeline import run

    if Path(bspline_data.__file__).resolve().parent != (BSPLINE_SRC / "data").resolve():
        raise RuntimeError(f"`data` resolved to {bspline_data.__file__}, not the BSplineEncoder package")

    if PRESETS[args.preset][1] != SUPPORTED_PRESETS[args.preset]:
        raise RuntimeError(f"BSplineEncoder preset {args.preset} writes to {PRESETS[args.preset][1]}, "
                           f"expected {SUPPORTED_PRESETS[args.preset]}")
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source = json.loads((raw_root / "SOURCE.json").read_text())
    summary = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "starvla_commit": git_head(REPO_ROOT), "bsplineencoder_commit": git_head(BSPLINE_SRC.parent),
               "raw_root": str(raw_root), "source_revision": source["revision"],
               "preset": args.preset, "fps": FPS, "spline": SPLINE,
               "validation_episodes": args.validation_episodes, "tasks": {}}
    for task in args.tasks:
        dataset, sidecar = dataset_path(output_root, task, args.preset), sidecar_path(output_root, task, args.preset)
        if args.skip_existing and dataset.exists() and sidecar.exists():
            print(f"[{task}] outputs exist; re-checking only", flush=True)
            reports = []
        else:
            config = config_from_mapping({
                "raw_root": str(raw_root),
                "tasks": [task],
                "task_description": DETAILED_TASK_INSTRUCTIONS[task] if args.detailed_instructions else None,
                "output_root": str(output_root),
                "setups": [args.preset],
                "cameras": {"global": "observation.images.global", "hand": "observation.images.hand"},
                "run": {"fps": FPS, "cleanup": True, "cleanup_motion": "auto"},
                "spline": {**SPLINE, "enabled": True, "validation_episodes": args.validation_episodes,
                           "output": str(sidecar)},
                "jobs": args.jobs,
                "overwrite": args.overwrite,
                "dry_run": args.dry_run,
            })
            reports = run(config, log=lambda text: print(f"[{task}] {text}", flush=True))
            if args.dry_run:
                print(json.dumps(reports, indent=2, default=str))
                continue
        geometry = check_encoder_geometry(sidecar)
        add_starvla_state_aliases(dataset)
        if args.detailed_instructions:
            replaced = set_task_instruction(dataset, DETAILED_TASK_INSTRUCTIONS[task])
            if replaced is not None:
                print(f"[{task}] instruction {replaced!r} -> {DETAILED_TASK_INSTRUCTIONS[task]!r}", flush=True)
        info = json.loads((dataset / "meta/info.json").read_text())
        summary["tasks"][task] = {
            "dataset": str(dataset), "sidecar": str(sidecar),
            "episodes": info["total_episodes"], "frames": info["total_frames"],
            "splits": info["splits"], "geometry": geometry,
            "instruction": json.loads((dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"],
            "pipeline_report": reports[0] if reports else None,
        }
        print(f"[{task}] {info['total_episodes']} episodes, {info['total_frames']} frames, "
              f"splits {info['splits']}; geometry OK", flush=True)
    if not args.dry_run:
        path = summary_path(output_root, args.preset)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(path.read_text()) if path.exists() else {"tasks": {}}
        for task, entry in summary["tasks"].items():
            # A --skip-existing re-check has no new pipeline report; keep the one from the build.
            if entry["pipeline_report"] is None:
                entry["pipeline_report"] = previous.get("tasks", {}).get(task, {}).get("pipeline_report")
        summary["tasks"] = {**previous.get("tasks", {}), **summary["tasks"]}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        os.replace(tmp, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
