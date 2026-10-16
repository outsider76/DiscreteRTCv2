"""Piper dataset registration for absolute joint-target imitation learning."""

import json
from pathlib import Path

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class PiperAbsoluteJointsDataConfig:
    """Two RGB views, 7-D proprioception, and 7-D absolute Piper commands."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.global", "video.hand"]
    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.joints", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    state_key_dims = {"state.joints": 6, "state.gripper": 1}
    action_key_dims = {"action.joints": 6, "action.gripper": 1}

    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
            ]
        )


class PiperAbsoluteJoints50StepDataConfig(PiperAbsoluteJointsDataConfig):
    """Piper absolute targets covering 1 second at the dataset's 50 Hz rate."""

    action_indices = list(range(50))


class PiperAbsoluteJoints30StepDataConfig(PiperAbsoluteJointsDataConfig):
    """Piper absolute targets covering 1 second at a dataset's 30 Hz rate."""

    action_indices = list(range(30))


def source_record_count(dataset_path: Path) -> int:
    """Rows over every episode in ``meta/episodes.jsonl``, ignoring ``episode_split``."""
    total = 0
    with open(Path(dataset_path) / "meta/episodes.jsonl", "r") as f:
        for line in f:
            if line.strip():
                total += int(json.loads(line)["length"])
    return total


class UniformBSplineLeRobotDataset(LeRobotSingleDataset):
    """Pair LeRobot observations with observation-aligned spline controls.

    The RGB/state/language observations come from the source LeRobot dataset.
    The raw action chunk is replaced before transforms run, so the control
    points use exactly the same per-dimension normalization as physical Piper
    actions.  The checkpoint can therefore decode in normalized coordinates
    and reuse the standard deployment un-normalizer afterward.
    """

    def __init__(
        self,
        *args,
        controls_dataset_path: str | Path,
        chunk_size: int = 20,
        num_basis: int = 13,
        action_dim: int = 7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controls_dataset_path = Path(controls_dataset_path).expanduser().resolve()
        self.spline_chunk_size = int(chunk_size)
        self.spline_num_basis = int(num_basis)
        self.control_action_dim = int(action_dim)
        self._controls_by_episode = self._load_controls()

    def _load_controls(self) -> dict[int, np.ndarray]:
        manifest_path = self.controls_dataset_path / "manifest.json"
        encoder_path = self.controls_dataset_path / "encoder.json"
        if not manifest_path.is_file() or not encoder_path.is_file():
            raise FileNotFoundError(
                "UniformBSpline dataset must contain manifest.json and encoder.json: "
                f"{self.controls_dataset_path}"
            )

        manifest = json.loads(manifest_path.read_text())
        encoder = json.loads(encoder_path.read_text())
        config = encoder.get("config", {})
        expected = {
            "mode": "uniform_left",
            "chunk_size": self.spline_chunk_size,
            "span_length_steps": 2,
            "num_basis": self.spline_num_basis,
            "action_dim": self.control_action_dim,
        }
        actual = {key: config.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Unexpected UniformBSpline encoder config: expected {expected}, got {actual}")
        if manifest.get("source_dataset"):
            source_name = Path(manifest["source_dataset"]).name
            if source_name != self.dataset_path.name:
                raise ValueError(
                    f"Spline controls were built from {source_name!r}, but observations come from "
                    f"{self.dataset_path.name!r}"
                )

        controls_by_episode: dict[int, np.ndarray] = {}
        total_records = 0
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            episode_id = int(trajectory_id)
            episode_path = self.controls_dataset_path / "data" / f"episode_{episode_id:06d}.npz"
            if not episode_path.is_file():
                raise FileNotFoundError(f"Missing spline controls: {episode_path}")
            with np.load(episode_path) as episode:
                frame_index = np.asarray(episode["frame_index"], dtype=np.int64)
                controls = np.asarray(episode["controls"], dtype=np.float32)
            expected_frames = np.arange(int(trajectory_length), dtype=np.int64)
            if not np.array_equal(frame_index, expected_frames):
                raise ValueError(
                    f"Spline frame alignment mismatch in episode {episode_id}: "
                    f"expected frames 0..{int(trajectory_length) - 1}, got {frame_index.shape}"
                )
            expected_shape = (
                int(trajectory_length),
                self.spline_num_basis,
                self.control_action_dim,
            )
            if controls.shape != expected_shape:
                raise ValueError(
                    f"Unexpected controls shape in episode {episode_id}: {controls.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.isfinite(controls).all():
                raise ValueError(f"Non-finite spline controls in episode {episode_id}")
            controls_by_episode[episode_id] = controls
            total_records += len(controls)

        # With ``episode_split`` only that split's episodes are loaded, so the
        # whole-sidecar count is checked against every source episode instead.
        expected_records = (
            source_record_count(self.dataset_path)
            if self._split_episode_ids is not None
            else total_records
        )
        if expected_records != int(manifest.get("record_count", -1)):
            raise ValueError(
                f"Spline record count mismatch: loaded {total_records} "
                f"(source dataset {expected_records}), manifest reports {manifest.get('record_count')}"
            )
        return controls_by_episode

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        controls = self._controls_by_episode[int(trajectory_id)][int(base_index)]
        data["action.joints"] = controls[:, :6].copy()
        data["action.gripper"] = controls[:, 6:].copy()
        return data


class PiperUniformLeftBSplineDataConfig(PiperAbsoluteJointsDataConfig):
    """Piper targets represented by 13 controls decoding to 20 steps at 25 Hz."""

    action_indices = list(range(13))

    def make_dataset(self, **kwargs):
        data_cfg = kwargs.get("data_cfg")
        controls_path = data_cfg.get("controls_dataset_path") if data_cfg is not None else None
        if not controls_path:
            raise ValueError(
                "datasets.vla_data.controls_dataset_path is required for piper_uniform_left_bspline"
            )
        kwargs.pop("dataset_name", None)
        return UniformBSplineLeRobotDataset(
            **kwargs,
            controls_dataset_path=controls_path,
            chunk_size=20,
            num_basis=13,
            action_dim=7,
        )


class PiperUniformLeftBSpline30StepDataConfig(PiperAbsoluteJointsDataConfig):
    """Eighteen uniform-left controls decoding to 30 physical action steps."""

    action_indices = list(range(18))

    def make_dataset(self, **kwargs):
        data_cfg = kwargs.get("data_cfg")
        controls_path = data_cfg.get("controls_dataset_path") if data_cfg is not None else None
        if not controls_path:
            raise ValueError(
                "datasets.vla_data.controls_dataset_path is required for "
                "piper_uniform_left_bspline_30step"
            )
        kwargs.pop("dataset_name", None)
        return UniformBSplineLeRobotDataset(
            **kwargs,
            controls_dataset_path=controls_path,
            chunk_size=30,
            num_basis=18,
            action_dim=7,
        )


class AdaptiveBSplineLeRobotDataset(LeRobotSingleDataset):
    """Observation-aligned adaptive controls with interleaved knot durations.

    Physical controls are normalized through the unchanged 7-D Piper action
    metadata. The duration column is normalized separately and appended only
    after the standard action transforms, preserving the canonical row order
    ``[control dims 0..6, duration]`` without contaminating action statistics.
    """

    def __init__(
        self,
        *args,
        controls_dataset_path: str | Path,
        duration_min: int,
        duration_max: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controls_dataset_path = Path(controls_dataset_path).expanduser().resolve()
        self.duration_min = int(duration_min)
        self.duration_max = int(duration_max)
        if self.duration_min < 1 or self.duration_max <= self.duration_min:
            raise ValueError("adaptive duration bounds must satisfy 1 <= min < max")
        self._parameters_by_episode = self._load_parameters()

    def _load_parameters(self) -> dict[int, np.ndarray]:
        manifest_path = self.controls_dataset_path / "manifest.json"
        encoder_path = self.controls_dataset_path / "encoder.json"
        if not manifest_path.is_file() or not encoder_path.is_file():
            raise FileNotFoundError(
                "AdaptiveBSpline dataset must contain manifest.json and encoder.json: "
                f"{self.controls_dataset_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        config = json.loads(encoder_path.read_text()).get("config", {})
        expected = {
            "mode": "adaptive_left",
            "chunk_size": 20,
            "span_length_steps": 2,
            "num_basis": 13,
            "action_dim": 7,
            "fit_tolerance": 1e-4,
            "knot_insertion_excluded_dimensions": [6],
        }
        actual = {key: config.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Unexpected AdaptiveBSpline encoder config: expected {expected}, got {actual}")
        if manifest.get("policy_layout") != "control_point_duration_interleaved_v1":
            raise ValueError("AdaptiveBSpline sidecar does not use canonical interleaving")
        if manifest.get("source_dataset"):
            source_name = Path(manifest["source_dataset"]).name
            if source_name != self.dataset_path.name:
                raise ValueError(
                    f"Adaptive parameters were built from {source_name!r}, but observations come from "
                    f"{self.dataset_path.name!r}"
                )

        parameters_by_episode: dict[int, np.ndarray] = {}
        total_records = 0
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            episode_id = int(trajectory_id)
            episode_path = self.controls_dataset_path / "data" / f"episode_{episode_id:06d}.npz"
            if not episode_path.is_file():
                raise FileNotFoundError(f"Missing adaptive parameters: {episode_path}")
            with np.load(episode_path) as episode:
                frame_index = np.asarray(episode["frame_index"], dtype=np.int64)
                parameters = np.asarray(
                    episode["control_duration_interleaved"], dtype=np.float32
                )
                controls = np.asarray(episode["controls"], dtype=np.float32)
                durations = np.asarray(episode["duration_steps"], dtype=np.int64)
            expected_frames = np.arange(int(trajectory_length), dtype=np.int64)
            if not np.array_equal(frame_index, expected_frames):
                raise ValueError(f"Adaptive frame alignment mismatch in episode {episode_id}")
            expected_shape = (int(trajectory_length), 13, 8)
            if parameters.shape != expected_shape:
                raise ValueError(
                    f"Unexpected adaptive parameter shape {parameters.shape}; expected {expected_shape}"
                )
            np.testing.assert_array_equal(parameters[..., :7], controls)
            np.testing.assert_array_equal(parameters[..., 7], durations)
            if durations.min() < self.duration_min or durations.max() > self.duration_max:
                raise ValueError(
                    f"Adaptive durations in episode {episode_id} exceed configured "
                    f"[{self.duration_min}, {self.duration_max}]"
                )
            if not np.isfinite(parameters).all():
                raise ValueError(f"Non-finite adaptive parameters in episode {episode_id}")
            parameters_by_episode[episode_id] = parameters
            total_records += len(parameters)
        expected_records = (
            source_record_count(self.dataset_path)
            if self._split_episode_ids is not None
            else total_records
        )
        if expected_records != int(manifest.get("record_count", -1)):
            raise ValueError("Adaptive parameter record count differs from manifest")
        return parameters_by_episode

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        parameters = self._parameters_by_episode[int(trajectory_id)][int(base_index)]
        data["action.joints"] = parameters[:, :6].copy()
        data["action.gripper"] = parameters[:, 6:7].copy()
        duration = parameters[:, 7:8]
        data["_adaptive_duration_normalized"] = (
            2.0 * (duration - self.duration_min) / (self.duration_max - self.duration_min) - 1.0
        ).astype(np.float32)
        return data

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        duration = np.asarray(data["_adaptive_duration_normalized"], dtype=np.float16)
        sample["action"] = np.concatenate((sample["action"], duration), axis=1).astype(
            np.float16, copy=False
        )
        return sample


class PiperAdaptiveLeftBSplineDataConfig(PiperAbsoluteJointsDataConfig):
    """Thirteen adaptive rows: seven controls followed by one duration."""

    action_indices = list(range(13))

    def make_dataset(self, **kwargs):
        data_cfg = kwargs.get("data_cfg")
        controls_path = data_cfg.get("controls_dataset_path") if data_cfg is not None else None
        if not controls_path:
            raise ValueError(
                "datasets.vla_data.controls_dataset_path is required for piper_adaptive_left_bspline"
            )
        duration_min = int(data_cfg.get("duration_min", 1))
        duration_max = int(data_cfg.get("duration_max", 21))
        kwargs.pop("dataset_name", None)
        return AdaptiveBSplineLeRobotDataset(
            **kwargs,
            controls_dataset_path=controls_path,
            duration_min=duration_min,
            duration_max=duration_max,
        )


ROBOT_TYPE_CONFIG_MAP = {
    "piper_abs_joints": PiperAbsoluteJointsDataConfig(),
    "piper_abs_joints_30step": PiperAbsoluteJoints30StepDataConfig(),
    "piper_abs_joints_50step": PiperAbsoluteJoints50StepDataConfig(),
    "piper_uniform_left_bspline": PiperUniformLeftBSplineDataConfig(),
    "piper_uniform_left_bspline_30step": PiperUniformLeftBSpline30StepDataConfig(),
    "piper_adaptive_left_bspline": PiperAdaptiveLeftBSplineDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "piper_pick_white_block": [
        ("piper_pick_white_block", 1.0, "piper_abs_joints"),
    ],
    "piper_pick_white_block_20260814_50hz_h50": [
        (
            "piper_pick_white_block_20260814_50hz",
            1.0,
            "piper_abs_joints_50step",
        ),
    ],
    "piper_pick_white_block_20260818_measured_50hz_h50": [
        (
            "20260818_piper_pick_white_block_50hz",
            1.0,
            "piper_abs_joints_50step",
        ),
    ],
    "piper_pick_white_block_20260818_25hz_uniform_left_bspline": [
        (
            "20260818_piper_pick_white_block_25hz",
            1.0,
            "piper_uniform_left_bspline",
        ),
    ],
    "piper_pick_white_block_20260818_25hz_adaptive_left_bspline": [
        (
            "20260818_piper_pick_white_block_25hz",
            1.0,
            "piper_adaptive_left_bspline",
        ),
    ],
    "cubes_classification_30hz_raw_h30": [
        (
            "cubes_classification_30hz",
            1.0,
            "piper_abs_joints_30step",
        ),
    ],
    "cubes_classification_30hz_uniform_left_bspline": [
        (
            "cubes_classification_30hz",
            1.0,
            "piper_uniform_left_bspline_30step",
        ),
    ],
    "stacking_cups_20260911_30hz_raw_h30": [
        (
            "stacking_cups/20260911",
            1.0,
            "piper_abs_joints_30step",
        ),
    ],
    "stacking_cups_20260911_30hz_uniform_left_bspline": [
        (
            "stacking_cups/20260911",
            1.0,
            "piper_uniform_left_bspline_30step",
        ),
    ],
    "stacking_cups_action_30hz_raw_h30": [
        (
            "stacking_cups_action_30hz",
            1.0,
            "piper_abs_joints_30step",
        ),
    ],
    "stacking_cups_action_30hz_uniform_left_bspline": [
        (
            "stacking_cups_action_30hz",
            1.0,
            "piper_uniform_left_bspline_30step",
        ),
    ],
}

# DiscreteRTC/dRTC ``piper_demos`` tasks processed by
# examples/realRobots/Piper/data_files/preprocess_piper_demos.py (preset
# ``joint-command``).  Each task has ONE LeRobot dataset; the raw and B-spline
# mixtures differ only in the action target (30 raw steps vs 18 controls from
# the sidecar given by ``datasets.vla_data.controls_dataset_path``).
# Alias: "classification_cube" / "cubes_classification" == ``classify_blocks``.
PIPER_DEMOS_PROCESSED_ROOT = "processed/piper_demos/ActionCommand/State_Joint"
PIPER_DEMOS_TASKS = ("classify_blocks", "hanging_mug", "stacking_cup")
for _task in PIPER_DEMOS_TASKS:
    DATASET_NAMED_MIXTURES[f"piper_demos_{_task}_30hz_raw_h30"] = [
        (f"{PIPER_DEMOS_PROCESSED_ROOT}/{_task}_30hz_cleanup", 1.0, "piper_abs_joints_30step"),
    ]
    DATASET_NAMED_MIXTURES[f"piper_demos_{_task}_30hz_uniform_left_bspline"] = [
        (f"{PIPER_DEMOS_PROCESSED_ROOT}/{_task}_30hz_cleanup", 1.0, "piper_uniform_left_bspline_30step"),
    ]
del _task
