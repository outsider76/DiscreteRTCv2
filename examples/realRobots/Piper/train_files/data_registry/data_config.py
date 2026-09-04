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


class UniformBSplineLeRobotDataset(LeRobotSingleDataset):
    """Pair LeRobot observations with observation-aligned spline controls.

    The RGB/state/language observations come from the source LeRobot dataset.
    The raw action chunk is replaced before transforms run, so the 13 control
    points use exactly the same per-dimension normalization as physical Piper
    actions.  The checkpoint can therefore decode in normalized coordinates
    and reuse the standard deployment un-normalizer afterward.
    """

    def __init__(self, *args, controls_dataset_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.controls_dataset_path = Path(controls_dataset_path).expanduser().resolve()
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
            "chunk_size": 20,
            "span_length_steps": 2,
            "num_basis": 13,
            "action_dim": 7,
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
            if controls.shape != (int(trajectory_length), 13, 7):
                raise ValueError(
                    f"Unexpected controls shape in episode {episode_id}: {controls.shape}; "
                    f"expected {(int(trajectory_length), 13, 7)}"
                )
            if not np.isfinite(controls).all():
                raise ValueError(f"Non-finite spline controls in episode {episode_id}")
            controls_by_episode[episode_id] = controls
            total_records += len(controls)

        if total_records != int(manifest.get("record_count", -1)):
            raise ValueError(
                f"Spline record count mismatch: loaded {total_records}, "
                f"manifest reports {manifest.get('record_count')}"
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
        )


ROBOT_TYPE_CONFIG_MAP = {
    "piper_abs_joints": PiperAbsoluteJointsDataConfig(),
    "piper_abs_joints_50step": PiperAbsoluteJoints50StepDataConfig(),
    "piper_uniform_left_bspline": PiperUniformLeftBSplineDataConfig(),
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
}
