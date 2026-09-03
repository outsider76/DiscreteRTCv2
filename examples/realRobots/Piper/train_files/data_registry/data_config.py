"""Piper dataset registration for absolute joint-target imitation learning."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
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


ROBOT_TYPE_CONFIG_MAP = {
    "piper_abs_joints": PiperAbsoluteJointsDataConfig(),
    "piper_abs_joints_50step": PiperAbsoluteJoints50StepDataConfig(),
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
}
