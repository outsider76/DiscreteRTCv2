from __future__ import annotations

import numpy as np
import pytest
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.model.framework.VLM4A.QwenPI_v3_AdaptiveBSpline import (
    QwenPI_v3_AdaptiveBSpline,
)
from starVLA.model.framework.VLM4A.QwenPI_v3_BSpline_TrainingRTC import (
    QwenPI_v3_AdaptiveBSpline_TrainingRTC,
    QwenPI_v3_UniformBSpline_TrainingRTC,
)
from starVLA.model.framework.VLM4A.QwenPI_v3_TrainingRTC import (
    Qwen_PI_v3_TrainingRTC,
)
from starVLA.model.framework.VLM4A.QwenPI_v3_UniformBSpline import (
    QwenPI_v3_UniformBSpline,
)
from starVLA.model.modules.action_model.TrainingRTC_LayerwiseFM_ActionHeader import (
    TrainingRTCLayerwiseFlowmatchingActionHead,
)


@pytest.mark.parametrize(
    ("framework", "representation"),
    [
        (QwenPI_v3_UniformBSpline_TrainingRTC, QwenPI_v3_UniformBSpline),
        (QwenPI_v3_AdaptiveBSpline_TrainingRTC, QwenPI_v3_AdaptiveBSpline),
    ],
)
def test_bspline_training_rtc_preserves_representation_and_rtc_inheritance(
    framework, representation
):
    assert issubclass(framework, representation)
    assert issubclass(framework, Qwen_PI_v3_TrainingRTC)


@pytest.mark.parametrize(
    ("framework", "parameter_dim"),
    [
        (QwenPI_v3_UniformBSpline_TrainingRTC, 7),
        (QwenPI_v3_AdaptiveBSpline_TrainingRTC, 8),
    ],
)
def test_bspline_training_rtc_rejects_decoded_actions_as_parameter_prefix(
    framework, parameter_dim
):
    instance = framework.__new__(framework)
    instance.action_horizon = 13
    valid = np.zeros((2, 13, parameter_dim), dtype=np.float32)
    np.testing.assert_array_equal(
        instance._validate_previous_parameters(valid, parameter_dim), valid
    )
    with pytest.raises(ValueError, match="Decoded robot actions are not parameter rows"):
        instance._validate_previous_parameters(
            np.zeros((2, 20, 7), dtype=np.float32), parameter_dim
        )


def test_policy_wrapper_refuses_parameter_row_rtc_through_action_api():
    wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
    wrapper._supports_rtc = True
    wrapper._rtc_conditioning_space = "spline_parameter_rows"
    with pytest.raises(RuntimeError, match="representation-aware parameter prefix adapter"):
        wrapper.predict_action_realtime(
            examples=[{}],
            prev_action_chunk=np.zeros((20, 7), dtype=np.float32),
            inference_delay=2,
        )


def test_training_rtc_samples_completed_prefix_from_four_through_nine():
    head = TrainingRTCLayerwiseFlowmatchingActionHead.__new__(
        TrainingRTCLayerwiseFlowmatchingActionHead
    )
    head.rtc_min_delay_steps = 4
    head.rtc_max_delay_steps = 9
    torch.manual_seed(42)
    prefixes = head._sample_training_delays(4096, torch.device("cpu"))
    assert prefixes.min().item() == 4
    assert prefixes.max().item() == 9
    assert set(prefixes.tolist()) == set(range(4, 10))
    generated_rows = 13 - prefixes
    assert generated_rows.min().item() == 4
    assert generated_rows.max().item() == 9
