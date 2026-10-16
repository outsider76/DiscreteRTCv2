"""Representation-aware training-time RTC frameworks for B-spline policies.

TrainingRTC operates on the action head's temporal tokens.  For these models,
those tokens are 13 B-spline parameter rows rather than decoded robot actions.
The classes below retain each representation's normal decode path while
replacing only the parameter-compatible LayerwiseFM head implementation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from starVLA.model.framework.VLM4A.QwenPI_v3_AdaptiveBSpline import (
    QwenPI_v3_AdaptiveBSpline,
)
from starVLA.model.framework.VLM4A.QwenPI_v3_TrainingRTC import (
    Qwen_PI_v3_TrainingRTC,
)
from starVLA.model.framework.VLM4A.QwenPI_v3_UniformBSpline import (
    QwenPI_v3_UniformBSpline,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


class _BSplineTrainingRTCContract:
    """Validate and expose the parameter-row RTC contract."""

    def _validate_training_rtc_contract(self) -> None:
        representation = self.config.framework.action_representation
        conditioning_space = str(representation.rtc_conditioning_space)
        delay_unit = str(representation.rtc_delay_unit)
        if conditioning_space != "spline_parameter_rows":
            raise ValueError(
                "B-spline TrainingRTC requires rtc_conditioning_space="
                "spline_parameter_rows"
            )
        if delay_unit != "parameter_rows":
            raise ValueError("B-spline TrainingRTC delay unit must be parameter_rows")
        self.rtc_conditioning_space = conditioning_space
        self.rtc_delay_unit = delay_unit
        self.rtc_min_ready_rows = int(
            self.config.framework.action_model.rtc_min_delay_steps
        )
        self.rtc_max_ready_rows = int(
            self.config.framework.action_model.rtc_max_delay_steps
        )
        configured_min = int(representation.rtc_min_ready_rows)
        configured_max = int(representation.rtc_max_ready_rows)
        if configured_min != self.rtc_min_ready_rows:
            raise ValueError(
                "action-model and representation RTC minimum prefix disagree: "
                f"{self.rtc_min_ready_rows} != {configured_min}"
            )
        if configured_max != self.rtc_max_ready_rows:
            raise ValueError(
                "action-model and representation RTC maximum prefix disagree: "
                f"{self.rtc_max_ready_rows} != {configured_max}"
            )
        if not (
            0
            < self.rtc_min_ready_rows
            <= self.rtc_max_ready_rows
            < self.action_horizon
        ):
            raise ValueError("RTC ready-prefix range must be within the parameter horizon")

    def _validate_previous_parameters(self, values, parameter_dim: int) -> np.ndarray:
        previous = np.asarray(values, dtype=np.float32)
        expected = (self.action_horizon, parameter_dim)
        if previous.ndim != 3 or previous.shape[1:] != expected:
            raise ValueError(
                "B-spline TrainingRTC expects normalized previous spline parameters "
                f"with shape [B, {expected[0]}, {expected[1]}], got {previous.shape}. "
                "Decoded robot actions are not parameter rows."
            )
        return previous


@FRAMEWORK_REGISTRY.register("QwenPI_v3_UniformBSpline_TrainingRTC")
class QwenPI_v3_UniformBSpline_TrainingRTC(
    _BSplineTrainingRTCContract,
    QwenPI_v3_UniformBSpline,
    Qwen_PI_v3_TrainingRTC,
):
    """Uniform-left B-spline policy with parameter-row TrainingRTC."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._validate_training_rtc_contract()

    def predict_action_realtime(
        self,
        examples: List[dict] = None,
        prev_action_chunk_normalized=None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        if prev_action_chunk_normalized is None or inference_delay <= 0:
            return self.predict_action(examples=examples, **kwargs)
        previous = self._validate_previous_parameters(
            prev_action_chunk_normalized,
            int(self.config.framework.action_model.action_dim),
        )
        predicted = Qwen_PI_v3_TrainingRTC.predict_action_realtime(
            self,
            examples=examples,
            prev_action_chunk_normalized=previous,
            inference_delay=inference_delay,
            **kwargs,
        )["normalized_actions"]
        return {"normalized_actions": self.decode_normalized_controls(predicted)}


@FRAMEWORK_REGISTRY.register("QwenPI_v3_AdaptiveBSpline_TrainingRTC")
class QwenPI_v3_AdaptiveBSpline_TrainingRTC(
    _BSplineTrainingRTCContract,
    QwenPI_v3_AdaptiveBSpline,
    Qwen_PI_v3_TrainingRTC,
):
    """Adaptive-left B-spline policy with parameter-row TrainingRTC."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._validate_training_rtc_contract()

    def predict_action_realtime(
        self,
        examples: List[dict] = None,
        prev_action_chunk_normalized=None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        if prev_action_chunk_normalized is None or inference_delay <= 0:
            return self.predict_action(examples=examples, **kwargs)
        previous = self._validate_previous_parameters(
            prev_action_chunk_normalized,
            int(self.config.framework.action_model.action_dim),
        )
        predicted = Qwen_PI_v3_TrainingRTC.predict_action_realtime(
            self,
            examples=examples,
            prev_action_chunk_normalized=previous,
            inference_delay=inference_delay,
            **kwargs,
        )["normalized_actions"]
        decoded, action_lengths, uncapped_lengths = self.decode_normalized_parameters(
            predicted
        )
        return {
            "normalized_actions": decoded,
            "action_lengths": action_lengths,
            "action_lengths_uncapped": uncapped_lengths,
        }
