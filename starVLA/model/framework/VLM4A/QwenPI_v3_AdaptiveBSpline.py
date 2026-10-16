"""QwenPI_v3 with interleaved adaptive-left B-spline parameters."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.interpolate import BSpline

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.tools import FRAMEWORK_REGISTRY


def decode_adaptive_parameters(
    parameters: np.ndarray,
    *,
    action_dim: int,
    degree: int,
    duration_min: int,
    duration_max: int,
    output_horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode normalized rows on the native grid, preserving native lengths.

    The first ``num_controls - degree`` durations define each plan's true
    executable length. A dense return tensor is padded by repeating the last
    native sample; callers must use the returned lengths as the validity mask.
    ``output_horizon`` is only a memory/safety ceiling, never a resampling size.
    """
    values = np.asarray(parameters, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != action_dim + 1:
        raise ValueError(
            f"Expected adaptive parameters [B, C, {action_dim + 1}], got {values.shape}"
        )
    if duration_min < 1 or duration_max <= duration_min:
        raise ValueError("invalid duration normalization bounds")
    if values.shape[1] <= degree:
        raise ValueError("adaptive control count must exceed degree")
    if output_horizon is not None and output_horizon < 1:
        raise ValueError("output_horizon must be positive")
    controls = values[..., :action_dim]
    duration_normalized = values[..., action_dim]
    durations = np.rint(
        (duration_normalized + 1.0) * 0.5 * (duration_max - duration_min)
        + duration_min
    ).astype(np.int64)
    durations = np.clip(durations, duration_min, duration_max)
    executable_rows = values.shape[1] - degree
    native_horizons = durations[:, :executable_rows].sum(axis=1, dtype=np.int64)
    dense_horizon = (
        int(native_horizons.max()) if output_horizon is None else int(output_horizon)
    )
    decoded = np.empty((values.shape[0], dense_horizon, action_dim), dtype=np.float32)
    for batch_index in range(values.shape[0]):
        distinct = np.concatenate(([0], np.cumsum(durations[batch_index])))
        knots = np.concatenate((np.zeros(degree + 1, dtype=np.int64), distinct[1:]))
        native_horizon = int(native_horizons[batch_index])
        decoded_count = min(native_horizon, dense_horizon)
        spline = BSpline(
            knots,
            controls[batch_index],
            degree,
            extrapolate=False,
            axis=0,
        )
        decoded[batch_index, :decoded_count] = spline(
            np.arange(decoded_count, dtype=np.float64)
        )
        if decoded_count < dense_horizon:
            decoded[batch_index, decoded_count:] = decoded[batch_index, decoded_count - 1]
    if not np.isfinite(decoded).all():
        raise ValueError("Decoded adaptive B-spline actions contain non-finite values")
    return decoded, durations, native_horizons


@FRAMEWORK_REGISTRY.register("QwenPI_v3_AdaptiveBSpline")
class QwenPI_v3_AdaptiveBSpline(Qwen_PI_v3):
    """Predict 13 interleaved control/duration rows and decode a safe prefix."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        representation = self.config.framework.get("action_representation")
        if representation is None:
            raise ValueError("framework.action_representation is required")
        if str(representation.type) != "continuous_adaptive_bspline":
            raise ValueError("adaptive framework requires continuous_adaptive_bspline")
        if str(representation.mode) != "adaptive_left":
            raise ValueError("adaptive framework requires adaptive_left mode")
        if str(representation.policy_layout) != "control_point_duration_interleaved_v1":
            raise ValueError("adaptive framework requires canonical control/duration interleaving")
        self.spline_action_dim = int(representation.action_dim)
        self.spline_degree = int(representation.degree)
        self.spline_frequency_hz = float(representation.frequency_hz)
        self.nominal_chunk_size = int(representation.nominal_chunk_size)
        self.nominal_span_length_steps = int(representation.nominal_span_length_steps)
        self.fit_tolerance = float(representation.fit_tolerance)
        self.knot_insertion_excluded_dimensions = tuple(
            int(value) for value in representation.knot_insertion_excluded_dimensions
        )
        self.spline_tokenizer_id = str(representation.tokenizer_id)
        self.duration_min = int(representation.duration_min)
        self.duration_max = int(representation.duration_max)
        if str(representation.execution_horizon_mode) != "predicted_duration_sum":
            raise ValueError("adaptive-left requires predicted_duration_sum execution")
        self.max_native_horizon = int(representation.max_native_horizon)
        if self.max_native_horizon < 1:
            raise ValueError("max_native_horizon must be positive")
        self.num_control_points = int(representation.num_control_points)
        parameter_dim = int(representation.parameter_dim)
        if self.spline_frequency_hz <= 0 or self.fit_tolerance <= 0:
            raise ValueError("adaptive frequency and fit tolerance must be positive")
        if (
            (self.num_control_points - self.spline_degree)
            * self.nominal_span_length_steps
            != self.nominal_chunk_size
        ):
            raise ValueError("adaptive nominal span layout does not match chunk size")
        if self.knot_insertion_excluded_dimensions != (6,):
            raise ValueError("this Piper representation must exclude only gripper dimension 6")
        if not self.spline_tokenizer_id:
            raise ValueError("adaptive tokenizer_id must be nonempty")
        if self.action_horizon != self.num_control_points:
            raise ValueError("Action-head horizon must equal adaptive control count")
        if parameter_dim != self.spline_action_dim + 1:
            raise ValueError("Adaptive parameter_dim must be action_dim + one duration")
        if int(self.config.framework.action_model.action_dim) != parameter_dim:
            raise ValueError("Action-head dimension must match adaptive parameter_dim")

    def decode_normalized_parameters(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        decoded, _, native_horizons = decode_adaptive_parameters(
            parameters,
            action_dim=self.spline_action_dim,
            degree=self.spline_degree,
            duration_min=self.duration_min,
            duration_max=self.duration_max,
            output_horizon=self.max_native_horizon,
        )
        effective_horizons = np.minimum(native_horizons, self.max_native_horizon)
        batch_horizon = int(effective_horizons.max())
        return decoded[:, :batch_horizon], effective_horizons, native_horizons

    def predict_action_parameters(self, examples: List[dict] = None, **kwargs) -> dict:
        return super().predict_action(examples=examples, **kwargs)

    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        predicted = self.predict_action_parameters(examples=examples, **kwargs)
        decoded, action_lengths, uncapped_lengths = self.decode_normalized_parameters(
            predicted["normalized_actions"]
        )
        return {
            "normalized_actions": decoded,
            "action_lengths": action_lengths,
            "action_lengths_uncapped": uncapped_lengths,
        }
