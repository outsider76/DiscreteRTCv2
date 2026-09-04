"""QwenPI_v3 with a continuous uniform-left B-spline action representation."""

from __future__ import annotations

from typing import List

import numpy as np

from spline_encoder import UniformLeftBSplineConfig, create_encoder
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenPI_v3_UniformBSpline")
class QwenPI_v3_UniformBSpline(Qwen_PI_v3):
    """Train on spline controls and decode them before policy un-normalization.

    The neural architecture is QwenPI_v3 unchanged: only the action head's
    temporal axis represents control points rather than executable timesteps.
    ``predict_action`` converts the predicted normalized controls into a
    normalized executable trajectory, preserving the standard policy-server
    interface.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        representation = self.config.framework.get("action_representation")
        if representation is None:
            raise ValueError("framework.action_representation is required")

        encoder_config = UniformLeftBSplineConfig(
            action_dim=int(representation.action_dim),
            chunk_size=int(representation.executable_horizon),
            frequency_hz=float(representation.frequency_hz),
            degree=int(representation.degree),
            num_basis=int(representation.num_control_points),
            span_length_steps=int(representation.span_length_steps),
            vocab_size=int(representation.get("vocab_size", 256)),
            regularization=float(representation.get("regularization", 1e-4)),
            end_padding=bool(representation.get("end_padding", True)),
            implementation_version=str(
                representation.get(
                    "implementation_version", "uniform_left_extended_double_fit_v1"
                )
            ),
        )
        if encoder_config.mode != "uniform_left":
            raise ValueError(f"Expected uniform_left encoder, got {encoder_config.mode}")
        if self.action_horizon != encoder_config.num_basis:
            raise ValueError(
                "Action-head horizon must equal the number of B-spline controls: "
                f"{self.action_horizon} != {encoder_config.num_basis}"
            )
        if int(self.config.framework.action_model.action_dim) != encoder_config.action_dim:
            raise ValueError(
                "Action-head dimension must match B-spline action_dim: "
                f"{self.config.framework.action_model.action_dim} != {encoder_config.action_dim}"
            )

        encoder = create_encoder(encoder_config)
        self._bspline_decode_basis = np.asarray(encoder.basis, dtype=np.float32)
        expected_shape = (encoder_config.chunk_size, encoder_config.num_basis)
        if self._bspline_decode_basis.shape != expected_shape:
            raise ValueError(
                f"Unexpected B-spline decode basis {self._bspline_decode_basis.shape}; "
                f"expected {expected_shape}"
            )
        self.executable_action_horizon = encoder_config.chunk_size

    def decode_normalized_controls(self, controls: np.ndarray) -> np.ndarray:
        """Decode ``[B, C, D]`` controls to ``[B, T, D]`` actions."""
        values = np.asarray(controls, dtype=np.float32)
        expected_tail = (self.action_horizon, int(self.config.framework.action_model.action_dim))
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                f"Expected normalized controls shaped [B, {expected_tail[0]}, {expected_tail[1]}], "
                f"got {values.shape}"
            )
        decoded = np.einsum("tc,bcd->btd", self._bspline_decode_basis, values, optimize=True)
        if not np.isfinite(decoded).all():
            raise ValueError("Decoded B-spline actions contain non-finite values")
        return decoded.astype(np.float32, copy=False)

    def predict_action_parameters(self, examples: List[dict] = None, **kwargs) -> dict:
        """Return the normalized control points for representation-space eval."""
        return super().predict_action(examples=examples, **kwargs)

    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        predicted = self.predict_action_parameters(examples=examples, **kwargs)
        controls = predicted["normalized_actions"]
        return {"normalized_actions": self.decode_normalized_controls(controls)}
