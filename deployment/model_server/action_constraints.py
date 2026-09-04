"""Small, deterministic constraints applied to decoded physical actions.

These constraints intentionally run after action-representation decoding and
training-time un-normalization.  In particular, a B-spline remains a genuine
B-spline; only the command sent to the environment is projected onto the
environment's admissible gripper command set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GripperConstraint:
    """Project one action dimension onto a safe continuous or binary range."""

    mode: str = "binary"
    dimension: int = -1
    minimum: float = 0.0
    maximum: float = 1.0
    threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.mode not in {"clip", "binary"}:
            raise ValueError(f"gripper constraint mode must be 'clip' or 'binary', got {self.mode!r}")
        if not np.isfinite([self.minimum, self.maximum, self.threshold]).all():
            raise ValueError("gripper constraint values must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("gripper constraint minimum must be smaller than maximum")
        if self.mode == "binary" and not self.minimum < self.threshold < self.maximum:
            raise ValueError("binary gripper threshold must lie strictly inside [minimum, maximum]")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "GripperConstraint":
        return cls(
            mode=str(config.get("mode", "binary")),
            dimension=int(config.get("dimension", -1)),
            minimum=float(config.get("minimum", 0.0)),
            maximum=float(config.get("maximum", 1.0)),
            threshold=float(config.get("threshold", 0.3)),
        )

    def apply(self, actions: np.ndarray) -> np.ndarray:
        """Return a constrained copy of an ``[..., action_dim]`` array."""
        values = np.asarray(actions)
        if values.ndim < 1:
            raise ValueError(f"actions must have at least one dimension, got {values.shape}")
        dimension = self.dimension % values.shape[-1]
        result = values.copy()
        gripper = np.clip(result[..., dimension], self.minimum, self.maximum)
        if self.mode == "binary":
            gripper = np.where(gripper > self.threshold, self.maximum, self.minimum)
        result[..., dimension] = gripper
        return result

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "mode": self.mode,
            "dimension": self.dimension,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "threshold": self.threshold,
        }


__all__ = ["GripperConstraint"]
