#!/usr/bin/env python3
"""Asynchronous Piper control using a training-time RTC hard prefix.

This entry point reuses the existing asynchronous controller but sends
``mode=simulated_delay``.  It never falls back silently to the ΠGDM sampler.
Dry-run is the default; ``--execute`` is required to publish robot commands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.realRobots.Piper.eval_files import piper_async_common as common


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAINING_RTC_STATS = (
    REPO_ROOT
    / "results/Checkpoints"
    / "piper_pick_white_block_20260818_qwenpi_training_rtc_d15_50hz_h50"
    / "dataset_statistics.json"
)


class TrainingRTCPolicyConnection(common.AsyncPolicyConnection):
    """RTC connection that only requests the learned hard-prefix sampler."""

    def __init__(self, host: str, port: int, timeout: float):
        super().__init__(host, port, timeout)
        rtc_mode = self.metadata.get("rtc_mode")
        if rtc_mode != "training_time_hard_prefix":
            raise RuntimeError(
                "The connected server is not a TrainingRTC server: "
                f"rtc_mode={rtc_mode!r}. Start run_policy_server_training_rtc.sh."
            )
        self.max_trained_delay = int(
            self.metadata.get("rtc_max_delay_steps", 0)
        )
        if self.max_trained_delay <= 0:
            raise RuntimeError(
                "TrainingRTC metadata has no positive rtc_max_delay_steps"
            )

    def predict_rtc(
        self,
        example: dict,
        previous_chunk: np.ndarray,
        inference_delay: int,
        suffix_length: int,
        unnorm_key: str,
        prefix_attention_schedule: str,
        max_guidance_weight: float,
    ) -> np.ndarray:
        del suffix_length, prefix_attention_schedule, max_guidance_weight
        requested_delay = int(inference_delay)
        conditioned_delay = min(requested_delay, self.max_trained_delay)
        if requested_delay != conditioned_delay:
            print(
                "WARNING: TrainingRTC delay clamp: "
                f"estimated={requested_delay} steps, "
                f"conditioned={conditioned_delay} steps, "
                f"trained_max={self.max_trained_delay}. "
                "Actions after the conditioned prefix are model-generated."
            )
        return self._predict(
            "infer_realtime",
            {
                "examples": [example],
                "prev_action_chunk": np.asarray(
                    previous_chunk, dtype=np.float32
                ),
                "inference_delay": conditioned_delay,
                "unnorm_key": unnorm_key,
                "mode": "simulated_delay",
            },
        )


def main() -> None:
    # run_async_controller constructs the connection from this module-level
    # hook. Replacing it here keeps all existing ΠGDM/TE files unchanged.
    common.AsyncPolicyConnection = TrainingRTCPolicyConnection
    parser = common.build_async_parser("rtc", __doc__)
    # Ignore one-time cold-start timing when forecasting later asynchronous
    # calls. Use the trained maximum as the initial conservative forecast;
    # measured four-step requests on the local RTX 5090 can take 11--14 steps.
    parser.set_defaults(
        stats=DEFAULT_TRAINING_RTC_STATS,
        initial_delay_steps=15,
        gripper_threshold=0.5,
        viewer_method="training-rtc",
    )
    args = parser.parse_args()
    common.run_async_controller("rtc", args)


if __name__ == "__main__":
    main()
