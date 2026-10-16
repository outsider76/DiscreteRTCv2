from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.model.framework.VLM4A.QwenPI_v3_AdaptiveBSpline import (
    decode_adaptive_parameters,
)


def _duration_to_normalized(values, low=1, high=21):
    values = np.asarray(values, dtype=np.float32)
    return 2.0 * (values - low) / (high - low) - 1.0


def test_adaptive_decode_preserves_native_lengths_and_only_pads_tensor():
    rng = np.random.default_rng(42)
    controls = rng.normal(size=(2, 13, 7)).astype(np.float32)
    durations = np.asarray([
        [1] * 13,
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3],
    ])
    parameters = np.concatenate(
        (controls, _duration_to_normalized(durations)[..., None]), axis=2
    )
    decoded, recovered_durations, native_horizons = decode_adaptive_parameters(
        parameters,
        action_dim=7,
        degree=3,
        duration_min=1,
        duration_max=21,
        output_horizon=32,
    )
    np.testing.assert_array_equal(recovered_durations, durations)
    np.testing.assert_array_equal(native_horizons, durations[:, :10].sum(axis=1))
    for index, native_horizon in enumerate(native_horizons):
        distinct = np.r_[0, np.cumsum(durations[index])]
        knots = np.r_[np.zeros(4, dtype=np.int64), distinct[1:]]
        expected = BSpline(knots, controls[index], 3, extrapolate=False)(
            np.arange(native_horizon)
        )
        np.testing.assert_allclose(
            decoded[index, :native_horizon], expected, rtol=2e-6, atol=2e-6
        )
        np.testing.assert_array_equal(
            decoded[index, native_horizon:],
            np.repeat(
                decoded[index, native_horizon - 1][None],
                decoded.shape[1] - native_horizon,
                axis=0,
            ),
        )


class _IdentityProcessor:
    def unapply_actions(self, values):
        return np.asarray(values)


class _FakeFramework:
    def __init__(self, output):
        self.output = output

    def predict_action(self, **_kwargs):
        return self.output


def _bare_wrapper(output, *, variable):
    wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
    wrapper._framework = _FakeFramework(output)
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._norm_processors = {"new_embodiment": _IdentityProcessor()}
    wrapper._variable_action_horizon = variable
    wrapper._gripper_constraint = None
    return wrapper


def test_policy_wrapper_propagates_adaptive_action_lengths():
    actions = np.zeros((2, 5, 7), dtype=np.float32)
    wrapper = _bare_wrapper(
        {
            "normalized_actions": actions,
            "action_lengths": np.asarray([3, 5]),
            "action_lengths_uncapped": np.asarray([3, 8]),
        },
        variable=True,
    )
    result = wrapper.predict_action(examples=[{}, {}])
    np.testing.assert_array_equal(result["actions"], actions)
    np.testing.assert_array_equal(result["action_lengths"], [3, 5])
    np.testing.assert_array_equal(result["action_lengths_uncapped"], [3, 8])


def test_policy_wrapper_keeps_fixed_horizon_output_unchanged():
    actions = np.zeros((1, 20, 7), dtype=np.float32)
    wrapper = _bare_wrapper({"normalized_actions": actions}, variable=False)
    result = wrapper.predict_action(examples=[{}])
    assert set(result) == {"actions"}
    np.testing.assert_array_equal(result["actions"], actions)
