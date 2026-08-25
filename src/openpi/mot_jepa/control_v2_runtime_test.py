from __future__ import annotations

import numpy as np
import pytest

from openpi.mot_jepa.control_v2_runtime import DEFAULT_ENSEMBLE_K
from openpi.mot_jepa.control_v2_runtime import GRIPPER_INDEX
from openpi.mot_jepa.control_v2_runtime import J5_INDEX
from openpi.mot_jepa.control_v2_runtime import QPOS8_DIM
from openpi.mot_jepa.control_v2_runtime import V2_HORIZON
from openpi.mot_jepa.control_v2_runtime import SafetyLimiter
from openpi.mot_jepa.control_v2_runtime import SafetyLimits
from openpi.mot_jepa.control_v2_runtime import TemporalEnsemblerV2
from openpi.mot_jepa.control_v2_runtime import active_mixed_chunk_to_absolute
from openpi.mot_jepa.control_v2_runtime import paired_n100_acceptance
from openpi.mot_jepa.control_v2_runtime import qualify_control_trace
from openpi.mot_jepa.control_v2_runtime import safe_hold_chunk


def _limits(*, max_delta: float = 10.0, joint_bound: float = 4.0) -> SafetyLimits:
    return SafetyLimits(
        joint_lower=np.full(QPOS8_DIM, -joint_bound),
        joint_upper=np.full(QPOS8_DIM, joint_bound),
        max_delta=np.full(QPOS8_DIM, max_delta),
    )


def _chunk(value: float, *, gripper: float = 0.02) -> np.ndarray:
    chunk = np.zeros((V2_HORIZON, QPOS8_DIM), dtype=np.float32)
    chunk[:, :7] = value
    chunk[:, GRIPPER_INDEX] = gripper
    return chunk


def test_active_mixed_chunk_skips_placeholder_and_does_not_integrate_offsets() -> None:
    base = np.arange(QPOS8_DIM, dtype=np.float32)
    chunk = _chunk(0.0, gripper=0.25)
    chunk[0] = 999.0  # A malicious placeholder must have no effect.
    chunk[1, :7] = 0.1
    chunk[2, :7] = 0.2

    absolute = active_mixed_chunk_to_absolute(chunk, base)

    assert absolute.shape == (31, QPOS8_DIM)
    np.testing.assert_allclose(absolute[0, :7], base[:7] + 0.1)
    np.testing.assert_allclose(absolute[1, :7], base[:7] + 0.2)
    assert absolute[0, GRIPPER_INDEX] == pytest.approx(0.25)


def test_safe_hold_chunk_is_finite_and_resolves_to_exact_hold() -> None:
    qpos = np.linspace(-0.3, 0.4, QPOS8_DIM, dtype=np.float32)

    chunk = safe_hold_chunk(qpos)
    absolute = active_mixed_chunk_to_absolute(chunk, qpos)

    np.testing.assert_allclose(absolute, np.broadcast_to(qpos, absolute.shape))


def test_chunk_conversion_rejects_wrong_shape_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="shape"):
        active_mixed_chunk_to_absolute(np.zeros((31, 8)), np.zeros(8))
    bad = np.zeros((32, 8))
    bad[4, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        active_mixed_chunk_to_absolute(bad, np.zeros(8))


def test_temporal_ensemble_matches_official_oldest_to_newest_weights() -> None:
    runtime = TemporalEnsemblerV2(_limits(joint_bound=100.0))
    current = np.zeros(QPOS8_DIM, dtype=np.float32)
    runtime.reset(current)

    first = _chunk(0.0)
    first[1, :7] = 1.0
    first[2, :7] = 2.0
    np.testing.assert_allclose(runtime.step(first, current)[:7], 1.0)

    second = _chunk(0.0)
    second[1, :7] = 10.0
    expected_weights = np.exp(-DEFAULT_ENSEMBLE_K * np.arange(2, dtype=np.float32))
    expected_weights /= expected_weights.sum()
    expected = 2.0 * expected_weights[0] + 10.0 * expected_weights[1]

    output = runtime.step(second, current)

    np.testing.assert_allclose(output[:7], expected, rtol=1e-6)
    assert runtime.last_debug is not None
    assert runtime.last_debug.candidate_count == 2
    np.testing.assert_allclose(runtime.last_debug.weights, expected_weights)


def test_temporal_ensemble_uses_only_first_twenty_executable_rows() -> None:
    runtime = TemporalEnsemblerV2(_limits(max_delta=100.0))
    current = np.zeros(8, dtype=np.float32)
    runtime.reset(current)

    for step in range(21):
        chunk = _chunk(float(step))
        runtime.step(chunk, current)

    assert runtime.last_debug is not None
    assert runtime.last_debug.candidate_count == 20


def test_nonfinite_chunk_and_observation_clear_history_and_return_finite_hold() -> None:
    runtime = TemporalEnsemblerV2(_limits())
    current = np.zeros(8, dtype=np.float32)
    current[GRIPPER_INDEX] = 0.02
    runtime.reset(current)
    runtime.step(_chunk(0.1), current)

    bad_chunk = _chunk(0.2)
    bad_chunk[3, 4] = np.nan
    np.testing.assert_array_equal(runtime.step(bad_chunk, current), current)
    assert runtime.counters.nonfinite_fallbacks == 1
    assert runtime.last_debug is not None
    assert runtime.last_debug.used_fallback

    bad_current = current.copy()
    bad_current[0] = np.inf
    last_safe = runtime.safe_hold()
    np.testing.assert_array_equal(runtime.step(_chunk(0.3), bad_current), last_safe)
    assert np.isfinite(runtime.safe_hold()).all()
    assert runtime.counters.nonfinite_fallbacks == 2


def test_first_nonfinite_observation_cannot_invent_a_hold() -> None:
    runtime = TemporalEnsemblerV2(_limits())
    with pytest.raises(RuntimeError, match="first qpos8_current must be finite"):
        runtime.step(_chunk(0.0), np.full(8, np.nan))


def test_safety_limiter_counts_rate_and_joint_clamps() -> None:
    limits = SafetyLimits(
        joint_lower=np.full(8, -1.0),
        joint_upper=np.full(8, 1.0),
        max_delta=np.full(8, 0.1),
    )
    limiter = SafetyLimiter(limits)
    desired = np.zeros(8, dtype=np.float32)
    desired[0] = 2.0  # absolute and rate clamp
    desired[1] = 0.5  # rate clamp only

    output = limiter.apply(desired, np.zeros(8, dtype=np.float32))

    np.testing.assert_allclose(output[:2], [0.1, 0.1])
    assert limiter.counters.command_count == 1
    assert limiter.counters.joint_clamp_commands == 1
    assert limiter.counters.joint_clamped_values == 1
    assert limiter.counters.rate_clamp_commands == 1
    assert limiter.counters.rate_clamped_values == 2
    assert limiter.counters.clamp_commands == 2


def test_safety_limiter_counts_final_joint_clip_from_out_of_bounds_current() -> None:
    limiter = SafetyLimiter(
        SafetyLimits(
            joint_lower=(-1.0,) * 8,
            joint_upper=(1.0,) * 8,
            max_delta=(0.01,) * 8,
        )
    )
    current = np.zeros(8, dtype=np.float32)
    current[0] = 1.05
    desired = np.zeros(8, dtype=np.float32)
    desired[0] = 0.99

    safe = limiter.apply(desired, current)

    assert safe[0] == pytest.approx(1.0)
    assert limiter.counters.rate_clamp_commands == 1
    assert limiter.counters.rate_clamped_values == 1
    assert limiter.counters.joint_clamp_commands == 1
    assert limiter.counters.joint_clamped_values == 1
    assert limiter.counters.clamp_commands == 2


def test_safety_limiter_nonfinite_desired_is_a_hold() -> None:
    limiter = SafetyLimiter(_limits())
    current = np.linspace(0.0, 0.7, 8, dtype=np.float32)
    desired = current.copy()
    desired[2] = np.inf

    np.testing.assert_array_equal(limiter.apply(desired, current), current)
    assert limiter.counters.nonfinite_fallbacks == 1


def _passing_trace() -> tuple[np.ndarray, np.ndarray]:
    before = np.zeros((4, 8), dtype=np.float32)
    sent = before.copy()
    sent[:, GRIPPER_INDEX] = [0.001, 0.002, 0.003, 0.004]
    sent[:, J5_INDEX] = [-0.1, -0.05, 0.05, 0.1]
    return before, sent


def test_trace_qualification_requires_reversal_and_no_other_violation() -> None:
    before, sent = _passing_trace()
    result = qualify_control_trace(
        before,
        sent,
        _limits(),
        first_gripper_jump_max=0.005,
    )

    assert result.passed
    assert result.j5_negative_to_positive_reversal
    assert not result.j5_limit_contact
    assert not result.first_gripper_jump
    assert not result.clamp_used


@pytest.mark.parametrize("failure", ["j5_limit", "gripper_jump", "no_reversal", "clamp", "nonfinite"])
def test_trace_qualification_reports_each_failure(failure: str) -> None:
    before, sent = _passing_trace()
    clamp_count = 0
    if failure == "j5_limit":
        sent[-1, J5_INDEX] = 4.0
    elif failure == "gripper_jump":
        sent[0, GRIPPER_INDEX] = 0.02
    elif failure == "no_reversal":
        sent[:, J5_INDEX] = -0.1
    elif failure == "clamp":
        clamp_count = 1
    elif failure == "nonfinite":
        sent[1, 0] = np.nan

    result = qualify_control_trace(
        before,
        sent,
        _limits(),
        clamp_count=clamp_count,
        first_gripper_jump_max=0.005,
    )

    assert not result.passed


def test_paired_n100_acceptance_passes_at_one_official_only_failure() -> None:
    official = np.ones(100, dtype=bool)
    student = np.ones(100, dtype=bool)
    student[0] = False

    result = paired_n100_acceptance(official, student)

    assert result.accepted
    assert result.student_successes == 99
    assert result.official_only_discordances == 1
    assert result.discordance_upper_95 == pytest.approx(0.0465598, rel=1e-5)


def test_paired_n100_acceptance_fails_at_two_student_failures() -> None:
    official = np.ones(100, dtype=bool)
    student = np.ones(100, dtype=bool)
    student[:2] = False

    result = paired_n100_acceptance(official, student)

    assert not result.accepted
    assert result.student_successes == 98
    assert result.discordance_upper_95 > 0.05


def test_paired_n100_acceptance_requires_boolean_paired_n100_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        paired_n100_acceptance(np.ones(99, dtype=bool), np.ones(99, dtype=bool))
    with pytest.raises(TypeError, match="booleans"):
        paired_n100_acceptance(np.ones(100, dtype=np.int8), np.ones(100, dtype=np.int8))
