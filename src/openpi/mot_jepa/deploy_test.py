from __future__ import annotations

import numpy as np
import pytest

from openpi.mot_jepa.action_parse import ACTION_DIM
from openpi.mot_jepa.action_parse import actions_from_state
from openpi.mot_jepa.deploy import ARM_JOINT_SLICE
from openpi.mot_jepa.deploy import GRIPPER_SLOT
from openpi.mot_jepa.deploy import chunk_to_absolute_qpos8


def test_constant_delta_advances_linearly():
    """THE test for the cumsum bug, and the one the plan called for.

    A policy emitting a fixed delta must drive the arm k*delta after k steps. Feed the same chunk
    through FTP-1's chunk-relative resolver instead and every target is base + delta -- the arm
    moves once and stops. Nothing raises either way, so only this assertion separates them.
    """
    horizon, delta = 6, 0.01
    chunk = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    chunk[:, ARM_JOINT_SLICE] = delta
    base = np.arange(8, dtype=np.float32)

    out = chunk_to_absolute_qpos8(chunk, base)
    for k in range(horizon):
        np.testing.assert_allclose(out[k, :7], base[:7] + (k + 1) * delta, rtol=0, atol=1e-6)
    # And state it as the negative: the single-base convention would give this instead.
    assert not np.allclose(out[-1, :7], base[:7] + delta), "targets did not integrate"


def test_gripper_is_passed_through_not_integrated():
    """The asymmetry. ABSOLUTE_COLUMNS exists so the gripper is never differenced.

    Integrating it would accumulate an open/close command into a runaway value while the arm
    looked perfectly correct -- the failure would read as a grasping problem, not a units bug.
    """
    horizon = 5
    chunk = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    chunk[:, GRIPPER_SLOT] = 0.3
    out = chunk_to_absolute_qpos8(chunk, np.full(8, 99.0, dtype=np.float32))
    np.testing.assert_allclose(out[:, 7], 0.3, rtol=0, atol=1e-6)


def test_round_trips_a_real_trajectory():
    """Verify against the motivating case: derive actions the way training does, then invert.

    A test built only from synthetic deltas agrees with any convention that integrates. This one
    starts from states, runs the SAME function the clip dataset uses to build training targets,
    and asserts the deployment path recovers the states those targets came from. If the two ever
    disagree the policy would be trained on one convention and executed under another.
    """
    rng = np.random.default_rng(0)
    steps = 9
    states = np.zeros((steps, ACTION_DIM), dtype=np.float32)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[ARM_JOINT_SLICE] = True
    mask[GRIPPER_SLOT] = True
    states[:, ARM_JOINT_SLICE] = np.cumsum(rng.normal(scale=0.02, size=(steps, 7)), axis=0)
    states[:, GRIPPER_SLOT] = rng.uniform(size=steps)

    actions = actions_from_state(states, mask)  # (steps - 1, 120), what training sees
    base = np.concatenate([states[0, ARM_JOINT_SLICE], [states[0, GRIPPER_SLOT]]]).astype(np.float32)

    out = chunk_to_absolute_qpos8(actions, base)
    np.testing.assert_allclose(out[:, :7], states[1:, ARM_JOINT_SLICE], rtol=0, atol=1e-5)
    np.testing.assert_allclose(out[:, 7], states[1:, GRIPPER_SLOT], rtol=0, atol=1e-5)


def test_rejects_wrong_shapes():
    """Deployment is where a silent shape slip is most expensive; make both cases raise."""
    with pytest.raises(ValueError, match="chunk must be"):
        chunk_to_absolute_qpos8(np.zeros((4, 7), dtype=np.float32), np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="qpos8_base must be"):
        chunk_to_absolute_qpos8(np.zeros((4, ACTION_DIM), dtype=np.float32), np.zeros(7, dtype=np.float32))
