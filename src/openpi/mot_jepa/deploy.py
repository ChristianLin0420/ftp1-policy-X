"""Turning a MoT-JEPA action chunk into the absolute qpos targets UniVTAC executes.

This is the one conversion in the deployment path that fails *silently*. Everything else -- a
wrong image size, a missing module, a bad checkpoint -- raises. This does not: feed per-step
deltas through FTP-1's chunk-relative resolver and the commanded target simply never advances
past one step's motion, so the robot under-actuates, the episode fails, and the number looks like
a bad policy rather than a bad unit conversion.

The two conventions, measured from the code rather than assumed:

**MoT-JEPA** (:func:`openpi.mot_jepa.action_parse.states_to_actions_from_mask`) emits
``state[i+1] - state[i]`` -- a per-step FIRST DIFFERENCE -- for every live slot except the
columns in ``ABSOLUTE_COLUMNS`` (44 and 92, the grippers), which carry ``state[i+1]`` directly.

**FTP-1** (``UniVTAC/scripts/eval_ftp1.py:_resolve_univtac_abs_action_from_ftp1``) adds a SINGLE
``qpos8_base`` to every index of the chunk, because its actions are chunk-relative: each entry is
already an offset from the chunk's start, not from its predecessor.

So the arm needs a cumulative sum before re-basing, and the gripper needs neither -- it is already
absolute. That asymmetry is the part worth being careful about: applying the cumsum uniformly
would corrupt the gripper just as surely as omitting it corrupts the arm, and both produce
plausible-looking trajectories.
"""

from __future__ import annotations

import numpy as np

from openpi.mot_jepa.action_parse import ABSOLUTE_COLUMNS
from openpi.mot_jepa.action_parse import ACTION_DIM
from openpi.mot_jepa.action_parse import SLICES

#: UniVTAC executes 8 numbers: 7 arm joints then 1 gripper (``eval_ftp1.py:_get_qpos8``).
QPOS8_DIM = 8

#: Where those 8 live in the 120-slot FTP-1 layout. Read from SLICES rather than written as
#: literals so a layout change breaks the import instead of silently shifting the mapping.
ARM_JOINT_SLICE = slice(*SLICES["right-arm-joints"])
GRIPPER_SLOT = ABSOLUTE_COLUMNS[0]


def chunk_to_absolute_qpos8(chunk: np.ndarray, qpos8_base: np.ndarray) -> np.ndarray:
    """``(H, 120)`` MoT-JEPA chunk + current ``(8,)`` qpos -> ``(H, 8)`` absolute targets.

    The arm is integrated then re-based; the gripper is passed through. Returns one target per
    horizon step, so the caller can execute the chunk open-loop or re-plan at any index.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    base = np.asarray(qpos8_base, dtype=np.float32).reshape(-1)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"chunk must be (H, {ACTION_DIM}), got {chunk.shape}")
    if base.shape[0] != QPOS8_DIM:
        raise ValueError(f"qpos8_base must be ({QPOS8_DIM},), got {base.shape}")

    out = np.empty((chunk.shape[0], QPOS8_DIM), dtype=np.float32)
    # Integrate: target at step k is the base plus every delta up to and including k. Without the
    # cumsum each target would be base + one step, i.e. the arm would stop after the first move.
    out[:, :7] = base[:7] + np.cumsum(chunk[:, ARM_JOINT_SLICE], axis=0)
    # Already absolute -- ABSOLUTE_COLUMNS exists precisely so the gripper is not differenced.
    out[:, 7] = chunk[:, GRIPPER_SLOT]
    return out
