"""Canonical parsing of FTP-1 proprioception into a unified action space.

No domain in the pretraining release carries an ``action`` array. What every domain carries
is proprioception, and it is as heterogeneous as the tactile streams were:

===========================  =======  ==============================================
key                          domains  note
===========================  =======  ==============================================
``{side}_hand_joints``        15/15   width 1 (gripper), 6 (MotionTrans), 22 (sharpa)
``{side}_hand_joints_idx``    15/15   the FTP-1 hand slot of each column
``{side}_wrist_pose``         11/15   ``(T, 6)`` position + rotation **vector**
``{side}_arm_joints``          6/15   ``(T, 7)``
``camera_ego_pose``            6/15   head/camera pose, same 6-D convention
``supplementary_joints``       1/15   sharpa only, ``(T, 9)``
===========================  =======  ==============================================

Rather than invent a layout, this module targets **FTP-1's own 120-D action space** --
``2 * 48 + 9 + 15``, with slot boundaries taken from
:func:`openpi.ftp1_action_groups.get_ftp1_action_group_slices` rather than re-derived here --
plus FTP-1's per-dimension ``action_mask`` convention for absent groups. That is what makes a
world model trained on these actions transferable to the policy later.

Three things this module exists to get right:

1. **``hand_joints_idx`` is a scatter, not a slice.** MotionTrans records 6 hand values whose
   slots are ``[1, 2, 7, 12, 17, 22]`` and sharpa records 22 whose slots skip 5, 10, 15 and
   20. Writing them contiguously would put every finger in the wrong place, and nothing
   downstream would say so. ``dataset_zarr.py:1419`` uses the raw index value as the offset,
   so slots are **not** decremented here either.

2. **The camera projection cancels, so it is deliberately not stored.** FTP-1 expresses the
   wrist in camera frame via ``cam_proj = inv(pose_to_mat(camera_ego_pose[idx]))``
   (``dataset_zarr.py:1660``), a single 4x4 constant per sample. For a *relative* action
   ``inv(cam_proj @ P_t) @ (cam_proj @ P_t') == inv(P_t) @ P_t'``: the projection drops out
   exactly. Storing world-frame absolute poses therefore loses nothing an action needs, and
   avoids baking in an observation index the builder does not know.

3. **The gripper is absolute; every other joint is relative.** FTP-1's default
   ``action_joint_rep="mix"`` makes hand slot 28 -- and only slot 28 -- an absolute width
   target (``dataset_zarr.py:58,85-90``). A gripper command is a position, not an increment,
   and differencing it produces an action that cannot be executed.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import zarr

from openpi.ftp1_action_groups import get_ftp1_action_group_slices
from openpi.pose_utils import mat_to_pose9d
from openpi.pose_utils import pose10d_to_mat
from openpi.pose_utils import pose_to_mat

logger = logging.getLogger(__name__)

#: Mirrors ``models_pytorch/ftp1_model_config.py``. Duplicated rather than imported because
#: that module pulls in ``openpi.models.gemma`` (and therefore jax) at import time, which is
#: unacceptable in a DataLoader worker. ``action_parse_test.py`` asserts the two agree, so
#: drift fails a test rather than silently reshaping the action space.
SINGLE_ARM_JOINT_DIM = 7
SINGLE_HAND_JOINT_DIM = 32
SINGLE_ARM_ACTION_REP_DIM = 9 + SINGLE_ARM_JOINT_DIM + SINGLE_HAND_JOINT_DIM  # 48
RESERVED_ACTION_DIM = 15

#: ``dataset_zarr.FTP1_GRIPPER_HAND_SLOT_INDEX``. Absolute under ``action_joint_rep="mix"``.
GRIPPER_HAND_SLOT = 28

#: Every slot boundary below is *read from* FTP-1's own group map rather than recomputed, so
#: a layout change there propagates here instead of silently disagreeing.
SLICES = get_ftp1_action_group_slices(SINGLE_ARM_ACTION_REP_DIM, SINGLE_ARM_JOINT_DIM, RESERVED_ACTION_DIM)

ACTION_DIM = SLICES["supplementary-joints"][1]  # 120
HEAD_START = SLICES["head-track-pos"][0]  # 96
SUPP_START = SLICES["supplementary-joints"][0]  # 105

#: Sides in FTP-1 slot order: right occupies ``[0, 48)``, left ``[48, 96)``.
SIDES: tuple[tuple[str, int], ...] = (
    ("right", SLICES["right-wrist-pos"][0]),
    ("left", SLICES["left-wrist-pos"][0]),
)

#: Sensor glitches are orders of magnitude above +-pi (``dataset_zarr.py:228``).
JOINT_GLITCH_THRESH = 100.0


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    """How one side's proprioception maps into its 48-slot block."""

    side: str
    base: int
    """Column of the 120-D vector where this arm's block starts."""
    hand_key: str
    hand_slots: tuple[int, ...]
    """FTP-1 hand slot for each column of ``hand_key``, from ``{side}_hand_joints_idx``."""
    wrist_key: str | None = None
    arm_key: str | None = None
    arm_width: int = 0

    @property
    def wrist_slice(self) -> slice:
        return slice(self.base, self.base + 9)

    @property
    def arm_slice(self) -> slice:
        return slice(self.base + 9, self.base + 9 + self.arm_width)

    @property
    def hand_columns(self) -> np.ndarray:
        return self.base + 9 + SINGLE_ARM_JOINT_DIM + np.asarray(self.hand_slots, dtype=np.int64)


@dataclasses.dataclass(frozen=True)
class StoreStateSpec:
    """Every proprioceptive stream of one store, in FTP-1 slot terms."""

    arms: tuple[ArmSpec, ...] = ()
    head_key: str | None = None
    supp_key: str | None = None
    supp_width: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.arms and self.head_key is None and self.supp_key is None


def replace_joint_glitches(joints: np.ndarray) -> np.ndarray:
    """Replace frames with non-finite or absurd joint values by the nearest clean frame.

    Same rule and threshold as ``dataset_zarr._replace_joint_glitches``; reimplemented rather
    than imported for the reason given on :data:`SINGLE_ARM_JOINT_DIM` -- ``dataset_zarr``
    imports torch, cv2, scipy and the FTP-1 model config. ``action_parse_test`` pins the two
    against each other on the glitch shapes that actually occur.
    """
    bad = np.any(~np.isfinite(joints) | (np.abs(joints) > JOINT_GLITCH_THRESH), axis=-1)
    if not np.any(bad):
        return joints
    good = np.where(~bad)[0]
    if good.size == 0:
        logger.warning("every frame of a joint stream is a glitch; returning zeros")
        return np.zeros_like(joints)
    joints = joints.copy()
    joints[bad] = joints[good[np.argmin(np.abs(good[None, :] - np.where(bad)[0][:, None]), axis=1)]]
    return joints


def _pose9d(pose6d: np.ndarray) -> np.ndarray:
    """``(T, 6)`` position + rotation vector -> ``(T, 9)`` position + 6-D rotation.

    Non-finite rows would make ``Rotation.from_rotvec`` raise and take a whole domain down, so
    they are zeroed first and land on the identity rotation. They are rare and isolated; the
    alternative is losing the store.
    """
    pose = np.asarray(pose6d, dtype=np.float64)
    bad = ~np.all(np.isfinite(pose), axis=-1)
    if np.any(bad):
        logger.warning("%d non-finite wrist/head pose row(s) zeroed", int(bad.sum()))
        pose = pose.copy()
        pose[bad] = 0.0
    return mat_to_pose9d(pose_to_mat(pose))


def specs_for_store(data: zarr.Group) -> StoreStateSpec:
    """Classify a *source* store's proprioceptive keys into FTP-1 slot groups.

    A side is present only if it has ``{side}_hand_joints``, matching
    ``dataset_zarr.py:1311`` -- a wrist pose with no hand is not an arm FTP-1 can represent.
    """
    keys = set(data.array_keys())
    arms: list[ArmSpec] = []
    for side, base in SIDES:
        hand_key = f"{side}_hand_joints"
        idx_key = f"{side}_hand_joints_idx"
        if hand_key not in keys or idx_key not in keys:
            continue
        slots = np.asarray(data[idx_key][0], dtype=np.int64).reshape(-1)
        width = int(data[hand_key].shape[1]) if data[hand_key].ndim > 1 else 1
        if slots.size != width:
            raise ValueError(f"{idx_key} has {slots.size} slots but {hand_key} has {width} columns")
        if slots.min() < 0 or slots.max() >= SINGLE_HAND_JOINT_DIM:
            raise ValueError(f"{idx_key} values {slots.tolist()} outside [0, {SINGLE_HAND_JOINT_DIM})")

        wrist_key = f"{side}_wrist_pose" if f"{side}_wrist_pose" in keys else None
        arm_key = f"{side}_arm_joints" if f"{side}_arm_joints" in keys else None
        arm_width = 0
        if arm_key is not None:
            arm_width = int(data[arm_key].shape[1])
            if arm_width > SINGLE_ARM_JOINT_DIM:
                raise ValueError(f"{arm_key} has {arm_width} joints, more than {SINGLE_ARM_JOINT_DIM}")
        arms.append(
            ArmSpec(
                side=side,
                base=base,
                hand_key=hand_key,
                hand_slots=tuple(int(s) for s in slots),
                wrist_key=wrist_key,
                arm_key=arm_key,
                arm_width=arm_width,
            )
        )

    supp_key = "supplementary_joints" if "supplementary_joints" in keys else None
    supp_width = min(int(data[supp_key].shape[1]), RESERVED_ACTION_DIM) if supp_key else 0
    return StoreStateSpec(
        arms=tuple(arms),
        head_key="camera_ego_pose" if "camera_ego_pose" in keys else None,
        supp_key=supp_key,
        supp_width=supp_width,
    )


def state_mask(spec: StoreStateSpec) -> np.ndarray:
    """``(120,) uint8`` -- which slots this store actually populates.

    FTP-1's ``action_mask`` convention: a zero slot is *absent*, not *zero-valued*, and every
    downstream consumer must weight by it rather than regress the zeros.
    """
    mask = np.zeros(ACTION_DIM, dtype=np.uint8)
    for arm in spec.arms:
        if arm.wrist_key is not None:
            mask[arm.wrist_slice] = 1
        if arm.arm_key is not None:
            mask[arm.arm_slice] = 1
        mask[arm.hand_columns] = 1
    if spec.head_key is not None:
        mask[HEAD_START : HEAD_START + 9] = 1
    if spec.supp_key is not None:
        mask[SUPP_START : SUPP_START + spec.supp_width] = 1
    return mask


def sample_actions(
    state: np.ndarray,
    episode_ends: np.ndarray,
    mask: np.ndarray,
    *,
    num_frames: int = 16,
    tubelet: int = 2,
    num_clips: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """``(n, num_steps - 1, 120)`` actions from clips that stay inside one episode.

    Clip starts must respect ``episode_ends``: a window straddling a boundary would show a
    jump that no action produced, which is exactly the spurious variation this is meant to
    detect the absence of.
    """
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    span = (num_frames - 1) * 1
    windows = [(lo, hi - span) for lo, hi in zip(starts, ends, strict=True) if hi - span > lo]
    if not windows:
        return np.zeros((0, num_frames // tubelet - 1, ACTION_DIM), dtype=np.float32)

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_clips):
        lo, hi = windows[rng.integers(len(windows))]
        begin = int(rng.integers(lo, hi))
        out.append(actions_from_state(state[begin : begin + num_frames : tubelet], mask))
    return np.stack(out)


def drop_constant_columns(actions: np.ndarray, mask: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    """Clear mask bits whose ACTION never varies.

    Measured on the *action*, not the state, because the action is what the mask gates. The
    distinction is not academic: RH20T repositions its camera between episodes but holds it
    fixed within one, so ``camera_ego_pose`` has a state spread of ~3e-5 across the store --
    above any sane tolerance -- while the within-clip relative head transform is identically
    zero in all 128 sampled clips. Testing the state keeps nine dead slots that testing the
    action removes.

    The declared presence of a stream is not evidence that it recorded anything, which is the
    lesson ``is_degenerate`` encodes for tactile. Dead slots are worse than absent ones: they
    dilute every masked mean, let the action embedder spend capacity on a constant, and make
    ``mask.sum()`` overstate how much proprioception a domain contributes.
    """
    mask = np.asarray(mask).reshape(-1).copy()
    if actions.size == 0:
        return mask
    flat = actions.reshape(-1, actions.shape[-1])
    dead = mask.astype(bool) & (flat.std(axis=0) <= tol)
    if np.any(dead):
        logger.info("dropping %d constant slot(s) from the action mask: %s", int(dead.sum()), np.flatnonzero(dead))
    mask[dead] = 0
    return mask


def read_state(data: zarr.Group, spec: StoreStateSpec, begin: int, end: int) -> np.ndarray:
    """``(end - begin, 120) float32`` absolute state in FTP-1 slot order.

    "Absolute" in the source's own frame: world for the wrist, joint angles for the rest. The
    camera projection is intentionally omitted -- see the module docstring.
    """
    length = end - begin
    out = np.zeros((length, ACTION_DIM), dtype=np.float32)

    for arm in spec.arms:
        if arm.wrist_key is not None:
            out[:, arm.wrist_slice] = _pose9d(data[arm.wrist_key][begin:end])
        if arm.arm_key is not None:
            joints = replace_joint_glitches(np.asarray(data[arm.arm_key][begin:end], dtype=np.float64))
            out[:, arm.arm_slice] = joints[:, : arm.arm_width]
        hand = np.asarray(data[arm.hand_key][begin:end], dtype=np.float64).reshape(length, -1)
        out[:, arm.hand_columns] = replace_joint_glitches(hand)

    if spec.head_key is not None:
        out[:, HEAD_START : HEAD_START + 9] = _pose9d(data[spec.head_key][begin:end])
    if spec.supp_key is not None:
        supp = replace_joint_glitches(np.asarray(data[spec.supp_key][begin:end], dtype=np.float64))
        out[:, SUPP_START : SUPP_START + spec.supp_width] = supp[:, : spec.supp_width]
    return out


def relative_pose(state: np.ndarray, columns: slice) -> np.ndarray:
    """Pairwise ``inv(P_i) @ P_{i+1}`` over a 9-D pose block, returned as 9-D.

    This is ``convert_pose_mat_rep(..., pose_rep="relative")`` with a *moving* base, which
    that helper cannot express (it takes one base for the whole slice). ``action_parse_test``
    asserts step-by-step equality against it so the semantics stay pinned to the repository's
    definition rather than to this implementation.

    The inverse is taken analytically -- ``inv([[R, t], [0, 1]]) == [[R^T, -R^T t], [0, 1]]``
    -- rather than with ``np.linalg.inv``. That is both cheaper and, more importantly, it
    cannot raise: a degenerate rotation block makes ``inv`` throw ``LinAlgError``, which in a
    DataLoader worker kills the worker and takes the whole job with it. One malformed frame in
    17.7 M must cost that frame, not the run.
    """
    mats = pose10d_to_mat(np.asarray(state[:, columns], dtype=np.float64))
    rotation, translation = mats[:-1, :3, :3], mats[:-1, :3, 3]
    inverse_rotation = rotation.transpose(0, 2, 1)

    relative = np.zeros_like(mats[1:])
    relative[:, :3, :3] = inverse_rotation @ mats[1:, :3, :3]
    relative[:, :3, 3] = np.einsum("nij,nj->ni", inverse_rotation, mats[1:, :3, 3] - translation)
    relative[:, 3, 3] = 1.0

    out = mat_to_pose9d(relative)
    # A non-orthonormal block yields finite garbage rather than an exception; a NaN one would
    # still poison the loss, so it is zeroed and reported instead.
    bad = ~np.all(np.isfinite(out), axis=-1)
    if np.any(bad):
        logger.warning("%d degenerate pose transition(s) zeroed", int(bad.sum()))
        out[bad] = 0.0
    return out


#: The 9-D pose blocks of the 120-D layout. Fixed by the FTP-1 slot map, so a reader can
#: identify them without the source store's spec -- which is what lets the *derived* store,
#: which carries only ``state`` and ``action_mask``, produce actions on its own.
POSE_BLOCKS: tuple[slice, ...] = (
    slice(SLICES["right-wrist-pos"][0], SLICES["right-wrist-rot"][1]),
    slice(SLICES["left-wrist-pos"][0], SLICES["left-wrist-rot"][1]),
    slice(SLICES["head-track-pos"][0], SLICES["head-track-rot"][1]),
)

#: Gripper column of each hand, absolute under ``action_joint_rep="mix"``.
ABSOLUTE_COLUMNS: tuple[int, ...] = (
    SLICES["right-hand-joints"][0] + GRIPPER_HAND_SLOT,
    SLICES["left-hand-joints"][0] + GRIPPER_HAND_SLOT,
)

#: ``mat_to_pose9d`` of the identity transform: position 0 and the first two rows of I.
#: A *relative* pose action for a stationary frame lands exactly here, so the rot6d block of
#: a slow motion is this constant plus a perturbation three orders of magnitude smaller.
#: Measured on RH20TCfg5Franka: ``right-wrist-rot[0]`` had mean 0.99994 and sd 0.00034, a
#: 0.03% relative signal that no linear projection is going to recover. Subtracting the
#: identity makes "no motion" exactly zero and puts the whole block on the scale of the
#: motion rather than on the scale of the encoding.
IDENTITY_POSE9D: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def actions_from_state(state: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """``(N, 120)`` states -> ``(N - 1, 120)`` actions, using only the mask.

    Equivalent to :func:`states_to_actions` but needs no :class:`StoreStateSpec`, because the
    pose blocks and the gripper columns are properties of the *layout*, not of the store. That
    is what lets the derived clip store -- which carries ``state`` and ``action_mask`` and
    nothing else -- derive actions at read time for whatever stride the clip used.

    ``action_parse_test`` asserts this agrees with the spec-driven version on every observed
    layout, so the two cannot drift apart.
    """
    if state.ndim != 2 or state.shape[1] != ACTION_DIM:
        raise ValueError(f"state must be (N, {ACTION_DIM}), got {state.shape}")
    if state.shape[0] < 2:
        raise ValueError("need at least two states to form one action")

    mask = np.asarray(action_mask).reshape(-1).astype(bool)
    actions = state[1:] - state[:-1]
    for block in POSE_BLOCKS:
        if mask[block].any():
            actions[:, block] = relative_pose(state, block) - IDENTITY_POSE9D
    for column in ABSOLUTE_COLUMNS:
        if mask[column]:
            actions[:, column] = state[1:, column]
    actions[:, ~mask] = 0.0
    return actions.astype(np.float32)


def states_to_actions(
    state: np.ndarray,
    spec: StoreStateSpec,
    *,
    absolute_hand_slots: tuple[int, ...] = (GRIPPER_HAND_SLOT,),
) -> np.ndarray:
    """``(N, 120)`` states at clip timestamps -> ``(N - 1, 120)`` per-step actions.

    The base of step *i* is the state the action departs from, so the result is the dynamics
    quantity a world model consumes. FTP-1's own action uses a single chunk-relative base
    instead; the *layout* and *mask* here are FTP-1's, the choice of base is not.

    Callers pass ``state`` already sampled at the clip's frames, so an arbitrary stride is
    handled without the builder having to know about it -- which is why actions are derived
    here rather than stored.
    """
    if state.ndim != 2 or state.shape[1] != ACTION_DIM:
        raise ValueError(f"state must be (N, {ACTION_DIM}), got {state.shape}")
    if state.shape[0] < 2:
        raise ValueError("need at least two states to form one action")

    actions = np.zeros((state.shape[0] - 1, ACTION_DIM), dtype=np.float32)
    absolute = set(absolute_hand_slots)

    for arm in spec.arms:
        if arm.wrist_key is not None:
            actions[:, arm.wrist_slice] = relative_pose(state, arm.wrist_slice) - IDENTITY_POSE9D
        if arm.arm_key is not None:
            block = state[:, arm.arm_slice]
            actions[:, arm.arm_slice] = block[1:] - block[:-1]
        columns = arm.hand_columns
        block = state[:, columns]
        stepped = block[1:] - block[:-1]
        keep = np.isin(np.asarray(arm.hand_slots), list(absolute)) if absolute else np.zeros(len(columns), bool)
        stepped[:, keep] = block[1:][:, keep]
        actions[:, columns] = stepped

    if spec.head_key is not None:
        head = slice(HEAD_START, HEAD_START + 9)
        actions[:, head] = relative_pose(state, head) - IDENTITY_POSE9D
    if spec.supp_key is not None:
        supp = slice(SUPP_START, SUPP_START + spec.supp_width)
        actions[:, supp] = state[1:, supp] - state[:-1, supp]
    return actions
