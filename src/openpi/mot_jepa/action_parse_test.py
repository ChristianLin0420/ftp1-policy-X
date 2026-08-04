from __future__ import annotations

import numpy as np
import pytest
import zarr

from openpi.dataset_zarr import FTP1_GRIPPER_HAND_SLOT_INDEX
from openpi.dataset_zarr import _replace_joint_glitches as reference_glitch_fix
from openpi.models_pytorch import ftp1_model_config as ftp1_cfg
from openpi.mot_jepa import action_parse as ap
from openpi.pose_repr_utils import convert_pose_mat_rep
from openpi.pose_utils import mat_to_pose9d
from openpi.pose_utils import pose10d_to_mat
from openpi.pose_utils import pose_to_mat

#: Every proprioceptive layout the corpus survey found, one entry per shape that differs.
#: Adding a domain here is how a new layout gets covered; the parametrized tests below then
#: exercise it end to end.
OBSERVED = {
    # gripper only, no arm, no wrist -- FreeTacMan has a wrist, Unit does not
    "FreeTacMan": {"right_hand_joints": 1, "right_hand_joints_idx": [28], "right_wrist_pose": True},
    "Unit": {"right_hand_joints": 1, "right_hand_joints_idx": [28], "right_arm_joints": 7},
    "VLA_touch": {
        "right_hand_joints": 1,
        "right_hand_joints_idx": [28],
        "right_wrist_pose": True,
        "right_arm_joints": 7,
    },
    # non-contiguous hand slots
    "MotionTrans": {
        "right_hand_joints": 6,
        "right_hand_joints_idx": [1, 2, 7, 12, 17, 22],
        "right_wrist_pose": True,
    },
    # bimanual gripper + head pose
    "RDP_Bimanual": {
        "right_hand_joints": 1,
        "right_hand_joints_idx": [28],
        "right_wrist_pose": True,
        "left_hand_joints": 1,
        "left_hand_joints_idx": [28],
        "left_wrist_pose": True,
        "camera_ego_pose": True,
    },
    # bimanual arm + gripper, no wrist, no head
    "VisuoTactile_D-WHEEL": {
        "right_hand_joints": 1,
        "right_hand_joints_idx": [28],
        "right_arm_joints": 7,
        "left_hand_joints": 1,
        "left_hand_joints_idx": [28],
        "left_arm_joints": 7,
    },
    # the widest case: 22 dexterous hand joints per side, head pose, supplementary joints
    "sharpa": {
        "right_hand_joints": 22,
        "right_hand_joints_idx": [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26],
        "right_wrist_pose": True,
        "right_arm_joints": 7,
        "left_hand_joints": 22,
        "left_hand_joints_idx": [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26],
        "left_wrist_pose": True,
        "left_arm_joints": 7,
        "camera_ego_pose": True,
        "supplementary_joints": 9,
    },
}


def make_store(path, layout: dict, *, frames: int = 40, seed: int = 0) -> zarr.Group:
    """A source store carrying exactly one domain's proprioceptive layout."""
    rng = np.random.default_rng(seed)
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    meta.create_array("episode_ends", shape=(1,), dtype="int64")
    meta["episode_ends"][:] = [frames]

    for key, value in layout.items():
        if key.endswith("_idx"):
            slots = np.tile(np.asarray(value, dtype=np.int64), (frames, 1))
            data.create_array(key, shape=slots.shape, dtype="int64")
            data[key][:] = slots
        elif value is True:  # a 6-D pose: position + rotation vector
            pose = rng.normal(0.0, 0.5, (frames, 6))
            data.create_array(key, shape=pose.shape, dtype="float64")
            data[key][:] = pose
        else:  # joints, `value` columns wide
            joints = rng.normal(0.0, 0.4, (frames, int(value)))
            data.create_array(key, shape=joints.shape, dtype="float32")
            data[key][:] = joints
    return root


@pytest.fixture(params=sorted(OBSERVED), name="domain")
def domain_fixture(request):
    return request.param


def test_dimension_constants_match_the_ftp1_model_config():
    """These are duplicated to keep jax out of a DataLoader worker; drift must fail here."""
    assert ap.SINGLE_ARM_JOINT_DIM == ftp1_cfg.FTP1_SINGLE_ARM_JOINT_DIM
    assert ap.SINGLE_HAND_JOINT_DIM == ftp1_cfg.FTP1_SINGLE_HAND_JOINT_DIM
    assert ap.SINGLE_ARM_ACTION_REP_DIM == ftp1_cfg.FTP1_SINGLE_ARM_ACTION_REP_DIM
    assert ap.RESERVED_ACTION_DIM == ftp1_cfg.FTP1_RESERVED_ACTION_DIM
    expected = 2 * ftp1_cfg.FTP1_SINGLE_ARM_ACTION_REP_DIM + 9 + ftp1_cfg.FTP1_RESERVED_ACTION_DIM
    assert ap.ACTION_DIM == expected == 120


def test_gripper_slot_matches_dataset_zarr():
    assert ap.GRIPPER_HAND_SLOT == FTP1_GRIPPER_HAND_SLOT_INDEX


def test_state_is_120_wide_for_every_observed_layout(tmp_path, domain):
    store = make_store(tmp_path / f"{domain}.zarr", OBSERVED[domain])
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)
    assert state.shape == (40, ap.ACTION_DIM)
    assert state.dtype == np.float32
    assert np.all(np.isfinite(state))


def test_populated_columns_are_exactly_the_mask(tmp_path, domain):
    """A slot outside the mask must stay zero, and every masked slot must carry signal.

    This is the test that catches a scatter written to the wrong slots: the values would
    still be finite and the shape still right, but they would land outside the mask.
    """
    store = make_store(tmp_path / f"{domain}.zarr", OBSERVED[domain])
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)
    mask = ap.state_mask(spec)

    nonzero = np.any(state != 0.0, axis=0)
    assert not np.any(nonzero & (mask == 0)), "signal landed in an unmasked slot"
    # The rot6d block of an identity-ish rotation can contain exact zeros, so only require
    # that each masked *group* is alive rather than every individual column.
    for name, (lo, hi) in ap.SLICES.items():
        if mask[lo:hi].any():
            assert nonzero[lo:hi].any(), f"masked group {name} carries no signal"


def test_hand_slots_scatter_to_the_ftp1_index(tmp_path):
    """MotionTrans records 6 hand values whose slots are [1, 2, 7, 12, 17, 22].

    Writing them contiguously would place every finger wrongly with no downstream complaint.
    """
    layout = OBSERVED["MotionTrans"]
    store = make_store(tmp_path / "mt.zarr", layout)
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)

    raw = np.asarray(store["data"]["right_hand_joints"][0:40])
    hand_base = ap.SLICES["right-hand-joints"][0]
    for column, slot in enumerate(layout["right_hand_joints_idx"]):
        np.testing.assert_allclose(state[:, hand_base + slot], raw[:, column], rtol=1e-6)

    occupied = {hand_base + slot for slot in layout["right_hand_joints_idx"]}
    hand_lo, hand_hi = ap.SLICES["right-hand-joints"]
    for column in range(hand_lo, hand_hi):
        if column not in occupied:
            assert np.all(state[:, column] == 0.0)


def test_relative_pose_matches_the_repository_helper(tmp_path):
    """``relative_pose`` is a moving-base version of ``convert_pose_mat_rep``.

    That helper takes a single base for a whole slice and so cannot express this directly.
    Pinning it step by step keeps the semantics defined by the repository, not by us.
    """
    rng = np.random.default_rng(7)
    poses = mat_to_pose9d(pose_to_mat(rng.normal(0.0, 0.5, (9, 6))))
    state = np.zeros((9, ap.ACTION_DIM), dtype=np.float32)
    state[:, 0:9] = poses

    got = ap.relative_pose(state, slice(0, 9))
    # Build the expectation from the same float32 state the implementation reads, so this
    # asserts the *semantics* rather than re-measuring the float32 store round-trip.
    mats = pose10d_to_mat(np.asarray(state[:, 0:9], dtype=np.float64))
    for step in range(8):
        expected = mat_to_pose9d(convert_pose_mat_rep(mats[step + 1], mats[step], pose_rep="relative"))
        np.testing.assert_allclose(got[step], expected, rtol=1e-9, atol=1e-9)


def test_glitch_replacement_matches_dataset_zarr():
    """Pins our copy against FTP-1's original, which is why the private name is imported.

    ``action_parse`` reimplements this rather than importing it because ``dataset_zarr``
    pulls in torch, cv2 and scipy, which a DataLoader worker must not pay for.
    """
    joints = np.linspace(0.0, 1.0, 12 * 3).reshape(12, 3)
    joints[4] = 1e30  # the RH20TCfg1OptoForce failure mode
    joints[9, 1] = np.nan
    np.testing.assert_array_equal(ap.replace_joint_glitches(joints.copy()), reference_glitch_fix(joints.copy()))


def test_glitch_replacement_survives_an_all_bad_stream():
    joints = np.full((5, 3), np.inf)
    assert np.all(ap.replace_joint_glitches(joints) == 0.0)


def test_actions_are_one_shorter_and_stride_aware(tmp_path, domain):
    """Actions are derived from the clip's own frames, which is why they are not stored.

    A stored per-frame action would be silently wrong for every stride-2 clip, and the
    dataset samples stride in {1, 2}.
    """
    store = make_store(tmp_path / f"{domain}.zarr", OBSERVED[domain])
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)

    stride1 = ap.states_to_actions(state[0:9:1], spec)
    stride2 = ap.states_to_actions(state[0:18:2], spec)
    assert stride1.shape == stride2.shape == (8, ap.ACTION_DIM)
    mask = ap.state_mask(spec)
    moving = mask.astype(bool) & np.any(state != state[0], axis=0)
    if moving.any():
        assert not np.allclose(stride1[:, moving], stride2[:, moving]), "stride was ignored"


def test_gripper_is_absolute_while_other_joints_are_relative(tmp_path):
    """FTP-1's ``action_joint_rep='mix'``: slot 28 is a width target, not an increment.

    Differencing a gripper command produces an action that cannot be executed.
    """
    store = make_store(tmp_path / "sharpa.zarr", OBSERVED["sharpa"])
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)
    actions = ap.states_to_actions(state[:9], spec)

    hand_base = ap.SLICES["right-hand-joints"][0]
    relative_slot = hand_base + OBSERVED["sharpa"]["right_hand_joints_idx"][0]
    np.testing.assert_allclose(
        actions[:, relative_slot], state[1:9, relative_slot] - state[0:8, relative_slot], rtol=1e-5
    )

    # Give the same store a gripper slot and check it comes back absolute.
    gripper = make_store(tmp_path / "grip.zarr", OBSERVED["FreeTacMan"])
    grip_spec = ap.specs_for_store(gripper["data"])
    grip_state = ap.read_state(gripper["data"], grip_spec, 0, 40)
    grip_actions = ap.states_to_actions(grip_state[:9], grip_spec)
    column = ap.SLICES["right-hand-joints"][0] + ap.GRIPPER_HAND_SLOT
    np.testing.assert_allclose(grip_actions[:, column], grip_state[1:9, column], rtol=1e-5)


def test_unmasked_slots_stay_zero_in_the_action(tmp_path, domain):
    store = make_store(tmp_path / f"{domain}.zarr", OBSERVED[domain])
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)
    actions = ap.states_to_actions(state[:9], spec)
    mask = ap.state_mask(spec)
    assert np.all(actions[:, mask == 0] == 0.0)


def test_a_side_without_hand_joints_is_not_an_arm(tmp_path):
    """Mirrors ``dataset_zarr.py:1311``: a wrist with no hand is not a representable arm."""
    store = make_store(tmp_path / "wrist_only.zarr", {"right_wrist_pose": True})
    spec = ap.specs_for_store(store["data"])
    assert spec.arms == ()
    assert spec.is_empty
    assert not ap.state_mask(spec).any()


def test_a_hand_slot_outside_the_layout_is_refused(tmp_path):
    store = make_store(tmp_path / "bad.zarr", {"right_hand_joints": 1, "right_hand_joints_idx": [32]})
    with pytest.raises(ValueError, match="outside"):
        ap.specs_for_store(store["data"])


def test_more_arm_joints_than_the_layout_is_refused(tmp_path):
    layout = {"right_hand_joints": 1, "right_hand_joints_idx": [28], "right_arm_joints": 9}
    store = make_store(tmp_path / "wide.zarr", layout)
    with pytest.raises(ValueError, match="more than"):
        ap.specs_for_store(store["data"])


def test_non_finite_poses_do_not_take_down_a_store(tmp_path):
    """One bad row must cost that row, not the domain."""
    store = make_store(tmp_path / "nan.zarr", OBSERVED["FreeTacMan"])
    pose = np.asarray(store["data"]["right_wrist_pose"][:])
    pose[3] = np.nan
    store["data"]["right_wrist_pose"][:] = pose

    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)
    assert np.all(np.isfinite(state))


def test_mask_only_actions_agree_with_the_spec_driven_version(tmp_path, domain):
    """The derived store carries state and action_mask but no spec, so it needs a spec-free
    path. If the two ever disagree, actions computed at train time stop matching the ones the
    builder's semantics describe -- and nothing downstream would report it.
    """
    store = make_store(tmp_path / f"{domain}.zarr", OBSERVED[domain])
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)
    mask = ap.state_mask(spec)

    np.testing.assert_allclose(
        ap.actions_from_state(state[0:17:2], mask),
        ap.states_to_actions(state[0:17:2], spec),
        rtol=1e-5,
        atol=1e-6,
    )


def test_pose_blocks_and_gripper_columns_land_where_ftp1_puts_them():
    """These constants are what make the spec-free path possible; a wrong one is silent."""
    assert ap.POSE_BLOCKS[0] == slice(0, 9)
    assert ap.POSE_BLOCKS[1] == slice(48, 57)
    assert ap.POSE_BLOCKS[2] == slice(96, 105)
    assert ap.ABSOLUTE_COLUMNS == (16 + 28, 64 + 28)


def test_mask_only_actions_zero_absent_slots(tmp_path):
    store = make_store(tmp_path / "u.zarr", OBSERVED["Unit"])
    spec = ap.specs_for_store(store["data"])
    state = ap.read_state(store["data"], spec, 0, 40)
    mask = ap.state_mask(spec)
    actions = ap.actions_from_state(state[:9], mask)
    assert np.all(actions[:, mask == 0] == 0.0)
    # Unit has no wrist, so the pose block must stay zero rather than compose garbage.
    assert np.all(actions[:, ap.POSE_BLOCKS[0]] == 0.0)


def test_a_degenerate_pose_block_does_not_raise():
    """``np.linalg.inv`` throws LinAlgError on a singular rotation, which inside a DataLoader
    worker kills the worker and the whole job. One malformed frame in 17.7M must cost that
    frame only, so the inverse is taken analytically and non-finite rows are zeroed.
    """
    state = np.zeros((5, ap.ACTION_DIM), dtype=np.float32)
    state[:, 0:9] = np.arange(5, dtype=np.float32)[:, None]  # rot6d block is all-equal: singular
    mask = np.zeros(ap.ACTION_DIM, dtype=np.uint8)
    mask[0:9] = 1

    actions = ap.actions_from_state(state, mask)
    assert actions.shape == (4, ap.ACTION_DIM)
    assert np.all(np.isfinite(actions))


def test_relative_pose_uses_an_exact_rigid_inverse():
    """The analytic inverse must agree with the general one on genuine rotations."""
    rng = np.random.default_rng(11)
    poses = mat_to_pose9d(pose_to_mat(rng.normal(0.0, 0.7, (6, 6))))
    state = np.zeros((6, ap.ACTION_DIM))
    state[:, 0:9] = poses
    mats = pose10d_to_mat(poses)
    for step in range(5):
        expected = mat_to_pose9d(np.linalg.inv(mats[step]) @ mats[step + 1])
        np.testing.assert_allclose(ap.relative_pose(state, slice(0, 9))[step], expected, rtol=1e-9, atol=1e-9)
