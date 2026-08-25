from __future__ import annotations

# Tests deliberately exercise deployment internals and keep heavyweight config imports local.
# ruff: noqa: PLC0415, SLF001
import numpy as np
import pytest
import torch

from openpi.mot_jepa.action_parse import ACTION_DIM
from openpi.mot_jepa.action_parse import actions_from_state
from openpi.mot_jepa.deploy import ARM_JOINT_SLICE
from openpi.mot_jepa.deploy import GRIPPER_SLOT
from openpi.mot_jepa.deploy import MotJepaFlowPolicy
from openpi.mot_jepa.deploy import chunk_to_absolute_qpos8
from openpi.mot_jepa.deploy import univtac_action_mask


def test_univtac_action_mask_has_only_robot_live_slots():
    mask = univtac_action_mask()
    assert mask.shape == (ACTION_DIM,)
    assert mask.dtype == torch.float32
    assert torch.nonzero(mask, as_tuple=False).flatten().tolist() == [*range(9, 16), 44]


class _ConstantFlowHead:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.calls = []

    def sample(self, encoded, action_mask, domain_id, *, num_steps, generator):
        del encoded
        self.calls.append((action_mask.clone(), domain_id.clone(), num_steps, generator))
        return torch.ones(1, self.horizon, ACTION_DIM)


class _RandomFlowHead:
    def __init__(self, horizon: int):
        self.horizon = horizon

    def sample(self, encoded, action_mask, domain_id, *, num_steps, generator):
        del encoded, action_mask, domain_id, num_steps
        return torch.randn(1, self.horizon, ACTION_DIM, generator=generator)


def _flow_policy(head, normalizer, *, num_samples: int = 1, sample_seed: int = 5):
    from types import SimpleNamespace

    return MotJepaFlowPolicy(
        backbone=None,
        head=head,
        normalizer=normalizer,
        layout=SimpleNamespace(),
        domain_id=1,
        device=torch.device("cpu"),
        observation_stride=2,
        action_stride=1,
        domain_names=("other", "lift_bottle"),
        num_inference_steps=7,
        num_samples=num_samples,
        sample_seed=sample_seed,
    )


def test_flow_sampling_uses_sparse_mask_euler_k_and_denormalizes():
    from openpi.mot_jepa.action_dit import ActionNormalizer

    head = _ConstantFlowHead(horizon=3)
    normalizer = ActionNormalizer(num_domains=2)
    normalizer.load_stats(torch.full((2, ACTION_DIM), 0.25), torch.full((2, ACTION_DIM), 2.0))
    policy = _flow_policy(head, normalizer, num_samples=3)

    raw = policy._sample_chunk(encoded=None)

    assert len(head.calls) == 3
    for mask, domain, num_steps, generator in head.calls:
        torch.testing.assert_close(mask, univtac_action_mask().unsqueeze(0), rtol=0, atol=0)
        assert domain.tolist() == [1]
        assert num_steps == 7
        assert generator is policy._generator
    expected = np.zeros((3, ACTION_DIM), dtype=np.float32)
    expected[:, 9:16] = 2.25
    expected[:, 44] = 2.25
    np.testing.assert_allclose(raw, expected, rtol=0, atol=0)


def test_flow_reset_reseeds_its_episode_local_generator():
    from openpi.mot_jepa.action_dit import ActionNormalizer

    policy = _flow_policy(_RandomFlowHead(horizon=2), ActionNormalizer(num_domains=2), num_samples=2)
    policy.reset(seed=13)
    first = policy._sample_chunk(encoded=None)
    policy.reset(seed=13)
    repeated = policy._sample_chunk(encoded=None)
    policy.reset(seed=14)
    different = policy._sample_chunk(encoded=None)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first[:, 9:16], different[:, 9:16])


def test_observe_preserves_univtac_native_channels_and_left_right_pad_order():
    from types import SimpleNamespace

    from openpi.mot_jepa.deploy import MotJepaRidgePolicy

    policy = MotJepaRidgePolicy.__new__(MotJepaRidgePolicy)
    policy.layout = SimpleNamespace(video_size=2, gel_size=2)
    policy.num_frames = 2
    policy.frame_stride = 1
    policy._frames = []
    policy._resize = lambda image, size: np.asarray(image)

    head = np.full((2, 2, 3), [1, 2, 3], dtype=np.uint8)
    left = np.full((2, 2, 3), [4, 5, 6], dtype=np.uint8)
    right = np.full((2, 2, 3), [7, 8, 9], dtype=np.uint8)
    observation = {
        "observation": {"head": {"rgb": head}},
        "tactile": {
            "left_gsmini": {"rgb_marker": left},
            "right_gsmini": {"rgb_marker": right},
        },
    }

    policy.observe(observation)

    np.testing.assert_array_equal(policy._frames[0]["video"], head)
    np.testing.assert_array_equal(policy._frames[0]["gel"][0], left)
    np.testing.assert_array_equal(policy._frames[0]["gel"][1], right)


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


def _stub_policy(frame_stride: int, num_frames: int = 4):
    """A policy with no backbone or head: only the ring buffer is under test."""
    from openpi.mot_jepa.config import CONFIGS
    from openpi.mot_jepa.deploy import MotJepaRidgePolicy

    policy = MotJepaRidgePolicy.__new__(MotJepaRidgePolicy)
    policy.layout = CONFIGS["mot_jepa_pilot"].layout
    policy.num_frames = num_frames
    policy.frame_stride = frame_stride
    policy._frames = []
    return policy


def _push(policy, tag: int) -> None:
    policy._frames.append({"video": tag, "gel": tag})
    if len(policy._frames) > (policy.num_frames - 1) * policy.frame_stride + 1:
        policy._frames.pop(0)


def _window_tags(policy) -> list[int]:
    frames = policy._frames[::-1][:: policy.frame_stride][::-1]
    pad = [frames[0]] * (policy.num_frames - len(frames))
    return [f["video"] for f in (pad + frames)[-policy.num_frames :]]


def test_frame_stride_decimates_and_always_keeps_the_newest_observation():
    """The encoder must see the trained frame RATE, anchored to now.

    RoPE `t` is the tubelet index and carries no rate, so an encoder handed a different frame
    spacing than it trained on cannot tell -- it silently sees a slower or faster world. And the
    newest frame must always survive decimation: the action depends on where the motion is *now*,
    so anchoring the subsample to the oldest frame would act on stale observations forever.
    """
    policy = _stub_policy(frame_stride=2, num_frames=4)
    for tag in range(20):
        _push(policy, tag)
        window = _window_tags(policy)
        assert len(window) == policy.num_frames, f"window is {len(window)}, expected {policy.num_frames}"
        assert window[-1] == tag, f"newest frame {tag} missing from window {window}"
    # Once warm, spacing is exactly frame_stride control steps.
    assert _window_tags(policy) == [13, 15, 17, 19]


def test_stride_one_reproduces_consecutive_frames():
    """The widened stride must not change behaviour when it is set back to 1."""
    policy = _stub_policy(frame_stride=1, num_frames=4)
    for tag in range(10):
        _push(policy, tag)
    assert _window_tags(policy) == [6, 7, 8, 9]


def test_cold_start_pads_with_the_oldest_frame_and_still_ends_at_now():
    """Before the buffer fills, edge-padding is a documented deviation -- but not at the tail."""
    policy = _stub_policy(frame_stride=2, num_frames=4)
    _push(policy, 0)
    assert _window_tags(policy) == [0, 0, 0, 0]
    _push(policy, 1)
    assert _window_tags(policy)[-1] == 1


def test_deployment_derives_observation_stride_from_the_artifact():
    from types import SimpleNamespace

    from openpi.mot_jepa.config import CONFIGS
    from openpi.mot_jepa.deploy import MotJepaRidgePolicy

    head = SimpleNamespace(observation_stride=2, action_stride=1)
    policy = MotJepaRidgePolicy(None, head, CONFIGS["mot_jepa_pilot"].layout, 0, torch.device("cpu"))
    assert policy.frame_stride == 2
    assert policy.action_rep == "absolute"
    assert policy.chunk_action_rep == "mix"
    assert policy.first_executable_index == 0


def test_deployment_refuses_incompatible_sampling_metadata():
    from types import SimpleNamespace

    from openpi.mot_jepa.config import CONFIGS
    from openpi.mot_jepa.deploy import MotJepaRidgePolicy

    layout = CONFIGS["mot_jepa_pilot"].layout
    with pytest.raises(ValueError, match="executes every control step"):
        MotJepaRidgePolicy(None, SimpleNamespace(observation_stride=2, action_stride=2), layout, 0, torch.device("cpu"))
    with pytest.raises(ValueError, match="disagrees"):
        MotJepaRidgePolicy(
            None,
            SimpleNamespace(observation_stride=2, action_stride=1),
            layout,
            0,
            torch.device("cpu"),
            frame_stride=1,
        )
