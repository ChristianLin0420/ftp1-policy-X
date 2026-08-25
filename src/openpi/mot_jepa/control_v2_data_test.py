from __future__ import annotations

import json

import pytest
import torch

from openpi.mot_jepa import control_v2_data as cv2


def _state_from_qpos(qpos: torch.Tensor) -> torch.Tensor:
    return cv2.scatter_qpos8(qpos)


def test_qpos8_extract_scatter_round_trip_and_preserve_inactive_slots():
    qpos = torch.arange(2 * 3 * cv2.QPOS_DIM, dtype=torch.float32).reshape(2, 3, cv2.QPOS_DIM)
    template = torch.full((2, 3, cv2.STATE_DIM), -7.0)
    state = cv2.scatter_qpos8(qpos, template=template)

    torch.testing.assert_close(cv2.extract_qpos8(state), qpos)
    inactive = torch.ones(cv2.STATE_DIM, dtype=torch.bool)
    inactive[cv2.ARM_STATE_SLICE] = False
    inactive[cv2.GRIPPER_STATE_SLOT] = False
    assert torch.all(state[..., inactive] == -7.0)
    # Callers retain their own template; scatter never mutates it.
    assert torch.all(template == -7.0)


@pytest.mark.parametrize(
    ("fn", "value", "match"),
    [
        (cv2.extract_qpos8, torch.zeros(4, 119), "120"),
        (cv2.extract_qpos8, torch.full((4, 120), torch.nan), "non-finite"),
        (cv2.scatter_qpos8, torch.zeros(4, 7), "8"),
        (cv2.scatter_qpos8, torch.zeros(4, 8, dtype=torch.long), "floating"),
    ],
)
def test_qpos_contract_rejects_bad_shape_dtype_and_values(fn, value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        fn(value)


def test_mixed_h32_target_has_placeholder_relative_arm_and_absolute_gripper():
    time = torch.arange(cv2.ACTION_HORIZON, dtype=torch.float32)
    base_arm = torch.arange(cv2.ARM_DIM, dtype=torch.float32) + 10.0
    arm = base_arm + time[:, None] * torch.arange(1, cv2.ARM_DIM + 1)
    gripper = 0.02 - time[:, None] * 0.0004
    qpos = torch.cat((arm, gripper), dim=-1)

    chunk = cv2.build_mixed_action_chunk(_state_from_qpos(qpos))

    torch.testing.assert_close(chunk[0, : cv2.ARM_DIM], torch.zeros(cv2.ARM_DIM))
    torch.testing.assert_close(chunk[0, cv2.ARM_DIM], qpos[0, cv2.ARM_DIM])
    torch.testing.assert_close(chunk[17, : cv2.ARM_DIM], qpos[17, : cv2.ARM_DIM] - qpos[0, : cv2.ARM_DIM])
    torch.testing.assert_close(chunk[:, cv2.ARM_DIM], qpos[:, cv2.ARM_DIM])
    torch.testing.assert_close(cv2.mixed_chunk_to_absolute_qpos8(chunk, qpos[0]), qpos)


def test_mixed_target_rejects_wrong_horizon_or_width():
    with pytest.raises(ValueError, match="horizon 32"):
        cv2.build_mixed_action_chunk(torch.zeros(31, cv2.STATE_DIM))
    with pytest.raises(ValueError, match="32, 8"):
        cv2.mixed_action_chunk_from_qpos(torch.zeros(32, 7))
    with pytest.raises(ValueError, match="current_qpos8"):
        cv2.mixed_chunk_to_absolute_qpos8(torch.zeros(3, 32, 8), torch.zeros(8))


def test_h32_indices_keep_current_plus_31_futures_inside_episode():
    indices = cv2.control_v2_chunk_indices(8, episode_start=4, episode_end=71, action_stride=2)
    assert indices.tolist() == list(range(8, 71, 2))

    with pytest.raises(ValueError, match="crosses episode"):
        cv2.control_v2_chunk_indices(9, episode_start=4, episode_end=71, action_stride=2)
    with pytest.raises(ValueError, match="crosses episode"):
        cv2.control_v2_chunk_indices(3, episode_start=4, episode_end=71)
    with pytest.raises(ValueError, match="positive"):
        cv2.control_v2_chunk_indices(8, episode_start=4, episode_end=71, action_stride=0)


def test_explicit_prefix_padding_matches_deployment_cold_start_for_batch_and_single():
    history = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    padded = cv2.prefix_pad_history(history, torch.tensor([1, 3]))

    torch.testing.assert_close(padded.values[0], history[0, -1:].expand_as(history[0]))
    torch.testing.assert_close(padded.values[1], torch.stack((history[1, 2], history[1, 2], *history[1, 2:])))
    assert padded.valid.tolist() == [[False, False, False, False, True], [False, False, True, True, True]]

    # Deployment applies the same helper to one history at a time.
    single = cv2.prefix_pad_history(history[1], 3)
    torch.testing.assert_close(single.values, padded.values[1])
    torch.testing.assert_close(single.valid, padded.valid[1])


def test_random_prefix_padding_is_deterministic_only_from_supplied_generator():
    history = torch.arange(4 * 8 * 2, dtype=torch.float32).reshape(4, 8, 2)
    first_generator = torch.Generator().manual_seed(9284)
    second_generator = torch.Generator().manual_seed(9284)

    first = cv2.random_prefix_pad_history(history, generator=first_generator, min_available=2, max_available=6)
    second = cv2.random_prefix_pad_history(history, generator=second_generator, min_available=2, max_available=6)

    torch.testing.assert_close(first.available_lengths, second.available_lengths)
    torch.testing.assert_close(first.values, second.values)
    torch.testing.assert_close(first.valid, second.valid)
    assert bool(((first.available_lengths >= 2) & (first.available_lengths <= 6)).all())


@pytest.mark.parametrize("length", [0, 6])
def test_prefix_padding_checks_available_length_bounds(length):
    with pytest.raises(ValueError, match=r"\[1, 5\]"):
        cv2.prefix_pad_history(torch.zeros(5, 2), length)


def test_normalization_fits_only_present_values_and_round_trips_artifact(tmp_path):
    # Two batches, three instants.  The first item of batch 1 is left padding and excluded.
    qpos = torch.tensor(
        [
            [[0.0] * 8, [1.0] * 8, [3.0] * 8],
            [[100.0] * 8, [5.0] * 8, [9.0] * 8],
        ]
    )
    valid = torch.tensor([[True, True, True], [False, True, True]])
    command = qpos + 10.0
    command_present = torch.tensor([[False, True, True], [False, False, True]])
    stats = cv2.fit_control_v2_normalization(
        qpos,
        history_valid=valid,
        previous_command=command,
        previous_command_present=command_present,
        dt=2.0,
    )

    assert stats.qpos.count == 5
    assert stats.velocity.count == 3
    assert stats.previous_command.count == 3
    torch.testing.assert_close(stats.qpos.mean, torch.full((8,), 3.6))
    # Valid velocities are (1-0)/2, (3-1)/2, and (9-5)/2.
    torch.testing.assert_close(stats.velocity.mean, torch.full((8,), 7.0 / 6.0))
    torch.testing.assert_close(stats.previous_command.mean, torch.full((8,), 43.0 / 3.0))

    path = tmp_path / "norm.json"
    cv2.save_normalization_artifact(stats, path)
    loaded = cv2.load_normalization_artifact(path)
    assert loaded.to_artifact() == stats.to_artifact()
    assert json.loads(path.read_text())["schema"] == cv2.NORMALIZATION_SCHEMA


def test_normalizer_zeroes_missing_velocity_and_previous_command_features():
    qpos = torch.arange(4, dtype=torch.float32)[:, None].expand(4, 8)
    state = _state_from_qpos(qpos)
    padded = cv2.prefix_pad_history(state, 2)
    command = qpos + 0.25
    command_present = torch.tensor([False, False, True, False])
    stats = cv2.fit_control_v2_normalization(
        qpos.unsqueeze(0),
        previous_command=command.unsqueeze(0),
        previous_command_present=command_present.unsqueeze(0),
    )
    features = cv2.build_state_features(
        padded.values,
        history_valid=padded.valid,
        previous_command=command,
        previous_command_present=command_present,
        normalizer=cv2.ControlV2Normalizer(stats),
    )

    assert features.tensor.shape == (4, 27)
    assert features.velocity_present.tolist() == [False, False, False, True]
    assert features.previous_command_present.tolist() == [False, False, True, False]
    assert torch.all(features.tensor[~features.velocity_present, 8:16] == 0)
    assert torch.all(features.tensor[~features.previous_command_present, 16:24] == 0)
    torch.testing.assert_close(features.velocity[-1], torch.ones(8))


def test_absent_command_has_neutral_stats_and_explicit_false_presence():
    qpos = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(4))
    stats = cv2.fit_control_v2_normalization(qpos)
    assert stats.previous_command.count == 0
    torch.testing.assert_close(stats.previous_command.mean, torch.zeros(8))
    torch.testing.assert_close(stats.previous_command.std, torch.ones(8))

    features = cv2.build_state_features(_state_from_qpos(qpos), normalizer=cv2.ControlV2Normalizer(stats))
    assert not bool(features.previous_command_present.any())
    assert torch.all(features.tensor[..., 16:24] == 0)


def test_phase_ids_are_pinned_to_named_contract_and_validated():
    assert cv2.PHASE_NAMES == ("approach", "close", "lift", "release")
    assert [int(phase) for phase in cv2.Phase] == [0, 1, 2, 3]
    labels = cv2.validate_phase_ids(torch.tensor([0, 1, 2, 3], dtype=torch.int16))
    assert labels.dtype == torch.long

    with pytest.raises(ValueError, match="approach"):
        cv2.validate_phase_ids(torch.tensor([4]))
    with pytest.raises(ValueError, match="phase_ids"):
        cv2.validate_phase_ids(torch.tensor([-1]))
    assert cv2.validate_phase_ids(torch.tensor([-1, 0]), allow_unlabeled=True).tolist() == [-1, 0]
    with pytest.raises(TypeError, match="integer"):
        cv2.validate_phase_ids(torch.tensor([0.0]))


def test_normalization_artifact_rejects_schema_and_nonpositive_scale():
    stats = cv2.fit_control_v2_normalization(torch.zeros(1, 2, 8))
    artifact = stats.to_artifact()
    artifact["schema"] = "future_schema"
    with pytest.raises(ValueError, match="unsupported"):
        cv2.ControlV2NormStats.from_artifact(artifact)

    artifact = stats.to_artifact()
    artifact["qpos"]["std"][0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        cv2.ControlV2NormStats.from_artifact(artifact)
