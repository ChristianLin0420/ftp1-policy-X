import inspect

import numpy as np
import torch

from openpi.mot_jepa.clip_dataset import ClipIndex
from openpi.mot_jepa.control_v2_data import ACTION_HORIZON
from openpi.mot_jepa.control_v2_dataset import ControlV2Dataset
from openpi.mot_jepa.control_v2_dataset import build_command_action_target
from openpi.mot_jepa.control_v2_dataset import infer_legacy_phase_ids
from openpi.mot_jepa.control_v2_dataset import split_control_v2_index


def test_legacy_phase_inference_is_episode_local() -> None:
    one = np.array([0.02, 0.02, 0.015, 0.01, 0.01, 0.015, 0.02], dtype=np.float32)
    phases = infer_legacy_phase_ids(np.concatenate([one, one]), np.array([len(one), 2 * len(one)]))
    np.testing.assert_array_equal(phases[: len(one)], phases[len(one) :])
    assert set(phases.tolist()) == {0, 1, 2, 3}


def test_split_is_by_episode_and_disjoint() -> None:
    rows = np.array(
        [[0, episode * 100 + clip, 2, episode] for episode in range(20) for clip in range(3)],
        dtype=np.int64,
    )
    index = ClipIndex(rows, ["/tmp/lift_bottle.zarr"])
    episode_seeds = (np.arange(2_000_000, 2_000_020, dtype=np.int64),)
    train = split_control_v2_index(
        index,
        split="train",
        source_episode_seeds=episode_seeds,
        validation_fraction=0.25,
        seed=7,
    )
    validation = split_control_v2_index(
        index,
        split="validation",
        source_episode_seeds=episode_seeds,
        validation_fraction=0.25,
        seed=7,
    )
    train_episodes = set(train.entries[:, 3].tolist())
    validation_episodes = set(validation.entries[:, 3].tolist())
    assert train_episodes.isdisjoint(validation_episodes)
    assert train_episodes | validation_episodes == set(range(20))
    assert len(train) + len(validation) == len(index)


def test_split_is_invariant_to_prepared_store_path() -> None:
    rows = np.array(
        [[0, episode * 100 + clip, 2, episode] for episode in range(50) for clip in range(2)],
        dtype=np.int64,
    )
    seeds = (np.arange(2_100_000, 2_100_050, dtype=np.int64),)
    old = ClipIndex(rows, ["/prepared/lift_bottle_6716916/clips/lift_bottle_head.zarr"])
    rebuilt = ClipIndex(rows, ["/prepared/lift_bottle_9999999/clips/lift_bottle_head.zarr"])

    old_validation = split_control_v2_index(
        old,
        split="validation",
        source_episode_seeds=seeds,
        validation_fraction=0.15,
        seed=42,
    )
    rebuilt_validation = split_control_v2_index(
        rebuilt,
        split="validation",
        source_episode_seeds=seeds,
        validation_fraction=0.15,
        seed=42,
    )

    np.testing.assert_array_equal(old_validation.entries, rebuilt_validation.entries)


def test_action_target_is_next_sent_command_not_lagging_future_qpos() -> None:
    current_qpos = torch.arange(8, dtype=torch.float32) / 10
    future_command = current_qpos + torch.arange(1, ACTION_HORIZON, dtype=torch.float32)[:, None] / 100
    lagging_qpos = future_command - 0.25
    valid = torch.ones(ACTION_HORIZON - 1, dtype=torch.bool)
    valid[4] = False

    mixed, absolute = build_command_action_target(
        current_qpos,
        future_command,
        command_valid=valid,
        fallback_future_qpos=lagging_qpos,
    )

    torch.testing.assert_close(absolute[0], current_qpos)
    torch.testing.assert_close(absolute[1:5], future_command[:4])
    torch.testing.assert_close(absolute[5], lagging_qpos[4])
    torch.testing.assert_close(mixed[1, :7], future_command[0, :7] - current_qpos[:7])
    torch.testing.assert_close(mixed[1, 7], future_command[0, 7])


def test_dataset_exposes_explicit_simulator_control_cadence() -> None:
    assert "control_step_stride" in inspect.signature(ControlV2Dataset).parameters
