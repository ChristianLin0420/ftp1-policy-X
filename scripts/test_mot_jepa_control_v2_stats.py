import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from openpi.mot_jepa.clip_dataset import ClipIndex
from scripts.mot_jepa_control_v2_stats import command_target_residuals
from scripts.mot_jepa_control_v2_stats import conservative_command_delta_limit
from scripts.mot_jepa_control_v2_stats import store_contract_digest
from scripts.mot_jepa_control_v2_stats import validate_source_control_contract
from scripts.mot_jepa_control_v2_stats import validate_validation_oracle_safety


def _write_array(group: zarr.Group, name: str, value: np.ndarray) -> None:
    array = group.create_array(name, shape=value.shape, dtype=value.dtype)
    array[:] = value


def _control_store(
    tmp_path,
    *,
    include_contact: bool = True,
    valid: bool = True,
    control_step_stride: int = 1,
    episode_count: int = 1,
) -> str:
    path = tmp_path / "control.zarr"
    root = zarr.open_group(path, mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    rows = 12
    _write_array(data, "state", np.zeros((rows, 120), dtype=np.float32))
    _write_array(data, "command8", np.zeros((rows, 8), dtype=np.float32))
    _write_array(data, "command_valid", np.full(rows, valid, dtype=np.uint8))
    _write_array(data, "control_step", np.arange(rows, dtype=np.int64) * control_step_stride)
    _write_array(data, "control_step_valid", np.full(rows, valid, dtype=np.uint8))
    _write_array(data, "phase_id", np.tile(np.arange(4, dtype=np.int64), 3))
    if include_contact:
        _write_array(data, "contact", np.tile(np.array([0.0, 1.0], dtype=np.float32), rows // 2))
        _write_array(data, "contact_valid", np.full(rows, valid, dtype=np.uint8))
    if not 1 <= episode_count <= rows:
        raise ValueError("test episode_count must be within the row count")
    episode_ends = np.concatenate(
        (
            np.arange(1, episode_count, dtype=np.int64),
            np.array([rows], dtype=np.int64),
        )
    )
    _write_array(meta, "episode_ends", episode_ends)
    _write_array(meta, "source_episode_seed", np.arange(2_000_000, 2_000_000 + episode_count, dtype=np.int64))
    return str(path)


def test_authoritative_source_contract_reports_episode_and_phase_counts(tmp_path) -> None:
    path = _control_store(tmp_path)
    authoritative, episodes, phase_counts, contact_counts = validate_source_control_contract(
        [path], require_authoritative=True, minimum_total_episodes=1
    )
    assert authoritative
    assert episodes == 1
    assert phase_counts == (3, 3, 3, 3)
    assert contact_counts == (6, 6)


def test_source_contract_rejects_missing_or_invalid_authoritative_labels(tmp_path) -> None:
    missing = _control_store(tmp_path / "missing", include_contact=False)
    with pytest.raises(ValueError, match="authoritative V3 arrays"):
        validate_source_control_contract([missing], require_authoritative=True, minimum_total_episodes=1)

    invalid = _control_store(tmp_path / "invalid", valid=False)
    with pytest.raises(ValueError, match="must be authoritative"):
        validate_source_control_contract([invalid], require_authoritative=True, minimum_total_episodes=1)


def test_source_contract_enforces_episode_floor(tmp_path) -> None:
    path = _control_store(tmp_path)
    with pytest.raises(ValueError, match="require at least 1000"):
        validate_source_control_contract([path], require_authoritative=True, minimum_total_episodes=1000)


def test_source_contract_rejects_overfull_exact_episode_target(tmp_path) -> None:
    path = _control_store(tmp_path, episode_count=2)
    with pytest.raises(ValueError, match="require exactly 1"):
        validate_source_control_contract(
            [path],
            require_authoritative=True,
            minimum_total_episodes=1,
            exact_total_episodes=1,
        )


def test_source_contract_rejects_offline_online_control_cadence_mismatch(tmp_path) -> None:
    path = _control_store(tmp_path, control_step_stride=2)
    with pytest.raises(ValueError, match="one simulator control step"):
        validate_source_control_contract([path], require_authoritative=True, minimum_total_episodes=1)


def test_action_scale_samples_future_sent_commands_relative_to_current_state() -> None:
    rows = 40
    qpos = np.zeros((rows, 8), dtype=np.float32)
    command = np.arange(rows, dtype=np.float32)[:, None] * np.ones((1, 8), dtype=np.float32)

    residual = command_target_residuals(qpos, command, np.array([2], dtype=np.int64))

    assert residual.shape == (1, 31, 8)
    np.testing.assert_allclose(residual[0, 0], 3.0)
    np.testing.assert_allclose(residual[0, -1], 33.0)


def test_rate_limit_covers_expert_command_from_observed_qpos_without_crossing_episodes() -> None:
    qpos = np.zeros((6, 8), dtype=np.float32)
    command = np.zeros((6, 8), dtype=np.float32)
    command[1:3] = 0.4
    # A large reset discontinuity begins episode two and must not enter the transition maximum.
    qpos[3:] = 10.0
    command[3:] = 10.0
    command[4:] = 10.2

    limit = conservative_command_delta_limit(qpos, command, np.array([3, 6]), margin=0.1)

    within_episode = np.concatenate((command[1:3] - qpos[:2], command[4:6] - qpos[3:5]))
    assert np.all(np.abs(within_episode) < limit)
    assert np.all(limit < 1.0)


def test_rate_limit_can_fit_train_episodes_without_validation_leakage() -> None:
    qpos = np.zeros((6, 8), dtype=np.float32)
    command = np.zeros((6, 8), dtype=np.float32)
    command[1:3] = 0.2
    command[4:6] = 9.0  # held-out episode must not inflate the deploy limit

    limit = conservative_command_delta_limit(
        qpos,
        command,
        np.array([3, 6]),
        episode_indices=np.array([0]),
        margin=0.1,
    )

    assert np.all(limit > 0.2)
    assert np.all(limit < 0.3)


def test_store_contract_digest_is_invariant_to_prepared_path(tmp_path) -> None:
    first = _control_store(tmp_path / "first")
    second = tmp_path / "second" / "control.zarr"
    second.parent.mkdir(parents=True)
    shutil.copytree(first, second)

    assert store_contract_digest([first]) == store_contract_digest([str(second)])


def _oracle_dataset(tmp_path, *, delta: float, horizon: int = 1):
    path = tmp_path / "oracle.zarr"
    root = zarr.open_group(path, mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    rows = 80
    state = np.zeros((rows, 120), dtype=np.float32)
    command = np.zeros((rows, 8), dtype=np.float32)
    current = 30
    command[current + horizon, 0] = delta
    _write_array(data, "state", state)
    _write_array(data, "command8", command)
    _write_array(data, "command_valid", np.ones(rows, dtype=np.uint8))
    _write_array(meta, "episode_ends", np.array([rows], dtype=np.int64))
    _write_array(meta, "source_episode_seed", np.array([2_000_000], dtype=np.int64))
    index = ClipIndex(np.array([[0, 0, 2, 0]], dtype=np.int64), [str(path)])
    return SimpleNamespace(
        clip_index=index,
        store_paths=[str(path)],
        base=SimpleNamespace(layout=SimpleNamespace(num_frames=16)),
        action_stride=1,
    )


def test_validation_oracle_safety_requires_strictly_positive_slack(tmp_path) -> None:
    safe = _oracle_dataset(tmp_path / "safe", delta=0.1)
    audit = validate_validation_oracle_safety(safe, (0,), np.full(8, 0.2), first_n=20)
    assert audit.violation_values == 0
    assert audit.min_slack > 0
    assert audit.max_rate_ratio == pytest.approx(0.5)

    equal = _oracle_dataset(tmp_path / "equal", delta=0.2)
    with pytest.raises(ValueError, match="expert is infeasible"):
        validate_validation_oracle_safety(equal, (0,), np.full(8, 0.2), first_n=20)

    nonfinite = _oracle_dataset(tmp_path / "nonfinite", delta=float("nan"))
    with pytest.raises(ValueError, match="expert is infeasible"):
        validate_validation_oracle_safety(nonfinite, (0,), np.full(8, 0.2), first_n=20)


def test_validation_oracle_safety_ignores_undeployed_rows(tmp_path) -> None:
    dataset = _oracle_dataset(tmp_path, delta=1.0, horizon=21)
    audit = validate_validation_oracle_safety(dataset, (0,), np.full(8, 0.2), first_n=20)
    assert audit.violation_values == 0
