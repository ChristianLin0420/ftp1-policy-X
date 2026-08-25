"""Safety checks for the policy evaluator's frozen training contract."""

# ruff: noqa: SLF001 -- these small pure helpers are intentionally tested directly.

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mot_jepa_policy_eval as policy_eval


def _policy_config():
    return SimpleNamespace(
        data=SimpleNamespace(strides=(2,), action_stride=1),
        head=SimpleNamespace(horizon=16),
        num_train_steps=20_000,
    )


def _write_stats(path, *, observation_strides=(2,), action_stride=1, horizon=16):
    np.savez(
        path,
        domains=np.asarray(["a", "b"]),
        observation_strides=np.asarray(observation_strides),
        action_stride=np.asarray(action_stride),
        horizon=np.asarray(horizon),
    )


def test_match_trained_value_infers_saved_value_and_rejects_override():
    assert policy_eval._match_trained_value("index_step", None, 4) == 4
    assert policy_eval._match_trained_value("index_step", 4, 4) == 4
    with pytest.raises(ValueError, match="does not match trained"):
        policy_eval._match_trained_value("index_step", 17, 4)


def test_action_stats_metadata_must_match_sampling_contract(tmp_path):
    stats = tmp_path / "action_stats.npz"
    _write_stats(stats)
    policy_eval._validate_action_stats_metadata(stats, _policy_config(), ["a", "b"])

    _write_stats(stats, action_stride=2)
    with pytest.raises(ValueError, match="sampling metadata"):
        policy_eval._validate_action_stats_metadata(stats, _policy_config(), ["a", "b"])


def test_action_stats_metadata_requires_domain_mapping(tmp_path):
    stats = tmp_path / "action_stats.npz"
    np.savez(
        stats,
        observation_strides=np.asarray([2]),
        action_stride=np.asarray(1),
        horizon=np.asarray(16),
    )
    with pytest.raises(ValueError, match="lacks metadata"):
        policy_eval._validate_action_stats_metadata(stats, _policy_config(), ["a", "b"])


def test_completed_run_gate_requires_done_at_configured_and_selected_step(tmp_path):
    cfg = _policy_config()
    with pytest.raises(RuntimeError, match="incomplete or preempted"):
        policy_eval._require_completed_run(tmp_path, cfg, None)

    (tmp_path / "DONE").write_text("19000")
    with pytest.raises(ValueError, match="requires 20000"):
        policy_eval._require_completed_run(tmp_path, cfg, None)

    (tmp_path / "DONE").write_text("20000")
    assert policy_eval._require_completed_run(tmp_path, cfg, None) == 20_000
    with pytest.raises(ValueError, match="requested head step"):
        policy_eval._require_completed_run(tmp_path, cfg, 19_000)
