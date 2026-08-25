from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "UniVTAC" / "scripts"))

from episode_trace import EpisodeTraceRecorder  # noqa: E402
from episode_trace import validate_episode_trace  # noqa: E402


def _append(recorder: EpisodeTraceRecorder, index: int, *, success: bool = False, early_stop: bool = False) -> bool:
    qpos_before = np.linspace(0.0, 0.7, 8, dtype=np.float32) + index
    sent = qpos_before + 0.01
    raw = np.zeros(120, dtype=np.float32)
    raw[9:16] = 0.01
    raw[44] = 0.02
    return recorder.append(
        action_index=index,
        sim_step_before=100 + index - 1,
        sim_step_after=100 + index,
        qpos8_before=qpos_before,
        qpos8_after=sent,
        sent_action8=sent,
        raw_first_vector=raw,
        raw_first_action8=np.r_[np.full(7, 0.01, dtype=np.float32), np.float32(0.02)],
        resolved_first_action8=sent,
        first_executable_index=0,
        raw_action_rep="mix",
        exec_success=True,
        eval_success_after=success,
        early_stop_after=early_stop,
    )


def test_episode_trace_round_trip_is_seeded_atomic_and_pickle_free(tmp_path: Path) -> None:
    path = tmp_path / "trajectory" / "worker_0" / "1000007.npz"
    recorder = EpisodeTraceRecorder(seed=1000007, output_path=path, max_actions=500)
    assert _append(recorder, 1)
    assert _append(recorder, 2, early_stop=True)

    metadata = recorder.write(termination_reason="task_early_stop", result="failed")

    assert metadata["seed"] == 1000007
    assert metadata["recorded_actions"] == 2
    assert metadata["total_actions_seen"] == 2
    assert metadata["raw_action_dim"] == 120
    assert metadata["raw_action_rep"] == "mix"
    assert len(metadata["sha256"]) == 64
    assert not list(path.parent.glob(f".{path.name}.tmp.*"))
    validated = validate_episode_trace(path, expected_seed=1000007)
    assert validated["seed"] == metadata["seed"]
    assert validated["recorded_actions"] == metadata["recorded_actions"]
    assert validated["sha256"] == metadata["sha256"]
    assert validated["termination_reason"] == metadata["termination_reason"]
    assert validated["result"] == "failed"
    with np.load(path, allow_pickle=False) as trace:
        assert trace["qpos8_before"].shape == (2, 8)
        assert trace["raw_first_vector"].shape == (2, 120)
        assert trace["sent_action8"].dtype == np.float32
        assert trace["action_index"].tolist() == [1, 2]
        assert trace["early_stop_after"].tolist() == [False, True]
        assert str(trace["raw_action_rep"]) == "mix"
        assert str(trace["termination_reason"]) == "task_early_stop"


def test_episode_trace_cap_is_explicit_and_keeps_size_bounded(tmp_path: Path) -> None:
    path = tmp_path / "trace.npz"
    recorder = EpisodeTraceRecorder(seed=4, output_path=path, max_actions=2)
    assert _append(recorder, 1)
    assert _append(recorder, 2)
    assert not _append(recorder, 3, success=True)

    metadata = recorder.write(termination_reason="success", result="success")

    assert metadata["recorded_actions"] == 2
    assert metadata["total_actions_seen"] == 3
    assert metadata["truncated"] is True
    with np.load(path, allow_pickle=False) as trace:
        assert int(trace["num_actions"]) == 2
        assert int(trace["total_actions_seen"]) == 3
        assert bool(trace["truncated"])


def test_episode_trace_can_record_an_error_before_any_action(tmp_path: Path) -> None:
    path = tmp_path / "empty_error.npz"
    recorder = EpisodeTraceRecorder(seed=9, output_path=path, max_actions=5)

    recorder.write(termination_reason="error", result="error")

    with np.load(path, allow_pickle=False) as trace:
        assert trace["qpos8_before"].shape == (0, 8)
        assert trace["raw_first_vector"].shape == (0, 0)
        assert int(trace["seed"]) == 9


def test_episode_trace_rejects_bad_seed_nonfinite_values_and_inconsistent_result(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    recorder = EpisodeTraceRecorder(seed=12, output_path=path, max_actions=5)
    with pytest.raises(ValueError, match="non-finite"):
        recorder.append(
            action_index=1,
            sim_step_before=0,
            sim_step_after=1,
            qpos8_before=np.full(8, np.nan),
            qpos8_after=np.zeros(8),
            sent_action8=np.zeros(8),
            raw_first_vector=np.zeros(120),
            raw_first_action8=np.zeros(8),
            resolved_first_action8=np.zeros(8),
            first_executable_index=0,
            raw_action_rep="mix",
            exec_success=True,
            eval_success_after=False,
            early_stop_after=False,
        )

    assert _append(recorder, 1)
    with pytest.raises(ValueError, match="requires result"):
        recorder.write(termination_reason="action_limit", result="success")
    recorder.write(termination_reason="action_limit", result="failed")
    with pytest.raises(ValueError, match="does not match expected seed"):
        validate_episode_trace(path, expected_seed=13)
