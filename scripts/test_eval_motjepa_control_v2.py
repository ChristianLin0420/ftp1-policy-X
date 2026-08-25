from __future__ import annotations

import hashlib
import pathlib
import types

import numpy as np

from scripts import eval_motjepa_control_v2


class _FakePolicy:
    artifacts = types.SimpleNamespace(
        step=4,
        completed_step=10,
        selection_metric="validation_action_loss",
        selection_value=0.125,
        metadata={"artifact_sha256": "a" * 64},
    )

    def get_safety_counters(self) -> dict[str, int]:
        return {
            "command_count": 4,
            "joint_clamp_commands": 1,
            "joint_clamped_values": 2,
            "rate_clamp_commands": 3,
            "rate_clamped_values": 5,
            "nonfinite_fallbacks": 0,
        }


def test_trace_extension_persists_safety_counters_and_updates_digest(tmp_path: pathlib.Path, monkeypatch) -> None:
    class Recorder:
        def __init__(self, output_path: pathlib.Path) -> None:
            self.output_path = output_path
            self.seed = 9

        def write(self, *, termination_reason: str, result: str) -> dict:
            del termination_reason, result
            np.savez_compressed(self.output_path, schema_version=np.asarray(1, dtype=np.int32))
            return {"sha256": "stale"}

    validated: list[tuple[pathlib.Path, int]] = []
    fake_eval = types.SimpleNamespace(
        EpisodeTraceRecorder=Recorder,
        validate_episode_trace=lambda path, expected_seed: validated.append((path, expected_seed)),
    )
    monkeypatch.setattr(eval_motjepa_control_v2, "_ACTIVE_POLICY", _FakePolicy())
    eval_motjepa_control_v2._install_trace_counter_extension(fake_eval)  # noqa: SLF001
    path = tmp_path / "trace.npz"

    metadata = fake_eval.EpisodeTraceRecorder(path).write(termination_reason="success", result="success")

    with np.load(path, allow_pickle=False) as trace:
        assert int(trace["control_v2_trace_schema"]) == 1
        assert int(trace["control_v2_clamp_count"]) == 4
        assert int(trace["control_v2_joint_clamped_values"]) == 2
        assert int(trace["control_v2_nonfinite_fallbacks"]) == 0
        assert int(trace["control_v2_policy_step"]) == 4
        assert int(trace["control_v2_completed_step"]) == 10
        assert str(trace["control_v2_selection_metric"]) == "validation_action_loss"
        assert str(trace["control_v2_artifact_sha256"]) == "a" * 64
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["control_v2_safety"]["clamp_count"] == 4
    assert metadata["control_v2_policy"]["selected_step"] == 4
    assert validated == [(path, 9)]
