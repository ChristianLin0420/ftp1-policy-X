from __future__ import annotations

import dataclasses
import json

import pytest
import torch
from torch import nn

from openpi.mot_jepa import runtime
from openpi.mot_jepa.config import CONFIGS
from openpi.mot_jepa.config import MotJepaTrainConfig
from openpi.mot_jepa.ema import EmaTeacher


def tiny_state(tmp_path, step: int) -> None:
    model = nn.Linear(4, 4)
    teacher = EmaTeacher(model, runtime_dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime.save_checkpoint(
        tmp_path, step, student=model, teacher=teacher, optimizer=optimizer, config_json="{}", keep_last=99
    )


def test_find_latest_step_returns_none_on_a_fresh_run(tmp_path):
    """Idempotent auto-resume needs a real None; the FTP-1 helper raises instead, which left
    its own caller's None branch as dead code."""
    assert runtime.find_latest_step(tmp_path / "missing") is None
    (tmp_path / "empty").mkdir()
    assert runtime.find_latest_step(tmp_path / "empty") is None


def test_find_latest_step_prefers_the_pointer_but_falls_back_to_a_scan(tmp_path):
    tiny_state(tmp_path, 10)
    tiny_state(tmp_path, 20)
    assert runtime.find_latest_step(tmp_path) == 20

    (tmp_path / "latest").write_text("not-a-number")
    assert runtime.find_latest_step(tmp_path) == 20

    (tmp_path / "latest").unlink()
    assert runtime.find_latest_step(tmp_path) == 20


def test_find_latest_step_ignores_a_pointer_to_a_missing_directory(tmp_path):
    tiny_state(tmp_path, 10)
    (tmp_path / "latest").write_text("999")
    assert runtime.find_latest_step(tmp_path) == 10


def test_checkpoint_roundtrip_restores_step_and_weights(tmp_path):
    model = nn.Linear(4, 4)
    teacher = EmaTeacher(model, runtime_dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime.save_checkpoint(tmp_path, 7, student=model, teacher=teacher, optimizer=optimizer, config_json="{}")

    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    restored_step = runtime.load_checkpoint(
        tmp_path, 7, student=model, teacher=teacher, optimizer=optimizer, device=torch.device("cpu")
    )
    assert restored_step == 7
    assert torch.equal(model.weight, original)


def test_retention_keeps_last_n_plus_every_period(tmp_path):
    for step in range(0, 60, 10):
        tiny_state(tmp_path, step)
    kept_before = sorted(int(p.name) for p in tmp_path.iterdir() if p.is_dir())
    assert kept_before == [0, 10, 20, 30, 40, 50]

    runtime.prune_checkpoints(tmp_path, keep_last=2, keep_period=30)
    kept = sorted(int(p.name) for p in tmp_path.iterdir() if p.is_dir())
    assert kept == [0, 30, 40, 50], "keeps the newest 2 plus every multiple of 30"


def test_save_never_leaves_zero_checkpoints_on_disk(tmp_path):
    """Pruning runs only after the new checkpoint is committed."""
    model = nn.Linear(4, 4)
    teacher = EmaTeacher(model, runtime_dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(5):
        runtime.save_checkpoint(
            tmp_path, step, student=model, teacher=teacher, optimizer=optimizer, config_json="{}", keep_last=1
        )
        assert any(p.is_dir() for p in tmp_path.iterdir())
    assert runtime.find_latest_step(tmp_path) == 4


def test_save_is_atomic_and_leaves_no_staging_directory(tmp_path):
    model = nn.Linear(4, 4)
    teacher = EmaTeacher(model, runtime_dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime.save_checkpoint(tmp_path, 3, student=model, teacher=teacher, optimizer=optimizer, config_json="{}")
    assert not list(tmp_path.glob("tmp_*"))
    assert (tmp_path / "3" / "student.pt").exists()
    assert (tmp_path / "3" / "teacher_ema.pt").exists()


def test_resolve_run_config_freezes_on_first_launch(tmp_path):
    frozen, drift = runtime.resolve_run_config(tmp_path, json.dumps({"lr": 1.0}))
    assert json.loads(frozen) == {"lr": 1.0}
    assert drift == {}


def test_resolve_run_config_keeps_the_frozen_value_and_reports_drift(tmp_path):
    """A requeue three days later must not silently pick up an edited launcher."""
    runtime.resolve_run_config(tmp_path, json.dumps({"lr": 1.0, "steps": 10}))
    frozen, drift = runtime.resolve_run_config(tmp_path, json.dumps({"lr": 2.0, "steps": 10}))
    assert json.loads(frozen)["lr"] == 1.0, "the frozen config must win"
    assert drift == {"lr": (1.0, 2.0)}


class _FakeMonitor(runtime.PreemptionMonitor):
    def __init__(self, flag_path, device):
        super().__init__(flag_path, device, signals=())


def test_preemption_monitor_detects_the_flag_file(tmp_path):
    flag = tmp_path / "PREEMPT_REQUEST"
    monitor = _FakeMonitor(flag, torch.device("cpu"))
    assert not monitor.should_stop(0)
    flag.touch()
    assert monitor.should_stop(0)


def test_preemption_decision_latches(tmp_path):
    """Once decided, removing the flag must not un-decide: the checkpoint is already in flight."""
    flag = tmp_path / "PREEMPT_REQUEST"
    monitor = _FakeMonitor(flag, torch.device("cpu"))
    flag.touch()
    assert monitor.should_stop(0)
    flag.unlink()
    assert monitor.should_stop(10)


def test_preemption_only_polls_on_the_check_interval(tmp_path):
    flag = tmp_path / "PREEMPT_REQUEST"
    monitor = _FakeMonitor(flag, torch.device("cpu"))
    flag.touch()
    assert not monitor.should_stop(3), "step 3 is not a multiple of the default interval of 10"
    assert monitor.should_stop(10)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_named_configs_roundtrip_through_json(name):
    """The frozen-config path serializes and rebuilds the whole nested tree.

    ``from __future__ import annotations`` makes ``field.type`` a string, so a naive rebuild
    silently leaves nested sections as plain dicts and only fails much later in the trainer.
    """
    original = CONFIGS[name]
    restored = MotJepaTrainConfig.from_json(original.to_json())
    assert restored == original
    assert restored.data.num_workers == original.data.num_workers
    assert isinstance(restored.masking.mode_probs, tuple)
    assert restored.layout.num_tokens == original.layout.num_tokens


def test_config_diff_reports_changed_fields():
    base = CONFIGS["mot_jepa_debug"]
    changed = dataclasses.replace(base, seed=999)
    assert base.diff(changed) == {"seed": (base.seed, 999)}
