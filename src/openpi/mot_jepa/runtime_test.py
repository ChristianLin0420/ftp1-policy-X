from __future__ import annotations

import concurrent.futures
import dataclasses
import json

import numpy as np
import pytest
import torch
from torch import nn

from openpi.mot_jepa import runtime
from openpi.mot_jepa.config import CONFIGS
from openpi.mot_jepa.config import MotJepaTrainConfig
from openpi.mot_jepa.ema import EmaTeacher
from scripts.mot_jepa_train import InfiniteBatchSampler


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


def test_checkpoint_roundtrip_restores_loss_module_state(tmp_path):
    """The synchrony projectors live outside the student and were silently lost on requeue.

    ``MotJepaLoss`` owns trainable projectors and a running ``lowdim_scale``, and its parameters
    are in the optimizer's list (mot_jepa_train.py:306). Saving only the student meant every
    requeue rebuilt them at random init while the RESTORED optimizer reapplied the old Adam
    moments to those fresh weights -- worse than a clean restart. probe3's requeue at step 37970
    showed it: retrieval 0.586 -> 0.176, loss 0.66 -> 1.04 in a single interval.
    """
    model = nn.Linear(4, 4)
    loss_fn = nn.Module()
    loss_fn.projector = nn.Linear(4, 4)
    loss_fn.register_buffer("lowdim_scale", torch.tensor(3.5))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)
    runtime.save_checkpoint(
        tmp_path, 7, student=model, teacher=None, optimizer=optimizer, config_json="{}", loss_fn=loss_fn
    )

    original = loss_fn.projector.weight.detach().clone()
    with torch.no_grad():
        loss_fn.projector.weight.add_(1.0)
        loss_fn.lowdim_scale.fill_(1.0)

    runtime.load_checkpoint(
        tmp_path, 7, student=model, teacher=None, optimizer=optimizer, device=torch.device("cpu"), loss_fn=loss_fn
    )
    assert torch.equal(loss_fn.projector.weight, original)
    assert float(loss_fn.lowdim_scale) == 3.5


def test_checkpoint_without_loss_state_still_loads(tmp_path):
    """Checkpoints written before loss.pt existed must still resume, with a warning not a crash."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime.save_checkpoint(tmp_path, 5, student=model, teacher=None, optimizer=optimizer, config_json="{}")
    assert not (tmp_path / "5" / "loss.pt").exists()

    loss_fn = nn.Module()
    loss_fn.projector = nn.Linear(4, 4)
    step = runtime.load_checkpoint(
        tmp_path, 5, student=model, teacher=None, optimizer=optimizer, device=torch.device("cpu"), loss_fn=loss_fn
    )
    assert step == 5


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


def test_resolve_run_config_is_safe_when_many_ranks_race(tmp_path):
    """Reproduces the two-node failure: the claim is atomic, the content was not.

    Every rank calls this at once. The old implementation claimed the destination with
    O_CREAT|O_EXCL and wrote into that descriptor, so losers caught FileExistsError and read
    a file that existed but was still EMPTY -- json.loads("") raised on every rank but one. A
    single node survived on page-cache timing; two nodes on Lustre did not.
    """
    payload = json.dumps({"lr": 1.0, "steps": 10, "pad": "x" * 200_000})

    def worker(_):
        frozen, drift = runtime.resolve_run_config(tmp_path, payload)
        return json.loads(frozen)["lr"], drift

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(worker, range(16)))

    assert all(lr == 1.0 for lr, _ in results), "every rank must read the same frozen config"
    assert all(drift == {} for _, drift in results)


def test_resolve_run_config_never_leaves_a_partial_file(tmp_path):
    """The destination must never exist in a half-written state."""
    payload = json.dumps({"a": 1})
    runtime.resolve_run_config(tmp_path, payload)
    assert json.loads((tmp_path / "run_config.json").read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp")), "staging file must be renamed away"


def test_resolve_run_config_times_out_rather_than_hanging(tmp_path):
    """A claim with no config behind it must fail loudly, not block a whole job forever."""
    (tmp_path / "run_config.claim").touch()  # winner claimed then died before writing
    with pytest.raises(TimeoutError, match="did not become readable"):
        runtime.resolve_run_config(tmp_path, json.dumps({"a": 1}), timeout_s=0.5, poll_s=0.05)


def test_domain_pure_batches_never_mix_domains():
    """Every batch must be single-domain, or Level-A InfoNCE is solvable by dataset identity.

    Measured on a mixed-domain run: the positional-shortcut control reached the SAME top-1
    as real tactile (retrieval_gap 0.0 at step 6000, ratio-to-chance 14.0). A GelSight clip
    and a uSkin clip differ so obviously that matching video to touch needs no temporal
    correspondence at all.
    """
    domain_of_sample = np.repeat(np.arange(4), 500)  # 4 domains x 500 clips
    samplers = [
        InfiniteBatchSampler(2000, 8, rank=r, world_size=2, seed=42, domain_of_sample=domain_of_sample)
        for r in range(2)
    ]
    for step in range(60):
        per_rank = [s.indices_for_step(step) for s in samplers]
        domains = {int(domain_of_sample[i]) for idx in per_rank for i in idx}
        assert len(domains) == 1, f"step {step} mixed domains {domains}"
        assert not set(per_rank[0]) & set(per_rank[1]), "ranks must get disjoint clips"

    # Every domain should be reachable, weighted by size.
    seen = {int(domain_of_sample[samplers[0].indices_for_step(s)[0]]) for s in range(300)}
    assert seen == {0, 1, 2, 3}


def test_domain_smaller_than_a_global_batch_is_dropped_not_padded():
    domain_of_sample = np.concatenate([np.zeros(500, int), np.ones(3, int)])
    s = InfiniteBatchSampler(503, 8, rank=0, world_size=2, seed=1, domain_of_sample=domain_of_sample)
    assert len(s.domain_pools) == 1, "the 3-clip domain cannot fill a 16-clip batch"
