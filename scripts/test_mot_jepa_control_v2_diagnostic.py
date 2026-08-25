from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch

from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.control_v2_deploy import load_trained_control_v2_artifacts
from openpi.mot_jepa.control_v2_deploy_test import _write_completed_run
from scripts import mot_jepa_control_v2_diagnostic as diagnostic

_TRAIN_JOB = "123456"


def _write_unqualified_final_run(tmp_path: pathlib.Path) -> pathlib.Path:
    run = _write_completed_run(tmp_path)
    selected = json.loads((run / "BEST_VALIDATION.json").read_text())
    (run / "DONE").unlink()
    (run / "BEST_VALIDATION.json").unlink()
    shutil.rmtree(run / "checkpoints" / "best_validation_1")
    (run / "FAILED").write_text(f"{diagnostic.EXPECTED_TRAINING_FAILURE}\n")
    (run / f"FAILED.train.{_TRAIN_JOB}.json").write_text(
        json.dumps({"status": "FAILED", "job_id": _TRAIN_JOB, "exit_code": 1, "updated": "unit"})
    )
    (run / "TRAIN_SOURCE_PROVENANCE.json").write_text("{}\n")

    config = ControlV2TrainConfig.from_json((run / "run_config.json").read_text())
    metadata_path = run / "checkpoints" / "1" / "metadata.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    metrics = {name: value for name, value in selected.items() if name.startswith("validation_")}
    metrics.update(
        validation_rate_violation_values=7.0,
        validation_safety_violation_values=7.0,
        validation_rate_max_ratio=3.0,
    )
    metadata.update(
        best_validation_step=None,
        best_validation_chunk_loss=None,
        world_size=4,
        optimizer_step_min=1,
        optimizer_step_max=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        local_batch_size=config.local_batch_size,
        last_validation_metrics=metrics,
    )
    torch.save(metadata, metadata_path)
    return run


def test_strict_loader_rejects_but_diagnostic_loader_restores_exact_final(tmp_path: pathlib.Path) -> None:
    run = _write_unqualified_final_run(tmp_path)

    with pytest.raises(RuntimeError, match="require both DONE"):
        load_trained_control_v2_artifacts(run, "cpu", step=1, verify_source_stores=False)

    contract = diagnostic.validate_unqualified_final_run(run, step=1, train_job_id=_TRAIN_JOB)
    loaded = diagnostic.load_unqualified_final_control_v2_artifacts(
        run,
        "cpu",
        step=1,
        train_job_id=_TRAIN_JOB,
        verify_source_stores=False,
    )

    assert contract["status"] == diagnostic.DIAGNOSTIC_STATUS
    assert contract["final_validation_metrics"]["validation_safety_violation_values"] == 7
    assert loaded.step == loaded.completed_step == 1
    assert loaded.selection_metric == diagnostic.DIAGNOSTIC_SELECTION_METRIC
    assert loaded.checkpoint == (run / "checkpoints" / "1")


@pytest.mark.parametrize("mutation", ["done", "best", "wrong_failure", "qualified"])
def test_diagnostic_loader_rejects_nonexact_terminal_contract(tmp_path: pathlib.Path, mutation: str) -> None:
    run = _write_unqualified_final_run(tmp_path)
    if mutation == "done":
        (run / "DONE").write_text("1\n")
    elif mutation == "best":
        (run / "BEST_VALIDATION.json").write_text("{}\n")
    elif mutation == "wrong_failure":
        (run / "FAILED").write_text("RuntimeError: unrelated crash\n")
    else:
        metadata_path = run / "checkpoints" / "1" / "metadata.pt"
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        metadata["last_validation_metrics"].update(
            validation_rate_violation_values=0.0,
            validation_safety_violation_values=0.0,
            validation_rate_max_ratio=0.9,
        )
        torch.save(metadata, metadata_path)

    with pytest.raises((FileNotFoundError, ValueError)):
        diagnostic.validate_unqualified_final_run(run, step=1, train_job_id=_TRAIN_JOB)


def test_diagnostic_launchers_never_claim_formal_acceptance() -> None:
    root = pathlib.Path(__file__).parents[1]
    sbatch = (root / "scripts_exp_zarr/mot_jepa/control_v2_closedloop_diagnostic.sbatch").read_text()
    inner = (root / "scripts_exp_zarr/mot_jepa/control_v2_closedloop_diagnostic_inner.sh").read_text()
    submit = (root / "scripts_exp_zarr/mot_jepa/control_v2_submit_diagnostic.sh").read_text()

    assert "afterany:${TRAIN_JOB_ID}" in submit
    assert "afterok:${smoke_job}" in submit
    assert "START_SEED=900000" in submit
    assert "START_SEED=900100" in submit
    assert 'mot_jepa_control_v2_acceptance.py" "${acceptance_args' not in sbatch
    assert "DIAGNOSTIC_EVAL_COMPLETE" in sbatch
    assert '> "${PAIR_ROOT}/DONE"' not in sbatch
    assert "official FTP1 evaluation start" not in inner
    assert "MOT_CONTROL_V2_DIAGNOSTIC_UNQUALIFIED_FINAL=1" in inner
    assert "MOT_CONTROL_V2_DIAGNOSTIC_OUTER_STORE_VERIFIED=1" in sbatch
    assert "diagnostic rollout requires outer prepared-store verification" in inner


def test_diagnostic_entrypoint_imports_without_zarr(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"split_identity": "source_episode_seed_v1"}))
    script = pathlib.Path(__file__).with_name("eval_motjepa_control_v2_diagnostic.py")
    probe = """
import builtins
import runpy
import sys

real_import = builtins.__import__

def no_zarr(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "zarr" or name.startswith("zarr."):
        raise ModuleNotFoundError("blocked diagnostic dependency probe", name="zarr")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = no_zarr
runpy.run_path(sys.argv[1], run_name="diagnostic_import_probe")
"""
    environment = os.environ.copy()
    environment.update(
        MOT_CONTROL_V2_RUN=str(run),
        MOT_CONTROL_V2_DIAGNOSTIC_UNQUALIFIED_FINAL="1",
        MOT_CONTROL_V2_DIAGNOSTIC_OUTER_STORE_VERIFIED="1",
    )
    subprocess.run(
        [sys.executable, "-c", probe, str(script)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
