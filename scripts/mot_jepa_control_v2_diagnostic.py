#!/usr/bin/env python
"""Fail-closed diagnostic loading and result aggregation for an unqualified V3 final checkpoint.

This module is intentionally separate from the production deployment and acceptance paths.  It
never creates ``DONE`` or ``BEST_VALIDATION.json`` in a training run and never turns a diagnostic
rollout into an official acceptance result.  The only supported input is the exact final numeric
checkpoint from a run that completed every optimizer update and then failed solely because no
held-out checkpoint satisfied the zero-violation safety selector.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import tempfile
import threading
from typing import Any

import numpy as np
import torch

from openpi.mot_jepa.control_v2_config import CONTROL_V2_ARTIFACT_SCHEMA
from openpi.mot_jepa.control_v2_config import ControlV2Artifact
from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.control_v2_deploy import TrainedControlV2Artifacts
from openpi.mot_jepa.control_v2_runtime import SafetyLimits

try:
    from scripts import mot_jepa_control_v2_acceptance as acceptance
except ImportError:  # Direct execution puts the repository's scripts directory on sys.path.
    import mot_jepa_control_v2_acceptance as acceptance


DIAGNOSTIC_STATUS = "DIAGNOSTIC_ONLY_UNQUALIFIED_FINAL"
DIAGNOSTIC_SELECTION_METRIC = "diagnostic_unqualified_final_validation_action_loss"
EXPECTED_TRAINING_FAILURE = (
    "RuntimeError: training completed without any checkpoint satisfying zero held-out safety-limit violations"
)
REQUIRED_CHECKPOINT_FILES = (
    "student.pt",
    "backbone.pt",
    "loss.pt",
    "optimizer.pt",
    "metadata.pt",
    "train_config.json",
)
_VALIDATION_STRING_FIELDS = {"validation_sampling", "validation_sample_sha256"}
_VALIDATION_NUMERIC_FIELDS = {
    "validation_action_loss",
    "validation_action_mae",
    "validation_phase_loss",
    "validation_contact_loss",
    "validation_contact_count",
    "validation_clean_order_loss",
    "validation_overlap_loss",
    "validation_safety_loss",
    "validation_safety_tube_loss",
    "validation_safety_tube_mean_loss",
    "validation_safety_tube_tail_loss",
    "validation_safety_tube_max_ratio",
    "validation_safety_tube_min_slack",
    "validation_joint_violation_values",
    "validation_rate_violation_values",
    "validation_safety_violation_values",
    "validation_joint_max_excess_ratio",
    "validation_rate_max_ratio",
    "validation_examples",
    "validation_episode_count",
    "validation_objective",
}
_VALIDATION_FIELDS = _VALIDATION_STRING_FIELDS | _VALIDATION_NUMERIC_FIELDS
_LOADER_LOCK = threading.Lock()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(raw_temporary)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validation_is_qualified(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics["validation_joint_violation_values"]) == 0
        and float(metrics["validation_rate_violation_values"]) == 0
        and float(metrics["validation_safety_violation_values"]) == 0
        and float(metrics["validation_joint_max_excess_ratio"]) == 0
        and float(metrics["validation_rate_max_ratio"]) <= 1
    )


def validate_unqualified_final_run(
    run: str | pathlib.Path,
    *,
    step: int,
    train_job_id: str,
) -> dict[str, Any]:
    """Validate and describe the sole checkpoint eligible for diagnostic rollout."""

    run = pathlib.Path(run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"missing V3 run {run}")
    if not train_job_id.isdigit() or int(train_job_id) <= 0:
        raise ValueError("train_job_id must be one positive numeric Slurm job ID")

    required_run_files = (
        "run_config.json",
        "artifact.json",
        "normalization.json",
        "STATS_DONE",
        "VALIDATION_SAMPLES.json",
        "TRAIN_SOURCE_PROVENANCE.json",
        "FAILED",
    )
    missing_run = [name for name in required_run_files if not (run / name).is_file()]
    if missing_run:
        raise FileNotFoundError(f"diagnostic run lacks required artifacts {missing_run}")
    forbidden = [name for name in ("DONE", "BEST_VALIDATION.json") if (run / name).exists()]
    forbidden.extend(path.name for path in (run / "checkpoints").glob("best_validation_*") if path.exists())
    if forbidden:
        raise ValueError(f"diagnostic-unqualified run unexpectedly has formal completion artifacts {forbidden}")
    failure = (run / "FAILED").read_text().strip()
    if failure != EXPECTED_TRAINING_FAILURE:
        raise ValueError(f"training failure is not the expected qualification-only failure: {failure!r}")

    wrapper_failure_path = run / f"FAILED.train.{train_job_id}.json"
    wrapper_failure = _read_json(wrapper_failure_path)
    if (
        wrapper_failure.get("status") != "FAILED"
        or str(wrapper_failure.get("job_id")) != train_job_id
        or isinstance(wrapper_failure.get("exit_code"), bool)
        or not isinstance(wrapper_failure.get("exit_code"), int)
        or int(wrapper_failure["exit_code"]) == 0
    ):
        raise ValueError(f"invalid training wrapper failure marker {wrapper_failure_path}")

    config_path = run / "run_config.json"
    artifact_path = run / "artifact.json"
    normalization_path = run / "normalization.json"
    config_text = config_path.read_text()
    config = ControlV2TrainConfig.from_json(config_text)
    artifact = ControlV2Artifact.from_json(artifact_path.read_text())
    artifact.require_production_data()
    if isinstance(step, bool) or not isinstance(step, int) or step != config.num_train_steps:
        raise ValueError(f"diagnostic step {step!r} must equal configured final step {config.num_train_steps}")
    if config.backbone_train_mode != "frozen":
        raise ValueError(
            f"this diagnostic path is restricted to the current frozen head, got {config.backbone_train_mode}"
        )
    latest_path = run / "checkpoints" / "latest"
    try:
        latest = int(latest_path.read_text().strip())
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("checkpoints/latest must contain the final integer step") from error
    if latest != step:
        raise ValueError(f"latest checkpoint {latest} does not equal diagnostic final step {step}")

    checkpoint = run / "checkpoints" / str(step)
    missing_checkpoint = [
        name
        for name in REQUIRED_CHECKPOINT_FILES
        if not (checkpoint / name).is_file() or (checkpoint / name).is_symlink()
    ]
    if missing_checkpoint:
        raise FileNotFoundError(f"final checkpoint {checkpoint} lacks direct regular files {missing_checkpoint}")
    if json.loads((checkpoint / "train_config.json").read_text()) != json.loads(config_text):
        raise ValueError("final checkpoint train_config.json differs from run_config.json")

    metadata = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
    if not isinstance(metadata, Mapping):
        raise TypeError("final checkpoint metadata must be a mapping")
    expected_metadata = {
        "global_step": step,
        "control_v2_checkpoint_schema": CONTROL_V2_ARTIFACT_SCHEMA,
        "backbone_train_mode": "frozen",
        "backbone_last_n_blocks": config.backbone_last_n_blocks,
        "source_backbone_run": str(pathlib.Path(config.pretrained_run).resolve()),
        "source_backbone_step": config.pretrained_step,
        "artifact_sha256": _sha256(artifact_path),
        "normalization_sha256": _sha256(normalization_path),
        "source_store_sha256": artifact.source_store_sha256,
        "world_size": 4,
        "optimizer_step_min": step,
        "optimizer_step_max": step,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "local_batch_size": config.local_batch_size,
    }
    mismatches = {
        name: (metadata.get(name), expected)
        for name, expected in expected_metadata.items()
        if metadata.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"final checkpoint metadata differs from the production contract: {mismatches}")
    if metadata.get("best_validation_step") is not None or metadata.get("best_validation_chunk_loss") is not None:
        raise ValueError("diagnostic-unqualified metadata must not name a qualified best checkpoint")

    raw_validation = metadata.get("last_validation_metrics")
    if not isinstance(raw_validation, Mapping) or set(raw_validation) != _VALIDATION_FIELDS:
        actual = sorted(raw_validation) if isinstance(raw_validation, Mapping) else type(raw_validation).__name__
        raise ValueError(f"final validation metrics have an invalid field set: {actual}")
    validation = dict(raw_validation)
    for name in _VALIDATION_NUMERIC_FIELDS:
        value = validation[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"final validation metric {name} must be finite numeric data")
        if float(value) < 0:
            raise ValueError(f"final validation metric {name} must be non-negative")

    # Reuse the production manifest parser without changing its strict deployment loader.
    from openpi.mot_jepa import control_v2_deploy as deploy  # noqa: PLC0415

    manifest = deploy._load_validation_manifest(run)  # noqa: SLF001
    expected_validation = {
        "validation_sampling": manifest["sampling"],
        "validation_sample_sha256": manifest["sample_sha256"],
        "validation_examples": float(manifest["sample_count"]),
        "validation_episode_count": float(manifest["episode_count"]),
    }
    validation_mismatches = {
        name: (validation.get(name), expected)
        for name, expected in expected_validation.items()
        if validation.get(name) != expected
    }
    if validation_mismatches:
        raise ValueError(f"final validation metrics differ from the frozen panel: {validation_mismatches}")
    if _validation_is_qualified(validation):
        raise ValueError("final validation is qualified; use the formal deployment path instead")

    checkpoint_files = {
        name: {
            "path": str((checkpoint / name).resolve()),
            "size": (checkpoint / name).stat().st_size,
            "sha256": _sha256(checkpoint / name),
        }
        for name in REQUIRED_CHECKPOINT_FILES
    }
    return {
        "schema_version": 1,
        "status": DIAGNOSTIC_STATUS,
        "warning": "NOT_OFFICIAL: held-out safety qualification failed and was explicitly bypassed for rollout only",
        "run": str(run),
        "train_job_id": train_job_id,
        "step": step,
        "checkpoint": str(checkpoint.resolve()),
        "selection_metric": DIAGNOSTIC_SELECTION_METRIC,
        "selection_value": float(validation["validation_action_loss"]),
        "training_failure": failure,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": _sha256(artifact_path),
        "normalization_sha256": _sha256(normalization_path),
        "run_config_sha256": _sha256(config_path),
        "validation_manifest_sha256": _sha256(run / "VALIDATION_SAMPLES.json"),
        "train_source_provenance_sha256": _sha256(run / "TRAIN_SOURCE_PROVENANCE.json"),
        "checkpoint_files": checkpoint_files,
        "final_validation_metrics": validation,
    }


def load_unqualified_final_control_v2_artifacts(
    run: str | pathlib.Path,
    device: str | torch.device,
    *,
    step: int,
    train_job_id: str,
    verify_source_stores: bool = True,
) -> TrainedControlV2Artifacts:
    """Restore an exact final numeric checkpoint through an isolated diagnostic-only override."""

    contract = validate_unqualified_final_run(run, step=step, train_job_id=train_job_id)
    run_path = pathlib.Path(contract["run"])
    checkpoint = pathlib.Path(contract["checkpoint"])

    from openpi.mot_jepa import control_v2_deploy as deploy  # noqa: PLC0415

    original_read_completed = deploy._read_completed_step  # noqa: SLF001
    original_select = deploy._selected_validation_checkpoint  # noqa: SLF001
    original_require = deploy._require_metadata  # noqa: SLF001

    def diagnostic_completed(candidate: pathlib.Path, config: ControlV2TrainConfig) -> int:
        if pathlib.Path(candidate).resolve() != run_path or config.num_train_steps != step:
            raise ValueError("diagnostic loader run/config changed after validation")
        return step

    def diagnostic_selection(
        candidate: pathlib.Path,
        config: ControlV2TrainConfig,
        *,
        completed_step: int,
        requested_step: int | None,
        artifact_sha256: str,
        normalization_sha256: str,
        source_store_sha256: str,
    ) -> tuple[int, pathlib.Path, dict[str, Any]]:
        if (
            pathlib.Path(candidate).resolve() != run_path
            or config.num_train_steps != step
            or completed_step != step
            or requested_step != step
        ):
            raise ValueError("diagnostic loader requires the explicit exact final step")
        expected_digests = (
            contract["artifact_sha256"],
            contract["normalization_sha256"],
            ControlV2Artifact.from_json((run_path / "artifact.json").read_text()).source_store_sha256,
        )
        if (artifact_sha256, normalization_sha256, source_store_sha256) != expected_digests:
            raise ValueError("diagnostic loader artifact digests changed after validation")
        return (
            step,
            checkpoint,
            {
                "selection_metric": DIAGNOSTIC_SELECTION_METRIC,
                "validation_action_loss": contract["selection_value"],
            },
        )

    def diagnostic_require(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
        expected = dict(expected)
        if "best_validation_step" in expected or "best_validation_chunk_loss" in expected:
            if (
                metadata.get("best_validation_step") is not None
                or metadata.get("best_validation_chunk_loss") is not None
            ):
                raise ValueError("diagnostic checkpoint unexpectedly contains a qualified selection")
            expected.pop("best_validation_step", None)
            expected.pop("best_validation_chunk_loss", None)
        original_require(metadata, expected)

    # The production function remains unchanged.  A process-local, single-threaded patch only
    # replaces its two qualification selectors while all artifact, normalizer, source-store,
    # checkpoint, and strict state-dict checks continue to run.
    with _LOADER_LOCK:
        deploy._read_completed_step = diagnostic_completed  # type: ignore[attr-defined]  # noqa: SLF001
        deploy._selected_validation_checkpoint = diagnostic_selection  # type: ignore[attr-defined]  # noqa: SLF001
        deploy._require_metadata = diagnostic_require  # type: ignore[attr-defined]  # noqa: SLF001
        try:
            loaded = deploy.load_trained_control_v2_artifacts(
                run_path,
                device,
                step=step,
                verify_source_stores=verify_source_stores,
            )
        finally:
            deploy._read_completed_step = original_read_completed  # type: ignore[attr-defined]  # noqa: SLF001
            deploy._selected_validation_checkpoint = original_select  # type: ignore[attr-defined]  # noqa: SLF001
            deploy._require_metadata = original_require  # type: ignore[attr-defined]  # noqa: SLF001
    if loaded.selection_metric != DIAGNOSTIC_SELECTION_METRIC or loaded.checkpoint.resolve() != checkpoint:
        raise RuntimeError("diagnostic loader returned the wrong checkpoint identity")
    return loaded


def _qualify_diagnostic_traces(
    run_root: pathlib.Path,
    entries: Mapping[int, Mapping[str, Any]],
    *,
    limits: SafetyLimits,
    expected_policy_step: int,
    expected_selection_value: float,
    expected_artifact_sha256: str,
    first_gripper_jump_max: float = 0.005,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    root = pathlib.Path(run_root)
    trace_paths = sorted((root / "trajectory").glob("worker_*/*.npz"))
    by_seed: dict[int, pathlib.Path] = {}
    for path in trace_paths:
        try:
            seed = int(path.stem)
        except ValueError as error:
            raise ValueError(f"trajectory filename is not a seed: {path}") from error
        if seed in by_seed:
            raise ValueError(f"duplicate trajectory for seed {seed}")
        by_seed[seed] = path
    if set(by_seed) != set(entries):
        raise ValueError(
            f"student trajectory seeds differ: missing={sorted(set(entries) - set(by_seed))}, "
            f"extra={sorted(set(by_seed) - set(entries))}"
        )

    reports: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    for seed, entry in sorted(entries.items()):
        path = by_seed[seed]
        with np.load(path, allow_pickle=False) as trace:
            count = int(trace["num_actions"])
            if int(trace["seed"]) != seed:
                raise ValueError(f"trajectory path/payload seed mismatch for {path}")
            expected_result = "success" if entry["result"] == "success" else "failed"
            if str(trace["result"]) != expected_result:
                raise ValueError(f"trajectory/metadata result mismatch for seed {seed}")
            if count <= 0 or bool(trace["truncated"]):
                raise ValueError(f"seed {seed} trace is empty or truncated")
            first_indices = np.asarray(trace["first_executable_index"])
            if first_indices.shape != (count,) or not np.all(first_indices == 1):
                raise ValueError(f"seed {seed} does not consistently execute chunk index 1")
            expected_ints = {
                "control_v2_trace_schema": 1,
                "control_v2_policy_step": expected_policy_step,
                "control_v2_completed_step": expected_policy_step,
            }
            for name, expected in expected_ints.items():
                if name not in trace or int(np.asarray(trace[name]).item()) != expected:
                    raise ValueError(f"seed {seed} trace {name} does not equal {expected}")
            metric = str(np.asarray(trace["control_v2_selection_metric"]).item())
            if metric != DIAGNOSTIC_SELECTION_METRIC:
                raise ValueError(f"seed {seed} trace lacks the diagnostic-only checkpoint identity")
            value = float(np.asarray(trace["control_v2_selection_value"]).item())
            if not np.isclose(value, expected_selection_value, rtol=1e-7, atol=0.0):
                raise ValueError(f"seed {seed} trace has the wrong final validation value")
            if str(np.asarray(trace["control_v2_artifact_sha256"]).item()) != expected_artifact_sha256:
                raise ValueError(f"seed {seed} trace has the wrong artifact digest")
            before = np.asarray(trace["qpos8_before"], dtype=np.float32)
            sent = np.asarray(trace["sent_action8"], dtype=np.float32)
            clamp_count = acceptance._load_trace_counter(trace, entry)  # noqa: SLF001
            nonfinite_fallbacks = acceptance._load_nonfinite_fallbacks(trace, entry)  # noqa: SLF001
            trace_sha256 = _sha256(path)
            trace_meta = entry.get("trajectory")
            if isinstance(trace_meta, Mapping) and trace_meta.get("sha256") not in {None, trace_sha256}:
                raise ValueError(f"seed {seed} trajectory digest differs from worker metadata")

        qualification = acceptance.qualify_control_trace(
            before,
            sent,
            limits,
            clamp_count=clamp_count,
            first_gripper_jump_max=first_gripper_jump_max,
        )
        report = dataclasses.asdict(qualification)
        report.update(
            clamp_count=clamp_count,
            nonfinite_fallbacks=nonfinite_fallbacks,
            path=str(path),
            sha256=trace_sha256,
        )
        if nonfinite_fallbacks:
            report["passed"] = False
        reports[seed] = report
        if not report["passed"]:
            failures.append(f"seed {seed}: {report}")
    return reports, failures


def evaluate_diagnostic(
    *,
    run: pathlib.Path,
    train_job_id: str,
    step: int,
    student_root: pathlib.Path,
    qualified_runtime_path: pathlib.Path,
    start_seed: int,
    trials: int,
    contract_path: pathlib.Path,
) -> dict[str, Any]:
    if (trials, start_seed) not in {(1, 900_000), (20, 900_100)}:
        raise ValueError("diagnostic evaluation is fixed to smoke seed 900000 or n=20 seeds 900100..900119")
    contract = validate_unqualified_final_run(run, step=step, train_job_id=train_job_id)
    frozen_contract = _read_json(contract_path)
    if frozen_contract != contract:
        raise ValueError("diagnostic checkpoint contract changed after rollout")
    artifact_path = pathlib.Path(contract["artifact"])
    artifact = ControlV2Artifact.from_json(artifact_path.read_text())
    artifact.require_production_data()
    qualified_runtime = acceptance.load_qualified_runtime(qualified_runtime_path, require_official=False)
    results, entries = acceptance.load_episode_results(
        student_root,
        start_seed=start_seed,
        trials=trials,
    )
    reports, trace_failures = _qualify_diagnostic_traces(
        student_root,
        entries,
        limits=SafetyLimits(
            joint_lower=artifact.joint_lower,
            joint_upper=artifact.joint_upper,
            max_delta=artifact.max_command_delta,
        ),
        expected_policy_step=step,
        expected_selection_value=float(contract["selection_value"]),
        expected_artifact_sha256=contract["artifact_sha256"],
    )
    successes = sum(results.values())
    return {
        "schema_version": 1,
        "status": "DIAGNOSTIC_COMPLETE",
        "diagnostic_only": True,
        "official_result": False,
        "validation_gate_bypassed": True,
        "warning": contract["warning"],
        "trials": trials,
        "successes": successes,
        "failures": trials - successes,
        "success_rate": successes / trials,
        "start_seed": start_seed,
        "seed_results": {str(seed): results[seed] for seed in sorted(results)},
        "trace_qualified": len(reports) - len(trace_failures),
        "trace_total": len(reports),
        "trace_failures": trace_failures,
        "traces": {str(seed): report for seed, report in reports.items()},
        "checkpoint_contract": contract,
        "checkpoint_contract_sha256": _sha256(contract_path),
        "student_root": str(student_root.resolve()),
        "student_metadata": acceptance.result_metadata_manifest(student_root),
        "qualified_runtime": str(qualified_runtime_path.resolve()),
        "qualified_runtime_sha256": _sha256(qualified_runtime_path),
        "runtime_qualification": qualified_runtime["qualification"],
        "runtime_qualification_sha256": qualified_runtime["qualification_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--run", type=pathlib.Path, required=True)
    validate_parser.add_argument("--step", type=int, required=True)
    validate_parser.add_argument("--train-job-id", required=True)
    validate_parser.add_argument("--output", type=pathlib.Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--run", type=pathlib.Path, required=True)
    aggregate_parser.add_argument("--step", type=int, required=True)
    aggregate_parser.add_argument("--train-job-id", required=True)
    aggregate_parser.add_argument("--student-root", type=pathlib.Path, required=True)
    aggregate_parser.add_argument("--qualified-runtime", type=pathlib.Path, required=True)
    aggregate_parser.add_argument("--start-seed", type=int, required=True)
    aggregate_parser.add_argument("--trials", type=int, required=True)
    aggregate_parser.add_argument("--contract", type=pathlib.Path, required=True)
    aggregate_parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            payload = validate_unqualified_final_run(args.run, step=args.step, train_job_id=args.train_job_id)
        else:
            payload = evaluate_diagnostic(
                run=args.run,
                train_job_id=args.train_job_id,
                step=args.step,
                student_root=args.student_root,
                qualified_runtime_path=args.qualified_runtime,
                start_seed=args.start_seed,
                trials=args.trials,
                contract_path=args.contract,
            )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "status": "INVALID",
            "diagnostic_only": True,
            "official_result": False,
            "error": f"{type(error).__name__}: {error}",
        }
        _atomic_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
