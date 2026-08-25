#!/usr/bin/env python
"""Fail-closed acceptance gate for the paired MoT-Control V2 n=100 run.

The two evaluators must use exactly the same 100 seeds.  Success is necessary but not sufficient:
every student trajectory must also satisfy the pre-registered command-trace invariants.  This
script intentionally consumes worker-level records rather than trusting only aggregate rates.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

import numpy as np

from openpi.mot_jepa.control_v2_config import CONTROL_V2_ARTIFACT_SCHEMA
from openpi.mot_jepa.control_v2_config import ControlV2Artifact
from openpi.mot_jepa.control_v2_runtime import SafetyLimits
from openpi.mot_jepa.control_v2_runtime import paired_n100_acceptance
from openpi.mot_jepa.control_v2_runtime import qualify_control_trace

TRIALS = 100
VALID_RESULTS = {"success", "failed"}
VALIDATION_SAMPLING = "all_episode_time_phase_contact_cold_start_stratified_v2"
VALIDATION_TIME_BINS = ("early", "middle", "late")
VALIDATION_HISTORY_STEPS = 16
VALIDATION_PREFIX_SCENARIOS = {
    (length, oldest_present) for length in range(1, VALIDATION_HISTORY_STEPS + 1) for oldest_present in (0, 1)
}
SELECTION_CONSTRAINT = "validation_safety_violation_values==0"
COMMON_RUNTIME_QUALIFICATION = "univtac_common_runtime_v1"
OFFICIAL_RUNTIME_QUALIFICATION = "univtac_ftp1_official_v1"
BEST_VALIDATION_FIELDS = {
    "schema_version",
    "selection_metric",
    "selection_constraint",
    "selection_mode",
    "tie_break",
    "step",
    "checkpoint",
    "artifact_sha256",
    "normalization_sha256",
    "source_store_sha256",
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
    "validation_sampling",
    "validation_sample_sha256",
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def result_metadata_manifest(run_root: pathlib.Path) -> dict[str, Any]:
    """Hash the exact aggregate and per-worker records consumed by acceptance."""

    root = pathlib.Path(run_root).resolve()
    paths = [root / "metadata.json", *sorted((root / "metadata").glob("worker_*.json"))]
    if len(paths) < 2 or any(not path.is_file() for path in paths):
        raise ValueError(f"result metadata is incomplete under {root}")
    files = {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in paths
    }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "content_sha256": hashlib.sha256(canonical).hexdigest()}


def load_qualified_runtime(path: pathlib.Path, *, require_official: bool) -> dict[str, Any]:
    """Load a self-authenticating runtime qualification produced by the byte verifier."""
    path = pathlib.Path(path)
    payload = _read_json(path)
    expected_fields = {"schema_version", "qualification", "common", "qualification_sha256"}
    expected_qualification = COMMON_RUNTIME_QUALIFICATION
    if require_official:
        expected_fields.add("official")
        expected_qualification = OFFICIAL_RUNTIME_QUALIFICATION
    if set(payload) != expected_fields:
        raise ValueError(f"{path} has an invalid qualified-runtime field set")
    if payload["schema_version"] != 1 or payload["qualification"] != expected_qualification:
        raise ValueError(f"{path} is not the required {expected_qualification} qualification")
    common = payload["common"]
    if not isinstance(common, dict) or set(common) != {"container", "external_assets", "local_assets"}:
        raise ValueError(f"{path} has an invalid common-runtime qualification")
    if require_official:
        official = payload["official"]
        if not isinstance(official, dict) or set(official) != {"checkpoint", "overlay", "ready", "tokenizer"}:
            raise ValueError(f"{path} has an invalid official-runtime qualification")
    claimed = payload["qualification_sha256"]
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{path} has an invalid qualification digest")
    unsigned = {key: value for key, value in payload.items() if key != "qualification_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if claimed != actual:
        raise ValueError(f"{path} qualified-runtime digest does not close")
    return payload


def _load_validation_manifest(run: pathlib.Path) -> dict[str, Any]:
    manifest = _read_json(run / "VALIDATION_SAMPLES.json")
    expected_fields = {
        "schema_version",
        "sampling",
        "requested_examples",
        "sample_count",
        "episode_count",
        "sample_sha256",
        "episodes",
        "samples",
    }
    if set(manifest) != expected_fields:
        raise ValueError("VALIDATION_SAMPLES.json has an invalid field set")
    if manifest["schema_version"] != 1 or manifest["sampling"] != VALIDATION_SAMPLING:
        raise ValueError("VALIDATION_SAMPLES.json has an unsupported schema or sampling rule")
    samples, episodes = manifest["samples"], manifest["episodes"]
    if not isinstance(samples, list) or not isinstance(episodes, list) or not samples or not episodes:
        raise ValueError("VALIDATION_SAMPLES.json must contain non-empty sample and episode lists")
    if any(not isinstance(item, dict) or set(item) != {"dataset_index", "identity"} for item in samples):
        raise ValueError("VALIDATION_SAMPLES.json has malformed sample identities")
    if any(not isinstance(item, dict) or set(item) != {"identity", "sample_count"} for item in episodes):
        raise ValueError("VALIDATION_SAMPLES.json has malformed episode counts")
    for name in ("requested_examples", "sample_count", "episode_count"):
        if isinstance(manifest[name], bool) or not isinstance(manifest[name], int) or manifest[name] <= 0:
            raise ValueError(f"VALIDATION_SAMPLES.json {name} must be a positive integer")
    identities = [item["identity"] for item in samples]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("VALIDATION_SAMPLES.json sample identities must be non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("VALIDATION_SAMPLES.json contains duplicate sample identities")
    dataset_indices = [item["dataset_index"] for item in samples]
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in dataset_indices):
        raise ValueError("VALIDATION_SAMPLES.json dataset indices must be non-negative integers")
    if len(set(dataset_indices)) != len(dataset_indices):
        raise ValueError("VALIDATION_SAMPLES.json contains duplicate dataset indices")

    observed_episode_counts: dict[str, int] = {}
    prefix_scenarios: set[tuple[int, int]] = set()
    observed_time_bins: set[str] = set()
    observed_phases: set[int] = set()
    observed_contacts: set[int] = set()
    for item in samples:
        parts = item["identity"].split("\0")
        expected_names = (
            "episode",
            "start",
            "stride",
            "dataset_index",
            "time_bin",
            "phase",
            "contact",
            "history_length",
            "oldest_command_present",
        )
        if len(parts) != len(expected_names) + 1 or not pathlib.Path(parts[0]).is_absolute():
            raise ValueError("VALIDATION_SAMPLES.json has a malformed structured sample identity")
        fields: dict[str, str] = {}
        for part, expected_name in zip(parts[1:], expected_names, strict=True):
            name, separator, value = part.partition("=")
            if separator != "=" or name != expected_name or not value:
                raise ValueError("VALIDATION_SAMPLES.json has a malformed structured sample identity")
            fields[name] = value
        try:
            episode_idx = int(fields["episode"])
            start = int(fields["start"])
            stride = int(fields["stride"])
            embedded_dataset_index = int(fields["dataset_index"])
            history_length = int(fields["history_length"])
            oldest_present = int(fields["oldest_command_present"])
        except ValueError as error:
            raise ValueError("VALIDATION_SAMPLES.json structured identity has non-integer fields") from error
        if episode_idx < 0 or start < 0 or stride < 1 or not 1 <= history_length <= VALIDATION_HISTORY_STEPS:
            raise ValueError("VALIDATION_SAMPLES.json structured identity has out-of-range indices")
        if embedded_dataset_index != item["dataset_index"]:
            raise ValueError("VALIDATION_SAMPLES.json dataset index differs from its immutable identity")
        if fields["time_bin"] not in VALIDATION_TIME_BINS:
            raise ValueError("VALIDATION_SAMPLES.json has an invalid episode-time bin")
        if fields["phase"] not in {"0", "1", "2", "3"}:
            raise ValueError("VALIDATION_SAMPLES.json requires authoritative phase strata")
        if fields["contact"] not in {"0", "1"}:
            raise ValueError("VALIDATION_SAMPLES.json requires authoritative contact strata")
        if oldest_present not in {0, 1}:
            raise ValueError("VALIDATION_SAMPLES.json has an invalid cold-start parity")
        episode_identity = f"{parts[0]}\0episode={episode_idx}"
        observed_episode_counts[episode_identity] = observed_episode_counts.get(episode_identity, 0) + 1
        prefix_scenarios.add((history_length, oldest_present))
        observed_time_bins.add(fields["time_bin"])
        observed_phases.add(int(fields["phase"]))
        observed_contacts.add(int(fields["contact"]))

    declared_episode_counts: dict[str, int] = {}
    for item in episodes:
        identity, count = item["identity"], item["sample_count"]
        if not isinstance(identity, str) or not identity or identity in declared_episode_counts:
            raise ValueError("VALIDATION_SAMPLES.json episode identities must be unique non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("VALIDATION_SAMPLES.json episode sample counts must be positive integers")
        declared_episode_counts[identity] = count
    if declared_episode_counts != observed_episode_counts:
        raise ValueError("VALIDATION_SAMPLES.json episode counts differ from its sample identities")
    if len(samples) < len(VALIDATION_PREFIX_SCENARIOS) or not prefix_scenarios >= VALIDATION_PREFIX_SCENARIOS:
        raise ValueError("VALIDATION_SAMPLES.json does not cover the fixed V3 cold-start scenarios")
    required_strata = {
        "episode-time": (set(VALIDATION_TIME_BINS), observed_time_bins),
        "phase": (set(range(4)), observed_phases),
        "contact": (set(range(2)), observed_contacts),
    }
    missing_strata = {
        name: sorted(required - observed)
        for name, (required, observed) in required_strata.items()
        if required - observed
    }
    if missing_strata:
        raise ValueError(f"VALIDATION_SAMPLES.json does not cover required validation strata: {missing_strata}")

    digest = hashlib.sha256(json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if (
        manifest["sample_count"] != len(samples)
        or manifest["episode_count"] != len(episodes)
        or manifest["sample_sha256"] != digest
        or len(samples) < len(episodes)
    ):
        raise ValueError("VALIDATION_SAMPLES.json identity/count closure is invalid")
    return manifest


def load_episode_results(
    run_root: pathlib.Path,
    *,
    start_seed: int,
    trials: int = TRIALS,
) -> tuple[dict[int, bool], dict[int, dict]]:
    """Load exactly one worker record for each seed in the requested interval."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    root = pathlib.Path(run_root)
    summary = _read_json(root / "metadata.json")
    worker_paths = sorted((root / "metadata").glob("worker_*.json"))
    if not worker_paths:
        raise ValueError(f"no worker metadata found under {root / 'metadata'}")

    entries: dict[int, dict] = {}
    for path in worker_paths:
        for raw_seed, raw_entry in _read_json(path).items():
            try:
                seed = int(raw_seed)
            except ValueError as exc:
                raise ValueError(f"invalid seed key {raw_seed!r} in {path}") from exc
            if seed in entries:
                raise ValueError(f"duplicate seed {seed} across worker metadata")
            if not isinstance(raw_entry, dict):
                raise ValueError(f"seed {seed} entry in {path} is not an object")
            result = raw_entry.get("result")
            if result not in VALID_RESULTS:
                raise ValueError(f"seed {seed} has incomplete result {result!r}")
            entries[seed] = raw_entry

    expected = set(range(start_seed, start_seed + trials))
    actual = set(entries)
    if actual != expected:
        raise ValueError(
            f"seed set differs for {root}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    successes = {seed: entries[seed]["result"] == "success" for seed in sorted(entries)}
    success_count = sum(successes.values())
    if (
        int(summary.get("total_episodes", -1)) != trials
        or int(summary.get("success", -1)) != success_count
        or int(summary.get("failed", -1)) != trials - success_count
        or int(summary.get("error", -1)) != 0
    ):
        raise ValueError(f"aggregate metadata disagrees with the {trials} worker records in {root}")
    eval_args = summary.get("eval_args")
    if not isinstance(eval_args, dict):
        raise ValueError(f"{root / 'metadata.json'} lacks eval_args")
    if int(eval_args.get("start_seed", -1)) != start_seed or int(eval_args.get("total_num", -1)) != trials:
        raise ValueError("recorded evaluator seed interval is not the requested interval")
    return successes, entries


def load_deployment_contract(artifact_path: pathlib.Path) -> tuple[ControlV2Artifact, dict[str, Any], pathlib.Path]:
    artifact_path = pathlib.Path(artifact_path)
    artifact = ControlV2Artifact.from_json(artifact_path.read_text())
    artifact.require_production_data()
    normalization_path = artifact_path.parent / "normalization.json"
    if not normalization_path.is_file():
        raise FileNotFoundError(f"missing deployment normalization artifact {normalization_path}")
    selection_path = artifact_path.parent / "BEST_VALIDATION.json"
    selection = _read_json(selection_path)
    if set(selection) != BEST_VALIDATION_FIELDS:
        raise ValueError("BEST_VALIDATION.json has an invalid field set")
    manifest = _load_validation_manifest(artifact_path.parent)
    if selection.get("selection_metric") != "validation_action_loss":
        raise ValueError("BEST_VALIDATION.json does not select validation_action_loss")
    expected_strings = {
        "selection_constraint": SELECTION_CONSTRAINT,
        "selection_mode": "min",
        "tie_break": "earliest_step",
        "validation_sampling": manifest["sampling"],
        "validation_sample_sha256": manifest["sample_sha256"],
    }
    for name, expected in expected_strings.items():
        value = selection.get(name)
        if not isinstance(value, str) or value != expected:
            raise ValueError(f"BEST_VALIDATION.json {name} does not equal {expected!r}")
    expected_digests = {
        "artifact_sha256": _sha256(artifact_path),
        "normalization_sha256": _sha256(normalization_path),
        "source_store_sha256": artifact.source_store_sha256,
    }
    for name, expected in expected_digests.items():
        value = selection.get(name)
        if not isinstance(value, str) or value != expected:
            raise ValueError(f"BEST_VALIDATION.json {name} does not close the deployed artifacts")
    numeric_fields = BEST_VALIDATION_FIELDS - {
        "selection_metric",
        "selection_constraint",
        "selection_mode",
        "tie_break",
        "checkpoint",
        "artifact_sha256",
        "normalization_sha256",
        "source_store_sha256",
        "validation_sampling",
        "validation_sample_sha256",
    }
    for name in numeric_fields:
        value = selection[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"BEST_VALIDATION.json {name} must be finite numeric data")
        if value < 0:
            raise ValueError(f"BEST_VALIDATION.json {name} must be non-negative")
    if selection["schema_version"] != CONTROL_V2_ARTIFACT_SCHEMA:
        raise ValueError("BEST_VALIDATION.json has an unsupported schema")
    if selection["validation_examples"] != manifest["sample_count"]:
        raise ValueError("BEST_VALIDATION.json validation_examples differs from its sample manifest")
    if selection["validation_episode_count"] != manifest["episode_count"]:
        raise ValueError("BEST_VALIDATION.json validation_episode_count differs from its sample manifest")
    safety_counts = {
        name: selection[name]
        for name in (
            "validation_joint_violation_values",
            "validation_rate_violation_values",
            "validation_safety_violation_values",
        )
    }
    if any(value != 0 for value in safety_counts.values()):
        raise ValueError(f"BEST_VALIDATION.json violates its zero held-out safety constraint: {safety_counts}")
    if selection["validation_joint_max_excess_ratio"] != 0:
        raise ValueError("BEST_VALIDATION.json validation_joint_max_excess_ratio must be zero")
    if selection["validation_rate_max_ratio"] > 1:
        raise ValueError("BEST_VALIDATION.json validation_rate_max_ratio must be at most 1")
    selected_step = int(selection.get("step", 0))
    selected_loss = float(selection.get("validation_action_loss", float("nan")))
    selected_checkpoint = pathlib.Path(str(selection.get("checkpoint", "")))
    if selected_step < 0 or not np.isfinite(selected_loss) or not selected_checkpoint.is_dir():
        raise ValueError("BEST_VALIDATION.json does not identify a finite, existing selected checkpoint")
    expected_checkpoint = (artifact_path.parent / "checkpoints" / f"best_validation_{selected_step}").resolve()
    if selected_checkpoint.resolve() != expected_checkpoint:
        raise ValueError(f"selected checkpoint {selected_checkpoint} differs from {expected_checkpoint}")
    return artifact, selection, selection_path


def _metadata_counter(entry: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    candidates: list[Mapping[str, Any]] = [entry]
    for container in ("safety", "safety_counters", "trajectory"):
        value = entry.get(container)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        for name in names:
            if name in candidate:
                return int(candidate[name])
    return None


def _load_trace_counter(trace: Mapping[str, np.ndarray], entry: Mapping[str, Any]) -> int:
    for name in ("control_v2_clamp_count", "clamp_count", "clamp_commands", "safety_clamp_commands"):
        if name in trace:
            return int(np.asarray(trace[name]).item())
    counter = _metadata_counter(entry, ("clamp_count", "clamp_commands"))
    if counter is None:
        raise ValueError("missing per-episode clamp counter (refuse to infer zero from silence)")
    return counter


def _load_nonfinite_fallbacks(trace: Mapping[str, np.ndarray], entry: Mapping[str, Any]) -> int:
    for name in ("control_v2_nonfinite_fallbacks", "nonfinite_fallbacks"):
        if name in trace:
            return int(np.asarray(trace[name]).item())
    counter = _metadata_counter(entry, ("nonfinite_fallbacks",))
    if counter is None:
        raise ValueError("missing per-episode non-finite fallback counter")
    return counter


def qualify_student_traces(
    run_root: pathlib.Path,
    entries: Mapping[int, Mapping[str, Any]],
    *,
    limits: SafetyLimits,
    first_gripper_jump_max: float,
    expected_policy_step: int,
    expected_completed_step: int,
    expected_selection_value: float,
    expected_artifact_sha256: str,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    root = pathlib.Path(run_root)
    trace_paths = sorted((root / "trajectory").glob("worker_*/*.npz"))
    by_seed: dict[int, pathlib.Path] = {}
    for path in trace_paths:
        try:
            seed = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"trajectory filename is not a seed: {path}") from exc
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
            payload_seed = int(trace["seed"])
            result = str(trace["result"])
            count = int(trace["num_actions"])
            if payload_seed != seed:
                raise ValueError(f"trajectory path/payload seed mismatch for {path}")
            expected_result = "success" if entry["result"] == "success" else "failed"
            if result != expected_result:
                raise ValueError(f"trajectory/metadata result mismatch for seed {seed}")
            if count <= 0 or bool(trace["truncated"]):
                raise ValueError(f"seed {seed} trace is empty or truncated")
            first_indices = np.asarray(trace["first_executable_index"])
            if first_indices.shape != (count,) or not np.all(first_indices == 1):
                raise ValueError(f"seed {seed} does not consistently execute chunk index 1")
            provenance_expected = {
                "control_v2_trace_schema": 1,
                "control_v2_policy_step": expected_policy_step,
                "control_v2_completed_step": expected_completed_step,
            }
            for name, expected in provenance_expected.items():
                if name not in trace or int(np.asarray(trace[name]).item()) != expected:
                    raise ValueError(f"seed {seed} trace {name} does not equal {expected}")
            if str(np.asarray(trace["control_v2_selection_metric"]).item()) != "validation_action_loss":
                raise ValueError(f"seed {seed} trace has the wrong selection metric")
            selection_value = float(np.asarray(trace["control_v2_selection_value"]).item())
            if not np.isclose(selection_value, expected_selection_value, rtol=1e-7, atol=0.0):
                raise ValueError(f"seed {seed} trace has the wrong selected validation value")
            if str(np.asarray(trace["control_v2_artifact_sha256"]).item()) != expected_artifact_sha256:
                raise ValueError(f"seed {seed} trace has the wrong artifact digest")
            before = np.asarray(trace["qpos8_before"], dtype=np.float32)
            sent = np.asarray(trace["sent_action8"], dtype=np.float32)
            clamp_count = _load_trace_counter(trace, entry)
            nonfinite_fallbacks = _load_nonfinite_fallbacks(trace, entry)
            trace_sha256 = _sha256(path)
            trace_meta = entry.get("trajectory")
            if isinstance(trace_meta, Mapping) and trace_meta.get("sha256") not in {None, trace_sha256}:
                raise ValueError(f"seed {seed} trajectory digest differs from worker metadata")

        qualification = qualify_control_trace(
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


def evaluate_acceptance(
    official_root: pathlib.Path,
    student_root: pathlib.Path,
    artifact_path: pathlib.Path,
    qualified_runtime_path: pathlib.Path,
    *,
    start_seed: int,
    first_gripper_jump_max: float,
) -> dict[str, Any]:
    artifact_path = pathlib.Path(artifact_path)
    artifact, selection, selection_path = load_deployment_contract(artifact_path)
    completed_step = int((artifact_path.parent / "DONE").read_text().strip())
    artifact_sha256 = _sha256(artifact_path)
    qualified_runtime = load_qualified_runtime(qualified_runtime_path, require_official=True)
    limits = SafetyLimits(
        joint_lower=artifact.joint_lower,
        joint_upper=artifact.joint_upper,
        max_delta=artifact.max_command_delta,
    )
    official, _official_entries = load_episode_results(official_root, start_seed=start_seed)
    student, student_entries = load_episode_results(student_root, start_seed=start_seed)
    official_metadata = result_metadata_manifest(official_root)
    student_metadata = result_metadata_manifest(student_root)
    seeds = list(range(start_seed, start_seed + TRIALS))
    paired = paired_n100_acceptance(
        np.asarray([official[seed] for seed in seeds], dtype=np.bool_),
        np.asarray([student[seed] for seed in seeds], dtype=np.bool_),
    )
    traces, trace_failures = qualify_student_traces(
        student_root,
        student_entries,
        limits=limits,
        first_gripper_jump_max=first_gripper_jump_max,
        expected_policy_step=int(selection["step"]),
        expected_completed_step=completed_step,
        expected_selection_value=float(selection["validation_action_loss"]),
        expected_artifact_sha256=artifact_sha256,
    )
    accepted = paired.accepted and not trace_failures
    return {
        "schema_version": 1,
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECT",
        "criteria": {
            "trials": TRIALS,
            "student_success_min": 99,
            "official_only_discordance_upper_95_max": 0.05,
            "all_student_traces_qualified": True,
            "first_gripper_jump_max": first_gripper_jump_max,
        },
        "paired": dataclasses.asdict(paired),
        "paired_seed_results": {str(seed): {"official": official[seed], "student": student[seed]} for seed in seeds},
        "trace_qualified": len(traces) - len(trace_failures),
        "trace_total": len(traces),
        "trace_failures": trace_failures,
        "traces": {str(seed): report for seed, report in traces.items()},
        "official_root": str(pathlib.Path(official_root).resolve()),
        "student_root": str(pathlib.Path(student_root).resolve()),
        "official_metadata": official_metadata,
        "student_metadata": student_metadata,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "selection": selection,
        "selection_sha256": _sha256(selection_path),
        "qualified_runtime": str(pathlib.Path(qualified_runtime_path).resolve()),
        "qualified_runtime_sha256": _sha256(pathlib.Path(qualified_runtime_path)),
        "runtime_qualification": qualified_runtime["qualification"],
        "runtime_qualification_sha256": qualified_runtime["qualification_sha256"],
        "start_seed": start_seed,
    }


def evaluate_student_gate(
    student_root: pathlib.Path,
    artifact_path: pathlib.Path,
    qualified_runtime_path: pathlib.Path,
    *,
    start_seed: int,
    trials: int,
    minimum_successes: int,
    first_gripper_jump_max: float,
) -> dict[str, Any]:
    if not 0 <= minimum_successes <= trials:
        raise ValueError("minimum_successes must be in [0, trials]")
    artifact_path = pathlib.Path(artifact_path)
    artifact, selection, selection_path = load_deployment_contract(artifact_path)
    completed_step = int((artifact_path.parent / "DONE").read_text().strip())
    artifact_sha256 = _sha256(artifact_path)
    qualified_runtime = load_qualified_runtime(qualified_runtime_path, require_official=False)
    limits = SafetyLimits(
        joint_lower=artifact.joint_lower,
        joint_upper=artifact.joint_upper,
        max_delta=artifact.max_command_delta,
    )
    student, entries = load_episode_results(student_root, start_seed=start_seed, trials=trials)
    student_metadata = result_metadata_manifest(student_root)
    traces, trace_failures = qualify_student_traces(
        student_root,
        entries,
        limits=limits,
        first_gripper_jump_max=first_gripper_jump_max,
        expected_policy_step=int(selection["step"]),
        expected_completed_step=completed_step,
        expected_selection_value=float(selection["validation_action_loss"]),
        expected_artifact_sha256=artifact_sha256,
    )
    successes = sum(student.values())
    accepted = successes >= minimum_successes and not trace_failures
    return {
        "schema_version": 1,
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECT",
        "gate": "student_trace",
        "trials": trials,
        "successes": successes,
        "minimum_successes": minimum_successes,
        "success_rate": successes / trials,
        "trace_qualified": len(traces) - len(trace_failures),
        "trace_total": len(traces),
        "trace_failures": trace_failures,
        "traces": {str(seed): report for seed, report in traces.items()},
        "student_root": str(pathlib.Path(student_root).resolve()),
        "student_seed_results": {str(seed): student[seed] for seed in sorted(student)},
        "student_metadata": student_metadata,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "selection": selection,
        "selection_sha256": _sha256(selection_path),
        "qualified_runtime": str(pathlib.Path(qualified_runtime_path).resolve()),
        "qualified_runtime_sha256": _sha256(pathlib.Path(qualified_runtime_path)),
        "runtime_qualification": qualified_runtime["qualification"],
        "runtime_qualification_sha256": qualified_runtime["qualification_sha256"],
        "start_seed": start_seed,
    }


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = pathlib.Path(raw_tmp)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=pathlib.Path)
    parser.add_argument("--student-root", type=pathlib.Path, required=True)
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--qualified-runtime", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--start-seed", type=int, default=1_000_000)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--minimum-successes", type=int)
    parser.add_argument("--first-gripper-jump-max", type=float, default=0.005)
    args = parser.parse_args()
    try:
        if args.official_root is not None:
            if args.trials != TRIALS or args.minimum_successes not in {None, 99}:
                raise ValueError("paired official acceptance is fixed to trials=100 and minimum_successes=99")
            payload = evaluate_acceptance(
                args.official_root,
                args.student_root,
                args.artifact,
                args.qualified_runtime,
                start_seed=args.start_seed,
                first_gripper_jump_max=args.first_gripper_jump_max,
            )
        else:
            if args.minimum_successes is None:
                raise ValueError("student-only trace gate requires --minimum-successes")
            payload = evaluate_student_gate(
                args.student_root,
                args.artifact,
                args.qualified_runtime,
                start_seed=args.start_seed,
                trials=args.trials,
                minimum_successes=args.minimum_successes,
                first_gripper_jump_max=args.first_gripper_jump_max,
            )
    except Exception as exc:  # fail closed while still writing an auditable verdict
        payload = {
            "schema_version": 1,
            "accepted": False,
            "decision": "INVALID",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
