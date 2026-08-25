#!/usr/bin/env python
"""Validate five paired n=20 shards and synthesize the exact paired n=100 acceptance input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import stat
import tempfile
from typing import Any

from openpi.mot_jepa.control_v2_runtime import SafetyLimits
from scripts import mot_jepa_control_v2_acceptance as acceptance
from scripts import mot_jepa_control_v2_provenance as provenance
from scripts import mot_jepa_control_v2_qualified_runtime as runtime_qualifier

N100_START_SEED = 1_000_000
SHARD_TRIALS = 20
FIRST_GRIPPER_JUMP_MAX = 0.005
EVAL_NODE = "gpu-h100-0044"
SHARD_STARTS = tuple(range(N100_START_SEED, N100_START_SEED + 100, SHARD_TRIALS))
N100_SEEDS = frozenset(range(N100_START_SEED, N100_START_SEED + 100))
EXTERNAL_RUNTIME_ROOT_NAMES = ("container", "external_assets", "overlay", "checkpoint", "tokenizer")
REQUIRED_STATIC_PROFILE_ROOTS = (
    "scripts/eval_motjepa_control_v2.py",
    "scripts/mot_jepa_control_v2_acceptance.py",
    "scripts/mot_jepa_control_v2_merge_paired_eval.py",
    "scripts/mot_jepa_control_v2_provenance.py",
    "scripts/mot_jepa_control_v2_qualified_runtime.py",
    "scripts_exp_zarr/mot_jepa/control_v2_closedloop.sbatch",
    "scripts_exp_zarr/mot_jepa/control_v2_closedloop_paired_shard.sbatch",
    "scripts_exp_zarr/mot_jepa/control_v2_closedloop_inner.sh",
    "scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh",
    "UniVTAC/scripts",
    "UniVTAC/task_config/demo.yml",
    "UniVTAC/envs",
    "UniVTAC/encoder",
    "UniVTAC/policy",
    "UniVTAC/assets",
    "UniVTAC/third_party/TacEx/source/tacex/tacex",
    "UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors",
    "UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/robots",
    "UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/__init__.py",
    "UniVTAC/third_party/TacEx/source/tacex_assets/config/extension.toml",
    "UniVTAC/third_party/TacEx/source/tacex_tasks/tacex_tasks",
    "UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc",
    "src/openpi",
)
SHARD_MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "root",
    "start_seed",
    "end_seed",
    "trials",
    "artifact",
    "artifact_sha256",
    "selected_step",
    "completed_step",
    "selection_value",
    "qualified_runtime",
    "qualified_runtime_sha256",
    "runtime_qualification_sha256",
    "provenance",
    "provenance_sha256",
    "source_content_sha256",
    "static_source_sha256",
    "external_runtime_roots",
    "external_runtime_roots_sha256",
    "append_manifest",
    "append_manifest_sha256",
    "append_launcher",
    "append_launcher_sha256",
    "append_submission_sha256",
    "n20_job_id",
    "evaluation_contract",
    "evaluation_contract_sha256",
    "student_root",
    "official_root",
    "student_metadata",
    "official_metadata",
    "student_eval_args",
    "official_eval_args",
    "student_seed_results",
    "official_seed_results",
    "student_trace_manifest",
    "student_trace_content_sha256",
    "model_contract_sha256",
    "contract_sha256",
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = pathlib.Path(raw)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _canonical_member(root: pathlib.Path, raw: Any, relative: str, *, directory: bool = False) -> pathlib.Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"paired shard does not declare {relative}")
    declared = pathlib.Path(raw)
    if not declared.is_absolute() or ".." in declared.parts:
        raise ValueError(f"paired shard member is not absolute/traversal-free: {declared}")
    root = root.resolve()
    expected = root / relative
    if declared != expected:
        raise ValueError(f"paired shard member is not canonical: {declared} != {expected}")
    current = root
    for part in pathlib.Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"paired shard member may not contain a symlink: {current}")
    if not expected.resolve().is_relative_to(root):
        raise ValueError(f"paired shard member escapes its root: {expected}")
    mode = expected.lstat().st_mode if expected.exists() else 0
    exists_as_expected_type = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not exists_as_expected_type:
        raise ValueError(f"paired shard member is missing or has the wrong type: {expected}")
    return expected


def _normalized_eval_args(summary: dict[str, Any]) -> dict[str, Any]:
    raw = summary.get("eval_args")
    if not isinstance(raw, dict):
        raise ValueError("evaluator metadata lacks an eval_args object")
    normalized = dict(raw)
    for field in ("start_seed", "total_num", "save_root"):
        normalized.pop(field, None)
    return normalized


def _runtime_record_path(record: Any, label: str, *, directory: bool) -> pathlib.Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"qualified runtime lacks a path for {label}")
    raw = pathlib.Path(record["path"])
    if not raw.is_absolute() or ".." in raw.parts or raw != raw.resolve():
        raise ValueError(f"qualified runtime path is not canonical for {label}: {raw}")
    mode = raw.lstat().st_mode if raw.exists() else 0
    if (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)) is False:
        raise ValueError(f"qualified runtime path is missing or has the wrong type for {label}: {raw}")
    return raw


def _runtime_external_roots(qualified_runtime: dict[str, Any], repo_root: pathlib.Path) -> dict[str, pathlib.Path]:
    common = qualified_runtime["common"]
    official = qualified_runtime["official"]
    expected_records = {
        "container": {
            "bytes": runtime_qualifier.CONTAINER[0],
            "sha256": runtime_qualifier.CONTAINER[1],
        },
        "external_assets": {
            "file_count": runtime_qualifier.EXTERNAL_ASSET_TREE[0],
            "bytes": runtime_qualifier.EXTERNAL_ASSET_TREE[1],
            "sha256": runtime_qualifier.EXTERNAL_ASSET_TREE[2],
        },
        "local_assets": {
            "file_count": runtime_qualifier.LOCAL_ASSET_TREE[0],
            "bytes": runtime_qualifier.LOCAL_ASSET_TREE[1],
            "sha256": runtime_qualifier.LOCAL_ASSET_TREE[2],
        },
        "overlay": {
            "file_count": runtime_qualifier.FTP1_OVERLAY_TREE[0],
            "bytes": runtime_qualifier.FTP1_OVERLAY_TREE[1],
            "sha256": runtime_qualifier.FTP1_OVERLAY_TREE[2],
        },
        "ready": {"bytes": runtime_qualifier.READY[0], "sha256": runtime_qualifier.READY[1]},
        "tokenizer": {"bytes": runtime_qualifier.TOKENIZER[0], "sha256": runtime_qualifier.TOKENIZER[1]},
    }
    for name in ("container", "external_assets", "local_assets"):
        record = common.get(name)
        expected = expected_records[name]
        if not isinstance(record, dict) or set(record) != {"path", *expected}:
            raise ValueError(f"qualified runtime {name} record has an invalid field set")
        if any(_canonical_digest(record[field]) != _canonical_digest(value) for field, value in expected.items()):
            raise ValueError(f"qualified runtime {name} identity is not official")
    for name in ("overlay", "ready", "tokenizer"):
        record = official.get(name)
        expected = expected_records[name]
        if not isinstance(record, dict) or set(record) != {"path", *expected}:
            raise ValueError(f"qualified runtime {name} record has an invalid field set")
        if any(_canonical_digest(record[field]) != _canonical_digest(value) for field, value in expected.items()):
            raise ValueError(f"qualified runtime {name} identity is not official")
    container = _runtime_record_path(common["container"], "container", directory=False)
    external_assets = _runtime_record_path(common["external_assets"], "external assets", directory=True)
    local_assets = _runtime_record_path(common["local_assets"], "local assets", directory=True)
    expected_local_assets = (repo_root / "UniVTAC").resolve()
    if local_assets != expected_local_assets:
        raise ValueError(f"qualified local-assets root differs: {local_assets} != {expected_local_assets}")

    overlay = _runtime_record_path(official["overlay"], "FTP1 overlay", directory=True)
    ready = _runtime_record_path(official["ready"], "FTP1 overlay READY", directory=False)
    if ready != overlay / "READY.json":
        raise ValueError("qualified FTP1 READY is not canonical under the overlay")
    tokenizer = _runtime_record_path(official["tokenizer"], "FTP1 tokenizer", directory=False)

    checkpoint_records = official["checkpoint"]
    if not isinstance(checkpoint_records, dict) or set(checkpoint_records) != set(runtime_qualifier.FTP1_FILES):
        raise ValueError("qualified runtime does not have the exact official FTP1 checkpoint members")
    checkpoint_roots: set[pathlib.Path] = set()
    for relative, record in checkpoint_records.items():
        expected_bytes, expected_sha256 = runtime_qualifier.FTP1_FILES[relative]
        expected_record = {"bytes": expected_bytes, "sha256": expected_sha256}
        if not isinstance(record, dict) or set(record) != {"path", *expected_record}:
            raise ValueError(f"qualified FTP1 checkpoint record is invalid: {relative}")
        if any(
            _canonical_digest(record[field]) != _canonical_digest(value) for field, value in expected_record.items()
        ):
            raise ValueError(f"qualified FTP1 checkpoint member identity is not official: {relative}")
        member = pathlib.Path(relative) if isinstance(relative, str) else pathlib.Path()
        if not relative or member.is_absolute() or ".." in member.parts:
            raise ValueError(f"qualified FTP1 checkpoint member is invalid: {relative!r}")
        path = _runtime_record_path(record, f"FTP1 checkpoint/{relative}", directory=False)
        candidate = path
        for _part in member.parts:
            candidate = candidate.parent
        checkpoint_roots.add(candidate)
    if len(checkpoint_roots) != 1:
        raise ValueError("qualified FTP1 checkpoint records do not share one canonical root")
    checkpoint = checkpoint_roots.pop()
    for relative, record in checkpoint_records.items():
        if pathlib.Path(record["path"]) != checkpoint / relative:
            raise ValueError(f"qualified FTP1 checkpoint member escapes its root: {relative}")

    return {
        "container": container,
        "external_assets": external_assets,
        "overlay": overlay,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
    }


def _canonical_external_file(raw: Any, label: str) -> pathlib.Path:
    path = pathlib.Path(raw) if isinstance(raw, str) else pathlib.Path()
    if not isinstance(raw, str) or not path.is_absolute() or ".." in path.parts or path != path.resolve():
        raise ValueError(f"{label} is not a canonical absolute path: {raw!r}")
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"{label} is missing, symlinked, or not a regular file: {path}")
    return path


def _validate_append_submission(
    append_manifest_path: pathlib.Path,
    append_launcher_path: pathlib.Path,
    *,
    shard_root: pathlib.Path,
    repo_root: pathlib.Path,
    artifact_path: pathlib.Path,
    evaluation_contract: dict[str, Any],
    start_seed: int,
) -> tuple[dict[str, Any], str]:
    append_manifest_path = _canonical_external_file(str(append_manifest_path), "append submission manifest")
    append_launcher_path = _canonical_external_file(str(append_launcher_path), "append launcher")
    expected_launcher = (repo_root / "scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh").resolve()
    if append_launcher_path != expected_launcher:
        raise ValueError(f"append launcher differs from the versioned repository path: {append_launcher_path}")
    payload = _read_json(append_manifest_path)
    expected_fields = {
        "schema_version",
        "status",
        "v2_run",
        "n20_job_id",
        "n20_dependency",
        "pair_parent",
        "eval_node",
        "shard",
        "merge",
        "job_ids",
        "sources",
        "submitted_unix",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1 or payload.get("status") != "SUBMITTED":
        raise ValueError("append submission manifest has an invalid top-level contract")
    n20_job = payload.get("n20_job_id")
    if not isinstance(n20_job, str) or not n20_job.isdigit() or payload.get("n20_dependency") != f"afterok:{n20_job}":
        raise ValueError("append submission manifest has an invalid n20 dependency")
    if pathlib.Path(str(payload.get("v2_run", ""))).resolve() != artifact_path.parent:
        raise ValueError("append submission manifest names a different V2 run")
    pair_parent = pathlib.Path(str(payload.get("pair_parent", "")))
    if not pair_parent.is_absolute() or ".." in pair_parent.parts or pair_parent.resolve() != shard_root.parent:
        raise ValueError("append submission manifest names a different paired-output parent")
    if payload.get("eval_node") != EVAL_NODE:
        raise ValueError("append submission manifest names a non-preregistered evaluation node")

    shard = payload.get("shard")
    expected_shard_fields = {
        "partition",
        "time_limit",
        "run_seconds_per_policy",
        "trials_each",
        "start_seeds",
        "job_ids",
        "roots",
    }
    if not isinstance(shard, dict) or set(shard) != expected_shard_fields:
        raise ValueError("append submission manifest has an invalid shard contract")
    if any(
        (
            shard.get("partition") != "batch",
            shard.get("time_limit") != "04:00:00",
            shard.get("run_seconds_per_policy") != 5400,
            shard.get("trials_each") != SHARD_TRIALS,
            shard.get("start_seeds") != list(SHARD_STARTS),
        )
    ):
        raise ValueError("append submission manifest differs from the preregistered shard protocol")
    shard_jobs = shard.get("job_ids")
    if (
        not isinstance(shard_jobs, list)
        or len(shard_jobs) != len(SHARD_STARTS)
        or len(set(shard_jobs)) != len(SHARD_STARTS)
        or any(not isinstance(job, str) or not job.isdigit() for job in shard_jobs)
    ):
        raise ValueError("append submission manifest has invalid shard job IDs")
    expected_roots = [
        str(pair_parent.resolve() / f"lift_bottle_s{seed}_n20_{job}")
        for seed, job in zip(SHARD_STARTS, shard_jobs, strict=True)
    ]
    if shard.get("roots") != expected_roots:
        raise ValueError("append submission manifest shard roots differ from job IDs and seeds")
    shard_index = SHARD_STARTS.index(start_seed)
    if evaluation_contract["job_id"] != shard_jobs[shard_index] or str(shard_root) != expected_roots[shard_index]:
        raise ValueError("current paired shard differs from its append submission slot")

    merge = payload.get("merge")
    expected_merge_fields = {"partition", "time_limit", "dependency", "job_id", "root"}
    if not isinstance(merge, dict) or set(merge) != expected_merge_fields:
        raise ValueError("append submission manifest has an invalid merge contract")
    merge_job = merge.get("job_id")
    expected_dependency = f"afterok:{':'.join(shard_jobs)}"
    if (
        merge.get("partition") != "cpu"
        or merge.get("time_limit") != "02:00:00"
        or merge.get("dependency") != expected_dependency
        or not isinstance(merge_job, str)
        or not merge_job.isdigit()
        or merge_job in shard_jobs
        or merge.get("root") != str(pair_parent.resolve() / f"lift_bottle_s1000000_n100_{merge_job}")
    ):
        raise ValueError("append submission manifest differs from the preregistered merge protocol")
    if payload.get("job_ids") != [*shard_jobs, merge_job]:
        raise ValueError("append submission manifest global job list differs")

    source_paths = {
        "launcher": append_launcher_path,
        "shard_sbatch": (repo_root / "scripts_exp_zarr/mot_jepa/control_v2_closedloop_paired_shard.sbatch").resolve(),
        "merge_sbatch": (repo_root / "scripts_exp_zarr/mot_jepa/control_v2_merge_paired_eval.sbatch").resolve(),
    }
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(source_paths):
        raise ValueError("append submission manifest has an invalid source set")
    for name, expected_path in source_paths.items():
        record = sources.get(name)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"append submission source record is invalid: {name}")
        live_path = _canonical_external_file(record["path"], f"append submission source {name}")
        if live_path != expected_path or record["sha256"] != _sha256(live_path):
            raise ValueError(f"append submission source changed or has the wrong path: {name}")
    submitted = payload.get("submitted_unix")
    if type(submitted) not in {int, float} or not math.isfinite(submitted) or submitted <= 0:
        raise ValueError("append submission manifest has an invalid submission time")
    if evaluation_contract["append_manifest"] != str(append_manifest_path):
        raise ValueError("evaluation contract names a different append manifest")
    if evaluation_contract["append_launcher"] != str(append_launcher_path):
        raise ValueError("evaluation contract names a different append launcher")
    return payload, _canonical_digest(payload)


def _static_source_contract(
    provenance_path: pathlib.Path,
    shard_root: pathlib.Path,
    qualified_runtime: dict[str, Any],
    append_manifest_path: pathlib.Path,
    append_launcher_path: pathlib.Path,
) -> tuple[str, dict[str, Any], pathlib.Path, dict[str, pathlib.Path]]:
    payload = _read_json(provenance_path)
    provenance.verify(payload)
    files = {
        path: record
        for path, record in payload["files"].items()
        if not pathlib.Path(path).resolve().is_relative_to(shard_root)
    }
    roots = []
    roots_by_path: dict[str, dict[str, Any]] = {}
    for raw_record in payload["roots"]:
        if not isinstance(raw_record, dict) or set(raw_record) != {"path", "content_hashed"}:
            raise ValueError("paired shard provenance has an invalid root record")
        path = pathlib.Path(raw_record["path"])
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"paired shard provenance root is not canonical: {path}")
        resolved = path.resolve()
        if resolved.is_relative_to(shard_root):
            continue
        record = {"path": str(resolved), "content_hashed": raw_record["content_hashed"]}
        if record["path"] in roots_by_path:
            raise ValueError(f"paired shard provenance repeats root: {record['path']}")
        roots.append(record)
        roots_by_path[record["path"]] = record
    roots.sort(key=lambda record: record["path"])
    launcher_suffix = "/scripts_exp_zarr/mot_jepa/control_v2_closedloop_paired_shard.sbatch"
    launchers = [pathlib.Path(path) for path in roots_by_path if path.endswith(launcher_suffix)]
    if len(launchers) != 1:
        raise ValueError("paired shard source closure lacks exactly one versioned shard launcher root")
    repo_root = launchers[0].parents[2]
    for relative in REQUIRED_STATIC_PROFILE_ROOTS:
        required = (repo_root / relative).resolve()
        record = roots_by_path.get(str(required))
        if record is None or record.get("content_hashed") is not True:
            raise ValueError(f"paired shard source closure lacks required static profile root: {required}")
    expected_append_launcher = (repo_root / "scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh").resolve()
    if append_launcher_path.resolve() != expected_append_launcher:
        raise ValueError("paired shard declares the wrong append launcher")
    for label, required in (("append manifest", append_manifest_path), ("append launcher", append_launcher_path)):
        record = roots_by_path.get(str(required.resolve()))
        if record is None or record.get("content_hashed") is not True:
            raise ValueError(f"paired shard source closure lacks required hashed {label} root: {required}")
    external_roots = _runtime_external_roots(qualified_runtime, repo_root)
    for name, path in external_roots.items():
        record = roots_by_path.get(str(path))
        if record is None or record.get("content_hashed") is not False:
            raise ValueError(f"paired shard source closure lacks required stat-only {name} root: {path}")
    observed_stat_only = {path for path, record in roots_by_path.items() if record.get("content_hashed") is False}
    expected_stat_only = {str(path) for path in external_roots.values()}
    if observed_stat_only != expected_stat_only:
        raise ValueError(
            "paired shard source closure has an unexpected stat-only root set: "
            f"missing={sorted(expected_stat_only - observed_stat_only)}, "
            f"extra={sorted(observed_stat_only - expected_stat_only)}"
        )
    static = {"files": files, "roots": roots}
    return _canonical_digest(static), payload, repo_root, external_roots


def _validate_evaluation_contract(
    evaluation_contract: dict[str, Any],
    *,
    start_seed: int,
    selected_step: int,
    completed_step: int,
) -> None:
    expected_fields = {
        "schema_version",
        "job_id",
        "requested_node",
        "actual_node_list",
        "mode",
        "trials",
        "start_seed",
        "selected_step",
        "completed_step",
        "append_manifest",
        "append_launcher",
        "created_unix",
    }
    if set(evaluation_contract) != expected_fields:
        raise ValueError("paired-shard evaluation contract has an invalid field set")
    expected = {
        "schema_version": 2,
        "requested_node": EVAL_NODE,
        "actual_node_list": EVAL_NODE,
        "mode": "paired_shard",
        "trials": SHARD_TRIALS,
        "start_seed": start_seed,
        "selected_step": selected_step,
        "completed_step": completed_step,
    }
    mismatches = {
        field: (evaluation_contract.get(field), value)
        for field, value in expected.items()
        if _canonical_digest(evaluation_contract.get(field)) != _canonical_digest(value)
    }
    if mismatches:
        raise ValueError(f"paired-shard evaluation contract differs: {mismatches}")
    if not isinstance(evaluation_contract.get("job_id"), str) or not evaluation_contract["job_id"].isdigit():
        raise ValueError("paired-shard evaluation contract lacks a numeric job_id")
    for field in ("append_manifest", "append_launcher"):
        raw = evaluation_contract.get(field)
        path = pathlib.Path(raw) if isinstance(raw, str) else pathlib.Path()
        if not isinstance(raw, str) or not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"paired-shard evaluation contract has an invalid {field}")
    created = evaluation_contract.get("created_unix")
    if type(created) not in {int, float} or not math.isfinite(created) or created <= 0:
        raise ValueError("paired-shard evaluation contract has an invalid creation time")


def _expected_eval_args(
    *,
    policy: str,
    shard_root: pathlib.Path,
    repo_root: pathlib.Path,
    artifact_path: pathlib.Path,
    official_checkpoint: pathlib.Path,
    start_seed: int,
) -> dict[str, Any]:
    if policy not in {"student", "official"}:
        raise ValueError(policy)
    args: dict[str, Any] = {
        "checkpoint_dir": str(artifact_path.parent if policy == "student" else official_checkpoint),
        "policy_name": "mot_control_v2" if policy == "student" else "official_ftp1",
        "domain_name": "UniVTAC_lift_bottle",
        "task_list": "lift_bottle",
        "task_config": "demo.yml",
        "total_num": SHARD_TRIALS,
        "start_seed": start_seed,
        "workers": 1,
        "gpu": "",
        "ftp1_device": "cuda:1",
        "low_memory": False,
        "num_inference_steps": 10,
        "chunk_first_n": 20,
        "instruction_type": "seen",
        "save_root": str(shard_root / policy),
        "run_suffix": "",
        "run_id": policy,
        "tactile_key": "right_tactile_gripper",
        "tactile_sensor": "GelSightMini",
        "no_video": True,
        "video_frequency": 0,
        "save_trajectory": True,
        "trajectory_max_actions": 500,
        "gripper_slot_idx": 28,
        "arm_slice": "0:7" if policy == "student" else "9:16",
        "gripper_index": 7 if policy == "student" else 44,
        "action_rep": "absolute" if policy == "student" else "mix",
        "save_infer_input_dir": None,
        "policy_sample_seed": 0,
        "max_steps": 0,
        "temporal_ensemble": True,
        "ensemble_K": 0.01,
        "summary_only_run_roots": "",
        "resume": False,
        "task_run_ids": {"lift_bottle": policy},
        "task_config_path": str((repo_root / "UniVTAC/task_config/demo.yml").resolve()),
    }
    return args


def _validate_eval_args(summary: dict[str, Any], expected: dict[str, Any], *, policy: str) -> dict[str, Any]:
    actual = summary.get("eval_args")
    if not isinstance(actual, dict):
        raise ValueError(f"{policy} evaluator metadata lacks an eval_args object")
    canonical_actual = dict(actual)
    canonical_expected = dict(expected)
    for field in ("checkpoint_dir", "save_root", "task_config_path"):
        for label, values in (("actual", canonical_actual), ("expected", canonical_expected)):
            raw = values.get(field)
            path = pathlib.Path(raw) if isinstance(raw, str) else pathlib.Path()
            if not isinstance(raw, str) or not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{policy} evaluator {label} {field} is not an absolute canonicalizable path")
            values[field] = str(path.resolve())
    if _canonical_digest(canonical_actual) != _canonical_digest(canonical_expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            field
            for field in set(actual) & set(expected)
            if _canonical_digest(canonical_actual[field]) != _canonical_digest(canonical_expected[field])
        )
        raise ValueError(f"{policy} evaluator args differ: missing={missing}, extra={extra}, changed={changed}")
    return _normalized_eval_args({"eval_args": canonical_actual})


def _policy_records(root: pathlib.Path, *, start_seed: int) -> tuple[dict[int, bool], dict[int, dict], dict]:
    results, entries = acceptance.load_episode_results(root, start_seed=start_seed, trials=SHARD_TRIALS)
    return results, entries, acceptance.result_metadata_manifest(root)


def build_shard_manifest(
    shard_root: pathlib.Path,
    artifact_path: pathlib.Path,
    *,
    start_seed: int,
) -> dict[str, Any]:
    """Revalidate one raw paired shard without imposing a task-success threshold."""

    root = pathlib.Path(shard_root).resolve()
    artifact_path = pathlib.Path(artifact_path).resolve()
    if start_seed not in SHARD_STARTS:
        raise ValueError(f"paired shard start_seed must be one of {SHARD_STARTS}")
    evaluation_contract_path = root / "evaluation_contract.json"
    provenance_path = root / "provenance.json"
    runtime_before = root / "qualified_runtime.json"
    runtime_after = root / "qualified_runtime_after.json"
    student_root = root / "student/lift_bottle/student"
    official_root = root / "official/lift_bottle/official"
    for path in (evaluation_contract_path, provenance_path, runtime_before, runtime_after, artifact_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if runtime_before.read_bytes() != runtime_after.read_bytes():
        raise ValueError("paired-shard runtime qualification changed during evaluation")

    artifact, selection, _selection_path = acceptance.load_deployment_contract(artifact_path)
    completed_step = int((artifact_path.parent / "DONE").read_text().strip())
    selected_step = int(selection["step"])
    evaluation_contract = _read_json(evaluation_contract_path)
    _validate_evaluation_contract(
        evaluation_contract,
        start_seed=start_seed,
        selected_step=selected_step,
        completed_step=completed_step,
    )
    append_manifest_path = _canonical_external_file(
        evaluation_contract["append_manifest"],
        "evaluation append manifest",
    )
    append_launcher_path = _canonical_external_file(
        evaluation_contract["append_launcher"],
        "evaluation append launcher",
    )

    qualified_runtime = acceptance.load_qualified_runtime(runtime_after, require_official=True)
    static_source_sha256, source_payload, repo_root, external_roots = _static_source_contract(
        provenance_path,
        root,
        qualified_runtime,
        append_manifest_path,
        append_launcher_path,
    )
    append_submission, append_submission_sha256 = _validate_append_submission(
        append_manifest_path,
        append_launcher_path,
        shard_root=root,
        repo_root=repo_root,
        artifact_path=artifact_path,
        evaluation_contract=evaluation_contract,
        start_seed=start_seed,
    )
    student_results, student_entries, student_metadata = _policy_records(student_root, start_seed=start_seed)
    official_results, _official_entries, official_metadata = _policy_records(official_root, start_seed=start_seed)
    if set(student_results) != set(official_results):
        raise ValueError("paired-shard student and official seed sets differ")

    limits = SafetyLimits(
        joint_lower=artifact.joint_lower,
        joint_upper=artifact.joint_upper,
        max_delta=artifact.max_command_delta,
    )
    artifact_sha256 = _sha256(artifact_path)
    traces, trace_failures = acceptance.qualify_student_traces(
        student_root,
        student_entries,
        limits=limits,
        first_gripper_jump_max=FIRST_GRIPPER_JUMP_MAX,
        expected_policy_step=selected_step,
        expected_completed_step=completed_step,
        expected_selection_value=float(selection["validation_action_loss"]),
        expected_artifact_sha256=artifact_sha256,
    )
    if trace_failures:
        raise ValueError(f"paired-shard student traces violate the deployment contract: {trace_failures}")

    student_summary = _read_json(student_root / "metadata.json")
    official_summary = _read_json(official_root / "metadata.json")
    student_eval_args = _validate_eval_args(
        student_summary,
        _expected_eval_args(
            policy="student",
            shard_root=root,
            repo_root=repo_root,
            artifact_path=artifact_path,
            official_checkpoint=external_roots["checkpoint"],
            start_seed=start_seed,
        ),
        policy="student",
    )
    official_eval_args = _validate_eval_args(
        official_summary,
        _expected_eval_args(
            policy="official",
            shard_root=root,
            repo_root=repo_root,
            artifact_path=artifact_path,
            official_checkpoint=external_roots["checkpoint"],
            start_seed=start_seed,
        ),
        policy="official",
    )
    external_runtime_roots = {name: str(path) for name, path in sorted(external_roots.items())}
    trace_manifest = {
        str(seed): {"path": report["path"], "sha256": report["sha256"], "passed": bool(report["passed"])}
        for seed, report in sorted(traces.items())
    }
    model_contract = {
        "artifact_sha256": artifact_sha256,
        "selected_step": selected_step,
        "completed_step": completed_step,
        "selection_value": float(selection["validation_action_loss"]),
    }
    contract = {
        "model_contract_sha256": _canonical_digest(model_contract),
        "runtime_qualification_sha256": qualified_runtime["qualification_sha256"],
        "static_source_sha256": static_source_sha256,
        "external_runtime_roots_sha256": _canonical_digest(external_runtime_roots),
        "append_submission_sha256": append_submission_sha256,
        "student_eval_args": student_eval_args,
        "official_eval_args": official_eval_args,
    }
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "root": str(root),
        "start_seed": start_seed,
        "end_seed": start_seed + SHARD_TRIALS - 1,
        "trials": SHARD_TRIALS,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "selected_step": selected_step,
        "completed_step": completed_step,
        "selection_value": float(selection["validation_action_loss"]),
        "qualified_runtime": str(runtime_after),
        "qualified_runtime_sha256": _sha256(runtime_after),
        "runtime_qualification_sha256": qualified_runtime["qualification_sha256"],
        "provenance": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "source_content_sha256": source_payload["content_sha256"],
        "static_source_sha256": static_source_sha256,
        "external_runtime_roots": external_runtime_roots,
        "external_runtime_roots_sha256": _canonical_digest(external_runtime_roots),
        "append_manifest": str(append_manifest_path),
        "append_manifest_sha256": _sha256(append_manifest_path),
        "append_launcher": str(append_launcher_path),
        "append_launcher_sha256": _sha256(append_launcher_path),
        "append_submission_sha256": append_submission_sha256,
        "n20_job_id": append_submission["n20_job_id"],
        "evaluation_contract": str(evaluation_contract_path),
        "evaluation_contract_sha256": _sha256(evaluation_contract_path),
        "student_root": str(student_root),
        "official_root": str(official_root),
        "student_metadata": student_metadata,
        "official_metadata": official_metadata,
        "student_eval_args": student_eval_args,
        "official_eval_args": official_eval_args,
        "student_seed_results": {str(seed): result for seed, result in sorted(student_results.items())},
        "official_seed_results": {str(seed): result for seed, result in sorted(official_results.items())},
        "student_trace_manifest": trace_manifest,
        "student_trace_content_sha256": _canonical_digest(trace_manifest),
        "model_contract_sha256": _canonical_digest(model_contract),
        "contract_sha256": _canonical_digest(contract),
    }


def write_shard_manifest(
    shard_root: pathlib.Path,
    artifact_path: pathlib.Path,
    *,
    start_seed: int,
    output: pathlib.Path,
    done: pathlib.Path,
) -> dict[str, Any]:
    root = pathlib.Path(shard_root).resolve()
    output = pathlib.Path(output)
    done = pathlib.Path(done)
    if output.resolve() != root / "PAIRED_SHARD.json" or done.resolve() != root / "DONE":
        raise ValueError("paired shard manifest/DONE paths must be canonical under the shard root")
    if output.exists() or done.exists():
        raise FileExistsError(output if output.exists() else done)
    payload = build_shard_manifest(root, artifact_path, start_seed=start_seed)
    _atomic_json(output, payload)
    marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "start_seed": start_seed,
        "trials": SHARD_TRIALS,
        "manifest_sha256": _sha256(output),
    }
    _atomic_json(done, marker)
    return payload


def _validate_envelope(root: pathlib.Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "PAIRED_SHARD.json"
    done_path = root / "DONE"
    manifest = _read_json(manifest_path)
    marker = _read_json(done_path)
    if set(manifest) != SHARD_MANIFEST_FIELDS:
        raise ValueError(f"paired shard manifest has an invalid field set: {root}")
    expected_marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "start_seed": manifest.get("start_seed"),
        "trials": SHARD_TRIALS,
        "manifest_sha256": _sha256(manifest_path),
    }
    if marker != expected_marker:
        raise ValueError(f"paired shard DONE/manifest digest differs: {root}")
    if manifest["root"] != str(root) or manifest["status"] != "COMPLETE" or manifest["schema_version"] != 1:
        raise ValueError(f"paired shard manifest identity differs: {root}")
    start = manifest["start_seed"]
    if type(start) is not int or start not in SHARD_STARTS or manifest["trials"] != SHARD_TRIALS:
        raise ValueError(f"paired shard seed interval is invalid: {root}")
    if manifest["end_seed"] != start + SHARD_TRIALS - 1:
        raise ValueError(f"paired shard end seed is invalid: {root}")
    _canonical_member(root, manifest["provenance"], "provenance.json")
    _canonical_member(root, manifest["qualified_runtime"], "qualified_runtime_after.json")
    _canonical_member(root, manifest["evaluation_contract"], "evaluation_contract.json")
    _canonical_member(root, manifest["student_root"], "student/lift_bottle/student", directory=True)
    _canonical_member(root, manifest["official_root"], "official/lift_bottle/official", directory=True)
    return manifest


def load_shard_envelopes(shard_roots: list[pathlib.Path]) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    """Validate the exact five envelope/seed/contract closures before touching an output root."""

    roots = [pathlib.Path(root).resolve() for root in shard_roots]
    if len(roots) != 5 or len(set(roots)) != 5:
        raise ValueError("paired n100 merge requires exactly five distinct shard roots")
    envelopes = [(root, _validate_envelope(root)) for root in roots]
    envelopes.sort(key=lambda item: item[1]["start_seed"])
    starts = tuple(manifest["start_seed"] for _, manifest in envelopes)
    if starts != SHARD_STARTS:
        raise ValueError(f"paired shard starts differ: got={starts}, expected={SHARD_STARTS}")

    observed: dict[str, set[int]] = {"student": set(), "official": set(), "trace": set()}
    fields = {
        "student": "student_seed_results",
        "official": "official_seed_results",
        "trace": "student_trace_manifest",
    }
    for _root, manifest in envelopes:
        local_expected = set(range(manifest["start_seed"], manifest["start_seed"] + SHARD_TRIALS))
        for name, field in fields.items():
            raw = manifest[field]
            if not isinstance(raw, dict):
                raise ValueError(f"paired shard {field} is not an object")
            try:
                seeds = {int(seed) for seed in raw}
            except ValueError as exc:
                raise ValueError(f"paired shard {field} has a non-integer seed") from exc
            duplicates = observed[name] & seeds
            if duplicates:
                raise ValueError(f"duplicate {name} seeds across paired shards: {sorted(duplicates)}")
            observed[name].update(seeds)
            if seeds != local_expected:
                raise ValueError(
                    f"paired shard {field} seed set differs: missing={sorted(local_expected - seeds)}, "
                    f"extra={sorted(seeds - local_expected)}"
                )
    for name, seeds in observed.items():
        if seeds != N100_SEEDS:
            raise ValueError(f"combined {name} seeds do not close exact n100")

    identical_fields = (
        "artifact",
        "artifact_sha256",
        "selected_step",
        "completed_step",
        "selection_value",
        "qualified_runtime_sha256",
        "runtime_qualification_sha256",
        "static_source_sha256",
        "external_runtime_roots",
        "external_runtime_roots_sha256",
        "append_manifest",
        "append_manifest_sha256",
        "append_launcher",
        "append_launcher_sha256",
        "append_submission_sha256",
        "n20_job_id",
        "student_eval_args",
        "official_eval_args",
        "model_contract_sha256",
        "contract_sha256",
    )
    for field in identical_fields:
        values = {_canonical_digest(manifest[field]) for _, manifest in envelopes}
        if len(values) != 1:
            raise ValueError(f"paired shard contracts differ for {field}")
    return envelopes


def _hardlink(source: pathlib.Path, target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def _synthesize_policy_root(
    envelopes: list[tuple[pathlib.Path, dict[str, Any]]],
    *,
    policy: str,
    temp_root: pathlib.Path,
    final_root: pathlib.Path,
) -> pathlib.Path:
    relative = pathlib.Path(policy) / "lift_bottle" / policy
    target = temp_root / relative
    target.mkdir(parents=True)
    all_entries: dict[str, Any] = {}
    first_summary: dict[str, Any] | None = None
    for index, (_shard_root, manifest) in enumerate(envelopes):
        source_root = pathlib.Path(manifest[f"{policy}_root"])
        worker_paths = sorted((source_root / "metadata").glob("worker_*.json"))
        if len(worker_paths) != 1:
            raise ValueError(f"{source_root} must contain exactly one raw worker metadata file")
        entries = _read_json(worker_paths[0])
        overlap = set(all_entries) & set(entries)
        if overlap:
            raise ValueError(f"duplicate {policy} worker seeds during synthesis: {sorted(overlap)}")
        all_entries.update(entries)
        _hardlink(worker_paths[0], target / "metadata" / f"worker_{index}.json")
        summary = _read_json(source_root / "metadata.json")
        first_summary = summary if first_summary is None else first_summary

        if policy == "student":
            trace_paths = sorted((source_root / "trajectory").glob("worker_*/*.npz"))
            if len(trace_paths) != SHARD_TRIALS:
                raise ValueError(f"{source_root} does not contain exactly {SHARD_TRIALS} student traces")
            for trace in trace_paths:
                _hardlink(trace, target / "trajectory" / f"worker_{index}" / trace.name)

    if len(all_entries) != 100 or first_summary is None:
        raise ValueError(f"combined {policy} worker metadata does not contain exact n100")
    success = sum(entry.get("result") == "success" for entry in all_entries.values())
    failed = sum(entry.get("result") == "failed" for entry in all_entries.values())
    if success + failed != 100:
        raise ValueError(f"combined {policy} records contain incomplete results")
    eval_args = dict(first_summary["eval_args"])
    eval_args["start_seed"] = N100_START_SEED
    eval_args["total_num"] = 100
    if "save_root" in eval_args:
        eval_args["save_root"] = str(final_root / policy)
    summary = {
        "total_episodes": 100,
        "success": success,
        "failed": failed,
        "error": 0,
        "eval_args": eval_args,
    }
    _atomic_json(target / "metadata.json", summary)
    return target


def merge_paired_shards(shard_roots: list[pathlib.Path], output_root: pathlib.Path) -> dict[str, Any]:
    """Revalidate raw shards, synthesize canonical roots, and leave acceptance to the unchanged CLI."""

    envelopes = load_shard_envelopes(shard_roots)
    for root, stored in envelopes:
        recomputed = build_shard_manifest(root, pathlib.Path(stored["artifact"]), start_seed=stored["start_seed"])
        if recomputed != stored:
            raise ValueError(f"paired shard raw records/contracts changed after completion: {root}")

    output_root = pathlib.Path(output_root).resolve()
    append_submission = _read_json(pathlib.Path(envelopes[0][1]["append_manifest"]))
    expected_shard_roots = append_submission["shard"]["roots"]
    if [str(root) for root, _manifest in envelopes] != expected_shard_roots:
        raise ValueError("paired merge inputs differ from the append submission manifest")
    if str(output_root) != append_submission["merge"]["root"]:
        raise ValueError("paired merge output differs from the append submission manifest")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        student_temp = _synthesize_policy_root(
            envelopes,
            policy="student",
            temp_root=temp_root,
            final_root=output_root,
        )
        official_temp = _synthesize_policy_root(
            envelopes,
            policy="official",
            temp_root=temp_root,
            final_root=output_root,
        )
        runtime_source = pathlib.Path(envelopes[0][1]["qualified_runtime"])
        _hardlink(runtime_source, temp_root / "qualified_runtime.json")
        student_final = output_root / student_temp.relative_to(temp_root)
        official_final = output_root / official_temp.relative_to(temp_root)
        shard_records = [
            {
                "root": str(root),
                "start_seed": manifest["start_seed"],
                "end_seed": manifest["end_seed"],
                "manifest": str(root / "PAIRED_SHARD.json"),
                "manifest_sha256": _sha256(root / "PAIRED_SHARD.json"),
            }
            for root, manifest in envelopes
        ]
        payload = {
            "schema_version": 1,
            "status": "MERGED",
            "start_seed": N100_START_SEED,
            "trials": 100,
            "shard_starts": list(SHARD_STARTS),
            "shards": shard_records,
            "artifact": envelopes[0][1]["artifact"],
            "artifact_sha256": envelopes[0][1]["artifact_sha256"],
            "selected_step": envelopes[0][1]["selected_step"],
            "completed_step": envelopes[0][1]["completed_step"],
            "contract_sha256": envelopes[0][1]["contract_sha256"],
            "append_manifest": envelopes[0][1]["append_manifest"],
            "append_manifest_sha256": envelopes[0][1]["append_manifest_sha256"],
            "append_launcher": envelopes[0][1]["append_launcher"],
            "append_launcher_sha256": envelopes[0][1]["append_launcher_sha256"],
            "append_submission_sha256": envelopes[0][1]["append_submission_sha256"],
            "n20_job_id": envelopes[0][1]["n20_job_id"],
            "qualified_runtime": str(output_root / "qualified_runtime.json"),
            "qualified_runtime_sha256": envelopes[0][1]["qualified_runtime_sha256"],
            "student_root": str(student_final),
            "official_root": str(official_final),
            "student_seed_results": {
                seed: result
                for _root, manifest in envelopes
                for seed, result in manifest["student_seed_results"].items()
            },
            "official_seed_results": {
                seed: result
                for _root, manifest in envelopes
                for seed, result in manifest["official_seed_results"].items()
            },
            "student_metadata": acceptance.result_metadata_manifest(student_temp),
            "official_metadata": acceptance.result_metadata_manifest(official_temp),
        }
        _atomic_json(temp_root / "PAIRED_N100.json", payload)
        os.rename(temp_root, output_root)
        return payload
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("verify-shard")
    shard.add_argument("--shard-root", type=pathlib.Path, required=True)
    shard.add_argument("--artifact", type=pathlib.Path, required=True)
    shard.add_argument("--start-seed", type=int, required=True)
    shard.add_argument("--output", type=pathlib.Path, required=True)
    shard.add_argument("--done", type=pathlib.Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--shard-roots", required=True, help="Colon-separated exact five shard roots.")
    merge.add_argument("--output-root", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify-shard":
        payload = write_shard_manifest(
            args.shard_root,
            args.artifact,
            start_seed=args.start_seed,
            output=args.output,
            done=args.done,
        )
        print(json.dumps({"verified": True, "start_seed": payload["start_seed"], "trials": payload["trials"]}))
        return
    roots = [pathlib.Path(raw) for raw in args.shard_roots.split(":")]
    payload = merge_paired_shards(roots, args.output_root)
    print(json.dumps({"merged": True, "start_seed": payload["start_seed"], "trials": payload["trials"]}))


if __name__ == "__main__":
    main()
