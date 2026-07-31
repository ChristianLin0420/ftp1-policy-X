#!/usr/bin/env python3
"""Validate FTP-1 dataset configurations and PyTorch checkpoints before a run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import pathlib
import sys
from typing import Any

import numpy as np
import zarr

RGB_KEYS = (
    "camera_main_rgb",
    "camera_ego_rgb",
    "right_wrist_camera_rgb",
    "left_wrist_camera_rgb",
)
STATE_KEYS = (
    "left_wrist_pose",
    "right_wrist_pose",
    "left_arm_joints",
    "right_arm_joints",
    "left_hand_joints",
    "right_hand_joints",
    "supplementary_joints",
)
REQUIRED_KEYS = ("timestamps", "sub_task_instruction")
CHECKPOINT_FILES = ("model.safetensors", "model_config.json", "train_config.json")


@dataclass(frozen=True)
class Finding:
    severity: str
    location: str
    message: str


def _finding(severity: str, location: pathlib.Path | str, message: str) -> Finding:
    return Finding(severity=severity, location=str(location), message=message)


def _sample_indices(length: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.int64)
    return np.unique(np.array([0, length // 2, length - 1], dtype=np.int64))


def _array_is_finite(array: Any) -> bool:
    if not np.issubdtype(array.dtype, np.number):
        return True
    values = np.asarray(array[_sample_indices(array.shape[0])])
    return bool(np.all(np.isfinite(values)))


def _validate_tactile_group(data: Any, key: str, total_steps: int, location: pathlib.Path) -> list[Finding]:
    findings: list[Finding] = []
    prefix, group_name = key.rsplit("_data_", maxsplit=1)
    companion_keys = {
        "area": f"{prefix}_area_{group_name}",
        "sensor": f"{prefix}_sensor_{group_name}",
        "type": f"{prefix}_type_{group_name}",
    }
    tactile = data[key]
    if tactile.ndim < 3:
        findings.append(_finding("error", location, f"{key} must have shape (T, N, ...), got {tactile.shape}"))
    if tactile.shape[0] != total_steps:
        findings.append(_finding("error", location, f"{key} time dimension does not match episode data"))

    for role, companion_key in companion_keys.items():
        if companion_key not in data:
            findings.append(_finding("error", location, f"{key} is missing tactile {role} key {companion_key}"))

    if companion_keys["area"] in data and tactile.ndim >= 2:
        area = data[companion_keys["area"]]
        if area.ndim != 2 or area.shape != tactile.shape[:2]:
            findings.append(
                _finding(
                    "error",
                    location,
                    f"{companion_keys['area']} shape {area.shape} must match tactile (T, N) {tactile.shape[:2]}",
                )
            )

    if companion_keys["type"] in data and total_steps:
        tactile_types = {str(value) for value in np.asarray(data[companion_keys["type"]][_sample_indices(total_steps)])}
        unsupported = tactile_types - {"state", "binary", "image"}
        if unsupported:
            findings.append(_finding("error", location, f"unsupported tactile types for {key}: {sorted(unsupported)}"))
        if "image" in tactile_types and (tactile.ndim != 5 or tactile.shape[-1] != 3):
            findings.append(_finding("error", location, f"image tactile {key} must have shape (T, N, H, W, 3)"))

    if not _array_is_finite(tactile):
        findings.append(_finding("error", location, f"{key} contains NaN or infinite sampled values"))
    return findings


def validate_zarr(zarr_path: pathlib.Path) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    summary: dict[str, Any] = {"path": str(zarr_path), "episodes": 0, "steps": 0, "keys": []}
    try:
        root = zarr.open_group(store=str(zarr_path), mode="r")
    except Exception as exc:
        return [_finding("error", zarr_path, f"cannot open Zarr: {exc}")], summary

    if "data" not in root or "meta" not in root:
        return [_finding("error", zarr_path, "expected data/ and meta/episode_ends")], summary
    if "episode_ends" not in root["meta"]:
        return [_finding("error", zarr_path, "expected data/ and meta/episode_ends")], summary

    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    keys = sorted(data.keys())
    summary["keys"] = keys
    summary["episodes"] = int(episode_ends.size)

    if episode_ends.size == 0:
        findings.append(_finding("error", zarr_path, "meta/episode_ends is empty"))
        return findings, summary
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0):
        findings.append(_finding("error", zarr_path, "meta/episode_ends must be positive and strictly increasing"))
    total_steps = int(episode_ends[-1])
    summary["steps"] = total_steps

    findings.extend(
        _finding("error", zarr_path, f"missing required key data/{key}") for key in REQUIRED_KEYS if key not in data
    )
    if not any(key in data for key in RGB_KEYS):
        findings.append(_finding("error", zarr_path, f"missing supported RGB key; expected one of {RGB_KEYS}"))
    if not any(key in data for key in STATE_KEYS):
        findings.append(_finding("error", zarr_path, "missing embodiment state needed to derive actions"))

    for key in keys:
        array = data[key]
        if array.ndim == 0:
            findings.append(_finding("error", zarr_path, f"data/{key} must be time-major, got scalar"))
            continue
        if array.shape[0] != total_steps:
            findings.append(
                _finding("error", zarr_path, f"data/{key} length {array.shape[0]} does not match {total_steps} steps")
            )
        if not _array_is_finite(array):
            findings.append(_finding("error", zarr_path, f"data/{key} contains NaN or infinite sampled values"))

    for key in RGB_KEYS:
        if key not in data:
            continue
        image = data[key]
        if image.ndim != 4 or image.shape[-1] != 3:
            findings.append(_finding("error", zarr_path, f"data/{key} must have shape (T, H, W, 3), got {image.shape}"))

    for side in ("left", "right"):
        joints_key = f"{side}_hand_joints"
        indices_key = f"{side}_hand_joints_idx"
        if joints_key in data and indices_key not in data:
            findings.append(_finding("error", zarr_path, f"data/{joints_key} requires data/{indices_key}"))
        if joints_key in data and indices_key in data:
            joints = data[joints_key]
            indices = data[indices_key]
            if joints.shape != indices.shape:
                findings.append(_finding("error", zarr_path, f"{joints_key} and {indices_key} shapes must match"))
            sampled_indices = np.asarray(indices[_sample_indices(total_steps)])
            if np.any((sampled_indices < 0) | (sampled_indices >= 32)):
                findings.append(_finding("error", zarr_path, f"{indices_key} contains values outside [0, 31]"))

    tactile_keys = [key for key in keys if "_tactile_data_" in key]
    for key in tactile_keys:
        findings.extend(_validate_tactile_group(data, key, total_steps, zarr_path))
    if not tactile_keys:
        findings.append(_finding("warning", zarr_path, "no tactile stream found; valid only for a non-tactile checkpoint"))

    return findings, summary


def _find_zarrs(dataset_path: pathlib.Path) -> list[pathlib.Path]:
    if dataset_path.name.endswith(".zarr") and dataset_path.is_dir():
        return [dataset_path]
    if not dataset_path.is_dir():
        return []
    return sorted(path for path in dataset_path.iterdir() if path.is_dir() and path.name.endswith(".zarr"))


def validate_dataset_config(config_path: pathlib.Path, max_zarr: int | None = None) -> tuple[list[Finding], list[dict[str, Any]]]:
    findings: list[Finding] = []
    summaries: list[dict[str, Any]] = []
    try:
        config = json.loads(config_path.read_text())
    except Exception as exc:
        return [_finding("error", config_path, f"cannot parse dataset config: {exc}")], summaries

    datasets = [entry for entry in config.get("datasets", []) if entry.get("enabled", True)]
    if not datasets:
        return [_finding("error", config_path, "no enabled datasets")], summaries

    seen_names: set[str] = set()
    for entry in datasets:
        name = str(entry.get("name", "")).strip()
        if not name:
            findings.append(_finding("error", config_path, "enabled dataset is missing a name"))
        elif name in seen_names:
            findings.append(_finding("error", config_path, f"duplicate enabled dataset name {name!r}"))
        seen_names.add(name)

        ratio = float(entry.get("use_trajectory_ratio", config.get("default_use_trajectory_ratio", 1.0)))
        if not 0.0 < ratio <= 1.0:
            findings.append(_finding("error", config_path, f"{name or '<unnamed>'} has invalid trajectory ratio {ratio}"))

        raw_path = entry.get("path")
        if not raw_path:
            findings.append(_finding("error", config_path, f"{name or '<unnamed>'} is missing path"))
            continue
        dataset_path = pathlib.Path(raw_path).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (config_path.parent / dataset_path).resolve()
        zarr_paths = _find_zarrs(dataset_path)
        if not zarr_paths:
            findings.append(_finding("error", dataset_path, "no immediate *.zarr directories found"))
            continue
        if max_zarr is not None:
            zarr_paths = zarr_paths[:max_zarr]
        for zarr_path in zarr_paths:
            zarr_findings, summary = validate_zarr(zarr_path)
            findings.extend(zarr_findings)
            summary["domain"] = name
            summaries.append(summary)
    return findings, summaries


def _resolve_checkpoint(checkpoint_path: pathlib.Path) -> pathlib.Path:
    if (checkpoint_path / "model.safetensors").is_file():
        return checkpoint_path
    step_dirs = [path for path in checkpoint_path.iterdir() if path.is_dir() and path.name.isdigit()]
    if not step_dirs:
        return checkpoint_path
    return max(step_dirs, key=lambda path: int(path.name))


def validate_checkpoint(checkpoint_path: pathlib.Path, domain_name: str | None = None) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    summary: dict[str, Any] = {"path": str(checkpoint_path)}
    if not checkpoint_path.is_dir():
        return [_finding("error", checkpoint_path, "checkpoint directory does not exist")], summary
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    summary["resolved_path"] = str(checkpoint_path)
    findings.extend(
        _finding("error", checkpoint_path, f"missing {filename}")
        for filename in CHECKPOINT_FILES
        if not (checkpoint_path / filename).is_file()
    )

    model_config: dict[str, Any] = {}
    model_config_path = checkpoint_path / "model_config.json"
    if model_config_path.is_file():
        try:
            model_config = json.loads(model_config_path.read_text())
        except Exception as exc:
            findings.append(_finding("error", model_config_path, f"cannot parse model config: {exc}"))
    use_tactile = bool(model_config.get("use_tactile_input", True))
    summary["use_tactile_input"] = use_tactile
    summary["action_dim"] = model_config.get("action_dim")
    summary["action_horizon"] = model_config.get("action_horizon")
    if use_tactile and not (checkpoint_path / "hpt_tokenizer").is_dir():
        findings.append(_finding("error", checkpoint_path, "tactile checkpoint is missing hpt_tokenizer/"))

    normalization = checkpoint_path / "normalization"
    if not normalization.is_dir():
        findings.append(_finding("error", checkpoint_path, "missing normalization/"))
    elif domain_name and not (normalization / domain_name).is_dir():
        findings.append(_finding("error", normalization, f"missing normalization domain {domain_name!r}"))
    return findings, summary


def _print_human(findings: list[Finding], dataset_summaries: list[dict[str, Any]], checkpoint: dict[str, Any] | None) -> None:
    for summary in dataset_summaries:
        print(
            f"[OK] {summary['domain']}: {summary['path']} "
            f"({summary['episodes']} episodes, {summary['steps']} steps, {len(summary['keys'])} keys)"
        )
    if checkpoint is not None:
        print(f"[OK] checkpoint: {checkpoint.get('resolved_path', checkpoint['path'])}")
    for finding in findings:
        print(f"[{finding.severity.upper()}] {finding.location}: {finding.message}")
    errors = sum(finding.severity == "error" for finding in findings)
    warnings = sum(finding.severity == "warning" for finding in findings)
    print(f"Preflight complete: {errors} error(s), {warnings} warning(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--domain-name")
    parser.add_argument("--max-zarr", type=int, default=None, help="Validate at most N Zarr stores per domain")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    args = parser.parse_args(argv)
    if args.dataset_config is None and args.checkpoint is None:
        parser.error("at least one of --dataset-config or --checkpoint is required")

    findings: list[Finding] = []
    dataset_summaries: list[dict[str, Any]] = []
    checkpoint_summary: dict[str, Any] | None = None
    if args.dataset_config is not None:
        dataset_findings, dataset_summaries = validate_dataset_config(args.dataset_config, args.max_zarr)
        findings.extend(dataset_findings)
    if args.checkpoint is not None:
        checkpoint_findings, checkpoint_summary = validate_checkpoint(args.checkpoint, args.domain_name)
        findings.extend(checkpoint_findings)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not any(finding.severity == "error" for finding in findings),
                    "findings": [asdict(finding) for finding in findings],
                    "datasets": dataset_summaries,
                    "checkpoint": checkpoint_summary,
                },
                indent=2,
            )
        )
    else:
        _print_human(findings, dataset_summaries, checkpoint_summary)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
