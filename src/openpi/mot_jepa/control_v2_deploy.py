"""Strict deployment closure and stateful inference for MoT-Control V2.

The legacy MoT-JEPA deployment path predicts a generic 120-slot first-difference chunk.  V2 is a
task-specific controller with a deliberately different contract: dense video/GEL tokens, a
normalized 16-frame robot-state history, and a 32-row mixed eight-dimensional action chunk.  This
module keeps that contract fail-closed from checkpoint loading through the command sent to UniVTAC.

There is no FTP-1 teacher, fallback policy, or open-loop action cache here.  The controller replans
on every control step.  Its only fallback is a finite hold for a non-finite model output or a
transient non-finite observation after an episode has already established a safe command.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import glob
import hashlib
import json
import pathlib
from typing import Any

import cv2
import numpy as np
import torch

from openpi.mot_jepa.control_v2 import ACTIVE_ACTION_DIM
from openpi.mot_jepa.control_v2 import ARM_DIM
from openpi.mot_jepa.control_v2 import ControlV2
from openpi.mot_jepa.control_v2_config import CONTROL_V2_ARTIFACT_SCHEMA
from openpi.mot_jepa.control_v2_config import CONTROL_V2_PRODUCTION_EPISODES
from openpi.mot_jepa.control_v2_config import ControlV2Artifact
from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.control_v2_data import ControlV2Normalizer
from openpi.mot_jepa.control_v2_data import build_state_features
from openpi.mot_jepa.control_v2_data import load_normalization_artifact
from openpi.mot_jepa.control_v2_data import scatter_qpos8
from openpi.mot_jepa.control_v2_dataset import CONTROL_V2_SPLIT_IDENTITY
from openpi.mot_jepa.control_v2_runtime import FIRST_EXECUTABLE_INDEX
from openpi.mot_jepa.control_v2_runtime import V2_HORIZON
from openpi.mot_jepa.control_v2_runtime import SafetyLimits
from openpi.mot_jepa.control_v2_runtime import TemporalEnsemblerV2
from openpi.mot_jepa.control_v2_runtime import safe_hold_chunk
from openpi.mot_jepa.model import ClipInputs
from openpi.mot_jepa.model import MotJepaBackbone

CHECKPOINT_SCHEMA = CONTROL_V2_ARTIFACT_SCHEMA


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_completed_step(run: pathlib.Path, config: ControlV2TrainConfig) -> int:
    done_path = run / "DONE"
    latest_path = run / "checkpoints" / "latest"
    if not done_path.is_file() or not latest_path.is_file():
        raise RuntimeError(f"{run} is incomplete: require both DONE and checkpoints/latest")
    try:
        done = int(done_path.read_text().strip())
        latest = int(latest_path.read_text().strip())
    except ValueError as error:
        raise ValueError("DONE and checkpoints/latest must each contain one integer step") from error
    if done != config.num_train_steps or latest != done:
        raise ValueError(f"completion mismatch: DONE={done}, latest={latest}, configured={config.num_train_steps}")
    return done


_BEST_VALIDATION_FIELDS = {
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

_VALIDATION_SAMPLING = "all_episode_time_phase_contact_cold_start_stratified_v2"
_VALIDATION_TIME_BINS = ("early", "middle", "late")
_VALIDATION_HISTORY_STEPS = 16
_VALIDATION_PREFIX_SCENARIOS = {
    (length, oldest_present) for length in range(1, _VALIDATION_HISTORY_STEPS + 1) for oldest_present in (0, 1)
}
_SELECTION_CONSTRAINT = "validation_safety_violation_values==0"
_SAFETY_COUNT_FIELDS = (
    "validation_joint_violation_values",
    "validation_rate_violation_values",
    "validation_safety_violation_values",
)


def _load_validation_manifest(run: pathlib.Path) -> dict[str, Any]:
    path = run / "VALIDATION_SAMPLES.json"
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("VALIDATION_SAMPLES.json must contain valid JSON") from error
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
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("VALIDATION_SAMPLES.json has an invalid field set")
    if manifest["schema_version"] != 1 or manifest["sampling"] != _VALIDATION_SAMPLING:
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
        if episode_idx < 0 or start < 0 or stride < 1 or not 1 <= history_length <= _VALIDATION_HISTORY_STEPS:
            raise ValueError("VALIDATION_SAMPLES.json structured identity has out-of-range indices")
        if embedded_dataset_index != item["dataset_index"]:
            raise ValueError("VALIDATION_SAMPLES.json dataset index differs from its immutable identity")
        if fields["time_bin"] not in _VALIDATION_TIME_BINS:
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
    if len(samples) < len(_VALIDATION_PREFIX_SCENARIOS) or not prefix_scenarios >= _VALIDATION_PREFIX_SCENARIOS:
        raise ValueError("VALIDATION_SAMPLES.json does not cover the fixed V3 cold-start scenarios")
    required_strata = {
        "episode-time": (set(_VALIDATION_TIME_BINS), observed_time_bins),
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
    expected = {
        "sample_count": len(samples),
        "episode_count": len(episodes),
        "sample_sha256": digest,
    }
    _require_metadata(manifest, expected)
    if len(samples) < len(episodes):
        raise ValueError("VALIDATION_SAMPLES.json does not cover every declared episode")
    return manifest


def _selected_validation_checkpoint(
    run: pathlib.Path,
    config: ControlV2TrainConfig,
    *,
    completed_step: int,
    requested_step: int | None,
    artifact_sha256: str,
    normalization_sha256: str,
    source_store_sha256: str,
) -> tuple[int, pathlib.Path, dict[str, Any]]:
    marker_path = run / "BEST_VALIDATION.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"{marker_path} missing; production deployment requires held-out selection")
    try:
        marker = json.loads(marker_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("BEST_VALIDATION.json must contain valid JSON") from error
    if not isinstance(marker, dict) or set(marker) != _BEST_VALIDATION_FIELDS:
        actual = sorted(marker) if isinstance(marker, dict) else type(marker).__name__
        raise ValueError(
            f"BEST_VALIDATION.json fields {actual!r} differ from the schema-{CONTROL_V2_ARTIFACT_SCHEMA} contract"
        )
    manifest = _load_validation_manifest(run)
    expected = {
        "schema_version": CONTROL_V2_ARTIFACT_SCHEMA,
        "selection_metric": "validation_action_loss",
        "selection_constraint": _SELECTION_CONSTRAINT,
        "selection_mode": "min",
        "tie_break": "earliest_step",
        "artifact_sha256": artifact_sha256,
        "normalization_sha256": normalization_sha256,
        "source_store_sha256": source_store_sha256,
        "validation_sampling": manifest["sampling"],
        "validation_sample_sha256": manifest["sample_sha256"],
        "validation_episode_count": float(manifest["episode_count"]),
        "validation_examples": float(manifest["sample_count"]),
    }
    _require_metadata(marker, expected)
    selected_step = marker["step"]
    if not isinstance(selected_step, int) or isinstance(selected_step, bool):
        raise TypeError("BEST_VALIDATION step must be an integer")
    if not 0 <= selected_step <= completed_step:
        raise ValueError(
            f"selected validation step {selected_step} is outside completed training [0, {completed_step}]"
        )
    if selected_step != completed_step and selected_step % config.validation_interval:
        raise ValueError(
            f"selected step {selected_step} is neither final nor aligned to validation interval "
            f"{config.validation_interval}"
        )
    if requested_step is not None:
        if not isinstance(requested_step, int) or isinstance(requested_step, bool):
            raise TypeError("requested step must be an integer")
        if requested_step != selected_step:
            raise ValueError(f"requested step {requested_step} is not held-out-selected step {selected_step}")

    checkpoint = run / "checkpoints" / f"best_validation_{selected_step}"
    marker_checkpoint = pathlib.Path(marker["checkpoint"])
    if not marker_checkpoint.is_absolute() or marker_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError(f"BEST_VALIDATION checkpoint {marker_checkpoint} != durable selection {checkpoint.resolve()}")
    numeric_fields = _BEST_VALIDATION_FIELDS - {
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
        "validation_sampling",
        "validation_sample_sha256",
    }
    for name in numeric_fields:
        value = marker[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"BEST_VALIDATION {name} must be finite numeric data")
        if value < 0:
            raise ValueError(f"BEST_VALIDATION {name} must be non-negative")
    if (
        marker["validation_examples"] <= 0
        or marker["validation_episode_count"] <= 0
        or marker["validation_contact_count"] <= 0
    ):
        raise ValueError("BEST_VALIDATION must cover positive validation and contact example counts")
    if any(marker[name] != 0 for name in _SAFETY_COUNT_FIELDS):
        counts = {name: marker[name] for name in _SAFETY_COUNT_FIELDS}
        raise ValueError(f"BEST_VALIDATION violates its zero held-out safety constraint: {counts}")
    if marker["validation_joint_max_excess_ratio"] != 0:
        raise ValueError("BEST_VALIDATION validation_joint_max_excess_ratio must be zero")
    if marker["validation_rate_max_ratio"] > 1:
        raise ValueError("BEST_VALIDATION validation_rate_max_ratio must be at most 1")
    return selected_step, checkpoint, marker


def source_store_contract_digest(stores: tuple[str, ...]) -> str:
    """Recompute the same non-image data digest written by the V2 statistics job."""
    import zarr  # noqa: PLC0415

    digest = hashlib.sha256()
    for store_idx, raw_path in enumerate(stores):
        path = pathlib.Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"source Zarr store is unavailable: {path}")
        group = zarr.open_group(str(path), mode="r")
        digest.update(f"store={store_idx}".encode())
        for section in ("data", "meta"):
            if section not in group:
                raise ValueError(f"source store {path} lacks {section!r}")
            for name in sorted(group[section].array_keys()):
                array = group[section][name]
                digest.update(f"{section}/{name}:{array.shape}:{array.dtype}".encode())
        digest.update(np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64).tobytes())
        if "source_episode_seed" in set(group["meta"].array_keys()):
            digest.update(np.asarray(group["meta"]["source_episode_seed"][:], dtype=np.int64).tobytes())
        arrays = set(group["data"].array_keys())
        for name in (
            "state",
            "command8",
            "command_valid",
            "phase_id",
            "contact",
            "contact_valid",
            "control_step",
            "control_step_valid",
        ):
            if name in arrays:
                digest.update(np.asarray(group["data"][name][:]).tobytes())
    return digest.hexdigest()


def _normalization_counts(normalizer: ControlV2Normalizer) -> dict[str, int]:
    return {
        "qpos": int(normalizer.qpos_count),
        "velocity": int(normalizer.velocity_count),
        "previous_command": int(normalizer.previous_command_count),
    }


def _require_metadata(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    missing = sorted(set(expected) - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata lacks required fields {missing}")
    for name, value in expected.items():
        if metadata[name] != value:
            raise ValueError(f"checkpoint metadata {name}={metadata[name]!r}, expected {value!r}")


@dataclasses.dataclass(frozen=True)
class TrainedControlV2Artifacts:
    """One complete, immutable V2 policy selected for deployment."""

    config: ControlV2TrainConfig
    artifact: ControlV2Artifact
    backbone: MotJepaBackbone
    head: ControlV2
    normalizer: ControlV2Normalizer
    step: int
    completed_step: int
    selection_metric: str
    selection_value: float
    checkpoint: pathlib.Path
    metadata: Mapping[str, Any]


def load_trained_control_v2_artifacts(
    run: str | pathlib.Path,
    device: str | torch.device,
    *,
    step: int | None = None,
    verify_source_stores: bool = True,
) -> TrainedControlV2Artifacts:
    """Strictly restore the exact completed V2 backbone, controller, and normalizer.

    ``verify_source_stores=False`` exists only for hermetic unit tests and artifact inspection.
    Closed-loop deployment uses the default and therefore refuses a store whose non-image control
    arrays or episode boundaries changed after the statistics artifact was fitted.
    """
    run = pathlib.Path(run)
    required_run_files = (
        "run_config.json",
        "artifact.json",
        "normalization.json",
        "STATS_DONE",
        "VALIDATION_SAMPLES.json",
    )
    missing = [name for name in required_run_files if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run} lacks required V2 run artifacts {missing}")

    config_text = (run / "run_config.json").read_text()
    config = ControlV2TrainConfig.from_json(config_text)
    artifact_path = run / "artifact.json"
    normalization_path = run / "normalization.json"
    artifact = ControlV2Artifact.from_json(artifact_path.read_text())
    if artifact.schema_version != CONTROL_V2_ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported artifact schema {artifact.schema_version}")
    artifact.require_production_data()
    if not config.require_authoritative_control or config.minimum_total_episodes != CONTROL_V2_PRODUCTION_EPISODES:
        raise ValueError("deployable V2 run config must require authoritative control and exactly 1000 episodes")
    if artifact.task != config.task:
        raise ValueError(f"artifact task {artifact.task!r} differs from config task {config.task!r}")
    if (artifact.observation_stride, artifact.action_stride, artifact.control_step_stride) != (
        config.observation_stride,
        config.action_stride,
        1,
    ):
        raise ValueError("artifact and run config observation/action/control-step cadences differ")
    try:
        stats_done = json.loads((run / "STATS_DONE").read_text())
    except json.JSONDecodeError as error:
        raise ValueError("STATS_DONE must be a valid JSON closure marker") from error
    validation_manifest = _load_validation_manifest(run)
    expected_stats_fields = {
        "schema",
        "split_identity",
        "train_episodes",
        "validation_episodes",
        "source_store_sha256",
        "validation_sampling",
        "validation_sample_sha256",
        "validation_examples",
        "validation_episode_count",
        "validation_oracle_safety_violation_values",
        "validation_oracle_min_slack",
        "validation_oracle_max_rate_ratio",
    }
    if not isinstance(stats_done, dict) or set(stats_done) != expected_stats_fields:
        raise ValueError("STATS_DONE has an invalid field set")
    expected_stats_done = {
        "schema": CONTROL_V2_ARTIFACT_SCHEMA,
        "split_identity": CONTROL_V2_SPLIT_IDENTITY,
        "train_episodes": artifact.train_episodes,
        "validation_episodes": artifact.validation_episodes,
        "source_store_sha256": artifact.source_store_sha256,
        "validation_sampling": validation_manifest["sampling"],
        "validation_sample_sha256": validation_manifest["sample_sha256"],
        "validation_examples": validation_manifest["sample_count"],
        "validation_episode_count": validation_manifest["episode_count"],
        "validation_oracle_safety_violation_values": 0,
    }
    stats_mismatches = {
        name: (stats_done.get(name), value)
        for name, value in expected_stats_done.items()
        if stats_done.get(name) != value
    }
    min_slack = stats_done.get("validation_oracle_min_slack")
    max_rate_ratio = stats_done.get("validation_oracle_max_rate_ratio")
    if (
        isinstance(min_slack, bool)
        or not isinstance(min_slack, (int, float))
        or not np.isfinite(float(min_slack))
        or float(min_slack) <= 0
    ):
        stats_mismatches["validation_oracle_min_slack"] = (min_slack, "finite and > 0")
    if (
        isinstance(max_rate_ratio, bool)
        or not isinstance(max_rate_ratio, (int, float))
        or not np.isfinite(float(max_rate_ratio))
        or not 0 <= float(max_rate_ratio) < 1
    ):
        stats_mismatches["validation_oracle_max_rate_ratio"] = (max_rate_ratio, "finite and in [0, 1)")
    if stats_mismatches:
        raise ValueError(f"STATS_DONE does not close artifact.json: {stats_mismatches}")

    configured_stores = tuple(str(pathlib.Path(path).resolve()) for path in sorted(glob.glob(config.store_glob)))
    if (verify_source_stores or configured_stores) and configured_stores != artifact.source_stores:
        raise ValueError(
            f"store_glob currently resolves to {configured_stores}, not artifact sources {artifact.source_stores}"
        )
    if verify_source_stores:
        source_digest = source_store_contract_digest(artifact.source_stores)
        if source_digest != artifact.source_store_sha256:
            raise ValueError(
                f"source store digest {source_digest} differs from artifact {artifact.source_store_sha256}"
            )

    completed_step = _read_completed_step(run, config)
    artifact_sha256 = _sha256(artifact_path)
    normalization_sha256 = _sha256(normalization_path)
    selected_step, checkpoint, selection = _selected_validation_checkpoint(
        run,
        config,
        completed_step=completed_step,
        requested_step=step,
        artifact_sha256=artifact_sha256,
        normalization_sha256=normalization_sha256,
        source_store_sha256=artifact.source_store_sha256,
    )
    required_checkpoint_files = (
        "student.pt",
        "backbone.pt",
        "loss.pt",
        "optimizer.pt",
        "metadata.pt",
        "train_config.json",
    )
    missing = [name for name in required_checkpoint_files if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{checkpoint} lacks selected V2 checkpoint artifacts {missing}")
    if json.loads((checkpoint / "train_config.json").read_text()) != json.loads(config_text):
        raise ValueError("selected checkpoint train_config.json differs from run_config.json")

    metadata = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    normalizer = ControlV2Normalizer(load_normalization_artifact(normalization_path))
    expected_counts = _normalization_counts(normalizer)
    expected_metadata = {
        "global_step": selected_step,
        "control_v2_checkpoint_schema": CHECKPOINT_SCHEMA,
        "backbone_train_mode": config.backbone_train_mode,
        "backbone_last_n_blocks": config.backbone_last_n_blocks,
        "source_backbone_run": str(pathlib.Path(config.pretrained_run).resolve()),
        "source_backbone_step": config.pretrained_step,
        "artifact_sha256": artifact_sha256,
        "normalization_sha256": normalization_sha256,
        "source_store_sha256": artifact.source_store_sha256,
        "normalization_counts": expected_counts,
        "best_validation_step": selected_step,
        "best_validation_chunk_loss": float(selection["validation_action_loss"]),
    }
    _require_metadata(metadata, expected_metadata)

    # DONE/latest prove the entire preregistered schedule ran, independently of which earlier
    # checkpoint held-out validation selected.  Validate that numeric final checkpoint instead of
    # trusting two text markers that could point at an incomplete/pruned directory.
    final_checkpoint = run / "checkpoints" / str(completed_step)
    missing_final = [name for name in required_checkpoint_files if not (final_checkpoint / name).is_file()]
    if missing_final:
        raise FileNotFoundError(f"completed checkpoint {final_checkpoint} lacks {missing_final}")
    if json.loads((final_checkpoint / "train_config.json").read_text()) != json.loads(config_text):
        raise ValueError("completed checkpoint train_config.json differs from run_config.json")
    final_metadata = torch.load(final_checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
    if not isinstance(final_metadata, Mapping):
        raise TypeError("completed checkpoint metadata must be a mapping")
    _require_metadata(
        final_metadata,
        {
            "global_step": completed_step,
            "control_v2_checkpoint_schema": CHECKPOINT_SCHEMA,
            "artifact_sha256": artifact_sha256,
            "normalization_sha256": normalization_sha256,
            "source_store_sha256": artifact.source_store_sha256,
            "normalization_counts": expected_counts,
        },
    )

    dev = torch.device(device)
    backbone = MotJepaBackbone(
        config.layout,
        config.encoder,
        lowdim_channels=config.data.lowdim_channels,
        lowdim_log_compress=config.data.lowdim_log_compress,
    ).to(dev)
    head = ControlV2(config.head, config.layout).to(dev)
    expected_normalizer_state = {name: value.clone() for name, value in normalizer.state_dict().items()}
    normalizer = normalizer.to(dev)

    backbone_state = torch.load(checkpoint / "backbone.pt", map_location=dev, weights_only=True)
    head_state = torch.load(checkpoint / "student.pt", map_location=dev, weights_only=True)
    normalizer_state = torch.load(checkpoint / "loss.pt", map_location=dev, weights_only=True)
    backbone.load_state_dict(backbone_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    normalizer.load_state_dict(normalizer_state, strict=True)

    for name, expected in expected_normalizer_state.items():
        actual = normalizer.state_dict()[name].detach().cpu()
        if not torch.equal(actual, expected):
            raise ValueError(f"checkpoint normalizer {name} differs from normalization.json")
    expected_scale = torch.tensor(artifact.action_scale, dtype=torch.float32)
    if not torch.equal(head.action_scale.detach().cpu(), expected_scale):
        raise ValueError("checkpoint action_scale differs from artifact.json")

    backbone.eval().requires_grad_(requires_grad=False)
    head.eval().requires_grad_(requires_grad=False)
    normalizer.eval().requires_grad_(requires_grad=False)
    return TrainedControlV2Artifacts(
        config=config,
        artifact=artifact,
        backbone=backbone,
        head=head,
        normalizer=normalizer,
        step=selected_step,
        completed_step=completed_step,
        selection_metric=str(selection["selection_metric"]),
        selection_value=float(selection["validation_action_loss"]),
        checkpoint=checkpoint,
        metadata=dict(metadata),
    )


@dataclasses.dataclass(frozen=True)
class _ObservationFrame:
    video: np.ndarray
    gel: np.ndarray
    qpos8: np.ndarray
    previous_command: np.ndarray
    command_present: bool


def _to_uint8(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty HWC RGB array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite pixels")
    if array.dtype != np.uint8:
        multiplier = 255.0 if float(array.max()) <= 1.0 else 1.0
        array = np.clip(array * multiplier, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _qpos8(observation: Mapping[str, Any], *, finite: bool) -> np.ndarray:
    try:
        value = observation["embodiment"]["joint"][:ACTIVE_ACTION_DIM]
    except (KeyError, TypeError) as error:
        raise KeyError("observation must contain embodiment/joint with at least eight values") from error
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (ACTIVE_ACTION_DIM,):
        raise ValueError(f"observation qpos must have shape ({ACTIVE_ACTION_DIM},), got {result.shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError("observation qpos contains non-finite values")
    return result.copy()


class MotJepaControlV2Policy:
    """Replanning V2 policy with exact cold-start, sampling, and execution semantics."""

    def __init__(
        self,
        artifacts: TrainedControlV2Artifacts,
        *,
        device: str | torch.device,
        save_infer_input_dir: str | pathlib.Path | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.config = artifacts.config
        self.artifact = artifacts.artifact
        self.backbone = artifacts.backbone
        self.head = artifacts.head
        self.normalizer = artifacts.normalizer
        self.layout = artifacts.config.layout
        self.device = torch.device(device)
        if self.layout.num_frames != self.config.head.qpos_history:
            raise ValueError("visual and state history lengths must match")

        # Attributes consumed by UniVTAC/scripts/eval_ftp1.py.  The retained raw chunk is native
        # mixed 8-D data; act() itself returns an already-resolved absolute qpos8 command.
        self.action_rep = "absolute"
        self.chunk_action_rep = "mix"
        self.first_executable_index = FIRST_EXECUTABLE_INDEX
        self.action_dim = ACTIVE_ACTION_DIM
        self.model_action_dim = ACTIVE_ACTION_DIM
        self.state_dim = ACTIVE_ACTION_DIM
        self.use_temporal_ensemble = True
        self.mapping: Any = None
        self.domain_names = (self.config.task,)
        self.save_infer_input_dir = pathlib.Path(save_infer_input_dir) if save_infer_input_dir else None
        self._saved_infer_input = False
        self._task = self.config.task
        self._episode_seed: int | None = None
        self._reset_state()

    def _new_ensembler(self) -> TemporalEnsemblerV2:
        limits = SafetyLimits(
            joint_lower=self.artifact.joint_lower,
            joint_upper=self.artifact.joint_upper,
            max_delta=self.artifact.max_command_delta,
        )
        return TemporalEnsemblerV2(
            limits,
            first_n=self.artifact.chunk_first_n,
            ensemble_k=self.artifact.temporal_ensemble_k,
        )

    def _reset_state(self) -> None:
        self._frames: list[_ObservationFrame] = []
        self._last_command: np.ndarray | None = None
        self._chunk_history: list[tuple[np.ndarray, int, np.ndarray]] = []
        self._last_debug: dict[str, Any] | None = None
        self._ensembler = self._new_ensembler()

    def reset(self, seed: int | None = None) -> None:
        self._reset_state()
        self._episode_seed = seed

    def set_task(self, task_name: str) -> None:
        canonical = str(task_name).replace("UniVTAC_", "")
        if canonical != self.config.task:
            raise KeyError(f"V2 run is task-specific to {self.config.task!r}, not {task_name!r}")
        self._task = canonical

    def get_last_action_debug(self) -> dict[str, Any] | None:
        return self._last_debug

    def get_safety_counters(self) -> dict[str, int]:
        return dataclasses.asdict(self._ensembler.counters)

    @staticmethod
    def _resize(image: np.ndarray, size: int, *, area: bool = False) -> np.ndarray:
        if image.shape[:2] == (size, size):
            return image
        interpolation = cv2.INTER_AREA if area else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interpolation)

    @staticmethod
    def _jpeg_roundtrip(image: Any, *, name: str) -> np.ndarray:
        """Reproduce the loss introduced by UniVTAC's HDF5 image writer."""

        source = _to_uint8(image, name=name)
        success, encoded = cv2.imencode(".jpg", source)
        if not success:
            raise ValueError(f"failed to JPEG-encode {name}")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != source.shape or decoded.dtype != np.uint8:
            shape = None if decoded is None else decoded.shape
            raise ValueError(f"failed to JPEG-decode {name} with shape {source.shape}; got {shape}")
        return np.ascontiguousarray(decoded)

    def _prepare_video(self, image: Any, *, name: str) -> np.ndarray:
        return self._resize(self._jpeg_roundtrip(image, name=name), self.config.video_size)

    def _prepare_gel(self, image: Any, *, name: str) -> np.ndarray:
        # Offline V3 preprocessing first JPEG-decodes and resizes every HDF5 image to 224 with
        # bilinear interpolation, then the derived-store builder downsizes GEL from 224 to 112
        # with area interpolation. Reproduce the lossy storage and both resize stages.
        image = self._resize(self._jpeg_roundtrip(image, name=name), self.config.video_size)
        return self._resize(image, self.layout.gel_size, area=True)

    def _observe(self, observation: Mapping[str, Any], qpos8: np.ndarray) -> None:
        try:
            head = observation["observation"]["head"]["rgb"]
            tactile = observation["tactile"]
        except (KeyError, TypeError) as error:
            raise KeyError("V2 requires observation/head/rgb and two tactile marker images") from error
        left_key = "left_gsmini" if "left_gsmini" in tactile else "left_tactile"
        right_key = "right_gsmini" if "right_gsmini" in tactile else "right_tactile"
        try:
            left = tactile[left_key]["rgb_marker"]
            right = tactile[right_key]["rgb_marker"]
        except (KeyError, TypeError) as error:
            raise KeyError("V2 requires left and right tactile rgb_marker observations") from error

        command_present = self._last_command is not None
        previous = np.zeros(ACTIVE_ACTION_DIM, dtype=np.float32)
        if command_present:
            previous = np.asarray(self._last_command, dtype=np.float32).copy()
        frame = _ObservationFrame(
            video=self._prepare_video(head, name="head rgb"),
            gel=np.stack(
                (
                    self._prepare_gel(left, name="left tactile rgb_marker"),
                    self._prepare_gel(right, name="right tactile rgb_marker"),
                )
            ),
            qpos8=qpos8.copy(),
            previous_command=previous,
            command_present=command_present,
        )
        self._frames.append(frame)
        raw_history = (self.layout.num_frames - 1) * self.config.observation_stride + 1
        if len(self._frames) > raw_history:
            self._frames.pop(0)

    def _model_inputs(self) -> tuple[ClipInputs, torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
        if not self._frames:
            raise RuntimeError("an observation is required before inference")
        real = self._frames[::-1][:: self.config.observation_stride][::-1]
        real = real[-self.layout.num_frames :]
        pad_count = self.layout.num_frames - len(real)
        window = [real[0]] * pad_count + real
        valid = torch.tensor([False] * pad_count + [True] * len(real), dtype=torch.bool, device=self.device)
        command_present = torch.tensor(
            [False] * pad_count + [frame.command_present for frame in real],
            dtype=torch.bool,
            device=self.device,
        )

        video_np = np.stack([frame.video for frame in window])
        gel_np = np.stack([frame.gel for frame in window])
        qpos_np = np.stack([frame.qpos8 for frame in window]).astype(np.float32, copy=False)
        command_np = np.stack([frame.previous_command for frame in window]).astype(np.float32, copy=False)

        def prepare(array: np.ndarray, axes: tuple[int, ...]) -> torch.Tensor:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
            return tensor.permute(*axes).float().div_(127.5).sub_(1.0).unsqueeze(0)

        clip = ClipInputs(
            video=prepare(video_np, (0, 3, 1, 2)),
            gel=prepare(gel_np, (0, 1, 4, 2, 3)),
            lowdim=torch.zeros(
                1,
                self.layout.num_frames,
                self.layout.lowdim_slots,
                self.config.lowdim_channels,
                device=self.device,
            ),
        )
        qpos = torch.from_numpy(qpos_np).to(self.device)
        state120 = scatter_qpos8(qpos).unsqueeze(0)
        command = torch.from_numpy(command_np).to(self.device).unsqueeze(0)
        features = build_state_features(
            state120,
            history_valid=valid.unsqueeze(0),
            previous_command=command,
            previous_command_present=command_present.unsqueeze(0),
            normalizer=self.normalizer,
            dt=float(self.config.observation_stride),
        ).tensor
        current = qpos[-1:].contiguous()
        exact = {
            "video": clip.video.detach().cpu().numpy(),
            "gel": clip.gel.detach().cpu().numpy(),
            "lowdim": clip.lowdim.detach().cpu().numpy(),
            "state_features": features.detach().cpu().numpy(),
            "current_qpos8": current.detach().cpu().numpy(),
            "history_valid": valid.detach().cpu().numpy(),
            "command_present": command_present.detach().cpu().numpy(),
        }
        return clip, features, current, exact

    def _save_inference_input_once(self, exact: Mapping[str, np.ndarray]) -> None:
        if self.save_infer_input_dir is None or self._saved_infer_input:
            return
        output = self.save_infer_input_dir
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "mot_control_v2_input.npz", **exact)
        metadata = {
            "schema_version": 1,
            "channel_order": "univtac_native_cv2_decoded",
            "tactile_pad_order": ["left/thumb", "right/index"],
            "episode_seed": self._episode_seed,
            "observation_stride": self.config.observation_stride,
            "state_history": self.layout.num_frames,
            "policy_step": self.artifacts.step,
            "artifact_sha256": self.artifacts.metadata["artifact_sha256"],
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self._saved_infer_input = True

    @torch.no_grad()
    def act(self, observation: Mapping[str, Any], prompt: str | None = None) -> np.ndarray:
        del prompt
        qpos8 = _qpos8(observation, finite=False)
        if not np.isfinite(qpos8).all():
            # A post-reset non-finite observation may hold the last finite command.  The first
            # observation cannot invent a state and therefore fails closed in TemporalEnsemblerV2.
            hold = self._ensembler.safe_hold() if self._last_command is not None else np.zeros(8, dtype=np.float32)
            chunk = safe_hold_chunk(hold)
            action = self._ensembler.step(chunk, qpos8)
            debug_chunk = chunk
        else:
            self._observe(observation, qpos8)
            clip, state_features, current, exact = self._model_inputs()
            self._save_inference_input_once(exact)
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
                encoded = self.backbone.encode_full(clip)
            output = self.head(encoded, state_features.float(), current.float())
            chunk = output.action_chunk[0].float().detach().cpu().numpy()
            if chunk.shape != (V2_HORIZON, ACTIVE_ACTION_DIM):
                raise ValueError(f"V2 controller produced chunk {chunk.shape}, expected {(V2_HORIZON, 8)}")
            expected_placeholder = np.concatenate((np.zeros(ARM_DIM, dtype=np.float32), qpos8[ARM_DIM:]))
            if np.isfinite(chunk[0]).all() and not np.allclose(chunk[0], expected_placeholder, rtol=0, atol=1e-6):
                raise ValueError("V2 chunk row zero is not the current-state placeholder")
            action = self._ensembler.step(chunk, qpos8)
            debug_chunk = chunk if np.isfinite(chunk).all() else safe_hold_chunk(qpos8)

        infer_step = self._ensembler.exec_step - 1
        self._chunk_history.append((debug_chunk.copy(), infer_step, qpos8.copy()))
        ensemble_debug = self._ensembler.last_debug
        counters = self.get_safety_counters()
        self._last_debug = {
            "exec_step": infer_step,
            "latest_chunk_shape": list(debug_chunk.shape),
            "latest_chunk_first_mixed8": debug_chunk[FIRST_EXECUTABLE_INDEX].tolist(),
            "ensemble_candidates": 0 if ensemble_debug is None else ensemble_debug.candidate_count,
            "ensemble_weights": [] if ensemble_debug is None else list(ensemble_debug.weights),
            "used_fallback": False if ensemble_debug is None else ensemble_debug.used_fallback,
            "sent_action8": action.tolist(),
            "safety_counters": counters,
        }
        self._last_command = action.copy()
        return action.astype(np.float32, copy=False)
