#!/usr/bin/env python
"""Train the state-conditioned, dense-token MoT-Control V3 policy.

This trainer intentionally has a narrower contract than the generic MoT-JEPA policy trainer:

* one ``lift_bottle`` task and one eight-dimensional active action view,
* the qualified 100k EMA backbone at official input resolution,
* an H32 mixed target (arm offsets, absolute gripper), and
* one DDP module containing the backbone and deterministic control head.

Head-only training and final-block adaptation are separate completed runs.  A fresh adaptation
run must name its completed head run with ``--init-run``; requeues resume only their own atomic
checkpoints.  This keeps optimizer topology, sample order, and random augmentation streams exact.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import hashlib
import json
import logging
import os
import pathlib
import shutil
import tempfile
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch.nn.parallel import DistributedDataParallel
import zarr

from openpi.mot_jepa import runtime
from openpi.mot_jepa.control_v2 import ControlV2
from openpi.mot_jepa.control_v2_config import CONTROL_V2_ARTIFACT_SCHEMA
from openpi.mot_jepa.control_v2_config import CONTROL_V2_PRODUCTION_EPISODES
from openpi.mot_jepa.control_v2_config import ControlV2Artifact
from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.control_v2_config import control_v2_cli
from openpi.mot_jepa.control_v2_data import ControlV2Normalizer
from openpi.mot_jepa.control_v2_data import build_state_features
from openpi.mot_jepa.control_v2_data import load_normalization_artifact
from openpi.mot_jepa.control_v2_dataset import CONTROL_V2_SPLIT_IDENTITY
from openpi.mot_jepa.control_v2_dataset import ControlV2Dataset
from openpi.mot_jepa.control_v2_dataset import collate_control_v2
from openpi.mot_jepa.model import ClipInputs
from openpi.mot_jepa.mot_encoder import EncoderOutput
from openpi.mot_jepa.policy_finetune import BackboneSelection
from openpi.mot_jepa.policy_finetune import configure_backbone_trainability
from openpi.mot_jepa.policy_finetune import grad_norm
from openpi.mot_jepa.policy_finetune import make_step_generator
from openpi.shared.wandb_compat import wandb
from scripts.mot_jepa_train import InfiniteBatchSampler
from scripts.mot_jepa_train import lr_at
from scripts.mot_jepa_train import reduce_metrics

logger = logging.getLogger("mot_jepa.control_v2")
VALIDATION_SAMPLING = "all_episode_time_phase_contact_cold_start_stratified_v2"
SELECTION_CONSTRAINT = "validation_safety_violation_values==0"
_VALIDATION_TIME_BINS = ("early", "middle", "late")
_VALIDATION_HISTORY_STEPS = 16
_VALIDATION_PREFIX_SCENARIOS = tuple(
    (length, oldest_present) for length in range(1, _VALIDATION_HISTORY_STEPS + 1) for oldest_present in (False, True)
)

_QUALIFIED_BACKBONE_FILES = {
    "checkpoints/100000/metadata.pt": (
        1_267,
        "202764d003c37c2247f60b72978c72ec7732ab7ff128b35470565349308089c7",
    ),
    "checkpoints/100000/teacher_ema.pt": (
        119_633_043,
        "46af86aaafbd7c2b9a9c065653d5de917bc48d7e02726689a0e6aa4a666c461d",
    ),
    "checkpoints/100000/train_config.json": (
        2_566,
        "9ab7df985458799738455dcc5bac85ea9ad05af12b699d88a1cad7a98aa993ea",
    ),
    "checkpoints/latest": (
        7,
        "b80500a01f984c764f1a3b486622d0ef7cc5b13fa9bd57ec9015113eaf875597",
    ),
}


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def store_contract_digest(stores: list[str]) -> str:
    """Recompute the exact non-image control contract stamped by the stats job."""

    digest = hashlib.sha256()
    for store_idx, path in enumerate(stores):
        group = zarr.open_group(path, mode="r")
        digest.update(f"store={store_idx}".encode())
        for section in ("data", "meta"):
            for name in sorted(group[section].array_keys()):
                array = group[section][name]
                digest.update(f"{section}/{name}:{array.shape}:{array.dtype}".encode())
        digest.update(np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64).tobytes())
        if "source_episode_seed" in set(group["meta"].array_keys()):
            digest.update(np.asarray(group["meta"]["source_episode_seed"][:], dtype=np.int64).tobytes())
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
            if name in set(group["data"].array_keys()):
                digest.update(np.asarray(group["data"][name][:]).tobytes())
    return digest.hexdigest()


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_staging = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    staging = pathlib.Path(raw_staging)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class ArtifactBundle:
    artifact: ControlV2Artifact
    normalizer: ControlV2Normalizer
    artifact_sha256: str
    normalization_sha256: str
    normalization_counts: dict[str, int]


@dataclasses.dataclass(frozen=True)
class ValidationSelection:
    """Frozen, auditable held-out examples used for every checkpoint comparison."""

    indices: tuple[int, ...]
    identities: tuple[str, ...]
    episode_sample_counts: tuple[tuple[str, int], ...]
    sample_sha256: str
    requested_examples: int
    time_bins: tuple[str, ...]
    phase_ids: tuple[int, ...]
    contact_classes: tuple[int, ...]
    history_lengths: tuple[int, ...]
    oldest_command_present: tuple[bool, ...]

    def __post_init__(self) -> None:
        count = len(self.indices)
        aligned = {
            "identities": len(self.identities),
            "time_bins": len(self.time_bins),
            "phase_ids": len(self.phase_ids),
            "contact_classes": len(self.contact_classes),
            "history_lengths": len(self.history_lengths),
            "oldest_command_present": len(self.oldest_command_present),
        }
        if not count or any(value != count for value in aligned.values()):
            raise ValueError(f"validation selection fields are empty or misaligned: indices={count}, {aligned}")
        if len(set(self.indices)) != count:
            raise ValueError("validation selection contains duplicate dataset indices")
        if self.requested_examples <= 0:
            raise ValueError("requested validation examples must be positive")
        if any(value not in _VALIDATION_TIME_BINS for value in self.time_bins):
            raise ValueError("validation time bins must be early/middle/late")
        if any(value not in {-1, 0, 1, 2, 3} for value in self.phase_ids):
            raise ValueError("validation phase ids must be unknown (-1) or in [0, 4)")
        if any(value not in {-1, 0, 1} for value in self.contact_classes):
            raise ValueError("validation contact classes must be unknown (-1), absent, or contact")
        if any(not 1 <= value <= _VALIDATION_HISTORY_STEPS for value in self.history_lengths):
            raise ValueError(f"validation history lengths must be in [1, {_VALIDATION_HISTORY_STEPS}]")
        scenarios = set(zip(self.history_lengths, self.oldest_command_present, strict=True))
        if not set(_VALIDATION_PREFIX_SCENARIOS) <= scenarios:
            raise ValueError("validation selection does not cover every V3 cold-start history/parity scenario")
        episode_counts = [value for _identity, value in self.episode_sample_counts]
        if not episode_counts or any(value <= 0 for value in episode_counts) or sum(episode_counts) != count:
            raise ValueError("validation episode counts must be positive and close over every sample")
        canonical = json.dumps(self.identities, ensure_ascii=False, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != self.sample_sha256:
            raise ValueError("validation sample digest does not close over its immutable identities")

    @property
    def episode_count(self) -> int:
        return len(self.episode_sample_counts)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sampling": VALIDATION_SAMPLING,
            "requested_examples": self.requested_examples,
            "sample_count": len(self.indices),
            "episode_count": self.episode_count,
            "sample_sha256": self.sample_sha256,
            "episodes": [
                {"identity": identity, "sample_count": count} for identity, count in self.episode_sample_counts
            ],
            "samples": [
                {"dataset_index": index, "identity": identity}
                for index, identity in zip(self.indices, self.identities, strict=True)
            ],
        }


def build_validation_selection(dataset: ControlV2Dataset, max_examples: int) -> ValidationSelection:
    """Build one deterministic, stratified validation panel covering every held-out episode.

    ``max_examples`` is the nominal compute budget.  It becomes a floor when there are more
    held-out episodes: silently omitting an episode would make checkpoint selection depend on a
    lexicographic subset of the holdout.  Remaining slots are divided into non-overlapping time
    segments within each episode, then chosen to balance phase, contact, and coarse episode time
    globally.  A fixed prefix scenario is attached to every row so checkpoint selection exercises
    cold start as well as saturated history without stochastic validation augmentation.
    """

    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    entries = np.asarray(dataset.clip_index.entries, dtype=np.int64)
    if entries.ndim != 2 or entries.shape[1] != 4 or len(entries) == 0:
        raise ValueError("validation clip index must be a non-empty (N, 4) array")

    pools: dict[tuple[str, int], list[int]] = {}
    for dataset_index, row in enumerate(entries):
        store_idx, start, stride, episode_idx = (int(value) for value in row)
        if not 0 <= store_idx < len(dataset.store_paths) or start < 0 or stride < 1 or episode_idx < 0:
            raise ValueError(f"validation clip index row {dataset_index} has out-of-range values {row.tolist()}")
        store = str(pathlib.Path(dataset.store_paths[store_idx]).resolve())
        pools.setdefault((store, episode_idx), []).append(dataset_index)
    for pool in pools.values():
        pool.sort(key=lambda dataset_index: (int(entries[dataset_index, 1]), int(entries[dataset_index, 2])))
    keys = sorted(pools)
    minimum_budget = max(len(keys), len(_VALIDATION_PREFIX_SCENARIOS))
    if len(entries) < minimum_budget:
        raise ValueError(
            f"validation index has {len(entries)} rows but needs at least {minimum_budget} to cover "
            "every episode and all 32 V3 cold-start scenarios"
        )
    budget = min(max(max_examples, minimum_budget), len(entries))
    allocations = dict.fromkeys(keys, 1)
    remaining = budget - len(keys)
    while remaining:
        progressed = False
        for key in keys:
            if allocations[key] >= len(pools[key]):
                continue
            allocations[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:  # pragma: no cover - budget is capped by the total pool size
            raise RuntimeError("unable to allocate the validation sample budget")

    phase_cache: dict[int, np.ndarray] = {}
    contact_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}

    def labels(dataset_index: int) -> tuple[int, int]:
        store_idx, start, stride, _episode_idx = (int(value) for value in entries[dataset_index])
        history_steps = int(getattr(getattr(getattr(dataset, "base", None), "layout", None), "num_frames", 16))
        current = start + (history_steps - 1) * stride
        provider = getattr(dataset, "validation_stratum", None)
        if callable(provider):
            phase_id, contact_class = provider(dataset_index, store_idx, current)
        elif callable(getattr(dataset, "_phase_ids", None)) and hasattr(dataset, "base"):
            if store_idx not in phase_cache:
                phase_cache[store_idx] = np.asarray(dataset._phase_ids(store_idx), dtype=np.int64)  # noqa: SLF001
            phase_id = int(phase_cache[store_idx][current])
            if store_idx not in contact_cache:
                group, _ = dataset.base._store(store_idx)  # noqa: SLF001
                data = group["data"]
                arrays = set(data.array_keys())
                if "contact" not in arrays:
                    contact_cache[store_idx] = None
                else:
                    contact = np.asarray(data["contact"][:], dtype=np.float32).reshape(-1)
                    valid = (
                        np.asarray(data["contact_valid"][:], dtype=np.uint8).astype(bool).reshape(-1)
                        if "contact_valid" in arrays
                        else np.ones(contact.shape, dtype=bool)
                    )
                    contact_cache[store_idx] = contact, valid
            contact_entry = contact_cache[store_idx]
            contact_class = -1
            if contact_entry is not None and bool(contact_entry[1][current]):
                contact_class = int(float(contact_entry[0][current]) >= 0.5)
        else:
            # Lightweight index-only datasets remain useful in unit tests and legacy analysis.
            # Production ControlV2Dataset instances always take the strict branch above.
            phase_id, contact_class = -1, -1
        phase_id, contact_class = int(phase_id), int(contact_class)
        if phase_id not in {-1, 0, 1, 2, 3}:
            raise ValueError(f"validation sample {dataset_index} has invalid phase id {phase_id}")
        if contact_class not in {-1, 0, 1}:
            raise ValueError(f"validation sample {dataset_index} has invalid contact class {contact_class}")
        return phase_id, contact_class

    annotations: dict[int, tuple[str, int, int]] = {}
    for key in keys:
        pool = pools[key]
        for position, dataset_index in enumerate(pool):
            time_bin_index = min(
                len(_VALIDATION_TIME_BINS) - 1,
                int(np.floor((position + 0.5) * len(_VALIDATION_TIME_BINS) / len(pool))),
            )
            phase_id, contact_class = labels(dataset_index)
            annotations[dataset_index] = (_VALIDATION_TIME_BINS[time_bin_index], phase_id, contact_class)

    phase_availability = dict.fromkeys(range(4), 0)
    contact_availability = dict.fromkeys(range(2), 0)
    for _time_bin, phase_id, contact_class in annotations.values():
        if phase_id >= 0:
            phase_availability[phase_id] += 1
        if contact_class >= 0:
            contact_availability[contact_class] += 1
    selected_indices: list[int] = []
    time_counts = dict.fromkeys(_VALIDATION_TIME_BINS, 0)
    phase_counts = dict.fromkeys(range(4), 0)
    contact_counts = dict.fromkeys(range(2), 0)
    joint_counts: dict[tuple[str, int, int], int] = {}
    for slot in range(max(allocations.values())):
        for key in keys:
            count = allocations[key]
            if slot >= count:
                continue
            pool = pools[key]
            segment_start = slot * len(pool) // count
            segment_stop = (slot + 1) * len(pool) // count
            candidates = pool[segment_start:segment_stop]
            if not candidates:  # pragma: no cover - allocation never exceeds pool size
                raise RuntimeError(f"empty validation time segment for {key} slot {slot}/{count}")

            midpoint = (segment_start + segment_stop - 1) / 2
            positions = {dataset_index: pools[key].index(dataset_index) for dataset_index in candidates}
            scored_candidates: list[tuple[tuple[int, int, int, int, int, int, int, float, int], int]] = []
            for dataset_index in candidates:
                time_bin, phase_id, contact_class = annotations[dataset_index]
                phase_count = phase_counts[phase_id] if phase_id >= 0 else len(selected_indices) + 1
                contact_count = contact_counts[contact_class] if contact_class >= 0 else len(selected_indices) + 1
                joint = (time_bin, phase_id, contact_class)
                coverage_gain = int(time_counts[time_bin] == 0)
                coverage_gain += int(phase_id >= 0 and phase_count == 0)
                coverage_gain += int(contact_class >= 0 and contact_count == 0)
                scored_candidates.append(
                    (
                        (
                            -coverage_gain,
                            time_counts[time_bin],
                            phase_count,
                            contact_count,
                            phase_availability[phase_id] if phase_id >= 0 else len(entries) + 1,
                            contact_availability[contact_class] if contact_class >= 0 else len(entries) + 1,
                            joint_counts.get(joint, 0),
                            abs(positions[dataset_index] - midpoint),
                            dataset_index,
                        ),
                        dataset_index,
                    )
                )

            selected = min(scored_candidates)[1]
            selected_indices.append(selected)
            time_bin, phase_id, contact_class = annotations[selected]
            time_counts[time_bin] += 1
            if phase_id >= 0:
                phase_counts[phase_id] += 1
            if contact_class >= 0:
                contact_counts[contact_class] += 1
            joint = (time_bin, phase_id, contact_class)
            joint_counts[joint] = joint_counts.get(joint, 0) + 1

    history_steps = int(getattr(getattr(getattr(dataset, "base", None), "layout", None), "num_frames", 16))
    if history_steps != _VALIDATION_HISTORY_STEPS:
        raise ValueError(f"V3 validation requires {_VALIDATION_HISTORY_STEPS} history steps, got {history_steps}")
    history_lengths = tuple(
        _VALIDATION_PREFIX_SCENARIOS[index % len(_VALIDATION_PREFIX_SCENARIOS)][0] for index in range(budget)
    )
    oldest_command_present = tuple(
        _VALIDATION_PREFIX_SCENARIOS[index % len(_VALIDATION_PREFIX_SCENARIOS)][1] for index in range(budget)
    )

    available_time_bins = {annotation[0] for annotation in annotations.values()}
    selected_time_bins = {annotations[index][0] for index in selected_indices}
    available_phases = {annotation[1] for annotation in annotations.values() if annotation[1] >= 0}
    selected_phases = {annotations[index][1] for index in selected_indices if annotations[index][1] >= 0}
    available_contacts = {annotation[2] for annotation in annotations.values() if annotation[2] >= 0}
    selected_contacts = {annotations[index][2] for index in selected_indices if annotations[index][2] >= 0}
    missing_strata = {
        "time_bin": sorted(available_time_bins - selected_time_bins),
        "phase": sorted(available_phases - selected_phases),
        "contact": sorted(available_contacts - selected_contacts),
    }
    missing_strata = {name: values for name, values in missing_strata.items() if values}
    if missing_strata:
        raise RuntimeError(f"validation stratification omitted available labels: {missing_strata}")

    identities = []
    time_bins = []
    phase_ids = []
    contact_classes = []
    for selection_index, dataset_index in enumerate(selected_indices):
        store_idx, start, stride, episode_idx = (int(value) for value in entries[dataset_index])
        store = str(pathlib.Path(dataset.store_paths[store_idx]).resolve())
        time_bin, phase_id, contact_class = annotations[dataset_index]
        phase = "unknown" if phase_id < 0 else str(phase_id)
        contact = "unknown" if contact_class < 0 else str(contact_class)
        identities.append(
            f"{store}\0episode={episode_idx}\0start={start}\0stride={stride}"
            f"\0dataset_index={dataset_index}\0time_bin={time_bin}\0phase={phase}\0contact={contact}"
            f"\0history_length={history_lengths[selection_index]}"
            f"\0oldest_command_present={int(oldest_command_present[selection_index])}"
        )
        time_bins.append(time_bin)
        phase_ids.append(phase_id)
        contact_classes.append(contact_class)
    canonical = json.dumps(identities, ensure_ascii=False, separators=(",", ":"))
    sample_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    episode_sample_counts = tuple(
        (f"{store}\0episode={episode_idx}", allocations[(store, episode_idx)])
        for store, episode_idx in keys
        if allocations[(store, episode_idx)]
    )
    return ValidationSelection(
        indices=tuple(selected_indices),
        identities=tuple(identities),
        episode_sample_counts=episode_sample_counts,
        sample_sha256=sample_sha256,
        requested_examples=max_examples,
        time_bins=tuple(time_bins),
        phase_ids=tuple(phase_ids),
        contact_classes=tuple(contact_classes),
        history_lengths=history_lengths,
        oldest_command_present=oldest_command_present,
    )


def freeze_validation_selection(run_dir: pathlib.Path, selection: ValidationSelection) -> None:
    """Write the exact held-out identities once and reject any later selection drift."""

    path = run_dir / "VALIDATION_SAMPLES.json"
    expected = json.dumps(selection.to_manifest(), indent=2, sort_keys=True) + "\n"
    error: str | None = None
    if runtime.is_main_process():
        try:
            if path.is_file() and path.read_text() != expected:
                raise ValueError(f"{path} differs from the deterministic held-out selection")
            if not path.is_file():
                _atomic_write(path, expected)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if runtime.get_world_size() > 1:
        payload: list[str | None] = [error]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error is not None:
        raise RuntimeError(f"failed to freeze held-out selection: {error}")


def load_validation_manifest(run_dir: pathlib.Path) -> dict[str, Any]:
    """Validate and return the immutable held-out identity manifest."""

    path = run_dir / "VALIDATION_SAMPLES.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing held-out identity manifest {path}")
    manifest = json.loads(path.read_text())
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
    if manifest["schema_version"] != 1 or manifest["sampling"] != VALIDATION_SAMPLING:
        raise ValueError("VALIDATION_SAMPLES.json has an unsupported schema or sampling rule")
    samples = manifest["samples"]
    episodes = manifest["episodes"]
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
    required_prefix_scenarios = {
        (length, int(oldest_present)) for length, oldest_present in _VALIDATION_PREFIX_SCENARIOS
    }
    if len(samples) < len(required_prefix_scenarios) or not required_prefix_scenarios <= prefix_scenarios:
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

    canonical = json.dumps(identities, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    expected_counts = {
        "sample_count": len(samples),
        "episode_count": len(episodes),
        "sample_sha256": digest,
    }
    mismatches = {
        name: (manifest.get(name), value) for name, value in expected_counts.items() if manifest.get(name) != value
    }
    if len(samples) < len(episodes):
        mismatches["all_episode_coverage"] = (len(samples), len(episodes))
    if mismatches:
        raise ValueError(f"VALIDATION_SAMPLES.json identity/count closure differs: {mismatches}")
    return manifest


def validate_validation_metric_stamp(metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = {
        "validation_sampling": manifest["sampling"],
        "validation_sample_sha256": manifest["sample_sha256"],
        "validation_episode_count": float(manifest["episode_count"]),
        "validation_examples": float(manifest["sample_count"]),
    }
    mismatches = {name: (metrics.get(name), value) for name, value in expected.items() if metrics.get(name) != value}
    if mismatches:
        raise ValueError(f"held-out metric identity/count stamp differs from manifest: {mismatches}")


def _normalization_counts(normalizer: ControlV2Normalizer) -> dict[str, int]:
    return {
        "qpos": int(normalizer.qpos_count),
        "velocity": int(normalizer.velocity_count),
        "previous_command": int(normalizer.previous_command_count),
    }


def load_and_validate_artifacts(cfg: ControlV2TrainConfig, *, validate_store_content: bool) -> ArtifactBundle:
    """Load the stats bundle and reject any stale data/config pairing."""

    run = cfg.run_dir
    required = (
        run / "run_config.json",
        run / "artifact.json",
        run / "normalization.json",
        run / "VALIDATION_SAMPLES.json",
        run / "STATS_DONE",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"run the V3 stats job first; missing {missing}")

    frozen_cfg = ControlV2TrainConfig.from_json((run / "run_config.json").read_text())
    if frozen_cfg != cfg:
        raise ValueError("CLI config differs from the stats-frozen run_config.json; refusing configuration drift")

    artifact_path = run / "artifact.json"
    normalization_path = run / "normalization.json"
    artifact = ControlV2Artifact.from_json(artifact_path.read_text())
    if cfg.require_authoritative_control and not artifact.authoritative_control:
        raise ValueError("configuration requires authoritative control labels but the artifact is legacy/inferred")
    if cfg.require_authoritative_control:
        artifact.require_production_data()
    elif artifact.total_episodes < cfg.minimum_total_episodes:
        raise ValueError(
            f"artifact contains {artifact.total_episodes} episodes; configuration requires {cfg.minimum_total_episodes}"
        )
    if cfg.require_authoritative_control and cfg.minimum_total_episodes != CONTROL_V2_PRODUCTION_EPISODES:
        raise ValueError(
            f"authoritative production config must require exactly {CONTROL_V2_PRODUCTION_EPISODES} episodes"
        )
    stats = load_normalization_artifact(normalization_path)
    normalizer = ControlV2Normalizer(stats)
    validation_manifest = load_validation_manifest(run)
    marker = json.loads((run / "STATS_DONE").read_text())
    expected_marker_fields = {
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
    if not isinstance(marker, dict) or set(marker) != expected_marker_fields:
        raise ValueError("STATS_DONE has an invalid field set")
    expected_marker = {
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
    marker_mismatches = {
        name: (marker.get(name), value) for name, value in expected_marker.items() if marker.get(name) != value
    }
    min_slack = marker.get("validation_oracle_min_slack")
    max_rate_ratio = marker.get("validation_oracle_max_rate_ratio")
    if (
        isinstance(min_slack, bool)
        or not isinstance(min_slack, (int, float))
        or not np.isfinite(float(min_slack))
        or float(min_slack) <= 0
    ):
        marker_mismatches["validation_oracle_min_slack"] = (min_slack, "finite and > 0")
    if (
        isinstance(max_rate_ratio, bool)
        or not isinstance(max_rate_ratio, (int, float))
        or not np.isfinite(float(max_rate_ratio))
        or not 0 <= float(max_rate_ratio) < 1
    ):
        marker_mismatches["validation_oracle_max_rate_ratio"] = (max_rate_ratio, "finite and in [0, 1)")
    if marker_mismatches:
        raise ValueError(f"STATS_DONE does not close the artifact bundle: {marker_mismatches}")

    stores = sorted(glob.glob(cfg.store_glob))
    if not stores:
        raise ValueError(f"no Zarr stores matched {cfg.store_glob!r}")
    resolved_stores = tuple(str(pathlib.Path(path).resolve()) for path in stores)
    if resolved_stores != artifact.source_stores:
        raise ValueError(
            f"current source stores differ from artifact: {resolved_stores!r} != {artifact.source_stores!r}"
        )
    if validate_store_content:
        current_digest = store_contract_digest(stores)
        if current_digest != artifact.source_store_sha256:
            raise ValueError(
                "source Zarr control contract changed after stats fitting: "
                f"{current_digest} != {artifact.source_store_sha256}"
            )
    return ArtifactBundle(
        artifact=artifact,
        normalizer=normalizer,
        artifact_sha256=file_sha256(artifact_path),
        normalization_sha256=file_sha256(normalization_path),
        normalization_counts=_normalization_counts(normalizer),
    )


def _collective_validate_artifacts(cfg: ControlV2TrainConfig) -> ArtifactBundle:
    """Hash large stores once on rank zero, then make every rank fail together."""

    error: str | None = None
    if runtime.is_main_process():
        try:
            load_and_validate_artifacts(cfg, validate_store_content=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if runtime.get_world_size() > 1:
        message: list[str | None] = [error]
        dist.broadcast_object_list(message, src=0)
        error = message[0]
    if error is not None:
        raise RuntimeError(f"rank-zero artifact validation failed: {error}")
    return load_and_validate_artifacts(cfg, validate_store_content=False)


def build_dataset(cfg: ControlV2TrainConfig, *, split: str = "train") -> ControlV2Dataset:
    stores = sorted(glob.glob(cfg.store_glob))
    return ControlV2Dataset(
        stores,
        cfg.layout,
        observation_stride=cfg.observation_stride,
        action_stride=cfg.action_stride,
        control_step_stride=1,
        index_step=cfg.index_step,
        lowdim_channels=cfg.lowdim_channels,
        split=split,
        validation_fraction=cfg.validation_fraction,
        split_seed=cfg.split_seed,
    )


def _history_indices(lengths: torch.Tensor, time_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-align each available suffix and repeat its oldest value on the left."""

    if lengths.ndim != 1 or lengths.dtype != torch.long:
        raise ValueError("lengths must be an int64 vector")
    if bool(((lengths < 1) | (lengths > time_steps)).any()):
        raise ValueError(f"history lengths must be in [1, {time_steps}]")
    positions = torch.arange(time_steps, device=lengths.device).expand(lengths.shape[0], -1)
    starts = time_steps - lengths
    indices = torch.maximum(positions, starts[:, None])
    valid = positions >= starts[:, None]
    return indices, valid


def _gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or values.shape[:2] != indices.shape:
        raise ValueError(f"values/indices batch-time shapes differ: {values.shape[:2]} != {indices.shape}")
    view = indices.reshape(*indices.shape, *((1,) * (values.ndim - 2)))
    return torch.gather(values, 1, view.expand_as(values))


def _prefix_command_presence(
    command_present: torch.Tensor,
    history_valid: torch.Tensor,
    available_lengths: torch.Tensor,
    oldest_command_present: torch.Tensor,
) -> torch.Tensor:
    """Mask padding and reproduce both stride-2 online cold-start parities.

    With observation stride two, rollout ages ``2L-1`` and ``2L`` both produce an ``L``-frame
    model history.  The former still contains the episode's first observation (no policy-issued
    predecessor); the latter starts one raw control step later and does have one.  The caller
    samples that parity explicitly instead of always deleting the oldest command.
    """

    if command_present.shape != history_valid.shape or command_present.dtype != torch.bool:
        raise ValueError("command_present/history_valid must be equally shaped bool tensors")
    if available_lengths.shape != command_present.shape[:1] or available_lengths.dtype != torch.long:
        raise ValueError("available_lengths must be an int64 batch vector")
    if oldest_command_present.shape != command_present.shape[:1] or oldest_command_present.dtype != torch.bool:
        raise ValueError("oldest_command_present must be a bool batch vector")
    output = command_present & history_valid
    output = output.clone()
    rows = torch.arange(command_present.shape[0], device=command_present.device)
    first_real = command_present.shape[1] - available_lengths
    output[rows, first_real] &= oldest_command_present
    return output


def _apply_validation_history_scenarios(
    inputs: ClipInputs,
    state_history: torch.Tensor,
    command_history: torch.Tensor,
    command_present: torch.Tensor,
    history_lengths: torch.Tensor,
    oldest_command_present: torch.Tensor,
) -> tuple[ClipInputs, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply frozen cold-start scenarios without introducing validation randomness."""

    batch, time_steps = state_history.shape[:2]
    if history_lengths.shape != (batch,) or history_lengths.dtype != torch.long:
        raise ValueError("validation history_lengths must be an int64 batch vector")
    if oldest_command_present.shape != (batch,) or oldest_command_present.dtype != torch.bool:
        raise ValueError("validation oldest_command_present must be a bool batch vector")
    indices, history_valid = _history_indices(history_lengths, time_steps)
    inputs = _permute_inputs(inputs, indices)
    state_history = _gather_time(state_history, indices)
    command_history = _gather_time(command_history, indices)
    command_present = _prefix_command_presence(
        _gather_time(command_present, indices),
        history_valid,
        history_lengths,
        oldest_command_present,
    )
    return inputs, state_history, command_history, command_present, history_valid


def _select_batch(values: torch.Tensor | None, selection: torch.Tensor) -> torch.Tensor | None:
    return None if values is None else values.index_select(0, selection)


def _select_inputs(inputs: ClipInputs, selection: torch.Tensor) -> ClipInputs:
    return ClipInputs(
        video=inputs.video.index_select(0, selection),
        gel=inputs.gel.index_select(0, selection),
        lowdim=inputs.lowdim.index_select(0, selection),
        gel_sensor_ids=_select_batch(inputs.gel_sensor_ids, selection),
        lowdim_sensor_ids=_select_batch(inputs.lowdim_sensor_ids, selection),
    )


def _permute_inputs(inputs: ClipInputs, indices: torch.Tensor) -> ClipInputs:
    return ClipInputs(
        video=_gather_time(inputs.video, indices),
        gel=_gather_time(inputs.gel, indices),
        lowdim=_gather_time(inputs.lowdim, indices),
        gel_sensor_ids=inputs.gel_sensor_ids,
        lowdim_sensor_ids=inputs.lowdim_sensor_ids,
    )


def _order_indices(
    history_valid: torch.Tensor,
    labels: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Corrupt only the real suffix; padding stays left and the last state stays present."""

    if history_valid.ndim != 2 or history_valid.dtype != torch.bool:
        raise ValueError("history_valid must be a bool (B, T) tensor")
    if labels.shape != history_valid.shape[:1] or labels.dtype != torch.long:
        raise ValueError("labels must be an int64 (B,) tensor")
    batch, time_steps = history_valid.shape
    output = torch.arange(time_steps, device=history_valid.device).expand(batch, -1).clone()
    for row in range(batch):
        valid_positions = torch.nonzero(history_valid[row], as_tuple=False).flatten()
        length = int(valid_positions.numel())
        label = int(labels[row])
        if length < 2 or label == 0:
            continue
        if label == 1:
            permutation = torch.arange(length - 1, -1, -1, device=output.device)
        elif label == 2:
            midpoint = length // 2
            permutation = torch.cat(
                (torch.arange(midpoint, length, device=output.device), torch.arange(midpoint, device=output.device))
            )
        elif label == 3:
            permutation = torch.randperm(length, generator=generator, device=output.device)
            if bool(torch.equal(permutation, torch.arange(length, device=output.device))):
                permutation = permutation.roll(1)
        else:
            raise ValueError("order labels must be in [0, 4)")
        output[row, valid_positions] = valid_positions[permutation]
    return output


def _fp32_encoded(encoded: EncoderOutput) -> EncoderOutput:
    return EncoderOutput(
        tokens=[tensor.float() for tensor in encoded.tokens],
        sync_readout=[tensor.float() for tensor in encoded.sync_readout],
        final_readout=[tensor.float() for tensor in encoded.final_readout],
    )


def control_safety_terms(
    predicted_absolute: torch.Tensor,
    rate_reference_qpos: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    max_command_delta: torch.Tensor,
    *,
    rate_margin: float,
    first_n: int,
) -> dict[str, torch.Tensor]:
    """Return differentiable safety loss and exact deployment-limit diagnostics.

    Row zero is the observation-time placeholder and is deliberately excluded.  For future row
    ``h``, the runtime limiter compares the proposed absolute command with the measured qpos at
    ``t + h - 1``.  The supplied reference follows that exact contract; using the preceding
    demonstrated command would train against a different limiter than the one used in rollout.
    """

    if predicted_absolute.ndim != 3 or predicted_absolute.shape[-1] != 8:
        raise ValueError("predicted_absolute must have shape (B, H, 8)")
    expected_reference = (predicted_absolute.shape[0], predicted_absolute.shape[1] - 1, 8)
    if rate_reference_qpos.shape != expected_reference:
        raise ValueError(f"rate_reference_qpos must have shape {expected_reference}")
    for name, tensor in (
        ("joint_lower", joint_lower),
        ("joint_upper", joint_upper),
        ("max_command_delta", max_command_delta),
    ):
        if tensor.shape != (8,):
            raise ValueError(f"{name} must have shape (8,)")
    if not 0 < rate_margin <= 1:
        raise ValueError("rate_margin must be in (0, 1]")
    if not 1 <= first_n <= predicted_absolute.shape[1] - 1:
        raise ValueError("first_n must select executable rows inside the predicted horizon")

    # Runtime retains exactly the first ``chunk_first_n`` executable rows.  Predictions beyond
    # that slice are supervised for representation/forecast quality but never enter the temporal
    # ensemble, so they must not influence the deployment-safety gate.
    future = predicted_absolute[:, 1 : 1 + first_n].float()
    reference = rate_reference_qpos[:, :first_n].float()
    lower = joint_lower.to(device=future.device, dtype=future.dtype)
    upper = joint_upper.to(device=future.device, dtype=future.dtype)
    max_delta = max_command_delta.to(device=future.device, dtype=future.dtype)
    joint_range = upper - lower

    lower_excess_ratio = F.relu((lower - future) / joint_range)
    upper_excess_ratio = F.relu((future - upper) / joint_range)
    joint_excess_ratio = lower_excess_ratio + upper_excess_ratio
    rate_ratio = (future - reference).abs() / max_delta
    rate_margin_excess = F.relu(rate_ratio - rate_margin)
    loss = joint_excess_ratio.square().mean() + rate_margin_excess.square().mean()
    return {
        "loss": loss,
        "joint_violation_values": (joint_excess_ratio > 0).sum(),
        "rate_violation_values": (rate_ratio > 1).sum(),
        "joint_max_excess_ratio": joint_excess_ratio.max(),
        "rate_max_ratio": rate_ratio.max(),
    }


def control_safety_tube_terms(
    predicted_absolute: torch.Tensor,
    expert_absolute: torch.Tensor,
    rate_reference_qpos: torch.Tensor,
    max_command_delta: torch.Tensor,
    *,
    first_n: int,
) -> dict[str, torch.Tensor]:
    """Penalize imitation error in units of the expert's remaining rate-limit slack.

    The exact held-out gate is a forall constraint, while the original safety hinge averages only
    values that already crossed the limit.  For an expert-safe target ``e`` and rate reference
    ``r``, the triangle inequality proves that ``|prediction - e| <= max_delta - |e - r|`` is a
    sufficient condition for the unchanged deployment rate gate.  Optimizing the normalized tube
    everywhere, plus each example's worst retained value, supplies gradients before violations and
    aligns the training tail with that final gate.
    """

    if predicted_absolute.ndim != 3 or predicted_absolute.shape[-1] != 8:
        raise ValueError("predicted_absolute must have shape (B, H, 8)")
    if expert_absolute.shape != predicted_absolute.shape:
        raise ValueError("expert_absolute must match predicted_absolute")
    expected_reference = (predicted_absolute.shape[0], predicted_absolute.shape[1] - 1, 8)
    if rate_reference_qpos.shape != expected_reference:
        raise ValueError(f"rate_reference_qpos must have shape {expected_reference}")
    if max_command_delta.shape != (8,):
        raise ValueError("max_command_delta must have shape (8,)")
    if not 1 <= first_n <= predicted_absolute.shape[1] - 1:
        raise ValueError("first_n must select executable rows inside the predicted horizon")

    predicted = predicted_absolute[:, 1 : 1 + first_n].float()
    expert = expert_absolute[:, 1 : 1 + first_n].to(device=predicted.device, dtype=predicted.dtype)
    reference = rate_reference_qpos[:, :first_n].to(device=predicted.device, dtype=predicted.dtype)
    max_delta = max_command_delta.to(device=predicted.device, dtype=predicted.dtype)
    slack = max_delta - (expert - reference).abs()
    valid_slack = torch.isfinite(slack).all() & (slack > 0).all()
    if slack.device.type == "cuda":
        # Avoid one host synchronization per microbatch while still failing the CUDA stream closed.
        torch._assert_async(  # noqa: SLF001
            valid_slack,
            "expert safety slack must be finite and strictly positive",
        )
    elif not bool(valid_slack):
        raise ValueError("expert safety slack must be finite and strictly positive")

    tube_ratio = (predicted - expert).abs() / slack
    per_value = F.smooth_l1_loss(tube_ratio, torch.zeros_like(tube_ratio), beta=1.0, reduction="none")
    mean_loss = per_value.mean()
    tail_loss = per_value.flatten(1).max(dim=1).values.mean()
    return {
        "loss": mean_loss + tail_loss,
        "mean_loss": mean_loss,
        "tail_loss": tail_loss,
        "max_ratio": tube_ratio.max(),
        "min_slack": slack.min(),
    }


class ControlV2TrainModel(nn.Module):
    """Backbone and controller behind one DDP forward/loss boundary."""

    def __init__(
        self,
        backbone: nn.Module,
        head: ControlV2,
        normalizer: ControlV2Normalizer,
        *,
        train_backbone: bool,
        anchor_backbone: nn.Module | None,
        overlap_loss_weight: float,
        anchor_loss_weight: float,
        order_batch_fraction: float,
        safety_loss_weight: float,
        safety_tube_loss_weight: float,
        safety_rate_margin: float,
        joint_lower: tuple[float, ...],
        joint_upper: tuple[float, ...],
        max_command_delta: tuple[float, ...],
        chunk_first_n: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.normalizer = normalizer
        self.train_backbone = bool(train_backbone)
        self.anchor_backbone = anchor_backbone
        self.overlap_loss_weight = float(overlap_loss_weight)
        self.anchor_loss_weight = float(anchor_loss_weight)
        self.order_batch_fraction = float(order_batch_fraction)
        self.safety_loss_weight = float(safety_loss_weight)
        self.safety_tube_loss_weight = float(safety_tube_loss_weight)
        self.safety_rate_margin = float(safety_rate_margin)
        self.chunk_first_n = int(chunk_first_n)
        limit_device = self.head.action_scale.device
        self.register_buffer(
            "joint_lower",
            torch.as_tensor(joint_lower, dtype=torch.float32, device=limit_device),
            persistent=False,
        )
        self.register_buffer(
            "joint_upper",
            torch.as_tensor(joint_upper, dtype=torch.float32, device=limit_device),
            persistent=False,
        )
        self.register_buffer(
            "max_command_delta",
            torch.as_tensor(max_command_delta, dtype=torch.float32, device=limit_device),
            persistent=False,
        )
        if min(self.safety_loss_weight, self.safety_tube_loss_weight) < 0:
            raise ValueError("safety loss weights must be non-negative")
        if not 0 < self.safety_rate_margin <= 1:
            raise ValueError("safety_rate_margin must be in (0, 1]")
        if not 1 <= self.chunk_first_n <= self.head.config.horizon - 1:
            raise ValueError("chunk_first_n must select executable rows inside the head horizon")
        if self.joint_lower.shape != (8,) or self.joint_upper.shape != (8,) or self.max_command_delta.shape != (8,):
            raise ValueError("safety limits must contain exactly eight values")
        if bool((self.joint_lower >= self.joint_upper).any()) or bool((self.max_command_delta <= 0).any()):
            raise ValueError("safety limits must be ordered and strictly positive")

    @staticmethod
    def _encode(backbone: nn.Module, inputs: ClipInputs, *, needs_grad: bool) -> EncoderOutput:
        gradient = contextlib.nullcontext() if needs_grad else torch.no_grad()
        with (
            gradient,
            torch.autocast(
                device_type=inputs.video.device.type,
                dtype=torch.bfloat16,
                enabled=inputs.video.device.type == "cuda",
            ),
        ):
            encoded = backbone.encode_full(inputs)
        return _fp32_encoded(encoded)

    def _prefix_augment(
        self,
        inputs: ClipInputs,
        state_history: torch.Tensor,
        command_history: torch.Tensor,
        command_present: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[ClipInputs, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time_steps = state_history.shape[:2]
        lengths = torch.randint(
            1,
            time_steps + 1,
            (batch,),
            generator=generator,
            device=state_history.device,
            dtype=torch.long,
        )
        indices, history_valid = _history_indices(lengths, time_steps)
        # At every retained-history length deployment can be observed on either side of the raw
        # stride-2 boundary.  Train both exact command-presence states with caller-owned RNG.
        oldest_command_present = torch.randint(
            0,
            2,
            (batch,),
            generator=generator,
            device=state_history.device,
            dtype=torch.long,
        ).bool()
        inputs = _permute_inputs(inputs, indices)
        state_history = _gather_time(state_history, indices)
        command_history = _gather_time(command_history, indices)
        command_present = _prefix_command_presence(
            _gather_time(command_present, indices),
            history_valid,
            lengths,
            oldest_command_present,
        )
        return inputs, state_history, command_history, command_present, history_valid, lengths

    def _corrupted_order_loss(
        self,
        inputs: ClipInputs,
        state_features: torch.Tensor,
        current_qpos: torch.Tensor,
        history_valid: torch.Tensor,
        available_lengths: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        eligible = torch.nonzero(available_lengths >= 4, as_tuple=False).flatten()
        requested = max(1, round(state_features.shape[0] * self.order_batch_fraction))
        count = min(requested, int(eligible.numel())) if self.order_batch_fraction > 0 else 0
        if count == 0:
            return self.head.order_output.weight.sum() * 0
        selection_order = torch.randperm(eligible.numel(), generator=generator, device=eligible.device)[:count]
        selection = eligible[selection_order]
        labels = torch.randint(1, 4, (count,), generator=generator, device=selection.device, dtype=torch.long)
        selected_valid = history_valid.index_select(0, selection)
        order_indices = _order_indices(selected_valid, labels, generator=generator)
        corrupted_inputs = _permute_inputs(_select_inputs(inputs, selection), order_indices)
        corrupted_state = _gather_time(state_features.index_select(0, selection), order_indices)
        corrupted_qpos = current_qpos.index_select(0, selection)
        encoded = self._encode(self.backbone, corrupted_inputs, needs_grad=self.train_backbone)
        output = self.head(encoded, corrupted_state, corrupted_qpos)
        return F.cross_entropy(output.order_logits, labels)

    def forward(
        self,
        inputs: ClipInputs,
        state_history: torch.Tensor,
        command_history: torch.Tensor,
        command_present: torch.Tensor,
        target_chunk: torch.Tensor,
        absolute_qpos_chunk: torch.Tensor,
        rate_reference_qpos: torch.Tensor,
        phase_id: torch.Tensor,
        contact: torch.Tensor,
        contact_valid: torch.Tensor,
        sample_weight: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs, state_history, command_history, command_present, history_valid, lengths = self._prefix_augment(
            inputs,
            state_history,
            command_history,
            command_present,
            generator=generator,
        )
        features = build_state_features(
            state_history,
            history_valid=history_valid,
            previous_command=command_history,
            previous_command_present=command_present,
            normalizer=self.normalizer,
            dt=2.0,
        )
        current_qpos = features.qpos[:, -1].float()
        encoded = self._encode(self.backbone, inputs, needs_grad=self.train_backbone)
        output = self.head(encoded, features.tensor.float(), current_qpos)

        correct_order = torch.zeros(target_chunk.shape[0], dtype=torch.long, device=target_chunk.device)
        behavior = self.head.loss(
            output,
            target_chunk,
            sample_weight=sample_weight,
            phase_labels=phase_id,
            order_labels=correct_order,
        )
        if bool(contact_valid.any()):
            contact_loss = F.binary_cross_entropy_with_logits(
                output.contact_logits[contact_valid, 0], contact[contact_valid].float()
            )
        else:
            # Keep the contact head in DDP's executed graph even for a legacy-data batch.
            contact_loss = output.contact_logits.sum() * 0

        predicted_delta = output.absolute_qpos_chunk[:, 2:] - output.absolute_qpos_chunk[:, 1:-1]
        target_delta = absolute_qpos_chunk[:, 2:] - absolute_qpos_chunk[:, 1:-1]
        overlap_loss = F.smooth_l1_loss(
            predicted_delta,
            target_delta,
            beta=self.head.config.smooth_l1_beta,
        )
        safety = control_safety_terms(
            output.absolute_qpos_chunk,
            rate_reference_qpos,
            self.joint_lower,
            self.joint_upper,
            self.max_command_delta,
            rate_margin=self.safety_rate_margin,
            first_n=self.chunk_first_n,
        )
        safety_tube = control_safety_tube_terms(
            output.absolute_qpos_chunk,
            absolute_qpos_chunk,
            rate_reference_qpos,
            self.max_command_delta,
            first_n=self.chunk_first_n,
        )

        anchor_loss = output.action_chunk.sum() * 0
        if self.anchor_backbone is not None:
            reference = self._encode(self.anchor_backbone, inputs, needs_grad=False)
            anchor_terms = [
                1 - F.cosine_similarity(current, frozen, dim=-1).mean()
                for current, frozen in zip(encoded.tokens, reference.tokens, strict=True)
            ]
            anchor_loss = torch.stack(anchor_terms).mean()

        corrupt_order_loss = self._corrupted_order_loss(
            inputs,
            features.tensor.float(),
            current_qpos,
            history_valid,
            lengths,
            generator=generator,
        )
        total = (
            behavior.total
            + self.head.config.contact_loss_weight * contact_loss
            + self.overlap_loss_weight * overlap_loss
            + self.safety_loss_weight * safety["loss"]
            + self.safety_tube_loss_weight * safety_tube["loss"]
            + self.anchor_loss_weight * anchor_loss
            + self.head.config.order_loss_weight * corrupt_order_loss
        )
        metrics = {
            "loss": total.detach(),
            "chunk_loss": behavior.chunk.detach(),
            "phase_loss": behavior.phase.detach(),
            "contact_loss": contact_loss.detach(),
            "clean_order_loss": behavior.order.detach(),
            "corrupt_order_loss": corrupt_order_loss.detach(),
            "overlap_loss": overlap_loss.detach(),
            "safety_loss": safety["loss"].detach(),
            "safety_tube_loss": safety_tube["loss"].detach(),
            "safety_tube_mean_loss": safety_tube["mean_loss"].detach(),
            "safety_tube_tail_loss": safety_tube["tail_loss"].detach(),
            "safety_tube_max_ratio": safety_tube["max_ratio"].detach(),
            "safety_tube_min_slack": safety_tube["min_slack"].detach(),
            "safety_joint_violation_values": safety["joint_violation_values"].float().detach(),
            "safety_rate_violation_values": safety["rate_violation_values"].float().detach(),
            "safety_joint_max_excess_ratio": safety["joint_max_excess_ratio"].detach(),
            "safety_rate_max_ratio": safety["rate_max_ratio"].detach(),
            "anchor_loss": anchor_loss.detach(),
            "mean_prefix_length": lengths.float().mean().detach(),
        }
        return total, metrics

    @torch.no_grad()
    def validation_batch(
        self,
        inputs: ClipInputs,
        state_history: torch.Tensor,
        command_history: torch.Tensor,
        command_present: torch.Tensor,
        target_chunk: torch.Tensor,
        absolute_qpos_chunk: torch.Tensor,
        rate_reference_qpos: torch.Tensor,
        phase_id: torch.Tensor,
        contact: torch.Tensor,
        contact_valid: torch.Tensor,
        *,
        history_lengths: torch.Tensor | None = None,
        oldest_command_present: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate a clean held-out batch under its frozen history scenarios."""

        if (history_lengths is None) != (oldest_command_present is None):
            raise ValueError("validation history length and command parity must be provided together")
        if history_lengths is None:
            history_valid = torch.ones(state_history.shape[:2], dtype=torch.bool, device=state_history.device)
        else:
            inputs, state_history, command_history, command_present, history_valid = (
                _apply_validation_history_scenarios(
                    inputs,
                    state_history,
                    command_history,
                    command_present,
                    history_lengths,
                    oldest_command_present,
                )
            )
        features = build_state_features(
            state_history,
            history_valid=history_valid,
            previous_command=command_history,
            previous_command_present=command_present,
            normalizer=self.normalizer,
            dt=2.0,
        )
        current_qpos = features.qpos[:, -1].float()
        encoded = self._encode(self.backbone, inputs, needs_grad=False)
        output = self.head(encoded, features.tensor.float(), current_qpos)
        deployed_chunk_mask = torch.zeros(
            target_chunk.shape[:2],
            dtype=target_chunk.dtype,
            device=target_chunk.device,
        )
        deployed_chunk_mask[:, 1 : 1 + self.chunk_first_n] = 1
        action_loss = self.head.loss(output, target_chunk, chunk_mask=deployed_chunk_mask).chunk
        phase_loss = F.cross_entropy(output.phase_logits, phase_id)
        clean_order_loss = F.cross_entropy(
            output.order_logits,
            torch.zeros(target_chunk.shape[0], dtype=torch.long, device=target_chunk.device),
        )
        if bool(contact_valid.any()):
            contact_loss_sum = F.binary_cross_entropy_with_logits(
                output.contact_logits[contact_valid, 0],
                contact[contact_valid].float(),
                reduction="sum",
            )
        else:
            contact_loss_sum = output.contact_logits.sum() * 0
        deployed_prediction = output.absolute_qpos_chunk[:, 1 : 1 + self.chunk_first_n]
        deployed_target = absolute_qpos_chunk[:, 1 : 1 + self.chunk_first_n]
        predicted_delta = deployed_prediction[:, 1:] - deployed_prediction[:, :-1]
        target_delta = deployed_target[:, 1:] - deployed_target[:, :-1]
        overlap_loss = F.smooth_l1_loss(
            predicted_delta,
            target_delta,
            beta=self.head.config.smooth_l1_beta,
        )
        safety = control_safety_terms(
            output.absolute_qpos_chunk,
            rate_reference_qpos,
            self.joint_lower,
            self.joint_upper,
            self.max_command_delta,
            rate_margin=self.safety_rate_margin,
            first_n=self.chunk_first_n,
        )
        safety_tube = control_safety_tube_terms(
            output.absolute_qpos_chunk,
            absolute_qpos_chunk,
            rate_reference_qpos,
            self.max_command_delta,
            first_n=self.chunk_first_n,
        )
        action_mae = (deployed_prediction - deployed_target).abs().mean()
        return {
            "action_loss": action_loss,
            "action_mae": action_mae,
            "phase_loss": phase_loss,
            "contact_loss_sum": contact_loss_sum,
            "contact_count": contact_valid.sum(),
            "clean_order_loss": clean_order_loss,
            "overlap_loss": overlap_loss,
            "safety_loss": safety["loss"],
            "safety_tube_loss": safety_tube["loss"],
            "safety_tube_mean_loss": safety_tube["mean_loss"],
            "safety_tube_tail_loss": safety_tube["tail_loss"],
            "safety_tube_max_ratio": safety_tube["max_ratio"],
            "safety_tube_min_slack": safety_tube["min_slack"],
            "joint_violation_values": safety["joint_violation_values"],
            "rate_violation_values": safety["rate_violation_values"],
            "joint_max_excess_ratio": safety["joint_max_excess_ratio"],
            "rate_max_ratio": safety["rate_max_ratio"],
            "examples": torch.tensor(target_chunk.shape[0], device=target_chunk.device),
        }


def _to_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> ClipInputs:
    def image(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)

    return ClipInputs(
        video=image(batch["video"]),
        gel=image(batch["gel"]),
        lowdim=batch["lowdim"].to(device, non_blocking=True).float(),
    )


def _device_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "state_history",
        "command_history",
        "command_present",
        "action_chunk",
        "absolute_qpos_chunk",
        "rate_reference_qpos",
        "phase_id",
        "contact",
        "contact_valid",
        "sample_weight",
    )
    return {name: batch[name].to(device, non_blocking=True) for name in names}


@torch.no_grad()
def evaluate_validation(
    model: ControlV2TrainModel,
    dataset: ControlV2Dataset,
    selection: ValidationSelection,
    cfg: ControlV2TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one frozen, episode-balanced selection from the held-out split."""

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selection.indices),
        batch_size=cfg.local_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_control_v2,
    )
    backbone_training, head_training = model.backbone.training, model.head.training
    model.backbone.eval()
    model.head.eval()
    weighted = {
        "action_loss": 0.0,
        "action_mae": 0.0,
        "phase_loss": 0.0,
        "clean_order_loss": 0.0,
        "overlap_loss": 0.0,
        "safety_loss": 0.0,
        "safety_tube_loss": 0.0,
        "safety_tube_mean_loss": 0.0,
        "safety_tube_tail_loss": 0.0,
    }
    examples = 0
    contact_sum = 0.0
    contact_count = 0
    joint_violation_values = 0
    rate_violation_values = 0
    joint_max_excess_ratio = 0.0
    rate_max_ratio = 0.0
    safety_tube_max_ratio = 0.0
    safety_tube_min_slack = float("inf")
    selection_offset = 0
    try:
        for batch in loader:
            inputs = _to_inputs(batch, device)
            tensors = _device_batch(batch, device)
            metrics = model.validation_batch(
                inputs,
                tensors["state_history"].float(),
                tensors["command_history"].float(),
                tensors["command_present"].bool(),
                tensors["action_chunk"].float(),
                tensors["absolute_qpos_chunk"].float(),
                tensors["rate_reference_qpos"].float(),
                tensors["phase_id"].long(),
                tensors["contact"].float(),
                tensors["contact_valid"].bool(),
                history_lengths=torch.tensor(
                    selection.history_lengths[selection_offset : selection_offset + tensors["state_history"].shape[0]],
                    dtype=torch.long,
                    device=device,
                ),
                oldest_command_present=torch.tensor(
                    selection.oldest_command_present[
                        selection_offset : selection_offset + tensors["state_history"].shape[0]
                    ],
                    dtype=torch.bool,
                    device=device,
                ),
            )
            batch_examples = int(metrics["examples"].item())
            selection_offset += batch_examples
            examples += batch_examples
            for name in weighted:
                weighted[name] += float(metrics[name]) * batch_examples
            contact_sum += float(metrics["contact_loss_sum"])
            contact_count += int(metrics["contact_count"].item())
            joint_violation_values += int(metrics["joint_violation_values"].item())
            rate_violation_values += int(metrics["rate_violation_values"].item())
            joint_max_excess_ratio = max(joint_max_excess_ratio, float(metrics["joint_max_excess_ratio"]))
            rate_max_ratio = max(rate_max_ratio, float(metrics["rate_max_ratio"]))
            safety_tube_max_ratio = max(safety_tube_max_ratio, float(metrics["safety_tube_max_ratio"]))
            safety_tube_min_slack = min(safety_tube_min_slack, float(metrics["safety_tube_min_slack"]))
    finally:
        model.backbone.train(backbone_training)
        model.head.train(head_training)
    if examples == 0:
        raise ValueError("validation split produced no batches")
    result = {f"validation_{name}": value / examples for name, value in weighted.items()}
    result["validation_contact_loss"] = contact_sum / max(contact_count, 1)
    result["validation_contact_count"] = float(contact_count)
    result["validation_joint_violation_values"] = float(joint_violation_values)
    result["validation_rate_violation_values"] = float(rate_violation_values)
    result["validation_safety_violation_values"] = float(joint_violation_values + rate_violation_values)
    result["validation_joint_max_excess_ratio"] = joint_max_excess_ratio
    result["validation_rate_max_ratio"] = rate_max_ratio
    result["validation_safety_tube_max_ratio"] = safety_tube_max_ratio
    result["validation_safety_tube_min_slack"] = safety_tube_min_slack
    result["validation_examples"] = float(examples)
    result["validation_episode_count"] = float(selection.episode_count)
    result["validation_objective"] = (
        result["validation_action_loss"]
        + cfg.head.phase_loss_weight * result["validation_phase_loss"]
        + cfg.head.contact_loss_weight * result["validation_contact_loss"]
        + cfg.head.order_loss_weight * result["validation_clean_order_loss"]
        + cfg.overlap_loss_weight * result["validation_overlap_loss"]
        + cfg.safety_loss_weight * result["validation_safety_loss"]
        + cfg.safety_tube_loss_weight * result["validation_safety_tube_loss"]
    )
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"validation produced non-finite metrics: {result}")
    if examples != len(selection.indices):
        raise RuntimeError(f"validation evaluated {examples} examples, expected {len(selection.indices)}")
    if selection_offset != len(selection.indices):
        raise RuntimeError(f"validation consumed {selection_offset} scenarios, expected {len(selection.indices)}")
    result["validation_sampling"] = VALIDATION_SAMPLING
    result["validation_sample_sha256"] = selection.sample_sha256
    return result


def collective_validation(
    model: ControlV2TrainModel,
    dataset: ControlV2Dataset,
    selection: ValidationSelection,
    cfg: ControlV2TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Run held-out inference on rank zero and broadcast its result or failure."""

    result: dict[str, Any] | None = None
    error: str | None = None
    if runtime.is_main_process():
        try:
            result = evaluate_validation(model, dataset, selection, cfg, device)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if runtime.get_world_size() > 1:
        payload: list[Any] = [result, error]
        dist.broadcast_object_list(payload, src=0)
        result, error = payload
    if error is not None:
        raise RuntimeError(f"held-out validation failed: {error}")
    if result is None:
        raise RuntimeError("held-out validation returned no metrics")
    return result


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def validation_safety_qualified(metrics: dict[str, Any]) -> bool:
    """Whether held-out predictions pass the exact limiter contract with no clamp."""

    required = (
        "validation_joint_violation_values",
        "validation_rate_violation_values",
        "validation_safety_violation_values",
        "validation_joint_max_excess_ratio",
        "validation_rate_max_ratio",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"validation safety metrics are missing: {missing}")
    values = {name: float(metrics[name]) for name in required}
    if not all(np.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError(f"validation safety metrics must be finite and non-negative: {values}")
    return (
        values["validation_joint_violation_values"] == 0
        and values["validation_rate_violation_values"] == 0
        and values["validation_safety_violation_values"] == 0
        and values["validation_joint_max_excess_ratio"] == 0
        and values["validation_rate_max_ratio"] <= 1
    )


def snapshot_best_validation(
    run_dir: pathlib.Path,
    step: int,
    metrics: dict[str, Any],
    bundle: ArtifactBundle,
) -> pathlib.Path:
    """Preserve one selected checkpoint outside numeric-checkpoint pruning."""

    if not validation_safety_qualified(metrics):
        raise ValueError("cannot select a checkpoint with held-out safety-limit violations")

    checkpoint_dir = run_dir / "checkpoints"
    source = checkpoint_dir / str(step)
    if not source.is_dir():
        raise FileNotFoundError(f"cannot select missing checkpoint {source}")
    destination = checkpoint_dir / f"best_validation_{step}"
    if not destination.exists():
        staging = checkpoint_dir / f".best_validation_{step}.{os.getpid()}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging, copy_function=_link_or_copy)
        os.replace(staging, destination)
    marker = {
        "schema_version": CONTROL_V2_ARTIFACT_SCHEMA,
        "checkpoint": str(destination.resolve()),
        "selection_metric": "validation_action_loss",
        "selection_constraint": SELECTION_CONSTRAINT,
        "selection_mode": "min",
        "tie_break": "earliest_step",
        "step": step,
        "artifact_sha256": bundle.artifact_sha256,
        "normalization_sha256": bundle.normalization_sha256,
        "source_store_sha256": bundle.artifact.source_store_sha256,
        **metrics,
    }
    _atomic_write(run_dir / "BEST_VALIDATION.json", json.dumps(marker, indent=2, sort_keys=True) + "\n")
    for obsolete in checkpoint_dir.glob("best_validation_*"):
        if obsolete != destination and obsolete.is_dir():
            shutil.rmtree(obsolete)
    return destination


def _source_backbone_contract(cfg: ControlV2TrainConfig) -> pathlib.Path:
    """Require the byte-identical forecast100k/QK-norm source qualified for this method."""

    if not cfg.pretrained_run:
        raise ValueError("--pretrained-run is required")
    source = pathlib.Path(cfg.pretrained_run).resolve()
    checkpoint = source / "checkpoints" / str(cfg.pretrained_step)
    required = tuple(source / relative for relative in _QUALIFIED_BACKBONE_FILES)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"qualified 100k source is incomplete: {missing}")
    mismatches = {}
    for relative, (expected_size, expected_sha256) in _QUALIFIED_BACKBONE_FILES.items():
        path = source / relative
        actual = (path.stat().st_size, file_sha256(path))
        expected = (expected_size, expected_sha256)
        if actual != expected:
            mismatches[relative] = (actual, expected)
    if mismatches:
        raise ValueError(f"pretrained_run is not the qualified forecast100k/QK-norm snapshot: {mismatches}")
    metadata_path = checkpoint / "metadata.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    if int(metadata.get("global_step", -1)) != cfg.pretrained_step:
        raise ValueError(f"source metadata does not identify step {cfg.pretrained_step}")
    latest = int((source / "checkpoints" / "latest").read_text().strip())
    if latest != cfg.pretrained_step:
        raise ValueError(f"source latest pointer {latest} does not identify step {cfg.pretrained_step}")
    done_path = source / "DONE"
    if done_path.is_file() and int(done_path.read_text().strip()) != cfg.pretrained_step:
        raise ValueError(f"source DONE marker does not identify step {cfg.pretrained_step}")
    return source


_STAGE_EXACT_FIELDS = (
    "task",
    "pretrained_step",
    "require_authoritative_control",
    "minimum_total_episodes",
    "split_identity",
    "split_seed",
    "validation_fraction",
    "observation_stride",
    "action_stride",
    "index_step",
    "video_size",
    "gel_size",
    "lowdim_channels",
    "encoder",
    "predictor",
    "data",
    "head",
    "overlap_loss_weight",
    "anchor_loss_weight",
    "order_batch_fraction",
    "safety_loss_weight",
    "safety_tube_loss_weight",
    "safety_rate_margin",
    "validation_interval",
    "validation_batches",
    "seed",
)


def _stage_initialization(
    cfg: ControlV2TrainConfig,
    bundle: ArtifactBundle,
    *,
    head: ControlV2,
    backbone: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Strictly load one completed prior stage, or stamp a scratch head stage."""

    empty = {
        "init_run": None,
        "init_step": None,
        "init_student_sha256": None,
        "init_backbone_sha256": None,
        "init_config_sha256": None,
    }
    if not cfg.init_run:
        if cfg.backbone_train_mode != "frozen":
            raise ValueError("last-block adaptation must initialize from a completed --init-run")
        return empty

    source_run = pathlib.Path(cfg.init_run).resolve()
    source_done = source_run / "DONE"
    source_latest = source_run / "checkpoints" / "latest"
    source_best = source_run / "BEST_VALIDATION.json"
    if not source_done.is_file() or not source_latest.is_file() or not source_best.is_file():
        raise FileNotFoundError(f"init_run is not complete: {source_run}")
    done_step = int(source_done.read_text().strip())
    latest_step = int(source_latest.read_text().strip())
    source_cfg = ControlV2TrainConfig.from_json((source_run / "run_config.json").read_text())
    if done_step != latest_step or done_step != source_cfg.num_train_steps:
        raise ValueError(
            f"init stage did not complete its configured schedule: done/latest/target="
            f"{done_step}/{latest_step}/{source_cfg.num_train_steps}"
        )
    best_marker = json.loads(source_best.read_text())
    expected_best_marker = {
        "schema_version": CONTROL_V2_ARTIFACT_SCHEMA,
        "selection_metric": "validation_action_loss",
        "selection_constraint": SELECTION_CONSTRAINT,
        "selection_mode": "min",
        "tie_break": "earliest_step",
        "artifact_sha256": bundle.artifact_sha256,
        "normalization_sha256": bundle.normalization_sha256,
        "source_store_sha256": bundle.artifact.source_store_sha256,
    }
    marker_mismatches = {
        key: (best_marker.get(key), value)
        for key, value in expected_best_marker.items()
        if best_marker.get(key) != value
    }
    if marker_mismatches:
        raise ValueError(f"BEST_VALIDATION artifact closure differs: {marker_mismatches}")
    source_validation_manifest = load_validation_manifest(source_run)
    validate_validation_metric_stamp(best_marker, source_validation_manifest)
    selected_step = int(best_marker.get("step", -1))
    if not 0 <= selected_step <= done_step:
        raise ValueError(f"BEST_VALIDATION step {selected_step} is outside completed schedule [0, {done_step}]")
    heldout_metric_names = (
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
    )
    missing_metrics = [name for name in heldout_metric_names if name not in best_marker]
    if missing_metrics or not all(
        np.isfinite(float(best_marker[name])) for name in heldout_metric_names if name in best_marker
    ):
        raise ValueError(f"BEST_VALIDATION has missing/non-finite held-out metrics: {missing_metrics}")
    if not validation_safety_qualified(best_marker):
        raise ValueError("BEST_VALIDATION does not satisfy the zero held-out safety-violation constraint")
    if cfg.init_step is not None and cfg.init_step != selected_step:
        raise ValueError(f"init_step {cfg.init_step} must equal completed run's held-out selection {selected_step}")
    checkpoint = source_run / "checkpoints" / f"best_validation_{selected_step}"
    marker_checkpoint = pathlib.Path(str(best_marker.get("checkpoint", ""))).resolve()
    if marker_checkpoint != checkpoint.resolve():
        raise ValueError(f"BEST_VALIDATION checkpoint {marker_checkpoint} != durable selection {checkpoint}")
    required = (
        source_run / "run_config.json",
        source_run / "artifact.json",
        source_run / "normalization.json",
        source_run / "VALIDATION_SAMPLES.json",
        checkpoint / "student.pt",
        checkpoint / "backbone.pt",
        checkpoint / "loss.pt",
        checkpoint / "optimizer.pt",
        checkpoint / "metadata.pt",
        checkpoint / "train_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"completed init stage lacks closure files: {missing}")
    selected_cfg = ControlV2TrainConfig.from_json((checkpoint / "train_config.json").read_text())
    if selected_cfg != source_cfg:
        raise ValueError("selected checkpoint train_config.json differs from its completed run_config.json")

    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in _STAGE_EXACT_FIELDS:
        source_value = getattr(source_cfg, field)
        target_value = getattr(cfg, field)
        if source_value != target_value:
            mismatches[field] = (source_value, target_value)
    if pathlib.Path(source_cfg.pretrained_run).resolve() != pathlib.Path(cfg.pretrained_run).resolve():
        mismatches["pretrained_run"] = (source_cfg.pretrained_run, cfg.pretrained_run)
    if file_sha256(source_run / "artifact.json") != bundle.artifact_sha256:
        mismatches["artifact_sha256"] = (file_sha256(source_run / "artifact.json"), bundle.artifact_sha256)
    if file_sha256(source_run / "normalization.json") != bundle.normalization_sha256:
        mismatches["normalization_sha256"] = (
            file_sha256(source_run / "normalization.json"),
            bundle.normalization_sha256,
        )
    if mismatches:
        raise ValueError(f"init stage and target scientific contracts differ: {mismatches}")

    metadata = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
    expected_metadata = {
        "control_v2_checkpoint_schema": CONTROL_V2_ARTIFACT_SCHEMA,
        "global_step": selected_step,
        "backbone_train_mode": source_cfg.backbone_train_mode,
        "backbone_last_n_blocks": source_cfg.backbone_last_n_blocks,
        "source_backbone_run": str(pathlib.Path(source_cfg.pretrained_run).resolve()),
        "source_backbone_step": source_cfg.pretrained_step,
        "artifact_sha256": bundle.artifact_sha256,
        "normalization_sha256": bundle.normalization_sha256,
        "source_store_sha256": bundle.artifact.source_store_sha256,
        "normalization_counts": bundle.normalization_counts,
        "best_validation_step": selected_step,
        "best_validation_chunk_loss": float(best_marker["validation_action_loss"]),
    }
    metadata_mismatches = {
        key: (metadata.get(key), value) for key, value in expected_metadata.items() if metadata.get(key) != value
    }
    if metadata_mismatches:
        raise ValueError(f"init checkpoint metadata differs from its artifact closure: {metadata_mismatches}")

    head.load_state_dict(torch.load(checkpoint / "student.pt", map_location=device, weights_only=True), strict=True)
    backbone.load_state_dict(
        torch.load(checkpoint / "backbone.pt", map_location=device, weights_only=True), strict=True
    )
    normalizer_state = torch.load(checkpoint / "loss.pt", map_location="cpu", weights_only=True)
    expected_normalizer = bundle.normalizer.state_dict()
    if set(normalizer_state) != set(expected_normalizer) or any(
        not torch.equal(normalizer_state[name].cpu(), value.cpu()) for name, value in expected_normalizer.items()
    ):
        raise ValueError("init checkpoint loss.pt differs from normalization.json")
    return {
        "init_run": str(source_run),
        "init_step": selected_step,
        "init_student_sha256": file_sha256(checkpoint / "student.pt"),
        "init_backbone_sha256": file_sha256(checkpoint / "backbone.pt"),
        "init_config_sha256": file_sha256(source_run / "run_config.json"),
    }


def optimizer_step_bounds(optimizer: torch.optim.Optimizer) -> tuple[int, int] | None:
    steps: list[int] = []
    for state in optimizer.state.values():
        if "step" in state:
            raw = state["step"]
            steps.append(int(raw.item()) if isinstance(raw, torch.Tensor) else int(raw))
    return (min(steps), max(steps)) if steps else None


def _checkpoint_metadata(
    *,
    cfg: ControlV2TrainConfig,
    bundle: ArtifactBundle,
    source_backbone: pathlib.Path,
    source_backbone_step: int,
    selection: BackboneSelection,
    optimizer: torch.optim.Optimizer,
    world_size: int,
    initialization: dict[str, Any],
    last_metrics: dict[str, float] | None,
    last_validation_metrics: dict[str, Any] | None,
    best_validation_chunk_loss: float | None,
    best_validation_step: int | None,
) -> dict[str, Any]:
    bounds = optimizer_step_bounds(optimizer)
    return {
        "control_v2_checkpoint_schema": CONTROL_V2_ARTIFACT_SCHEMA,
        "backbone_train_mode": cfg.backbone_train_mode,
        "backbone_last_n_blocks": cfg.backbone_last_n_blocks,
        "source_backbone_run": str(source_backbone),
        "source_backbone_step": source_backbone_step,
        "source_store_sha256": bundle.artifact.source_store_sha256,
        "artifact_sha256": bundle.artifact_sha256,
        "normalization_sha256": bundle.normalization_sha256,
        "normalization_counts": bundle.normalization_counts,
        "trainable_backbone_tensors": len(selection.parameters),
        "trainable_backbone_parameters": selection.num_parameters,
        "trainable_backbone_name_sha256": selection.name_digest,
        "trainable_head_parameters": sum(parameter.numel() for parameter in optimizer.param_groups[0]["params"]),
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "local_batch_size": cfg.local_batch_size,
        "world_size": world_size,
        "optimizer_step_min": bounds[0] if bounds else 0,
        "optimizer_step_max": bounds[1] if bounds else 0,
        "last_train_metrics": dict(last_metrics or {}),
        "last_validation_metrics": dict(last_validation_metrics or {}),
        "best_validation_chunk_loss": best_validation_chunk_loss,
        "best_validation_step": best_validation_step,
        **initialization,
    }


def _validate_resume_metadata(
    metadata: dict[str, Any],
    *,
    step: int,
    cfg: ControlV2TrainConfig,
    bundle: ArtifactBundle,
    source_backbone: pathlib.Path,
    source_backbone_step: int,
    selection: BackboneSelection,
    world_size: int,
    initialization: dict[str, Any],
) -> None:
    expected = {
        "control_v2_checkpoint_schema": CONTROL_V2_ARTIFACT_SCHEMA,
        "global_step": step,
        "backbone_train_mode": cfg.backbone_train_mode,
        "backbone_last_n_blocks": cfg.backbone_last_n_blocks,
        "source_backbone_run": str(source_backbone),
        "source_backbone_step": source_backbone_step,
        "source_store_sha256": bundle.artifact.source_store_sha256,
        "artifact_sha256": bundle.artifact_sha256,
        "normalization_sha256": bundle.normalization_sha256,
        "normalization_counts": bundle.normalization_counts,
        "trainable_backbone_name_sha256": selection.name_digest,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "local_batch_size": cfg.local_batch_size,
        "world_size": world_size,
        **initialization,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise RuntimeError(f"resume provenance/topology differs from checkpoint: {mismatches}")


def _init_tracking(cfg: ControlV2TrainConfig, *, resuming: bool) -> None:
    if not cfg.wandb_enabled or not runtime.is_main_process():
        return
    id_path = cfg.run_dir / "wandb_id.txt"
    if resuming and id_path.is_file():
        wandb.init(id=id_path.read_text().strip(), resume="must", project="mot-jepa-control-v2")
    else:
        wandb.init(
            name=cfg.exp_name,
            project="mot-jepa-control-v2",
            config=json.loads(cfg.to_json()),
        )
        _atomic_write(id_path, str(wandb.run.id))


def _worker_init(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)


def train(cfg: ControlV2TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    use_ddp, local_rank, device = runtime.setup_ddp()
    rank, world_size = runtime.get_rank(), runtime.get_world_size()
    try:
        if not cfg.run_dir.is_dir():
            raise FileNotFoundError(f"run the stats job before training: {cfg.run_dir}")
        bundle = _collective_validate_artifacts(cfg)
        if runtime.is_main_process():
            (cfg.run_dir / "FAILED").unlink(missing_ok=True)
        runtime.barrier()
        source_backbone = _source_backbone_contract(cfg)

        torch.manual_seed(cfg.seed + rank)
        np.random.seed(cfg.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed + rank)

        backbone, source_step = runtime.load_frozen_backbone(
            source_backbone,
            cfg.pretrained_step,
            cfg,
            device,
        )
        if source_step != cfg.pretrained_step:
            raise ValueError(f"loaded source backbone step {source_step} != configured {cfg.pretrained_step}")
        selection = configure_backbone_trainability(
            backbone,
            mode=cfg.backbone_train_mode,
            last_n_blocks=cfg.backbone_last_n_blocks,
        )
        backbone.set_gradient_checkpointing(enabled=cfg.gradient_checkpointing and cfg.backbone_train_mode != "frozen")
        head = ControlV2(cfg.head, cfg.layout).to(device).float()
        head.load_action_scale(torch.tensor(bundle.artifact.action_scale, dtype=torch.float32, device=device))
        normalizer = bundle.normalizer.to(device)
        initialization = _stage_initialization(
            cfg,
            bundle,
            head=head,
            backbone=backbone,
            device=device,
        )

        anchor_backbone: nn.Module | None = None
        if cfg.backbone_train_mode == "last_blocks":
            anchor_backbone, anchor_step = runtime.load_frozen_backbone(
                source_backbone,
                cfg.pretrained_step,
                cfg,
                device,
            )
            if anchor_step != source_step:
                raise ValueError(f"anchor/source backbone steps differ: {anchor_step} != {source_step}")
            anchor_backbone.eval().requires_grad_(requires_grad=False)

        dataset = build_dataset(cfg, split="train")
        validation_dataset = build_dataset(cfg, split="validation")
        train_episodes = {(int(row[0]), int(row[3])) for row in dataset.clip_index.entries}
        validation_episodes = {(int(row[0]), int(row[3])) for row in validation_dataset.clip_index.entries}
        overlap = train_episodes & validation_episodes
        if overlap:
            raise RuntimeError(f"train/validation episode leakage detected: {sorted(overlap)[:8]}")
        validation_selection = build_validation_selection(
            validation_dataset,
            cfg.validation_batches * cfg.local_batch_size,
        )
        freeze_validation_selection(cfg.run_dir, validation_selection)
        validation_manifest = load_validation_manifest(cfg.run_dir)
        logger.info(
            "V3 data: %d/%d train/validation clips from %d/%d disjoint episodes; "
            "%d fixed validation clips across %d episodes",
            len(dataset),
            len(validation_dataset),
            len(train_episodes),
            len(validation_episodes),
            len(validation_selection.indices),
            validation_selection.episode_count,
        )
        sampler = InfiniteBatchSampler(
            len(dataset),
            cfg.local_batch_size,
            rank=rank,
            world_size=world_size,
            seed=cfg.seed,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(cfg.seed + rank)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_control_v2,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            worker_init_fn=_worker_init,
            generator=loader_generator,
        )

        head_parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
        groups: list[dict[str, Any]] = [
            {"params": head_parameters, "lr": cfg.lr_head, "lr_multiplier": 1.0, "group_name": "head"}
        ]
        if selection.parameters:
            groups.append(
                {
                    "params": selection.parameters,
                    "lr": cfg.lr_backbone,
                    "lr_multiplier": cfg.lr_backbone / cfg.lr_head,
                    "group_name": "backbone",
                }
            )
        optimizer = torch.optim.AdamW(
            groups,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

        global_step = 0
        last_validation_metrics: dict[str, Any] | None = None
        best_validation_chunk_loss: float | None = None
        best_validation_step: int | None = None
        resume_step = runtime.find_latest_step(cfg.checkpoint_dir)
        if resume_step is not None:
            checkpoint = cfg.checkpoint_dir / str(resume_step)
            metadata = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
            _validate_resume_metadata(
                metadata,
                step=resume_step,
                cfg=cfg,
                bundle=bundle,
                source_backbone=source_backbone,
                source_backbone_step=source_step,
                selection=selection,
                world_size=world_size,
                initialization=initialization,
            )
            raw_last_validation = metadata.get("last_validation_metrics", {})
            if not isinstance(raw_last_validation, dict):
                raise RuntimeError("resume last_validation_metrics is not a dictionary")
            last_validation_metrics = {str(name): value for name, value in raw_last_validation.items()} or None
            if last_validation_metrics is not None:
                validate_validation_metric_stamp(last_validation_metrics, validation_manifest)
            raw_best_loss = metadata.get("best_validation_chunk_loss")
            raw_best_step = metadata.get("best_validation_step")
            if (raw_best_loss is None) != (raw_best_step is None):
                raise RuntimeError("resume checkpoint has a partial best-validation selection")
            if raw_best_loss is not None:
                best_validation_chunk_loss = float(raw_best_loss)
                best_validation_step = int(raw_best_step)
                best_marker_path = cfg.run_dir / "BEST_VALIDATION.json"
                if not best_marker_path.is_file():
                    raise FileNotFoundError(
                        "resume metadata names a best checkpoint but BEST_VALIDATION.json is missing"
                    )
                best_marker = json.loads(best_marker_path.read_text())
                validate_validation_metric_stamp(best_marker, validation_manifest)
                expected_best = {
                    "schema_version": CONTROL_V2_ARTIFACT_SCHEMA,
                    "selection_metric": "validation_action_loss",
                    "selection_constraint": SELECTION_CONSTRAINT,
                    "selection_mode": "min",
                    "tie_break": "earliest_step",
                    "step": best_validation_step,
                    "artifact_sha256": bundle.artifact_sha256,
                    "normalization_sha256": bundle.normalization_sha256,
                    "source_store_sha256": bundle.artifact.source_store_sha256,
                    "validation_action_loss": best_validation_chunk_loss,
                }
                best_mismatches = {
                    key: (best_marker.get(key), value)
                    for key, value in expected_best.items()
                    if best_marker.get(key) != value
                }
                durable = cfg.checkpoint_dir / f"best_validation_{best_validation_step}"
                if pathlib.Path(str(best_marker.get("checkpoint", ""))).resolve() != durable.resolve():
                    best_mismatches["checkpoint"] = (best_marker.get("checkpoint"), str(durable.resolve()))
                if not durable.is_dir():
                    best_mismatches["durable_checkpoint"] = (False, True)
                if best_mismatches:
                    raise RuntimeError(f"resume best-validation closure differs: {best_mismatches}")
                if not validation_safety_qualified(best_marker):
                    raise RuntimeError("resume BEST_VALIDATION violates its zero-safety selection constraint")
            global_step = runtime.load_checkpoint(
                cfg.checkpoint_dir,
                resume_step,
                student=head,
                teacher=None,
                optimizer=optimizer,
                device=device,
                loss_fn=normalizer,
                extra_modules={"backbone": backbone},
            )
            bounds = optimizer_step_bounds(optimizer)
            if bounds is not None and bounds != (global_step, global_step):
                raise RuntimeError(f"optimizer counters {bounds} do not equal resumed update {global_step}")
            logger.info("resumed exact V3 state from update %d", global_step)

        done_path = cfg.run_dir / "DONE"
        if done_path.exists():
            done_step = int(done_path.read_text().strip())
            if done_step != cfg.num_train_steps or global_step != done_step:
                raise ValueError(
                    f"DONE/latest/config disagree: done={done_step}, latest={global_step}, target={cfg.num_train_steps}"
                )
            logger.info("run already complete at update %d", done_step)
            return

        sampler.set_start_step(global_step * cfg.gradient_accumulation_steps)
        _init_tracking(cfg, resuming=resume_step is not None)
        train_model = ControlV2TrainModel(
            backbone,
            head,
            normalizer,
            train_backbone=bool(selection.parameters),
            anchor_backbone=anchor_backbone,
            overlap_loss_weight=cfg.overlap_loss_weight,
            anchor_loss_weight=cfg.anchor_loss_weight,
            order_batch_fraction=cfg.order_batch_fraction,
            safety_loss_weight=cfg.safety_loss_weight,
            safety_tube_loss_weight=cfg.safety_tube_loss_weight,
            safety_rate_margin=cfg.safety_rate_margin,
            joint_lower=bundle.artifact.joint_lower,
            joint_upper=bundle.artifact.joint_upper,
            max_command_delta=bundle.artifact.max_command_delta,
            chunk_first_n=bundle.artifact.chunk_first_n,
        )
        model: nn.Module = train_model
        if use_ddp:
            model = DistributedDataParallel(
                train_model,
                device_ids=[local_rank],
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )

        monitor = runtime.PreemptionMonitor(cfg.run_dir / "PREEMPT_REQUEST", device)
        iterator = iter(loader)
        records: list[dict[str, float]] = []
        last_metrics: dict[str, float] | None = None
        window_samples = 0
        window_start = time.time()
        preempted = False
        last_saved_step: int | None = resume_step

        if resume_step is None:
            last_validation_metrics = collective_validation(
                train_model,
                validation_dataset,
                validation_selection,
                cfg,
                device,
            )
            initial_safety_qualified = validation_safety_qualified(last_validation_metrics)
            best_validation_chunk_loss = (
                last_validation_metrics["validation_action_loss"] if initial_safety_qualified else None
            )
            best_validation_step = 0 if initial_safety_qualified else None
            if runtime.is_main_process():
                logger.info(
                    "initial validation: chunk %.5f mae %.5f phase %.4f contact %.4f overlap %.5f "
                    "tube=%.4f tube_max=%.4f safety=%d rate_max=%.4f qualified=%s",
                    last_validation_metrics["validation_action_loss"],
                    last_validation_metrics["validation_action_mae"],
                    last_validation_metrics["validation_phase_loss"],
                    last_validation_metrics["validation_contact_loss"],
                    last_validation_metrics["validation_overlap_loss"],
                    last_validation_metrics["validation_safety_tube_loss"],
                    last_validation_metrics["validation_safety_tube_max_ratio"],
                    int(last_validation_metrics["validation_safety_violation_values"]),
                    last_validation_metrics["validation_rate_max_ratio"],
                    initial_safety_qualified,
                )
                runtime.save_checkpoint(
                    cfg.checkpoint_dir,
                    0,
                    student=head,
                    teacher=None,
                    optimizer=optimizer,
                    config_json=cfg.to_json(),
                    loss_fn=normalizer,
                    extra_modules={"backbone": backbone},
                    keep_last=cfg.keep_last,
                    keep_period=cfg.keep_period,
                    extra=_checkpoint_metadata(
                        cfg=cfg,
                        bundle=bundle,
                        source_backbone=source_backbone,
                        source_backbone_step=source_step,
                        selection=selection,
                        optimizer=optimizer,
                        world_size=world_size,
                        initialization=initialization,
                        last_metrics=None,
                        last_validation_metrics=last_validation_metrics,
                        best_validation_chunk_loss=best_validation_chunk_loss,
                        best_validation_step=best_validation_step,
                    ),
                )
                if initial_safety_qualified:
                    snapshot_best_validation(cfg.run_dir, 0, last_validation_metrics, bundle)
                if cfg.wandb_enabled:
                    wandb.log(
                        {
                            name: value
                            for name, value in last_validation_metrics.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                        },
                        step=0,
                    )
            runtime.barrier()
            last_saved_step = 0

        while global_step < cfg.num_train_steps:
            if monitor.should_stop(global_step):
                preempted = True
                logger.info("preemption requested at optimizer update %d", global_step)
                break

            optimizer.zero_grad(set_to_none=True)
            update_metrics: dict[str, float] = {}
            next_step = global_step + 1
            for micro_step in range(cfg.gradient_accumulation_steps):
                batch = next(iterator)
                inputs = _to_inputs(batch, device)
                tensors = _device_batch(batch, device)
                random_step = global_step * cfg.gradient_accumulation_steps + micro_step
                generator = make_step_generator(device, base_seed=cfg.seed, step=random_step, rank=rank)
                synchronise = micro_step == cfg.gradient_accumulation_steps - 1
                sync_context = contextlib.nullcontext() if synchronise or not use_ddp else model.no_sync()
                with sync_context:
                    loss, metrics = model(
                        inputs,
                        tensors["state_history"].float(),
                        tensors["command_history"].float(),
                        tensors["command_present"].bool(),
                        tensors["action_chunk"].float(),
                        tensors["absolute_qpos_chunk"].float(),
                        tensors["rate_reference_qpos"].float(),
                        tensors["phase_id"].long(),
                        tensors["contact"].float(),
                        tensors["contact_valid"].bool(),
                        tensors["sample_weight"].float(),
                        generator=generator,
                    )
                    if not bool(torch.isfinite(loss.detach())):
                        raise FloatingPointError(
                            f"non-finite V3 loss at update/micro {next_step}/{micro_step}: {float(loss.detach())}"
                        )
                    (loss / cfg.gradient_accumulation_steps).backward()
                for name, value in metrics.items():
                    update_metrics[name] = (
                        update_metrics.get(name, 0.0) + float(value) / cfg.gradient_accumulation_steps
                    )

            missing_head = [
                name
                for name, parameter in head.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            missing_backbone = [
                name
                for name, parameter in backbone.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing_head or missing_backbone:
                raise RuntimeError(
                    f"trainable tensors missing gradients at update {next_step}: "
                    f"head={missing_head[:8]}, backbone={missing_backbone[:8]}"
                )

            head_grad = grad_norm(head_parameters, device=device)
            backbone_grad = grad_norm(selection.parameters, device=device)
            trainable = (*head_parameters, *selection.parameters)
            total_grad = torch.nn.utils.clip_grad_norm_(trainable, cfg.clip_grad_norm)
            norms = torch.stack((head_grad, backbone_grad, total_grad.float()))
            if not bool(torch.isfinite(norms).all()):
                raise FloatingPointError(f"non-finite gradient norms at update {next_step}: {norms.tolist()}")

            head_lr = lr_at(
                global_step,
                peak=cfg.lr_head,
                end=cfg.lr_head * cfg.lr_end_ratio,
                warmup=cfg.warmup_steps,
                total=cfg.num_train_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = head_lr * float(group["lr_multiplier"])
            optimizer.step()
            global_step = next_step
            update_metrics.update(
                {
                    "learning_rate": head_lr,
                    "grad_norm": float(total_grad),
                    "head_grad_norm": float(head_grad),
                    "backbone_grad_norm": float(backbone_grad),
                }
            )
            records.append(update_metrics)
            last_metrics = update_metrics
            window_samples += cfg.local_batch_size * world_size * cfg.gradient_accumulation_steps

            if global_step % cfg.log_interval == 0:
                metrics = reduce_metrics(records, device)
                records.clear()
                elapsed = max(time.time() - window_start, 1e-6)
                metrics["samples_per_second"] = window_samples / elapsed
                window_samples, window_start = 0, time.time()
                if runtime.is_main_process():
                    logger.info(
                        "update %d loss %.4f chunk %.4f anchor %.4f grad %.3f lr %.2e %.1f samples/s",
                        global_step,
                        metrics["loss"],
                        metrics["chunk_loss"],
                        metrics["anchor_loss"],
                        metrics["grad_norm"],
                        head_lr,
                        metrics["samples_per_second"],
                    )
                    if cfg.wandb_enabled:
                        wandb.log({f"control_v2/{name}": value for name, value in metrics.items()}, step=global_step)

            validation_due = global_step % cfg.validation_interval == 0 or global_step == cfg.num_train_steps
            selected_new_best = False
            if validation_due:
                last_validation_metrics = collective_validation(
                    train_model,
                    validation_dataset,
                    validation_selection,
                    cfg,
                    device,
                )
                candidate = last_validation_metrics["validation_action_loss"]
                candidate_safety_qualified = validation_safety_qualified(last_validation_metrics)
                if candidate_safety_qualified and (
                    best_validation_chunk_loss is None or candidate < best_validation_chunk_loss
                ):
                    best_validation_chunk_loss = candidate
                    best_validation_step = global_step
                    selected_new_best = True
                if runtime.is_main_process():
                    logger.info(
                        "validation update %d: chunk %.5f mae %.5f phase %.4f contact %.4f overlap %.5f "
                        "tube=%.4f tube_max=%.4f safety=%d rate_max=%.4f qualified=%s; "
                        "selected=%s@%s",
                        global_step,
                        last_validation_metrics["validation_action_loss"],
                        last_validation_metrics["validation_action_mae"],
                        last_validation_metrics["validation_phase_loss"],
                        last_validation_metrics["validation_contact_loss"],
                        last_validation_metrics["validation_overlap_loss"],
                        last_validation_metrics["validation_safety_tube_loss"],
                        last_validation_metrics["validation_safety_tube_max_ratio"],
                        int(last_validation_metrics["validation_safety_violation_values"]),
                        last_validation_metrics["validation_rate_max_ratio"],
                        candidate_safety_qualified,
                        best_validation_chunk_loss,
                        best_validation_step,
                    )
                    if cfg.wandb_enabled:
                        wandb.log(
                            {
                                name: value
                                for name, value in last_validation_metrics.items()
                                if isinstance(value, (int, float)) and not isinstance(value, bool)
                            },
                            step=global_step,
                        )

            if global_step % cfg.save_interval == 0 or validation_due:
                if runtime.is_main_process():
                    runtime.save_checkpoint(
                        cfg.checkpoint_dir,
                        global_step,
                        student=head,
                        teacher=None,
                        optimizer=optimizer,
                        config_json=cfg.to_json(),
                        loss_fn=normalizer,
                        extra_modules={"backbone": backbone},
                        keep_last=cfg.keep_last,
                        keep_period=cfg.keep_period,
                        extra=_checkpoint_metadata(
                            cfg=cfg,
                            bundle=bundle,
                            source_backbone=source_backbone,
                            source_backbone_step=source_step,
                            selection=selection,
                            optimizer=optimizer,
                            world_size=world_size,
                            initialization=initialization,
                            last_metrics=last_metrics,
                            last_validation_metrics=last_validation_metrics,
                            best_validation_chunk_loss=best_validation_chunk_loss,
                            best_validation_step=best_validation_step,
                        ),
                    )
                    if selected_new_best:
                        snapshot_best_validation(
                            cfg.run_dir,
                            global_step,
                            last_validation_metrics,
                            bundle,
                        )
                runtime.barrier()
                last_saved_step = global_step

        if last_saved_step != global_step:
            if runtime.is_main_process():
                runtime.save_checkpoint(
                    cfg.checkpoint_dir,
                    global_step,
                    student=head,
                    teacher=None,
                    optimizer=optimizer,
                    config_json=cfg.to_json(),
                    loss_fn=normalizer,
                    extra_modules={"backbone": backbone},
                    keep_last=cfg.keep_last,
                    keep_period=cfg.keep_period,
                    extra=_checkpoint_metadata(
                        cfg=cfg,
                        bundle=bundle,
                        source_backbone=source_backbone,
                        source_backbone_step=source_step,
                        selection=selection,
                        optimizer=optimizer,
                        world_size=world_size,
                        initialization=initialization,
                        last_metrics=last_metrics,
                        last_validation_metrics=last_validation_metrics,
                        best_validation_chunk_loss=best_validation_chunk_loss,
                        best_validation_step=best_validation_step,
                    ),
                )
            runtime.barrier()
        if not preempted and global_step == cfg.num_train_steps:
            if best_validation_step is None or best_validation_chunk_loss is None:
                raise RuntimeError(
                    "training completed without any checkpoint satisfying zero held-out safety-limit violations"
                )
            if runtime.is_main_process():
                _atomic_write(cfg.run_dir / "DONE", f"{global_step}\n")
                logger.info("V3 training complete at optimizer update %d", global_step)
            runtime.barrier()
        if runtime.is_main_process() and cfg.wandb_enabled:
            wandb.finish()
    finally:
        runtime.cleanup_ddp()


def main() -> None:
    cfg = control_v2_cli()
    try:
        train(cfg)
    except Exception as exc:
        # The traceback remains authoritative.  The marker gives a SLURM monitor a non-silent
        # terminal state without relying on pgrep or completion text.
        logger.exception("MoT-Control V3 training failed")
        try:
            if int(os.environ.get("RANK", "0")) == 0:
                _atomic_write(cfg.run_dir / "FAILED", f"{type(exc).__name__}: {exc}\n")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
