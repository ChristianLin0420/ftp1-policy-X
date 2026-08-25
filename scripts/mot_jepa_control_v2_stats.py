#!/usr/bin/env python
"""Fit the train-only normalization and safety artifact for MoT-Control V3."""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import pathlib
import tempfile

import numpy as np
import torch
import zarr

from openpi.mot_jepa.control_v2_config import CONTROL_V2_ARTIFACT_SCHEMA
from openpi.mot_jepa.control_v2_config import CONTROL_V2_PRODUCTION_EPISODES
from openpi.mot_jepa.control_v2_config import ControlV2Artifact
from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.control_v2_config import control_v2_cli
from openpi.mot_jepa.control_v2_data import ACTION_HORIZON
from openpi.mot_jepa.control_v2_data import PHASE_NAMES
from openpi.mot_jepa.control_v2_data import control_v2_chunk_indices
from openpi.mot_jepa.control_v2_data import extract_qpos8
from openpi.mot_jepa.control_v2_data import fit_control_v2_normalization
from openpi.mot_jepa.control_v2_data import save_normalization_artifact
from openpi.mot_jepa.control_v2_dataset import CONTROL_V2_DATA_SCHEMA
from openpi.mot_jepa.control_v2_dataset import CONTROL_V2_SPLIT_IDENTITY
from openpi.mot_jepa.control_v2_dataset import ControlV2Dataset
from scripts.mot_jepa_control_v2_train import ValidationSelection
from scripts.mot_jepa_control_v2_train import build_validation_selection
from scripts.mot_jepa_control_v2_train import freeze_validation_selection

PANDA_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0)
PANDA_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04)


def command_target_residuals(
    qpos: np.ndarray,
    command: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """Physical residuals for command rows t+1:t+32 relative to observed qpos at t."""

    qpos = np.asarray(qpos, dtype=np.float32)
    command = np.asarray(command, dtype=np.float32)
    current = np.asarray(current, dtype=np.int64).reshape(-1)
    if qpos.shape != command.shape or qpos.ndim != 2 or qpos.shape[1] != 8:
        raise ValueError("qpos and command must have matching (T, 8) shapes")
    future = current[:, None] + np.arange(1, ACTION_HORIZON, dtype=np.int64)[None, :]
    if current.size == 0 or np.any(current < 0) or np.any(future >= qpos.shape[0]):
        raise ValueError("current indices do not admit 31 future commands")
    return command[future] - qpos[current, None]


def conservative_command_delta_limit(
    qpos: np.ndarray,
    command: np.ndarray,
    episode_ends: np.ndarray,
    *,
    episode_indices: np.ndarray | None = None,
    margin: float = 0.1,
) -> np.ndarray:
    """Maximum expert ``|command[t+1] - qpos[t]|`` over selected episodes.

    ``episode_indices`` is used by artifact fitting to exclude the held-out validation episodes
    from this learned deployment limit.  The default remains all episodes for standalone audits.
    """

    qpos = np.asarray(qpos, dtype=np.float32)
    command = np.asarray(command, dtype=np.float32)
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if qpos.shape != command.shape or qpos.ndim != 2 or qpos.shape[1] != 8:
        raise ValueError("qpos and command must have matching (T, 8) shapes")
    if ends.size == 0 or int(ends[-1]) != qpos.shape[0] or not 0 <= margin <= 1:
        raise ValueError("episode_ends/margin do not define a valid command stream")
    selected = (
        np.arange(ends.size, dtype=np.int64)
        if episode_indices is None
        else np.unique(np.asarray(episode_indices, dtype=np.int64).reshape(-1))
    )
    if selected.size == 0 or np.any((selected < 0) | (selected >= ends.size)):
        raise ValueError("episode_indices must identify at least one valid episode")
    deltas: list[np.ndarray] = []
    for episode_idx in selected:
        begin = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
        raw_end = ends[episode_idx]
        end = int(raw_end)
        if end - begin > 1:
            deltas.append(np.abs(command[begin + 1 : end] - qpos[begin : end - 1]))
    if not deltas:
        raise ValueError("selected episodes contain no within-episode command transitions")
    return np.maximum(np.max(np.concatenate(deltas), axis=0) * (1 + margin) + 1e-5, 1e-5)


def discover_stores(pattern: str) -> list[str]:
    stores = sorted(glob.glob(pattern))
    if not stores:
        raise ValueError(f"no stores matched {pattern!r}")
    if any(not pathlib.Path(path).is_dir() for path in stores):
        raise ValueError("every store match must be a Zarr directory")
    return stores


def store_contract_digest(stores: list[str]) -> str:
    """Digest schemas and episode boundaries without rereading multi-terabyte image payloads."""
    digest = hashlib.sha256()
    for store_idx, path in enumerate(stores):
        group = zarr.open_group(path, mode="r")
        # Store paths are job-owned preparation outputs, not scientific identities.  The ordered
        # store slot plus content contract is stable across byte-identical rebuilds.
        digest.update(f"store={store_idx}".encode())
        for section in ("data", "meta"):
            for name in sorted(group[section].array_keys()):
                array = group[section][name]
                digest.update(f"{section}/{name}:{array.shape}:{array.dtype}".encode())
        digest.update(np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64).tobytes())
        if "source_episode_seed" in set(group["meta"].array_keys()):
            digest.update(np.asarray(group["meta"]["source_episode_seed"][:], dtype=np.int64).tobytes())
        # State/command/labels are the complete non-image control contract and cheap enough to
        # hash fully. Image chunks are closed separately by the prepared-data content manifest.
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


def validate_source_control_contract(
    stores: list[str],
    *,
    require_authoritative: bool,
    minimum_total_episodes: int,
    exact_total_episodes: int | None = None,
) -> tuple[bool, int, tuple[int, int, int, int], tuple[int, int]]:
    """Fail before fitting when production collection labels are missing or incomplete."""

    required = {
        "command8",
        "command_valid",
        "phase_id",
        "contact",
        "contact_valid",
        "control_step",
        "control_step_valid",
    }
    authoritative = True
    total_episodes = 0
    phase_counts = np.zeros(4, dtype=np.int64)
    contact_counts = np.zeros(2, dtype=np.int64)
    source_episode_seeds: set[int] = set()
    for path in stores:
        group = zarr.open_group(path, mode="r")
        data = group["data"]
        meta = group["meta"]
        keys = set(data.array_keys())
        missing = required - keys
        if missing:
            authoritative = False
            if require_authoritative:
                raise ValueError(f"{path} lacks authoritative V3 arrays: {sorted(missing)}")
            continue
        if "source_episode_seed" not in set(meta.array_keys()):
            if require_authoritative:
                raise ValueError(f"{path} lacks authoritative meta/source_episode_seed")
        else:
            raw_seeds = meta["source_episode_seed"]
            if not np.issubdtype(raw_seeds.dtype, np.integer):
                raise ValueError(f"{path}: meta/source_episode_seed must be integer")
            episode_seeds = np.asarray(raw_seeds[:], dtype=np.int64).reshape(-1)
            episode_count = int(meta["episode_ends"].shape[0])
            if episode_seeds.shape != (episode_count,) or np.any(episode_seeds < 0):
                raise ValueError(f"{path}: meta/source_episode_seed must contain one non-negative seed per episode")
            episode_seed_set = {int(value) for value in episode_seeds}
            duplicate_seeds = source_episode_seeds.intersection(episode_seed_set)
            if duplicate_seeds or len(episode_seed_set) != episode_count:
                raise ValueError(f"source episode seeds must be globally unique: {sorted(duplicate_seeds)[:8]}")
            source_episode_seeds.update(episode_seed_set)
        rows = int(data["state"].shape[0])
        for name in required:
            if data[name].shape[0] != rows:
                raise ValueError(f"{path}: data/{name} is not time-aligned with data/state")
        command = np.asarray(data["command8"][:], dtype=np.float32)
        if command.shape != (rows, 8) or not np.isfinite(command).all():
            raise ValueError(f"{path}: data/command8 must be finite with shape ({rows}, 8)")
        command_valid = np.asarray(data["command_valid"][:], dtype=np.uint8).reshape(-1)
        contact_valid = np.asarray(data["contact_valid"][:], dtype=np.uint8).reshape(-1)
        step_valid = np.asarray(data["control_step_valid"][:], dtype=np.uint8).reshape(-1)
        if require_authoritative and (
            not np.all(command_valid == 1) or not np.all(contact_valid == 1) or not np.all(step_valid == 1)
        ):
            raise ValueError(f"{path}: every production command/contact/control-step row must be authoritative")
        authoritative &= bool(np.all(command_valid == 1) and np.all(contact_valid == 1) and np.all(step_valid == 1))
        steps = np.asarray(data["control_step"][:], dtype=np.int64).reshape(-1)
        begin = 0
        for raw_end in np.asarray(meta["episode_ends"][:], dtype=np.int64):
            end = int(raw_end)
            if np.any(np.diff(steps[begin:end]) != 1):
                raise ValueError(f"{path}: saved rows are not one simulator control step apart")
            begin = end
        phases = np.asarray(data["phase_id"][:], dtype=np.int64).reshape(-1)
        if np.any((phases < 0) | (phases > 3)):
            raise ValueError(f"{path}: data/phase_id contains values outside [0, 3]")
        phase_counts += np.bincount(phases, minlength=4)[:4]
        contact = np.asarray(data["contact"][:], dtype=np.float32).reshape(-1)
        if not np.isfinite(contact).all() or np.any((contact < 0) | (contact > 1)):
            raise ValueError(f"{path}: data/contact must contain finite binary/probability labels")
        binary_contact = contact >= 0.5
        contact_counts += np.array([np.count_nonzero(~binary_contact), np.count_nonzero(binary_contact)])
        total_episodes += int(meta["episode_ends"].shape[0])

    if exact_total_episodes is not None and total_episodes != exact_total_episodes:
        raise ValueError(f"source has {total_episodes} authoritative episodes; require exactly {exact_total_episodes}")
    if exact_total_episodes is None and total_episodes < minimum_total_episodes:
        raise ValueError(
            f"source has {total_episodes} authoritative episodes; require at least {minimum_total_episodes}"
        )
    if np.any(phase_counts == 0):
        raise ValueError(f"source has empty control phases: counts={phase_counts.tolist()}")
    if np.any(contact_counts == 0):
        raise ValueError(f"source contact target has only one class: counts={contact_counts.tolist()}")
    if require_authoritative and len(source_episode_seeds) != total_episodes:
        raise ValueError("authoritative source episode seeds do not close over every episode")
    return (
        authoritative,
        total_episodes,
        tuple(int(value) for value in phase_counts),
        tuple(int(value) for value in contact_counts),
    )


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = pathlib.Path(raw_tmp)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class ValidationOracleSafetyAudit:
    examples: int
    values: int
    violation_values: int
    min_slack: float
    max_rate_ratio: float


def validate_validation_oracle_safety(
    dataset: ControlV2Dataset,
    dataset_indices: tuple[int, ...],
    max_command_delta: np.ndarray,
    *,
    first_n: int,
) -> ValidationOracleSafetyAudit:
    """Prove on CPU that the frozen expert panel is feasible under the deployment rate limit."""

    if not dataset_indices or len(set(dataset_indices)) != len(dataset_indices):
        raise ValueError("oracle safety audit requires unique validation dataset indices")
    if not 1 <= first_n <= ACTION_HORIZON - 1:
        raise ValueError("oracle safety first_n must select executable H32 rows")
    limits = np.asarray(max_command_delta, dtype=np.float32).reshape(-1)
    if limits.shape != (8,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("oracle safety max_command_delta must contain eight finite positive values")

    entries = np.asarray(dataset.clip_index.entries, dtype=np.int64)
    if any(index < 0 or index >= len(entries) for index in dataset_indices):
        raise ValueError("oracle safety selection contains an out-of-range dataset index")
    history_steps = int(dataset.base.layout.num_frames)
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    min_slack = float("inf")
    max_ratio = 0.0
    violations = 0
    first_failure: tuple[int, int, int, int, float, float] | None = None

    for dataset_index in dataset_indices:
        store_idx, start, stride, episode_idx = (int(value) for value in entries[dataset_index])
        if store_idx not in cache:
            group = zarr.open_group(dataset.store_paths[store_idx], mode="r")
            data = group["data"]
            state = torch.from_numpy(np.asarray(data["state"][:], dtype=np.float32))
            qpos = extract_qpos8(state).numpy()
            command = np.asarray(data["command8"][:], dtype=np.float32)
            command_valid = np.asarray(data["command_valid"][:], dtype=np.uint8).astype(bool)
            episode_ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
            cache[store_idx] = qpos, command, command_valid, episode_ends
        qpos, command, command_valid, episode_ends = cache[store_idx]
        episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        episode_end = int(episode_ends[episode_idx])
        current = start + (history_steps - 1) * stride
        future = control_v2_chunk_indices(
            current,
            episode_start=episode_start,
            episode_end=episode_end,
            action_stride=dataset.action_stride,
        ).numpy()
        executable = future[1 : 1 + first_n]
        reference = qpos[future[:first_n]]
        expert = np.where(command_valid[executable, None], command[executable], qpos[executable])
        delta = np.abs(expert.astype(np.float32) - reference.astype(np.float32))
        slack = limits[None] - delta
        ratio = delta / limits[None]
        finite = np.isfinite(slack) & np.isfinite(ratio)
        invalid = (~finite) | (slack <= 0)
        invalid_count = int(np.count_nonzero(invalid))
        violations += invalid_count
        min_slack = min(min_slack, float(np.nanmin(slack)))
        max_ratio = max(max_ratio, float(np.nanmax(ratio)))
        if invalid_count and first_failure is None:
            horizon, joint = (int(value) for value in np.argwhere(invalid)[0])
            first_failure = (
                dataset_index,
                episode_idx,
                horizon + 1,
                joint,
                float(delta[horizon, joint]),
                float(slack[horizon, joint]),
            )

    values = len(dataset_indices) * first_n * 8
    if violations or not np.isfinite(min_slack) or not np.isfinite(max_ratio):
        raise ValueError(
            "frozen validation expert is infeasible under the deployment rate limit: "
            f"violations={violations}/{values}, min_slack={min_slack:.9g}, "
            f"max_rate_ratio={max_ratio:.9g}, first={first_failure}"
        )
    return ValidationOracleSafetyAudit(
        examples=len(dataset_indices),
        values=values,
        violation_values=0,
        min_slack=min_slack,
        max_rate_ratio=max_ratio,
    )


def fit_artifacts(
    cfg: ControlV2TrainConfig,
) -> tuple[object, ControlV2Artifact, ValidationSelection, ValidationOracleSafetyAudit]:
    stores = discover_stores(cfg.store_glob)
    authoritative, source_episode_count, _raw_phase_counts, _raw_contact_counts = validate_source_control_contract(
        stores,
        require_authoritative=cfg.require_authoritative_control,
        minimum_total_episodes=cfg.minimum_total_episodes,
        exact_total_episodes=CONTROL_V2_PRODUCTION_EPISODES if cfg.require_authoritative_control else None,
    )
    train = ControlV2Dataset(
        stores,
        cfg.layout,
        observation_stride=cfg.observation_stride,
        action_stride=cfg.action_stride,
        control_step_stride=1,
        index_step=cfg.index_step,
        lowdim_channels=cfg.lowdim_channels,
        split="train",
        validation_fraction=cfg.validation_fraction,
        split_seed=cfg.split_seed,
    )
    validation = ControlV2Dataset(
        stores,
        cfg.layout,
        observation_stride=cfg.observation_stride,
        action_stride=cfg.action_stride,
        control_step_stride=1,
        index_step=cfg.index_step,
        lowdim_channels=cfg.lowdim_channels,
        split="validation",
        validation_fraction=cfg.validation_fraction,
        split_seed=cfg.split_seed,
    )

    # Gate the labels the optimizer can actually see at clip-current instants.  Raw-row coverage
    # is insufficient: a short phase entirely before the first T16 history would otherwise pass
    # the artifact gate yet contribute zero auxiliary examples.
    eligible_rows = np.concatenate((train.clip_index.entries, validation.clip_index.entries), axis=0)
    phase_counts_array = np.zeros(4, dtype=np.int64)
    contact_counts_array = np.zeros(2, dtype=np.int64)
    for store_idx, path in enumerate(stores):
        selected = eligible_rows[eligible_rows[:, 0] == store_idx]
        current = selected[:, 1] + (cfg.layout.num_frames - 1) * selected[:, 2]
        group = zarr.open_group(path, mode="r")
        phase = np.asarray(group["data"]["phase_id"][:], dtype=np.int64)
        contact = np.asarray(group["data"]["contact"][:], dtype=np.float32) >= 0.5
        phase_counts_array += np.bincount(phase[current], minlength=4)[:4]
        contact_counts_array += np.bincount(contact[current].astype(np.int64), minlength=2)[:2]
    if np.any(phase_counts_array == 0):
        raise ValueError(f"eligible V3 clips have empty phase classes: {phase_counts_array.tolist()}")
    if np.any(contact_counts_array == 0):
        raise ValueError(f"eligible V3 clips have only one contact class: {contact_counts_array.tolist()}")
    phase_counts = tuple(int(value) for value in phase_counts_array)
    contact_counts = tuple(int(value) for value in contact_counts_array)

    # Fit state moments from a deterministic, evenly spread subset of training histories.
    max_histories = 8192
    positions = np.linspace(0, len(train) - 1, min(max_histories, len(train)), dtype=np.int64)
    histories: list[torch.Tensor] = []
    commands: list[torch.Tensor] = []
    command_present: list[torch.Tensor] = []
    selected_entries = train.clip_index.entries[positions]
    for store_idx, path in enumerate(stores):
        rows = selected_entries[selected_entries[:, 0] == store_idx]
        if not len(rows):
            continue
        frames = rows[:, 1, None] + np.arange(cfg.layout.num_frames, dtype=np.int64)[None, :] * rows[:, 2, None]
        group = zarr.open_group(path, mode="r")
        data = group["data"]
        state = torch.from_numpy(np.asarray(data["state"][:], dtype=np.float32)[frames])
        histories.append(extract_qpos8(state))
        arrays = set(group["data"].array_keys())
        if "command8" in arrays:
            commands.append(torch.from_numpy(np.asarray(data["command8"][:], dtype=np.float32)[frames]))
            valid = (
                np.asarray(data["command_valid"][:], dtype=np.uint8)[frames].astype(bool)
                if "command_valid" in arrays
                else np.ones(frames.shape, dtype=bool)
            )
            command_present.append(torch.from_numpy(valid))
        else:
            commands.append(torch.zeros_like(histories[-1]))
            command_present.append(torch.zeros(frames.shape, dtype=torch.bool))
    qpos_history = torch.cat(histories)
    command_history = torch.cat(commands)
    command_mask = torch.cat(command_present)
    norm = fit_control_v2_normalization(
        qpos_history,
        previous_command=command_history,
        previous_command_present=command_mask,
        dt=float(cfg.observation_stride),
    )

    # Scales are p99.5 absolute residuals over every train clip.  They normalize the decoder while
    # retaining zero as the exact safe-hold output.
    residuals: list[np.ndarray] = []
    for store_idx, path in enumerate(stores):
        rows = train.clip_index.entries[train.clip_index.entries[:, 0] == store_idx]
        if not len(rows):
            continue
        current = rows[:, 1] + (cfg.layout.num_frames - 1) * rows[:, 2]
        group = zarr.open_group(path, mode="r")
        data = group["data"]
        state = torch.from_numpy(np.asarray(data["state"][:], dtype=np.float32))
        qpos = extract_qpos8(state).numpy()
        command = np.asarray(data["command8"][:], dtype=np.float32)
        residual = command_target_residuals(qpos, command, current)
        residuals.append(np.abs(residual).reshape(-1, 8))
    action_scale = np.maximum(np.quantile(np.concatenate(residuals), 0.995, axis=0), 1e-5)

    # Deployment gates desired command against CURRENT observed qpos, so fit that same quantity.
    # A command-to-command percentile can be below normal expert tracking lag and would clamp the
    # demonstrator itself.  The observed maximum plus a small margin keeps every training expert
    # transition executable while remaining finite and task-specific.
    deltas: list[np.ndarray] = []
    for store_idx, path in enumerate(stores):
        train_rows = train.clip_index.entries[train.clip_index.entries[:, 0] == store_idx]
        train_episode_indices = np.unique(train_rows[:, 3])
        if train_episode_indices.size == 0:
            continue
        group = zarr.open_group(path, mode="r")
        command = np.asarray(group["data"]["command8"][:], dtype=np.float32)
        state = torch.from_numpy(np.asarray(group["data"]["state"][:], dtype=np.float32))
        qpos = extract_qpos8(state).numpy()
        episode_ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
        deltas.append(
            conservative_command_delta_limit(
                qpos,
                command,
                episode_ends,
                episode_indices=train_episode_indices,
            )
        )
    max_delta = np.max(np.stack(deltas), axis=0)

    train_episodes = len({(int(row[0]), int(row[3])) for row in train.clip_index.entries})
    validation_episodes = len({(int(row[0]), int(row[3])) for row in validation.clip_index.entries})
    artifact = ControlV2Artifact(
        schema_version=CONTROL_V2_ARTIFACT_SCHEMA,
        data_schema=CONTROL_V2_DATA_SCHEMA,
        task=cfg.task,
        context_mode="dense_video_gel",
        state_mode="qpos8_history_command",
        target_mode="next_command_chunk_relative_mix8",
        horizon=ACTION_HORIZON,
        first_executable_index=1,
        chunk_first_n=20,
        temporal_ensemble_k=0.01,
        observation_stride=cfg.observation_stride,
        action_stride=cfg.action_stride,
        control_step_stride=1,
        phase_names=PHASE_NAMES,
        action_scale=tuple(float(value) for value in action_scale),
        joint_lower=PANDA_LOWER,
        joint_upper=PANDA_UPPER,
        max_command_delta=tuple(float(value) for value in max_delta),
        authoritative_control=authoritative,
        total_episodes=train_episodes + validation_episodes,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        phase_counts=phase_counts,
        contact_counts=contact_counts,
        source_stores=tuple(str(pathlib.Path(path).resolve()) for path in stores),
        source_store_sha256=store_contract_digest(stores),
    )
    if artifact.total_episodes != source_episode_count:
        raise ValueError(
            "one or more source episodes cannot form an H32/T16 training clip; "
            f"eligible={artifact.total_episodes}, source={source_episode_count}"
        )
    if cfg.require_authoritative_control:
        artifact.require_production_data()
    validation_selection = build_validation_selection(
        validation,
        cfg.validation_batches * cfg.local_batch_size,
    )
    oracle_audit = validate_validation_oracle_safety(
        validation,
        validation_selection.indices,
        max_delta,
        first_n=artifact.chunk_first_n,
    )
    return norm, artifact, validation_selection, oracle_audit


def main() -> None:
    cfg = control_v2_cli()
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / "run_config.json").exists():
        frozen = ControlV2TrainConfig.from_json((run / "run_config.json").read_text())
        if frozen != cfg:
            raise ValueError("existing run_config.json differs; refuse to replace fitted artifacts")
    else:
        _atomic_write(run / "run_config.json", cfg.to_json() + "\n")
    # A failed or interrupted recomputation must never leave a stale success marker that could
    # release the GPU dependency.  The launcher normally uses a fresh run root; this is defense
    # in depth for direct invocations and scheduler retries.
    (run / "STATS_DONE").unlink(missing_ok=True)
    norm, artifact, validation_selection, oracle_audit = fit_artifacts(cfg)
    freeze_validation_selection(run, validation_selection)
    save_normalization_artifact(norm, run / "normalization.json")
    _atomic_write(run / "artifact.json", artifact.to_json())
    _atomic_write(
        run / "STATS_DONE",
        json.dumps(
            {
                "schema": CONTROL_V2_ARTIFACT_SCHEMA,
                "split_identity": CONTROL_V2_SPLIT_IDENTITY,
                "train_episodes": artifact.train_episodes,
                "validation_episodes": artifact.validation_episodes,
                "source_store_sha256": artifact.source_store_sha256,
                "validation_sampling": validation_selection.to_manifest()["sampling"],
                "validation_sample_sha256": validation_selection.sample_sha256,
                "validation_examples": oracle_audit.examples,
                "validation_episode_count": validation_selection.episode_count,
                "validation_oracle_safety_violation_values": oracle_audit.violation_values,
                "validation_oracle_min_slack": oracle_audit.min_slack,
                "validation_oracle_max_rate_ratio": oracle_audit.max_rate_ratio,
            },
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps(json.loads(artifact.to_json()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
