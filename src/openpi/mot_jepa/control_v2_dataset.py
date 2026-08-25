"""Episode-safe UniVTAC dataset for the MoT-Control V3 contract."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib

import numpy as np
import torch
import zarr

from openpi.mot_jepa.clip_dataset import ClipIndex
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.control_v2_data import ACTION_HORIZON
from openpi.mot_jepa.control_v2_data import control_v2_chunk_indices
from openpi.mot_jepa.control_v2_data import extract_qpos8
from openpi.mot_jepa.control_v2_data import mixed_action_chunk_from_qpos
from openpi.mot_jepa.layout import TokenLayout

CONTROL_V2_DATA_SCHEMA = "mot_jepa_control_v2_data_v2"
CONTROL_V2_SPLIT_IDENTITY = "source_episode_seed_v1"


def _source_episode_seeds(store_paths: Sequence[str]) -> tuple[np.ndarray, ...]:
    """Load the immutable collection seed for every source episode.

    Prepared-store paths are job-owned and therefore cannot participate in a scientific split.
    The collection seed survives every deterministic preparation and is already closed by the
    authoritative collection and prepared-data provenance manifests.
    """

    seeds_by_store: list[np.ndarray] = []
    observed: set[int] = set()
    for path in store_paths:
        group = zarr.open_group(path, mode="r")
        meta = group["meta"]
        if "source_episode_seed" not in set(meta.array_keys()):
            raise ValueError(f"{path} lacks required meta/source_episode_seed")
        episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64).reshape(-1)
        raw = meta["source_episode_seed"]
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{path}: meta/source_episode_seed must be integer")
        seeds = np.asarray(raw[:], dtype=np.int64).reshape(-1)
        if seeds.shape != episode_ends.shape or np.any(seeds < 0):
            raise ValueError(f"{path}: meta/source_episode_seed must contain one non-negative value per episode")
        seed_set = {int(value) for value in seeds}
        duplicates = observed.intersection(seed_set)
        if duplicates or len(seed_set) != len(seeds):
            raise ValueError(f"source episode seeds must be globally unique; duplicates={sorted(duplicates)[:8]}")
        observed.update(seed_set)
        seeds_by_store.append(seeds)
    return tuple(seeds_by_store)


def build_command_action_target(
    current_qpos: torch.Tensor,
    future_command: torch.Tensor,
    *,
    command_valid: torch.Tensor | None = None,
    fallback_future_qpos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build row-0 state + rows-1:31 sent-command targets.

    A collected row stores the command that was applied immediately before that observation.
    Therefore the first action available *after* the current observation is the following row's
    command, not the following row's lagging measured qpos.  Legacy invalid command rows may fall
    back explicitly to measured future state, but production artifacts reject that data upstream.
    """

    if current_qpos.shape != (8,) or future_command.shape != (ACTION_HORIZON - 1, 8):
        raise ValueError("current_qpos/future_command must have shapes (8,) and (31, 8)")
    selected = future_command
    if command_valid is not None:
        if command_valid.shape != (ACTION_HORIZON - 1,) or command_valid.dtype != torch.bool:
            raise ValueError("command_valid must be a bool (31,) tensor")
        if fallback_future_qpos is None or fallback_future_qpos.shape != future_command.shape:
            raise ValueError("invalid command rows require fallback_future_qpos with shape (31, 8)")
        selected = torch.where(command_valid[:, None], future_command, fallback_future_qpos)
    absolute_command = torch.cat((current_qpos[None], selected), dim=0)
    return mixed_action_chunk_from_qpos(absolute_command), absolute_command


def _episode_bounds(episode_ends: np.ndarray, episode_idx: int) -> tuple[int, int]:
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if not 0 <= episode_idx < ends.size:
        raise IndexError(f"episode {episode_idx} outside {ends.size} episodes")
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    end = int(ends[episode_idx])
    if not 0 <= start < end:
        raise ValueError(f"invalid episode bounds [{start}, {end})")
    return start, end


def infer_legacy_phase_ids(gripper: np.ndarray, episode_ends: np.ndarray, *, threshold: float = 1e-5) -> np.ndarray:
    """Infer settle/close/lift/release labels when the legacy store has no phase array."""

    values = np.asarray(gripper, dtype=np.float32).reshape(-1)
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if ends.size == 0 or int(ends[-1]) != values.size:
        raise ValueError("episode_ends must terminate at the gripper array length")
    phases = np.full(values.size, 2, dtype=np.int64)
    start = 0
    for raw_end in ends:
        end = int(raw_end)
        episode = values[start:end]
        delta = np.diff(episode, prepend=episode[0])
        closing = np.flatnonzero(delta < -threshold)
        opening = np.flatnonzero((delta > threshold) & (np.arange(episode.size) > episode.size // 2))
        close_start = int(closing[0]) if closing.size else min(1, episode.size)
        release_start = int(opening[0]) if opening.size else episode.size
        phases[start : start + close_start] = 0
        if closing.size:
            close_stop = min(int(closing[-1]) + 2, release_start)
            if release_start > close_start + 1:
                close_stop = min(close_stop, release_start - 1)
            phases[start + close_start : start + close_stop] = 1
        phases[start + release_start : end] = 3
        start = end
    return phases


def split_control_v2_index(
    index: ClipIndex,
    *,
    split: str,
    source_episode_seeds: Sequence[np.ndarray],
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> ClipIndex:
    """Split by immutable source-episode seed, never prepared path or timestep."""

    if split not in {"train", "validation", "all"}:
        raise ValueError("split must be train, validation, or all")
    if split == "all":
        return index
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(source_episode_seeds) != len(index.store_paths):
        raise ValueError("source_episode_seeds must align with store_paths")
    keep = []
    for row in index.entries:
        store_idx, _start, _stride, episode_idx = (int(value) for value in row)
        try:
            episode_seed = int(source_episode_seeds[store_idx][episode_idx])
        except (IndexError, TypeError) as error:
            raise ValueError(f"clip identifies missing source episode ({store_idx}, {episode_idx})") from error
        identity = f"{seed}\0episode_seed={episode_seed}".encode()
        unit = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
        is_validation = unit < validation_fraction
        keep.append(is_validation if split == "validation" else not is_validation)
    entries = index.entries[np.asarray(keep, dtype=bool)]
    if entries.size == 0:
        raise ValueError(f"{split} split is empty")
    return ClipIndex(entries.copy(), list(index.store_paths))


@dataclasses.dataclass(frozen=True)
class ControlV2Sample:
    video: torch.Tensor
    gel: torch.Tensor
    lowdim: torch.Tensor
    gel_valid: torch.Tensor
    lowdim_valid: torch.Tensor
    state_history: torch.Tensor
    command_history: torch.Tensor
    command_present: torch.Tensor
    history_valid: torch.Tensor
    action_chunk: torch.Tensor
    absolute_qpos_chunk: torch.Tensor
    rate_reference_qpos: torch.Tensor
    phase_id: torch.Tensor
    contact: torch.Tensor
    contact_valid: torch.Tensor
    sample_weight: torch.Tensor
    episode_idx: torch.Tensor
    store_idx: torch.Tensor
    current_index: torch.Tensor


class ControlV2Dataset(torch.utils.data.Dataset):
    """A strict single-task wrapper over :class:`MotJepaClipDataset`.

    The wrapped index reserves only 31 future transitions because the H32 target's first row is
    the observation-time placeholder.  This distinction is intentionally encoded here rather
    than left to each trainer/evaluator.
    """

    def __init__(
        self,
        store_paths: list[str],
        layout: TokenLayout,
        *,
        observation_stride: int = 2,
        action_stride: int = 1,
        control_step_stride: int = 1,
        index_step: int = 1,
        lowdim_channels: int | None = None,
        split: str = "train",
        validation_fraction: float = 0.15,
        split_seed: int = 42,
    ) -> None:
        if observation_stride < 1 or action_stride != 1 or control_step_stride != 1:
            raise ValueError("V3 requires positive observation stride and action/control_step stride=1")
        self.action_stride = action_stride
        self.control_step_stride = control_step_stride
        self.base = MotJepaClipDataset(
            store_paths,
            layout,
            domain_ids=[0] * len(store_paths),
            strides=(observation_stride,),
            index_step=index_step,
            lowdim_channels=lowdim_channels,
            with_conditioning=True,
            action_horizon=ACTION_HORIZON - 1,
            action_stride=action_stride,
        )
        self.source_episode_seeds = _source_episode_seeds(self.base.store_paths)
        self.base.clip_index = split_control_v2_index(
            self.base.clip_index,
            split=split,
            source_episode_seeds=self.source_episode_seeds,
            validation_fraction=validation_fraction,
            seed=split_seed,
        )
        self._phase_cache: dict[int, np.ndarray] = {}

    @property
    def clip_index(self) -> ClipIndex:
        return self.base.clip_index

    @property
    def store_paths(self) -> list[str]:
        return self.base.store_paths

    def __len__(self) -> int:
        return len(self.base)

    def _phase_ids(self, store_idx: int) -> np.ndarray:
        if store_idx in self._phase_cache:
            return self._phase_cache[store_idx]
        group, _ = self.base._store(store_idx)  # noqa: SLF001 - package-local dataset extension
        data = group["data"]
        if "phase_id" in set(data.array_keys()):
            phase = np.asarray(data["phase_id"][:], dtype=np.int64)
        else:
            state = torch.from_numpy(np.asarray(data["state"][:], dtype=np.float32))
            phase = infer_legacy_phase_ids(
                extract_qpos8(state)[:, 7].numpy(),
                np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64),
            )
        if np.any((phase < 0) | (phase > 3)):
            raise ValueError(f"store {self.store_paths[store_idx]} has invalid phase ids")
        self._phase_cache[store_idx] = phase
        return phase

    def __getitem__(self, index: int) -> ControlV2Sample:
        entry = self.base.clip_index[index]
        base_sample = self.base[index]
        group, _ = self.base._store(entry.store_idx)  # noqa: SLF001
        data = group["data"]
        arrays = set(data.array_keys())
        if "state" not in arrays:
            raise ValueError(f"{self.store_paths[entry.store_idx]} lacks required data/state")

        frames = entry.start + np.arange(self.base.layout.num_frames, dtype=np.int64) * entry.stride
        current = int(frames[-1])
        episode_ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
        episode_start, episode_end = _episode_bounds(episode_ends, entry.episode_idx)
        future = control_v2_chunk_indices(
            current,
            episode_start=episode_start,
            episode_end=episode_end,
            action_stride=self.action_stride,
        ).numpy()
        future_state = torch.from_numpy(np.asarray(data["state"][future], dtype=np.float32))
        future_qpos = extract_qpos8(future_state)
        if "control_step" in arrays:
            cadence_indices = np.concatenate((frames, future))
            cadence_valid = (
                np.asarray(data["control_step_valid"][cadence_indices], dtype=np.uint8).astype(bool)
                if "control_step_valid" in arrays
                else np.ones(cadence_indices.shape, dtype=bool)
            )
            if bool(cadence_valid.all()):
                frame_steps = np.asarray(data["control_step"][frames], dtype=np.int64)
                future_steps = np.asarray(data["control_step"][future], dtype=np.int64)
                expected_frame = entry.stride * self.control_step_stride
                expected_future = self.action_stride * self.control_step_stride
                if np.any(np.diff(frame_steps) != expected_frame) or np.any(np.diff(future_steps) != expected_future):
                    raise ValueError(
                        f"store {self.store_paths[entry.store_idx]} control cadence differs from "
                        f"observation/action={expected_frame}/{expected_future} simulator steps"
                    )

        current_qpos = future_qpos[0]
        if "command8" in arrays:
            future_command = torch.from_numpy(np.asarray(data["command8"][future[1:]], dtype=np.float32))
            future_command_valid = (
                torch.from_numpy(np.asarray(data["command_valid"][future[1:]], dtype=np.uint8).astype(bool))
                if "command_valid" in arrays
                else torch.ones(ACTION_HORIZON - 1, dtype=torch.bool)
            )
            action_chunk, absolute_qpos = build_command_action_target(
                current_qpos,
                future_command,
                command_valid=future_command_valid,
                fallback_future_qpos=future_qpos[1:],
            )
        else:
            action_chunk, absolute_qpos = build_command_action_target(current_qpos, future_qpos[1:])

        state_history = base_sample.state
        qpos_history = extract_qpos8(state_history)
        if "command8" in arrays:
            command = torch.from_numpy(np.asarray(data["command8"][frames], dtype=np.float32))
            if command.shape != qpos_history.shape:
                raise ValueError(f"data/command8 has incompatible shape {tuple(command.shape)}")
            if "command_valid" in arrays:
                command_present = torch.from_numpy(
                    np.asarray(data["command_valid"][frames], dtype=np.uint8).astype(bool)
                )
            else:
                command_present = torch.ones(len(frames), dtype=torch.bool)
        else:
            command = torch.cat((qpos_history[:1], qpos_history[:-1]), dim=0)
            command_present = torch.zeros(len(frames), dtype=torch.bool)

        phase = self._phase_ids(entry.store_idx)
        phase_id = torch.tensor(int(phase[current]), dtype=torch.long)
        future_phase = phase[future]
        transition = bool(np.any(future_phase[1:] != future_phase[:-1]))
        if "contact" in arrays:
            contact = torch.tensor(float(np.asarray(data["contact"][current]).item()), dtype=torch.float32)
            valid = bool(np.asarray(data["contact_valid"][current]).item()) if "contact_valid" in arrays else True
            contact_valid = torch.tensor(valid, dtype=torch.bool)
        else:
            contact = torch.tensor(0.0, dtype=torch.float32)
            contact_valid = torch.zeros((), dtype=torch.bool)
        recovery = bool(np.asarray(data["recovery"][current]).item()) if "recovery" in arrays else False
        sample_weight = torch.tensor(3.0 if transition or recovery else 1.0, dtype=torch.float32)

        return ControlV2Sample(
            video=base_sample.video,
            gel=base_sample.gel,
            lowdim=base_sample.lowdim,
            gel_valid=base_sample.gel_valid,
            lowdim_valid=base_sample.lowdim_valid,
            state_history=state_history,
            command_history=command,
            command_present=command_present,
            history_valid=torch.ones(len(frames), dtype=torch.bool),
            action_chunk=action_chunk,
            absolute_qpos_chunk=absolute_qpos,
            # The runtime limiter compares command[t + h] with the measured qpos at the
            # immediately preceding control observation, not with command[t + h - 1].
            rate_reference_qpos=future_qpos[:-1],
            phase_id=phase_id,
            contact=contact,
            contact_valid=contact_valid,
            sample_weight=sample_weight,
            episode_idx=torch.tensor(entry.episode_idx, dtype=torch.long),
            store_idx=torch.tensor(entry.store_idx, dtype=torch.long),
            current_index=torch.tensor(current, dtype=torch.long),
        )


def collate_control_v2(samples: list[ControlV2Sample]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {
        field.name: torch.stack([getattr(sample, field.name) for sample in samples])
        for field in dataclasses.fields(ControlV2Sample)
    }
