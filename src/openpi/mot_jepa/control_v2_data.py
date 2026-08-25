"""Data contract for the state-conditioned MoT control-v2 policy.

Control-v2 intentionally uses a much smaller action/state view than the generic FTP-1
120-dimensional representation.  UniVTAC ``lift_bottle`` exposes one seven-joint arm in
slots ``[9, 16)`` and its absolute gripper width in slot ``44``.  Keeping all conversions
here prevents the training target and the deployment command from quietly drifting apart.

The action chunk has 32 rows.  Row zero represents the observation instant (zero arm
offset and the current absolute gripper width); rows 1--31 represent future instants.  Arm
positions are offsets from the current arm state while the gripper remains absolute.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import json
import os
import pathlib
import tempfile
from typing import Any

import torch
from torch import nn

STATE_DIM = 120
QPOS_DIM = 8
ARM_DIM = 7
ACTION_HORIZON = 32
ARM_STATE_SLICE = slice(9, 16)
GRIPPER_STATE_SLOT = 44
NORMALIZATION_SCHEMA = "mot_jepa_control_v2_norm_v1"


class Phase(enum.IntEnum):
    """Supervised lift-bottle stages in execution order."""

    APPROACH = 0
    CLOSE = 1
    LIFT = 2
    RELEASE = 3


PHASE_NAMES = tuple(phase.name.lower() for phase in Phase)
NUM_PHASES = len(PHASE_NAMES)


def _require_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")


def _require_floating(value: torch.Tensor, name: str) -> None:
    _require_tensor(value, name)
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype, got {value.dtype}")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def extract_qpos8(state: torch.Tensor) -> torch.Tensor:
    """Extract ``[..., 8]`` UniVTAC qpos from an exact ``[..., 120]`` FTP-1 state.

    A new contiguous tensor is returned rather than a view assembled through advanced
    indexing, which makes it safe for callers to retain while recycling a batch buffer.
    """

    _require_floating(state, "state")
    if state.ndim < 1 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have shape [..., {STATE_DIM}], got {tuple(state.shape)}")
    _require_finite(state, "state")
    return torch.cat((state[..., ARM_STATE_SLICE], state[..., GRIPPER_STATE_SLOT : GRIPPER_STATE_SLOT + 1]), dim=-1)


def scatter_qpos8(qpos8: torch.Tensor, *, template: torch.Tensor | None = None) -> torch.Tensor:
    """Scatter ``[..., 8]`` qpos into the active slots of a ``[..., 120]`` state.

    When ``template`` is supplied, inactive slots are preserved in a clone.  Otherwise they
    are zero-filled.  The strict shape/device/dtype checks avoid an accidental broadcast
    across a batch or conversion between normalized and physical units.
    """

    _require_floating(qpos8, "qpos8")
    if qpos8.ndim < 1 or qpos8.shape[-1] != QPOS_DIM:
        raise ValueError(f"qpos8 must have shape [..., {QPOS_DIM}], got {tuple(qpos8.shape)}")
    _require_finite(qpos8, "qpos8")
    expected = (*qpos8.shape[:-1], STATE_DIM)
    if template is None:
        state = qpos8.new_zeros(expected)
    else:
        _require_floating(template, "template")
        if tuple(template.shape) != expected:
            raise ValueError(f"template must have shape {expected}, got {tuple(template.shape)}")
        if template.dtype != qpos8.dtype or template.device != qpos8.device:
            raise ValueError("template and qpos8 must have the same dtype and device")
        _require_finite(template, "template")
        state = template.clone()
    state[..., ARM_STATE_SLICE] = qpos8[..., :ARM_DIM]
    state[..., GRIPPER_STATE_SLOT] = qpos8[..., ARM_DIM]
    return state


def mixed_action_chunk_from_qpos(qpos_trajectory: torch.Tensor) -> torch.Tensor:
    """Build the exact H32 mixed action target from ``[..., 32, 8]`` absolute qpos.

    All arm rows are relative to row zero, not step-to-step deltas.  The gripper column is
    absolute on every row.  Consequently row zero is ``[0, ..., 0, current_gripper]``.
    """

    _require_floating(qpos_trajectory, "qpos_trajectory")
    if qpos_trajectory.ndim < 2 or qpos_trajectory.shape[-2:] != (ACTION_HORIZON, QPOS_DIM):
        raise ValueError(
            f"qpos_trajectory must have shape [..., {ACTION_HORIZON}, {QPOS_DIM}], got {tuple(qpos_trajectory.shape)}"
        )
    _require_finite(qpos_trajectory, "qpos_trajectory")
    arm_offsets = qpos_trajectory[..., :, :ARM_DIM] - qpos_trajectory[..., :1, :ARM_DIM]
    return torch.cat((arm_offsets, qpos_trajectory[..., :, ARM_DIM:]), dim=-1)


def control_v2_chunk_indices(
    current_index: int,
    *,
    episode_start: int,
    episode_end: int,
    action_stride: int = 1,
) -> torch.Tensor:
    """Return the current + 31 future indices after proving they stay in one episode.

    ``episode_end`` is exclusive, matching the Zarr ``episode_ends`` convention.  This helper
    centralizes the otherwise easy-to-miss distinction between a 32-row target and 31 future
    transitions.
    """

    for value, name in (
        (current_index, "current_index"),
        (episode_start, "episode_start"),
        (episode_end, "episode_end"),
        (action_stride, "action_stride"),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int")
    if episode_start < 0 or episode_end <= episode_start:
        raise ValueError(f"invalid half-open episode bounds [{episode_start}, {episode_end})")
    if action_stride < 1:
        raise ValueError(f"action_stride must be positive, got {action_stride}")
    last_index = current_index + (ACTION_HORIZON - 1) * action_stride
    if current_index < episode_start or last_index >= episode_end:
        raise ValueError(
            f"H{ACTION_HORIZON} chunk [{current_index}, {last_index}] crosses episode [{episode_start}, {episode_end})"
        )
    return current_index + torch.arange(ACTION_HORIZON, dtype=torch.long) * action_stride


def build_mixed_action_chunk(state_trajectory: torch.Tensor) -> torch.Tensor:
    """Build an H32 control-v2 target from ``[..., 32, 120]`` absolute state."""

    _require_tensor(state_trajectory, "state_trajectory")
    if state_trajectory.ndim < 2 or state_trajectory.shape[-2] != ACTION_HORIZON:
        raise ValueError(f"state_trajectory must have horizon {ACTION_HORIZON}, got {tuple(state_trajectory.shape)}")
    return mixed_action_chunk_from_qpos(extract_qpos8(state_trajectory))


def mixed_chunk_to_absolute_qpos8(action_chunk: torch.Tensor, current_qpos8: torch.Tensor) -> torch.Tensor:
    """Invert the mixed representation for execution or a contract round-trip test."""

    _require_floating(action_chunk, "action_chunk")
    _require_floating(current_qpos8, "current_qpos8")
    if action_chunk.ndim < 2 or action_chunk.shape[-2:] != (ACTION_HORIZON, QPOS_DIM):
        raise ValueError(
            f"action_chunk must have shape [..., {ACTION_HORIZON}, {QPOS_DIM}], got {tuple(action_chunk.shape)}"
        )
    if tuple(current_qpos8.shape) != (*action_chunk.shape[:-2], QPOS_DIM):
        raise ValueError(
            f"current_qpos8 must have shape {(*action_chunk.shape[:-2], QPOS_DIM)}, got {tuple(current_qpos8.shape)}"
        )
    if action_chunk.dtype != current_qpos8.dtype or action_chunk.device != current_qpos8.device:
        raise ValueError("action_chunk and current_qpos8 must have the same dtype and device")
    _require_finite(action_chunk, "action_chunk")
    _require_finite(current_qpos8, "current_qpos8")
    arm = action_chunk[..., :, :ARM_DIM] + current_qpos8[..., None, :ARM_DIM]
    return torch.cat((arm, action_chunk[..., :, ARM_DIM:]), dim=-1)


@dataclasses.dataclass(frozen=True)
class PrefixPaddedHistory:
    """A left-padded history and the positions backed by real observations."""

    values: torch.Tensor
    valid: torch.Tensor
    available_lengths: torch.Tensor


def prefix_pad_history(history: torch.Tensor, available_lengths: int | torch.Tensor) -> PrefixPaddedHistory:
    """Repeat the oldest available item on the left, exactly as deployment cold start does.

    ``history`` is ``[..., T, D]`` with a batch/feature-independent time axis at
    ``-2``.  For length ``L``, the most recent ``L`` values are retained and positions before
    them repeat value ``T-L``.  ``valid`` distinguishes repeated padding from observations.
    """

    _require_tensor(history, "history")
    if history.ndim < 2:
        raise ValueError(f"history must have shape [..., T, D], got {tuple(history.shape)}")
    time = int(history.shape[-2])
    if time < 1:
        raise ValueError("history time dimension must be non-empty")
    batch_shape = history.shape[:-2]
    if isinstance(available_lengths, int):
        lengths = torch.full(batch_shape or (), available_lengths, dtype=torch.long, device=history.device)
    else:
        _require_tensor(available_lengths, "available_lengths")
        if available_lengths.is_floating_point() or available_lengths.dtype == torch.bool:
            raise TypeError("available_lengths must have an integer dtype")
        if tuple(available_lengths.shape) != tuple(batch_shape):
            raise ValueError(
                f"available_lengths must have batch shape {tuple(batch_shape)}, got {tuple(available_lengths.shape)}"
            )
        lengths = available_lengths.to(device=history.device, dtype=torch.long)
    if bool(((lengths < 1) | (lengths > time)).any()):
        raise ValueError(f"available_lengths must be in [1, {time}]")

    positions = torch.arange(time, device=history.device)
    starts = time - lengths
    indices = torch.maximum(positions, starts[..., None])
    valid = positions >= starts[..., None]
    gather_index = indices.unsqueeze(-1).expand(*history.shape)
    values = torch.gather(history, dim=-2, index=gather_index)
    return PrefixPaddedHistory(values=values, valid=valid, available_lengths=lengths)


def random_prefix_pad_history(
    history: torch.Tensor,
    *,
    generator: torch.Generator,
    min_available: int = 1,
    max_available: int | None = None,
) -> PrefixPaddedHistory:
    """Sample cold-start lengths with only the caller-owned generator as randomness."""

    _require_tensor(history, "history")
    if history.ndim < 2:
        raise ValueError(f"history must have shape [..., T, D], got {tuple(history.shape)}")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    time = int(history.shape[-2])
    upper = time if max_available is None else int(max_available)
    if not 1 <= min_available <= upper <= time:
        raise ValueError(f"require 1 <= min_available <= max_available <= {time}")
    batch_shape = tuple(history.shape[:-2])
    sample_shape = batch_shape or (1,)
    sample_device = generator.device
    lengths = torch.randint(
        min_available,
        upper + 1,
        sample_shape,
        generator=generator,
        device=sample_device,
        dtype=torch.long,
    ).to(history.device)
    if not batch_shape:
        lengths = lengths.squeeze(0)
    return prefix_pad_history(history, lengths)


@dataclasses.dataclass(frozen=True)
class _MomentStats:
    mean: torch.Tensor
    std: torch.Tensor
    count: int

    def __post_init__(self) -> None:
        _require_floating(self.mean, "mean")
        _require_floating(self.std, "std")
        if self.mean.shape != (QPOS_DIM,) or self.std.shape != (QPOS_DIM,):
            raise ValueError(f"normalization moments must each have shape ({QPOS_DIM},)")
        if self.mean.device.type != "cpu" or self.std.device.type != "cpu":
            raise ValueError("normalization artifacts must be stored on CPU")
        _require_finite(self.mean, "mean")
        _require_finite(self.std, "std")
        if bool((self.std <= 0).any()):
            raise ValueError("normalization std must be strictly positive")
        if self.count < 0:
            raise ValueError("normalization count must be non-negative")

    def to_artifact(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "count": self.count}

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> _MomentStats:
        if set(payload) != {"mean", "std", "count"}:
            raise ValueError("moment artifact must contain exactly mean, std, and count")
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
            count=int(payload["count"]),
        )


@dataclasses.dataclass(frozen=True)
class ControlV2NormStats:
    """Training-split moments for physical qpos, velocity, and previous commands."""

    qpos: _MomentStats
    velocity: _MomentStats
    previous_command: _MomentStats

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-serializable, schema-stamped checkpoint payload."""

        return {
            "schema": NORMALIZATION_SCHEMA,
            "qpos": self.qpos.to_artifact(),
            "velocity": self.velocity.to_artifact(),
            "previous_command": self.previous_command.to_artifact(),
        }

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> ControlV2NormStats:
        expected = {"schema", "qpos", "velocity", "previous_command"}
        if set(payload) != expected:
            raise ValueError(f"normalization artifact must contain exactly {sorted(expected)}")
        if payload["schema"] != NORMALIZATION_SCHEMA:
            raise ValueError(f"unsupported normalization schema {payload['schema']!r}")
        return cls(
            qpos=_MomentStats.from_artifact(payload["qpos"]),
            velocity=_MomentStats.from_artifact(payload["velocity"]),
            previous_command=_MomentStats.from_artifact(payload["previous_command"]),
        )


def _validate_history_valid(history_valid: torch.Tensor | None, qpos: torch.Tensor) -> torch.Tensor:
    expected = qpos.shape[:-1]
    if history_valid is None:
        return torch.ones(expected, dtype=torch.bool, device=qpos.device)
    _require_tensor(history_valid, "history_valid")
    if history_valid.dtype != torch.bool:
        raise TypeError("history_valid must have dtype bool")
    if tuple(history_valid.shape) != tuple(expected):
        raise ValueError(f"history_valid must have shape {tuple(expected)}, got {tuple(history_valid.shape)}")
    return history_valid.to(qpos.device)


def _moments(values: torch.Tensor, present: torch.Tensor, *, eps: float) -> _MomentStats:
    selected = values[present]
    if selected.numel() == 0:
        return _MomentStats(torch.zeros(QPOS_DIM), torch.ones(QPOS_DIM), 0)
    selected = selected.detach().float().cpu()
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(eps)
    return _MomentStats(mean, std, int(selected.shape[0]))


def fit_control_v2_normalization(
    qpos_history: torch.Tensor,
    *,
    history_valid: torch.Tensor | None = None,
    previous_command: torch.Tensor | None = None,
    previous_command_present: torch.Tensor | None = None,
    dt: float = 1.0,
    eps: float = 1e-6,
) -> ControlV2NormStats:
    """Fit normalization using only explicitly present training-split observations."""

    _require_floating(qpos_history, "qpos_history")
    if qpos_history.ndim < 2 or qpos_history.shape[-1] != QPOS_DIM:
        raise ValueError(f"qpos_history must have shape [..., T, {QPOS_DIM}], got {tuple(qpos_history.shape)}")
    if qpos_history.shape[-2] < 1:
        raise ValueError("qpos_history time dimension must be non-empty")
    _require_finite(qpos_history, "qpos_history")
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    valid = _validate_history_valid(history_valid, qpos_history)

    velocity = torch.zeros_like(qpos_history)
    velocity[..., 1:, :] = (qpos_history[..., 1:, :] - qpos_history[..., :-1, :]) / dt
    velocity_present = torch.zeros_like(valid)
    velocity_present[..., 1:] = valid[..., 1:] & valid[..., :-1]

    if previous_command is None:
        command = torch.zeros_like(qpos_history)
        command_present = torch.zeros_like(valid)
        if previous_command_present is not None:
            raise ValueError("previous_command_present was supplied without previous_command")
    else:
        _require_floating(previous_command, "previous_command")
        if tuple(previous_command.shape) != tuple(qpos_history.shape):
            raise ValueError(
                f"previous_command must have shape {tuple(qpos_history.shape)}, got {tuple(previous_command.shape)}"
            )
        if previous_command.dtype != qpos_history.dtype or previous_command.device != qpos_history.device:
            raise ValueError("previous_command and qpos_history must have the same dtype and device")
        _require_finite(previous_command, "previous_command")
        command = previous_command
        if previous_command_present is None:
            command_present = valid
        else:
            _require_tensor(previous_command_present, "previous_command_present")
            if previous_command_present.dtype != torch.bool:
                raise TypeError("previous_command_present must have dtype bool")
            if tuple(previous_command_present.shape) != tuple(valid.shape):
                raise ValueError(
                    f"previous_command_present must have shape {tuple(valid.shape)}, "
                    f"got {tuple(previous_command_present.shape)}"
                )
            command_present = previous_command_present.to(valid.device) & valid

    return ControlV2NormStats(
        qpos=_moments(qpos_history, valid, eps=eps),
        velocity=_moments(velocity, velocity_present, eps=eps),
        previous_command=_moments(command, command_present, eps=eps),
    )


class ControlV2Normalizer(nn.Module):
    """Checkpointable normalization with explicit missing-value masking."""

    def __init__(self, stats: ControlV2NormStats):
        super().__init__()
        for name, moments in (
            ("qpos", stats.qpos),
            ("velocity", stats.velocity),
            ("previous_command", stats.previous_command),
        ):
            self.register_buffer(f"{name}_mean", moments.mean.clone())
            self.register_buffer(f"{name}_std", moments.std.clone())
        self.qpos_count = stats.qpos.count
        self.velocity_count = stats.velocity.count
        self.previous_command_count = stats.previous_command.count

    def _normalize(self, value: torch.Tensor, kind: str, present: torch.Tensor | None = None) -> torch.Tensor:
        mean = getattr(self, f"{kind}_mean")
        std = getattr(self, f"{kind}_std")
        normalized = (value - mean) / std
        if present is not None:
            normalized = torch.where(present[..., None], normalized, torch.zeros_like(normalized))
        return normalized

    def normalize_qpos(self, value: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        return self._normalize(value, "qpos", present)

    def normalize_velocity(self, value: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        return self._normalize(value, "velocity", present)

    def normalize_previous_command(self, value: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        return self._normalize(value, "previous_command", present)

    def to_stats(self) -> ControlV2NormStats:
        return ControlV2NormStats(
            qpos=_MomentStats(self.qpos_mean.detach().cpu(), self.qpos_std.detach().cpu(), self.qpos_count),
            velocity=_MomentStats(
                self.velocity_mean.detach().cpu(), self.velocity_std.detach().cpu(), self.velocity_count
            ),
            previous_command=_MomentStats(
                self.previous_command_mean.detach().cpu(),
                self.previous_command_std.detach().cpu(),
                self.previous_command_count,
            ),
        )


@dataclasses.dataclass(frozen=True)
class StateFeatures:
    """State-conditioning components and their concatenated 27-D representation."""

    qpos: torch.Tensor
    velocity: torch.Tensor
    previous_command: torch.Tensor
    history_valid: torch.Tensor
    velocity_present: torch.Tensor
    previous_command_present: torch.Tensor
    tensor: torch.Tensor


def build_state_features(
    state_history: torch.Tensor,
    *,
    history_valid: torch.Tensor | None = None,
    previous_command: torch.Tensor | None = None,
    previous_command_present: torch.Tensor | None = None,
    normalizer: ControlV2Normalizer | None = None,
    dt: float = 1.0,
) -> StateFeatures:
    """Construct qpos, velocity, previous-command, and three presence channels.

    Missing velocities and commands normalize to exact zero and carry a false presence bit;
    an actual physical zero therefore remains distinguishable from an unavailable value.
    """

    qpos = extract_qpos8(state_history)
    if qpos.ndim < 2 or qpos.shape[-2] < 1:
        raise ValueError(f"state_history must have shape [..., T, {STATE_DIM}] with T >= 1")
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    valid = _validate_history_valid(history_valid, qpos)
    velocity = torch.zeros_like(qpos)
    velocity[..., 1:, :] = (qpos[..., 1:, :] - qpos[..., :-1, :]) / dt
    velocity_present = torch.zeros_like(valid)
    velocity_present[..., 1:] = valid[..., 1:] & valid[..., :-1]

    if previous_command is None:
        command = torch.zeros_like(qpos)
        command_present = torch.zeros_like(valid)
        if previous_command_present is not None:
            raise ValueError("previous_command_present was supplied without previous_command")
    else:
        _require_floating(previous_command, "previous_command")
        if tuple(previous_command.shape) != tuple(qpos.shape):
            raise ValueError(
                f"previous_command must have shape {tuple(qpos.shape)}, got {tuple(previous_command.shape)}"
            )
        if previous_command.dtype != qpos.dtype or previous_command.device != qpos.device:
            raise ValueError("previous_command and state_history must have the same dtype and device")
        _require_finite(previous_command, "previous_command")
        command = previous_command
        if previous_command_present is None:
            command_present = valid
        else:
            _require_tensor(previous_command_present, "previous_command_present")
            if previous_command_present.dtype != torch.bool:
                raise TypeError("previous_command_present must have dtype bool")
            if tuple(previous_command_present.shape) != tuple(valid.shape):
                raise ValueError(
                    f"previous_command_present must have shape {tuple(valid.shape)}, "
                    f"got {tuple(previous_command_present.shape)}"
                )
            command_present = previous_command_present.to(valid.device) & valid

    if normalizer is None:
        qpos_features = qpos
        velocity_features = torch.where(velocity_present[..., None], velocity, torch.zeros_like(velocity))
        command_features = torch.where(command_present[..., None], command, torch.zeros_like(command))
    else:
        qpos_features = normalizer.normalize_qpos(qpos, valid)
        velocity_features = normalizer.normalize_velocity(velocity, velocity_present)
        command_features = normalizer.normalize_previous_command(command, command_present)
    tensor = torch.cat(
        (
            qpos_features,
            velocity_features,
            command_features,
            valid[..., None].to(qpos.dtype),
            velocity_present[..., None].to(qpos.dtype),
            command_present[..., None].to(qpos.dtype),
        ),
        dim=-1,
    )
    return StateFeatures(
        qpos=qpos,
        velocity=velocity,
        previous_command=command,
        history_valid=valid,
        velocity_present=velocity_present,
        previous_command_present=command_present,
        tensor=tensor,
    )


def save_normalization_artifact(stats: ControlV2NormStats, path: str | pathlib.Path) -> None:
    """Write a normalization artifact as deterministic JSON."""

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_staging = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    staging = pathlib.Path(raw_staging)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(stats.to_artifact(), indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)


def load_normalization_artifact(path: str | pathlib.Path) -> ControlV2NormStats:
    """Load and fully validate a normalization artifact."""

    payload = json.loads(pathlib.Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("normalization artifact root must be an object")
    return ControlV2NormStats.from_artifact(payload)


def validate_phase_ids(phase_ids: torch.Tensor, *, allow_unlabeled: bool = False) -> torch.Tensor:
    """Validate phase labels and return a ``long`` tensor on the same device."""

    _require_tensor(phase_ids, "phase_ids")
    if phase_ids.is_floating_point() or phase_ids.dtype == torch.bool or phase_ids.is_complex():
        raise TypeError("phase_ids must have an integer dtype")
    lower = -1 if allow_unlabeled else 0
    ids = phase_ids.to(dtype=torch.long)
    if bool(((ids < lower) | (ids >= NUM_PHASES)).any()):
        allowed = f"-1 or [0, {NUM_PHASES})" if allow_unlabeled else f"[0, {NUM_PHASES})"
        raise ValueError(f"phase_ids must be {allowed}; phases are {PHASE_NAMES}")
    return ids
