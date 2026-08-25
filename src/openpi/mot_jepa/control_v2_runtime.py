"""Runtime contract and qualification gates for the MoT-Control V2 policy.

V2 predicts one 32-row chunk in an active eight-dimensional layout.  Rows contain seven arm
offsets from the qpos at inference time and one absolute gripper target.  Row zero is a
non-executable current-time placeholder; rows 1--31 are future targets.  This module keeps that
contract separate from the legacy 120-slot, first-difference path in :mod:`openpi.mot_jepa.deploy`.

The temporal ensemble intentionally matches the official FTP1 evaluator: infer on every control
step, use at most the first 20 executable predictions from each chunk, order candidates from oldest
to newest, and weight them with ``exp(-0.01 * candidate_index)``.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import numpy as np

QPOS8_DIM = 8
ARM_DIM = 7
J5_INDEX = 5
GRIPPER_INDEX = 7
V2_HORIZON = 32
FIRST_EXECUTABLE_INDEX = 1
DEFAULT_ENSEMBLE_FIRST_N = 20
DEFAULT_ENSEMBLE_K = 0.01


def _vector8(value: np.ndarray | Sequence[float], *, name: str, finite: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (QPOS8_DIM,):
        raise ValueError(f"{name} must have shape ({QPOS8_DIM},), got {result.shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def safe_hold_chunk(qpos8: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a valid V2 chunk whose every executable row resolves to ``qpos8``.

    Arm entries are zero offsets and the gripper is copied as an absolute target.  The same values
    are also placed in row zero even though that placeholder is never executable.  This makes a
    zero-initialized/fallback chunk safe for both inspection and conversion.
    """
    qpos8 = _vector8(qpos8, name="qpos8")
    chunk = np.zeros((V2_HORIZON, QPOS8_DIM), dtype=np.float32)
    chunk[:, GRIPPER_INDEX] = qpos8[GRIPPER_INDEX]
    return chunk


def active_mixed_chunk_to_absolute(
    chunk: np.ndarray,
    qpos8_base: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Convert a V2 ``(32, 8)`` mixed chunk into 31 executable absolute qpos targets.

    The arm offsets are independently relative to ``qpos8_base``; they are *not* first differences
    and must not be cumulatively summed.  The gripper is already absolute.  Row zero is ignored.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    base = _vector8(qpos8_base, name="qpos8_base")
    if chunk.shape != (V2_HORIZON, QPOS8_DIM):
        raise ValueError(f"chunk must have shape ({V2_HORIZON}, {QPOS8_DIM}), got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("chunk must contain only finite values")

    executable = chunk[FIRST_EXECUTABLE_INDEX:]
    absolute = np.empty_like(executable)
    absolute[:, :ARM_DIM] = base[:ARM_DIM] + executable[:, :ARM_DIM]
    absolute[:, GRIPPER_INDEX] = executable[:, GRIPPER_INDEX]
    return absolute


@dataclasses.dataclass(frozen=True)
class SafetyLimits:
    """Hard absolute and per-command limits for the eight active joints."""

    joint_lower: Sequence[float]
    joint_upper: Sequence[float]
    max_delta: Sequence[float]

    def __post_init__(self) -> None:
        lower = _vector8(self.joint_lower, name="joint_lower")
        upper = _vector8(self.joint_upper, name="joint_upper")
        max_delta = _vector8(self.max_delta, name="max_delta")
        if np.any(lower >= upper):
            raise ValueError("every joint_lower value must be less than joint_upper")
        if np.any(max_delta <= 0):
            raise ValueError("every max_delta value must be positive")
        # Store immutable primitives rather than caller-owned mutable arrays.
        object.__setattr__(self, "joint_lower", tuple(float(value) for value in lower))
        object.__setattr__(self, "joint_upper", tuple(float(value) for value in upper))
        object.__setattr__(self, "max_delta", tuple(float(value) for value in max_delta))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.joint_lower, dtype=np.float32),
            np.asarray(self.joint_upper, dtype=np.float32),
            np.asarray(self.max_delta, dtype=np.float32),
        )


@dataclasses.dataclass
class ClampCounters:
    """Auditable counts; qualification requires every clamp count to remain zero."""

    command_count: int = 0
    joint_clamp_commands: int = 0
    joint_clamped_values: int = 0
    rate_clamp_commands: int = 0
    rate_clamped_values: int = 0
    nonfinite_fallbacks: int = 0

    @property
    def clamp_commands(self) -> int:
        return self.joint_clamp_commands + self.rate_clamp_commands

    @property
    def clamp_values(self) -> int:
        return self.joint_clamped_values + self.rate_clamped_values


class SafetyLimiter:
    """Apply hard limits and retain separate rate/joint clamp counters."""

    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits
        self.counters = ClampCounters()

    def reset_counters(self) -> None:
        self.counters = ClampCounters()

    def record_nonfinite_fallback(self) -> None:
        self.counters.command_count += 1
        self.counters.nonfinite_fallbacks += 1

    def apply(self, desired: np.ndarray | Sequence[float], current: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return a finite bounded command; a non-finite desired command becomes a hold."""
        desired = _vector8(desired, name="desired", finite=False)
        current = _vector8(current, name="current")
        self.counters.command_count += 1
        if not np.isfinite(desired).all():
            self.counters.nonfinite_fallbacks += 1
            return current.copy()

        lower, upper, max_delta = self.limits.arrays()
        joint_limited = np.clip(desired, lower, upper)
        initial_joint_mask = joint_limited != desired

        rate_limited = np.clip(joint_limited, current - max_delta, current + max_delta)
        rate_mask = rate_limited != joint_limited
        if np.any(rate_mask):
            self.counters.rate_clamp_commands += 1
            self.counters.rate_clamped_values += int(np.count_nonzero(rate_mask))

        # Absolute limits take precedence if an externally supplied current state is already out
        # of bounds. Count this second clip too, so every change to the emitted command remains
        # visible to the zero-clamp acceptance gate.
        safe = np.clip(rate_limited, lower, upper)
        final_joint_mask = safe != rate_limited
        joint_mask = initial_joint_mask | final_joint_mask
        if np.any(joint_mask):
            self.counters.joint_clamp_commands += 1
            self.counters.joint_clamped_values += int(np.count_nonzero(joint_mask))
        return safe.astype(np.float32, copy=False)


@dataclasses.dataclass(frozen=True)
class EnsembleDebug:
    exec_step: int
    candidate_count: int
    weights: tuple[float, ...]
    used_fallback: bool


class TemporalEnsemblerV2:
    """Stateful official-style temporal ensemble with a fail-closed hold path."""

    def __init__(
        self,
        limits: SafetyLimits,
        *,
        first_n: int = DEFAULT_ENSEMBLE_FIRST_N,
        ensemble_k: float = DEFAULT_ENSEMBLE_K,
    ) -> None:
        if not 1 <= first_n <= V2_HORIZON - FIRST_EXECUTABLE_INDEX:
            raise ValueError(f"first_n must be in [1, {V2_HORIZON - FIRST_EXECUTABLE_INDEX}]")
        if not np.isfinite(ensemble_k) or ensemble_k < 0:
            raise ValueError("ensemble_k must be finite and non-negative")
        self.first_n = int(first_n)
        self.ensemble_k = float(ensemble_k)
        self.limiter = SafetyLimiter(limits)
        self._history: list[tuple[int, np.ndarray]] = []
        self._exec_step = 0
        self._last_safe_target: np.ndarray | None = None
        self.last_debug: EnsembleDebug | None = None

    @property
    def counters(self) -> ClampCounters:
        return self.limiter.counters

    @property
    def exec_step(self) -> int:
        return self._exec_step

    def reset(self, qpos8: np.ndarray | Sequence[float]) -> np.ndarray:
        """Start an episode and return its finite initial hold command."""
        qpos8 = _vector8(qpos8, name="qpos8")
        self._history.clear()
        self._exec_step = 0
        self._last_safe_target = qpos8.copy()
        self.last_debug = None
        self.limiter.reset_counters()
        return self._last_safe_target.copy()

    def safe_hold(self, qpos8: np.ndarray | Sequence[float] | None = None) -> np.ndarray:
        """Return the current finite hold, falling back to the last emitted safe target."""
        if qpos8 is not None:
            candidate = _vector8(qpos8, name="qpos8", finite=False)
            if np.isfinite(candidate).all():
                return candidate.copy()
        if self._last_safe_target is None:
            raise RuntimeError("reset with a finite qpos8 before requesting a safe hold")
        return self._last_safe_target.copy()

    def step(self, chunk: np.ndarray, qpos8_current: np.ndarray | Sequence[float]) -> np.ndarray:
        """Add one replan and emit the bounded target for the current control step.

        A shape/contract mismatch raises immediately.  A correctly shaped chunk containing NaN or
        Inf, or a transient non-finite qpos observation after reset, clears all stale predictions
        and emits a finite hold.  This prevents a bad inference from being hidden by old chunks.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (V2_HORIZON, QPOS8_DIM):
            raise ValueError(f"chunk must have shape ({V2_HORIZON}, {QPOS8_DIM}), got {chunk.shape}")
        current = _vector8(qpos8_current, name="qpos8_current", finite=False)
        if self._last_safe_target is None:
            if not np.isfinite(current).all():
                raise RuntimeError("first qpos8_current must be finite; call reset before step")
            self.reset(current)

        if not np.isfinite(chunk).all() or not np.isfinite(current).all():
            hold = self.safe_hold(current)
            self._history.clear()
            self.limiter.record_nonfinite_fallback()
            self.last_debug = EnsembleDebug(
                exec_step=self._exec_step,
                candidate_count=0,
                weights=(),
                used_fallback=True,
            )
            self._exec_step += 1
            self._last_safe_target = hold.copy()
            return hold

        absolute = active_mixed_chunk_to_absolute(chunk, current)[: self.first_n]
        self._history.append((self._exec_step, absolute))

        candidates: list[np.ndarray] = []
        retained: list[tuple[int, np.ndarray]] = []
        for infer_step, future in self._history:  # insertion order is oldest -> newest
            delay = self._exec_step - infer_step
            if 0 <= delay < future.shape[0]:
                retained.append((infer_step, future))
                candidates.append(future[delay])
        self._history = retained
        if not candidates:  # Defensive: the newly appended chunk must always contribute row 1.
            raise RuntimeError(f"no valid temporal-ensemble candidate at exec step {self._exec_step}")

        actions = np.stack(candidates)
        weights = np.exp(-self.ensemble_k * np.arange(len(candidates), dtype=np.float32))
        weights /= weights.sum()
        desired = np.sum(actions * weights[:, None], axis=0)
        safe = self.limiter.apply(desired, current)
        self.last_debug = EnsembleDebug(
            exec_step=self._exec_step,
            candidate_count=len(candidates),
            weights=tuple(float(weight) for weight in weights),
            used_fallback=False,
        )
        self._exec_step += 1
        self._last_safe_target = safe.copy()
        return safe


@dataclasses.dataclass(frozen=True)
class TraceQualification:
    passed: bool
    finite: bool
    j5_limit_contact: bool
    first_gripper_jump: bool
    j5_negative_to_positive_reversal: bool
    clamp_used: bool
    first_gripper_delta: float


def qualify_control_trace(
    qpos8_before: np.ndarray,
    sent_action8: np.ndarray,
    limits: SafetyLimits,
    *,
    clamp_count: int = 0,
    first_gripper_jump_max: float,
    j5_limit_margin: float = 1e-4,
    reversal_epsilon: float = 1e-4,
) -> TraceQualification:
    """Evaluate the four mandatory closed-loop V2 trace invariants.

    J5 reversal is measured from commanded displacement (sent target minus qpos immediately before
    that command), not from target-to-target differences.  A valid trace must contain a negative
    command followed later by a positive command.
    """
    before = np.asarray(qpos8_before, dtype=np.float32)
    sent = np.asarray(sent_action8, dtype=np.float32)
    if before.ndim != 2 or before.shape[1:] != (QPOS8_DIM,):
        raise ValueError(f"qpos8_before must have shape (T, {QPOS8_DIM}), got {before.shape}")
    if sent.shape != before.shape:
        raise ValueError(f"sent_action8 must have shape {before.shape}, got {sent.shape}")
    if before.shape[0] == 0:
        raise ValueError("trace must contain at least one command")
    if clamp_count < 0:
        raise ValueError("clamp_count must be non-negative")
    if not np.isfinite(first_gripper_jump_max) or first_gripper_jump_max < 0:
        raise ValueError("first_gripper_jump_max must be finite and non-negative")
    if not np.isfinite(j5_limit_margin) or j5_limit_margin < 0:
        raise ValueError("j5_limit_margin must be finite and non-negative")
    if not np.isfinite(reversal_epsilon) or reversal_epsilon < 0:
        raise ValueError("reversal_epsilon must be finite and non-negative")

    finite = bool(np.isfinite(before).all() and np.isfinite(sent).all())
    first_gripper_delta = float(abs(sent[0, GRIPPER_INDEX] - before[0, GRIPPER_INDEX])) if finite else float("nan")
    first_gripper_jump = not finite or first_gripper_delta > first_gripper_jump_max

    lower, upper, _ = limits.arrays()
    if finite:
        j5_values = np.concatenate((before[:, J5_INDEX], sent[:, J5_INDEX]))
        j5_limit_contact = bool(
            np.any(j5_values <= lower[J5_INDEX] + j5_limit_margin)
            or np.any(j5_values >= upper[J5_INDEX] - j5_limit_margin)
        )
        j5_commands = sent[:, J5_INDEX] - before[:, J5_INDEX]
        negative_indices = np.flatnonzero(j5_commands < -reversal_epsilon)
        positive_indices = np.flatnonzero(j5_commands > reversal_epsilon)
        reversal = bool(
            negative_indices.size and positive_indices.size and np.any(positive_indices > negative_indices[0])
        )
    else:
        j5_limit_contact = True
        reversal = False

    clamp_used = clamp_count > 0
    passed = finite and not j5_limit_contact and not first_gripper_jump and reversal and not clamp_used
    return TraceQualification(
        passed=passed,
        finite=finite,
        j5_limit_contact=j5_limit_contact,
        first_gripper_jump=first_gripper_jump,
        j5_negative_to_positive_reversal=reversal,
        clamp_used=clamp_used,
        first_gripper_delta=first_gripper_delta,
    )


@dataclasses.dataclass(frozen=True)
class PairedAcceptance:
    accepted: bool
    trials: int
    student_successes: int
    official_successes: int
    official_only_discordances: int
    discordance_upper_95: float


def paired_n100_acceptance(
    official_success: Sequence[bool] | np.ndarray,
    student_success: Sequence[bool] | np.ndarray,
) -> PairedAcceptance:
    """Apply the preregistered official-level gate to exactly 100 paired seeds.

    Acceptance requires at least 99 student successes and a one-sided exact 95% Clopper--Pearson
    upper bound below 5% for ``official succeeds AND student fails`` over all 100 seeds.
    """
    official = np.asarray(official_success)
    student = np.asarray(student_success)
    if official.shape != (100,) or student.shape != (100,):
        raise ValueError(
            f"official_success and student_success must both have shape (100,), got {official.shape} and {student.shape}"
        )
    if official.dtype != np.bool_ or student.dtype != np.bool_:
        raise TypeError("official_success and student_success must contain booleans")

    student_count = int(np.count_nonzero(student))
    official_count = int(np.count_nonzero(official))
    discordances = int(np.count_nonzero(official & ~student))

    # Lazy import keeps the control path independent of scipy import time.  Scipy is already a
    # project dependency and its beta quantile is the exact binomial confidence endpoint.
    if discordances == 100:
        upper = 1.0
    else:
        from scipy.stats import beta  # noqa: PLC0415

        upper = float(beta.ppf(0.95, discordances + 1, 100 - discordances))
    accepted = student_count >= 99 and upper < 0.05
    return PairedAcceptance(
        accepted=accepted,
        trials=100,
        student_successes=student_count,
        official_successes=official_count,
        official_only_discordances=discordances,
        discordance_upper_95=upper,
    )
