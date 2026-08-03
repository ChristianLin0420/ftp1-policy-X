"""Deterministic mask generation for MoT-JEPA.

Two properties matter more than anything else in this file.

**No RNG ever runs inside the model.** ``torch.utils.checkpoint`` is used with
``preserve_rng_state=False`` throughout this repository (``ftp1_pytorch.py:127-133``,
``ftp1_gemma_pytorch.py:491-501``). Any randomness inside a checkpointed region re-samples
on recompute, so the backward pass would differentiate a *different* function than the one
that produced the loss -- with no crash, no NaN, and no assertion, just a model that
plateaus. Masks are therefore built here, from ``numpy.random.Generator(PCG64(seed))``
only, and handed to the model as plain tensors.

**The masking mode is a pure function of the optimizer step**, not of batch content. Every
DDP rank must choose the same mode on the same step: the mode determines which parameters
receive gradients, and ranks disagreeing about that deadlocks the reducer. Deriving the
mode from ``global_step`` also makes a 40-job requeue chain replay identically.

Target and context sets are complementary and disjoint: context is exactly the tokens not
selected as targets.
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np
import torch

from openpi.mot_jepa.layout import STREAM_ORDER
from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.layout import TokenLayout

_MASK_SALT = 0x4A455041  # "JEPA"; keeps mask seeds from colliding with other seed streams.


class MaskMode(enum.IntEnum):
    """Masking modes from the design document, in declaration order.

    ``V`` and ``V_HARD`` produce video targets and do nothing for cross-modal binding -- a
    model that ignores touch solves them. ``T`` and ``X`` are the train-time binding
    *meters*. ``T_HARD`` is the structurally forcing one: with tactile removed entirely, no
    tactile input, real or shuffled, can satisfy it.
    """

    V = 0
    T = 1
    T_HARD = 2
    V_HARD = 3
    X = 4


#: Design-document mode probabilities, indexed by :class:`MaskMode`.
DEFAULT_MODE_PROBS: tuple[float, float, float, float, float] = (0.35, 0.20, 0.10, 0.10, 0.25)


@dataclasses.dataclass(frozen=True)
class MaskSpec:
    """Masking hyperparameters.

    Args:
        layout: Token layout the masks address.
        mode_probs: Probability per :class:`MaskMode`, in enum order.
        video_mask_frac: Fraction of video spatial positions masked in mode ``V``.
        x_video_mask_frac: Same, for mode ``X``.
        tactile_window_steps: Allowed contiguous tactile window lengths, in tubelet steps.
        video_window_steps: Allowed contiguous video window lengths, in tubelet steps.
        num_blocks: Number of tube blocks unioned to build a spatial video mask.
        block_scale: Min/max block area as a fraction of the spatial grid.
        block_aspect: Min/max block aspect ratio.
        min_targets_per_stream: Target floor so every prediction head gets a gradient and
            the loss dict never reports a structurally-absent term.
    """

    layout: TokenLayout
    mode_probs: tuple[float, ...] = DEFAULT_MODE_PROBS
    video_mask_frac: float = 0.88
    x_video_mask_frac: float = 0.75
    tactile_window_steps: tuple[int, ...] = (2, 3, 4)
    video_window_steps: tuple[int, ...] = (2, 3, 4)
    num_blocks: int = 4
    block_scale: tuple[float, float] = (0.15, 0.35)
    block_aspect: tuple[float, float] = (0.66, 1.5)
    min_targets_per_stream: int = 8

    def __post_init__(self) -> None:
        if len(self.mode_probs) != len(MaskMode):
            raise ValueError(f"mode_probs must have {len(MaskMode)} entries, got {len(self.mode_probs)}")
        if not np.isclose(sum(self.mode_probs), 1.0):
            raise ValueError(f"mode_probs must sum to 1.0, got {sum(self.mode_probs)}")
        if min(self.mode_probs) < 0.0:
            raise ValueError("mode_probs must be non-negative")
        max_window = max(*self.tactile_window_steps, *self.video_window_steps)
        if max_window >= self.layout.num_steps:
            raise ValueError(
                f"window length {max_window} must be < num_steps={self.layout.num_steps} "
                "so that at least one instant survives"
            )


@dataclasses.dataclass
class ClipMasks:
    """Gather indices for one batch.

    ``ctx_index`` and ``tgt_index`` are row-wise ascending and disjoint, and every row has
    the same per-stream counts. That uniformity is what keeps each stream contiguous at
    identical offsets across the batch, which in turn is what lets modality-local attention
    run as slice-wise SDPA calls with no attention mask.
    """

    mode: torch.Tensor  # () int64
    ctx_index: torch.Tensor  # (B, L_ctx) int64
    tgt_index: torch.Tensor  # (B, L_tgt) int64
    ctx_bounds: torch.Tensor  # (3, 2) int64, per StreamId, into ctx_index columns
    tgt_bounds: torch.Tensor  # (3, 2) int64, per StreamId, into tgt_index columns
    ctx_expert_bounds: torch.Tensor  # (2, 2) int64, per ExpertId
    tgt_expert_bounds: torch.Tensor  # (2, 2) int64, per ExpertId
    seeds: torch.Tensor  # (B,) int64, echoed so tests can recompute the batch

    @property
    def batch_size(self) -> int:
        return int(self.ctx_index.shape[0])

    @property
    def mode_enum(self) -> MaskMode:
        return MaskMode(int(self.mode))

    def to(self, device: torch.device | str, *, non_blocking: bool = True) -> ClipMasks:
        moved = {f.name: getattr(self, f.name).to(device, non_blocking=non_blocking) for f in dataclasses.fields(self)}
        return ClipMasks(**moved)

    def pin_memory(self) -> ClipMasks:
        moved = {f.name: getattr(self, f.name).pin_memory() for f in dataclasses.fields(self)}
        return ClipMasks(**moved)


def derive_mask_seed(base_seed: int, step: int, sample_index: int) -> int:
    """Seed for one clip's mask.

    Depends only on values every rank can reproduce after a requeue, and never on worker id
    or ``num_workers``.
    """
    mixed = (int(base_seed) ^ _MASK_SALT) & 0xFFFFFFFFFFFF
    return int((mixed * 0x9E3779B97F4A7C15 + int(step) * 0x100000001B3 + int(sample_index)) % (2**63 - 1))


def draw_mode(spec: MaskSpec, step: int, base_seed: int) -> MaskMode:
    """Draw the batch's masking mode. Identical on every rank for a given ``step``."""
    rng = np.random.Generator(np.random.PCG64(derive_mask_seed(base_seed, step, -1)))
    return MaskMode(int(rng.choice(len(MaskMode), p=np.asarray(spec.mode_probs, dtype=np.float64))))


def _sample_tube_mask(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    num_masked: int,
    spec: MaskSpec,
) -> np.ndarray:
    """Spatial tube mask: union random blocks, then trim to *exactly* ``num_masked`` cells.

    Tube masking (a 2-D block extended across all time) rather than independent per-frame
    masking is what makes the video objective non-trivial: a temporally local copy cannot
    solve it.
    """
    total = grid_h * grid_w
    num_masked = int(np.clip(num_masked, 0, total))
    masked = np.zeros(total, dtype=bool)
    if num_masked == 0:
        return masked

    for _ in range(spec.num_blocks * 8):
        if masked.sum() >= num_masked:
            break
        scale = rng.uniform(*spec.block_scale)
        aspect = rng.uniform(*spec.block_aspect)
        area = max(1.0, scale * total)
        block_h = int(np.clip(round(np.sqrt(area * aspect)), 1, grid_h))
        block_w = int(np.clip(round(np.sqrt(area / aspect)), 1, grid_w))
        top = int(rng.integers(0, grid_h - block_h + 1))
        left = int(rng.integers(0, grid_w - block_w + 1))
        block = np.zeros((grid_h, grid_w), dtype=bool)
        block[top : top + block_h, left : left + block_w] = True
        masked |= block.reshape(-1)

    # Trim or grow to hit the exact count so every row of the batch has identical shape.
    excess = int(masked.sum()) - num_masked
    if excess > 0:
        on = np.flatnonzero(masked)
        masked[rng.choice(on, size=excess, replace=False)] = False
    elif excess < 0:
        off = np.flatnonzero(~masked)
        masked[rng.choice(off, size=-excess, replace=False)] = True
    return masked


def _video_target_indices(rng: np.random.Generator, layout: TokenLayout, spec: MaskSpec, frac: float) -> np.ndarray:
    grid_h, grid_w = layout.video_grid
    num_masked_cells = round(frac * grid_h * grid_w)
    spatial = _sample_tube_mask(rng, grid_h, grid_w, num_masked_cells, spec)
    cells = np.flatnonzero(spatial)
    base = layout.stream_slices[StreamId.VIDEO].start
    per_step = layout.video_tokens_per_step
    steps = np.arange(layout.num_steps, dtype=np.int64)
    return (base + steps[:, None] * per_step + cells[None, :]).reshape(-1)


def _window_start(rng: np.random.Generator, num_steps: int, window: int) -> int:
    return int(rng.integers(0, num_steps - window + 1))


def _floor_targets(
    rng: np.random.Generator,
    layout: TokenLayout,
    stream: StreamId,
    count: int,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Sample ``count`` scattered target tokens from ``stream``.

    Guarantees every prediction head receives a gradient in every mode, which keeps the
    loss dictionary's key set stable and its terms genuinely non-zero.
    """
    pool = layout.token_indices(stream)
    if exclude is not None and exclude.size:
        pool = np.setdiff1d(pool, exclude, assume_unique=False)
    count = int(min(count, pool.size))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(pool, size=count, replace=False).astype(np.int64)


def _targets_for_mode(
    rng: np.random.Generator,
    spec: MaskSpec,
    mode: MaskMode,
    window_tac: int,
    window_vid: int,
) -> dict[StreamId, np.ndarray]:
    """Target token indices per stream for one clip."""
    layout = spec.layout
    floor = spec.min_targets_per_stream
    out: dict[StreamId, np.ndarray] = {s: np.empty(0, dtype=np.int64) for s in STREAM_ORDER}

    def tactile_window(window: int) -> tuple[np.ndarray, np.ndarray]:
        start = _window_start(rng, layout.num_steps, window)
        steps = np.arange(start, start + window, dtype=np.int64)
        return (
            layout.step_token_indices(StreamId.GEL, steps),
            layout.step_token_indices(StreamId.LOWDIM, steps),
        )

    if mode is MaskMode.V:
        out[StreamId.VIDEO] = _video_target_indices(rng, layout, spec, spec.video_mask_frac)
        out[StreamId.GEL] = _floor_targets(rng, layout, StreamId.GEL, floor)
        out[StreamId.LOWDIM] = _floor_targets(rng, layout, StreamId.LOWDIM, floor)

    elif mode is MaskMode.T:
        out[StreamId.GEL], out[StreamId.LOWDIM] = tactile_window(window_tac)
        out[StreamId.VIDEO] = _floor_targets(rng, layout, StreamId.VIDEO, floor)

    elif mode is MaskMode.T_HARD:
        out[StreamId.GEL] = layout.token_indices(StreamId.GEL)
        out[StreamId.LOWDIM] = layout.token_indices(StreamId.LOWDIM)
        out[StreamId.VIDEO] = _floor_targets(rng, layout, StreamId.VIDEO, floor)

    elif mode is MaskMode.V_HARD:
        start = _window_start(rng, layout.num_steps, window_vid)
        steps = np.arange(start, start + window_vid, dtype=np.int64)
        out[StreamId.VIDEO] = layout.step_token_indices(StreamId.VIDEO, steps)
        out[StreamId.GEL] = _floor_targets(rng, layout, StreamId.GEL, floor)
        out[StreamId.LOWDIM] = _floor_targets(rng, layout, StreamId.LOWDIM, floor)

    elif mode is MaskMode.X:
        out[StreamId.VIDEO] = _video_target_indices(rng, layout, spec, spec.x_video_mask_frac)
        out[StreamId.GEL], out[StreamId.LOWDIM] = tactile_window(window_tac)

    else:  # pragma: no cover - exhaustive over MaskMode
        raise ValueError(f"unhandled mode {mode}")

    return out


def expected_target_counts(spec: MaskSpec, mode: MaskMode, window_tac: int, window_vid: int) -> dict[StreamId, int]:
    """Per-stream target counts for a mode. Identical for every clip in the batch."""
    layout = spec.layout
    floor = spec.min_targets_per_stream
    grid_h, grid_w = layout.video_grid

    def video_from_frac(frac: float) -> int:
        return layout.num_steps * round(frac * grid_h * grid_w)

    if mode is MaskMode.V:
        return {
            StreamId.VIDEO: video_from_frac(spec.video_mask_frac),
            StreamId.GEL: floor,
            StreamId.LOWDIM: floor,
        }
    if mode is MaskMode.T:
        return {
            StreamId.VIDEO: floor,
            StreamId.GEL: window_tac * layout.gel_tokens_per_step,
            StreamId.LOWDIM: window_tac * layout.lowdim_slots,
        }
    if mode is MaskMode.T_HARD:
        return {
            StreamId.VIDEO: floor,
            StreamId.GEL: layout.num_gel_tokens,
            StreamId.LOWDIM: layout.num_lowdim_tokens,
        }
    if mode is MaskMode.V_HARD:
        return {
            StreamId.VIDEO: window_vid * layout.video_tokens_per_step,
            StreamId.GEL: floor,
            StreamId.LOWDIM: floor,
        }
    if mode is MaskMode.X:
        return {
            StreamId.VIDEO: video_from_frac(spec.x_video_mask_frac),
            StreamId.GEL: window_tac * layout.gel_tokens_per_step,
            StreamId.LOWDIM: window_tac * layout.lowdim_slots,
        }
    raise ValueError(f"unhandled mode {mode}")  # pragma: no cover


def _bounds_from_counts(counts: dict[StreamId, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Column ranges per stream and per expert inside an ascending index row."""
    stream_bounds = torch.zeros(len(STREAM_ORDER), 2, dtype=torch.int64)
    offset = 0
    for stream in STREAM_ORDER:
        stream_bounds[int(stream), 0] = offset
        offset += counts[stream]
        stream_bounds[int(stream), 1] = offset

    expert_bounds = torch.zeros(len(ExpertId), 2, dtype=torch.int64)
    expert_bounds[int(ExpertId.VIDEO)] = stream_bounds[int(StreamId.VIDEO)]
    expert_bounds[int(ExpertId.TACTILE), 0] = stream_bounds[int(StreamId.GEL), 0]
    expert_bounds[int(ExpertId.TACTILE), 1] = stream_bounds[int(StreamId.LOWDIM), 1]
    return stream_bounds, expert_bounds


def build_batch_masks(
    spec: MaskSpec,
    *,
    step: int,
    batch_size: int,
    base_seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> ClipMasks:
    """Build masks for one optimizer step.

    The mode and all per-stream counts depend only on ``step``, so every rank agrees; the
    block positions and window offsets depend on the global sample index, so clips still
    differ within and across ranks.
    """
    layout = spec.layout
    mode = draw_mode(spec, step, base_seed)

    # Window lengths are per batch, not per sample, so every row has identical shape.
    shape_rng = np.random.Generator(np.random.PCG64(derive_mask_seed(base_seed, step, -2)))
    window_tac = int(shape_rng.choice(np.asarray(spec.tactile_window_steps)))
    window_vid = int(shape_rng.choice(np.asarray(spec.video_window_steps)))

    tgt_counts = expected_target_counts(spec, mode, window_tac, window_vid)
    ctx_counts = {s: layout.stream_counts()[s] - tgt_counts[s] for s in STREAM_ORDER}

    num_tgt = sum(tgt_counts.values())
    num_ctx = sum(ctx_counts.values())
    ctx_index = torch.empty(batch_size, num_ctx, dtype=torch.int64)
    tgt_index = torch.empty(batch_size, num_tgt, dtype=torch.int64)
    seeds = torch.empty(batch_size, dtype=torch.int64)

    all_tokens = np.arange(layout.num_tokens, dtype=np.int64)
    for row in range(batch_size):
        sample_index = rank * batch_size + row + step * world_size * batch_size
        seed = derive_mask_seed(base_seed, step, sample_index)
        rng = np.random.Generator(np.random.PCG64(seed))
        per_stream = _targets_for_mode(rng, spec, mode, window_tac, window_vid)

        tgt = np.concatenate([np.sort(per_stream[s]) for s in STREAM_ORDER])
        if tgt.size != num_tgt:
            raise RuntimeError(
                f"mode {mode.name} produced {tgt.size} targets, expected {num_tgt}; "
                "target counts must be data-independent"
            )
        is_target = np.zeros(layout.num_tokens, dtype=bool)
        is_target[tgt] = True

        tgt_index[row] = torch.from_numpy(tgt)
        ctx_index[row] = torch.from_numpy(all_tokens[~is_target])
        seeds[row] = seed

    ctx_bounds, ctx_expert_bounds = _bounds_from_counts(ctx_counts)
    tgt_bounds, tgt_expert_bounds = _bounds_from_counts(tgt_counts)
    return ClipMasks(
        mode=torch.tensor(int(mode), dtype=torch.int64),
        ctx_index=ctx_index,
        tgt_index=tgt_index,
        ctx_bounds=ctx_bounds,
        tgt_bounds=tgt_bounds,
        ctx_expert_bounds=ctx_expert_bounds,
        tgt_expert_bounds=tgt_expert_bounds,
        seeds=seeds,
    )


def assert_mask_invariants(masks: ClipMasks, spec: MaskSpec) -> None:
    """Validate the four properties the rest of the package relies on.

    1. Context and target rows are ascending and disjoint, and together cover every token.
    2. Row shapes are data-independent (already implied by the tensor shape, re-checked
       against the analytic counts).
    3. Every stream has at least one target, and every expert has at least one context
       token -- except in ``T_HARD``, where the total absence of tactile context is the
       entire point of the mode.
    4. Stream bounds partition each row into contiguous runs, the precondition for
       slice-wise modality-local attention.
    """
    layout = spec.layout
    mode = masks.mode_enum
    num_tokens = layout.num_tokens

    if masks.ctx_index.shape[1] + masks.tgt_index.shape[1] != num_tokens:
        raise AssertionError(f"ctx({masks.ctx_index.shape[1]}) + tgt({masks.tgt_index.shape[1]}) != L({num_tokens})")

    for row in range(masks.batch_size):
        ctx = masks.ctx_index[row]
        tgt = masks.tgt_index[row]
        if not bool(torch.all(ctx[1:] > ctx[:-1])):
            raise AssertionError(f"ctx_index row {row} is not strictly ascending")
        if not bool(torch.all(tgt[1:] > tgt[:-1])):
            raise AssertionError(f"tgt_index row {row} is not strictly ascending")
        union = torch.cat([ctx, tgt]).sort().values
        if not bool(torch.equal(union, torch.arange(num_tokens, dtype=torch.int64))):
            raise AssertionError(f"row {row}: ctx and tgt are not a disjoint cover of all tokens")

    for stream in STREAM_ORDER:
        lo, hi = (int(v) for v in masks.tgt_bounds[int(stream)])
        if hi - lo < 1:
            raise AssertionError(f"stream {stream.name} has no targets in mode {mode.name}")
        sl = layout.stream_slices[stream]
        block = masks.tgt_index[:, lo:hi]
        if block.numel() and not bool(((block >= sl.start) & (block < sl.stop)).all()):
            raise AssertionError(f"tgt_bounds for {stream.name} do not select that stream's tokens")

    for expert in ExpertId:
        lo, hi = (int(v) for v in masks.ctx_expert_bounds[int(expert)])
        if hi - lo < 1 and not (mode is MaskMode.T_HARD and expert is ExpertId.TACTILE):
            raise AssertionError(f"expert {expert.name} has no context tokens in mode {mode.name}")
