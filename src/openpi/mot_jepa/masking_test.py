"""Mask-generation determinism and invariants.

These are the cheapest guards against the highest-cost silent failure in this design:
``torch.utils.checkpoint(..., preserve_rng_state=False)`` re-samples any RNG inside a
checkpointed region on recompute, so a mask drawn inside the model would make the backward
pass differentiate a different function than the forward computed -- with no crash and no
NaN. Keeping every draw here, seeded and reproducible, is what makes that impossible.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from openpi.mot_jepa.layout import LAYOUT_PILOT
from openpi.mot_jepa.layout import STREAM_ORDER
from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.masking import ClipMasks
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import assert_mask_invariants
from openpi.mot_jepa.masking import build_batch_masks
from openpi.mot_jepa.masking import derive_mask_seed
from openpi.mot_jepa.masking import draw_mode
from openpi.mot_jepa.masking import expected_target_counts

BASE_SEED = 42


def spec_forcing(mode: MaskMode) -> MaskSpec:
    """A spec whose mode draw is deterministic, so each mode can be exercised directly."""
    probs = [0.0] * len(MaskMode)
    probs[int(mode)] = 1.0
    return MaskSpec(layout=LAYOUT_PILOT, mode_probs=tuple(probs))


ALL_MODES = list(MaskMode)


def test_build_batch_masks_is_bitwise_reproducible():
    spec = MaskSpec(layout=LAYOUT_PILOT)
    a = build_batch_masks(spec, step=17, batch_size=4, base_seed=BASE_SEED)
    b = build_batch_masks(spec, step=17, batch_size=4, base_seed=BASE_SEED)
    for field in dataclasses.fields(ClipMasks):
        lhs, rhs = getattr(a, field.name), getattr(b, field.name)
        assert torch.equal(lhs, rhs), field.name


def test_masks_differ_across_steps():
    spec = MaskSpec(layout=LAYOUT_PILOT)
    a = build_batch_masks(spec, step=1, batch_size=4, base_seed=BASE_SEED)
    b = build_batch_masks(spec, step=2, batch_size=4, base_seed=BASE_SEED)
    assert not torch.equal(a.seeds, b.seeds)


def test_mode_is_a_pure_function_of_step_so_every_rank_agrees():
    """Ranks that disagree about the mode use different parameter subsets and deadlock DDP."""
    spec = MaskSpec(layout=LAYOUT_PILOT)
    for step in range(200):
        modes = {draw_mode(spec, step, BASE_SEED) for _ in range(3)}
        assert len(modes) == 1
        # Independent of rank/batch_size, which only perturb per-sample block positions.
        rank0 = build_batch_masks(spec, step=step, batch_size=2, base_seed=BASE_SEED, rank=0, world_size=2)
        rank1 = build_batch_masks(spec, step=step, batch_size=2, base_seed=BASE_SEED, rank=1, world_size=2)
        assert int(rank0.mode) == int(rank1.mode)
        assert rank0.ctx_index.shape == rank1.ctx_index.shape
        assert rank0.tgt_index.shape == rank1.tgt_index.shape
        assert torch.equal(rank0.tgt_bounds, rank1.tgt_bounds)


def test_ranks_get_different_block_positions():
    spec = MaskSpec(layout=LAYOUT_PILOT)
    rank0 = build_batch_masks(spec, step=5, batch_size=2, base_seed=BASE_SEED, rank=0, world_size=2)
    rank1 = build_batch_masks(spec, step=5, batch_size=2, base_seed=BASE_SEED, rank=1, world_size=2)
    assert not torch.equal(rank0.seeds, rank1.seeds)


def test_mode_distribution_tracks_the_design_document_probabilities():
    spec = MaskSpec(layout=LAYOUT_PILOT)
    counts = np.zeros(len(MaskMode))
    trials = 20_000
    for step in range(trials):
        counts[int(draw_mode(spec, step, BASE_SEED))] += 1
    observed = counts / trials
    np.testing.assert_allclose(observed, np.asarray(spec.mode_probs), atol=0.02)


@pytest.mark.parametrize("mode", ALL_MODES, ids=[m.name for m in ALL_MODES])
def test_invariants_hold_for_every_mode(mode):
    spec = spec_forcing(mode)
    masks = build_batch_masks(spec, step=3, batch_size=6, base_seed=BASE_SEED)
    assert masks.mode_enum is mode
    assert_mask_invariants(masks, spec)


@pytest.mark.parametrize("mode", ALL_MODES, ids=[m.name for m in ALL_MODES])
def test_target_counts_are_data_independent(mode):
    """Every row must have identical per-stream counts, or the batched gather is ragged."""
    spec = spec_forcing(mode)
    masks = build_batch_masks(spec, step=9, batch_size=8, base_seed=BASE_SEED)
    for stream in STREAM_ORDER:
        lo, hi = (int(v) for v in masks.tgt_bounds[int(stream)])
        block = masks.tgt_index[:, lo:hi]
        sl = spec.layout.stream_slices[stream]
        assert bool(((block >= sl.start) & (block < sl.stop)).all())


@pytest.mark.parametrize("mode", ALL_MODES, ids=[m.name for m in ALL_MODES])
def test_every_stream_receives_targets_so_every_head_gets_a_gradient(mode):
    spec = spec_forcing(mode)
    masks = build_batch_masks(spec, step=11, batch_size=2, base_seed=BASE_SEED)
    for stream in STREAM_ORDER:
        lo, hi = (int(v) for v in masks.tgt_bounds[int(stream)])
        assert hi - lo >= 1, f"{stream.name} has no targets in {mode.name}"


def test_t_hard_removes_all_tactile_context():
    """The forcing property: no tactile input, real or shuffled, can satisfy T_HARD."""
    spec = spec_forcing(MaskMode.T_HARD)
    masks = build_batch_masks(spec, step=1, batch_size=3, base_seed=BASE_SEED)
    lo, hi = (int(v) for v in masks.ctx_expert_bounds[int(ExpertId.TACTILE)])
    assert hi - lo == 0
    layout = spec.layout
    assert masks.ctx_index.shape[1] == layout.num_video_tokens - spec.min_targets_per_stream


def test_v_mode_leaves_all_tactile_visible_apart_from_the_target_floor():
    spec = spec_forcing(MaskMode.V)
    masks = build_batch_masks(spec, step=1, batch_size=3, base_seed=BASE_SEED)
    layout = spec.layout
    lo, hi = (int(v) for v in masks.ctx_expert_bounds[int(ExpertId.TACTILE)])
    tactile_ctx = hi - lo
    tactile_total = layout.num_gel_tokens + layout.num_lowdim_tokens
    assert tactile_ctx == tactile_total - 2 * spec.min_targets_per_stream


@pytest.mark.parametrize("mode", [MaskMode.T, MaskMode.X], ids=["T", "X"])
def test_tactile_window_is_temporally_contiguous(mode):
    """Modes T and X must remove whole instants; scattered dropout is interpolable."""
    spec = spec_forcing(mode)
    masks = build_batch_masks(spec, step=7, batch_size=4, base_seed=BASE_SEED)
    layout = spec.layout
    lo, hi = (int(v) for v in masks.tgt_bounds[int(StreamId.GEL)])
    for row in range(masks.batch_size):
        gel_targets = masks.tgt_index[row, lo:hi]
        steps = torch.unique(layout.coords[gel_targets, 0])
        assert torch.equal(steps, torch.arange(int(steps[0]), int(steps[-1]) + 1))
        assert len(steps) in spec.tactile_window_steps


def test_v_hard_window_is_temporally_contiguous():
    spec = spec_forcing(MaskMode.V_HARD)
    masks = build_batch_masks(spec, step=7, batch_size=4, base_seed=BASE_SEED)
    layout = spec.layout
    lo, hi = (int(v) for v in masks.tgt_bounds[int(StreamId.VIDEO)])
    for row in range(masks.batch_size):
        steps = torch.unique(layout.coords[masks.tgt_index[row, lo:hi], 0])
        assert torch.equal(steps, torch.arange(int(steps[0]), int(steps[-1]) + 1))


def test_window_offsets_vary_across_the_batch():
    spec = spec_forcing(MaskMode.T)
    masks = build_batch_masks(spec, step=7, batch_size=16, base_seed=BASE_SEED)
    lo = int(masks.tgt_bounds[int(StreamId.GEL), 0])
    starts = {int(masks.tgt_index[row, lo]) for row in range(masks.batch_size)}
    assert len(starts) > 1


@pytest.mark.parametrize("mode", ALL_MODES, ids=[m.name for m in ALL_MODES])
def test_expected_target_counts_matches_realized_counts(mode):
    spec = spec_forcing(mode)
    masks = build_batch_masks(spec, step=13, batch_size=2, base_seed=BASE_SEED)
    realized = {
        stream: int(masks.tgt_bounds[int(stream), 1] - masks.tgt_bounds[int(stream), 0]) for stream in STREAM_ORDER
    }
    total = sum(realized.values())
    assert total == masks.tgt_index.shape[1]
    # The analytic helper must agree for at least one legal window pair.
    ok = any(
        expected_target_counts(spec, mode, wt, wv) == realized
        for wt in spec.tactile_window_steps
        for wv in spec.video_window_steps
    )
    assert ok, f"{mode.name}: realized {realized} matches no analytic count"


def test_video_mask_fraction_is_close_to_the_configured_value():
    spec = spec_forcing(MaskMode.V)
    masks = build_batch_masks(spec, step=2, batch_size=4, base_seed=BASE_SEED)
    layout = spec.layout
    lo, hi = (int(v) for v in masks.tgt_bounds[int(StreamId.VIDEO)])
    frac = (hi - lo) / layout.num_video_tokens
    assert abs(frac - spec.video_mask_frac) < 0.02


def test_derive_mask_seed_is_stable_and_well_separated():
    a = derive_mask_seed(BASE_SEED, 100, 7)
    assert a == derive_mask_seed(BASE_SEED, 100, 7)
    assert a != derive_mask_seed(BASE_SEED, 100, 8)
    assert a != derive_mask_seed(BASE_SEED, 101, 7)
    assert a != derive_mask_seed(BASE_SEED + 1, 100, 7)


def test_seed_does_not_depend_on_worker_or_thread_state():
    """Guards against a seed accidentally derived from worker id or global RNG state."""
    spec = MaskSpec(layout=LAYOUT_PILOT)
    baseline = build_batch_masks(spec, step=4, batch_size=4, base_seed=BASE_SEED)
    np.random.seed(1234)
    torch.manual_seed(4321)
    _ = np.random.rand(1000)
    _ = torch.randn(1000)
    perturbed = build_batch_masks(spec, step=4, batch_size=4, base_seed=BASE_SEED)
    assert torch.equal(baseline.ctx_index, perturbed.ctx_index)
    assert torch.equal(baseline.tgt_index, perturbed.tgt_index)


def test_invalid_specs_raise():
    with pytest.raises(ValueError, match="must have"):
        MaskSpec(layout=LAYOUT_PILOT, mode_probs=(0.5, 0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="must sum to"):
        MaskSpec(layout=LAYOUT_PILOT, mode_probs=(0.5, 0.1, 0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="at least one instant survives"):
        # A window as long as the clip would leave no surviving instant.
        MaskSpec(layout=LAYOUT_PILOT, tactile_window_steps=(LAYOUT_PILOT.num_steps,))
