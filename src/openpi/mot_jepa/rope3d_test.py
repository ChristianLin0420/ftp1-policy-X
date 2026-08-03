from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.mot_jepa.layout import LAYOUT_PILOT
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.rope3d import Rope3D
from openpi.mot_jepa.rope3d import Rope3DConfig
from openpi.mot_jepa.rope3d import apply_rope
from openpi.mot_jepa.rope3d import build_rope_tables
from openpi.mot_jepa.rope3d import relative_phase

CONFIG = Rope3DConfig()
LAYOUT = LAYOUT_PILOT


@pytest.fixture(name="rope")
def rope_fixture() -> Rope3D:
    return Rope3D(CONFIG, LAYOUT)


def _first_token_of_stream_at_step(stream: StreamId, step: int) -> int:
    return int(LAYOUT.step_token_indices(stream, np.array([step]))[0])


def test_time_slice_phase_is_identical_across_streams_at_the_same_instant(rope):
    """The load-bearing property: same instant, different modality, zero relative time phase.

    If this regresses, tokens carry a modality-specific temporal code and the synchrony
    objective becomes solvable by reading position rather than content -- which is exactly
    the ``shuffle == real`` failure mode.
    """
    cos, sin = rope.cos_table, rope.sin_table
    time_pairs = CONFIG.dim_t // 2
    for step in range(LAYOUT.num_steps):
        video = _first_token_of_stream_at_step(StreamId.VIDEO, step)
        gel = _first_token_of_stream_at_step(StreamId.GEL, step)
        lowdim = _first_token_of_stream_at_step(StreamId.LOWDIM, step)
        for other in (gel, lowdim):
            phase = relative_phase(cos, sin, video, other)[:time_pairs]
            torch.testing.assert_close(phase, torch.zeros_like(phase), atol=1e-6, rtol=0)


def test_time_slice_phase_differs_across_instants(rope):
    cos, sin = rope.cos_table, rope.sin_table
    time_pairs = CONFIG.dim_t // 2
    a = _first_token_of_stream_at_step(StreamId.VIDEO, 0)
    b = _first_token_of_stream_at_step(StreamId.VIDEO, 1)
    phase = relative_phase(cos, sin, a, b)[:time_pairs]
    assert float(phase.abs().max()) > 1e-3


def test_lowdim_tokens_have_no_spatial_rotation(rope):
    """Lowdim sits at h = w = 0, so its spatial slots are the identity rotation."""
    sl = LAYOUT.stream_slices[StreamId.LOWDIM]
    spatial = slice(CONFIG.dim_t // 2, CONFIG.head_dim // 2)
    cos = rope.cos_table[sl, spatial]
    sin = rope.sin_table[sl, spatial]
    torch.testing.assert_close(cos, torch.ones_like(cos))
    torch.testing.assert_close(sin, torch.zeros_like(sin))


def test_gel_pads_receive_identical_rotations(rope):
    """Pads are separated by an additive area embedding, never by position."""
    per_pad = LAYOUT.gel_grid[0] * LAYOUT.gel_grid[1]
    start = LAYOUT.stream_slices[StreamId.GEL].start
    pad0 = rope.cos_table[start : start + per_pad]
    pad1 = rope.cos_table[start + per_pad : start + 2 * per_pad]
    torch.testing.assert_close(pad0, pad1)


def test_apply_rope_preserves_norm(rope):
    torch.manual_seed(0)
    index = torch.arange(LAYOUT.num_tokens)
    cos, sin = rope.gather(index)
    x = torch.randn(2, 12, LAYOUT.num_tokens, CONFIG.head_dim, dtype=torch.float32)
    rotated = apply_rope(x, cos, sin)
    # Rotation is orthogonal, so per-pair (hence per-head) norms are invariant.
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rope_makes_attention_logits_depend_only_on_relative_position(rope):
    """The defining RoPE property, checked on the time axis with spatial slots neutralized."""
    torch.manual_seed(0)
    head_dim = CONFIG.head_dim
    # Two lowdim tokens (h = w = 0) separated by one step, versus another such pair.
    lowdim = LAYOUT.stream_slices[StreamId.LOWDIM]
    slots = LAYOUT.lowdim_slots
    a0 = lowdim.start
    a1 = lowdim.start + slots
    b0 = lowdim.start + 2 * slots
    b1 = lowdim.start + 3 * slots

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def logit(qi: int, kj: int) -> float:
        cq, sq = rope.gather(torch.tensor([qi]))
        ck, sk = rope.gather(torch.tensor([kj]))
        qr = apply_rope(q, cq, sq)
        kr = apply_rope(k, ck, sk)
        return float((qr * kr).sum())

    assert logit(a0, a1) == pytest.approx(logit(b0, b1), abs=1e-4)


def test_values_are_never_rotated_by_construction():
    """`apply_rope` is only ever meant for q/k; this documents the contract in a test.

    A non-identity rotation applied to the value path would inject a linearly readable
    positional code into the residual stream.
    """
    cfg = Rope3DConfig()
    cos, sin = build_rope_tables(cfg, torch.tensor([[3, 1, 2]], dtype=torch.int64))
    x = torch.randn(1, 1, 1, cfg.head_dim)
    assert not torch.allclose(apply_rope(x, cos, sin), x)


def test_tables_are_deterministic_and_independent_of_global_rng():
    torch.manual_seed(123)
    a = build_rope_tables(CONFIG, LAYOUT.coords)
    _ = torch.randn(1000)
    np.random.seed(7)
    b = build_rope_tables(CONFIG, LAYOUT.coords)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_buffers_are_not_persisted(rope):
    assert "cos_table" not in rope.state_dict()
    assert "sin_table" not in rope.state_dict()


def test_gather_supports_an_empty_selection(rope):
    """Mode T_HARD leaves the tactile expert with zero context tokens, by design."""
    empty = torch.zeros(4, 0, dtype=torch.int64)
    cos, sin = rope.gather(empty)
    assert cos.shape == (4, 0, CONFIG.head_dim // 2)
    assert sin.shape == (4, 0, CONFIG.head_dim // 2)
    x = torch.randn(4, 2, 0, CONFIG.head_dim)
    assert apply_rope(x, cos, sin).shape == x.shape


def test_gather_supports_batched_index(rope):
    index = torch.stack([torch.arange(10), torch.arange(10, 20)])
    cos, sin = rope.gather(index)
    assert cos.shape == (2, 10, CONFIG.head_dim // 2)
    x = torch.randn(2, 12, 10, CONFIG.head_dim)
    assert apply_rope(x, cos, sin).shape == x.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim_t": 30},  # slices no longer sum to head_dim
        {"dim_t": 31, "dim_h": 17, "dim_w": 16},  # odd slices
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError, match=r"must equal head_dim|must be even"):
        Rope3DConfig(**kwargs)


def test_build_rope_tables_rejects_bad_coords():
    with pytest.raises(ValueError, match=r"coords must be"):
        build_rope_tables(CONFIG, torch.zeros(4, 2, dtype=torch.int64))
