from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from openpi.mot_jepa.layout import LAYOUT_BASE
from openpi.mot_jepa.layout import LAYOUT_BASE_STAGES
from openpi.mot_jepa.layout import LAYOUT_PILOT
from openpi.mot_jepa.layout import STREAM_ORDER
from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.layout import TokenLayout


def test_base_layout_matches_design_document_arithmetic():
    # The design document's headline token budget: 1568 video + 784 gel + 96 lowdim.
    assert LAYOUT_BASE.num_steps == 8
    assert LAYOUT_BASE.num_video_tokens == 1568
    assert LAYOUT_BASE.num_gel_tokens == 784
    assert LAYOUT_BASE.num_lowdim_tokens == 96
    assert LAYOUT_BASE.num_tokens == 2448


def test_progressive_resolution_stages_match_design_document():
    assert [layout.num_video_tokens for layout in LAYOUT_BASE_STAGES] == [800, 1152, 1568]
    # Gel is fixed across the schedule; only video grows.
    assert {layout.num_gel_tokens for layout in LAYOUT_BASE_STAGES} == {784}


def test_stream_slices_are_contiguous_and_cover_every_token():
    offset = 0
    for stream in STREAM_ORDER:
        sl = LAYOUT_BASE.stream_slices[stream]
        assert sl.start == offset
        offset = sl.stop
    assert offset == LAYOUT_BASE.num_tokens


def test_expert_slices_are_contiguous_because_tactile_streams_are_adjacent():
    # This contiguity is the precondition for slice-wise modality-local attention.
    video = LAYOUT_BASE.expert_slices[ExpertId.VIDEO]
    tactile = LAYOUT_BASE.expert_slices[ExpertId.TACTILE]
    assert video.start == 0
    assert video.stop == tactile.start
    assert tactile.stop == LAYOUT_BASE.num_tokens

    expert_ids = LAYOUT_BASE.expert_ids
    assert torch.equal(expert_ids[video], torch.full((video.stop - video.start,), int(ExpertId.VIDEO)))
    assert torch.equal(expert_ids[tactile], torch.full((tactile.stop - tactile.start,), int(ExpertId.TACTILE)))


def test_time_axis_is_shared_across_streams():
    """Video, gel and lowdim tokens at the same instant carry the same ``t`` coordinate.

    This is the entire mechanism behind cross-modal attention at one instant having zero
    relative-time phase. If it regresses, the synchrony objective becomes solvable by
    reading a positional code instead of the content.
    """
    coords = LAYOUT_BASE.coords
    for step in range(LAYOUT_BASE.num_steps):
        for stream in STREAM_ORDER:
            sl = LAYOUT_BASE.stream_slices[stream]
            idx = LAYOUT_BASE.step_token_indices(stream, np.array([step]))
            assert idx.min() >= sl.start
            assert idx.max() < sl.stop
            assert torch.all(coords[idx, 0] == step), f"{stream.name} step {step}"


def test_lowdim_tokens_have_identity_spatial_coordinates():
    coords = LAYOUT_BASE.coords[LAYOUT_BASE.stream_slices[StreamId.LOWDIM]]
    assert torch.all(coords[:, 1] == 0)
    assert torch.all(coords[:, 2] == 0)


def test_gel_pads_share_positions_and_are_separated_by_area_embedding_instead():
    # Two pads observing the same instant and patch are positionally identical on purpose:
    # pad identity is carried by an additive area embedding, not by RoPE.
    layout = LAYOUT_BASE
    gel = layout.coords[layout.stream_slices[StreamId.GEL]]
    per_pad = layout.gel_grid[0] * layout.gel_grid[1]
    pad0 = gel[:per_pad]
    pad1 = gel[per_pad : 2 * per_pad]
    assert torch.equal(pad0, pad1)
    assert torch.equal(layout.gel_pad_ids[:per_pad], torch.zeros(per_pad, dtype=torch.int64))
    assert torch.equal(layout.gel_pad_ids[per_pad : 2 * per_pad], torch.ones(per_pad, dtype=torch.int64))


def test_step_token_indices_selects_exactly_one_instant():
    layout = LAYOUT_BASE
    for stream, per_step in [
        (StreamId.VIDEO, layout.video_tokens_per_step),
        (StreamId.GEL, layout.gel_tokens_per_step),
        (StreamId.LOWDIM, layout.lowdim_slots),
    ]:
        idx = layout.step_token_indices(stream, np.array([3, 4]))
        assert idx.size == 2 * per_step
        assert np.unique(idx).size == idx.size


def test_coords_and_ids_have_consistent_lengths():
    for layout in (LAYOUT_BASE, LAYOUT_PILOT):
        assert layout.coords.shape == (layout.num_tokens, 3)
        assert layout.expert_ids.shape == (layout.num_tokens,)
        assert layout.stream_ids.shape == (layout.num_tokens,)
        assert layout.step_ids.shape == (layout.num_tokens,)
        assert layout.gel_pad_ids.shape == (layout.num_gel_tokens,)


def test_pilot_layout_is_substantially_cheaper():
    assert LAYOUT_PILOT.num_tokens < LAYOUT_BASE.num_tokens / 3


def test_with_video_size_only_changes_video():
    resized = LAYOUT_BASE.with_video_size(160)
    assert resized.num_video_tokens == 800
    assert resized.num_gel_tokens == LAYOUT_BASE.num_gel_tokens
    assert resized.num_lowdim_tokens == LAYOUT_BASE.num_lowdim_tokens


def test_layout_is_frozen_and_hashable():
    assert dataclasses.is_dataclass(LAYOUT_BASE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        LAYOUT_BASE.num_frames = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_frames": 15},  # not divisible by tubelet_t
        {"video_size": 100},  # not divisible by video_patch
        {"gel_size": 100},  # not divisible by gel_patch
        {"num_gel_pads": 0},
    ],
)
def test_invalid_layouts_raise(kwargs):
    with pytest.raises(ValueError, match=r"divisible|must both be"):
        TokenLayout(**kwargs)
