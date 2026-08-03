"""Token layout contract shared by the MoT-JEPA data loader and model.

This module is deliberately dependency-light (numpy + torch only) and contains no
randomness. It is the single source of truth for how many tokens exist, which expert
owns each token, and what ``(t, h, w)`` coordinate each token carries.

The loader builds masks from a ``TokenLayout`` and the encoder builds RoPE tables from
the *same* ``TokenLayout``. If these two ever disagree, the model trains on
systematically mismatched targets without raising -- so every consumer must derive its
shapes from here rather than recomputing them.

Token order is always ``[video | gel | lowdim]`` and each stream is contiguous. That
contiguity is load-bearing: it is what lets modality-local attention run as two plain
SDPA calls over slices instead of materializing a ``(B, 1, L, L)`` mask.
"""

from __future__ import annotations

import dataclasses
import enum
import functools

import numpy as np
import torch


class ExpertId(enum.IntEnum):
    """Which set of per-modality weights (LN / qkv / out / FFN) processes a token."""

    VIDEO = 0
    TACTILE = 1


class StreamId(enum.IntEnum):
    """Which input stream a token came from.

    Distinct from :class:`ExpertId` on purpose: gel and low-dimensional tactile share the
    tactile expert's *weights* (so modality-local attention stays two SDPA calls) but need
    separate identities because the loss weights them differently (``L_tac`` vs
    ``L_lowdim``) and the masking modes address them separately.
    """

    VIDEO = 0
    GEL = 1
    LOWDIM = 2


STREAM_TO_EXPERT: dict[StreamId, ExpertId] = {
    StreamId.VIDEO: ExpertId.VIDEO,
    StreamId.GEL: ExpertId.TACTILE,
    StreamId.LOWDIM: ExpertId.TACTILE,
}

#: Streams in canonical token order. Do not reorder; masks and RoPE tables assume it.
STREAM_ORDER: tuple[StreamId, ...] = (StreamId.VIDEO, StreamId.GEL, StreamId.LOWDIM)


@dataclasses.dataclass(frozen=True)
class TokenLayout:
    """Immutable description of one clip's token grid.

    Args:
        num_frames: Frames per clip before temporal patching (``T``).
        tubelet_t: Temporal patch size. ``num_frames`` must be divisible by it.
        video_size: Square video input resolution in pixels.
        video_patch: Video spatial patch size in pixels.
        gel_size: Square gel (image-tactile) input resolution in pixels.
        gel_patch: Gel spatial patch size in pixels.
        num_gel_pads: Number of image-tactile pads (``N``), padded to a fixed maximum.
        lowdim_slots: Low-dimensional tactile tokens emitted per temporal step.
        video_width: Residual width of the video expert.
        tactile_width: Residual width of the tactile expert.
    """

    num_frames: int = 16
    tubelet_t: int = 2
    video_size: int = 224
    video_patch: int = 16
    gel_size: int = 112
    gel_patch: int = 16
    num_gel_pads: int = 2
    lowdim_slots: int = 12
    video_width: int = 768
    tactile_width: int = 384

    def __post_init__(self) -> None:
        if self.num_frames % self.tubelet_t != 0:
            raise ValueError(f"num_frames={self.num_frames} not divisible by tubelet_t={self.tubelet_t}")
        if self.video_size % self.video_patch != 0:
            raise ValueError(f"video_size={self.video_size} not divisible by video_patch={self.video_patch}")
        if self.gel_size % self.gel_patch != 0:
            raise ValueError(f"gel_size={self.gel_size} not divisible by gel_patch={self.gel_patch}")
        if self.num_gel_pads < 1 or self.lowdim_slots < 1:
            raise ValueError("num_gel_pads and lowdim_slots must both be >= 1")

    # -- grid geometry -------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        """Temporal token steps after tubelet patching (the shared time axis length)."""
        return self.num_frames // self.tubelet_t

    @property
    def video_grid(self) -> tuple[int, int]:
        side = self.video_size // self.video_patch
        return side, side

    @property
    def gel_grid(self) -> tuple[int, int]:
        side = self.gel_size // self.gel_patch
        return side, side

    @property
    def video_tokens_per_step(self) -> int:
        h, w = self.video_grid
        return h * w

    @property
    def gel_tokens_per_step(self) -> int:
        h, w = self.gel_grid
        return self.num_gel_pads * h * w

    # -- token counts --------------------------------------------------------------

    @property
    def num_video_tokens(self) -> int:
        return self.num_steps * self.video_tokens_per_step

    @property
    def num_gel_tokens(self) -> int:
        return self.num_steps * self.gel_tokens_per_step

    @property
    def num_lowdim_tokens(self) -> int:
        return self.num_steps * self.lowdim_slots

    @property
    def num_tokens(self) -> int:
        return self.num_video_tokens + self.num_gel_tokens + self.num_lowdim_tokens

    def stream_counts(self) -> dict[StreamId, int]:
        return {
            StreamId.VIDEO: self.num_video_tokens,
            StreamId.GEL: self.num_gel_tokens,
            StreamId.LOWDIM: self.num_lowdim_tokens,
        }

    @functools.cached_property
    def stream_slices(self) -> dict[StreamId, slice]:
        """Half-open ``[start, end)`` token ranges, in canonical order."""
        slices: dict[StreamId, slice] = {}
        counts = self.stream_counts()
        offset = 0
        for stream in STREAM_ORDER:
            slices[stream] = slice(offset, offset + counts[stream])
            offset += counts[stream]
        return slices

    @functools.cached_property
    def expert_slices(self) -> dict[ExpertId, slice]:
        """Half-open token ranges per expert.

        Valid only because ``STREAM_ORDER`` groups both tactile streams adjacently.
        """
        video = self.stream_slices[StreamId.VIDEO]
        gel = self.stream_slices[StreamId.GEL]
        lowdim = self.stream_slices[StreamId.LOWDIM]
        return {
            ExpertId.VIDEO: slice(video.start, video.stop),
            ExpertId.TACTILE: slice(gel.start, lowdim.stop),
        }

    def expert_width(self, expert: ExpertId) -> int:
        return self.video_width if expert is ExpertId.VIDEO else self.tactile_width

    # -- per-token metadata --------------------------------------------------------

    @functools.cached_property
    def stream_ids(self) -> torch.Tensor:
        """``(L,) int64`` stream id per token."""
        out = torch.empty(self.num_tokens, dtype=torch.int64)
        for stream, sl in self.stream_slices.items():
            out[sl] = int(stream)
        return out

    @functools.cached_property
    def expert_ids(self) -> torch.Tensor:
        """``(L,) int64`` expert id per token."""
        out = torch.empty(self.num_tokens, dtype=torch.int64)
        for expert, sl in self.expert_slices.items():
            out[sl] = int(expert)
        return out

    @functools.cached_property
    def coords(self) -> torch.Tensor:
        """``(L, 3) int64`` RoPE coordinates ``(t, h, w)``.

        ``t`` is the tubelet step index and is the *same* axis for every stream -- that
        shared time axis is the whole mechanism by which cross-modal attention at one
        instant carries zero relative-time phase. ``h``/``w`` are per-stream patch grids;
        low-dimensional tokens sit at ``h = w = 0`` so their rotation is purely temporal.

        The gel pad index is deliberately *not* a coordinate: pads are distinguished by an
        additive area embedding, so two pads observing the same instant and the same patch
        location are positionally identical, as the design requires.
        """
        out = torch.zeros(self.num_tokens, 3, dtype=torch.int64)

        vh, vw = self.video_grid
        t_idx, h_idx, w_idx = torch.meshgrid(
            torch.arange(self.num_steps),
            torch.arange(vh),
            torch.arange(vw),
            indexing="ij",
        )
        out[self.stream_slices[StreamId.VIDEO]] = torch.stack(
            [t_idx.reshape(-1), h_idx.reshape(-1), w_idx.reshape(-1)], dim=-1
        )

        gh, gw = self.gel_grid
        t_idx, _, h_idx, w_idx = torch.meshgrid(
            torch.arange(self.num_steps),
            torch.arange(self.num_gel_pads),
            torch.arange(gh),
            torch.arange(gw),
            indexing="ij",
        )
        out[self.stream_slices[StreamId.GEL]] = torch.stack(
            [t_idx.reshape(-1), h_idx.reshape(-1), w_idx.reshape(-1)], dim=-1
        )

        t_idx, _ = torch.meshgrid(
            torch.arange(self.num_steps),
            torch.arange(self.lowdim_slots),
            indexing="ij",
        )
        lowdim_coords = torch.zeros(self.num_lowdim_tokens, 3, dtype=torch.int64)
        lowdim_coords[:, 0] = t_idx.reshape(-1)
        out[self.stream_slices[StreamId.LOWDIM]] = lowdim_coords

        return out

    @functools.cached_property
    def step_ids(self) -> torch.Tensor:
        """``(L,) int64`` tubelet step per token. Equivalent to ``coords[:, 0]``."""
        return self.coords[:, 0].clone()

    @functools.cached_property
    def gel_pad_ids(self) -> torch.Tensor:
        """``(num_gel_tokens,) int64`` pad index for each gel token."""
        gh, gw = self.gel_grid
        pads = torch.arange(self.num_gel_pads).view(1, -1, 1, 1)
        pads = pads.expand(self.num_steps, self.num_gel_pads, gh, gw)
        return pads.reshape(-1).contiguous()

    # -- helpers -------------------------------------------------------------------

    def token_indices(self, stream: StreamId) -> np.ndarray:
        """``(n,) int64`` global token indices belonging to ``stream``."""
        sl = self.stream_slices[stream]
        return np.arange(sl.start, sl.stop, dtype=np.int64)

    def step_token_indices(self, stream: StreamId, steps: np.ndarray) -> np.ndarray:
        """Global token indices for ``stream`` restricted to the given tubelet ``steps``.

        Used by the temporal-window masking modes, which remove whole instants.
        """
        sl = self.stream_slices[stream]
        per_step = {
            StreamId.VIDEO: self.video_tokens_per_step,
            StreamId.GEL: self.gel_tokens_per_step,
            StreamId.LOWDIM: self.lowdim_slots,
        }[stream]
        steps = np.asarray(steps, dtype=np.int64).reshape(-1)
        offsets = sl.start + steps * per_step
        return (offsets[:, None] + np.arange(per_step, dtype=np.int64)[None, :]).reshape(-1)

    def with_video_size(self, video_size: int) -> TokenLayout:
        """Return a copy at a different video resolution (progressive-resolution schedule).

        Only legal because positions are RoPE rather than learned lookups, so changing the
        grid does not invalidate any parameter.
        """
        return dataclasses.replace(self, video_size=video_size)

    def summary(self) -> dict[str, int]:
        return {
            "num_frames": self.num_frames,
            "num_steps": self.num_steps,
            "video_tokens": self.num_video_tokens,
            "gel_tokens": self.num_gel_tokens,
            "lowdim_tokens": self.num_lowdim_tokens,
            "total_tokens": self.num_tokens,
        }


#: Full-resolution layout from the design doc: 1568 + 784 + 96 = 2448 tokens.
LAYOUT_BASE = TokenLayout()

#: Progressive-resolution stages for the base layout (gel stays fixed at 112).
LAYOUT_BASE_STAGES: tuple[TokenLayout, ...] = (
    LAYOUT_BASE.with_video_size(160),
    LAYOUT_BASE.with_video_size(192),
    LAYOUT_BASE.with_video_size(224),
)

#: Cheap ViT-S-class layout for the single-domain pilot that gates the full run.
LAYOUT_PILOT = TokenLayout(
    video_size=112,
    gel_size=64,
    lowdim_slots=8,
    video_width=384,
    tactile_width=192,
)
