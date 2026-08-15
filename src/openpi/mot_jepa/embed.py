"""Patch embedding for the three input streams.

Every stream is embedded at *full* length and the mask is applied afterwards by gathering.
Embedding only the visible patches would save a little compute, but the patch-embed convs
are a low-single-digit fraction of total FLOPs next to sixteen transformer layers, and the
teacher needs the full token set anyway. Paying it twice buys a much simpler contract:
token index ``i`` always means the same thing to the loader, the model and the probes.

Gel pads and low-dimensional sensors carry **additive** identity embeddings rather than
positional ones. Two pads observing the same instant sit at identical RoPE coordinates on
purpose -- position encodes *when and where*, the additive embedding encodes *which sensor*.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from openpi.mot_jepa.layout import TokenLayout


class VideoPatchEmbed(nn.Module):
    """``(B, T, 3, H, W)`` uint8-or-float video into ``(B, num_video_tokens, video_width)``."""

    def __init__(self, layout: TokenLayout, *, in_channels: int = 3) -> None:
        super().__init__()
        self.layout = layout
        kernel = (layout.tubelet_t, layout.video_patch, layout.video_patch)
        self.proj = nn.Conv3d(in_channels, layout.video_width, kernel_size=kernel, stride=kernel)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames = video.shape[0], video.shape[1]
        if frames != self.layout.num_frames:
            raise ValueError(f"expected {self.layout.num_frames} frames, got {frames}")
        # (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d.
        tokens = self.proj(video.permute(0, 2, 1, 3, 4))
        # (B, D, T', Hp, Wp) -> (B, T'*Hp*Wp, D), matching the layout's (t, h, w) row-major order.
        return tokens.flatten(2).transpose(1, 2).reshape(batch, self.layout.num_video_tokens, -1)


class GelPatchEmbed(nn.Module):
    """``(B, T, N, 3, H, W)`` gel frames into ``(B, num_gel_tokens, tactile_width)``.

    A single shared conv is applied to every pad; pads are then distinguished by an additive
    pad embedding and a sensor embedding. Sharing the conv is deliberate -- a GelSight and an
    MCTac pad observe the same physics and should not need separate feature extractors to be
    comparable in the synchrony objective.
    """

    def __init__(self, layout: TokenLayout, *, in_channels: int = 3, num_sensor_types: int = 16) -> None:
        super().__init__()
        self.layout = layout
        kernel = (layout.tubelet_t, layout.gel_patch, layout.gel_patch)
        self.proj = nn.Conv3d(in_channels, layout.tactile_width, kernel_size=kernel, stride=kernel)
        self.pad_embed = nn.Embedding(layout.num_gel_pads, layout.tactile_width)
        self.sensor_embed = nn.Embedding(num_sensor_types, layout.tactile_width)
        nn.init.zeros_(self.pad_embed.weight)
        nn.init.zeros_(self.sensor_embed.weight)

    def forward(self, gel: torch.Tensor, sensor_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, num_pads = gel.shape[0], gel.shape[1], gel.shape[2]
        if frames != self.layout.num_frames:
            raise ValueError(f"expected {self.layout.num_frames} frames, got {frames}")
        if num_pads != self.layout.num_gel_pads:
            raise ValueError(f"expected {self.layout.num_gel_pads} pads, got {num_pads}")

        # Fold pads into the batch so one conv serves all of them.
        merged = gel.permute(0, 2, 3, 1, 4, 5).reshape(batch * num_pads, gel.shape[3], frames, *gel.shape[4:])
        tokens = self.proj(merged)  # (B*N, D, T', hp, wp)
        width = tokens.shape[1]
        per_pad = self.layout.gel_grid[0] * self.layout.gel_grid[1]
        tokens = tokens.flatten(2).transpose(1, 2)  # (B*N, T'*hp*wp, D)
        tokens = tokens.reshape(batch, num_pads, self.layout.num_steps, per_pad, width)
        # Layout order is (t, pad, h, w), so time must lead pad.
        tokens = tokens.permute(0, 2, 1, 3, 4).reshape(batch, self.layout.num_gel_tokens, width)

        pad_ids = self.layout.gel_pad_ids.to(tokens.device)
        tokens = tokens + self.pad_embed(pad_ids)[None]
        if sensor_ids is not None:
            # (B, N) -> per-token, following the same (t, pad, h, w) ordering.
            per_token = sensor_ids[:, None, :, None].expand(batch, self.layout.num_steps, num_pads, per_pad)
            tokens = tokens + self.sensor_embed(per_token.reshape(batch, -1))
        return tokens


class LowDimEmbed(nn.Module):
    """``(B, T, S, C)`` low-dimensional tactile into ``(B, num_lowdim_tokens, tactile_width)``.

    One token per (tubelet step, sensor slot). Values are lifted through random Fourier
    features before the linear map: a 1-D force scalar has almost no linear structure to
    read, and the same trick is used by the FTP-1 state encoder
    (``ftp1_blocks.py:FourierStateEncoder``).
    """

    def __init__(
        self,
        layout: TokenLayout,
        *,
        in_channels: int = 1,
        fourier_dim: int = 16,
        num_sensor_types: int = 16,
        log_compress: bool = True,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.in_channels = in_channels
        # Compress the raw value before the Fourier lift. Parameter-free and stateless, so it
        # cannot drift between student and teacher, and it maps 0 -> 0 exactly -- which matters
        # because ~90% of the corpus zero-fills this stream and must stay inert.
        #
        # Without it the RAW value is concatenated as feature 0 below and handed to `proj`, while
        # every other feature is a sin/cos in [-1, 1] and both image streams arrive in [-1, 1].
        # Measured over the corpus this stream reaches |x| = 51,287, with |x| > 1000 in 2.7% of
        # clips -- so at a global batch of 512 it is in essentially every batch, a chronic source
        # of large activations rather than a rare spike. It also breaks the lift itself:
        # `scaled` reaches 51,287 * 2^15 = 1.7e9, where sin/cos under bf16 autocast carry no
        # information at all. `backbone.embed.lowdim.proj.bias` was among the fastest-growing
        # parameters in the run that diverged (see MoTEncoderConfig.qk_norm).
        self.log_compress = log_compress
        # log1p(1e4) = 9.21, so 51,287 -> 1.18 and 1000 -> 0.75: back inside the range the
        # Fourier frequencies were chosen for.
        self.log_reference = math.log1p(1e4)
        # Fixed, not learned, so the feature basis cannot drift between student and teacher.
        self.register_buffer(
            "fourier_freqs",
            2.0 ** torch.arange(fourier_dim, dtype=torch.float32),
            persistent=True,
        )
        feature_dim = layout.tubelet_t * in_channels * (1 + 2 * fourier_dim)
        self.proj = nn.Linear(feature_dim, layout.tactile_width)
        self.slot_embed = nn.Embedding(layout.lowdim_slots, layout.tactile_width)
        self.sensor_embed = nn.Embedding(num_sensor_types, layout.tactile_width)
        nn.init.zeros_(self.slot_embed.weight)
        nn.init.zeros_(self.sensor_embed.weight)

    def forward(self, lowdim: torch.Tensor, sensor_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, slots = lowdim.shape[0], lowdim.shape[1], lowdim.shape[2]
        if frames != self.layout.num_frames:
            raise ValueError(f"expected {self.layout.num_frames} frames, got {frames}")
        if slots != self.layout.lowdim_slots:
            raise ValueError(f"expected {self.layout.lowdim_slots} slots, got {slots}")

        if self.log_compress:
            lowdim = torch.sign(lowdim) * torch.log1p(lowdim.abs()) / self.log_reference

        scaled = lowdim[..., None] * self.fourier_freqs
        features = torch.cat([lowdim[..., None], torch.sin(scaled), torch.cos(scaled)], dim=-1)
        # (B, T, S, C, F) -> (B, T', S, tubelet*C*F): a tubelet's frames share one token.
        features = features.reshape(batch, self.layout.num_steps, self.layout.tubelet_t, slots, -1)
        features = features.permute(0, 1, 3, 2, 4).reshape(batch, self.layout.num_steps, slots, -1)
        tokens = self.proj(features).reshape(batch, self.layout.num_lowdim_tokens, -1)

        slot_ids = torch.arange(slots, device=tokens.device).repeat(self.layout.num_steps)
        tokens = tokens + self.slot_embed(slot_ids)[None]
        if sensor_ids is not None:
            per_token = sensor_ids[:, None, :].expand(batch, self.layout.num_steps, slots)
            tokens = tokens + self.sensor_embed(per_token.reshape(batch, -1))
        return tokens


class StreamEmbed(nn.Module):
    """Embeds all three streams and returns them keyed by expert.

    Returns ``(video_tokens, tactile_tokens)`` where the tactile tensor is the gel tokens
    followed by the lowdim tokens -- the same order the layout uses, so a global token index
    minus the video count indexes straight into it.
    """

    def __init__(
        self,
        layout: TokenLayout,
        *,
        video_channels: int = 3,
        gel_channels: int = 3,
        lowdim_channels: int = 1,
        num_sensor_types: int = 16,
        lowdim_log_compress: bool = True,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.video = VideoPatchEmbed(layout, in_channels=video_channels)
        self.gel = GelPatchEmbed(layout, in_channels=gel_channels, num_sensor_types=num_sensor_types)
        self.lowdim = LowDimEmbed(
            layout,
            in_channels=lowdim_channels,
            num_sensor_types=num_sensor_types,
            log_compress=lowdim_log_compress,
        )

    def forward(
        self,
        video: torch.Tensor,
        gel: torch.Tensor,
        lowdim: torch.Tensor,
        *,
        gel_sensor_ids: torch.Tensor | None = None,
        lowdim_sensor_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_tokens = self.video(video)
        tactile_tokens = torch.cat(
            [self.gel(gel, gel_sensor_ids), self.lowdim(lowdim, lowdim_sensor_ids)],
            dim=1,
        )
        return video_tokens, tactile_tokens
