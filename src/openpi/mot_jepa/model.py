"""Backbone and student assembly.

``MotJepaBackbone`` is embedding + encoder. It is the artifact that survives pretraining:
the EMA teacher shadows it, the probes read it, and the policy head freezes it.

``MotJepaStudent`` adds the predictor and is what DDP wraps -- one module, therefore one
reducer, rather than two independently-bucketed graphs.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from openpi.mot_jepa.embed import StreamEmbed
from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.masking import ClipMasks
from openpi.mot_jepa.mot_encoder import NUM_EXPERTS
from openpi.mot_jepa.mot_encoder import EncoderOutput
from openpi.mot_jepa.mot_encoder import MoTEncoder
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.mot_encoder import gather_tokens
from openpi.mot_jepa.mot_encoder import split_index_by_expert
from openpi.mot_jepa.predictor import MoTPredictor
from openpi.mot_jepa.predictor import MoTPredictorConfig
from openpi.mot_jepa.predictor import target_index_by_expert


@dataclasses.dataclass
class ClipInputs:
    """Raw clip tensors, already on device and float-normalized."""

    video: torch.Tensor  # (B, T, 3, H, W)
    gel: torch.Tensor  # (B, T, N, 3, h, w)
    lowdim: torch.Tensor  # (B, T, S, C)
    gel_sensor_ids: torch.Tensor | None = None  # (B, N)
    lowdim_sensor_ids: torch.Tensor | None = None  # (B, S)


@dataclasses.dataclass
class StudentOutput:
    """Predictions in teacher space plus the student's own unimodal synchrony readout."""

    predictions: list[torch.Tensor]  # per expert, (B, L_tgt_e, width_e)
    sync_readout: list[torch.Tensor]  # per expert, (B, num_steps, width_e)
    context: list[torch.Tensor]  # per expert, (B, L_ctx_e, width_e)


class MotJepaBackbone(nn.Module):
    """Embedding + MoT encoder. The deployable half of the model."""

    def __init__(
        self,
        layout: TokenLayout,
        encoder_config: MoTEncoderConfig,
        *,
        lowdim_channels: int = 1,
        num_sensor_types: int = 16,
        lowdim_log_compress: bool = True,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.embed = StreamEmbed(
            layout,
            lowdim_channels=lowdim_channels,
            num_sensor_types=num_sensor_types,
            lowdim_log_compress=lowdim_log_compress,
        )
        self.encoder = MoTEncoder(encoder_config, layout)

    def set_gradient_checkpointing(self, *, enabled: bool) -> None:
        self.encoder.set_gradient_checkpointing(enabled=enabled)

    def embed_all(self, inputs: ClipInputs) -> list[torch.Tensor]:
        """Full-length per-expert token tensors, before any masking."""
        video, tactile = self.embed(
            inputs.video,
            inputs.gel,
            inputs.lowdim,
            gel_sensor_ids=inputs.gel_sensor_ids,
            lowdim_sensor_ids=inputs.lowdim_sensor_ids,
        )
        return [video, tactile]

    def forward(self, inputs: ClipInputs, index: torch.Tensor, expert_bounds: torch.Tensor) -> EncoderOutput:
        """Encode the tokens selected by ``index``.

        Args:
            inputs: Raw clip tensors.
            index: ``(B, L_sel)`` global token indices, ascending.
            expert_bounds: ``(num_experts, 2)`` column ranges of ``index`` per expert.
        """
        full = self.embed_all(inputs)
        global_index, local_index = split_index_by_expert(self.layout, index, expert_bounds)
        selected = [gather_tokens(full[e], local_index[e]) for e in range(NUM_EXPERTS)]
        return self.encoder(selected, global_index)

    def encode_full(self, inputs: ClipInputs) -> EncoderOutput:
        """Encode every token. Used by the teacher and by the probes."""
        batch = inputs.video.shape[0]
        device = inputs.video.device
        video_index = torch.arange(self.layout.num_video_tokens, device=device).expand(batch, -1)
        tactile = self.layout.expert_slices[ExpertId.TACTILE]
        tactile_index = torch.arange(tactile.start, tactile.stop, device=device).expand(batch, -1)
        return self.encoder(self.embed_all(inputs), [video_index, tactile_index])


class MotJepaStudent(nn.Module):
    """Backbone + predictor, wrapped as one module so DDP builds a single reducer."""

    def __init__(
        self,
        layout: TokenLayout,
        encoder_config: MoTEncoderConfig,
        predictor_config: MoTPredictorConfig,
        *,
        lowdim_channels: int = 1,
        num_sensor_types: int = 16,
        lowdim_log_compress: bool = True,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.backbone = MotJepaBackbone(
            layout, encoder_config, lowdim_channels=lowdim_channels, num_sensor_types=num_sensor_types
        )
        self.predictor = MoTPredictor(predictor_config, layout)

    def set_gradient_checkpointing(self, *, enabled: bool) -> None:
        self.backbone.set_gradient_checkpointing(enabled=enabled)
        self.predictor.set_gradient_checkpointing(enabled=enabled)

    def forward(self, inputs: ClipInputs, masks: ClipMasks) -> StudentOutput:
        encoded = self.backbone(inputs, masks.ctx_index, masks.ctx_expert_bounds)
        context_index, _ = split_index_by_expert(self.layout, masks.ctx_index, masks.ctx_expert_bounds)
        target_index = target_index_by_expert(self.layout, masks.tgt_index, masks.tgt_expert_bounds)
        predictions = self.predictor(encoded.tokens, context_index, target_index, masks.mode)
        return StudentOutput(
            predictions=predictions,
            sync_readout=encoded.sync_readout,
            context=encoded.tokens,
        )
