"""State-conditioned deterministic controller over dense MoT-JEPA tokens.

The first MoT-JEPA policy read a per-step mean pool and predicted first differences without
observing robot state.  That contract admits an open-loop average trajectory.  ``ControlV2`` is
the deliberately small replacement used by the lift-bottle policy:

* cross-attention reads every video and GEL token (never the zero-filled low-dimensional stream),
* Fourier state tokens carry normalized qpos, velocity, command, and presence features,
* learned horizon queries deterministically emit a 32-step mixed action chunk, and
* auxiliary phase/contact/order heads make the transition signals visible during training.

The mixed action convention is load-bearing.  Dimensions 0:7 are offsets from the current arm
qpos and dimension 7 is an absolute gripper width.  Row zero is a non-learned current-state
placeholder: zero arm offsets and the current gripper width.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.mot_encoder import EncoderOutput

ACTIVE_ACTION_DIM = 8
ARM_DIM = 7
STATE_FEATURE_DIM = 3 * ACTIVE_ACTION_DIM + 3
NUM_PHASES = 4
NUM_ORDER_CLASSES = 4


@dataclasses.dataclass(frozen=True)
class ControlV2Config:
    """Architecture and loss contract for the task-specific V2 controller.

    The defaults are the production lift-bottle shape.  Width and depth remain configurable so
    the exact contract can be exercised cheaply in unit and one-step DDP tests.
    """

    width: int = 512
    depth: int = 8
    num_heads: int = 8
    horizon: int = 32
    qpos_history: int = 16
    mlp_ratio: float = 4.0
    fourier_dim: int = 8
    fourier_min_period: float = 1e-3
    fourier_max_period: float = 1.0
    smooth_l1_beta: float = 1e-2
    horizon_decay: float = 0.05
    gripper_loss_weight: float = 2.0
    phase_loss_weight: float = 0.25
    contact_loss_weight: float = 0.1
    order_loss_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.width <= 0 or self.depth <= 0 or self.num_heads <= 0:
            raise ValueError("width, depth, and num_heads must be positive")
        if self.width % self.num_heads:
            raise ValueError(f"width {self.width} must be divisible by num_heads {self.num_heads}")
        if self.horizon < 2:
            raise ValueError("horizon must include row zero and at least one executable row")
        if self.qpos_history < 1:
            raise ValueError("qpos_history must be positive")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.fourier_dim <= 0 or self.fourier_dim % 2:
            raise ValueError("fourier_dim must be a positive even number")
        if not 0 < self.fourier_min_period <= self.fourier_max_period:
            raise ValueError("Fourier periods must satisfy 0 < min_period <= max_period")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if self.horizon_decay < 0:
            raise ValueError("horizon_decay must be non-negative")
        if (
            min(
                self.gripper_loss_weight,
                self.phase_loss_weight,
                self.contact_loss_weight,
                self.order_loss_weight,
            )
            < 0
        ):
            raise ValueError("loss weights must be non-negative")


@dataclasses.dataclass(frozen=True)
class ControlV2Output:
    """Deterministic policy result and supervised auxiliary predictions."""

    action_chunk: torch.Tensor
    """``(B, H, 8)`` mixed target: arm offsets followed by absolute gripper width."""

    absolute_qpos_chunk: torch.Tensor
    """``(B, H, 8)`` absolute commands, ready for temporal ensembling/execution."""

    residual_chunk: torch.Tensor
    """``(B, H, 8)`` learned residuals; row zero is exactly zero."""

    normalized_residual_chunk: torch.Tensor
    """``(B, H, 8)`` residuals before multiplication by train-set action scales."""

    action_scale: torch.Tensor
    """Checkpointed positive ``(8,)`` scale used to reconstruct physical commands."""

    phase_logits: torch.Tensor
    contact_logits: torch.Tensor
    order_logits: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ControlV2Loss:
    """Differentiable loss components without device-synchronising metric conversion."""

    total: torch.Tensor
    chunk: torch.Tensor
    phase: torch.Tensor
    contact: torch.Tensor
    order: torch.Tensor


class FourierStateEncoder(nn.Module):
    """Fourier-feature MLP for normalized, continuous state sequences."""

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        *,
        fourier_dim: int,
        min_period: float,
        max_period: float,
    ) -> None:
        super().__init__()
        half = fourier_dim // 2
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), half)
        self.register_buffer("frequencies", (2 * math.pi / periods).float())
        input_dim = state_dim * (1 + fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * output_dim),
            nn.GELU(),
            nn.Linear(2 * output_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        angles = state.float().unsqueeze(-1) * self.frequencies
        features = torch.cat((state.float().unsqueeze(-1), angles.sin(), angles.cos()), dim=-1)
        features = features.flatten(start_dim=-2).to(self.mlp[0].weight.dtype)
        return self.mlp(features)


class _DecoderBlock(nn.Module):
    """Pre-norm query self-attention, memory cross-attention, and GELU MLP."""

    def __init__(self, width: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, num_heads, batch_first=True, dropout=0.0)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, num_heads, batch_first=True, dropout=0.0)
        self.mlp_norm = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        queries = queries + self.cross_attention(self.cross_norm(queries), memory, memory, need_weights=False)[0]
        return queries + self.mlp(self.mlp_norm(queries))


class ControlV2(nn.Module):
    """Dense-token, state-conditioned query decoder for one eight-dimensional task."""

    def __init__(self, config: ControlV2Config, layout: TokenLayout) -> None:
        super().__init__()
        self.config = config
        self.layout = layout
        width = config.width

        self.context_projection = nn.ModuleList(
            (nn.Linear(layout.video_width, width), nn.Linear(layout.tactile_width, width))
        )
        # Video, GEL, and robot state retain distinct identities after entering the shared space.
        self.memory_type_embedding = nn.Embedding(3, width)
        self.context_time_embedding = nn.Embedding(layout.num_steps, width)
        self.state_time_embedding = nn.Embedding(config.qpos_history, width)
        max_spatial = max(*layout.video_grid, *layout.gel_grid)
        self.row_embedding = nn.Embedding(max_spatial, width)
        self.column_embedding = nn.Embedding(max_spatial, width)

        video_slice = layout.stream_slices[StreamId.VIDEO]
        gel_slice = layout.stream_slices[StreamId.GEL]
        self.register_buffer("video_coordinates", layout.coords[video_slice].clone(), persistent=False)
        self.register_buffer("gel_coordinates", layout.coords[gel_slice].clone(), persistent=False)

        # The data contract supplies normalized qpos/velocity/command plus three presence bits.
        self.state_encoder = FourierStateEncoder(
            STATE_FEATURE_DIM,
            width,
            fourier_dim=config.fourier_dim,
            min_period=config.fourier_min_period,
            max_period=config.fourier_max_period,
        )
        self.memory_norm = nn.LayerNorm(width)

        self.query_embedding = nn.Parameter(torch.empty(1, config.horizon, width))
        nn.init.trunc_normal_(self.query_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            _DecoderBlock(width, config.num_heads, config.mlp_ratio) for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(width)
        self.action_output = nn.Linear(width, ACTIVE_ACTION_DIM)
        # Exact hold-current behavior at initialization.  No random action can reach deployment.
        nn.init.zeros_(self.action_output.weight)
        nn.init.zeros_(self.action_output.bias)

        self.auxiliary_norm = nn.LayerNorm(width)
        self.phase_output = nn.Linear(width, NUM_PHASES)
        self.contact_output = nn.Linear(width, 1)
        self.order_output = nn.Linear(width, NUM_ORDER_CLASSES)
        self.register_buffer("action_scale", torch.ones(ACTIVE_ACTION_DIM))

    def load_action_scale(self, scale: torch.Tensor) -> None:
        """Store positive train-split residual scales in the model checkpoint."""
        if scale.shape != (ACTIVE_ACTION_DIM,):
            raise ValueError(f"action_scale must have shape ({ACTIVE_ACTION_DIM},)")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("action_scale must be finite and strictly positive")
        with torch.no_grad():
            self.action_scale.copy_(scale.to(device=self.action_scale.device, dtype=self.action_scale.dtype))

    def _validate_inputs(
        self,
        encoded: EncoderOutput,
        state_features: torch.Tensor,
        current_qpos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(encoded.tokens) != 2:
            raise ValueError(f"expected two expert token tensors, got {len(encoded.tokens)}")
        video, tactile = encoded.tokens
        if video.ndim != 3 or tactile.ndim != 3:
            raise ValueError("encoder tokens must have shape (batch, tokens, width)")
        if video.shape[0] != tactile.shape[0]:
            raise ValueError("video and tactile token batches differ")
        if video.shape[1:] != (self.layout.num_video_tokens, self.layout.video_width):
            raise ValueError(
                "video tokens must be the dense full-clip output with shape "
                f"(B, {self.layout.num_video_tokens}, {self.layout.video_width})"
            )
        allowed_tactile_lengths = {
            self.layout.num_gel_tokens,
            self.layout.num_gel_tokens + self.layout.num_lowdim_tokens,
        }
        if tactile.shape[1] not in allowed_tactile_lengths or tactile.shape[2] != self.layout.tactile_width:
            raise ValueError(
                "tactile tokens must contain the dense GEL prefix, optionally followed by the full lowdim stream"
            )

        batch = video.shape[0]
        if state_features.ndim != 3 or state_features.shape[0] != batch or state_features.shape[2] != STATE_FEATURE_DIM:
            raise ValueError(f"state_features must have shape (B, T, {STATE_FEATURE_DIM})")
        if not 1 <= state_features.shape[1] <= self.config.qpos_history:
            raise ValueError(f"state feature history length must be in [1, {self.config.qpos_history}]")
        if current_qpos.shape != (batch, ACTIVE_ACTION_DIM):
            raise ValueError(f"current_qpos must have shape (B, {ACTIVE_ACTION_DIM})")
        if not state_features.is_floating_point() or not current_qpos.is_floating_point():
            raise TypeError("state_features and current_qpos must be floating-point tensors")
        if (
            state_features.device != video.device
            or current_qpos.device != video.device
            or tactile.device != video.device
        ):
            raise ValueError("encoder tokens and state inputs must be on the same device")
        if not torch.isfinite(state_features).all() or not torch.isfinite(current_qpos).all():
            raise ValueError("state inputs must be finite")

        presence = state_features[..., -3:]
        if not ((presence == 0) | (presence == 1)).all():
            raise ValueError("state presence features must be binary")
        history_valid = presence[..., 0]
        if not (history_valid[:, -1] == 1).all():
            raise ValueError("the current (last) state history entry must be present")
        if (presence[..., 1:] > history_valid[..., None]).any():
            raise ValueError("velocity/command cannot be present when the state history entry is absent")
        return video, tactile[:, : self.layout.num_gel_tokens]

    def _position(self, coordinates: torch.Tensor) -> torch.Tensor:
        coordinates = coordinates.to(self.context_time_embedding.weight.device)
        return (
            self.context_time_embedding(coordinates[:, 0])
            + self.row_embedding(coordinates[:, 1])
            + self.column_embedding(coordinates[:, 2])
        )

    def _state_tokens(self, state_features: torch.Tensor) -> torch.Tensor:
        tokens = self.state_encoder(state_features).to(self.memory_type_embedding.weight.dtype)
        # A short cold-start history is right-aligned with the same positions used by a full clip.
        start = self.config.qpos_history - state_features.shape[1]
        positions = torch.arange(start, self.config.qpos_history, device=state_features.device)
        return tokens + self.state_time_embedding(positions) + self.memory_type_embedding.weight[2]

    def build_memory(
        self,
        encoded: EncoderOutput,
        state_features: torch.Tensor,
        current_qpos: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``[dense video | dense GEL | state history]`` cross-attention memory."""
        video, gel = self._validate_inputs(encoded, state_features, current_qpos)
        dtype = self.context_projection[0].weight.dtype
        video_memory = self.context_projection[int(ExpertId.VIDEO)](video.to(dtype))
        video_memory = video_memory + self._position(self.video_coordinates) + self.memory_type_embedding.weight[0]
        gel_memory = self.context_projection[int(ExpertId.TACTILE)](gel.to(dtype))
        gel_memory = gel_memory + self._position(self.gel_coordinates) + self.memory_type_embedding.weight[1]
        state_memory = self._state_tokens(state_features)
        return self.memory_norm(torch.cat((video_memory, gel_memory, state_memory), dim=1))

    def forward(
        self,
        encoded: EncoderOutput,
        state_features: torch.Tensor,
        current_qpos: torch.Tensor,
    ) -> ControlV2Output:
        memory = self.build_memory(encoded, state_features, current_qpos)
        queries = self.query_embedding.expand(state_features.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)

        raw_residual = self.action_output(self.output_norm(queries))
        zero = raw_residual.new_zeros(raw_residual.shape[0], 1, ACTIVE_ACTION_DIM)
        # Physical control outputs stay FP32 even when the dense encoder/memory runs in BF16.
        normalized_residual = torch.cat((zero, raw_residual[:, 1:]), dim=1).float()
        action_scale = self.action_scale.to(device=raw_residual.device, dtype=torch.float32)
        residual = normalized_residual * action_scale
        current = current_qpos.float()
        arm_offsets = residual[..., :ARM_DIM]
        absolute_gripper = current[:, None, ARM_DIM:] + residual[..., ARM_DIM:]
        action_chunk = torch.cat((arm_offsets, absolute_gripper), dim=-1)
        absolute_qpos = torch.cat((current[:, None, :ARM_DIM] + arm_offsets, absolute_gripper), dim=-1)

        state_memory = memory[:, -state_features.shape[1] :]
        current_auxiliary = self.auxiliary_norm(queries[:, 0] + state_memory[:, -1])
        order_auxiliary = self.auxiliary_norm(queries.mean(dim=1) + state_memory.mean(dim=1))
        return ControlV2Output(
            action_chunk=action_chunk,
            absolute_qpos_chunk=absolute_qpos,
            residual_chunk=residual,
            normalized_residual_chunk=normalized_residual,
            action_scale=action_scale,
            phase_logits=self.phase_output(current_auxiliary),
            contact_logits=self.contact_output(current_auxiliary),
            order_logits=self.order_output(order_auxiliary),
        )

    def loss(
        self,
        output: ControlV2Output,
        target_chunk: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        phase_labels: torch.Tensor | None = None,
        contact_targets: torch.Tensor | None = None,
        order_labels: torch.Tensor | None = None,
    ) -> ControlV2Loss:
        return control_v2_loss(
            output,
            target_chunk,
            config=self.config,
            chunk_mask=chunk_mask,
            sample_weight=sample_weight,
            phase_labels=phase_labels,
            contact_targets=contact_targets,
            order_labels=order_labels,
        )


def make_chunk_relative_target(absolute_qpos_chunk: torch.Tensor, current_qpos: torch.Tensor) -> torch.Tensor:
    """Convert absolute qpos8 demonstrations to the V2 mixed action contract."""
    if absolute_qpos_chunk.ndim != 3 or absolute_qpos_chunk.shape[-1] != ACTIVE_ACTION_DIM:
        raise ValueError(f"absolute_qpos_chunk must have shape (B, H, {ACTIVE_ACTION_DIM})")
    if current_qpos.shape != (absolute_qpos_chunk.shape[0], ACTIVE_ACTION_DIM):
        raise ValueError(f"current_qpos must have shape (B, {ACTIVE_ACTION_DIM})")
    if absolute_qpos_chunk.device != current_qpos.device:
        raise ValueError("absolute_qpos_chunk and current_qpos must be on the same device")
    if not torch.isfinite(absolute_qpos_chunk).all() or not torch.isfinite(current_qpos).all():
        raise ValueError("action targets must be finite")

    arm_offsets = absolute_qpos_chunk[..., :ARM_DIM] - current_qpos[:, None, :ARM_DIM]
    mixed = torch.cat((arm_offsets, absolute_qpos_chunk[..., ARM_DIM:]), dim=-1)
    placeholder = torch.cat((torch.zeros_like(current_qpos[:, :ARM_DIM]), current_qpos[:, ARM_DIM:]), dim=-1)
    return torch.cat((placeholder[:, None], mixed[:, 1:]), dim=1)


def control_v2_loss(
    output: ControlV2Output,
    target_chunk: torch.Tensor,
    *,
    config: ControlV2Config,
    chunk_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    phase_labels: torch.Tensor | None = None,
    contact_targets: torch.Tensor | None = None,
    order_labels: torch.Tensor | None = None,
) -> ControlV2Loss:
    """Weighted Smooth-L1 chunk objective plus optional phase/contact/order supervision."""
    predicted_chunk = output.action_chunk
    expected_shape = (predicted_chunk.shape[0], config.horizon, ACTIVE_ACTION_DIM)
    if predicted_chunk.shape != expected_shape or target_chunk.shape != expected_shape:
        raise ValueError(f"predicted and target chunks must both have shape {expected_shape}")
    if output.normalized_residual_chunk.shape != expected_shape:
        raise ValueError(f"normalized residual chunk must have shape {expected_shape}")
    if output.action_scale.shape != (ACTIVE_ACTION_DIM,):
        raise ValueError(f"output action_scale must have shape ({ACTIVE_ACTION_DIM},)")
    if not torch.isfinite(output.action_scale).all() or (output.action_scale <= 0).any():
        raise ValueError("output action_scale must be finite and strictly positive")
    if target_chunk.device != predicted_chunk.device or not torch.isfinite(target_chunk).all():
        raise ValueError("target_chunk must be finite and on the output device")
    if not torch.allclose(target_chunk[:, 0], predicted_chunk[:, 0].detach(), rtol=0, atol=1e-6):
        raise ValueError("target row zero must be the current-state placeholder")

    target_residual = torch.cat(
        (
            target_chunk[..., :ARM_DIM],
            target_chunk[..., ARM_DIM:] - target_chunk[:, :1, ARM_DIM:],
        ),
        dim=-1,
    )
    target_normalized = target_residual.to(output.normalized_residual_chunk.dtype) / output.action_scale
    predicted = output.normalized_residual_chunk

    weights = torch.exp(
        -config.horizon_decay * torch.arange(config.horizon, device=predicted.device, dtype=predicted.dtype)
    )
    weights[0] = 0
    weights = weights[None, :, None].expand_as(predicted).clone()
    weights[..., ARM_DIM] *= config.gripper_loss_weight

    if chunk_mask is not None:
        if chunk_mask.shape == predicted.shape[:2]:
            chunk_mask = chunk_mask.unsqueeze(-1)
        if chunk_mask.shape not in (predicted.shape, (*predicted.shape[:2], 1)):
            raise ValueError("chunk_mask must have shape (B, H) or (B, H, 1|8)")
        if not torch.isfinite(chunk_mask).all() or (chunk_mask < 0).any():
            raise ValueError("chunk_mask must be finite and non-negative")
        weights = weights * chunk_mask.to(device=predicted.device, dtype=predicted.dtype)
    if sample_weight is not None:
        if sample_weight.shape != (predicted.shape[0],):
            raise ValueError("sample_weight must have shape (B,)")
        if not torch.isfinite(sample_weight).all() or (sample_weight < 0).any():
            raise ValueError("sample weights must be finite and non-negative")
        weights = weights * sample_weight.to(device=predicted.device, dtype=predicted.dtype)[:, None, None]

    elementwise = F.smooth_l1_loss(predicted, target_normalized, beta=config.smooth_l1_beta, reduction="none")
    chunk = (elementwise * weights).sum() / weights.sum().clamp_min(1)
    zero = predicted.sum() * 0

    phase = zero
    if phase_labels is not None:
        if phase_labels.shape != (predicted.shape[0],) or phase_labels.dtype != torch.long:
            raise ValueError("phase_labels must be int64 with shape (B,)")
        if ((phase_labels < 0) | (phase_labels >= NUM_PHASES)).any():
            raise ValueError(f"phase labels must be in [0, {NUM_PHASES})")
        phase = F.cross_entropy(output.phase_logits, phase_labels)

    contact = zero
    if contact_targets is not None:
        if contact_targets.shape == (predicted.shape[0],):
            contact_targets = contact_targets[:, None]
        if contact_targets.shape != output.contact_logits.shape:
            raise ValueError("contact_targets must have shape (B,) or (B, 1)")
        if not torch.isfinite(contact_targets).all() or ((contact_targets < 0) | (contact_targets > 1)).any():
            raise ValueError("contact targets must be finite probabilities in [0, 1]")
        contact = F.binary_cross_entropy_with_logits(output.contact_logits, contact_targets.to(predicted.dtype))

    order = zero
    if order_labels is not None:
        if order_labels.shape != (predicted.shape[0],) or order_labels.dtype != torch.long:
            raise ValueError("order_labels must be int64 with shape (B,)")
        if ((order_labels < 0) | (order_labels >= NUM_ORDER_CLASSES)).any():
            raise ValueError(f"order labels must be in [0, {NUM_ORDER_CLASSES})")
        order = F.cross_entropy(output.order_logits, order_labels)

    total = (
        chunk
        + config.phase_loss_weight * phase
        + config.contact_loss_weight * contact
        + config.order_loss_weight * order
    )
    return ControlV2Loss(total=total, chunk=chunk, phase=phase, contact=contact, order=order)
