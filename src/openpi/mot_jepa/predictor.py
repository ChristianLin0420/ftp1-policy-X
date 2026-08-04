"""Narrow MoT predictor: context embeddings + positioned mask tokens -> target embeddings.

Deliberately much narrower than the encoder (384 vs 768), following V-JEPA 2's finding that
a wide predictor lets the model hide capacity outside the representation that actually gets
deployed.

Two details are not cosmetic:

**Mask tokens are per expert and carry only position.** A mask token is told *where and
when* it must predict, via the same RoPE coordinates the real token would have had, and
nothing about *what* is there. One learned vector per modality, so the predictor cannot
smuggle content through a per-position lookup.

**The mode embedding is added to mask tokens only, never to context.** If it were added to
context, the encoder-side representation entering the predictor would become
mode-dependent, and the probes -- which compare representations across steps that used
different modes -- would silently be measuring the mode instead of the content.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.mot_encoder import NUM_EXPERTS
from openpi.mot_jepa.mot_encoder import MoTBlock
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.rope3d import Rope3D
from openpi.mot_jepa.rope3d import Rope3DConfig


@dataclasses.dataclass(frozen=True)
class MoTPredictorConfig:
    """Predictor shape. Defaults are the design document's 12 x 384 narrow predictor."""

    depth: int = 12
    width: int = 384
    num_local_layers: int = 0
    num_heads: int = 6
    head_dim: int = 64
    mlp_ratio: float = 4.0
    rope: Rope3DConfig = dataclasses.field(default_factory=Rope3DConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.num_local_layers <= self.depth:
            raise ValueError(f"num_local_layers={self.num_local_layers} must be in [0, depth={self.depth}]")
        if self.rope.head_dim != self.head_dim:
            raise ValueError(f"rope.head_dim={self.rope.head_dim} must equal head_dim={self.head_dim}")

    def as_block_config(self) -> MoTEncoderConfig:
        return MoTEncoderConfig(
            depth=self.depth,
            num_local_layers=self.num_local_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            rope=self.rope,
        )


class MoTPredictor(nn.Module):
    """Predicts teacher-space target embeddings from student context embeddings."""

    def __init__(self, config: MoTPredictorConfig, layout: TokenLayout) -> None:
        super().__init__()
        self.config = config
        self.layout = layout
        self.encoder_widths = (layout.video_width, layout.tactile_width)
        block_config = config.as_block_config()

        self.rope = Rope3D(config.rope, layout)
        self.in_proj = nn.ModuleList([nn.Linear(width, config.width) for width in self.encoder_widths])
        self.out_proj = nn.ModuleList([nn.Linear(config.width, width) for width in self.encoder_widths])
        self.blocks = nn.ModuleList(
            [
                MoTBlock((config.width,) * NUM_EXPERTS, block_config, local=i < config.num_local_layers)
                for i in range(config.depth)
            ]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(NUM_EXPERTS)])

        self.mask_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, config.width)) for _ in range(NUM_EXPERTS)])
        for token in self.mask_token:
            nn.init.trunc_normal_(token, std=0.02)
        self.mode_embed = nn.Embedding(len(MaskMode), config.width)
        nn.init.zeros_(self.mode_embed.weight)

        # Tubelet step of every global token index, used to place per-step conditioning on
        # the mask tokens. Non-persistent, like the RoPE tables: derived from the layout, so
        # it neither enters a checkpoint nor goes stale.
        self.register_buffer("token_step", layout.coords[:, 0].contiguous(), persistent=False)

        self._gradient_checkpointing = False

    def set_gradient_checkpointing(self, *, enabled: bool) -> None:
        self._gradient_checkpointing = bool(enabled)

    def _run_block(
        self,
        block: MoTBlock,
        xs: list[torch.Tensor],
        rope: list[tuple[torch.Tensor, torch.Tensor]],
        valid: list[torch.Tensor | None],
    ) -> list[torch.Tensor]:
        """Mirrors :meth:`MoTEncoder._run_block`; safe because this module has no RNG."""
        if self._gradient_checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                block, xs, rope, valid, use_reentrant=False, preserve_rng_state=False
            )
        return block(xs, rope, valid)

    def forward(
        self,
        context: list[torch.Tensor],
        context_index: list[torch.Tensor],
        target_index: list[torch.Tensor],
        mode: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Predict target embeddings.

        Args:
            context: Per-expert ``(B, L_ctx_e, width_e)`` encoder outputs.
            context_index: Per-expert ``(B, L_ctx_e)`` global token indices.
            target_index: Per-expert ``(B, L_tgt_e)`` global token indices.
            mode: Scalar :class:`MaskMode` value for this batch.
            cond: Optional ``(B, num_steps, width)`` per-timestep conditioning, added to each
                mask token at *that token's own* tubelet step. Used by action post-training
                so the action applied between steps *t* and *t+1* reaches exactly the tokens
                it explains. ``None`` reproduces pretraining bit-for-bit, and pretraining
                never passes it.

        Returns:
            Per-expert ``(B, L_tgt_e, width_e)`` predictions in encoder/teacher space.
        """
        if not len(context) == len(context_index) == len(target_index) == NUM_EXPERTS:
            raise ValueError("context, context_index and target_index must each have one entry per expert")
        if cond is not None and cond.shape[1] != self.layout.num_steps:
            raise ValueError(f"cond must be (B, {self.layout.num_steps}, width), got {tuple(cond.shape)}")

        mode_vector = self.mode_embed(mode.reshape(()).to(torch.long))

        xs, rope, splits = [], [], []
        for expert in range(NUM_EXPERTS):
            ctx = self.in_proj[expert](context[expert])
            num_targets = target_index[expert].shape[1]
            batch = ctx.shape[0]
            masks = self.mask_token[expert].expand(batch, num_targets, -1) + mode_vector
            if cond is not None:
                steps = self.token_step[target_index[expert]]  # (B, L_tgt_e)
                masks = masks + torch.gather(cond, 1, steps[..., None].expand(-1, -1, cond.shape[-1]))
            xs.append(torch.cat([ctx, masks], dim=1))
            joint_index = torch.cat([context_index[expert], target_index[expert]], dim=1)
            rope.append(self.rope.gather(joint_index))
            splits.append((ctx.shape[1], num_targets))

        valid: list[torch.Tensor | None] = [None] * NUM_EXPERTS
        for block in self.blocks:
            xs = self._run_block(block, xs, rope, valid)

        predictions = []
        for expert in range(NUM_EXPERTS):
            num_context, _ = splits[expert]
            hidden = self.norm[expert](xs[expert][:, num_context:])
            predictions.append(self.out_proj[expert](hidden))
        return predictions


def target_index_by_expert(
    layout: TokenLayout, target_index: torch.Tensor, expert_bounds: torch.Tensor
) -> list[torch.Tensor]:
    """Split a ``(B, L_tgt)`` global target row into per-expert global index tensors."""
    out = []
    for expert in ExpertId:
        lo, hi = (int(v) for v in expert_bounds[int(expert)])
        out.append(target_index[:, lo:hi])
    return out
