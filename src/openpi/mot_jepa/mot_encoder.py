"""Mixture-of-Transformers encoder: per-expert weights, one attention.

Each modality owns its LayerNorms, q/k/v projections, output projection and FFN; the two
experts meet only inside the attention operation. That is what lets the video expert run at
width 768 while the tactile expert runs at 384: both project into a *shared* ``num_heads x
head_dim`` attention space and project back out to their own residual width.

Layers ``[0, num_local_layers)`` attend within a modality only. Two reasons, both
load-bearing:

1. It gives the synchrony head a genuinely *unimodal* readout. If attention were global from
   layer 1, "the tactile representation" would already contain video and the synchrony loss
   would be satisfiable by copying rather than by binding.
2. Because the mask contract keeps each stream contiguous, modality-local attention is two
   plain SDPA calls over slices -- no ``(B, 1, L, L)`` mask is ever materialized, which at
   L ~ 2400 would be ~92 MB per sample per layer.

Gradient checkpointing is gated **only** by an explicit flag. The FTP-1 Gemma stack
force-enables it whenever ``self.training`` (``ftp1_gemma_pytorch.py:362-366``), overriding
its own config; copying that would make it impossible to run the determinism test with
checkpointing off and on and compare bitwise.

There is no RNG anywhere in this module, and a static test enforces that.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint

from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.rope3d import Rope3D
from openpi.mot_jepa.rope3d import Rope3DConfig
from openpi.mot_jepa.rope3d import apply_rope

NUM_EXPERTS = len(ExpertId)


@dataclasses.dataclass(frozen=True)
class MoTEncoderConfig:
    """Encoder shape.

    Defaults are the design document's ViT-B-class encoder: 16 layers, 12 heads of 64, with
    the first 6 layers modality-local.
    """

    depth: int = 16
    num_local_layers: int = 6
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 4.0
    rope: Rope3DConfig = dataclasses.field(default_factory=Rope3DConfig)
    qk_norm: bool = True
    """LayerNorm q and k over ``head_dim`` before the dot product. Bounds the attention logits.

    Set ``False`` ONLY to load a checkpoint trained without it -- it changes the parameter count,
    and ``load_frozen_backbone`` checks that count rather than failing silently.

    Without this, nothing bounds ``q.k``: ``project_qkv`` feeds ``qkv(norm1(x))`` straight into
    ``scaled_dot_product_attention``, whose only scaling is the constant ``1/sqrt(head_dim)``.
    Growth in ``norm1``'s gain and ``qkv``'s weights multiplies into the logits, the softmax
    saturates toward one-hot, gradients spike, and the growth feeds itself.

    Measured on job 6569421, predictor block 3, tactile expert -- the run that made this
    non-optional::

        step 40000   max|q|    42.6   max logit         2,680
        step 60000   max|q| 4,998.8   max logit   164,247,568

    Diverged at ~45k of 100k with gradient norms to 546 against a clip of 1.0, and the tactile
    representation was destroyed (RankMe 147 -> 11) while the LOSS FELL to 0.497, because a
    collapsed representation is trivially predictable. Weight growth was confined to
    ``predictor.blocks.3.experts.1``: ``norm1.bias`` 36x, ``qkv.weight`` 16x, everything else
    under 2x.

    This is the documented failure mode of mixed-modal transformers, and the reason it bites
    *here* specifically: ``MoTPredictorConfig.num_local_layers`` is 0, so every predictor block is
    global and all three experts' q/k are concatenated into ONE softmax (``MoTBlock.forward``).
    The modalities compete inside a single normalisation, which is exactly the setting where
    Chameleon (arXiv:2405.09818) reports logit drift and norm growth beyond bf16's range, and
    adopts QK-norm as the remedy; ViT-22B (arXiv:2302.05442) reports the same divergence from
    "extremely large values in attention logits" giving near-zero-entropy attention. FTP-1's own
    Gemma in this repo carries no softcap or QK-norm, so there was no in-house precedent to
    inherit."""

    def __post_init__(self) -> None:
        if not 0 <= self.num_local_layers <= self.depth:
            raise ValueError(f"num_local_layers={self.num_local_layers} must be in [0, depth={self.depth}]")
        if self.rope.head_dim != self.head_dim:
            raise ValueError(f"rope.head_dim={self.rope.head_dim} must equal head_dim={self.head_dim}")

    @property
    def attention_dim(self) -> int:
        return self.num_heads * self.head_dim


def _pool_by_step(tokens: torch.Tensor, steps: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Mean-pool ``(B, L, D)`` tokens into ``(B, num_steps, D)`` by their tubelet step.

    Steps are per-sample because block positions differ across the batch, so this is a
    batched scatter rather than a reshape. Steps with no surviving token pool to zero.
    """
    batch, _, width = tokens.shape
    index = steps[..., None].expand(-1, -1, width)
    summed = torch.zeros(batch, num_steps, width, dtype=tokens.dtype, device=tokens.device)
    summed.scatter_add_(1, index, tokens)
    counts = torch.zeros(batch, num_steps, 1, dtype=tokens.dtype, device=tokens.device)
    counts.scatter_add_(1, steps[..., None], torch.ones_like(tokens[..., :1]))
    return summed / counts.clamp(min=1.0)


class ExpertLayer(nn.Module):
    """One modality's parameters for one transformer layer."""

    def __init__(self, width: int, config: MoTEncoderConfig) -> None:
        super().__init__()
        attn_dim = config.attention_dim
        hidden = int(width * config.mlp_ratio)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        # Per-head, over head_dim. Identity (parameter-free) when disabled, so a config with
        # qk_norm=False has exactly the parameter count it had before this existed.
        self.q_norm = nn.LayerNorm(config.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(config.head_dim) if config.qk_norm else nn.Identity()

    def project_qkv(self, x: torch.Tensor, config: MoTEncoderConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.reshape(batch, length, 3, config.num_heads, config.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Before RoPE, which is a rotation and so norm-preserving -- normalising here therefore
        # bounds the logits after RoPE too, and matches the usual placement.
        return self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]


class MoTBlock(nn.Module):
    """One layer: per-expert projections around a single (local or global) attention."""

    def __init__(self, widths: tuple[int, ...], config: MoTEncoderConfig, *, local: bool) -> None:
        super().__init__()
        self.config = config
        self.local = local
        self.experts = nn.ModuleList([ExpertLayer(width, config) for width in widths])

    def forward(
        self,
        xs: list[torch.Tensor],
        rope: list[tuple[torch.Tensor, torch.Tensor]],
        valid: list[torch.Tensor | None],
    ) -> list[torch.Tensor]:
        queries, keys, values = [], [], []
        for expert, x in zip(self.experts, xs, strict=True):
            q, k, v = expert.project_qkv(x, self.config)
            cos, sin = rope[len(queries)]
            queries.append(apply_rope(q, cos, sin))
            keys.append(apply_rope(k, cos, sin))
            values.append(v)

        if self.local:
            # Two independent SDPA calls over contiguous slices; no attention mask allocated.
            attended = [
                F.scaled_dot_product_attention(q, k, v, attn_mask=_key_mask(m, q.dtype))
                for q, k, v, m in zip(queries, keys, values, valid, strict=True)
            ]
        else:
            lengths = [q.shape[2] for q in queries]
            joint_mask = _concat_valid(valid, lengths, queries[0].device)
            out = F.scaled_dot_product_attention(
                torch.cat(queries, dim=2),
                torch.cat(keys, dim=2),
                torch.cat(values, dim=2),
                attn_mask=_key_mask(joint_mask, queries[0].dtype),
            )
            attended = list(torch.split(out, lengths, dim=2))

        outputs = []
        for expert, x, head_out in zip(self.experts, xs, attended, strict=True):
            batch, _, length, _ = head_out.shape
            merged = head_out.transpose(1, 2).reshape(batch, length, self.config.attention_dim)
            hidden = x + expert.out_proj(merged)
            outputs.append(hidden + expert.mlp(expert.norm2(hidden)))
        return outputs


def _key_mask(valid: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """``(B, L)`` validity into the additive ``(B, 1, 1, L)`` mask SDPA expects."""
    if valid is None:
        return None
    mask = torch.zeros(valid.shape, dtype=dtype, device=valid.device)
    return mask.masked_fill(~valid, torch.finfo(dtype).min)[:, None, None, :]


def _concat_valid(valid: list[torch.Tensor | None], lengths: list[int], device: torch.device) -> torch.Tensor | None:
    if all(mask is None for mask in valid):
        return None
    parts = []
    for mask, length in zip(valid, lengths, strict=True):
        if mask is None:
            batch = next(m.shape[0] for m in valid if m is not None)
            parts.append(torch.ones(batch, length, dtype=torch.bool, device=device))
        else:
            parts.append(mask)
    return torch.cat(parts, dim=1)


@dataclasses.dataclass
class EncoderOutput:
    """Per-expert token states plus the unimodal readout used by the synchrony loss."""

    tokens: list[torch.Tensor]  # indexed by ExpertId, each (B, L_e, width_e)
    sync_readout: list[torch.Tensor]  # indexed by ExpertId, each (B, num_steps, width_e)
    final_readout: list[torch.Tensor]  # indexed by ExpertId, each (B, num_steps, width_e)
    """The POLICY feature. Same pooling as ``sync_readout`` but taken from the final normed
    tokens instead of the layer-``num_local_layers`` snapshot.

    The distinction decides whether a dynamics-aware objective reaches a policy at all. The
    forecasting gradient (``MaskMode.F``) arrives through ``tokens`` -- the output of the LAST
    block -- while ``sync_readout`` is snapshotted before the first *global* layer. Nothing
    stops the encoder from satisfying forecasting inside the global layers and leaving the
    early snapshot a per-step appearance code, in which case a policy reading ``sync_readout``
    would see no benefit and the run would be misread as "forecasting does not help".

    ``sync_readout`` stays the right input for the cross-modal retrieval probe, where mixing the
    two modalities inflates the score by self-matching. It is the wrong input for a policy, where
    mixing is exactly what is wanted. Deliberately has no default: a stale caller must fail loudly
    rather than silently feed a head the wrong tensor.
    """

    def expert(self, expert: ExpertId) -> torch.Tensor:
        return self.tokens[int(expert)]


class MoTEncoder(nn.Module):
    """Stack of :class:`MoTBlock` with a modality-local prefix and a global suffix."""

    def __init__(self, config: MoTEncoderConfig, layout: TokenLayout) -> None:
        super().__init__()
        self.config = config
        self.layout = layout
        self.widths = (layout.video_width, layout.tactile_width)
        self.rope = Rope3D(config.rope, layout)
        self.blocks = nn.ModuleList(
            [MoTBlock(self.widths, config, local=i < config.num_local_layers) for i in range(config.depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for width in self.widths])
        self._gradient_checkpointing = False

    def set_gradient_checkpointing(self, *, enabled: bool) -> None:
        """Explicit only. Never auto-enabled from ``self.training``."""
        self._gradient_checkpointing = bool(enabled)

    @property
    def gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    def _run_block(
        self,
        block: MoTBlock,
        xs: list[torch.Tensor],
        rope: list[tuple[torch.Tensor, torch.Tensor]],
        valid: list[torch.Tensor | None],
    ) -> list[torch.Tensor]:
        if self._gradient_checkpointing and torch.is_grad_enabled():
            # preserve_rng_state=False is safe here precisely because this module contains
            # no RNG: a recompute cannot disagree with the original forward.
            return torch.utils.checkpoint.checkpoint(
                block, xs, rope, valid, use_reentrant=False, preserve_rng_state=False
            )
        return block(xs, rope, valid)

    def forward(
        self,
        tokens: list[torch.Tensor],
        index: list[torch.Tensor],
        valid: list[torch.Tensor | None] | None = None,
    ) -> EncoderOutput:
        """Encode selected tokens.

        Args:
            tokens: Per-expert ``(B, L_e, width_e)`` embeddings, already gathered.
            index: Per-expert ``(B, L_e)`` *global* token indices, used for RoPE and for the
                step-wise synchrony pooling.
            valid: Optional per-expert ``(B, L_e)`` boolean key masks for padded sensors.
        """
        if len(tokens) != NUM_EXPERTS or len(index) != NUM_EXPERTS:
            raise ValueError(f"expected {NUM_EXPERTS} experts, got {len(tokens)} tokens and {len(index)} indices")
        valid = list(valid) if valid is not None else [None] * NUM_EXPERTS

        rope = [self.rope.gather(idx) for idx in index]
        step_ids = self.layout.step_ids.to(index[0].device)
        steps = [step_ids[idx] for idx in index]

        xs = list(tokens)
        readout: list[torch.Tensor] | None = None
        for depth, block in enumerate(self.blocks):
            if depth == self.config.num_local_layers:
                # Snapshot before the first global layer: this is the unimodal readout the
                # synchrony head consumes. Taken after the local prefix so it is a real
                # per-modality representation, not one that already contains the other.
                readout = [_pool_by_step(x, s, self.layout.num_steps) for x, s in zip(xs, steps, strict=True)]
            xs = self._run_block(block, xs, rope, valid)

        if readout is None:  # num_local_layers == depth
            readout = [_pool_by_step(x, s, self.layout.num_steps) for x, s in zip(xs, steps, strict=True)]

        normed = [norm(x) for norm, x in zip(self.norms, xs, strict=True)]
        return EncoderOutput(
            tokens=normed,
            sync_readout=readout,
            # Zero new parameters, and identical shape to sync_readout, so every consumer's
            # feature width is unchanged.
            final_readout=[_pool_by_step(x, s, self.layout.num_steps) for x, s in zip(normed, steps, strict=True)],
        )


def split_index_by_expert(
    layout: TokenLayout, index: torch.Tensor, expert_bounds: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split a ``(B, L_sel)`` global index row into per-expert global and local indices.

    Local indices are offsets into each expert's own full-length token tensor, which is what
    the gather from the embedding output needs. Global indices stay global because RoPE and
    the step pooling are defined on the shared token axis.
    """
    tactile_start = layout.expert_slices[ExpertId.TACTILE].start
    global_index, local_index = [], []
    for expert in ExpertId:
        lo, hi = (int(v) for v in expert_bounds[int(expert)])
        chunk = index[:, lo:hi]
        global_index.append(chunk)
        local_index.append(chunk if expert is ExpertId.VIDEO else chunk - tactile_start)
    return global_index, local_index


def gather_tokens(full: torch.Tensor, local_index: torch.Tensor) -> torch.Tensor:
    """Gather ``(B, L_sel, D)`` from ``(B, L_full, D)`` using per-sample indices."""
    return full.gather(1, local_index[..., None].expand(-1, -1, full.shape[-1]))
