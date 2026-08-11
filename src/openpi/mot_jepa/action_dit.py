"""A DiT action head over a frozen MoT-JEPA encoder.

Emits a future action chunk ``(B, H, 120)`` in the FTP-1 slot layout, conditioned on the frozen
encoder's per-tubelet readout. Two objectives share the same trunk:

``drifting``
    A pushforward map. The head draws its own noise and returns a chunk in **one** forward pass;
    there is no timestep, no noise schedule and no sampler. Trained by
    :mod:`openpi.mot_jepa.drifting`.

``flowmatch``
    Conditional flow matching, the control arm. Takes ``(x_t, t)`` and returns a velocity, mirroring
    ``models_pytorch/ftp1_pytorch.py:435-534`` for training and ``:541-675`` for Euler sampling.

The control arm exists because this pipeline changes two things at once relative to the FTP-1
policy -- the conditioning encoder *and* the generative objective. Without an arm that changes only
the encoder, a bad number cannot be attributed to either.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.mot_jepa.action_parse import ACTION_DIM
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.mot_encoder import EncoderOutput

OBJECTIVES = ("drifting", "flowmatch", "linear")


@dataclasses.dataclass(frozen=True)
class ActionDiTConfig:
    """Shape of the action head. The frozen encoder's widths come from the layout."""

    width: int = 512
    depth: int = 8
    num_heads: int = 8
    horizon: int = 32
    objective: str = "drifting"
    mlp_ratio: float = 4.0

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.width % self.num_heads:
            raise ValueError(f"width {self.width} must divide by num_heads {self.num_heads}")


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ActionNormalizer(nn.Module):
    """Per-domain, per-dimension z-scoring of action chunks. The head works in normalised space.

    Raw FTP-1 actions are first differences, so they are tiny -- measured mean |a| of 0.0013 to
    0.0107 depending on domain -- and their scale differs ~8x across domains. Both facts break
    training if left alone:

    * **Flow matching becomes trivial.** ``x_t = t*noise + (1-t)*a`` mixes N(0,1) noise with a
      signal of magnitude 0.01, so the action contributes ~1% of the input and the target
      ``u_t = noise - a`` is almost exactly the noise. The model scores near-zero loss by echoing
      back what it was given, having learned nothing about actions.
    * **Large-motion domains dominate.** A shared head trained on raw values weights sharpa's
      0.0013-scale actions at a fraction of D-WHEEL's 0.0107.

    Per *domain* rather than globally because batches are domain-pure and the embodiments are not
    comparable; the statistics are stored as buffers so they land in the checkpoint and evaluation
    can denormalise with exactly the values training used.
    """

    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, num_domains: int, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_domains, action_dim))
        self.register_buffer("scale", torch.ones(num_domains, action_dim))

    def load_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if mean.shape != self.mean.shape or scale.shape != self.scale.shape:
            raise ValueError(f"expected stats of shape {tuple(self.mean.shape)}")
        with torch.no_grad():
            self.mean.copy_(mean)
            # A dead or constant slot has zero spread; leaving it at 1.0 maps it to a constant
            # zero target rather than dividing by ~0 and manufacturing enormous values.
            self.scale.copy_(torch.where(scale > 1e-6, scale, torch.ones_like(scale)))

    def normalize(self, actions: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        return (actions - self.mean[domain_id].unsqueeze(1)) / self.scale[domain_id].unsqueeze(1)

    def denormalize(self, actions: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        return actions * self.scale[domain_id].unsqueeze(1) + self.mean[domain_id].unsqueeze(1)


class PerDomainLinear(nn.Module):
    """An affine map with its own weights per domain. The fix for the shared-map defect.

    Measured: eight per-domain ridges on the frozen readout reach R^2 0.65-0.93 on held-out
    UniVTAC, while ONE shared ridge on the same features -- even with per-domain z-scored targets,
    exactly what ActionNormalizer provides -- scores -0.0016. A shared map simply cannot serve
    eight embodiments, and a per-domain *bias* cannot close that gap either; it needs per-domain
    *weights*.

    Applied per unique domain in the batch rather than by gathering ``weight[domain_id]``. The
    gather would materialise ``(B, in, out)``, which at batch 64 and a 4608->3840 map is 500 MB.
    Batches are domain-pure during training, so the loop runs once; evaluation may mix, and the
    loop stays correct there without any assumption.
    """

    def __init__(self, num_domains: int, in_features: int, out_features: int, *, zero_init: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_domains, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(num_domains, out_features))
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            # Per domain, the fan-in is in_features, so scale as a normal Linear would.
            nn.init.normal_(self.weight, std=in_features**-0.5)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(*x.shape[:-1], self.weight.shape[-1])
        for domain in torch.unique(domain_id):
            rows = domain_id == domain
            out[rows] = x[rows] @ self.weight[domain] + self.bias[domain]
        return out


class DiTBlock(nn.Module):
    """Pre-norm self-attention + cross-attention + SwiGLU, with adaLN-Zero conditioning.

    adaLN-Zero rather than conditioning tokens: the gates are zero-initialised, so at step 0 every
    block is exactly the identity and the head starts as a pass-through. That matters here because
    the conditioning comes from a *frozen* encoder -- a randomly-scaled modulation at init would
    inject noise into a signal the head cannot correct by retraining the encoder.
    """

    def __init__(self, width: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.cross = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)

        hidden = int(width * mlp_ratio)
        self.mlp_in = nn.Linear(width, 2 * hidden)  # SwiGLU: value and gate
        self.mlp_out = nn.Linear(hidden, width)

        # Nine vectors: shift/scale/gate for self-attn, cross-attn and MLP.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.modulation(cond).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = params

        h = _modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]

        h = _modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self.cross(h, context, context, need_weights=False)[0]

        h = _modulate(self.norm3(x), shift_mlp, scale_mlp)
        value, gate = self.mlp_in(h).chunk(2, dim=-1)
        return x + gate_mlp.unsqueeze(1) * self.mlp_out(F.silu(gate) * value)


def timestep_embedding(t: torch.Tensor, width: int, *, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding of a continuous scalar, as in ``pi0_pytorch.create_sinusoidal_pos_embedding``."""
    half = width // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ActionDiT(nn.Module):
    """Frozen encoder readout -> future action chunk.

    The head owns no encoder. It consumes :attr:`EncoderOutput.sync_readout`, the per-tubelet
    unimodal readout, because that is the representation the viability gate measured -- the ridge
    that recovered the action chunk at R^2 0.32 pooled read exactly these tokens. Conditioning on
    the final tokens instead would condition on something never gated.

    A **domain embedding** is added to the conditioning vector. Without it the head sees only the
    readout and cannot route between embodiments, and the measurement says that is fatal here: a
    ridge fitted PER DOMAIN reaches R^2 0.65-0.93 on held-out UniVTAC while the same ridge fitted as
    ONE SHARED map over the eight domains scores -0.34. Every head trained before this was that one
    shared map, and every one lost to a per-domain constant.
    """

    def __init__(
        self,
        config: ActionDiTConfig,
        layout: TokenLayout,
        *,
        num_domains: int = 1,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        width = config.width
        # See the class docstring: one shared map over eight domains scores R^2 -0.34 where
        # per-domain maps score 0.65-0.93. This is what lets the head route.
        self.domain_embed = nn.Embedding(num_domains, width)
        nn.init.normal_(self.domain_embed.weight, std=0.02)

        # One projection per expert, so the two widths (384 video, 192 tactile at pilot) enter a
        # shared space before the head sees them.
        self.context_proj = nn.ModuleList(
            [nn.Linear(layout.video_width, width), nn.Linear(layout.tactile_width, width)]
        )
        self.context_norm = nn.LayerNorm(width)

        self.action_in = nn.Linear(action_dim, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.horizon, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([DiTBlock(width, config.num_heads, config.mlp_ratio) for _ in range(config.depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        # Per-domain, for the reason in PerDomainLinear's docstring. Cheap here: 8 x 512 x 120.
        self.action_out = PerDomainLinear(num_domains, width, action_dim, zero_init=True)

    def context_tokens(self, encoded: EncoderOutput) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, 2*num_steps, width)`` cross-attention memory and the ``(B, width)`` adaLN vector."""
        projected = [proj(readout.to(proj.weight.dtype)) for proj, readout in zip(self.context_proj, encoded.sync_readout, strict=True)]
        context = self.context_norm(torch.cat(projected, dim=1))
        return context, context.mean(dim=1)

    def forward(
        self,
        encoded: EncoderOutput,
        action_mask: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        """Returns ``(B, H, action_dim)``: a *sample* under ``drifting``, a *velocity* under ``flowmatch``.

        Both objectives share this signature. IDP is not the pure pushforward the original
        Drifting paper describes -- it evaluates the generator twice per step, once at
        ``(a_0, t=0)`` for the one-step prediction and once at ``(a* + (1-t_*)eps, t_*)`` for the
        expert-proximal probe -- so it needs the same ``(x, t)`` interface flow matching does.
        Inference is still one call; the second evaluation is a training-time term only.
        """
        context, pooled = self.context_tokens(encoded)
        x = noisy_actions.to(context.dtype)
        cond = pooled + timestep_embedding(timestep, self.config.width).to(context.dtype)
        cond = cond + self.domain_embed(domain_id).to(context.dtype)
        horizon = self.config.horizon

        # Mask BEFORE the projection, never after. An absent action group -- a domain with no left
        # arm, say -- must contribute exactly zero rather than a learned bias, or the head can read
        # embodiment identity off the input and appear to use the action without doing so.
        mask = action_mask.to(x.dtype)
        if mask.dim() == 2:  # (B, action_dim) -> broadcast over the horizon
            mask = mask.unsqueeze(1)
        x = self.action_in(x * mask)
        h = x + self.pos_embed[:, :horizon]

        for block in self.blocks:
            h = block(h, context, cond)

        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        return self.action_out(_modulate(self.final_norm(h), shift, scale), domain_id)

    @torch.no_grad()
    def sample(
        self,
        encoded: EncoderOutput,
        action_mask: torch.Tensor,
        domain_id: torch.Tensor,
        *,
        num_steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk. **One** forward pass under ``drifting``; Euler under ``flowmatch``.

        The 1-NFE claim is the headline reason to prefer drifting for a high-rate controller, so
        it is asserted by a test rather than left as a comment.
        """
        batch = encoded.sync_readout[0].shape[0]
        device = encoded.sync_readout[0].device
        x = torch.randn(batch, self.config.horizon, self.action_dim, device=device, generator=generator)

        if self.config.objective == "drifting":
            return self(encoded, action_mask, x, torch.zeros(batch, device=device), domain_id)

        dt = -1.0 / num_steps
        t = torch.ones(batch, device=device)
        for _ in range(num_steps):
            x = x + dt * self(encoded, action_mask, x, t, domain_id)
            t = t + dt
        return x


def flow_matching_loss(
    head: ActionDiT,
    encoded: EncoderOutput,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    chunk_mask: torch.Tensor,
    domain_id: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rectified-flow loss, mirroring ``ftp1_pytorch.py:472-534``.

    ``x_t = t * noise + (1 - t) * actions`` and ``u_t = noise - actions`` -- the optimal-transport
    path. The loss is masked and normalised by the mask sum, so a domain that populates 10 of 120
    slots contributes the same per-live-dimension weight as one that populates 40.
    """
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    # Beta(1.5, 1.0) concentrates samples near t=0 where the velocity field is hardest to fit.
    beta = torch.distributions.Beta(1.5, 1.0)
    t = beta.sample((actions.shape[0],)).to(actions.device) * 0.999 + 0.001
    t_b = t.reshape(-1, 1, 1)

    x_t = t_b * noise + (1.0 - t_b) * actions
    u_t = noise - actions
    velocity = head(encoded, action_mask, x_t, t, domain_id)

    error = (velocity - u_t) ** 2 * chunk_mask
    loss = error.sum() / chunk_mask.sum().clamp_min(1.0)
    return loss, {"flow_mse": float(loss.detach())}


class LinearHead(nn.Module):
    """A single affine map from the frozen readout to the action chunk. The diagnostic baseline.

    Exists to separate two explanations of a policy that loses to a constant: a defective *head*,
    or a defective *pipeline*. A ridge on these same features reaches R^2 0.6-0.95 on episode-held
    -out UniVTAC, so if this reproduces that through our own dataloader, normaliser, masking and
    loss plumbing, the pipeline is sound and the DiT is at fault. If this fails too, the fault is
    upstream of any architecture and the ridge comparison was never like-for-like.

    Deterministic on purpose. It ignores the noise and timestep arguments so it can be dropped into
    the same trainer and evaluator without special-casing, and so its score is a conditional mean
    rather than a sample -- which is the quantity the ridge reports.
    """

    def __init__(
        self,
        config: ActionDiTConfig,
        layout: TokenLayout,
        *,
        num_domains: int = 1,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        features = layout.num_steps * (layout.video_width + layout.tactile_width)
        # Normalise the readout first. It is NOT unit scale -- measured std 5.19, max |.| 54.5 --
        # so an unnormalised affine map diverges at any learning rate that trains in reasonable
        # time. The ridge this baseline reproduces standardises its features before fitting, and
        # the DiT normalises via context_norm, so without this the comparison is not like-for-like.
        self.norm = nn.LayerNorm(features)
        # Shared trunk, PER-DOMAIN output. A full per-domain 4608->3840 map would be 141M
        # parameters; factoring through `width` keeps the whole composition linear (rank `width`)
        # while giving each domain its own output weights. That distinction is the entire point:
        # a shared map with per-domain normalised targets measures R^2 -0.0016 where eight
        # per-domain maps measure 0.65-0.93, and a per-domain *bias* cannot close that.
        self.trunk = nn.Linear(features, config.width)
        # Zero-init, like the DiT's output layer: targets are unit-variance, so a zero output IS
        # the mean prediction and the loss starts at ~1.0 rather than near 9.
        self.proj = PerDomainLinear(num_domains, config.width, config.horizon * action_dim, zero_init=True)

    def forward(
        self,
        encoded: EncoderOutput,
        action_mask: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        del noisy_actions, timestep, action_mask  # deterministic: conditioning is the readout alone
        flat = torch.cat([r.flatten(start_dim=1) for r in encoded.sync_readout], dim=-1)
        hidden = self.trunk(self.norm(flat.to(self.trunk.weight.dtype)))
        out = self.proj(hidden, domain_id)
        return out.reshape(out.shape[0], self.config.horizon, self.action_dim)

    @torch.no_grad()
    def sample(self, encoded, action_mask, domain_id, *, num_steps: int = 10, generator=None) -> torch.Tensor:
        del num_steps, generator
        batch = encoded.sync_readout[0].shape[0]
        device = encoded.sync_readout[0].device
        zeros = torch.zeros(batch, self.config.horizon, self.action_dim, device=device)
        return self(encoded, action_mask, zeros, torch.zeros(batch, device=device), domain_id)


def regression_loss(
    head,
    encoded: EncoderOutput,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    chunk_mask: torch.Tensor,
    domain_id: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked MSE against the expert chunk. Normalised by the mask sum, like the other two."""
    del generator
    batch = actions.shape[0]
    zeros = torch.zeros_like(actions)
    predicted = head(encoded, action_mask, zeros, torch.zeros(batch, device=actions.device), domain_id)
    error = (predicted - actions) ** 2 * chunk_mask
    loss = error.sum() / chunk_mask.sum().clamp_min(1.0)
    return loss, {"regress_mse": float(loss.detach())}
