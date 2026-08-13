"""The four loss terms.

``L = L_vid + 1.0 * L_tac + 0.05 * L_lowdim + 0.2 * L_sync``

Only ``L_sync`` is *provably* minimized at the matched (video, tactile) pair. Every
JEPA-family reconstruction term has a degenerate solution -- masked video can be recovered
from visible video, masked tactile can be interpolated from visible tactile -- so on its own
none of them forces the model to read *which* tactile signal it was given. That is precisely
the failure the ablation measured: ``shuffle`` scored the same as ``real``. ``L_sync``'s
negatives *are* real tactile readings from other clips, so a model that cannot tell them
apart cannot minimize it.

Two structural safeguards live here:

**Stable key set.** ``forward`` always returns the same dictionary keys, emitting an exact
zero for a term that is inactive under the current mask mode. The trainer all-reduces the
auxiliary metrics as one stacked tensor with sorted keys; if the key set differed across
ranks the shapes would mismatch and DDP would hang.

**Low-dimensional anti-collapse.** A 4-D torque vector at 30 Hz is nearly perfectly
interpolable, so a raw regression on it is free accuracy that teaches nothing. The target is
therefore centered by its own per-slot temporal mean (predicting the mean now scores zero),
low-variance slots are gated out, and the term is self-normalized by a running estimate of
its own magnitude so it contributes a fixed share of gradient regardless of how easy it is.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812

from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.masking import ClipMasks


@dataclasses.dataclass(frozen=True)
class LossConfig:
    """Loss weights and knobs. Defaults are the design document's."""

    weight_video: float = 1.0
    weight_tactile: float = 1.0
    weight_lowdim: float = 0.05
    weight_sync: float = 0.2
    sync_warmup_steps: int = 5_000
    sync_level_a_weight: float = 0.8
    sync_level_b_weight: float = 0.2
    temperature: float = 0.07
    projector_dim: int = 256
    lowdim_variance_gate: float = 0.1
    lowdim_ema_momentum: float = 0.99
    gather_negatives: bool = True

    video_objective: str = "l1"
    gel_objective: str = "l1"
    """``l1`` or ``direction``. Per stream rather than one global knob: MoT shares a single
    attention, so switching only the tactile term also shifts that expert's gradient scale
    relative to video. Keeping them separable lets that be ablated instead of confounded.

    The low-dimensional term is deliberately not selectable -- see :func:`jepa_direction_loss`.
    """
    direction_beta: float = 0.25
    direction_delta: float = 1.0


def normalize_targets(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free LayerNorm over the feature axis, in fp32.

    The standard I-JEPA/V-JEPA target stabilizer. Computed in fp32 regardless of the
    surrounding autocast dtype, because the targets define the loss scale and a bf16
    normalization here quietly shifts it.
    """
    return F.layer_norm(x.to(torch.float32), x.shape[-1:])


def jepa_regression_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean L1 between a prediction and a stop-gradient normalized target."""
    if prediction.numel() == 0:
        return prediction.new_zeros(())
    error = (prediction.to(torch.float32) - target.detach()).abs().mean(dim=-1)
    return _reduce(error, valid)


def _reduce(error: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return error.mean()
    weights = valid.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp(min=1.0)


def jepa_direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    beta: float = 0.25,
    delta: float = 1.0,
) -> torch.Tensor:
    """``(1 - cos) + beta * huber(||pred|| - ||target||)``, direction first.

    ``normalize_targets`` is *parameter-free* LayerNorm, so every target has unit variance over
    its ``D`` channels and therefore ``||z*|| = sqrt(D)`` exactly. The target magnitude carries
    no information at all, yet the L1 form spends part of its gradient budget matching it --
    the tactile predictor has to learn to emit vectors of norm ~19.6 before direction matters.
    This objective optimizes only the informative part and anchors the norm cheaply.

    Two details that are not cosmetic:

    **The prediction is centered before the cosine.** Cosine is scale-invariant but *not*
    shift-invariant, and the target is centered by LayerNorm. Any component of the prediction
    along the all-ones direction is orthogonal to the target, contributes nothing to the
    numerator, and inflates the denominator -- it strictly lowers the cosine. The model would
    learn to center itself; doing it here is free and faster.

    **The Huber term is load-bearing, not decoration.** ``d(cos)/d(pred)`` scales as
    ``1/||pred||`` and diverges as predictions shrink, so an unanchored direction loss is
    unstable near zero. Because ``||z*||`` is a known constant this anchor is nearly free
    supervision.

    Not offered for the low-dimensional stream: that term is centered per slot, variance-gated
    and self-normalized (``_center_by_slot`` and the ``lowdim_scale`` buffer), so its target has
    no fixed norm and the premise above does not hold.
    """
    if prediction.numel() == 0:
        return prediction.new_zeros(())
    prediction = prediction.to(torch.float32)
    target = target.detach()

    centered = prediction - prediction.mean(dim=-1, keepdim=True)
    cosine = F.cosine_similarity(centered, target, dim=-1, eps=1e-6)
    norm_gap = centered.norm(dim=-1) - target.norm(dim=-1)
    huber = F.huber_loss(norm_gap, torch.zeros_like(norm_gap), reduction="none", delta=delta)
    return _reduce((1.0 - cosine) + beta * huber, valid)


#: Dispatch table. ``l1`` is the default everywhere, so an existing frozen config -- which has
#: no objective key at all -- rebuilds to exactly the behaviour it was trained with.
LATENT_OBJECTIVES = {"l1": jepa_regression_loss, "direction": jepa_direction_loss}


def latent_loss(
    objective: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    beta: float = 0.25,
    delta: float = 1.0,
) -> torch.Tensor:
    if objective not in LATENT_OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}, expected one of {sorted(LATENT_OBJECTIVES)}")
    if objective == "l1":
        return jepa_regression_loss(prediction, target, valid)
    return jepa_direction_loss(prediction, target, valid, beta=beta, delta=delta)


class AllGatherWithGrad(torch.autograd.Function):
    """All-gather that propagates gradients back to the contributing rank.

    ``dist.all_gather`` detaches, which would silently drop most of the InfoNCE gradient and
    leave the objective far weaker than the batch size suggests.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            ctx.rank, ctx.world_size = 0, 1
            return tensor
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        gathered[dist.get_rank()] = tensor  # keep this rank's autograd history
        ctx.rank, ctx.world_size = dist.get_rank(), world_size
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if ctx.world_size == 1:
            return grad_output
        chunk = grad_output.shape[0] // ctx.world_size
        return grad_output[ctx.rank * chunk : (ctx.rank + 1) * chunk]


def sync_loss_level_a(
    video: torch.Tensor, tactile: torch.Tensor, *, temperature: float, gather: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Across-sample synchrony: match clip *i*'s video to clip *i*'s touch.

    This is the ``shuffle`` ablation turned into a training signal -- the negatives are other
    clips' genuine tactile readings.

    No positional shortcut exists at this level: every candidate occupies the identical
    ``t = 0..T-1`` grid, so position carries exactly zero bits about which pairing is
    correct. That is a structural guarantee, not an empirical hope.
    """
    video = F.normalize(video.to(torch.float32), dim=-1)
    tactile = F.normalize(tactile.to(torch.float32), dim=-1)
    local_batch = video.shape[0]

    if gather and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        all_video = AllGatherWithGrad.apply(video)
        all_tactile = AllGatherWithGrad.apply(tactile)
        offset = dist.get_rank() * local_batch
    else:
        all_video, all_tactile, offset = video, tactile, 0

    logits = video @ all_tactile.t() / temperature
    labels = torch.arange(local_batch, device=video.device) + offset
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(tactile @ all_video.t() / temperature, labels)

    with torch.no_grad():
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        chance = torch.tensor(1.0 / max(all_tactile.shape[0], 1), device=video.device)
    return 0.5 * (loss_v2t + loss_t2v), {"sync_a_acc": accuracy, "sync_a_chance": chance}


def sync_loss_level_b(
    video: torch.Tensor, tactile: torch.Tensor, *, temperature: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Within-clip synchrony across time: match instant *t* to instant *t*.

    Negatives are the same scene, object, sensor and lighting -- only the dynamics differ --
    so identity carries no signal and only *when things happen* can solve it.

    Unlike level A, this level **does** admit a positional shortcut, which is why the
    positional-shortcut control probe carries a hard kill threshold.

    Args:
        video: ``(B, T, D)`` per-instant readout.
        tactile: ``(B, T, D)`` per-instant readout.
    """
    video = F.normalize(video.to(torch.float32), dim=-1)
    tactile = F.normalize(tactile.to(torch.float32), dim=-1)
    batch, steps, _ = video.shape

    logits = torch.einsum("btd,bsd->bts", video, tactile) / temperature
    labels = torch.arange(steps, device=video.device).expand(batch, -1)
    loss_v2t = F.cross_entropy(logits.reshape(batch * steps, steps), labels.reshape(-1))
    loss_t2v = F.cross_entropy(logits.transpose(1, 2).reshape(batch * steps, steps), labels.reshape(-1))

    with torch.no_grad():
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return 0.5 * (loss_v2t + loss_t2v), {"sync_b_acc": accuracy}


class Projector(nn.Module):
    """Two-layer head used only by the synchrony loss and discarded after pretraining."""

    def __init__(self, width: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _center_by_slot(values: torch.Tensor, slot_ids: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Subtract each ``(sample, slot)`` group's mean over time.

    Removes the constant component of a low-dimensional channel, which is the part a
    predictor can nail by emitting a per-slot bias. What is left is the time-varying signal
    -- contact onset, force ramp, slip -- which is the only part worth learning.
    """
    batch, _, width = values.shape
    index = slot_ids[..., None].expand(-1, -1, width)
    summed = torch.zeros(batch, num_slots, width, dtype=values.dtype, device=values.device)
    summed.scatter_add_(1, index, values)
    counts = torch.zeros(batch, num_slots, 1, dtype=values.dtype, device=values.device)
    counts.scatter_add_(1, slot_ids[..., None], torch.ones_like(values[..., :1]))
    means = summed / counts.clamp(min=1.0)
    return values - means.gather(1, index)


class MotJepaLoss(nn.Module):
    """Combines the four terms and reports a stable metric dictionary."""

    lowdim_scale: torch.Tensor

    def __init__(self, config: LossConfig, layout: TokenLayout) -> None:
        super().__init__()
        self.config = config
        self.layout = layout
        self.projectors = nn.ModuleList(
            [
                Projector(layout.video_width, config.projector_dim),
                Projector(layout.tactile_width, config.projector_dim),
            ]
        )
        # Running magnitude of the raw low-dim term, so its gradient share stays fixed.
        self.register_buffer("lowdim_scale", torch.ones(()))

    def sync_weight(self, step: int) -> float:
        """Warm the synchrony term in from zero so it cannot dominate a random init."""
        if self.config.sync_warmup_steps <= 0:
            return self.config.weight_sync
        ramp = min(max(step / self.config.sync_warmup_steps, 0.0), 1.0)
        return self.config.weight_sync * ramp

    def _split_tactile(self, tensor: torch.Tensor, masks: ClipMasks) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the tactile expert's target-aligned tensor into gel and low-dim parts."""
        gel_lo, gel_hi = (int(v) for v in masks.tgt_bounds[int(StreamId.GEL)])
        low_lo, low_hi = (int(v) for v in masks.tgt_bounds[int(StreamId.LOWDIM)])
        offset = int(masks.tgt_expert_bounds[int(ExpertId.TACTILE), 0])
        return tensor[:, gel_lo - offset : gel_hi - offset], tensor[:, low_lo - offset : low_hi - offset]

    def _lowdim_slot_ids(self, masks: ClipMasks, device: torch.device) -> torch.Tensor:
        low_lo, low_hi = (int(v) for v in masks.tgt_bounds[int(StreamId.LOWDIM)])
        indices = masks.tgt_index[:, low_lo:low_hi].to(device)
        local = indices - self.layout.stream_slices[StreamId.LOWDIM].start
        return local % self.layout.lowdim_slots

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        student_readout: list[torch.Tensor],
        masks: ClipMasks,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total loss and a fixed-key metric dictionary.

        Args:
            predictions: Per-expert ``(B, L_tgt_e, width_e)`` student predictions.
            targets: Per-expert ``(B, L_tgt_e, width_e)`` teacher outputs, already
                LayerNormed and detached.
            student_readout: Per-expert ``(B, num_steps, width_e)`` unimodal layer-S readout.
            masks: The batch's masks, for splitting tactile into gel and low-dim.
            step: Global optimizer step, for the synchrony warmup.
        """
        device = predictions[0].device
        video_pred, tactile_pred = predictions
        video_tgt, tactile_tgt = targets
        gel_pred, lowdim_pred = self._split_tactile(tactile_pred, masks)
        gel_tgt, lowdim_tgt = self._split_tactile(tactile_tgt, masks)

        loss_video = latent_loss(
            self.config.video_objective,
            video_pred,
            video_tgt,
            beta=self.config.direction_beta,
            delta=self.config.direction_delta,
        )
        loss_gel = latent_loss(
            self.config.gel_objective,
            gel_pred,
            gel_tgt,
            beta=self.config.direction_beta,
            delta=self.config.direction_delta,
        )

        # -- low-dim: centered, variance-gated, self-normalized ------------------------
        if lowdim_pred.shape[1] > 0:
            slot_ids = self._lowdim_slot_ids(masks, device)
            centered_tgt = _center_by_slot(lowdim_tgt.to(torch.float32), slot_ids, self.layout.lowdim_slots)
            centered_pred = _center_by_slot(lowdim_pred.to(torch.float32), slot_ids, self.layout.lowdim_slots)
            magnitude = centered_tgt.abs().mean(dim=-1)
            gate = magnitude > self.config.lowdim_variance_gate * magnitude.mean().clamp(min=1e-8)
            raw_lowdim = jepa_regression_loss(centered_pred, centered_tgt, valid=gate)
            if self.training:
                with torch.no_grad():
                    momentum = self.config.lowdim_ema_momentum
                    self.lowdim_scale.mul_(momentum).add_(raw_lowdim.detach().clamp(min=1e-6), alpha=1 - momentum)
            loss_lowdim = raw_lowdim / self.lowdim_scale.clamp(min=1e-6)
            lowdim_gate_frac = gate.float().mean()
        else:
            raw_lowdim = torch.zeros((), device=device)
            loss_lowdim = torch.zeros((), device=device)
            lowdim_gate_frac = torch.zeros((), device=device)

        # -- synchrony ------------------------------------------------------------------
        # Drop steps that have no context token. Only mode F creates any: its trailing forecast
        # window leaves those steps empty in BOTH experts, and `_pool_by_step` divides by a
        # count clamped to 1, so the readout there is exactly zero.
        #
        # Level A would survive that -- `F.normalize` absorbs the rescale -- but level B would
        # not. `projector(0)` is the projector's bias, the SAME vector for every sample and every
        # empty step, so after normalisation those rows are bit-identical and the cross-entropy
        # below would be asking the model to tell them apart. That is an irreducible floor plus a
        # real gradient pushing the bias around, and it would have been misread as mode F hurting
        # synchrony.
        #
        # Mode and horizon are pure functions of `step`, so every rank slices identically and DDP
        # stays in lockstep.
        valid_steps = int(masks.num_context_steps)
        student_readout = [readout[:, :valid_steps] for readout in student_readout]
        projected = [proj(readout) for proj, readout in zip(self.projectors, student_readout, strict=True)]
        video_steps, tactile_steps = projected
        loss_a, metrics_a = sync_loss_level_a(
            video_steps.mean(dim=1),
            tactile_steps.mean(dim=1),
            temperature=self.config.temperature,
            gather=self.config.gather_negatives,
        )
        loss_b, metrics_b = sync_loss_level_b(video_steps, tactile_steps, temperature=self.config.temperature)
        loss_sync = self.config.sync_level_a_weight * loss_a + self.config.sync_level_b_weight * loss_b

        sync_weight = self.sync_weight(step)
        total = (
            self.config.weight_video * loss_video
            + self.config.weight_tactile * loss_gel
            + self.config.weight_lowdim * loss_lowdim
            + sync_weight * loss_sync
        )

        # Key set is fixed for every mode and every rank; see the module docstring.
        extras = {
            "loss_video": loss_video.detach(),
            "loss_gel": loss_gel.detach(),
            "loss_lowdim_raw": raw_lowdim.detach(),
            "loss_lowdim_scaled": loss_lowdim.detach(),
            "loss_sync": loss_sync.detach(),
            "loss_sync_a": loss_a.detach(),
            "loss_sync_b": loss_b.detach(),
            "sync_weight": torch.tensor(sync_weight, device=device),
            "lowdim_gate_frac": lowdim_gate_frac.detach(),
            "lowdim_scale": self.lowdim_scale.detach().clone(),
            **{key: value.detach() for key, value in metrics_a.items()},
            **{key: value.detach() for key, value in metrics_b.items()},
        }
        return total, extras
