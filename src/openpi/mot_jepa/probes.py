"""The four probes that decide whether pretraining actually worked.

V-JEPA 2 reports that its training loss is uncorrelated with downstream accuracy, and a
*lower* reconstruction loss can indicate a more collapsed representation. So model selection
here is driven by these probes, never by the loss curve.

* **P1 cross-modal retrieval** is the primary metric and the cleanest number available. An
  FTP-1 encoder should sit at chance (0.39% among 256 candidates); a bound representation
  should be far above it. It measures binding directly, weeks before any closed-loop result.
* **P2 positional-shortcut control** asks whether P1 is real. Tactile *content* is replaced
  by its own clip-mean while positions are left untouched. If retrieval survives that, the
  head is reading position rather than touch, and the run must halt.
* **P3 donor ratio** measures binding inside the JEPA path: the prediction error against a
  donor clip's targets divided by the error against the true ones. At exactly 1.0 the
  predictor is emitting something clip-independent.
* **P4 RankMe** is a collapse detector, not a binding measure.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.losses import normalize_targets
from openpi.mot_jepa.model import ClipInputs


def rankme(features: torch.Tensor, *, eps: float = 1e-7) -> float:
    """Effective rank ``exp(H(sigma_hat))`` over normalized singular values.

    Needs roughly ``N >= 4D`` rows to mean anything, so callers should accumulate across
    several batches before trusting it.
    """
    matrix = features.detach().to(torch.float32)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return float("nan")
    try:
        singular = torch.linalg.svdvals(matrix)
    except RuntimeError:  # pragma: no cover - degenerate batch
        return float("nan")
    total = singular.sum()
    if not torch.isfinite(total) or total <= 0:
        return float("nan")
    probabilities = singular / total + eps
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.exp())


def cross_modal_retrieval(video: torch.Tensor, tactile: torch.Tensor, *, ks: tuple[int, ...] = (1, 5)) -> dict:
    """Top-k accuracy of matching each clip's video to its own tactile among the batch."""
    video = F.normalize(video.to(torch.float32), dim=-1)
    tactile = F.normalize(tactile.to(torch.float32), dim=-1)
    num = video.shape[0]
    if num < 2:
        return {f"retrieval_top{k}": float("nan") for k in ks} | {"retrieval_chance": float("nan")}

    similarity = video @ tactile.t()
    labels = torch.arange(num, device=video.device)
    ranking = similarity.argsort(dim=-1, descending=True)
    out = {}
    for k in ks:
        hits = (ranking[:, : min(k, num)] == labels[:, None]).any(dim=-1)
        out[f"retrieval_top{k}"] = float(hits.float().mean())
    out["retrieval_chance"] = 1.0 / num
    return out


def donor_ratio(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """``||pred - donor_target|| / ||pred - true_target||``.

    Rolling the batch by one gives a donor drawn from the same marginal. A value at 1.0 means
    the prediction is as close to another clip's contact state as to its own -- exactly the
    ``shuffle == real`` signature, measured inside the JEPA path.
    """
    prediction = prediction.to(torch.float32)
    target = target.to(torch.float32)
    if prediction.shape[0] < 2 or prediction.numel() == 0:
        return float("nan")
    true_error = (prediction - target).abs().mean()
    donor_error = (prediction - target.roll(1, dims=0)).abs().mean()
    if not torch.isfinite(true_error) or true_error <= 0:
        return float("nan")
    return float(donor_error / true_error)


def _flatten_clip_mean(tensor: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
    """Replace content with its temporal mean, broadcast back to the original shape."""
    return tensor.mean(dim=time_dim, keepdim=True).expand_as(tensor).contiguous()


def _permute_time(tensor: torch.Tensor, *, seed: int, time_dim: int = 1) -> torch.Tensor:
    """Shuffle frames within each clip, leaving the multiset of frames untouched.

    The marginal statistics of the clip are preserved *exactly* -- same frames, same mean,
    same variance, same histogram -- and only the ordering is destroyed. That is the contrast
    the clip-mean control cannot draw, and it is the one the thesis needs: the FTP-1 ablation
    found ``shuffle ~ real``, meaning the policy read tactile marginals rather than
    correspondence, and this control asks the same question of the encoder.

    A fixed permutation per call (not per sample) so the number is comparable across steps.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(tensor.shape[time_dim], generator=generator).to(tensor.device)
    return tensor.index_select(time_dim, order).contiguous()


def _between_clip_dispersion(tokens: torch.Tensor) -> float:
    """How much clips differ from each other, per channel. Collapse drives this to zero.

    Deliberately a *between-sample* statistic. Every within-token measure is blind here: the
    encoder ends in a learned LayerNorm, so the std inside a token reports the norm's gain, not
    the representation. Averaging tokens per clip first and then taking the spread across the
    batch is untouched by that, because LayerNorm normalizes across channels rather than across
    samples.
    """
    if tokens.shape[0] < 2:
        return float("nan")
    per_clip = tokens.float().mean(dim=1)
    return float(per_clip.std(dim=0).mean())


def _gather_targets(tokens: torch.Tensor, masks, layout: TokenLayout, expert: int) -> torch.Tensor:
    """Slice a teacher's full-clip output down to the masked target positions.

    Mirrors ``scripts/mot_jepa_train.gather_targets``; duplicated here rather than imported
    because ``probes`` must not depend on the training entrypoint.
    """
    offset = 0 if expert == 0 else layout.num_video_tokens
    lo, hi = (int(v) for v in masks.tgt_expert_bounds[expert])
    index = (masks.tgt_index[:, lo:hi] - offset).to(tokens.device)
    return tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1]))


class ProbeSuite:
    """Runs the four probes on a held-out batch."""

    #: P2 must stay at or below 1.5x chance. Above that, retrieval is reading position.
    SHORTCUT_TOLERANCE = 1.5

    def __init__(self, layout: TokenLayout, *, max_batches: int = 1, use_autocast: bool | None = None) -> None:
        self.layout = layout
        self.max_batches = max_batches
        #: ``None`` means "autocast iff CUDA". Tests force it on to exercise the mixed-dtype
        #: path on CPU, which is otherwise unreachable.
        self.use_autocast = use_autocast
        #: Tensors from the most recent run, kept only when ``collect_panels=True`` so a
        #: caller can render figures without paying for a second forward pass.
        self.panels: dict[str, torch.Tensor] = {}

    @staticmethod
    def _pool(encoder_output, projectors) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip-level video and tactile vectors, in the shared synchrony-head space.

        Two reasons for the layer-S readout rather than the final tokens: after the global
        layers each modality already contains the other, so retrieval on final tokens would
        be partly self-matching; and the experts have different residual widths (768 vs 384),
        so they are only directly comparable once the projectors have mapped both into one
        space. Those are the same projectors the synchrony loss trains, which makes this
        measurement of exactly what the objective learned rather than of a proxy.
        """
        video, tactile = encoder_output.sync_readout
        if projectors is not None:
            video = projectors[0](video)
            tactile = projectors[1](tactile)
        return video.mean(dim=1), tactile.mean(dim=1)

    @torch.no_grad()
    def run(
        self,
        student,
        teacher,
        loader_iter,
        device,
        *,
        layout: TokenLayout | None = None,
        projectors=None,
        masks=None,
        collect_panels: bool = False,
    ) -> dict:
        """Compute all probes on the next batch from ``loader_iter``.

        Consumes a training batch rather than holding a separate loader, which keeps the
        probe cheap; the batch is never trained on because this runs under ``no_grad`` and
        after the optimizer step.
        """
        layout = layout or self.layout
        was_training = student.training
        student.eval()
        metrics: dict[str, float] = {}
        try:
            batch = next(loader_iter)
            inputs = ClipInputs(
                video=batch["video"].to(device).float().div_(127.5).sub_(1.0),
                gel=batch["gel"].to(device).float().div_(127.5).sub_(1.0),
                lowdim=batch["lowdim"].to(device).float(),
            )

            # Every forward AND every projection must sit inside autocast. `encode_full`
            # under autocast returns bf16 readouts while the projector weights stay fp32, so
            # applying the projectors outside raises "mat1 and mat2 must have the same
            # dtype". `amp` is device-generic rather than hardcoded to CUDA precisely so a
            # CPU test can reproduce that -- with autocast keyed to "cuda" the bug is
            # invisible off-GPU, which is how it reached a real job.
            enabled = device.type == "cuda" if self.use_autocast is None else self.use_autocast

            def amp():
                return torch.autocast(device.type, torch.bfloat16, enabled=enabled)

            with amp():
                encoded = student.backbone.encode_full(inputs)
                video_vec, tactile_vec = self._pool(encoded, projectors)
            metrics.update(cross_modal_retrieval(video_vec, tactile_vec))

            if collect_panels:
                with amp():
                    steps_v = projectors[0](encoded.sync_readout[0]) if projectors else encoded.sync_readout[0]
                    steps_t = projectors[1](encoded.sync_readout[1]) if projectors else encoded.sync_readout[1]
                self.panels = {
                    "video_vec": video_vec.detach().float().cpu(),
                    "tactile_vec": tactile_vec.detach().float().cpu(),
                    "video_steps": steps_v[0].detach().float().cpu(),
                    "tactile_steps": steps_t[0].detach().float().cpu(),
                    "video_frames": batch["video"][0].cpu(),
                    "gel_frames": batch["gel"][0].cpu(),
                }

            # P2: identical positions, constant content. Retrieval must collapse to chance.
            control_inputs = ClipInputs(
                video=inputs.video,
                gel=_flatten_clip_mean(inputs.gel),
                lowdim=_flatten_clip_mean(inputs.lowdim),
            )
            with amp():
                control = student.backbone.encode_full(control_inputs)
                control_video, control_tactile = self._pool(control, projectors)
            control_metrics = cross_modal_retrieval(control_video, control_tactile, ks=(1,))
            metrics["shortcut_top1"] = control_metrics["retrieval_top1"]
            chance = control_metrics["retrieval_chance"]
            metrics["shortcut_ratio_to_chance"] = (
                control_metrics["retrieval_top1"] / chance if chance > 0 else float("nan")
            )
            metrics["shortcut_tripped"] = float(metrics["shortcut_ratio_to_chance"] > self.SHORTCUT_TOLERANCE)
            # The headline is the gap, not the raw number.
            metrics["retrieval_gap"] = metrics.get("retrieval_top1", float("nan")) - metrics["shortcut_top1"]

            # P2b: the control that actually tests BINDING. Permuting tactile along time within
            # each clip preserves that clip's marginal statistics *exactly* and destroys only
            # the timing, so retrieval that survives it is reading marginals -- which is the
            # `shuffle == real` failure this design exists to defeat. The clip-mean control
            # above cannot draw that distinction: it changes the marginals too, and each clip's
            # mean still differs, so across-sample retrieval can legitimately survive it.
            shuffle_inputs = ClipInputs(
                video=inputs.video,
                gel=_permute_time(inputs.gel, seed=0),
                lowdim=_permute_time(inputs.lowdim, seed=0),
            )
            with amp():
                shuffled = student.backbone.encode_full(shuffle_inputs)
                shuffle_video, shuffle_tactile = self._pool(shuffled, projectors)
            shuffle_metrics = cross_modal_retrieval(shuffle_video, shuffle_tactile, ks=(1,))
            metrics["timeshuffle_top1"] = shuffle_metrics["retrieval_top1"]
            metrics["timeshuffle_gap"] = metrics.get("retrieval_top1", float("nan")) - metrics["timeshuffle_top1"]

            with amp():
                teacher_out = teacher.module.encode_full(inputs)

            # P3 measured on the PREDICTOR, as documented. Comparing the student's own encoding
            # to the EMA teacher's encoding of the same clip -- which is what this used to do --
            # asks whether an EMA pair agrees with itself. It does, by construction, so it read
            # 10.7 and rising while saying nothing about the JEPA path.
            metrics["donor_ratio"] = float("nan")
            metrics["donor_ratio_video"] = float("nan")
            if masks is not None:
                with amp():
                    out = student(inputs, masks)
                targets = [
                    normalize_targets(_gather_targets(tokens, masks, layout, expert))
                    for expert, tokens in enumerate(teacher_out.tokens)
                ]
                metrics["donor_ratio"] = donor_ratio(out.predictions[1].to(torch.float32), targets[1])
                metrics["donor_ratio_video"] = donor_ratio(out.predictions[0].to(torch.float32), targets[0])

            metrics["rankme_video"] = rankme(encoded.tokens[0].flatten(0, 1))
            metrics["rankme_tactile"] = rankme(encoded.tokens[1].flatten(0, 1))
            # Collapse here means every CLIP mapping to the same representation, which is a
            # between-sample property. The old `teacher_target_std` measured the std *within* a
            # token, and no placement of that measurement can work: MoTEncoder ends in a learned
            # LayerNorm (`mot_encoder.py:261`), so its output std is ~||gamma|| regardless of
            # what the residual stream does. It read 0.9999931 for 14k steps because it was
            # reporting the norm's gain, not the representation.
            metrics["teacher_dispersion"] = _between_clip_dispersion(teacher_out.tokens[1])
            metrics["student_dispersion"] = _between_clip_dispersion(encoded.tokens[1])
        except StopIteration:  # pragma: no cover - infinite sampler makes this unreachable
            return metrics
        finally:
            if was_training:
                student.train()
        return metrics


@contextlib.contextmanager
def evaluating(module):
    """Temporarily switch a module to eval mode and restore the previous state."""
    was_training = module.training
    module.eval()
    try:
        yield module
    finally:
        if was_training:
            module.train()
