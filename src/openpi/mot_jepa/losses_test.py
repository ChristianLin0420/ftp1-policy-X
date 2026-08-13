from __future__ import annotations

import json

import pytest
import torch

from openpi.mot_jepa.config import _from_dict
from openpi.mot_jepa.ema import EmaTeacher
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.losses import LossConfig
from openpi.mot_jepa.losses import MotJepaLoss
from openpi.mot_jepa.losses import _center_by_slot
from openpi.mot_jepa.losses import jepa_direction_loss
from openpi.mot_jepa.losses import jepa_regression_loss
from openpi.mot_jepa.losses import latent_loss
from openpi.mot_jepa.losses import normalize_targets
from openpi.mot_jepa.losses import sync_loss_level_a
from openpi.mot_jepa.losses import sync_loss_level_b
from openpi.mot_jepa.masking import DEFAULT_MODE_PROBS
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import build_batch_masks
from openpi.mot_jepa.model import ClipInputs
from openpi.mot_jepa.model import MotJepaStudent
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.mot_encoder_test import TINY_LAYOUT
from openpi.mot_jepa.mot_encoder_test import TINY_ROPE
from openpi.mot_jepa.predictor import MoTPredictorConfig

BATCH = 4
ALL_MODES = list(MaskMode)


def tiny_spec(mode: MaskMode | None = None) -> MaskSpec:
    probs = list(DEFAULT_MODE_PROBS)
    if mode is not None:
        probs = [0.0] * len(MaskMode)
        probs[int(mode)] = 1.0
    return MaskSpec(
        layout=TINY_LAYOUT,
        mode_probs=tuple(probs),
        tactile_window_steps=(1,),
        video_window_steps=(1,),
        # Scaled to TINY_LAYOUT exactly as the windows above are. num_steps is 2 here, so the
        # production defaults (horizon 2-3, min_context 4) cannot fit -- and should not: the
        # guard that rejects them is the one that keeps a real run from starving its own context.
        forecast_horizon_steps=(1,),
        min_context_steps=1,
        min_targets_per_stream=2,
    )


def tiny_student() -> MotJepaStudent:
    torch.manual_seed(0)
    return MotJepaStudent(
        TINY_LAYOUT,
        MoTEncoderConfig(depth=3, num_local_layers=1, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE),
        MoTPredictorConfig(depth=2, width=16, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE),
    )


def tiny_clip() -> ClipInputs:
    generator = torch.Generator().manual_seed(1)
    return ClipInputs(
        video=torch.randn(BATCH, TINY_LAYOUT.num_frames, 3, 32, 32, generator=generator),
        gel=torch.randn(BATCH, TINY_LAYOUT.num_frames, 2, 3, 32, 32, generator=generator),
        lowdim=torch.randn(BATCH, TINY_LAYOUT.num_frames, TINY_LAYOUT.lowdim_slots, 1, generator=generator),
    )


def run_step(mode: MaskMode, step: int = 10_000) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    spec = tiny_spec(mode)
    masks = build_batch_masks(spec, step=0, batch_size=BATCH, base_seed=3)
    student = tiny_student()
    teacher = EmaTeacher(student.backbone, runtime_dtype=torch.float32)
    loss_fn = MotJepaLoss(LossConfig(), TINY_LAYOUT)

    clip = tiny_clip()
    with torch.no_grad():
        full = teacher.module.encode_full(clip)
    targets = []
    for expert, tokens in enumerate(full.tokens):
        offset = 0 if expert == 0 else TINY_LAYOUT.num_video_tokens
        lo, hi = (int(v) for v in masks.tgt_expert_bounds[expert])
        index = masks.tgt_index[:, lo:hi] - offset
        gathered = tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1]))
        targets.append(normalize_targets(gathered))

    out = student(clip, masks)
    return loss_fn(out.predictions, targets, out.sync_readout, masks, step)


@pytest.mark.parametrize("mode", ALL_MODES, ids=[m.name for m in ALL_MODES])
def test_loss_key_set_is_identical_across_every_mode(mode):
    """Ranks stack the aux dict into one tensor; a differing key set hangs DDP."""
    reference = set(run_step(MaskMode.V)[1])
    assert set(run_step(mode)[1]) == reference


@pytest.mark.parametrize("mode", ALL_MODES, ids=[m.name for m in ALL_MODES])
def test_loss_is_finite_and_backpropagates_in_every_mode(mode):
    loss, extras = run_step(mode)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in extras.values())


def test_every_expert_receives_gradient_except_tactile_encoder_under_t_hard():
    """T_HARD deliberately removes all tactile input, so its encoder gets no gradient.

    Documented here because it is the sole reason the trainer needs
    ``find_unused_parameters=True``.
    """
    spec = tiny_spec(MaskMode.T_HARD)
    masks = build_batch_masks(spec, step=0, batch_size=BATCH, base_seed=3)
    student = tiny_student()
    teacher = EmaTeacher(student.backbone, runtime_dtype=torch.float32)
    loss_fn = MotJepaLoss(LossConfig(), TINY_LAYOUT)
    clip = tiny_clip()

    with torch.no_grad():
        full = teacher.module.encode_full(clip)
    targets = []
    for expert, tokens in enumerate(full.tokens):
        offset = 0 if expert == 0 else TINY_LAYOUT.num_video_tokens
        lo, hi = (int(v) for v in masks.tgt_expert_bounds[expert])
        index = masks.tgt_index[:, lo:hi] - offset
        targets.append(normalize_targets(tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1]))))

    out = student(clip, masks)
    loss, _ = loss_fn(out.predictions, targets, out.sync_readout, masks, 10_000)
    loss.backward()

    gel_conv = student.backbone.embed.gel.proj.weight
    video_conv = student.backbone.embed.video.proj.weight
    assert video_conv.grad is not None
    assert gel_conv.grad is None or torch.equal(gel_conv.grad, torch.zeros_like(gel_conv.grad))


def test_sync_level_a_separates_matched_from_mismatched_pairs():
    """The design rule made concrete: matched pairs must score strictly better."""
    torch.manual_seed(0)
    video = torch.randn(16, 32)
    matched, _ = sync_loss_level_a(video, video.clone(), temperature=0.07, gather=False)
    shuffled, _ = sync_loss_level_a(video, video[torch.randperm(16)], temperature=0.07, gather=False)
    assert matched < shuffled


def test_sync_level_a_at_chance_for_independent_streams():
    torch.manual_seed(0)
    _, metrics = sync_loss_level_a(torch.randn(64, 32), torch.randn(64, 32), temperature=0.07, gather=False)
    assert float(metrics["sync_a_acc"]) < 0.2
    assert float(metrics["sync_a_chance"]) == pytest.approx(1 / 64)


def test_sync_level_b_collapses_to_chance_when_the_time_axis_is_shuffled():
    """Level B can only be solved by temporal correspondence, so destroying it must hurt."""
    torch.manual_seed(0)
    video = torch.randn(8, 6, 32)
    aligned, aligned_metrics = sync_loss_level_b(video, video.clone(), temperature=0.07)
    shuffled_tactile = video[:, torch.randperm(6)]
    shuffled, shuffled_metrics = sync_loss_level_b(video, shuffled_tactile, temperature=0.07)
    assert aligned < shuffled
    assert float(aligned_metrics["sync_b_acc"]) > float(shuffled_metrics["sync_b_acc"])


def test_center_by_slot_removes_the_per_slot_temporal_mean():
    """Centering is what stops a per-slot bias from scoring well on interpolable channels."""
    values = torch.tensor([[[1.0], [3.0], [10.0], [20.0]]])
    slot_ids = torch.tensor([[0, 0, 1, 1]])
    centered = _center_by_slot(values, slot_ids, num_slots=2)
    torch.testing.assert_close(centered.squeeze(-1), torch.tensor([[-1.0, 1.0, -5.0, 5.0]]))


def test_predicting_the_slot_mean_scores_zero_after_centering():
    """The anti-collapse property: emitting a per-slot bias earns nothing."""
    target = torch.tensor([[[1.0], [3.0], [10.0], [20.0]]])
    slot_ids = torch.tensor([[0, 0, 1, 1]])
    mean_prediction = torch.tensor([[[2.0], [2.0], [15.0], [15.0]]])

    centered_pred = _center_by_slot(mean_prediction, slot_ids, num_slots=2)
    centered_tgt = _center_by_slot(target, slot_ids, num_slots=2)

    # The mean-predictor collapses to exactly zero ...
    torch.testing.assert_close(centered_pred, torch.zeros_like(centered_pred))
    # ... while the real target retains its time-varying content, so the residual is large.
    assert float(centered_tgt.abs().mean()) > 0.0
    assert float(jepa_regression_loss(centered_pred, centered_tgt)) == pytest.approx(3.0)


def test_sync_warmup_ramps_from_zero():
    loss_fn = MotJepaLoss(LossConfig(weight_sync=0.2, sync_warmup_steps=1000), TINY_LAYOUT)
    assert loss_fn.sync_weight(0) == 0.0
    assert loss_fn.sync_weight(500) == pytest.approx(0.1)
    assert loss_fn.sync_weight(1000) == pytest.approx(0.2)
    assert loss_fn.sync_weight(50_000) == pytest.approx(0.2)


def test_lowdim_self_normalization_tracks_the_raw_magnitude():
    """The term must contribute a fixed share of gradient however easy it becomes."""
    loss_fn = MotJepaLoss(LossConfig(), TINY_LAYOUT)
    loss_fn.train()
    before = float(loss_fn.lowdim_scale)
    _, extras = run_step(MaskMode.T_HARD)
    assert before == pytest.approx(1.0)
    assert float(extras["loss_lowdim_scaled"]) > 0.0


def test_normalize_targets_is_fp32_and_zero_mean():
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    out = normalize_targets(x)
    assert out.dtype is torch.float32
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5, rtol=0)


def test_target_gathering_covers_every_stream():
    spec = tiny_spec(MaskMode.X)
    masks = build_batch_masks(spec, step=0, batch_size=BATCH, base_seed=3)
    for stream in (StreamId.VIDEO, StreamId.GEL, StreamId.LOWDIM):
        lo, hi = (int(v) for v in masks.tgt_bounds[int(stream)])
        assert hi > lo


# --------------------------------------------------------------------------------------
# Selectable latent objective
# --------------------------------------------------------------------------------------


def test_direction_loss_is_zero_on_a_perfect_prediction():
    torch.manual_seed(0)
    target = normalize_targets(torch.randn(4, 7, 32))
    assert float(jepa_direction_loss(target.clone(), target)) == pytest.approx(0.0, abs=1e-5)


def test_direction_loss_penalises_an_inverted_prediction():
    torch.manual_seed(0)
    target = normalize_targets(torch.randn(4, 7, 32))
    # Anti-parallel but same norm: direction term is 2, Huber term is 0.
    assert float(jepa_direction_loss(-target, target)) == pytest.approx(2.0, abs=1e-4)


def test_direction_loss_ignores_prediction_scale_but_not_direction():
    """Scale is uninformative -- LayerNorm fixes every target norm to sqrt(D) -- so only the
    Huber anchor should react to it, and the cosine part should not."""
    torch.manual_seed(0)
    target = normalize_targets(torch.randn(4, 7, 32))
    plain = jepa_direction_loss(target * 3.0, target, beta=0.0)
    assert float(plain) == pytest.approx(0.0, abs=1e-5)
    anchored = jepa_direction_loss(target * 3.0, target, beta=1.0)
    assert float(anchored) > 1.0, "the Huber anchor must react to a 3x norm"


def test_direction_loss_is_immune_to_a_constant_shift():
    """The target is centred by LayerNorm; a DC offset on the prediction carries no signal and
    would otherwise depress the cosine purely by inflating its norm."""
    torch.manual_seed(0)
    target = normalize_targets(torch.randn(4, 7, 32))
    shifted = target + 5.0
    assert float(jepa_direction_loss(shifted, target, beta=0.0)) == pytest.approx(0.0, abs=1e-5)


def test_direction_loss_honours_the_valid_mask():
    torch.manual_seed(0)
    target = normalize_targets(torch.randn(4, 6, 16))
    prediction = target.clone()
    prediction[:, 3:] = -prediction[:, 3:]  # corrupt the second half
    valid = torch.zeros(4, 6, dtype=torch.bool)
    valid[:, :3] = True
    assert float(jepa_direction_loss(prediction, target, valid)) == pytest.approx(0.0, abs=1e-5)


def test_empty_selection_returns_zero_for_both_objectives():
    empty = torch.zeros(2, 0, 8)
    assert float(jepa_direction_loss(empty, empty)) == 0.0
    assert float(jepa_regression_loss(empty, empty)) == 0.0


def test_latent_loss_dispatches_and_refuses_an_unknown_name():
    torch.manual_seed(0)
    target = normalize_targets(torch.randn(2, 5, 16))
    prediction = torch.randn(2, 5, 16)
    assert torch.equal(latent_loss("l1", prediction, target), jepa_regression_loss(prediction, target))
    assert torch.equal(latent_loss("direction", prediction, target), jepa_direction_loss(prediction, target))
    with pytest.raises(ValueError, match="unknown objective"):
        latent_loss("cosine", prediction, target)


def test_a_frozen_config_without_the_new_keys_rebuilds_as_l1():
    """probe2 is mid-flight with a run_config.json predating these fields. A requeue must
    reconstruct exactly the behaviour it has been training with, not silently switch objective.
    """
    frozen = json.dumps({"weight_video": 1.0, "weight_tactile": 1.0})
    rebuilt = _from_dict(LossConfig, json.loads(frozen))
    assert rebuilt.video_objective == "l1"
    assert rebuilt.gel_objective == "l1"
