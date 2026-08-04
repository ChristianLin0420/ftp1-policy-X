"""Probe coverage, including the mixed-precision path that only bites on GPU.

The autocast test below is the important one. `encode_full` under autocast returns bf16
readouts while the projector weights stay fp32, so applying the projectors outside the
autocast context raises "mat1 and mat2 must have the same dtype". With autocast keyed to
``device.type == "cuda"`` that is *structurally unreachable* on CPU, which is how it survived
a full CPU test suite and only surfaced on a real 8-GPU job. Forcing autocast on for CPU
reproduces it in under a second.
"""

from __future__ import annotations

import pytest
import torch

from openpi.mot_jepa.ema import EmaTeacher
from openpi.mot_jepa.losses import LossConfig
from openpi.mot_jepa.losses import MotJepaLoss
from openpi.mot_jepa.losses import normalize_targets
from openpi.mot_jepa.model import MotJepaStudent
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.mot_encoder_test import TINY_LAYOUT
from openpi.mot_jepa.mot_encoder_test import TINY_ROPE
from openpi.mot_jepa.predictor import MoTPredictorConfig
from openpi.mot_jepa.probes import ProbeSuite
from openpi.mot_jepa.probes import _between_clip_dispersion
from openpi.mot_jepa.probes import _permute_time
from openpi.mot_jepa.probes import cross_modal_retrieval
from openpi.mot_jepa.probes import donor_ratio
from openpi.mot_jepa.probes import rankme

BATCH = 4


def tiny_student() -> MotJepaStudent:
    torch.manual_seed(0)
    return MotJepaStudent(
        TINY_LAYOUT,
        MoTEncoderConfig(depth=2, num_local_layers=1, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE),
        MoTPredictorConfig(depth=1, width=16, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE),
    )


def batch_iter():
    generator = torch.Generator().manual_seed(1)
    while True:
        yield {
            "video": (torch.rand(BATCH, TINY_LAYOUT.num_frames, 3, 32, 32, generator=generator) * 255).to(torch.uint8),
            "gel": (torch.rand(BATCH, TINY_LAYOUT.num_frames, 2, 3, 32, 32, generator=generator) * 255).to(torch.uint8),
            "lowdim": torch.randn(BATCH, TINY_LAYOUT.num_frames, TINY_LAYOUT.lowdim_slots, 1, generator=generator),
        }


@pytest.mark.parametrize("use_autocast", [False, True], ids=["fp32", "autocast_bf16"])
def test_probe_suite_runs_under_both_precisions(use_autocast):
    """Regression guard for the GPU-only dtype mismatch between readouts and projectors."""
    student = tiny_student()
    teacher = EmaTeacher(student.backbone, runtime_dtype=torch.float32)
    loss_fn = MotJepaLoss(LossConfig(), TINY_LAYOUT)
    probes = ProbeSuite(TINY_LAYOUT, use_autocast=use_autocast)

    metrics = probes.run(student, teacher, batch_iter(), torch.device("cpu"), projectors=loss_fn.projectors)
    expected = {
        "retrieval_top1",
        "retrieval_top5",
        "retrieval_chance",
        "shortcut_top1",
        "shortcut_ratio_to_chance",
        "shortcut_tripped",
        "retrieval_gap",
        "donor_ratio",
        "rankme_video",
        "rankme_tactile",
        "teacher_dispersion",
    }
    assert expected <= set(metrics)
    assert all(isinstance(value, float) for value in metrics.values())


def test_probe_restores_training_mode():
    student = tiny_student()
    student.train()
    teacher = EmaTeacher(student.backbone, runtime_dtype=torch.float32)
    loss_fn = MotJepaLoss(LossConfig(), TINY_LAYOUT)
    ProbeSuite(TINY_LAYOUT).run(student, teacher, batch_iter(), torch.device("cpu"), projectors=loss_fn.projectors)
    assert student.training, "probe must not leave the model in eval mode"


def test_retrieval_is_perfect_when_the_two_views_agree():
    torch.manual_seed(0)
    vectors = torch.randn(16, 32)
    out = cross_modal_retrieval(vectors, vectors.clone())
    assert out["retrieval_top1"] == 1.0
    assert out["retrieval_chance"] == pytest.approx(1 / 16)


def test_retrieval_is_near_chance_for_independent_views():
    torch.manual_seed(0)
    out = cross_modal_retrieval(torch.randn(64, 32), torch.randn(64, 32))
    assert out["retrieval_top1"] < 0.15


def test_retrieval_degenerates_safely_on_a_single_sample():
    out = cross_modal_retrieval(torch.randn(1, 8), torch.randn(1, 8))
    assert out["retrieval_top1"] != out["retrieval_top1"]  # NaN


def test_donor_ratio_is_one_when_the_prediction_ignores_its_clip():
    """1.0 is the shuffle-equals-real signature, measured inside the JEPA path."""
    target = torch.randn(8, 4, 16)
    constant = torch.zeros_like(target)  # clip-independent prediction
    assert donor_ratio(constant, target) == pytest.approx(1.0, abs=0.25)


def test_donor_ratio_exceeds_one_when_the_prediction_tracks_its_own_clip():
    target = torch.randn(8, 4, 16)
    assert donor_ratio(target + 0.01 * torch.randn_like(target), target) > 2.0


def test_rankme_is_low_for_a_collapsed_matrix_and_high_for_a_full_rank_one():
    collapsed = torch.ones(64, 16) * torch.randn(1, 16)
    assert rankme(collapsed) < 2.0
    assert rankme(torch.randn(256, 16)) > 8.0


def test_rankme_handles_degenerate_input():
    assert rankme(torch.zeros(0, 4)) != rankme(torch.zeros(0, 4)) or True  # must not raise
    assert rankme(torch.randn(4)) != rankme(torch.randn(4)) or True


# --------------------------------------------------------------------------------------
# Repaired probes
# --------------------------------------------------------------------------------------


def test_time_permutation_preserves_marginals_exactly():
    """That preservation is the whole point: it isolates timing from content.

    The clip-mean control changes the marginals as well, so retrieval surviving it is
    ambiguous. Retrieval surviving THIS is unambiguous -- nothing but the ordering changed.
    """
    torch.manual_seed(0)
    clip = torch.randn(3, 8, 2, 3, 4, 4)
    shuffled = _permute_time(clip, seed=0)

    torch.testing.assert_close(clip.mean(dim=1), shuffled.mean(dim=1))
    torch.testing.assert_close(clip.std(dim=1), shuffled.std(dim=1))
    # Same multiset of frames per clip, in a different order.
    torch.testing.assert_close(clip.sort(dim=1).values, shuffled.sort(dim=1).values)
    assert not torch.equal(clip, shuffled), "the ordering must actually change"


def test_time_permutation_is_stable_across_calls():
    """A control that reshuffles every step measures its own noise, not the model."""
    torch.manual_seed(0)
    clip = torch.randn(2, 8, 4)
    assert torch.equal(_permute_time(clip, seed=0), _permute_time(clip, seed=0))


def test_no_within_token_std_can_detect_collapse():
    """Why the collapse detector had to change shape, not just move.

    MoTEncoder ends in a learned LayerNorm, so anything measured *inside* a token is
    re-standardised on the way out and reports the norm's gain. Even the parameter-free
    normalize_targets only attenuates once the per-token std nears its 1e-5 eps: a
    hundred-fold shrink (5.0 -> 0.05) moves it 0.2%. The live run read 0.9999931 -> 0.9999913
    across 14k steps -- 2e-6 of movement.
    """
    healthy = torch.randn(4, 16, 32) * 5.0
    shrinking = torch.randn(4, 16, 32) * 0.05  # 100x smaller, far above the eps floor
    assert float(normalize_targets(healthy).std()) == pytest.approx(1.0, abs=5e-3)
    assert float(normalize_targets(shrinking).std()) == pytest.approx(1.0, abs=5e-3)


def test_between_clip_dispersion_detects_collapse_that_std_cannot():
    """The failure worth catching is every clip mapping to the same representation."""
    torch.manual_seed(0)
    healthy = torch.randn(8, 16, 32)
    # Collapsed: identical per-clip content, with only within-clip token noise left.
    collapsed = torch.randn(1, 16, 32).expand(8, -1, -1) + 1e-4 * torch.randn(8, 16, 32)

    assert _between_clip_dispersion(healthy) > 0.1
    assert _between_clip_dispersion(collapsed) < 1e-3

    # And the point: a within-token std is identical for both, so it sees nothing.
    assert float(normalize_targets(healthy).std()) == pytest.approx(float(normalize_targets(collapsed).std()), abs=1e-2)


def test_donor_ratio_is_one_when_the_prediction_ignores_the_clip():
    """The documented semantics: 1.0 means the prediction is as close to another clip's
    target as to its own."""
    torch.manual_seed(0)
    target = torch.randn(8, 5, 16)
    constant = target.mean(dim=0, keepdim=True).expand_as(target)
    assert donor_ratio(constant, target) == pytest.approx(1.0, abs=0.15)
    # Near-perfect rather than exact: an exact match makes true_error zero, which the
    # div-by-zero guard reports as NaN by design.
    assert donor_ratio(target + 1e-3 * torch.randn_like(target), target) > 5.0
