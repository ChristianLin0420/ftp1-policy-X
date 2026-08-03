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
from openpi.mot_jepa.model import MotJepaStudent
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.mot_encoder_test import TINY_LAYOUT
from openpi.mot_jepa.mot_encoder_test import TINY_ROPE
from openpi.mot_jepa.predictor import MoTPredictorConfig
from openpi.mot_jepa.probes import ProbeSuite
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
        "teacher_target_std",
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
