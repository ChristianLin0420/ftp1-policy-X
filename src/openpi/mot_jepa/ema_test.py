from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from openpi.mot_jepa.ema import EmaTeacher
from openpi.mot_jepa.ema import ema_decay_at


def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))


def _perturb(model: nn.Module, delta: float) -> None:
    """Deterministic stand-in for optimizer steps."""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(delta)


def test_drift_is_zero_at_init_then_strictly_positive_and_rising():
    """A merely-nonzero check would pass on a teacher that moved once and then froze."""
    student = tiny_model()
    teacher = EmaTeacher(student, decay_start=0.99, decay_end=0.999, warmup_steps=10)
    assert teacher.drift_norm(student)[0] == 0.0

    drifts = []
    for step in range(50):
        _perturb(student, 1e-3)
        teacher.update(student, step)
        drifts.append(teacher.drift_norm(student)[0])

    assert drifts[0] > 0.0
    assert drifts[-1] > drifts[0], "EMA drift must keep growing while the student moves"
    assert all(math.isfinite(value) for value in drifts)


def test_fp32_shadow_moves_where_a_bf16_shadow_would_freeze():
    """Names the exact failure mode in the test.

    At decay 0.9999 the update is ``1e-4 * (student - shadow)``. For a relative gap of 1e-3
    that increment is ~1e-7 relative -- below bfloat16's ~2^-8 resolution, so it rounds to
    exactly zero on every step and the teacher stays at its random init forever. The loss
    still decreases (predicting a frozen random projection is learnable), so nothing raises.
    """
    decay = 0.9999
    shadow_value, student_value = 1.0, 1.001

    shadow_bf16 = torch.tensor([shadow_value], dtype=torch.bfloat16)
    student_bf16 = torch.tensor([student_value], dtype=torch.bfloat16)
    for _ in range(10):
        shadow_bf16 = shadow_bf16 + (1 - decay) * (student_bf16 - shadow_bf16)
    assert shadow_bf16.item() == pytest.approx(shadow_value, abs=0.0), "bf16 update should round to zero"

    shadow_fp32 = torch.tensor([shadow_value], dtype=torch.float32)
    student_fp32 = torch.tensor([student_value], dtype=torch.float32)
    for _ in range(10):
        shadow_fp32 = shadow_fp32 + (1 - decay) * (student_fp32 - shadow_fp32)
    assert shadow_fp32.item() > shadow_value, "fp32 update must actually move"


def test_teacher_shadow_is_fp32_even_when_runtime_is_bf16():
    student = tiny_model()
    teacher = EmaTeacher(student, runtime_dtype=torch.bfloat16)
    assert all(tensor.dtype is torch.float32 for tensor in teacher.state_dict().values())
    assert all(param.dtype is torch.bfloat16 for param in teacher.module.parameters())


def test_teacher_moves_at_high_decay_with_a_bf16_runtime_copy():
    """End-to-end version of the freeze test, through the real class."""
    student = tiny_model()
    teacher = EmaTeacher(student, decay_start=0.9999, decay_end=0.9999, warmup_steps=0)
    for step in range(20):
        _perturb(student, 1e-2)
        teacher.update(student, step)
    assert teacher.drift_norm(student)[0] > 0.0
    baseline = teacher.state_dict()["shadow.0"].clone()
    _perturb(student, 1e-2)
    teacher.update(student, 20)
    assert not torch.equal(teacher.state_dict()["shadow.0"], baseline)


def test_decay_schedule_is_a_pure_function_of_step():
    kwargs = {"decay_start": 0.99, "decay_end": 0.9999, "warmup_steps": 1000}
    assert ema_decay_at(1234, **kwargs) == ema_decay_at(1234, **kwargs)
    assert ema_decay_at(0, **kwargs) == pytest.approx(0.99)
    assert ema_decay_at(1000, **kwargs) == pytest.approx(0.9999)
    assert ema_decay_at(10_000, **kwargs) == pytest.approx(0.9999), "must clamp past warmup"
    assert ema_decay_at(500, **kwargs) == pytest.approx(0.99 + 0.5 * (0.9999 - 0.99))


def test_zero_warmup_returns_the_final_decay():
    assert ema_decay_at(0, decay_start=0.1, decay_end=0.9, warmup_steps=0) == 0.9


def test_state_dict_roundtrip_resumes_the_trajectory_bitwise():
    """A requeue must land on the same trajectory, not merely a similar one."""
    student = tiny_model()
    teacher = EmaTeacher(student, decay_start=0.99, decay_end=0.999, warmup_steps=10)
    for step in range(5):
        _perturb(student, 1e-3)
        teacher.update(student, step)

    saved_state = {key: value.clone() for key, value in teacher.state_dict().items()}
    saved_student = {key: value.clone() for key, value in student.state_dict().items()}

    for step in range(5, 10):
        _perturb(student, 1e-3)
        teacher.update(student, step)
    uninterrupted = teacher.state_dict()["shadow.0"].clone()

    resumed_student = tiny_model()
    resumed_student.load_state_dict(saved_student)
    resumed = EmaTeacher(resumed_student, decay_start=0.99, decay_end=0.999, warmup_steps=10)
    resumed.load_state_dict(saved_state)
    for step in range(5, 10):
        _perturb(resumed_student, 1e-3)
        resumed.update(resumed_student, step)

    assert torch.equal(resumed.state_dict()["shadow.0"], uninterrupted)


def test_teacher_parameters_require_no_grad():
    """The teacher must never be DDP-wrapped or receive gradients."""
    student = tiny_model()
    teacher = EmaTeacher(student)
    assert all(not param.requires_grad for param in teacher.module.parameters())
    assert not teacher.module.training


def test_shadow_checksum_changes_as_the_teacher_moves():
    student = tiny_model()
    teacher = EmaTeacher(student, decay_start=0.9, decay_end=0.9, warmup_steps=0)
    before = teacher.shadow_checksum()
    _perturb(student, 1.0)
    teacher.update(student, 0)
    assert not torch.equal(before, teacher.shadow_checksum())


def test_update_rejects_a_mismatched_student():
    student = tiny_model()
    teacher = EmaTeacher(student)
    with pytest.raises(ValueError, match="parameter list changed"):
        teacher.update(nn.Linear(8, 8), 0)
