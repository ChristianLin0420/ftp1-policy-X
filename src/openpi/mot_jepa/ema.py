"""Hand-rolled fp32 EMA teacher.

Two independent reasons this cannot be delegated:

**``config.ema_decay`` is a documented no-op in this repository's PyTorch path.** It is
consumed only by the JAX trainer (``scripts/train.py:171-175``); the PyTorch trainer prints
"EMA is not supported for PyTorch training" (``zarr_train_ftp1_pytorch.py:743``) and moves
on. Setting it and assuming a teacher exists would leave the targets tied to a randomly
initialized network.

**The shadow must be fp32.** bfloat16 carries an 8-bit mantissa, so its relative resolution
is about 2^-8. At decay 0.9999 the per-step update is ``1e-4 * (student - shadow)``; for a
typical relative gap of 1e-3 that increment is ~1e-7 relative, far below bf16's resolution,
and rounds to *exactly zero every step*. The teacher then stays frozen at its random
initialization -- and because predicting a frozen random projection is a learnable task, the
loss still decreases smoothly and nothing raises. Keeping the shadow in fp32 and casting
only the runtime copy is the whole defence. :func:`EmaTeacher.drift_norm` exists so the same
property is monitored live, not merely asserted in a unit test.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


def ema_decay_at(step: int, *, decay_start: float, decay_end: float, warmup_steps: int) -> float:
    """Linearly ramp the decay from ``decay_start`` to ``decay_end`` over ``warmup_steps``.

    A pure function of ``step``, exactly like the repository's LR schedule
    (``zarr_train_ftp1_pytorch.py:638-654``). Nothing about the schedule is stored, so a
    requeue that restores ``global_step`` restores the schedule too, with no extra state to
    get out of sync.
    """
    if warmup_steps <= 0:
        return decay_end
    alpha = min(max(step / warmup_steps, 0.0), 1.0)
    return decay_start + alpha * (decay_end - decay_start)


class EmaTeacher:
    """An exponentially-moving-average copy of a backbone, kept in fp32.

    Not an ``nn.Module``: it must never be registered as a child of the student, or DDP
    would try to reduce gradients for parameters that never receive any.
    """

    def __init__(
        self,
        student: nn.Module,
        *,
        decay_start: float = 0.996,
        decay_end: float = 0.9999,
        warmup_steps: int = 30_000,
        runtime_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.warmup_steps = warmup_steps
        self.runtime_dtype = runtime_dtype

        self.module = copy.deepcopy(student)
        self.module.requires_grad_(False)  # noqa: FBT003 - torch API takes a positional bool
        self.module.eval()
        if device is not None:
            self.module.to(device)

        self._student_params = list(student.parameters())
        # fp32 master copy, independent of whatever dtype the runtime module holds.
        self._shadow = [p.detach().clone().to(torch.float32) for p in self._student_params]
        self._runtime_params = list(self.module.parameters())
        if len(self._shadow) != len(self._runtime_params):
            raise ValueError("teacher and student parameter lists diverged")

        self.module.to(runtime_dtype)
        self._runtime_params = list(self.module.parameters())
        self._write_runtime()
        self.last_decay = decay_start

    # -- update ---------------------------------------------------------------------

    @torch.no_grad()
    def _write_runtime(self) -> None:
        for runtime, shadow in zip(self._runtime_params, self._shadow, strict=True):
            runtime.copy_(shadow)

    @torch.no_grad()
    def update(self, student: nn.Module, step: int) -> float:
        """One EMA step. Call *after* ``optimizer.step()``.

        Returns the decay used, for logging.
        """
        decay = ema_decay_at(
            step, decay_start=self.decay_start, decay_end=self.decay_end, warmup_steps=self.warmup_steps
        )
        params = [p.detach() for p in student.parameters()]
        if len(params) != len(self._shadow):
            raise ValueError("student parameter list changed shape since the teacher was built")
        # shadow <- shadow + (1 - decay) * (student - shadow), fused over all tensors.
        torch._foreach_lerp_(self._shadow, [p.to(torch.float32) for p in params], 1.0 - decay)
        self._write_runtime()
        self.last_decay = decay
        return decay

    # -- telemetry ------------------------------------------------------------------

    @torch.no_grad()
    def drift_norm(self, student: nn.Module) -> tuple[float, float]:
        """``(||shadow - student||_2, ||shadow - student|| / ||student||)``.

        Must be strictly positive and rising early in training. A value pinned at exactly
        zero is the bf16 freeze; a value that stops growing means the student has converged
        or the update is not being called.
        """
        diff_sq = 0.0
        ref_sq = 0.0
        for shadow, param in zip(self._shadow, student.parameters(), strict=True):
            other = param.detach().to(torch.float32)
            diff_sq += float((shadow - other).pow(2).sum())
            ref_sq += float(other.pow(2).sum())
        norm = math.sqrt(diff_sq)
        return norm, norm / math.sqrt(ref_sq) if ref_sq > 0 else 0.0

    @torch.no_grad()
    def shadow_checksum(self) -> torch.Tensor:
        """Cheap scalar fingerprint, used to assert ranks have not silently diverged."""
        total = torch.zeros((), dtype=torch.float64)
        for shadow in self._shadow:
            total += shadow.double().sum().cpu()
        return total

    # -- checkpoint -----------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the **fp32 shadow**, not the bf16 runtime copy.

        Saving the runtime copy would silently quantize the teacher on every requeue, which
        over a 40-job chain compounds into exactly the freeze this class exists to avoid.
        """
        return {f"shadow.{i}": tensor.cpu() for i, tensor in enumerate(self._shadow)}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if len(state) != len(self._shadow):
            raise ValueError(f"expected {len(self._shadow)} shadow tensors, got {len(state)}")
        for i, shadow in enumerate(self._shadow):
            shadow.copy_(state[f"shadow.{i}"].to(shadow.device, torch.float32))
        self._write_runtime()

    def to(self, device: torch.device | str) -> EmaTeacher:
        self.module.to(device)
        self._runtime_params = list(self.module.parameters())
        self._shadow = [tensor.to(device) for tensor in self._shadow]
        self._write_runtime()
        return self
