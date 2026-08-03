"""Test-time tactile ablations -- the measurement this whole design exists to move.

The four conditions and what each leaves intact:

===========  =========================================================================
``real``     everything
``zero``     nothing
``noise``    first and second moments, no structure
``shuffle``  moments **and** full structure, correspondence broken
===========  =========================================================================

The published FTP-1 measurement was ``zero`` = ``noise`` = +22.1% and ``shuffle`` = +0.7%.
Read together those say the policy responds to tactile *structure* but never to *which*
structure -- it has learned ``p(a | phi(tactile))`` for a marginal-statistics summary
``phi``, not a function of the current contact.

Two pre-registered predictions distinguish a bound model:

1. ``shuffle - real >= +10%``. Below +5% falsifies the thesis.
2. ``noise`` and ``zero`` must **separate**, with ordering ``real < zero <= shuffle <=
   noise``. A bound model should be hurt *more* by confidently-wrong evidence than by absent
   evidence, since an all-zeros pad is detectable and can be discounted. That ordering is
   mutually exclusive with the measured one, which is what makes the test sharp.

``donor_at_offset`` turns the single ``shuffle`` number into a curve: a bound model's error
rises monotonically with the donor's temporal offset and saturates at the contact process's
correlation time, while an unbound one is flat.
"""

from __future__ import annotations

import enum

import numpy as np
import torch


class TactileMode(enum.StrEnum):
    REAL = "real"
    ZERO = "zero"
    NOISE = "noise"
    SHUFFLE = "shuffle"


def apply_tactile_ablation(
    tactile: torch.Tensor,
    mode: TactileMode | str,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return an ablated copy of a ``(B, ...)`` tactile tensor.

    Args:
        tactile: Any tactile tensor whose first dimension is the batch.
        mode: One of :class:`TactileMode`.
        generator: Optional RNG, so an evaluation is reproducible.

    ``noise`` matches the per-sample mean and standard deviation of the real signal, so it
    preserves exactly the marginal statistics that ``zero`` destroys -- that is the whole
    point of having both conditions.

    ``shuffle`` permutes across the batch with a derangement, so every clip receives a real
    tactile reading that belongs to a *different* clip: structure and moments intact,
    correspondence destroyed.
    """
    mode = TactileMode(mode)
    if mode is TactileMode.REAL:
        return tactile
    if mode is TactileMode.ZERO:
        return torch.zeros_like(tactile)
    if mode is TactileMode.NOISE:
        flat = tactile.reshape(tactile.shape[0], -1).float()
        mean = flat.mean(dim=1).reshape(-1, *([1] * (tactile.ndim - 1)))
        std = flat.std(dim=1).reshape(-1, *([1] * (tactile.ndim - 1)))
        noise = torch.randn(tactile.shape, generator=generator, device=tactile.device, dtype=torch.float32)
        return (noise * std + mean).to(tactile.dtype)
    if mode is TactileMode.SHUFFLE:
        return tactile[derangement(tactile.shape[0], generator=generator, device=tactile.device)]
    raise ValueError(f"unhandled mode {mode}")  # pragma: no cover


def derangement(size: int, *, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    """A permutation with no fixed point, so no clip keeps its own tactile signal.

    A plain ``randperm`` leaves roughly one clip in the batch correctly paired, which
    contaminates the very comparison being made.
    """
    if size < 2:
        return torch.arange(size, device=device)
    permutation = torch.randperm(size, generator=generator, device=device)
    fixed = permutation == torch.arange(size, device=permutation.device)
    if bool(fixed.any()):
        # Rotating by one is guaranteed fixed-point-free and keeps the draw deterministic.
        permutation = permutation.roll(1)
        still_fixed = permutation == torch.arange(size, device=permutation.device)
        if bool(still_fixed.any()):
            permutation = torch.arange(size, device=permutation.device).roll(1)
    return permutation


def donor_at_offset(sequence: torch.Tensor, offset: int, *, time_dim: int = 0) -> torch.Tensor:
    """Take the donor from the *same* episode at a controlled temporal ``offset``.

    Replacing the random donor with this converts one scalar into a curve with a shape
    prediction, and reads out *what timescale* binding operates on rather than merely
    whether it exists.
    """
    if offset == 0:
        return sequence
    return sequence.roll(shifts=int(offset), dims=time_dim)


def summarize_ablation(results: dict[str, float]) -> dict[str, float | bool | str]:
    """Score a condition -> metric mapping against the pre-registered predictions."""
    real = results.get("real")
    out: dict[str, float | bool | str] = dict(results)
    if real is None or real == 0:
        return out

    for name in ("zero", "noise", "shuffle"):
        if name in results:
            out[f"{name}_pct"] = 100.0 * (results[name] - real) / real

    shuffle_gap = out.get("shuffle_pct")
    if shuffle_gap is not None:
        out["thesis_supported"] = bool(shuffle_gap >= 10.0)
        out["thesis_falsified"] = bool(shuffle_gap < 5.0)

    if "noise" in results and "zero" in results:
        # The signature of reading only marginal statistics is noise == zero to three digits.
        out["noise_zero_separated"] = bool(abs(results["noise"] - results["zero"]) > 1e-3 * abs(real))

    ordered = all(
        results.get(a, -np.inf) <= results.get(b, np.inf)
        for a, b in (("real", "zero"), ("zero", "shuffle"), ("shuffle", "noise"))
        if a in results and b in results
    )
    out["predicted_ordering_holds"] = bool(ordered)
    return out
