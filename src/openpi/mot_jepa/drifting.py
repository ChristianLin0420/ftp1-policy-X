"""Implicit Drifting Policy: one-step action generation via conditional expert geometry.

Follows arXiv 2606.01098 (Yang et al., 2026), which exists to fix a failure the plain Drifting
objective (arXiv 2602.04770) has on behaviour cloning.

**Why not plain Drifting.** Its field needs several real samples sharing a condition. A robot
observation has exactly one expert action chunk, and the paper's Proposition 3.1 shows the
empirical conditional field then evaluates to ``V = a* - a`` exactly, so the objective collapses
into isotropic MSE and every distribution-matching property is lost. A naive port would be
regression wearing a generative costume.

**The fix.** Never materialise a field. Estimate the *local geometry* of expert actions under
observations similar to this one, and minimise an anisotropic quadratic potential directly:

    E_i(a) = 1/2 [ ||a - a*_i||^2  +  (a - a*_i)^T M_i (a - a*_i) ]        (Eq. 5)
    L      = E_i(y_i) + lambda_prox * E_i(z_i)                             (Eq. 6)

``y_i`` is the one-step prediction from pure noise at ``t = 0``; ``z_i`` is an expert-proximal
probe at ``t_* = 0.95``, which is what keeps the geometry meaningful near the data rather than
only at the noise end.

**The sign that matters.** ``s^cond`` and ``s^ref`` in Eq. 9 are *precisions* -- ``(v + eps)^-1/2``
-- each normalised by its across-dimension mean, not variances. So a dimension where neighbouring
experts disagree (high variance, i.e. genuinely multimodal) has *low* precision, the ratio falls
below 1, the ReLU zeroes it, and ``(I + M)`` reduces to the identity: the weakest pull the
objective ever applies. Extra pull is added only where the observation has made a dimension
*tighter than the global prior*. Implementing this with variances instead of precisions inverts
the anisotropy -- strongest pull along the multimodal directions -- which collapses valid modes
toward their mean and looks exactly like ordinary underfitting.
"""

from __future__ import annotations

import dataclasses

import torch

EPS_EMBED = 1e-8
"""Eq. 7 guard on the observation-embedding norm."""

EPS_PRECISION = 1e-6
"""Eq. 9 guard inside the inverse square roots."""


@dataclasses.dataclass(frozen=True)
class DriftingConfig:
    """Table 4 of the paper; identical across all its benchmarks."""

    t_star: float = 0.95
    lambda_prox: float = 0.1


def neighbour_weights(embeddings: torch.Tensor) -> torch.Tensor:
    """``(B, B)`` row-stochastic weights over observation similarity (Eq. 7-8).

    Row-wise standardisation of the cosine similarities *is* the temperature -- the paper has no
    explicit one -- and it is adaptive per row, which is what stops weights collapsing in sparse
    regions of the embedding space.

    ``j = i`` is masked out. The paper's Eq. 8 excludes it to stop the zero-distance term
    collapsing the local variance estimate; its appendix notes the implementation may keep it
    because ``(a*_i - a*_i) = 0`` contributes nothing to ``G_i``. Both give nearly the same ``M``
    after the across-dimension normalisation, and masking is the safer reading of the stated
    intent.
    """
    normed = embeddings / (embeddings.norm(dim=-1, keepdim=True) + EPS_EMBED)
    similarity = normed @ normed.t()
    standardised = (similarity - similarity.mean(dim=-1, keepdim=True)) / (similarity.std(dim=-1, keepdim=True) + EPS_EMBED)
    standardised = standardised.masked_fill(torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device), float("-inf"))
    return torch.softmax(standardised, dim=-1)


def excess_precision(actions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``M_i`` as a per-dimension diagonal (Eq. 8-10). ``actions`` is ``(B, D)``, returns ``(B, D)``.

    Only ``diag(G_i)`` is ever used, so the full ``D x D`` second moment is never formed -- at
    ``H * 120 = 3840`` dimensions that matrix would be 14.7M entries per sample.
    """
    delta = actions.unsqueeze(0) - actions.unsqueeze(1)  # (B_j, B_i, D) -> difference to each query
    v_cond = torch.einsum("ij,jid->id", weights, delta.transpose(0, 1) ** 2)
    v_ref = actions.var(dim=0, unbiased=False)

    # Precisions, each normalised by its across-dimension mean. Without those denominators the
    # ratio carries an arbitrary overall scale and the ReLU's threshold at 1 means nothing.
    p_cond = (v_cond + EPS_PRECISION).rsqrt()
    p_cond = p_cond / p_cond.mean(dim=-1, keepdim=True)
    p_ref = (v_ref + EPS_PRECISION).rsqrt()
    p_ref = p_ref / p_ref.mean()

    return torch.relu(p_cond / (p_ref + EPS_PRECISION) - 1.0)


def geometric_energy(prediction: torch.Tensor, expert: torch.Tensor, excess: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Eq. 5, masked. ``(I + M)`` weighting, averaged over live dimensions only.

    Normalising by the mask sum rather than the element count keeps a domain that populates 10 of
    120 slots on the same per-live-dimension footing as one that populates 40.
    """
    residual = (prediction - expert) * mask
    return 0.5 * ((residual**2) * (1.0 + excess)).sum() / mask.sum().clamp_min(1.0)


def drifting_loss(
    head,
    encoded,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    chunk_mask: torch.Tensor,
    domain_id: torch.Tensor,
    *,
    config: DriftingConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq. 6. ``actions`` is ``(B, H, action_dim)``; the geometry is computed on the flat chunk.

    The observation embedding comes from the **frozen** encoder and is detached, per Eq. 7's
    ``sg[.]``: the geometry is a constant target, not something the head can reshape to make its
    own loss smaller.
    """
    config = config or DriftingConfig()
    batch, horizon, dim = actions.shape
    flat_actions = actions.reshape(batch, -1)
    flat_mask = chunk_mask.reshape(batch, -1)

    with torch.no_grad():
        pooled = torch.cat([readout.float().mean(dim=1) for readout in encoded.sync_readout], dim=-1)
        weights = neighbour_weights(pooled)
        excess = excess_precision(flat_actions.detach() * flat_mask, weights)

    # y: the one-step prediction, pure noise at t = 0. This is what inference runs.
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    zeros = torch.zeros(batch, device=actions.device)
    y = head(encoded, action_mask, noise, zeros, domain_id)

    # z: the expert-proximal probe. Without it the geometry is only ever evaluated far from the
    # data, where it says little about the modes the policy has to keep apart.
    probe_noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    perturbed = actions + (1.0 - config.t_star) * probe_noise
    t_star = torch.full((batch,), config.t_star, device=actions.device)
    z = head(encoded, action_mask, perturbed, t_star, domain_id)

    energy_y = geometric_energy(y.reshape(batch, -1), flat_actions, excess, flat_mask)
    energy_z = geometric_energy(z.reshape(batch, -1), flat_actions, excess, flat_mask)
    loss = energy_y + config.lambda_prox * energy_z

    del horizon, dim
    return loss, {
        "energy_one_step": float(energy_y.detach()),
        "energy_proximal": float(energy_z.detach()),
        "excess_mean": float(excess.mean()),
        "excess_active_frac": float((excess > 0).float().mean()),
    }
