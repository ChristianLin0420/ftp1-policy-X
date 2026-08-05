from __future__ import annotations

import torch

from openpi.mot_jepa.action_dit import ActionDiT
from openpi.mot_jepa.action_dit import ActionDiTConfig
from openpi.mot_jepa.config import CONFIGS
from openpi.mot_jepa.drifting import DriftingConfig
from openpi.mot_jepa.drifting import drifting_loss
from openpi.mot_jepa.drifting import excess_precision
from openpi.mot_jepa.drifting import geometric_energy
from openpi.mot_jepa.drifting import neighbour_weights
from openpi.mot_jepa.mot_encoder import EncoderOutput

LAYOUT = CONFIGS["mot_jepa_pilot"].layout


def fake_encoded(batch: int, *, generator: torch.Generator | None = None) -> EncoderOutput:
    return EncoderOutput(
        tokens=[],
        sync_readout=[
            torch.randn(batch, LAYOUT.num_steps, LAYOUT.video_width, generator=generator),
            torch.randn(batch, LAYOUT.num_steps, LAYOUT.tactile_width, generator=generator),
        ],
    )


def test_observation_constrained_dimensions_get_the_extra_pull():
    """THE sign test, and the reason the paper uses precisions rather than variances.

    The mechanism is *conditional* tightening, so the discriminating case needs a dimension whose
    variance is high globally but low among observation-similar neighbours. A dimension that is
    merely constant everywhere is not "constrained by the observation" and correctly gets nothing.

    Here two clusters sit at opposite values in dimension 0, so globally it has the largest
    variance of any dimension, while within a cluster it is nearly fixed. Dimension 1 is
    independent noise everywhere -- genuinely multimodal given the observation -- and must be left
    at weight 1.0. Implementing Eq. 9 with variances instead of precisions swaps these two, which
    would put the strongest pull on the multimodal dimension and collapse its modes toward their
    mean while looking like ordinary underfitting.
    """
    generator = torch.Generator().manual_seed(0)
    batch, dim, width = 64, 6, 16
    cluster = torch.arange(batch) % 2

    # Embeddings that genuinely separate the two clusters, so neighbours share a cluster.
    embeddings = torch.randn(batch, width, generator=generator) * 0.05
    embeddings[:, 0] += torch.where(cluster == 0, 5.0, -5.0)

    actions = torch.randn(batch, dim, generator=generator)
    actions[:, 0] = torch.where(cluster == 0, 8.0, -8.0) + 0.01 * torch.randn(batch, generator=generator)

    weights = neighbour_weights(embeddings)
    excess = excess_precision(actions, weights)

    assert actions[:, 0].var() > actions[:, 1].var(), "dimension 0 must be the globally wider one"
    constrained, multimodal = float(excess[:, 0].mean()), float(excess[:, 1].mean())
    assert constrained > 5 * multimodal, (
        f"the observation-constrained dimension must get far MORE pull than the multimodal one, "
        f"got {constrained:.3f} vs {multimodal:.3f}; if this inverts or narrows, Eq. 9 was "
        f"implemented with variances instead of precisions"
    )
    # (1 + M) stays close to the identity where the observation does not constrain, so those
    # directions keep the plain-MSE pull and their modes are not squeezed together.
    assert multimodal < 0.25, f"an unconstrained dimension must stay near weight 1.0, got 1+{multimodal:.3f}"


def test_excess_is_never_negative():
    """``M`` only ever ADDS pull; the one-sided ReLU must not be able to loosen a dimension."""
    generator = torch.Generator().manual_seed(1)
    actions = torch.randn(48, 12, generator=generator)
    weights = neighbour_weights(torch.randn(48, 16, generator=generator))
    assert (excess_precision(actions, weights) >= 0).all()


def test_zero_excess_reduces_to_masked_mse():
    """Proposition 3.1's degenerate case, made a test rather than a surprise.

    With no local geometry the objective must be exactly masked MSE (up to the 1/2). If this ever
    fails, the anisotropic term is leaking into the isotropic baseline.
    """
    generator = torch.Generator().manual_seed(2)
    prediction = torch.randn(6, 20, generator=generator)
    expert = torch.randn(6, 20, generator=generator)
    mask = torch.ones(6, 20)
    mask[:, 10:] = 0.0

    energy = geometric_energy(prediction, expert, torch.zeros_like(prediction), mask)
    expected = 0.5 * (((prediction - expert) * mask) ** 2).sum() / mask.sum()
    assert torch.allclose(energy, expected)


def test_neighbour_weights_exclude_self_and_are_row_stochastic():
    """Self-similarity is the largest cosine, so leaving it in would dominate every row."""
    generator = torch.Generator().manual_seed(3)
    weights = neighbour_weights(torch.randn(16, 32, generator=generator))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)
    assert torch.allclose(weights.diagonal(), torch.zeros(16), atol=1e-7)


def test_masked_out_dimensions_cannot_affect_the_loss():
    """An absent action group must contribute exactly zero, not a small number.

    Domains populate different slots of the 120-D layout; if dead slots leaked into the loss the
    head would be trained to regress zeros it should simply ignore.
    """
    generator = torch.Generator().manual_seed(4)
    prediction = torch.randn(5, 12, generator=generator)
    expert = torch.randn(5, 12, generator=generator)
    mask = torch.ones(5, 12)
    mask[:, 6:] = 0.0
    excess = torch.rand(5, 12, generator=generator)

    base = geometric_energy(prediction, expert, excess, mask)
    moved = prediction.clone()
    moved[:, 6:] += 1000.0
    assert torch.allclose(base, geometric_energy(moved, expert, excess, mask))


def test_drifting_loss_runs_and_reports_its_geometry():
    """End to end on the real head, including the expert-proximal probe."""
    generator = torch.Generator().manual_seed(5)
    batch, horizon = 12, 6
    head = ActionDiT(
        ActionDiTConfig(width=64, depth=2, num_heads=4, horizon=horizon, objective="drifting"), LAYOUT
    )
    encoded = fake_encoded(batch, generator=generator)
    actions = torch.randn(batch, horizon, 120, generator=generator)
    action_mask = torch.zeros(batch, 120)
    action_mask[:, :16] = 1.0
    chunk_mask = action_mask[:, None, :].expand(batch, horizon, 120)

    loss, metrics = drifting_loss(
        head, encoded, actions, action_mask, chunk_mask, config=DriftingConfig(), generator=generator
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert set(metrics) == {"energy_one_step", "energy_proximal", "excess_mean", "excess_active_frac"}


def test_drifting_inference_is_one_forward_pass():
    """1 NFE is the headline reason to prefer drifting over flow matching; assert it."""
    calls = []
    horizon = 6
    head = ActionDiT(
        ActionDiTConfig(width=64, depth=1, num_heads=4, horizon=horizon, objective="drifting"), LAYOUT
    )
    original = head.forward
    head.forward = lambda *a, **k: (calls.append(1), original(*a, **k))[1]

    encoded = fake_encoded(3)
    head.sample(encoded, torch.ones(3, 120))
    assert len(calls) == 1, f"drifting inference must be 1 NFE, took {len(calls)}"
