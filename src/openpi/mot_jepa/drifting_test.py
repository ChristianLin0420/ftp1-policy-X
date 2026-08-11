from __future__ import annotations

import torch

from openpi.mot_jepa.action_dit import ActionDiT
from openpi.mot_jepa.action_dit import ActionDiTConfig
from openpi.mot_jepa.action_dit import LinearHead
from openpi.mot_jepa.action_dit import PerDomainLinear
from openpi.mot_jepa.action_dit import flow_matching_loss
from openpi.mot_jepa.action_dit import regression_loss
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
        ActionDiTConfig(width=64, depth=2, num_heads=4, horizon=horizon, objective="drifting"),
        LAYOUT,
        num_domains=3,
    )
    encoded = fake_encoded(batch, generator=generator)
    actions = torch.randn(batch, horizon, 120, generator=generator)
    action_mask = torch.zeros(batch, 120)
    action_mask[:, :16] = 1.0
    chunk_mask = action_mask[:, None, :].expand(batch, horizon, 120)

    domain_id = torch.zeros(batch, dtype=torch.long)
    loss, metrics = drifting_loss(
        head, encoded, actions, action_mask, chunk_mask, domain_id, config=DriftingConfig(), generator=generator
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
        ActionDiTConfig(width=64, depth=1, num_heads=4, horizon=horizon, objective="drifting"),
        LAYOUT,
        num_domains=2,
    )
    original = head.forward
    head.forward = lambda *a, **k: (calls.append(1), original(*a, **k))[1]

    encoded = fake_encoded(3)
    head.sample(encoded, torch.ones(3, 120), torch.zeros(3, dtype=torch.long))
    assert len(calls) == 1, f"drifting inference must be 1 NFE, took {len(calls)}"


def test_domain_id_changes_the_prediction():
    """The head must actually route on domain, or the embedding is decoration.

    This is the defect the embedding exists to fix: a ridge fitted PER DOMAIN reaches R^2 0.65-0.93
    on held-out UniVTAC while the same ridge fitted as ONE SHARED map over the eight domains scores
    -0.34. Every head trained before this was that shared map -- none had any domain input at all --
    and every one lost to a per-domain constant. If swapping domain_id leaves the output unchanged,
    the head is still that shared map and the fix did nothing.
    """
    generator = torch.Generator().manual_seed(7)
    horizon, batch = 6, 4
    head = ActionDiT(
        ActionDiTConfig(width=64, depth=2, num_heads=4, horizon=horizon, objective="flowmatch"),
        LAYOUT,
        num_domains=5,
    )
    # A zero-init adaLN gate makes every block the identity at init, so route on a TRAINED head.
    with torch.no_grad():
        for block in head.blocks:
            block.modulation[1].weight.normal_(std=0.05)
            block.modulation[1].bias.normal_(std=0.05)
        head.action_out.weight.normal_(std=0.05)

    encoded = fake_encoded(batch, generator=generator)
    mask = torch.ones(batch, 120)
    x = torch.randn(batch, horizon, 120, generator=generator)
    t = torch.zeros(batch)

    first = head(encoded, mask, x, t, torch.zeros(batch, dtype=torch.long))
    second = head(encoded, mask, x, t, torch.full((batch,), 3, dtype=torch.long))
    assert not torch.allclose(first, second, atol=1e-6), "output is invariant to domain_id"


def test_linear_head_also_routes_on_domain():
    """Same requirement for the diagnostic baseline, which is the arm compared against the ridge."""
    generator = torch.Generator().manual_seed(8)
    horizon, batch = 6, 4
    head = LinearHead(
        ActionDiTConfig(width=64, depth=1, num_heads=4, horizon=horizon, objective="linear"),
        LAYOUT,
        num_domains=5,
    )
    with torch.no_grad():
        head.proj.weight.normal_(std=0.05)  # zero-init would make every domain identical

    encoded = fake_encoded(batch, generator=generator)
    mask = torch.ones(batch, 120)
    first = head.sample(encoded, mask, torch.zeros(batch, dtype=torch.long))
    second = head.sample(encoded, mask, torch.full((batch,), 4, dtype=torch.long))
    assert not torch.allclose(first, second, atol=1e-6), "linear head ignores domain_id"


def test_per_domain_linear_gives_weights_not_just_a_bias():
    """The distinction the whole fix turns on.

    A per-domain BIAS shifts every input by the same constant, so the DIFFERENCE between two
    inputs is identical across domains. Per-domain WEIGHTS change that difference. Measured, a
    shared map with per-domain normalised targets scores R^2 -0.0016 where eight per-domain maps
    score 0.65-0.93, so a bias cannot be enough and this asserts we did not ship one.
    """
    torch.manual_seed(0)
    layer = PerDomainLinear(num_domains=3, in_features=8, out_features=5)
    x = torch.randn(4, 8)
    a = torch.zeros(4, dtype=torch.long)
    b = torch.full((4,), 2, dtype=torch.long)

    delta_a = layer(x, a) - layer(torch.zeros_like(x), a)
    delta_b = layer(x, b) - layer(torch.zeros_like(x), b)
    assert not torch.allclose(delta_a, delta_b, atol=1e-6), (
        "input-to-output MAP is identical across domains; this is a per-domain bias, not weights"
    )


def test_per_domain_linear_routes_each_row_to_its_own_domain():
    """Mixed-domain batches occur at evaluation; each row must use its own weights."""
    torch.manual_seed(1)
    layer = PerDomainLinear(num_domains=4, in_features=6, out_features=3)
    x = torch.randn(5, 6)
    mixed = torch.tensor([0, 2, 0, 3, 2])

    got = layer(x, mixed)
    for row, domain in enumerate(mixed.tolist()):
        expected = x[row] @ layer.weight[domain] + layer.bias[domain]
        assert torch.allclose(got[row], expected, atol=1e-6), f"row {row} used the wrong domain"


def _loss_fixture(objective: str, *, batch: int = 6, horizon: int = 5, per_domain_trunk: bool = False):
    head_cls = LinearHead if objective == "linear" else ActionDiT
    head = head_cls(
        ActionDiTConfig(
            width=64,
            depth=2,
            num_heads=4,
            horizon=horizon,
            objective=objective,
            per_domain_trunk=per_domain_trunk,
        ),
        LAYOUT,
        num_domains=4,
    )
    encoded = fake_encoded(batch, generator=torch.Generator().manual_seed(11))
    actions = torch.randn(batch, horizon, 120)
    action_mask = torch.zeros(batch, 120)
    action_mask[:, :16] = 1.0
    chunk_mask = action_mask[:, None, :].expand(batch, horizon, 120)
    domain_id = torch.arange(batch) % 4
    return head, encoded, actions, action_mask, chunk_mask, domain_id


def test_every_loss_runs_end_to_end():
    """All three losses, not just drifting.

    The suite previously covered drifting_loss alone, so when flow_matching_loss was left calling
    the head WITHOUT domain_id it stayed green -- and the break only surfaced on a GPU four minutes
    into a real job. Each objective's loss is a distinct call path into the head and each needs its
    own exercise.
    """
    for objective, loss_fn in (("flowmatch", flow_matching_loss), ("linear", regression_loss)):
        head, encoded, actions, action_mask, chunk_mask, domain_id = _loss_fixture(objective)
        loss, metrics = loss_fn(head, encoded, actions, action_mask, chunk_mask, domain_id)
        assert torch.isfinite(loss), f"{objective} loss is not finite"
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()), objective
        assert metrics, f"{objective} reported no metrics"


def test_per_domain_trunk_gives_each_domain_an_independent_map():
    """The point of the flag: perturbing one domain's trunk must not move another domain's output.

    Asserted rather than assumed because a shared trunk passes every shape and finiteness check a
    per-domain one does, so nothing else in this suite can tell the two apart -- and the whole
    reason the flag exists is that a shared trunk is the suspected defect.
    """
    head, encoded, _, action_mask, _, _ = _loss_fixture("linear", per_domain_trunk=True)
    zero = torch.zeros(6, dtype=torch.long)
    one = torch.ones(6, dtype=torch.long)
    with torch.no_grad():
        # `proj` ships zero-initialised, so at init the head emits exactly zero and no trunk
        # perturbation is observable downstream. Give it weight before measuring.
        head.proj.weight.normal_(std=0.1)
        before_zero = head.sample(encoded, action_mask, zero)
        before_one = head.sample(encoded, action_mask, one)
        head.trunk.weight[0].add_(1.0)
        after_zero = head.sample(encoded, action_mask, zero)
        after_one = head.sample(encoded, action_mask, one)
    assert not torch.allclose(before_zero, after_zero), "domain 0's own trunk did not affect it"
    torch.testing.assert_close(before_one, after_one, msg="domain 0's trunk leaked into domain 1")


def test_shared_trunk_remains_the_default():
    """The three shipped presets trained with a shared trunk; their checkpoints must still load."""
    head, *_ = _loss_fixture("linear")
    assert isinstance(head.trunk, torch.nn.Linear), "default head must keep the shared nn.Linear trunk"


def test_every_objective_samples_end_to_end():
    """sample() is the deployment path for all three and must accept the same call."""
    for objective in ("drifting", "flowmatch", "linear"):
        head, encoded, _, action_mask, _, domain_id = _loss_fixture(objective)
        out = head.sample(encoded, action_mask, domain_id, num_steps=2)
        assert out.shape == (6, 5, 120), f"{objective} produced {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{objective} produced non-finite actions"
