from __future__ import annotations

import torch

from scripts.mot_jepa_timeshuffle_control import _time_residual
from scripts.mot_jepa_timeshuffle_control import between_clip_distance
from scripts.mot_jepa_timeshuffle_control import permute_time_per_sample
from scripts.mot_jepa_timeshuffle_control import permute_time_shared
from scripts.mot_jepa_timeshuffle_control import temporal_variation


def _clip(batch: int = 8, steps: int = 16, *, trailing: tuple[int, ...] = (3,)) -> torch.Tensor:
    """Distinct value per (sample, step) so a permutation is detectable by content alone."""
    base = torch.arange(batch * steps, dtype=torch.float32).view(batch, steps)
    return base.view(batch, steps, *([1] * len(trailing))).expand(batch, steps, *trailing).contiguous()


def test_per_sample_preserves_each_clips_multiset():
    """Order destroyed, content untouched -- the property the control depends on.

    If the permutation altered content, any measured displacement would confound 'the encoder
    tracks order' with 'the input changed', which is precisely what the probe must not do.
    """
    tensor = _clip()
    shuffled = permute_time_per_sample(tensor, seed=0)
    for row in range(tensor.shape[0]):
        assert torch.equal(
            tensor[row].flatten().sort().values, shuffled[row].flatten().sort().values
        ), f"row {row} changed content, not just order"


def test_per_sample_uses_a_different_order_per_clip():
    """The point of the per-sample variant: no shared time relabeling survives across clips.

    ``probes._permute_time`` applies one order to the whole batch, so every retrieval candidate
    is perturbed identically and the relative geometry can survive. This asserts the new control
    does not share that defect.
    """
    tensor = _clip(batch=8, steps=16)
    shuffled = permute_time_per_sample(tensor, seed=0)
    # Recover each row's order from its content, since values are unique per (row, step).
    orders = [(shuffled[row, :, 0] - row * 16).to(torch.int64).tolist() for row in range(8)]
    assert len({tuple(order) for order in orders}) > 1, "every clip got the same permutation"


def test_shared_variant_really_does_share_one_order():
    """Characterises the probe's own control, so the comparison in the script is honest."""
    tensor = _clip(batch=8, steps=16)
    shuffled = permute_time_shared(tensor, seed=0)
    orders = [(shuffled[row, :, 0] - row * 16).to(torch.int64).tolist() for row in range(8)]
    assert len({tuple(order) for order in orders}) == 1, "shared variant should reuse one order"


def test_both_variants_are_deterministic_given_a_seed():
    """Step-to-step comparability, the property the shared version's docstring cites."""
    tensor = _clip()
    for permute in (permute_time_shared, permute_time_per_sample):
        assert torch.equal(permute(tensor, seed=3), permute(tensor, seed=3))


def test_per_sample_handles_extra_trailing_dims():
    """gel arrives as (B, T, pads, H, W, C); the gather must not assume a flat trailing shape."""
    tensor = _clip(batch=4, steps=8, trailing=(2, 5, 5, 3))
    shuffled = permute_time_per_sample(tensor, seed=1)
    assert shuffled.shape == tensor.shape
    for row in range(4):
        assert torch.equal(tensor[row].flatten().sort().values, shuffled[row].flatten().sort().values)


def test_static_clip_reports_no_temporal_variation():
    """The case that flips the verdict: identical frames mean the shuffle is a no-op.

    If a clip's frames are duplicates, an order-blind encoder is correct behaviour and
    timeshuffle_gap ~ 0 carries no information about the objective. This is the denominator the
    latent-only measurement was missing.
    """
    static = torch.randn(4, 1, 3, 8, 8).expand(4, 16, 3, 8, 8).contiguous()
    assert temporal_variation(static) < 1e-6
    # ... and the shuffle genuinely cannot change it.
    assert torch.equal(permute_time_per_sample(static, seed=0), static)


def test_moving_clip_reports_temporal_variation():
    """A clip that actually changes over time must read clearly above zero, or the guard is dead."""
    moving = torch.randn(4, 16, 3, 8, 8)
    assert temporal_variation(moving) > 0.5


def test_time_residual_is_unchanged_by_the_clip_mean():
    """Centring must isolate the time-varying part, so a large DC offset cannot mask a null.

    Raw pixels share a huge constant component; without centring both the displacement and the
    between-clip spread would be dominated by it and their ratio would be uninformative.
    """
    signal = torch.randn(3, 8, 5)
    offset = signal + 100.0
    assert torch.allclose(_time_residual(signal), _time_residual(offset), atol=1e-4)


def test_shuffling_preserves_the_clip_mean_so_centring_is_comparable():
    """Real and shuffled are centred by the same vector -- otherwise the comparison is invalid."""
    clip = torch.randn(4, 16, 3, 8, 8)
    shuffled = permute_time_per_sample(clip, seed=0)
    assert torch.allclose(clip.mean(dim=1), shuffled.mean(dim=1), atol=1e-5)


def test_between_clip_distance_ignores_the_diagonal():
    """Identical clips must read 0 spread; the self-similarity term would mask a collapse."""
    identical = torch.ones(6, 12)
    assert between_clip_distance(identical) < 1e-6

    orthogonal = torch.eye(6)
    # Every off-diagonal pair is orthogonal => cos 0 => distance 1.
    assert abs(between_clip_distance(orthogonal) - 1.0) < 1e-6
