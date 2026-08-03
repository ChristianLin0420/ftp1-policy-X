from __future__ import annotations

import pytest
import torch

from openpi.mot_jepa.ablation import TactileMode
from openpi.mot_jepa.ablation import apply_tactile_ablation
from openpi.mot_jepa.ablation import derangement
from openpi.mot_jepa.ablation import donor_at_offset
from openpi.mot_jepa.ablation import summarize_ablation


@pytest.fixture(name="tactile")
def tactile_fixture() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 4, 3, 8, 8) * 2.0 + 1.0


def test_real_is_the_identity(tactile):
    assert torch.equal(apply_tactile_ablation(tactile, "real"), tactile)


def test_zero_destroys_everything(tactile):
    out = apply_tactile_ablation(tactile, TactileMode.ZERO)
    assert torch.equal(out, torch.zeros_like(tactile))


def test_noise_preserves_the_first_two_moments_but_not_structure(tactile):
    """`noise` and `zero` differ only in what survives, which is why both conditions exist."""
    out = apply_tactile_ablation(tactile, "noise", generator=torch.Generator().manual_seed(0))
    real_flat = tactile.reshape(8, -1)
    out_flat = out.reshape(8, -1)
    # Tolerance is set by the sampling error of 768 draws at sigma ~= 2, i.e. ~0.07 on the
    # mean; anything tighter would be testing the RNG rather than the transform.
    torch.testing.assert_close(out_flat.mean(dim=1), real_flat.mean(dim=1), atol=0.3, rtol=0)
    torch.testing.assert_close(out_flat.std(dim=1), real_flat.std(dim=1), atol=0.3, rtol=0)
    assert not torch.allclose(out, tactile)


def test_shuffle_preserves_every_value_but_breaks_correspondence(tactile):
    out = apply_tactile_ablation(tactile, "shuffle", generator=torch.Generator().manual_seed(0))
    # Every sample is a genuine tactile reading, just not its own.
    assert torch.equal(out.flatten().sort().values, tactile.flatten().sort().values)
    for i in range(tactile.shape[0]):
        assert not torch.equal(out[i], tactile[i]), f"sample {i} kept its own tactile signal"


def test_derangement_has_no_fixed_point():
    """A plain randperm leaves ~1 clip correctly paired, contaminating the comparison."""
    for seed in range(25):
        permutation = derangement(9, generator=torch.Generator().manual_seed(seed))
        assert not bool((permutation == torch.arange(9)).any())
        assert sorted(permutation.tolist()) == list(range(9))


def test_derangement_degenerates_gracefully():
    assert derangement(1).tolist() == [0]
    assert derangement(0).tolist() == []


def test_donor_at_offset_shifts_in_time():
    sequence = torch.arange(10).float()
    assert donor_at_offset(sequence, 0).tolist() == sequence.tolist()
    assert donor_at_offset(sequence, 2)[2:].tolist() == sequence[:-2].tolist()


def test_summary_flags_a_supported_thesis():
    # Satisfies the predicted ordering real < zero <= shuffle <= noise.
    summary = summarize_ablation({"real": 1.0, "zero": 1.12, "shuffle": 1.15, "noise": 1.30})
    assert summary["shuffle_pct"] == pytest.approx(15.0)
    assert summary["thesis_supported"] is True
    assert summary["thesis_falsified"] is False
    assert summary["noise_zero_separated"] is True
    assert summary["predicted_ordering_holds"] is True


def test_summary_reproduces_the_published_ftp1_failure():
    """The measured baseline: noise == zero to three digits and shuffle ~ real."""
    summary = summarize_ablation({"real": 1.0, "zero": 1.221, "noise": 1.221, "shuffle": 1.007})
    assert summary["shuffle_pct"] == pytest.approx(0.7, abs=0.05)
    assert summary["thesis_falsified"] is True
    assert summary["noise_zero_separated"] is False
    assert summary["predicted_ordering_holds"] is False


def test_summary_is_safe_without_a_real_condition():
    assert summarize_ablation({"zero": 1.0}) == {"zero": 1.0}


@pytest.mark.parametrize("mode", list(TactileMode), ids=[m.value for m in TactileMode])
def test_every_mode_preserves_shape_and_dtype(tactile, mode):
    out = apply_tactile_ablation(tactile, mode, generator=torch.Generator().manual_seed(1))
    assert out.shape == tactile.shape
    assert out.dtype == tactile.dtype
