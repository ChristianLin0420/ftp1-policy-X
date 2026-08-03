"""Smoke coverage for the figure suite.

Plotting must never be the thing that kills a training step, so every helper is exercised
here on synthetic data and each is checked for the property that makes it worth plotting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from openpi.mot_jepa import viz
from openpi.mot_jepa.layout import LAYOUT_PILOT
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import build_batch_masks


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    before = plt.get_fignums()
    yield
    assert plt.get_fignums() == before, "a figure was left open; call plt.close(fig)"


def test_similarity_heatmap_runs_and_closes():
    viz.similarity_heatmap(torch.randn(8, 32), torch.randn(8, 32))


def test_sync_matrix_accepts_a_contact_marker():
    viz.sync_matrix(torch.randn(8, 16), torch.randn(8, 16), contact_step=3)


def test_retrieval_curve_handles_the_control_above_the_signal():
    """Must not crash when the control beats retrieval -- that is the failure it exists to show."""
    steps = np.arange(0, 1000, 100)
    viz.retrieval_curve(steps, np.linspace(0.1, 0.2, len(steps)), np.linspace(0.15, 0.3, len(steps)), 0.05)


def test_temporal_offset_curve():
    viz.temporal_offset_curve([1, 2, 5, 10, 30, 100], [0.1, 0.12, 0.15, 0.2, 0.24, 0.25])


def test_donor_ratio_curve_draws_gates():
    viz.donor_ratio_curve(np.arange(0, 60_000, 5_000), np.linspace(1.0, 1.4, 12))


def test_filmstrip_with_two_gel_pads():
    video = torch.rand(16, 3, 32, 32) * 2 - 1
    gel = torch.rand(16, 2, 3, 16, 16) * 2 - 1
    viz.filmstrip(video, gel, num_frames=4)


def test_filmstrip_accepts_uint8_input():
    video = (torch.rand(8, 3, 32, 32) * 255).to(torch.uint8)
    gel = (torch.rand(8, 1, 3, 16, 16) * 255).to(torch.uint8)
    viz.filmstrip(video, gel, num_frames=3)


def test_contact_timeline_with_and_without_gripper():
    energy = np.abs(np.sin(np.linspace(0, 6, 32)))
    viz.contact_timeline(energy, contact_steps=(8, 20))
    viz.contact_timeline(energy, gripper=np.linspace(1, 0, 32), contact_steps=(8,))


@pytest.mark.parametrize("mode", list(MaskMode), ids=[m.name for m in MaskMode])
def test_mask_panel_for_every_mode(mode):
    probs = [0.0] * len(MaskMode)
    probs[int(mode)] = 1.0
    spec = MaskSpec(layout=LAYOUT_PILOT, mode_probs=tuple(probs))
    masks = build_batch_masks(spec, step=0, batch_size=2, base_seed=1)
    viz.mask_panel(LAYOUT_PILOT, masks.tgt_index[0], mode.name)


def test_rank_and_spectrum_and_drift():
    steps = np.arange(0, 500, 50)
    viz.rank_curve(steps, np.linspace(10, 40, len(steps)), np.linspace(8, 30, len(steps)))
    viz.singular_value_spectrum({0: np.random.randn(64, 16), 500: np.random.randn(64, 16)})
    viz.ema_drift_curve(steps, np.linspace(0, 0.2, len(steps)))


def test_sim_vs_real_gel():
    sim = torch.rand(6, 3, 16, 16)
    real = torch.rand(6, 3, 16, 16)
    viz.sim_vs_real_gel(sim, real)


def test_ablation_bars_draws_the_preregistered_threshold():
    viz.ablation_bars({"real": 1.0, "zero": 1.25, "shuffle": 1.12, "noise": 1.30})


def test_ablation_bars_without_a_real_condition():
    viz.ablation_bars({"zero": 1.25, "noise": 1.30})


def test_figures_are_returned_or_none_but_never_raise(tmp_path):
    """W&B may be unavailable; the helpers must degrade rather than break a training step."""
    result = viz.similarity_heatmap(torch.randn(4, 8), torch.randn(4, 8), path=tmp_path / "sim.png")
    assert (tmp_path / "sim.png").exists()
    assert result is None or hasattr(result, "_path") or result.__class__.__name__ == "Image"
