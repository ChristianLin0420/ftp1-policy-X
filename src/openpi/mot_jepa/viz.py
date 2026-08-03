"""Figures for reading a MoT-JEPA run.

Grouped by the question each one answers.

**Is it binding?** ``similarity_heatmap``, ``retrieval_curve``, ``sync_matrix``,
``temporal_offset_curve``, ``donor_ratio_curve``. The retrieval plot draws the shuffled
control alongside the real curve because *the gap is the metric* -- a high top-1 that the
control also achieves means the model is reading position, not touch.

**Does the correspondence look right?** ``filmstrip``, ``contact_timeline``, ``mask_panel``.

**Is the representation healthy?** ``rank_curve``, ``singular_value_spectrum``,
``ema_drift_curve``. These detect collapse, which a falling loss will happily hide.

**Is sim like real?** ``sim_vs_real_gel``. Load-bearing because all closed-loop evaluation
is simulated, so the gel appearance gap is a direct threat to transfer.

Follows ``scripts/zarr_train_ftp1_validation_func.py``: ``matplotlib.use("Agg")`` at import
and ``plt.close(fig)`` after every figure, so this is safe on a headless compute node.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.layout import TokenLayout

#: Colour-blind-safe qualitative palette, dark enough to survive greyscale printing.
PALETTE = ("#3B6FB6", "#D1651A", "#2E8B57", "#8B4A9C", "#B03A3A", "#7A7A7A")


def _to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor)


def _image_from_chw(frame) -> np.ndarray:
    """``(3, H, W)`` in [-1, 1] or uint8 into a displayable ``(H, W, 3)`` float array."""
    array = _to_numpy(frame)
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    # uint8-scaled inputs exceed 1.5; [-1, 1] inputs do not.
    array = array / 255.0 if array.max() > 1.5 else (array + 1.0) / 2.0
    return np.clip(array, 0.0, 1.0)


def _finish(fig, path=None):
    """Save if asked, always close, and return a ``wandb.Image`` when W&B is importable."""
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    image = None
    try:
        # Lazy: plotting must work without a tracking backend installed.
        from openpi.shared.wandb_compat import wandb  # noqa: PLC0415

        image = wandb.Image(fig)
    except Exception:
        image = None
    plt.close(fig)
    return image


# --------------------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------------------


def similarity_heatmap(video: torch.Tensor, tactile: torch.Tensor, *, path=None):
    """``B x B`` cosine similarity. A bound model glows on the diagonal."""
    video = torch.nn.functional.normalize(video.detach().float(), dim=-1)
    tactile = torch.nn.functional.normalize(tactile.detach().float(), dim=-1)
    matrix = _to_numpy(video @ tactile.t())

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("tactile clip")
    ax.set_ylabel("video clip")
    diagonal = float(np.mean(np.diag(matrix)))
    off = float((matrix.sum() - np.trace(matrix)) / max(matrix.size - matrix.shape[0], 1))
    ax.set_title(f"cross-modal similarity\ndiagonal {diagonal:.3f} vs off-diagonal {off:.3f}")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _finish(fig, path)


def sync_matrix(video_steps: torch.Tensor, tactile_steps: torch.Tensor, *, contact_step=None, path=None):
    """``T x T`` within-clip similarity for one clip.

    A bound model shows a bright diagonal band, and the *width* of that band reads out the
    temporal precision of binding -- which is a far more informative number than a scalar.
    """
    video = torch.nn.functional.normalize(video_steps.detach().float(), dim=-1)
    tactile = torch.nn.functional.normalize(tactile_steps.detach().float(), dim=-1)
    matrix = _to_numpy(video @ tactile.t())

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    image = ax.imshow(matrix, cmap="magma", origin="lower")
    ax.set_xlabel("tactile instant t'")
    ax.set_ylabel("video instant t")
    ax.plot([0, matrix.shape[1] - 1], [0, matrix.shape[0] - 1], color="white", lw=0.8, ls="--", alpha=0.6)
    if contact_step is not None:
        ax.axvline(contact_step, color="#00E5FF", lw=1.2, label="contact onset")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title("within-clip synchrony\n(diagonal band width = temporal precision)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _finish(fig, path)


def retrieval_curve(steps, top1, shortcut, chance, *, path=None):
    """Retrieval versus its two reference lines. The shaded gap is the real metric."""
    steps = np.asarray(steps)
    top1 = np.asarray(top1, dtype=float)
    shortcut = np.asarray(shortcut, dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(steps, top1, color=PALETTE[0], lw=2, label="retrieval top-1")
    ax.plot(steps, shortcut, color=PALETTE[1], lw=1.6, ls="--", label="positional-shortcut control")
    ax.axhline(chance, color=PALETTE[5], lw=1.2, ls=":", label=f"chance ({chance:.3f})")
    ax.fill_between(steps, shortcut, top1, where=top1 >= shortcut, color=PALETTE[0], alpha=0.15)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("binding: retrieval above its own shortcut control")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def temporal_offset_curve(offsets, errors, *, path=None):
    """Error against a donor taken from the same episode at a controlled offset.

    A bound model rises monotonically and saturates at the contact process's correlation
    time. An unbound model is flat -- which is the shape the current FTP-1 checkpoints show.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(offsets, errors, marker="o", color=PALETTE[0], lw=2)
    ax.axhline(errors[0], color=PALETTE[5], ls=":", lw=1.2, label="matched-pair error")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("donor offset (frames)")
    ax.set_ylabel("prediction error")
    ax.set_title("temporal-offset sweep\nrising = bound, flat = unbound")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def donor_ratio_curve(steps, ratios, *, gates=((20_000, 1.15), (50_000, 1.30)), path=None):
    """Donor ratio with its pre-registered gates drawn in."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(steps, ratios, color=PALETTE[2], lw=2, label=r"$\rho_T$")
    ax.axhline(1.0, color=PALETTE[4], ls="--", lw=1.2, label="1.0 = no binding")
    for step, value in gates:
        ax.plot(step, value, marker="*", ms=13, color=PALETTE[1])
        ax.annotate(f"gate {value}", (step, value), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(r"$L_T(\mathrm{donor})\ /\ L_T(\mathrm{true})$")
    ax.set_title("donor ratio: binding inside the JEPA path")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


# --------------------------------------------------------------------------------------
# Correspondence and rendering
# --------------------------------------------------------------------------------------


def filmstrip(video, gel, *, num_frames: int = 8, path=None):
    """Rows of ``[RGB | gel pad 0 | gel pad 1]`` across time. The 'does it look right' plot."""
    video = _to_numpy(video)
    gel = _to_numpy(gel)
    total = video.shape[0]
    picks = np.linspace(0, total - 1, min(num_frames, total)).astype(int)
    num_pads = gel.shape[1]
    rows = 1 + num_pads

    fig, axes = plt.subplots(rows, len(picks), figsize=(1.5 * len(picks), 1.6 * rows))
    axes = np.atleast_2d(axes)
    for column, frame in enumerate(picks):
        axes[0, column].imshow(_image_from_chw(video[frame]))
        axes[0, column].set_title(f"t={frame}", fontsize=8)
        for pad in range(num_pads):
            axes[pad + 1, column].imshow(_image_from_chw(gel[frame, pad]))
    for row, label in enumerate(["camera", *[f"gel {i}" for i in range(num_pads)]]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("clip filmstrip: vision and touch on a shared time axis", fontsize=11)
    fig.tight_layout()
    return _finish(fig, path)


def contact_timeline(tactile_energy, gripper=None, contact_steps=(), *, path=None):
    """Tactile activity over time, with contact onsets marked.

    Answers whether the model's tactile signal actually moves *when* the contact does, which
    a scalar aggregate cannot show.
    """
    energy = _to_numpy(tactile_energy).reshape(-1)
    steps = np.arange(energy.size)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(steps, energy, color=PALETTE[0], lw=1.8, label="tactile energy")
    if gripper is not None:
        twin = ax.twinx()
        twin.plot(steps, _to_numpy(gripper).reshape(-1), color=PALETTE[1], lw=1.4, ls="--", label="gripper")
        twin.set_ylabel("gripper", color=PALETTE[1], fontsize=9)
        twin.spines[["top"]].set_visible(False)
    for onset in contact_steps:
        ax.axvline(onset, color=PALETTE[4], lw=1.1, alpha=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("tactile energy", color=PALETTE[0], fontsize=9)
    ax.set_title("contact-event alignment")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def mask_panel(layout: TokenLayout, target_index, mode_name: str, *, path=None):
    """Which video and gel positions were masked, per stream.

    A masking bug is otherwise invisible: the loss still falls, it just falls on the wrong
    thing.
    """
    index = _to_numpy(target_index).astype(int).reshape(-1)
    coords = _to_numpy(layout.coords).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    for ax, stream, grid, title in (
        (axes[0], StreamId.VIDEO, layout.video_grid, "video"),
        (axes[1], StreamId.GEL, layout.gel_grid, "gel"),
    ):
        span = layout.stream_slices[stream]
        selected = index[(index >= span.start) & (index < span.stop)]
        heat = np.zeros(grid)
        for token in selected:
            _, row, column = coords[token]
            heat[row, column] += 1
        ax.imshow(heat, cmap="viridis")
        ax.set_title(f"{title}: {selected.size} target tokens", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"mask mode {mode_name}", fontsize=11)
    fig.tight_layout()
    return _finish(fig, path)


# --------------------------------------------------------------------------------------
# Representation health
# --------------------------------------------------------------------------------------


def rank_curve(steps, video_rank, tactile_rank, *, path=None):
    """Effective rank per modality. Falling rank is collapse, whatever the loss says."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(steps, video_rank, color=PALETTE[0], lw=2, label="video")
    ax.plot(steps, tactile_rank, color=PALETTE[1], lw=2, label="tactile")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("RankMe (effective rank)")
    ax.set_title("representation health: rank must not fall")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def singular_value_spectrum(features_by_step: dict, *, path=None):
    """Log-log singular-value spectra at several checkpoints; flattening indicates collapse."""
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for colour, (label, features) in zip(PALETTE, sorted(features_by_step.items()), strict=False):
        values = torch.linalg.svdvals(torch.as_tensor(features).float())
        values = _to_numpy(values)
        ax.loglog(np.arange(1, values.size + 1), values / values[0], color=colour, lw=1.8, label=str(label))
    ax.set_xlabel("index")
    ax.set_ylabel("normalized singular value")
    ax.set_title("feature spectrum")
    ax.legend(frameon=False, fontsize=8, title="step")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def ema_drift_curve(steps, drift, *, path=None):
    """``||EMA - student||`` over training.

    Pinned at exactly zero is the bf16 freeze: the teacher never moved off its random
    initialization, and the loss falls anyway because predicting a frozen random projection
    is a learnable task.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(steps, drift, color=PALETTE[2], lw=2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(r"$\|\mathrm{EMA} - \mathrm{student}\|$ (relative)")
    ax.set_title("EMA drift: exactly zero means the teacher is frozen")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


# --------------------------------------------------------------------------------------
# Sim-to-real
# --------------------------------------------------------------------------------------


def sim_vs_real_gel(sim_frames, real_frames, *, path=None):
    """Simulated versus real gel appearance, with per-channel histograms.

    All closed-loop evaluation is simulated, so this gap is a direct threat to transfer and
    deserves to be looked at rather than assumed away.
    """
    sim = _to_numpy(sim_frames)
    real = _to_numpy(real_frames)

    fig = plt.figure(figsize=(9.0, 4.6))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.4, 1.4])
    for row, (frames, label) in enumerate(((sim, "simulated"), (real, "real"))):
        for column in range(2):
            ax = fig.add_subplot(grid[row, column])
            ax.imshow(_image_from_chw(frames[min(column * 4, len(frames) - 1)]))
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(label, fontsize=10)

    ax_hist = fig.add_subplot(grid[:, 2])
    for channel, colour in zip(range(3), ("#B03A3A", "#2E8B57", "#3B6FB6"), strict=True):
        ax_hist.hist(sim[..., channel, :, :].ravel(), bins=48, histtype="step", color=colour, lw=1.6)
        ax_hist.hist(real[..., channel, :, :].ravel(), bins=48, histtype="step", color=colour, lw=1.2, ls="--")
    ax_hist.set_title("channel histograms\n(solid sim, dashed real)", fontsize=9)
    ax_hist.spines[["top", "right"]].set_visible(False)

    ax_pca = fig.add_subplot(grid[:, 3])
    flat = np.concatenate([sim.reshape(len(sim), -1), real.reshape(len(real), -1)], axis=0)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(flat, full_matrices=False)
    projected = flat @ components[:2].T
    ax_pca.scatter(projected[: len(sim), 0], projected[: len(sim), 1], s=14, color=PALETTE[0], label="sim")
    ax_pca.scatter(projected[len(sim) :, 0], projected[len(sim) :, 1], s=14, color=PALETTE[1], label="real")
    ax_pca.set_title("appearance PCA", fontsize=9)
    ax_pca.legend(frameon=False, fontsize=8)
    ax_pca.spines[["top", "right"]].set_visible(False)

    fig.suptitle("sim-to-real gel appearance gap", fontsize=11)
    fig.tight_layout()
    return _finish(fig, path)


def ablation_bars(conditions: dict, *, target_gap: float = 0.10, path=None):
    """Tactile-ablation results with the pre-registered prediction drawn on.

    The predicted ordering ``real < zero <= shuffle <= noise`` is mutually exclusive with the
    currently-measured one (``noise == zero``, ``shuffle ~ real``), which is what makes the
    test sharp rather than merely suggestive.
    """
    names = list(conditions)
    values = [conditions[name] for name in names]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colours = [PALETTE[2] if name == "real" else PALETTE[0] for name in names]
    ax.bar(names, values, color=colours)
    if "real" in conditions:
        ax.axhline(
            conditions["real"] * (1 + target_gap),
            color=PALETTE[4],
            ls="--",
            lw=1.3,
            label=f"pre-registered +{target_gap:.0%} threshold",
        )
        ax.legend(frameon=False, fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_title("tactile ablation\npredicted ordering: real < zero <= shuffle <= noise")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)


def training_panels(panels: dict, layout: TokenLayout, target_index=None, mode_name: str = "") -> dict:
    """Build the live W&B image panels from tensors the probe already computed.

    Only the four figures that need pixels are rendered here. Everything that is naturally a
    curve -- retrieval, rank, EMA drift, per-mode losses -- is logged as a scalar and drawn by
    W&B natively, which stays interactive and costs nothing per step.
    """
    images = {}
    if "video_vec" in panels:
        images["similarity"] = similarity_heatmap(panels["video_vec"], panels["tactile_vec"])
    if "video_steps" in panels:
        images["sync_matrix"] = sync_matrix(panels["video_steps"], panels["tactile_steps"])
    if "video_frames" in panels:
        images["filmstrip"] = filmstrip(panels["video_frames"], panels["gel_frames"], num_frames=8)
    if target_index is not None:
        images["mask_panel"] = mask_panel(layout, target_index, mode_name)
    return {key: value for key, value in images.items() if value is not None}
