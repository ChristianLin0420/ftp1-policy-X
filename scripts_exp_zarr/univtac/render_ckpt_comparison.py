#!/usr/bin/env python
"""Render a side-by-side comparison video of two FTP-1 checkpoints on held-out episodes.

Each frame shows the head camera and both gel pads at time t, a headline chart of how far each
checkpoint's predicted action chunk sits from ground truth, and per-joint small multiples.

Predictions are converted to absolute joint space for readability: `action_joint_rep=mix` means arm
joints are relative to the current state while the gripper (hand slot 28) is already absolute.

Usage:
    uv run python scripts_exp_zarr/univtac/render_ckpt_comparison.py \
        --zarr .../lift_bottle_head.zarr --split .../train_val_split.json \
        --ckpt-a <baseline>/19999 --ckpt-b <pact>/19999 --out out.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import FancyBboxPatch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

# FTP-1 canonical layout: right block starts at 0; 9-d wrist pose, then 7 arm joints, then 32 hand
# slots. UniVTAC maps its scalar gripper to hand slot 28.
ARM_SLICE = slice(9, 16)
GRIPPER_IDX = 9 + 7 + 28
JOINT_LABELS = [f"Joint {i + 1}" for i in range(7)] + ["Gripper"]

# Apple-ish system palette.
BG = "#F5F5F7"
CARD = "#FFFFFF"
INK = "#1D1D1F"
INK_2 = "#86868B"
INK_3 = "#C7C7CC"
GRID = "#EDEDF0"
COL = {"a": "#0071E3", "b": "#FF9500"}  # blue / orange

FONT = "Liberation Sans"  # metric-compatible with Helvetica
plt.rcParams.update(
    {
        "font.family": FONT,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "axes.facecolor": CARD,
        "figure.facecolor": BG,
    }
)


def episode_bounds(episode_ends: np.ndarray, ep: int) -> tuple[int, int]:
    start = 0 if ep == 0 else int(episode_ends[ep - 1])
    return start, int(episode_ends[ep])


def build_state(arm_joints: np.ndarray, gripper: float, state_dim: int) -> np.ndarray:
    state = np.zeros((1, state_dim), dtype=np.float32)
    state[0, ARM_SLICE] = arm_joints
    state[0, GRIPPER_IDX] = gripper
    return state


def predict_chunk(wrapper, head_rgb, gel, arm_joints, gripper, prompt) -> np.ndarray:
    """Return an absolute-joint (horizon, 8) prediction: 7 arm joints + gripper."""
    state = build_state(arm_joints, gripper, wrapper.get_state_dim())
    with contextlib.redirect_stdout(io.StringIO()):  # wrapper prints a timing line per call
        chunk = wrapper.infer(
            images={"camera_ego_rgb_0": head_rgb},
            state=state,
            prompt=prompt,
            tactiles={"right_tactile_gripper": gel[None].astype(np.float32)},
            tactile_function_areas={"right_tactile_gripper": [0, 1]},
            tactile_sensors={"right_tactile_gripper": "GelSightMini"},
        )
    absolute = np.zeros((chunk.shape[0], 8), dtype=np.float32)
    # mix: arm joints relative to the current state, gripper already absolute.
    absolute[:, :7] = arm_joints[None, :] + chunk[:, ARM_SLICE]
    absolute[:, 7] = chunk[:, GRIPPER_IDX]
    return absolute


def card(fig, x, y, w, h, radius=0.014):
    """Draw a rounded white panel in figure coordinates."""
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            transform=fig.transFigure,
            facecolor=CARD,
            edgecolor="none",
            zorder=0,
        )
    )


def rounded_image(ax, img, radius_frac=0.045):
    ax.set_facecolor(CARD)
    # Default 'antialiased' resampling keeps the gel marker lattice crisp; bilinear softens it.
    im = ax.imshow(img)
    h, w = img.shape[:2]
    clip = FancyBboxPatch(
        (-0.5, -0.5),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={min(h, w) * radius_frac}",
        transform=ax.transData,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(clip)
    im.set_clip_path(clip)
    ax.axis("off")


def strip(ax, *, ygrid=True):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.set_facecolor(CARD)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        ax.set_axisbelow(True)


def chunk_error(pred: np.ndarray, gt: np.ndarray, anchor: int) -> float:
    """Mean absolute deviation of a predicted arm-joint chunk from ground truth, in radians."""
    n = gt.shape[0]
    span = min(pred.shape[0], n - anchor)
    if span <= 0:
        return float("nan")
    return float(np.abs(pred[:span, :7] - gt[anchor : anchor + span, :7]).mean())


def render_episode(writer, z, wrappers, labels, ep, episode_ends, stride, horizon_draw):
    start, end = episode_bounds(episode_ends, ep)
    n = end - start

    head = z["data"]["camera_ego_rgb"]
    gel_arr = z["data"]["right_tactile_data_gripper"]
    arm = z["data"]["right_arm_joints"][start:end].astype(np.float32)
    grip = z["data"]["right_hand_joints"][start:end].astype(np.float32).reshape(-1)
    # sub_task_instruction is a fixed-width <U100 array, so it arrives space-padded.
    prompt = str(z["data"]["sub_task_instruction"][start]).strip()
    gt = np.concatenate([arm, grip[:, None]], axis=1)  # (n, 8) absolute

    anchors = list(range(0, n, stride))
    preds: dict[str, dict[int, np.ndarray]] = {k: {} for k in wrappers}
    err: dict[str, list[float]] = {k: [] for k in wrappers}
    for i, t in enumerate(anchors):
        head_t = np.asarray(head[start + t])
        gel_t = np.asarray(gel_arr[start + t])
        for key, wrapper in wrappers.items():
            p = predict_chunk(wrapper, head_t, gel_t, arm[t], float(grip[t]), prompt)
            preds[key][t] = p
            err[key].append(chunk_error(p, gt, t))
        if (i + 1) % 10 == 0:
            print(f"ep{ep}: inferred {i + 1}/{len(anchors)}", flush=True)

    fig = plt.figure(figsize=(16, 9), dpi=120)

    # Static panels.
    card(fig, 0.030, 0.400, 0.350, 0.470)   # head camera
    card(fig, 0.030, 0.075, 0.168, 0.285)   # gel pad 0
    card(fig, 0.212, 0.075, 0.168, 0.285)   # gel pad 1
    card(fig, 0.408, 0.560, 0.562, 0.310)   # headline chart
    card(fig, 0.408, 0.075, 0.562, 0.445)   # small multiples

    ax_head = fig.add_axes([0.042, 0.412, 0.326, 0.446])
    ax_gel = [fig.add_axes([0.042, 0.088, 0.144, 0.238]), fig.add_axes([0.224, 0.088, 0.144, 0.238])]
    ax_hero = fig.add_axes([0.442, 0.600, 0.505, 0.180])

    sm_axes = []
    for r in range(2):
        for c in range(4):
            sm_axes.append(fig.add_axes([0.437 + c * 0.1345, 0.290 - r * 0.200, 0.104, 0.140]))

    anchors_arr = np.asarray(anchors)
    for t in range(n):
        ai = min(len(anchors) - 1, t // stride)
        anchor = anchors[ai]

        fig.texts.clear()
        fig.text(0.030, 0.945, prompt, fontsize=15, color=INK, fontweight="medium")
        fig.text(0.030, 0.905, f"Episode {ep} · held out from training", fontsize=11, color=INK_2)
        fig.text(0.970, 0.945, f"{t:03d}", fontsize=15, color=INK, ha="right")
        fig.text(0.970, 0.907, f"of {n} frames", fontsize=11, color=INK_2, ha="right")

        ax_head.clear()
        rounded_image(ax_head, np.asarray(head[start + t]))
        ax_head.text(
            0.5, -0.045, "Head camera", transform=ax_head.transAxes,
            ha="center", fontsize=11, color=INK_2,
        )

        gel_t = np.asarray(gel_arr[start + t])
        for k, (ax, name) in enumerate(zip(ax_gel, ("Thumb pad", "Finger pad"))):
            ax.clear()
            rounded_image(ax, gel_t[k])
            ax.text(
                0.5, -0.075, name, transform=ax.transAxes,
                ha="center", fontsize=10, color=INK_2,
            )

        # ---- headline: distance from ground truth, both checkpoints, over the episode ----
        ax_hero.clear()
        strip(ax_hero)
        for key in ("a", "b"):
            e = np.asarray(err[key])
            ax_hero.plot(anchors_arr, e, color=COL[key], linewidth=2.4, solid_capstyle="round")
            ax_hero.fill_between(anchors_arr, 0, e, color=COL[key], alpha=0.08)
        ax_hero.axvline(t, color=INK_3, linewidth=1.2)
        ax_hero.set_xlim(0, n - 1)
        ax_hero.set_ylim(0, float(np.nanmax([np.nanmax(err[k]) for k in err])) * 1.15 + 1e-6)
        ax_hero.set_xticks([])

        cur = {k: err[k][ai] for k in err}
        better = min(cur, key=lambda k: cur[k])
        gap = abs(cur["a"] - cur["b"])

        fig.text(0.442, 0.815, "Deviation from ground truth", fontsize=13, color=INK, fontweight="medium")
        fig.text(0.442, 0.788, "mean absolute arm-joint error over the predicted chunk (rad)",
                 fontsize=9.5, color=INK_2)
        for j, key in enumerate(("a", "b")):
            x = 0.700 + j * 0.135
            fig.text(x, 0.822, "●", fontsize=13, color=COL[key])
            fig.text(x + 0.016, 0.822, labels[key], fontsize=10.5, color=INK_2)
            fig.text(x, 0.788, f"{cur[key]:.4f}", fontsize=17, color=COL[key], fontweight="medium")
        # Sits in the card gutter below the chart so it never collides with the plotted lines.
        fig.text(
            0.442, 0.573,
            f"{labels[better]} closer by {gap:.4f} rad" if gap > 1e-6 else "tied",
            fontsize=10.5, color=COL[better] if gap > 1e-6 else INK_2,
        )

        # ---- small multiples: per-joint trajectories ----
        for d, ax in enumerate(sm_axes):
            ax.clear()
            strip(ax)
            lo, hi = max(0, t - 45), min(n, t + horizon_draw + 12)
            ax.plot(range(lo, hi), gt[lo:hi, d], color=INK_3, linewidth=2.6, solid_capstyle="round")
            for key in ("a", "b"):
                chunk = preds[key][anchor]
                xs = np.arange(anchor, min(anchor + chunk.shape[0], n))
                ax.plot(xs, chunk[: len(xs), d], color=COL[key], linewidth=1.9, solid_capstyle="round")
            ax.axvline(t, color=INK_3, linewidth=0.9, alpha=0.7)
            ax.set_xlim(lo, hi)
            ax.set_xticks([])
            ax.set_title(JOINT_LABELS[d], fontsize=10, color=INK_2, pad=5, loc="left")

        fig.text(0.442, 0.487, "Predicted chunk vs ground truth", fontsize=13, color=INK,
                 fontweight="medium")
        fig.text(0.688, 0.489, "●", fontsize=12, color=INK_3)
        fig.text(0.700, 0.489, "ground truth", fontsize=9.5, color=INK_2)

        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--label-a", default="FTP-1 baseline")
    p.add_argument("--label-b", default="PACT")
    p.add_argument("--domain", default="UniVTAC_lift_bottle")
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--horizon-draw", type=int, default=32)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    labels = {"a": args.label_a, "b": args.label_b}
    z = zarr.open_group(args.zarr, mode="r")
    episode_ends = z["meta"]["episode_ends"][:]

    episodes = args.episodes or json.load(open(args.split))["val_episode_idx"][:4]
    print(f"rendering held-out episodes {episodes}", flush=True)

    wrappers = {}
    for key, ckpt in (("a", args.ckpt_a), ("b", args.ckpt_b)):
        print(f"loading {labels[key]}: {ckpt}", flush=True)
        wrappers[key] = FTP1InferenceWrapper(
            checkpoint_dir=ckpt, domain_name=args.domain, device="cuda", num_inference_steps=10
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=args.fps, macro_block_size=None, quality=8) as writer:
        for ep in episodes:
            render_episode(writer, z, wrappers, labels, ep, episode_ends, args.stride, args.horizon_draw)

    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
