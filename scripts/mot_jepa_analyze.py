"""Build the analysis report for a MoT-JEPA checkpoint.

Produces the figures that answer, in order: *is it binding?*, *does the correspondence look
right?*, and *is the representation healthy?* -- plus an ``index.html`` so the whole set can
be read in one scroll.

Deliberately does **not** read the training loss. V-JEPA 2 reports its loss is uncorrelated
with downstream accuracy, and a lower reconstruction loss can mean a *more* collapsed
representation, so every judgement here comes from the probes.

Usage::

    uv run python scripts/mot_jepa_analyze.py \
        --checkpoint .cache/mot_jepa/runs/mot_jepa_pilot/pilot01/checkpoints/50000 \
        --data-glob '/lustre/.../ftp1-clips/*/*.zarr' \
        --output reports/pilot01_50k
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("USE_SWANLAB", "false")

import numpy as np
import torch

from openpi.mot_jepa import viz
from openpi.mot_jepa.ablation import TactileMode
from openpi.mot_jepa.ablation import apply_tactile_ablation
from openpi.mot_jepa.ablation import donor_at_offset
from openpi.mot_jepa.ablation import summarize_ablation
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.config import MotJepaTrainConfig
from openpi.mot_jepa.losses import MotJepaLoss
from openpi.mot_jepa.losses import normalize_targets
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import build_batch_masks
from openpi.mot_jepa.model import ClipInputs
from openpi.mot_jepa.model import MotJepaStudent
from openpi.mot_jepa.probes import cross_modal_retrieval

FIGURES = [
    ("similarity", "Cross-modal similarity", "Diagonal should glow; off-diagonal should not."),
    ("sync_matrix", "Within-clip synchrony", "Diagonal band; its width is the temporal precision."),
    ("temporal_offset", "Temporal-offset sweep", "Rising and saturating = bound. Flat = unbound."),
    ("filmstrip", "Clip filmstrip", "Vision and touch on a shared time axis."),
    ("mask_panel", "Masking", "Which positions became targets, per stream."),
    ("ablation", "Tactile ablation", "Predicted ordering: real < zero <= shuffle <= noise."),
    ("spectrum", "Feature spectrum", "Flattening indicates collapse."),
]


def load_model(checkpoint: pathlib.Path, device: torch.device):
    config = MotJepaTrainConfig.from_json((checkpoint / "train_config.json").read_text())
    student = MotJepaStudent(
        config.layout, config.encoder, config.predictor, lowdim_channels=config.data.lowdim_channels
    ).to(device)
    student.load_state_dict(torch.load(checkpoint / "student.pt", map_location=device))
    student.eval()
    return student, config


def to_inputs(batch, device):
    return ClipInputs(
        video=batch["video"].to(device).float().div_(127.5).sub_(1.0),
        gel=batch["gel"].to(device).float().div_(127.5).sub_(1.0),
        lowdim=batch["lowdim"].to(device).float(),
    )


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="A checkpoints/<step> directory.")
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offsets", type=int, nargs="*", default=[1, 2, 5, 10, 30, 100])
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    student, config = load_model(args.checkpoint, device)
    layout = config.layout

    stores = sorted(glob.glob(args.data_glob))
    if not stores:
        parser.error(f"no *.zarr matched {args.data_glob}")
    dataset = MotJepaClipDataset(stores, layout, strides=(1,), index_step=max(1, len(stores) * 8))
    picks = np.linspace(0, len(dataset) - 1, args.batch_size).astype(int)
    batch = collate_clips([dataset[int(i)] for i in picks])
    inputs = to_inputs(batch, device)

    loss_fn = MotJepaLoss(config.loss, layout).to(device)
    encoded = student.backbone.encode_full(inputs)
    video_vec = loss_fn.projectors[0](encoded.sync_readout[0]).mean(dim=1)
    tactile_vec = loss_fn.projectors[1](encoded.sync_readout[1]).mean(dim=1)

    summary: dict[str, object] = {"checkpoint": str(args.checkpoint), "batch_size": args.batch_size}
    summary["retrieval"] = cross_modal_retrieval(video_vec, tactile_vec)

    viz.similarity_heatmap(video_vec, tactile_vec, path=args.output / "similarity.png")
    viz.sync_matrix(
        loss_fn.projectors[0](encoded.sync_readout[0])[0],
        loss_fn.projectors[1](encoded.sync_readout[1])[0],
        path=args.output / "sync_matrix.png",
    )
    viz.filmstrip(batch["video"][0], batch["gel"][0], path=args.output / "filmstrip.png")

    spec = MaskSpec(
        layout=layout,
        mode_probs=config.masking.mode_probs,
        tactile_window_steps=config.masking.tactile_window_steps,
        video_window_steps=config.masking.video_window_steps,
        min_targets_per_stream=config.masking.min_targets_per_stream,
    )
    masks = build_batch_masks(spec, step=0, batch_size=args.batch_size, base_seed=config.seed).to(device)
    viz.mask_panel(layout, masks.tgt_index[0].cpu(), masks.mode_enum.name, path=args.output / "mask_panel.png")

    # Tactile ablation on the representation: how far does the tactile embedding move?
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    reference = torch.nn.functional.normalize(tactile_vec.float(), dim=-1)
    ablation: dict[str, float] = {}
    for mode in TactileMode:
        ablated = ClipInputs(
            video=inputs.video,
            gel=apply_tactile_ablation(inputs.gel.cpu(), mode, generator=generator).to(device),
            lowdim=apply_tactile_ablation(inputs.lowdim.cpu(), mode, generator=generator).to(device),
        )
        out = student.backbone.encode_full(ablated)
        vector = torch.nn.functional.normalize(loss_fn.projectors[1](out.sync_readout[1]).mean(dim=1).float(), dim=-1)
        ablation[mode.value] = float((vector - reference).pow(2).sum(dim=-1).mean())
    summary["ablation"] = summarize_ablation(ablation)
    viz.ablation_bars(ablation, path=args.output / "ablation.png")

    # Temporal-offset sweep: donor from the same episode at a controlled offset.
    targets = normalize_targets(encoded.tokens[1])
    errors = [float((targets - donor_at_offset(targets, o, time_dim=0)).abs().mean()) for o in [0, *args.offsets]]
    summary["temporal_offset"] = dict(zip(["0", *map(str, args.offsets)], errors, strict=True))
    viz.temporal_offset_curve([0, *args.offsets], errors, path=args.output / "temporal_offset.png")

    viz.singular_value_spectrum(
        {"video": encoded.tokens[0].flatten(0, 1).cpu(), "tactile": encoded.tokens[1].flatten(0, 1).cpu()},
        path=args.output / "spectrum.png",
    )

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_index(args.output, summary)
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote report to {args.output}/index.html")
    return 0


def _write_index(output: pathlib.Path, summary: dict) -> None:
    cards = "\n".join(
        f'<section><h2>{title}</h2><p>{caption}</p><img src="{name}.png" loading="lazy"></section>'
        for name, title, caption in FIGURES
        if (output / f"{name}.png").exists()
    )
    output.joinpath("index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>MoT-JEPA report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;"
        "color:#1a1a1a}section{margin:2.5rem 0}img{max-width:100%;border:1px solid #e0e0e0;border-radius:6px}"
        "h2{margin-bottom:.2rem}p{color:#555;margin-top:0}pre{background:#f6f6f6;padding:1rem;border-radius:6px;"
        "overflow-x:auto;font-size:.85rem}</style>"
        f"<h1>MoT-JEPA report</h1><pre>{json.dumps(summary, indent=2, default=float)}</pre>{cards}"
    )


if __name__ == "__main__":
    sys.exit(main())
