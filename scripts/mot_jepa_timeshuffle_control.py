#!/usr/bin/env python
"""Positive control for ``timeshuffle_gap``: can that probe detect temporal order at all?

Both probe2 (L1) and probe3 (direction) report ``timeshuffle_gap`` indistinguishable from zero
-- +1.2 and +1.3 per 256 candidates respectively, across tens of thousands of steps. Read
naively that says the encoder matches video to tactile on marginal statistics rather than
temporal correspondence, which is the exact FTP-1 failure the architecture exists to remove.

That reading was never validated. The probe has no positive control anywhere in the repo, and
two structural features make a null the *expected* output regardless of what the encoder learned:

1. ``probes._permute_time`` draws ONE permutation per call and applies it to every clip in the
   batch (``probes.py:103-107``). ``timeshuffle_gap`` is then measured by retrieval, so the
   matching clip and all 255 distractors are perturbed the same way. A globally consistent time
   relabeling can leave the *relative* geometry -- and therefore the ranking -- intact even for
   an encoder that is fully order-sensitive.

2. The clip vector is ``sync_readout.mean(dim=1)`` (``probes.py:170``), and ``sync_readout`` is
   ``(B, num_steps, width)`` (``mot_encoder.py:181``). That mean runs over the time axis, and a
   mean over time is exactly permutation-invariant. Only cross-time attention *before* the
   readout can make a shuffle visible at all.

So this script does not measure retrieval. It measures whether the representation MOVES, which
is the precondition for any retrieval gap to be possible, and it separates the two candidate
explanations that the single ``timeshuffle_gap`` number cannot:

    pooled displacement   -- what the probe actually sees (mean over time)
    unpooled displacement -- what the encoder computes (per-step, time axis intact)

Each is reported as a fraction of the natural between-clip spread, because a bare cosine is
uninterpretable: moving 0.01 is large if different clips sit 0.02 apart and negligible if they
sit 0.9 apart. Both are computed under the shared permutation the probe uses AND under a
per-sample permutation, which is the control the probe should arguably have used.

Reading the output:

  unpooled LOW, pooled LOW      the encoder is genuinely order-blind. The probe was right and
                                the tactile objective has not fixed the FTP-1 failure.
  unpooled HIGH, pooled LOW     the encoder tracks temporal order and the mean-over-time readout
                                discards it. ``timeshuffle_gap`` is measuring the readout, not
                                the encoder, and tonight's negative result does not stand.
  unpooled HIGH, pooled HIGH    the representation moves and the probe should have been able to
                                see it; the confound is then the shared permutation, and the
                                per-sample column will show it.

Read-only. Loads a frozen EMA backbone from a snapshot and runs under ``no_grad``.

    uv run python scripts/mot_jepa_timeshuffle_control.py \
        --pretrained_run /lustre/.../ftp1-runs/backbones/probe2_s50000 \
        --clips '/lustre/.../ftp1-clips/*/*.zarr'
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pathlib

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("USE_SWANLAB", "false")

from openpi.mot_jepa import config as config_module
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.model import ClipInputs
from scripts.mot_jepa_action_probe import load_backbone

logger = logging.getLogger("timeshuffle_control")


def permute_time_shared(tensor: torch.Tensor, *, seed: int) -> torch.Tensor:
    """The probe's own control: one permutation, reused for every clip in the batch."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(tensor.shape[1], generator=generator).to(tensor.device)
    return tensor.index_select(1, order).contiguous()


def permute_time_per_sample(tensor: torch.Tensor, *, seed: int) -> torch.Tensor:
    """A distinct permutation per clip, so no shared time relabeling survives across candidates.

    Still fully determined by ``seed``, so the number stays comparable across invocations -- the
    property the shared version's docstring cites, obtained without the shared-permutation
    confound.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, steps = tensor.shape[0], tensor.shape[1]
    orders = torch.stack([torch.randperm(steps, generator=generator) for _ in range(batch)])
    shape = [batch, steps] + [1] * (tensor.dim() - 2)
    index = orders.view(shape).expand_as(tensor).to(tensor.device)
    return torch.gather(tensor, 1, index)


def _cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.to(torch.float32), dim=-1)
    b = torch.nn.functional.normalize(b.to(torch.float32), dim=-1)
    return (a * b).sum(dim=-1)


def between_clip_distance(vectors: torch.Tensor) -> float:
    """Mean ``1 - cos`` between DIFFERENT clips: the natural spread of the representation.

    This is the yardstick. A shuffle-induced displacement only means something relative to how
    far apart unrelated clips already sit.
    """
    normed = torch.nn.functional.normalize(vectors.to(torch.float32), dim=-1)
    similarity = normed @ normed.t()
    count = similarity.shape[0]
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    return float((1.0 - similarity[off_diagonal]).mean())


@torch.no_grad()
def collect(backbone, loader, device, num_batches: int) -> dict[str, list[float]]:
    """Displacement under each shuffle, pooled and unpooled, in units of between-clip spread."""
    out: dict[str, list[float]] = {key: [] for key in (
        "pooled_shared", "pooled_persample", "unpooled_shared", "unpooled_persample",
        "pooled_spread", "unpooled_spread",
    )}

    for index, batch in enumerate(loader):
        if index >= num_batches:
            break
        inputs = ClipInputs(
            video=batch["video"].to(device).float().div_(127.5).sub_(1.0),
            gel=batch["gel"].to(device).float().div_(127.5).sub_(1.0),
            lowdim=batch["lowdim"].to(device).float(),
        )

        def tactile_readout(clip: ClipInputs) -> torch.Tensor:
            # Expert 1 is the tactile expert; (B, num_steps, width).
            return backbone.encode_full(clip).sync_readout[1].float()

        real = tactile_readout(inputs)
        shared = tactile_readout(
            ClipInputs(inputs.video, permute_time_shared(inputs.gel, seed=0),
                       permute_time_shared(inputs.lowdim, seed=0))
        )
        per_sample = tactile_readout(
            ClipInputs(inputs.video, permute_time_per_sample(inputs.gel, seed=0),
                       permute_time_per_sample(inputs.lowdim, seed=0))
        )

        # Pooled: exactly what the probe compares -- mean over the time axis.
        pooled = {name: value.mean(dim=1) for name, value in
                  (("real", real), ("shared", shared), ("per_sample", per_sample))}
        # Unpooled: the same content with the time axis kept, flattened so a single cosine sees
        # per-step structure. This is what the encoder produced before the readout averaged it.
        flat = {name: value.flatten(start_dim=1) for name, value in
                (("real", real), ("shared", shared), ("per_sample", per_sample))}

        pooled_spread = between_clip_distance(pooled["real"])
        flat_spread = between_clip_distance(flat["real"])
        out["pooled_spread"].append(pooled_spread)
        out["unpooled_spread"].append(flat_spread)
        for tag, name in (("shared", "shared"), ("persample", "per_sample")):
            out[f"pooled_{tag}"].append(float((1.0 - _cosine_rows(pooled["real"], pooled[name])).mean()))
            out[f"unpooled_{tag}"].append(float((1.0 - _cosine_rows(flat["real"], flat[name])).mean()))

        if index == 0:
            logger.info("readout %s -> pooled %s, flat %s",
                        tuple(real.shape), tuple(pooled["real"].shape), tuple(flat["real"].shape))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_run", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained_step", type=int, default=None)
    parser.add_argument("--clips", required=True, help="glob for the derived *.zarr stores")
    parser.add_argument("--config", default="mot_jepa_pilot")
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_module.CONFIGS[args.config]
    backbone, step = load_backbone(args.pretrained_run, args.pretrained_step, cfg, device)

    stores = sorted(glob.glob(args.clips))
    names = sorted({pathlib.Path(p).parent.name for p in stores})
    domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]
    dataset = MotJepaClipDataset(stores, cfg.layout, domain_ids=domain_ids, strides=(1,), index_step=97)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_clips
    )
    logger.info("%d stores / %d domains / %d clips", len(stores), len(names), len(dataset))

    results = collect(backbone, loader, device, args.batches)

    pooled_spread = float(np.mean(results["pooled_spread"]))
    flat_spread = float(np.mean(results["unpooled_spread"]))
    print(f"\nbackbone step {step}, {args.batches} batches of {args.batch_size}")
    print(f"between-clip spread (1-cos):  pooled {pooled_spread:.5f}   unpooled {flat_spread:.5f}")
    print(f"\n{'readout':12s} {'shuffle':12s} {'displacement':>14s} {'/ spread':>12s}")
    verdict: dict[str, float] = {}
    for readout, spread in (("pooled", pooled_spread), ("unpooled", flat_spread)):
        for shuffle in ("shared", "persample"):
            value = float(np.mean(results[f"{readout}_{shuffle}"]))
            ratio = value / spread if spread > 0 else float("nan")
            verdict[f"{readout}_{shuffle}"] = ratio
            print(f"{readout:12s} {shuffle:12s} {value:14.5f} {ratio:12.3f}")

    # The precondition for a retrieval gap: shuffling must move a clip meaningfully relative to
    # how far apart unrelated clips sit. Below ~0.1 the representation has barely budged and no
    # retrieval-based probe could have resolved a gap, whatever the encoder learned.
    pooled_blind = verdict["pooled_persample"] < 0.1
    encoder_sees = verdict["unpooled_persample"] >= 0.1
    print("\nverdict:")
    if pooled_blind and not encoder_sees:
        print("  the encoder is order-blind -- shuffling moves nothing, pooled or not.")
        print("  timeshuffle_gap ~ 0 is a real finding and the tactile objective has not fixed it.")
    elif pooled_blind and encoder_sees:
        print("  the ENCODER tracks temporal order but the mean-over-time readout discards it.")
        print("  timeshuffle_gap measures the readout, not the encoder. The probe2/probe3 null")
        print("  does NOT support the claim that the encoder ignores temporal correspondence.")
    else:
        print("  the pooled representation does move under shuffling, so a retrieval gap was")
        print("  resolvable. Compare the shared and persample rows: a large difference means the")
        print("  shared permutation was the confound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
