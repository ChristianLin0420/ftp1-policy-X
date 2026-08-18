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
from openpi.mot_jepa import runtime
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.model import ClipInputs

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


def temporal_variation(tensor: torch.Tensor) -> float:
    """Fraction of a clip's magnitude that varies over time: ``||x - mean_t x|| / ||x||``.

    The denominator the latent measurement was missing. If the gel frames inside a clip are
    near-duplicates -- static contact, slow manipulation, too short a temporal span -- then
    shuffling them barely changes the input, an order-blind encoder is the CORRECT answer, and
    the defect is the clip configuration rather than the objective. Without this number a small
    latent displacement cannot distinguish "the encoder discards order" from "there was no order
    information in the input to begin with".
    """
    residual = tensor - tensor.mean(dim=1, keepdim=True)
    scale = tensor.flatten(start_dim=1).norm(dim=-1)
    return float((residual.flatten(start_dim=1).norm(dim=-1) / scale.clamp_min(1e-8)).mean())


def _time_residual(tensor: torch.Tensor) -> torch.Tensor:
    """Per-clip time-varying component, flattened.

    Shuffling frames leaves a clip's temporal mean untouched, so centring is identical for the
    real and shuffled versions and the comparison isolates exactly the part order can affect.
    Measuring raw pixels instead would let the shared DC component dominate both the
    displacement and the spread.
    """
    return (tensor - tensor.mean(dim=1, keepdim=True)).flatten(start_dim=1)


#: Both readouts, measured on the SAME clips in one pass so they are directly comparable.
#:
#: ``sync_readout`` is snapshotted at layer 4 of 12 (``mot_encoder.py:249``), before the first
#: global layer; ``final_readout`` is the layer-12 pooled output and is what every downstream
#: consumer deploys. Measuring only ``sync`` would risk a false null on the one number that gates
#: everything else: the forecasting gradient arrives via layer-12 tokens, and nothing forbids the
#: encoder from satisfying mode F inside layers 5-11 while leaving the layer-4 readout a pure
#: per-step appearance code. That would read as "forecasting did nothing" about a change that
#: worked. ``sync`` is kept because the pre-registered 0.005 baseline was measured on it.
READOUTS = ("sync", "final")


@torch.no_grad()
def collect(backbone, loader, device, num_batches: int) -> dict[str, list[float]]:
    """Displacement under each shuffle, per readout and pooled/unpooled, in units of spread."""
    keys = [
        f"{readout}_{pooling}_{suffix}"
        for readout in READOUTS
        for pooling in ("pooled", "unpooled")
        for suffix in ("shared", "persample", "spread")
    ]
    out: dict[str, list[float]] = {key: [] for key in (
        *keys, "input_persample", "input_spread", "gel_temporal_variation",
    )}

    for index, batch in enumerate(loader):
        if index >= num_batches:
            break
        inputs = ClipInputs(
            video=batch["video"].to(device).float().div_(127.5).sub_(1.0),
            gel=batch["gel"].to(device).float().div_(127.5).sub_(1.0),
            lowdim=batch["lowdim"].to(device).float(),
        )

        def tactile_readouts(clip: ClipInputs) -> dict[str, torch.Tensor]:
            # Expert 1 is the tactile expert; each is (B, num_steps, width). One forward, both
            # readouts, so sync and final are never compared across different clips.
            encoded = backbone.encode_full(clip)
            return {
                "sync": encoded.sync_readout[1].float(),
                "final": encoded.final_readout[1].float(),
            }

        variants = {
            "real": tactile_readouts(inputs),
            "shared": tactile_readouts(
                ClipInputs(inputs.video, permute_time_shared(inputs.gel, seed=0),
                           permute_time_shared(inputs.lowdim, seed=0))
            ),
            "per_sample": tactile_readouts(
                ClipInputs(inputs.video, permute_time_per_sample(inputs.gel, seed=0),
                           permute_time_per_sample(inputs.lowdim, seed=0))
            ),
        }

        for readout in READOUTS:
            # Pooled: exactly what the probe compares -- mean over the time axis.
            pooled = {name: v[readout].mean(dim=1) for name, v in variants.items()}
            # Unpooled: the same content with the time axis kept, flattened so a single cosine
            # sees per-step structure -- what the encoder produced before the readout averaged it.
            flat = {name: v[readout].flatten(start_dim=1) for name, v in variants.items()}

            out[f"{readout}_pooled_spread"].append(between_clip_distance(pooled["real"]))
            out[f"{readout}_unpooled_spread"].append(between_clip_distance(flat["real"]))
            for tag, name in (("shared", "shared"), ("persample", "per_sample")):
                out[f"{readout}_pooled_{tag}"].append(
                    float((1.0 - _cosine_rows(pooled["real"], pooled[name])).mean())
                )
                out[f"{readout}_unpooled_{tag}"].append(
                    float((1.0 - _cosine_rows(flat["real"], flat[name])).mean())
                )

        # Input side, on the same clips and the same per-sample permutation: how much does the
        # SHUFFLE ITSELF change what the encoder was given? This is the denominator for every
        # latent number above.
        gel_shuffled = permute_time_per_sample(inputs.gel, seed=0)
        residual_real = _time_residual(inputs.gel)
        out["gel_temporal_variation"].append(temporal_variation(inputs.gel))
        out["input_spread"].append(between_clip_distance(residual_real))
        out["input_persample"].append(
            float((1.0 - _cosine_rows(residual_real, _time_residual(gel_shuffled))).mean())
        )

        if index == 0:
            for readout in READOUTS:
                logger.info("%s readout %s", readout, tuple(variants["real"][readout].shape))
            logger.info("gel %s", tuple(inputs.gel.shape))
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
    parser.add_argument(
        "--strides", type=int, nargs="+", default=None,
        help="Frame strides to probe at. Defaults to the run's own training strides.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_module.CONFIGS[args.config]
    backbone, step = runtime.load_frozen_backbone(args.pretrained_run, args.pretrained_step, cfg, device)

    stores = sorted(glob.glob(args.clips))
    names = sorted({pathlib.Path(p).parent.name for p in stores})
    domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]
    # MUST match the stride the backbone was TRAINED at. RoPE `t` is the tubelet index and
    # carries no rate, so an encoder handed a different frame rate cannot tell -- it simply sees
    # a slower or faster world, with no error anywhere. Probing a (2,4)-trained encoder at stride
    # 1 puts it 2-4x outside its training distribution, and a null there says nothing about
    # whether it represents dynamics at the rate it actually saw.
    strides = args.strides or tuple(runtime.architecture_from_run(args.pretrained_run, cfg).data.strides)
    logger.info("probing at strides %s (training strides for this run)", strides)
    dataset = MotJepaClipDataset(stores, cfg.layout, domain_ids=domain_ids, strides=strides, index_step=97)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_clips
    )
    logger.info("%d stores / %d domains / %d clips", len(stores), len(names), len(dataset))

    results = collect(backbone, loader, device, args.batches)

    print(f"\nbackbone step {step}, {args.batches} batches of {args.batch_size}")
    print(f"\n{'layer':7s} {'pooling':10s} {'shuffle':11s} {'displacement':>14s} {'/ spread':>12s}")
    verdict: dict[str, float] = {}
    for readout in READOUTS:
        for pooling in ("pooled", "unpooled"):
            spread = float(np.mean(results[f"{readout}_{pooling}_spread"]))
            for shuffle in ("shared", "persample"):
                value = float(np.mean(results[f"{readout}_{pooling}_{shuffle}"]))
                ratio = value / spread if spread > 0 else float("nan")
                verdict[f"{readout}_{pooling}_{shuffle}"] = ratio
                print(f"{readout:7s} {pooling:10s} {shuffle:11s} {value:14.5f} {ratio:12.3f}")

    # The input side, on the same clips and the same permutation: the denominator for all of the
    # above. A latent that does not move is only evidence about the ENCODER if the shuffle moved
    # what the encoder was given.
    input_spread = float(np.mean(results["input_spread"]))
    input_displacement = float(np.mean(results["input_persample"]))
    input_ratio = input_displacement / input_spread if input_spread > 0 else float("nan")
    variation = float(np.mean(results["gel_temporal_variation"]))
    verdict["input_persample"] = input_ratio
    print(f"{'input':7s} {'(gel)':10s} {'persample':11s} {input_displacement:14.5f} {input_ratio:12.3f}")
    print(f"\ngel temporal variation ||x - mean_t x|| / ||x||:  {variation:.5f}")

    # Below ~0.1 a quantity has barely budged relative to the natural spread, so no
    # retrieval-based probe could have resolved a gap in it whatever the encoder learned.
    #
    # Judged on the BEST of the two readouts, not on `sync` alone. The question is whether the
    # ENCODER represents order anywhere; if only the layer-12 readout moves, order is present and
    # the layer-4 snapshot is simply the wrong place to look for it. Reporting the sync number
    # alone would turn that into a false null on the measurement that gates everything else.
    input_moves = input_ratio >= 0.1
    pooled_blind = max(verdict[f"{r}_pooled_persample"] for r in READOUTS) < 0.1
    encoder_sees = max(verdict[f"{r}_unpooled_persample"] for r in READOUTS) >= 0.1
    moved = [r for r in READOUTS if verdict[f"{r}_unpooled_persample"] >= 0.1]
    print("\nverdict:")
    if moved:
        print(f"  order is visible in: {', '.join(moved)} (of {', '.join(READOUTS)})")
    if not input_moves:
        print("  THE INPUT ITSELF BARELY CHANGES under a full temporal scramble. The frames in a")
        print("  clip are near-duplicates, so an order-blind encoder is the CORRECT answer and")
        print("  timeshuffle_gap ~ 0 says nothing about the objective. The defect is the clip")
        print("  temporal span -- widen stride or frame count and rebuild before touching losses.")
    elif pooled_blind and not encoder_sees:
        print("  the input changes but the encoder is order-blind -- it discards order that was")
        print("  present. timeshuffle_gap ~ 0 is a real finding about the OBJECTIVE: nothing in")
        print("  the loss rewards temporal correspondence. Clip-pooled L_sync is order-invariant")
        print("  after pooling by construction; a time-local sync term is the targeted fix.")
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
