#!/usr/bin/env python
"""Is the expert action multimodal given the observation? The premise check for Drifting.

Drifting exists to preserve multiple valid action modes. If the data has one mode -- which is the
default expectation for scripted simulator demonstrations -- it has nothing to preserve, and a
drifting-vs-flow-matching comparison on that data cannot discriminate the two objectives no matter
how well either is trained.

The measurement is the one IDP itself computes. For each clip, weight every other clip in the pool
by observation similarity in the frozen encoder's embedding space, then take the weighted spread of
their expert action chunks:

    v_cond = sum_j w_ij (a*_j - a*_i)^2        spread among OBSERVATION-SIMILAR clips
    v_ref  = Var_i(a*)                          spread across everything

    ratio  = mean_d sqrt(v_cond_d / (2 * v_ref_d))

The factor 2 is not cosmetic: for independent draws E[(a_j - a_i)^2] = 2 Var(a), so an
UNINFORMATIVE embedding gives ratio 1.0 by construction. Reading the raw ratio without it makes
every dataset look multimodal.

    ratio -> 1   observation-similar clips disagree as much as random pairs. Either genuinely
                 multimodal, or the embedding does not capture what determines the action.
    ratio -> 0   similar observations imply similar actions: near-deterministic, one mode.

The disambiguation matters and is available: the readout gate independently established that a
ridge predicts the action chunk from these same embeddings at R^2 0.6-0.95 on UniVTAC. So the
embedding IS informative here, and a low ratio means determinism rather than a blind metric.

Read-only.
"""

from __future__ import annotations

import argparse
import collections
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

logger = logging.getLogger("mot_jepa.multimodality")


@torch.no_grad()
def collect(backbone, loader, device, max_batches: int) -> dict[int, tuple[list, list, list]]:
    out: dict[int, tuple[list, list, list]] = collections.defaultdict(lambda: ([], [], []))
    for seen, batch in enumerate(loader):
        if seen >= max_batches:
            break
        video = batch["video"].to(device).float().div_(127.5).sub_(1.0)
        gel = batch["gel"].to(device).float().div_(127.5).sub_(1.0)
        lowdim = batch["lowdim"].to(device).float()
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            encoded = backbone.encode_full(ClipInputs(video=video, gel=gel, lowdim=lowdim))
        pooled = torch.cat([r.float().mean(dim=1) for r in encoded.sync_readout], dim=-1).cpu()
        chunk = (batch["action_chunk"] * batch["chunk_mask"]).flatten(start_dim=1)
        mask = batch["chunk_mask"].flatten(start_dim=1)
        for row, domain in enumerate(batch["domain_id"].tolist()):
            out[domain][0].append(pooled[row].numpy())
            out[domain][1].append(chunk[row].numpy())
            out[domain][2].append(mask[row].numpy())
    return out


def ratio_for(
    embeddings: np.ndarray, actions: np.ndarray, mask: np.ndarray, *, top_k: int = 8, shuffle: bool = False
) -> tuple[float, int]:
    """Conditional spread as a fraction of the marginal, over live dimensions.

    ``top_k`` restricts the weighting to the k nearest clips. IDP's own row-standardised softmax is
    deliberately broad -- it spreads mass over most of the batch -- which is right for its geometry
    but wrong as a multimodality diagnostic: averaging over ~63 clips approaches the marginal
    variance by construction, so it would report "multimodal" on any dataset.

    ``shuffle`` randomises the embeddings, destroying the observation-action correspondence while
    leaving every marginal untouched. It is the control: a metric that cannot tell the real
    embedding from a shuffled one is measuring its own weighting, not the data.
    """
    live = mask.mean(axis=0) > 0.5
    if live.sum() == 0 or embeddings.shape[0] < 32:
        return float("nan"), embeddings.shape[0]
    actions = actions[:, live]
    if shuffle:
        embeddings = embeddings[np.random.default_rng(0).permutation(embeddings.shape[0])]

    normed = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -np.inf)
    keep = np.argsort(-similarity, axis=-1)[:, :top_k]
    weights = np.zeros_like(similarity)
    np.put_along_axis(weights, keep, 1.0 / top_k, axis=-1)

    delta = actions[None, :, :] - actions[:, None, :]  # (i, j, d)
    v_cond = np.einsum("ij,ijd->id", weights, delta**2)
    v_ref = actions.var(axis=0)
    good = v_ref > 1e-12
    if not good.any():
        return float("nan"), embeddings.shape[0]
    # sqrt of the variance ratio, so the number is a spread ratio rather than a variance ratio.
    per_dim = np.sqrt(np.maximum(v_cond[:, good], 0.0) / (2.0 * v_ref[good]))
    return float(per_dim.mean()), embeddings.shape[0]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_run", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained_step", type=int, default=None)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--config", default="mot_jepa_pilot")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-step", type=int, default=53)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_module.CONFIGS[args.config]
    backbone, step = runtime.load_frozen_backbone(args.pretrained_run, args.pretrained_step, cfg, device)

    stores = sorted(glob.glob(args.clips))
    names = sorted({pathlib.Path(p).parent.name for p in stores})
    domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]
    dataset = MotJepaClipDataset(
        stores,
        cfg.layout,
        domain_ids=domain_ids,
        strides=(1,),
        index_step=args.index_step,
        with_conditioning=True,
        action_horizon=args.horizon,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_clips
    )
    collected = collect(backbone, loader, device, args.batches)

    print(f"\nbackbone step {step}, horizon {args.horizon}\n")
    print(f"{'domain':26s} {'clips':>7s} {'real emb':>12s} {'SHUFFLED (ctl)':>16s} {'gap':>10s}")
    values, ctl_values = [], []
    for domain in sorted(collected):
        emb = np.stack(collected[domain][0])
        act = np.stack(collected[domain][1])
        msk = np.stack(collected[domain][2])
        r, n = ratio_for(emb, act, msk)
        c, _ = ratio_for(emb, act, msk, shuffle=True)
        if np.isfinite(r):
            values.append(r)
            ctl_values.append(c)
        print(f"{names[domain]:26s} {n:7d} {r:12.4f} {c:16.4f} {c - r:10.4f}")

    overall = float(np.mean(values)) if values else float("nan")
    control = float(np.mean(ctl_values)) if ctl_values else float("nan")
    print(f"\n{'MEAN over domains':26s} {'':7s} {overall:12.4f} {control:16.4f} {control - overall:10.4f}")
    print(
        "\nThe control shuffles the embeddings, destroying the observation-action correspondence\n"
        "while leaving every marginal intact. If real and shuffled agree, the metric is blind and\n"
        "neither number says anything about the data."
    )
    print(
        "\n1.0 = observation-similar clips disagree as much as random pairs (multimodal, or the\n"
        "embedding is blind).  Toward 0 = similar observations imply similar actions (one mode).\n"
    )
    print(
        "verdict: "
        + (
            "the expert action is largely DETERMINED by the observation. Drifting has little\n"
            "  multimodality to preserve here, so it cannot be expected to beat flow matching\n"
            "  on this data, and a null comparison is the premise failing rather than the method."
            if overall < 0.7
            else "observation-similar clips carry substantial action spread. There is genuine\n"
            "  multimodality for a generative objective to model."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
