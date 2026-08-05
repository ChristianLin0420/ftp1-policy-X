#!/usr/bin/env python
"""Can the action chunk be read out of a frozen MoT-JEPA latent? The policy's viability gate.

The Stage 4 gate measured R^2(dz <- [action|state]) -- "do actions explain latent *change*" --
and got -0.219/-0.242 pooled. That closed Stage 4, but it is the wrong direction for a policy.
A policy asks the reverse: given the latent, can the future action chunk be recovered. Those are
different questions and the first being null does not imply the second is.

So this fits, held out, per domain:

    a_{t..t+H}  <-  [ sync_readout_video | sync_readout_tactile ]

``sync_readout`` rather than the final tokens because the final tokens have already mixed the two
modalities through the global layers, and because the per-step readout is the same conditioning
sequence the DiT head will consume -- measuring anything else would gate a different model.

Ridge, held out, and per domain, for the reasons documented in the Stage 4 gate: 3840 action
dimensions against a few hundred clips makes an in-sample fit meaningless, the action columns are
strongly collinear (a gripper moves with its arm), and the domains are not comparable to each
other.

**Interpretation, agreed in advance:** R^2 >= 0.05 on any domain with >=500 clips means the latent
carries action-relevant information and the DiT head has something to learn from. R^2 ~ 0
everywhere means a ridge cannot read the action out, a DiT will not either, and the correct move
is to fix pretraining rather than build a policy on a representation that does not carry it.

Read-only. Runs under no_grad against a frozen snapshot.

    uv run python scripts/mot_jepa_action_readout.py \
        --pretrained_run .../backbones/probe2_s50000 --pretrained_step 50000 \
        --clips '.../ftp1-clips/*/*.zarr'
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

from openpi.mot_jepa import action_parse as ap
from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.model import ClipInputs

logger = logging.getLogger("mot_jepa.action_readout")

MIN_CLIPS = 500
"""Below this a domain produces spectacular numbers that are small-sample artefacts. The Stage 4
gate learned this the hard way from a 14-clip domain."""


def r_squared(features: np.ndarray, targets: np.ndarray, *, holdout: float = 0.3) -> tuple[float, int]:
    """Held-out R^2 of a ridge fit ``targets ~ features``.

    Held out rather than in-sample: with thousands of action dimensions and a few hundred clips,
    an in-sample fit reports a high R^2 from sheer capacity and says nothing. Ridge rather than
    plain least squares because the latent columns are strongly correlated and a singular normal
    matrix would otherwise decide the answer.
    """
    if features.shape[0] < 20:
        return float("nan"), features.shape[0]
    split = int(features.shape[0] * (1.0 - holdout))
    x_train, x_test = features[:split], features[split:]
    y_train, y_test = targets[:split], targets[split:]

    # Standardise on train statistics only; a constant column would otherwise dominate.
    mean, scale = x_train.mean(0), x_train.std(0)
    keep = scale > 1e-8
    if not keep.any():
        return float("nan"), features.shape[0]
    x_train = (x_train[:, keep] - mean[keep]) / scale[keep]
    x_test = (x_test[:, keep] - mean[keep]) / scale[keep]
    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

    ridge = 1e-3 * np.eye(x_train.shape[1])
    weights = np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)
    residual = ((y_test - x_test @ weights) ** 2).sum()
    total = ((y_test - y_train.mean(0)) ** 2).sum()
    return float(1.0 - residual / max(total, 1e-12)), features.shape[0]


@torch.no_grad()
def collect(backbone, loader, device, max_batches: int, horizon: int) -> dict[int, tuple[list, list]]:
    """Per domain: the frozen latent readout, and the future action chunk it should explain."""
    out: dict[int, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
    for seen, batch in enumerate(loader):
        if seen >= max_batches:
            break
        video = batch["video"].to(device).float().div_(127.5).sub_(1.0)
        gel = batch["gel"].to(device).float().div_(127.5).sub_(1.0)
        lowdim = batch["lowdim"].to(device).float()

        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            encoded = backbone.encode_full(ClipInputs(video=video, gel=gel, lowdim=lowdim))

        # Both experts, flattened over the tubelet axis: this is exactly the context the DiT head
        # will cross-attend to, so the gate measures the representation the policy will actually
        # receive rather than a more favourable summary of it.
        readout = torch.cat([expert.float().flatten(start_dim=1) for expert in encoded.sync_readout], dim=-1)
        features = readout.cpu().numpy()

        # The future chunk, derived at the clip's own stride from the per-frame state. The clip
        # dataset's own `action` field is retrospective and only 8 steps long, so it cannot answer
        # this question -- the chunk has to be built here from `state`.
        state = batch["state"].numpy()
        masks = batch["action_mask"].numpy()
        for row, domain in enumerate(batch["domain_id"].tolist()):
            chunk = _future_chunk(state[row], masks[row], horizon)
            if chunk is None:
                continue
            out[domain][0].append(features[row])
            out[domain][1].append(chunk.reshape(-1))
    return out


def _future_chunk(state: np.ndarray, mask: np.ndarray, horizon: int) -> np.ndarray | None:
    """``(num_frames, 120)`` state -> ``(horizon, 120)`` actions, edge-padded at the clip end.

    The clip carries ``num_frames`` states, so at most ``num_frames - 1`` real actions exist. The
    last action is repeated to reach the horizon, matching ``dataset_zarr.py:1885-1889``. That is
    a stopgap for the gate only -- Phase C reads a genuinely longer window from the store. It is
    honest here because it measures the *hardest* part of the horizon, the near-term actions, and
    padding cannot inflate R^2: repeated rows are perfectly predictable and would only be added to
    both sides of the fit.
    """
    if state.shape[0] < 2:
        return None
    actions = ap.actions_from_state(state, mask.astype(np.uint8))
    if actions.shape[0] == 0:
        return None
    if actions.shape[0] >= horizon:
        return actions[:horizon]
    pad = np.repeat(actions[-1:], horizon - actions.shape[0], axis=0)
    return np.concatenate([actions, pad], axis=0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_run", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained_step", type=int, default=None)
    parser.add_argument("--clips", required=True, help="glob for the derived *.zarr stores")
    parser.add_argument("--config", default="mot_jepa_pilot")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--batches", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_module.CONFIGS[args.config]
    backbone, step = runtime.load_frozen_backbone(args.pretrained_run, args.pretrained_step, cfg, device)

    stores = sorted(glob.glob(args.clips))
    names = sorted({pathlib.Path(p).parent.name for p in stores})
    domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]
    dataset = MotJepaClipDataset(
        stores, cfg.layout, domain_ids=domain_ids, strides=(1,), index_step=37, with_conditioning=True
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_clips
    )
    logger.info("%d stores / %d domains / %d clips", len(stores), len(names), len(dataset))

    collected = collect(backbone, loader, device, args.batches, args.horizon)

    print(f"\nbackbone step {step}, horizon {args.horizon}, {args.batches} batches of {args.batch_size}")
    print(f"\n{'domain':26s} {'clips':>7s} {'R^2 action<-latent':>20s}")
    rows, all_x, all_y = [], [], []
    for domain in sorted(collected):
        features = np.stack(collected[domain][0])
        targets = np.stack(collected[domain][1])
        score, count = r_squared(features, targets)
        rows.append((names[domain], score, count))
        all_x.append(features)
        all_y.append(targets)
        print(f"{names[domain]:26s} {count:7d} {score:20.4f}")

    pooled, pooled_n = r_squared(np.concatenate(all_x), np.concatenate(all_y))
    print(f"\n{'POOLED':26s} {pooled_n:7d} {pooled:20.4f}")

    solid = [(n, s) for n, s, c in rows if c >= MIN_CLIPS and np.isfinite(s)]
    best = max((s for _, s in solid), default=float("nan"))
    best_name = next((n for n, s in solid if s == best), "-")
    print(f"\nbest R^2 over domains with >={MIN_CLIPS} clips = {best:.4f}  ({best_name})")
    print(f"  ({len(solid)} of {len(rows)} domains cleared that bar)")
    print(
        "\nverdict: "
        + (
            "the frozen latent carries action-relevant information -- a DiT head has something "
            "to learn from. Proceed to the policy build."
            if best > 0.05
            else "a ridge cannot read the action out of the frozen latent. A DiT will not "
            "either. Fix pretraining (time-local L_sync) before building a policy on it."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
