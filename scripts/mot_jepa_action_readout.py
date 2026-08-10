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

from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.model import ClipInputs

logger = logging.getLogger("mot_jepa.action_readout")

MIN_CLIPS = 500
"""Below this a domain produces spectacular numbers that are small-sample artefacts. The Stage 4
gate learned this the hard way from a 14-clip domain."""


RIDGE_GRID = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
"""The readout has 4608 features against a few hundred to a few thousand clips, so ``p >> n`` and
the penalty is not a detail -- it decides the answer. A fixed 1e-3 (inherited from the Stage 4
probe, where features were only 240-D) interpolates the training set and reports a spectacular
negative held-out R^2 that says nothing about the encoder. The penalty is chosen on a validation
split, never on the test split."""


def r_squared(
    features: np.ndarray, targets: np.ndarray, episodes: np.ndarray | None = None, *, holdout: float = 0.3
) -> tuple[float, int, float]:
    """Held-out R^2 of a ridge fit ``targets ~ features``, penalty tuned on a validation split.

    Three-way split: fit on train, pick the penalty on validation, report on test. Picking the
    penalty on the test split would report the best of eight tries as if it were one measurement.

    Returns ``(r2, n, chosen_lambda)``.
    """
    count = features.shape[0]
    if count < 60:  # need three usable splits
        return float("nan"), count, float("nan")

    # Split by EPISODE. Splitting by row -- which is what a positional slice over a shuffled
    # loader does -- puts clips of the SAME episode on both sides: same scene, same object
    # placement, same motion signature. That inflates R^2 and is exactly what made the original
    # gate disagree with the held-out policy evaluation.
    if episodes is None:
        fit_end = int(count * (1.0 - 2 * holdout / 3))
        val_end = int(count * (1.0 - holdout / 3))
        order = np.arange(count)
    else:
        unique = np.unique(episodes)
        if unique.size < 6:
            return float("nan"), count, float("nan")
        cut_fit = int(unique.size * (1.0 - 2 * holdout / 3))
        cut_val = int(unique.size * (1.0 - holdout / 3))
        fit_eps, val_eps = set(unique[:cut_fit].tolist()), set(unique[cut_fit:cut_val].tolist())
        in_fit = np.array([e in fit_eps for e in episodes])
        in_val = np.array([e in val_eps for e in episodes])
        order = np.concatenate([np.flatnonzero(in_fit), np.flatnonzero(in_val), np.flatnonzero(~in_fit & ~in_val)])
        fit_end, val_end = int(in_fit.sum()), int(in_fit.sum() + in_val.sum())
    features, targets = features[order], targets[order]
    x_fit, x_val, x_test = features[:fit_end], features[fit_end:val_end], features[val_end:]
    y_fit, y_val, y_test = targets[:fit_end], targets[fit_end:val_end], targets[val_end:]
    if min(x_val.shape[0], x_test.shape[0]) < 10:
        return float("nan"), count, float("nan")

    # Standardise on fit statistics only; a constant column would otherwise dominate.
    mean, scale = x_fit.mean(0), x_fit.std(0)
    keep = scale > 1e-8
    if not keep.any():
        return float("nan"), count, float("nan")

    def prep(x: np.ndarray) -> np.ndarray:
        z = (x[:, keep] - mean[keep]) / scale[keep]
        return np.hstack([z, np.ones((z.shape[0], 1))])

    x_fit, x_val, x_test = prep(x_fit), prep(x_val), prep(x_test)
    gram = x_fit.T @ x_fit
    rhs = x_fit.T @ y_fit
    eye = np.eye(gram.shape[0])

    def score(x: np.ndarray, y: np.ndarray, weights: np.ndarray, baseline: np.ndarray) -> float:
        residual = ((y - x @ weights) ** 2).sum()
        total = ((y - baseline) ** 2).sum()
        return float(1.0 - residual / max(total, 1e-12))

    best_lambda, best_val, best_weights = float("nan"), -np.inf, None
    for lam in RIDGE_GRID:
        weights = np.linalg.solve(gram + lam * eye, rhs)
        val = score(x_val, y_val, weights, y_fit.mean(0))
        if val > best_val:
            best_lambda, best_val, best_weights = lam, val, weights

    return score(x_test, y_test, best_weights, y_fit.mean(0)), count, best_lambda


@torch.no_grad()
def collect(
    backbone, loader, device, max_batches: int, *, layernorm_features: bool = False
) -> dict[int, tuple[list, list, list, list]]:
    """Per domain: the frozen readout, the future action chunk, and the positive-control state.

    The third target is the control. The clip's own proprioceptive state is fed straight into the
    encoder through the lowdim stream, so a readout that cannot recover it is a broken estimator,
    not an uninformative encoder. Without it a negative action R^2 is uninterpretable -- exactly
    the trap the timeshuffle probe fell into.
    """
    out: dict[int, tuple[list, list, list, list]] = collections.defaultdict(lambda: ([], [], [], []))
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
        if layernorm_features:
            readout = torch.nn.functional.layer_norm(readout, (readout.shape[-1],))
        features = readout.cpu().numpy()

        # The genuine future window, built by the dataset at this clip's own stride and guaranteed
        # by the index to lie inside the episode. The dataset's `action` field cannot answer this
        # question: it is retrospective over the observed clip, not the future.
        chunks = (batch["action_chunk"] * batch["chunk_mask"]).numpy()
        masks = batch["action_mask"].numpy()
        anchors = batch["state"][:, :: 2].numpy()  # tubelet anchors, the POSITIVE CONTROL target
        store = batch["store_idx"].tolist()
        episode = batch["episode_idx"].tolist()
        for row, domain in enumerate(batch["domain_id"].tolist()):
            out[domain][0].append(features[row])
            out[domain][1].append(chunks[row].reshape(-1))
            out[domain][2].append((anchors[row] * masks[row]).reshape(-1))
            # Globally unique per (store, episode) so the split cannot merge episode 0 of two
            # different stores into one group.
            out[domain][3].append(store[row] * 1_000_000 + episode[row])
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_run", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained_step", type=int, default=None)
    parser.add_argument("--clips", required=True, help="glob for the derived *.zarr stores")
    parser.add_argument("--config", default="mot_jepa_pilot")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--batches", type=int, default=192)
    parser.add_argument(
        "--layernorm-features",
        action="store_true",
        help=(
            "Apply LayerNorm to the readout before fitting -- per SAMPLE across all 4608 dims, "
            "discarding that sample's mean and magnitude. This reproduces what LinearHead and the "
            "DiT's context_norm do, and asks in closed form whether that transform is what costs "
            "the trained heads their accuracy. No training required."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
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
        index_step=37,
        with_conditioning=True,
        action_horizon=args.horizon,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_clips
    )
    logger.info("%d stores / %d domains / %d clips", len(stores), len(names), len(dataset))

    collected = collect(backbone, loader, device, args.batches, layernorm_features=args.layernorm_features)

    print(f"\nbackbone step {step}, horizon {args.horizon}, {args.batches} batches of {args.batch_size}"
          f"{'  [LayerNorm features]' if args.layernorm_features else ''}")
    print(f"\n{'domain':26s} {'clips':>7s} {'R2 action':>11s} {'lambda':>9s} {'R2 state(ctl)':>14s}")
    rows, all_x, all_y, all_c, all_e = [], [], [], [], []
    for domain in sorted(collected):
        features = np.stack(collected[domain][0])
        targets = np.stack(collected[domain][1])
        control = np.stack(collected[domain][2])
        eps = np.asarray(collected[domain][3])
        score, count, lam = r_squared(features, targets, eps)
        ctl, _, _ = r_squared(features, control, eps)
        rows.append((names[domain], score, count, ctl))
        all_x.append(features)
        all_y.append(targets)
        all_c.append(control)
        all_e.append(eps)
        print(f"{names[domain]:26s} {count:7d} {score:11.4f} {lam:9.0e} {ctl:14.4f}")

    x, y, c = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_c)
    e = np.concatenate(all_e)
    pooled, pooled_n, pooled_lam = r_squared(x, y, e)
    pooled_ctl, _, _ = r_squared(x, c, e)
    print(f"\n{'POOLED':26s} {pooled_n:7d} {pooled:11.4f} {pooled_lam:9.0e} {pooled_ctl:14.4f}")

    solid = [(n, s, ctl) for n, s, cnt, ctl in rows if cnt >= MIN_CLIPS and np.isfinite(s)]
    best = max((s for _, s, _ in solid), default=float("nan"))
    best_name = next((n for n, s, _ in solid if s == best), "-")
    best_ctl = max((ctl for _, _, ctl in solid if np.isfinite(ctl)), default=float("nan"))
    print(f"\nbest R2(action) over domains with >={MIN_CLIPS} clips = {best:.4f}  ({best_name})")
    print(f"best R2(state) -- the positive control -- over the same domains = {best_ctl:.4f}")
    print(f"  ({len(solid)} of {len(rows)} domains cleared that bar)")

    # Order matters: a failed control invalidates the action number entirely, so it is checked
    # first. Reporting "no action signal" from an estimator that cannot recover the state either
    # would be a null with no evidentiary value.
    print("\nverdict: ")
    if not np.isfinite(best_ctl) or best_ctl < 0.2:
        print("  CONTROL FAILED. The readout cannot recover the clip's own state, which is fed")
        print("  straight into the encoder via the lowdim stream. The estimator is broken, not")
        print("  the encoder -- the action number below it means nothing. Fix the probe.")
    elif best > 0.05:
        print("  the frozen latent carries action-relevant information and the control passes.")
        print("  A DiT head has something to learn from. Proceed to the policy build.")
    else:
        print("  the control passes but the action does not: the readout recovers the state and")
        print("  still cannot recover the action. That is a real null about the encoder. Fix")
        print("  pretraining (time-local L_sync) before building a policy on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
