#!/usr/bin/env python
"""Per-domain ridge policy, fitted through the real pipeline. The deployable floor.

Four trained heads -- linear, drifting, flowmatch, and a weight-decay sweep over all four settings
of the linear one -- all lose to predicting a per-domain constant on held-out UniVTAC. A ridge on
the same frozen features, measured by the readout gate, reaches R^2 0.65-0.93. This closes the last
gap between those two facts by fitting the ridge on **our own data path**: the same
MotJepaClipDataset, the same episode-level split, the same ActionNormalizer targets, and the same
RMSE the policy evaluator reports.

Two jobs at once.

**Diagnostic.** If this reaches ridge-like accuracy, the pipeline is sound and every remaining gap
belongs to the neural head's optimisation. If it does not, then the gate and this pipeline differ
somewhere still unfound, and the head-versus-ridge comparison was never like-for-like.

**Deliverable.** A per-domain closed-form policy is a real, deployable artefact: it predicts an
action chunk from a frozen encoder with no training loop. It is the floor any neural head must
clear, and having it on record beats having four heads that lose to a constant.

Fits one independent map per domain, full rank, with the penalty chosen on a validation split --
the three things the trained heads do not do. Saves the weights so the eval harness can load them.

    uv run python scripts/mot_jepa_ridge_policy.py \
        --pretrained_run .../backbones/probe2_s50000 --pretrained_step 50000 \
        --clips '.../univtac-clips/*/*.zarr' --out .../ridge_policy.npz
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

from openpi.ftp1_action_groups import get_ftp1_action_group_slices
from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import split_clip_index
from openpi.mot_jepa.model import ClipInputs

logger = logging.getLogger("mot_jepa.ridge_policy")

GROUPS = get_ftp1_action_group_slices(48, 7, 15)
RIDGE_GRID = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


@torch.no_grad()
def collect(backbone, loader, device, max_batches: int) -> dict[int, dict[str, list]]:
    """Frozen readout, action chunk and mask, grouped by domain."""
    out: dict[int, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for seen, batch in enumerate(loader):
        if seen >= max_batches:
            break
        video = batch["video"].to(device).float().div_(127.5).sub_(1.0)
        gel = batch["gel"].to(device).float().div_(127.5).sub_(1.0)
        lowdim = batch["lowdim"].to(device).float()
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            encoded = backbone.encode_full(ClipInputs(video=video, gel=gel, lowdim=lowdim))
        features = torch.cat([r.float().flatten(start_dim=1) for r in encoded.final_readout], dim=-1).cpu().numpy()
        chunk = batch["action_chunk"].numpy()
        mask = batch["chunk_mask"].numpy()
        for row, domain in enumerate(batch["domain_id"].tolist()):
            out[domain]["x"].append(features[row])
            out[domain]["y"].append(chunk[row].reshape(-1))
            out[domain]["m"].append(mask[row].reshape(-1))
    return out


def fit_one_domain(x: np.ndarray, y: np.ndarray, *, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ridge with the penalty chosen on a validation split. Returns (mean, scale, weights, lambda).

    The penalty is the whole game at ``p >> n``: 4608 features against a few hundred clips. A fixed
    value interpolates the training set and generalises at random, which is what the very first
    version of the readout gate did before the grid was added.
    """
    cut = int(x.shape[0] * (1.0 - val_frac))
    x_fit, x_val = x[:cut], x[cut:]
    y_fit, y_val = y[:cut], y[cut:]

    mean, scale = x_fit.mean(0), x_fit.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)

    def prep(a: np.ndarray) -> np.ndarray:
        return np.hstack([(a - mean) / scale, np.ones((a.shape[0], 1))])

    xf, xv = prep(x_fit), prep(x_val)
    gram, rhs = xf.T @ xf, xf.T @ y_fit
    eye = np.eye(gram.shape[0])

    best = (np.inf, None, float("nan"))
    for lam in RIDGE_GRID:
        weights = np.linalg.solve(gram + lam * eye, rhs)
        err = float(((y_val - xv @ weights) ** 2).mean())
        if err < best[0]:
            best = (err, weights, lam)
    return mean, scale, best[1], best[2]


def feature_share(weights: np.ndarray, x_std: np.ndarray) -> tuple[float, float]:
    """How much of the commanded action responds to the OBSERVATION, two ways.

    ``weights`` is ``(p + 1, out)`` with the intercept as its last row, and the features were
    standardised before fitting, so the two blocks' norms are directly comparable.

    This is the measurement that explains every other number in the offline evaluation. At 33% on
    ``lift_bottle`` two thirds of the command is a learned constant, and then: the ridge beats a
    per-domain constant by only 24%, no trained head beats the constant at all, integrated drift is
    89% correlated (a constant bias integrates linearly), and closed-loop failures retract away
    from the object and time out while successes are fast. An order-blind representation cannot
    separate approach from lift from retract at one visual pose, so the best available policy is
    the AVERAGE action at that pose -- which is what an intercept is.

    Returns ``(by_weight_norm, by_prediction)``:

    * ``by_weight_norm`` -- ||W_features|| / (||W_features|| + ||intercept||). The quantity the
      plan pre-registered, so this is the one to compare against the 33% baseline.
    * ``by_prediction`` -- ||pred - intercept|| / ||pred|| on the holdout itself. Less
      assumption-laden: it asks what fraction of the ACTUAL commanded magnitude moved in response
      to input, rather than what fraction of the weight budget was allocated to it.
    """
    w_features, w_intercept = weights[:-1], weights[-1]
    norm_f = float(np.linalg.norm(w_features))
    norm_i = float(np.linalg.norm(w_intercept))
    by_weight = norm_f / (norm_f + norm_i) if (norm_f + norm_i) > 0 else float("nan")

    pred = np.hstack([x_std, np.ones((x_std.shape[0], 1))]) @ weights
    driven = pred - w_intercept[None, :]
    total = float(np.linalg.norm(pred))
    by_pred = float(np.linalg.norm(driven)) / total if total > 0 else float("nan")
    return by_weight, by_pred


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_run", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained_step", type=int, default=None)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--config", default="mot_jepa_pilot")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--holdout_mod", type=int, default=10)
    parser.add_argument("--index-step", type=int, default=17)
    parser.add_argument("--batches", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--normalized-targets", action="store_true",
                        help="Fit on per-domain z-scored actions the way the trained heads do, then "
                             "denormalise before scoring. Isolates the target space as the cause of "
                             "the ridge-vs-head gap.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_module.CONFIGS[args.config]
    backbone, step = runtime.load_frozen_backbone(args.pretrained_run, args.pretrained_step, cfg, device)

    stores = sorted(glob.glob(args.clips))
    names = sorted({pathlib.Path(p).parent.name for p in stores})
    domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]

    def build(want: str) -> MotJepaClipDataset:
        dataset = MotJepaClipDataset(
            stores,
            cfg.layout,
            domain_ids=domain_ids,
            strides=(1,),
            index_step=args.index_step,
            with_conditioning=True,
            action_horizon=args.horizon,
        )
        dataset.clip_index = split_clip_index(dataset.clip_index, holdout_mod=args.holdout_mod, want=want)
        return dataset

    def gather(want: str) -> dict[int, dict[str, list]]:
        loader = torch.utils.data.DataLoader(
            build(want), batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=collate_clips
        )
        return collect(backbone, loader, device, args.batches)

    train = gather("train")
    held = gather("holdout")
    logger.info("collected %d train / %d holdout domains", len(train), len(held))

    # The trained heads regress onto per-domain z-scored actions (mot_jepa_policy_train.py:300) and
    # are scored after denormalising, while this ridge regresses onto raw ones. That is not a
    # cosmetic difference: z-scoring divides each of the 120 slots by its own spread, so the
    # training loss weights a slot that barely moves exactly as heavily as one that carries the
    # motion, whereas the eval metric -- raw-unit RMSE -- weights by actual magnitude. A head can
    # lower normalised MSE while raising the number we report. Fitting the ridge BOTH ways on
    # identical data isolates that, since nothing else differs between the two runs.
    action_stats: dict[int, tuple] = {}
    models: dict[int, tuple] = {}
    for domain, blob in sorted(train.items()):
        x, y = np.stack(blob["x"]), np.stack(blob["y"])
        if x.shape[0] < 40:
            logger.warning("%s has only %d train clips; skipping", names[domain], x.shape[0])
            continue
        if args.normalized_targets:
            # Per-slot over clips AND horizon steps, matching ActionNormalizer's (num_domains, 120).
            chunks = y.reshape(y.shape[0], args.horizon, -1)
            a_mean = chunks.mean(axis=(0, 1))
            a_scale = chunks.std(axis=(0, 1))
            a_scale = np.where(a_scale > 1e-6, a_scale, 1.0)
            action_stats[domain] = (a_mean, a_scale)
            y = ((chunks - a_mean) / a_scale).reshape(y.shape[0], -1)
        models[domain] = fit_one_domain(x, y)
        logger.info("%s: fitted on %d clips, lambda=%.0e", names[domain], x.shape[0], models[domain][3])

    # Score with the evaluator's metric: RMSE in raw action units over live slots, against the
    # holdout's own per-domain mean as the reference constant.
    print(f"\nridge policy on backbone step {step}, horizon {args.horizon}\n")
    print(f"{'domain':24s} {'clips':>7s} {'RMSE':>11s} {'constant':>11s} {'better?':>9s}")
    sq_r = sq_c = n_tot = 0.0
    shares: dict[int, tuple[float, float]] = {}
    for domain, blob in sorted(held.items()):
        if domain not in models:
            continue
        x, y, m = np.stack(blob["x"]), np.stack(blob["y"]), np.stack(blob["m"])
        mean, scale, weights, _ = models[domain]
        pred = np.hstack([(x - mean) / scale, np.ones((x.shape[0], 1))]) @ weights
        if args.normalized_targets:
            # Back to raw units, exactly as mot_jepa_policy_eval.py does, so the RMSE below stays
            # comparable to every other row in this table.
            a_mean, a_scale = action_stats[domain]
            pred = (pred.reshape(pred.shape[0], args.horizon, -1) * a_scale + a_mean).reshape(pred.shape[0], -1)
        err_r = float((((pred - y) * m) ** 2).sum())
        err_c = float((((y.mean(0) - y) * m) ** 2).sum())
        count = float(m.sum())
        sq_r += err_r
        sq_c += err_c
        n_tot += count
        rmse, const = np.sqrt(err_r / count), np.sqrt(err_c / count)
        shares[domain] = feature_share(weights, (x - mean) / scale)
        print(f"{names[domain]:24s} {x.shape[0]:7d} {rmse:11.5f} {const:11.5f} {'YES' if rmse < const else 'no':>9s}")

    if n_tot <= 0:
        # No domain ended up with BOTH a fitted model and holdout clips. Almost always too few
        # batches to reach every domain (the loader shuffles across 8 stores), which used to
        # surface as ZeroDivisionError three screens below the actual cause.
        logger.error(
            "no domain has both a fitted model and holdout clips: %d fitted, %d with holdout. "
            "Raise --batches (the index holds %s clips at the current --index-step) or lower "
            "--holdout_mod.",
            len(models),
            len(held),
            "unknown",
        )
        return 1

    pooled_r, pooled_c = np.sqrt(sq_r / n_tot), np.sqrt(sq_c / n_tot)
    print(f"{'POOLED':24s} {'':7s} {pooled_r:11.5f} {pooled_c:11.5f} {'YES' if pooled_r < pooled_c else 'no':>9s}")
    print(
        "\nCompare against the trained heads on the same holdout: linear 0.00462, drifting 0.00463,\n"
        "flowmatch 0.00484, best weight-decay 0.00454, constant 0.00436."
    )

    # The decomposition that interprets the table above. A ridge can only beat a constant by as
    # much as its features carry, so this bounds what any head on these features could do.
    if shares:
        print(f"\n{'domain':24s} {'obs-driven (weights)':>21s} {'obs-driven (pred)':>19s}")
        for domain, (by_weight, by_pred) in sorted(shares.items(), key=lambda kv: kv[1][0]):
            print(f"{names[domain]:24s} {by_weight:20.0%} {by_pred:18.0%}")
        mean_w = float(np.mean([v[0] for v in shares.values()]))
        print(f"{'MEAN':24s} {mean_w:20.0%}")
        print(
            "\nPre-registered baseline on the order-blind backbone (by weights): grasp_classify 5%,\n"
            "insert_HDMI 5%, insert_hole 22%, lift_can 28%, lift_bottle 33%, insert_tube 39%,\n"
            "pull_out_key 40%, put_bottle_in_shelf 70%. Thresholds: >= 50% the frozen features are\n"
            "load-bearing; 15-50% the features carry signal and the head is the ceiling; < 15%\n"
            "pretraining does not transfer to control on this corpus."
        )

    if args.out and models:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out,
            domains=np.array([names[d] for d in sorted(models)]),
            # Which encoder tensor these weights were fitted on. The policy feature moved from
            # sync_readout (layer-S snapshot) to final_readout (last block) and the WIDTH is
            # identical, so without this stamp a ridge fitted on the old tensor loads against the
            # new one with no error and quietly scores a different model.
            readout=np.array("final_readout"),
            **{f"mean_{d}": models[d][0] for d in models},
            **{f"scale_{d}": models[d][1] for d in models},
            **{f"weights_{d}": models[d][2] for d in models},
        )
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
