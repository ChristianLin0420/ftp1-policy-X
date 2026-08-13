#!/usr/bin/env python
"""Held-out action-chunk RMSE for a trained policy head. The drifting-vs-flowmatch comparison.

Training loss cannot answer which arm is better: drifting minimises an anisotropic geometric
energy and flow matching a velocity MSE, on different scales. This measures the one thing both
arms are ultimately for -- how close the generated action chunk is to what the human actually did
-- through each arm's own **deployment** path (`ActionDiT.sample`, so 1 NFE for drifting and an
Euler loop for flowmatch), on episodes the head never trained on.

RMSE is reported in **raw action units** (radians for joints), not normalised ones, so the number
means something physical and is comparable across arms and tasks whose normalisers differ.

Per FTP-1 action group as well as pooled, because a policy that nails the arm and drops the
gripper is a very different failure from the reverse, and a single scalar hides it.

Read-only.

    uv run python scripts/mot_jepa_policy_eval.py \
        --run .../mot_jepa_policy_drifting/ft_univtac_drifting \
        --pretrained_run .../backbones/probe2_s50000 --pretrained_step 50000 \
        --clips '.../univtac-clips/*/*.zarr' --holdout_mod 10
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import os
import pathlib

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from openpi.ftp1_action_groups import get_ftp1_action_group_slices
from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.action_dit import ActionDiT
from openpi.mot_jepa.action_dit import ActionNormalizer
from openpi.mot_jepa.action_dit import LinearHead
from openpi.mot_jepa.action_dit import RidgeHead
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import split_clip_index
from openpi.mot_jepa.model import ClipInputs

logger = logging.getLogger("mot_jepa.policy_eval")

GROUPS = get_ftp1_action_group_slices(48, 7, 15)


def load_head(run: pathlib.Path, step: int | None, cfg, num_domains: int, device: torch.device):
    """Head weights plus the normaliser fitted during that run.

    The normaliser is not optional. The head emits normalised actions, so without the exact
    per-domain statistics its run was trained against, every number below would be in the wrong
    units -- and plausibly so, which is worse than an error.
    """
    checkpoint_dir = run / "checkpoints"
    step = step or runtime.find_latest_step(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint under {checkpoint_dir}")
    path = checkpoint_dir / str(step)

    head_cls = LinearHead if cfg.head.objective == "linear" else ActionDiT
    head = head_cls(cfg.head, cfg.layout, num_domains=num_domains).to(device)
    weights = torch.load(path / "student.pt", map_location=device, weights_only=True)
    missing, unexpected = head.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"head load mismatched: {len(missing)} missing, {len(unexpected)} unexpected")
    head.eval()

    normalizer = ActionNormalizer(num_domains).to(device)
    loss_path = path / "loss.pt"
    if not loss_path.exists():
        raise FileNotFoundError(
            f"{loss_path} missing -- the run did not checkpoint its ActionNormalizer, so the "
            "predicted chunks cannot be returned to raw units. Re-run training with loss_fn passed "
            "to save_checkpoint."
        )
    normalizer.load_state_dict(torch.load(loss_path, map_location=device, weights_only=True))
    logger.info("loaded head + normalizer from %s step %d", run, step)
    return head, normalizer, step


@torch.no_grad()
def evaluate(backbone, head, normalizer, loader, device, names, *, num_steps: int, num_samples: int = 1,
             drift_ks: tuple[int, ...] = ()) -> dict:
    """Squared error accumulated per domain and per action group, over live slots only.

    ``drift_ks`` additionally measures INTEGRATED open-loop drift: the error in joint POSITION
    after executing k predicted deltas, rather than the error in each delta. That is the quantity
    the robot actually feels. MoT-JEPA emits per-step first differences and
    ``deploy.chunk_to_absolute_qpos8`` integrates them with a cumsum, so a per-step RMSE is a rate
    error whose integral over the chunk has never been computed.

    The gap between the two possibilities is the whole point. At lift_bottle's 0.00109 rad per
    step, 32 steps of INDEPENDENT error grow as sqrt(32) -> 0.0062 rad, about 3.7 mm at 0.6 m;
    PERFECTLY CORRELATED error grows as 32 -> 0.035 rad, about 21 mm. The first might grasp a
    bottle, the second misses it, and the per-step number is identical either way. Reporting the
    measured curve against both bounds reads off the correlation directly.
    """
    sq: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    count: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    drift_sq: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    drift_n: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    drift_clips: dict = collections.defaultdict(float)
    arm_lo, arm_hi = GROUPS["right-arm-joints"]

    for batch in loader:
        video = batch["video"].to(device).float().div_(127.5).sub_(1.0)
        gel = batch["gel"].to(device).float().div_(127.5).sub_(1.0)
        lowdim = batch["lowdim"].to(device).float()
        domain_id = batch["domain_id"].to(device)
        action_mask = batch["action_mask"].to(device).float()
        chunk_mask = batch["chunk_mask"].to(device).float()
        truth = batch["action_chunk"].to(device).float()

        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            encoded = backbone.encode_full(ClipInputs(video=video, gel=gel, lowdim=lowdim))
        encoded = type(encoded)(
            tokens=[t.float() for t in encoded.tokens],
            sync_readout=[t.float() for t in encoded.sync_readout],
            final_readout=[t.float() for t in encoded.final_readout],
        )

        # The deployment path, not the training path: 1 NFE under drifting, Euler under flowmatch.
        #
        # ``num_samples > 1`` averages independent samples to estimate the CONDITIONAL MEAN.
        # This matters because RMSE structurally penalises a generative model: a sampler that has
        # correctly learned p(a|o) pays Var[a|o] + bias^2, while a mean-predictor pays only
        # bias^2, so a perfectly calibrated sampler scores WORSE than a blurry averager. Comparing
        # the K-sample mean against the single-sample score separates "did not learn the
        # conditional structure" from "learned it and is paying honest sampling variance".
        predicted = torch.stack(
            [head.sample(encoded, action_mask, domain_id, num_steps=num_steps).float() for _ in range(num_samples)]
        ).mean(dim=0)
        # Back to raw units before scoring, so the RMSE is in radians and comparable across arms
        # whose normalisers differ.
        predicted = normalizer.denormalize(predicted, domain_id) * chunk_mask
        error = (predicted - truth * chunk_mask) ** 2

        for row, domain in enumerate(batch["domain_id"].tolist()):
            name = names[domain]
            for group, (lo, hi) in GROUPS.items():
                sq[name][group] += float(error[row][:, lo:hi].sum())
                count[name][group] += float(chunk_mask[row][:, lo:hi].sum())
            sq[name]["ALL"] += float(error[row].sum())
            count[name]["ALL"] += float(chunk_mask[row].sum())

        if drift_ks:
            # Only the arm integrates. The gripper sits in ABSOLUTE_COLUMNS and is executed as a
            # position, so cumsum-ing it would measure a quantity nothing computes.
            pred_arm = predicted[:, :, arm_lo:arm_hi]
            true_arm = (truth * chunk_mask)[:, :, arm_lo:arm_hi]
            live = chunk_mask[:, :, arm_lo:arm_hi]
            # A chunk that runs past the episode end has dead tail steps, and a cumsum through a
            # zero-filled step silently reports the drift as having stopped growing. Restrict to
            # clips whose arm slots are live at EVERY step, and report how many that drops.
            whole = live.reshape(live.shape[0], -1).min(dim=1).values > 0
            pred_cum = torch.cumsum(pred_arm, dim=1)
            true_cum = torch.cumsum(true_arm, dim=1)
            step_err = (pred_cum - true_cum) ** 2
            for row, domain in enumerate(batch["domain_id"].tolist()):
                if not bool(whole[row]):
                    continue
                name = names[domain]
                drift_clips[name] += 1.0
                for k in drift_ks:
                    if k > step_err.shape[1]:
                        continue
                    drift_sq[name][k] += float(step_err[row, k - 1].sum())
                    drift_n[name][k] += float(step_err.shape[2])

    return {
        "sq": {k: dict(v) for k, v in sq.items()},
        "count": {k: dict(v) for k, v in count.items()},
        "drift_sq": {k: dict(v) for k, v in drift_sq.items()},
        "drift_n": {k: dict(v) for k, v in drift_n.items()},
        "drift_clips": dict(drift_clips),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=pathlib.Path, default=None, help="Policy run directory.")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--ridge", type=pathlib.Path, default=None,
                        help="Score a fitted ridge (.npz from mot_jepa_ridge_policy.py) instead of a "
                             "trained run. Mutually exclusive with --run.")
    parser.add_argument("--pretrained_run", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained_step", type=int, default=None)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--config", default=None, help="Policy preset; inferred from the run dir if unset.")
    parser.add_argument("--holdout_mod", type=int, default=10)
    parser.add_argument("--index-step", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--euler-steps", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Average K samples per observation to estimate the conditional mean.")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--drift-ks", default="1,2,4,8,16,32",
                        help="Truncation lengths for the integrated open-loop drift table. Empty "
                             "string disables it.")
    args = parser.parse_args()
    drift_ks = tuple(int(k) for k in args.drift_ks.split(",") if k.strip())

    if bool(args.run) == bool(args.ridge):
        parser.error("pass exactly one of --run (a trained head) or --ridge (a fitted .npz)")

    if args.config:
        preset = args.config
    elif args.ridge:
        # Only the layout and the horizon are read for a ridge; every preset shares them.
        preset = "mot_jepa_policy_linear"
    elif args.run.parent.name in config_module.POLICY_CONFIGS:
        # submit.sh builds RUN_DIR as <run_root>/<CONFIG_NAME>/<EXP_NAME>, so the parent directory
        # IS the preset name. Exact, unlike the substring test this replaces -- which mapped
        # `mot_jepa_policy_linear_perdomain` onto `mot_jepa_policy_linear`, a preset with a
        # different trunk, and would have built a head whose weights could not load.
        preset = args.run.parent.name
    else:
        raise SystemExit(
            f"cannot infer the preset from {args.run}: its parent directory "
            f"{args.run.parent.name!r} is not a known policy config. Pass --config explicitly. "
            f"Known: {sorted(config_module.POLICY_CONFIGS)}"
        )
    cfg = config_module.POLICY_CONFIGS[preset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stores = sorted(glob.glob(args.clips))
    names = sorted({pathlib.Path(p).parent.name for p in stores})
    domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]

    backbone, backbone_step = runtime.load_frozen_backbone(args.pretrained_run, args.pretrained_step, cfg, device)
    if args.ridge:
        head = RidgeHead(args.ridge, cfg.layout, horizon=cfg.head.horizon).to(device)
        # The ridge regresses onto raw action_chunk, so there is nothing to invert. A
        # default-constructed ActionNormalizer is mean 0 / scale 1, making denormalize an exact
        # identity -- the alternative, special-casing the scoring loop, would leave two code paths
        # where the whole point of this flag is to have one.
        normalizer = ActionNormalizer(len(names)).to(device)
        step = -1
        missing = sorted(set(range(len(names))) - set(head.fitted_domains))
        if missing:
            logger.warning("no ridge fitted for %s; those domains predict zero", [names[d] for d in missing])
    else:
        head, normalizer, step = load_head(args.run, args.step, cfg, len(names), device)

    dataset = MotJepaClipDataset(
        stores,
        cfg.layout,
        domain_ids=domain_ids,
        strides=(1,),
        index_step=args.index_step,
        with_conditioning=True,
        action_horizon=cfg.head.horizon,
    )
    before = len(dataset)
    dataset.clip_index = split_clip_index(dataset.clip_index, holdout_mod=args.holdout_mod, want="holdout")
    logger.info("holdout: %d of %d clips (mod %d)", len(dataset), before, args.holdout_mod)
    if len(dataset) == 0:
        raise RuntimeError("holdout split is empty; check --holdout_mod against how the run was trained")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=collate_clips
    )
    acc = evaluate(backbone, head, normalizer, loader, device, names, num_steps=args.euler_steps,
                   num_samples=args.num_samples, drift_ks=drift_ks)

    def rmse(name: str, group: str) -> float:
        c = acc["count"][name].get(group, 0.0)
        return float("nan") if c <= 0 else float(np.sqrt(acc["sq"][name][group] / c))

    live = [g for g in (*GROUPS, "ALL") if any(acc["count"][n].get(g, 0.0) > 0 for n in acc["count"])]
    objective = "ridge" if args.ridge else cfg.head.objective
    print(f"\n{preset}  head step {step}  backbone step {backbone_step}  objective {objective}")
    print(f"held-out clips: {len(dataset)}   samples/obs: {args.num_samples}   (every {args.holdout_mod}th episode)\n")
    header = f"{'domain':24s}" + "".join(f"{g[:14]:>16s}" for g in live)
    print(header)
    for name in sorted(acc["count"]):
        print(f"{name:24s}" + "".join(f"{rmse(name, g):16.5f}" for g in live))

    pooled = {}
    for g in live:
        s = sum(acc["sq"][n].get(g, 0.0) for n in acc["sq"])
        c = sum(acc["count"][n].get(g, 0.0) for n in acc["count"])
        pooled[g] = float("nan") if c <= 0 else float(np.sqrt(s / c))
    print(f"{'POOLED':24s}" + "".join(f"{pooled[g]:16.5f}" for g in live))
    print("\nRMSE in raw action units (radians for joints); live slots only.")

    def drift(name: str, k: int) -> float:
        n = acc["drift_n"].get(name, {}).get(k, 0.0)
        return float("nan") if n <= 0 else float(np.sqrt(acc["drift_sq"][name][k] / n))

    drift_table: dict = {}
    if drift_ks and acc["drift_clips"]:
        print("\n=== integrated open-loop drift: arm POSITION error after k executed deltas ===")
        print("(what the robot feels. per-step RMSE is a rate; this is its integral.)\n")
        header = f"{'domain':24s}{'clips':>7s}" + "".join(f"{'k=' + str(k):>11s}" for k in drift_ks)
        print(header)
        for name in sorted(acc["drift_clips"]):
            row = "".join(f"{drift(name, k):11.5f}" for k in drift_ks)
            print(f"{name:24s}{int(acc['drift_clips'][name]):7d}" + row)
            drift_table[name] = {k: drift(name, k) for k in drift_ks}

        pooled_drift = {}
        for k in drift_ks:
            s_k = sum(acc["drift_sq"][n].get(k, 0.0) for n in acc["drift_sq"])
            n_k = sum(acc["drift_n"][n].get(k, 0.0) for n in acc["drift_n"])
            pooled_drift[k] = float("nan") if n_k <= 0 else float(np.sqrt(s_k / n_k))
        print(f"{'POOLED':24s}{int(sum(acc['drift_clips'].values())):7d}"
              + "".join(f"{pooled_drift[k]:11.5f}" for k in drift_ks))

        # The two bounds the measured curve must be read against. Independent per-step errors grow
        # as sqrt(k); perfectly correlated ones grow as k. Where the measurement falls between them
        # IS the error correlation, which is the thing no per-step metric can show.
        per_step = pooled.get("right-arm-joints", float("nan"))
        print(f"\n{'independent sqrt(k)':24s}{'':7s}"
              + "".join(f"{per_step * np.sqrt(k):11.5f}" for k in drift_ks))
        print(f"{'correlated k':24s}{'':7s}" + "".join(f"{per_step * k:11.5f}" for k in drift_ks))
        worst = drift_ks[-1]
        if np.isfinite(pooled_drift[worst]) and per_step > 0:
            lo_b, hi_b = per_step * np.sqrt(worst), per_step * worst
            frac = (pooled_drift[worst] - lo_b) / max(hi_b - lo_b, 1e-12)
            print(f"\nat k={worst}: measured {pooled_drift[worst]:.5f} rad sits {frac:.0%} of the way "
                  f"from independent ({lo_b:.5f}) to fully correlated ({hi_b:.5f}).")
            print(f"as a reach error at 0.6 m that is about {pooled_drift[worst] * 600:.1f} mm.")
        dropped = len(dataset) - int(sum(acc["drift_clips"].values()))
        if dropped:
            print(f"\n{dropped} of {len(dataset)} clips excluded: arm slots not live at every step.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "preset": preset,
                    "objective": objective,
                    "head_step": step,
                    "backbone_step": backbone_step,
                    "holdout_mod": args.holdout_mod,
                    "num_clips": len(dataset),
                    "pooled": pooled,
                    "per_domain": {n: {g: rmse(n, g) for g in live} for n in sorted(acc["count"])},
                    "drift_ks": list(drift_ks),
                    "integrated_drift": drift_table,
                },
                indent=2,
            )
        )
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
