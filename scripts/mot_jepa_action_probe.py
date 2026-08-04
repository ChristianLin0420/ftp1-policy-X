"""Does the frozen encoder's latent even respond to the action? A gate, not a training run.

Stage 4 exists to learn an action-conditioned world model, and its falsifier -- the action
donor ratio -- sat at 1.00 across every setting tried: loss weights a decade apart, prediction
horizons of one and four tubelets, and (after a real encoding defect was fixed) with donor
actions verified as near-orthogonal counterfactuals. The action buys about 1%.

Three explanations were eliminated by experiment. This script tests the fourth and cheapest to
check, which is also the only one no architecture can fix: **maybe the latent simply does not
move with the action at this timescale.** If a linear map from action to latent delta explains
nothing, then no predictor -- block-causal, autoregressive, or otherwise -- can do better,
because the information is not there to condition on.

Method: freeze the backbone, encode clips, pool tokens per tubelet step, and least-squares fit

    dz_t  <-  [action_t | state_t]

reporting R^2 on **held-out clips**, overall and per domain. Per domain matters because the
domains are not comparable: VisuoTactile_D-WHEEL carries real joint motion across 16 live slots
while RH20TCfg5Franka carries 10, and D-WHEEL was the one domain whose actions had healthy
variance when every other was dominated by the rot6d constant.

Read it as: R^2 near zero means the corpus, not the design, is the limit.

Usage::

    uv run python scripts/mot_jepa_action_probe.py \
        --pretrained_run /lustre/.../ftp1-runs/backbones/pilot02_s18500 \
        --clips '/lustre/.../ftp1-clips/*/*.zarr'
"""

from __future__ import annotations

import argparse
import collections
import glob
import logging
import os
import pathlib
import sys

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("USE_SWANLAB", "false")

from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.model import MotJepaStudent

logger = logging.getLogger("mot_jepa.action_probe")


def load_backbone(run: pathlib.Path, step: int | None, cfg, device: torch.device):
    """Positional shadow copy -- ``EmaTeacher`` keys by index, not by parameter name."""
    checkpoint_dir = run / "checkpoints"
    step = step or runtime.find_latest_step(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint under {checkpoint_dir}")
    shadow = torch.load(checkpoint_dir / str(step) / "teacher_ema.pt", map_location="cpu", weights_only=True)
    student = MotJepaStudent(cfg.layout, cfg.encoder, cfg.predictor, lowdim_channels=cfg.data.lowdim_channels)
    params = list(student.backbone.parameters())
    if len(shadow) != len(params):
        raise RuntimeError(f"{len(shadow)} shadow tensors for {len(params)} parameters; config mismatch")
    with torch.no_grad():
        for index, param in enumerate(params):
            param.copy_(shadow[f"shadow.{index}"].to(param.dtype))
    logger.info("loaded frozen backbone from %s step %d", checkpoint_dir, step)
    return student.to(device).eval().backbone, step


def r_squared(features: np.ndarray, targets: np.ndarray, *, holdout: float = 0.3) -> tuple[float, int]:
    """Held-out R^2 of a ridge fit ``targets ~ features``.

    Held out rather than in-sample: with 240 action dims and a few hundred clips, an in-sample
    fit would report a high R^2 from sheer capacity and say nothing. Ridge rather than plain
    least squares because the action columns are strongly correlated -- a gripper and its arm
    joints move together -- and a singular normal matrix would otherwise decide the answer.
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
def collect(backbone, loader, layout, device, max_batches: int) -> dict[int, tuple[list, list]]:
    """Per domain: the action/state features and the latent deltas they should explain."""
    out: dict[int, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
    for seen, batch in enumerate(loader):
        if seen >= max_batches:
            break
        video = batch["video"].to(device).float().div_(127.5).sub_(1.0)
        gel = batch["gel"].to(device).float().div_(127.5).sub_(1.0)
        lowdim = batch["lowdim"].to(device).float()

        from openpi.mot_jepa.model import ClipInputs

        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            encoded = backbone.encode_full(ClipInputs(video=video, gel=gel, lowdim=lowdim))

        # Pool the TACTILE expert per tubelet step -- that is the stream a world model has to
        # roll forward, and the one Stage 4's loss is dominated by.
        tactile = encoded.tokens[1].float()
        steps = layout.num_steps
        per_step = tactile.reshape(tactile.shape[0], steps, -1, tactile.shape[-1]).mean(dim=2)
        delta = (per_step[:, 1:] - per_step[:, :-1]).cpu().numpy()

        action = batch["action"][:, :-1].numpy()
        state = batch["state"][:, :: layout.tubelet_t][:, :-1].numpy()
        features = np.concatenate([action, state], axis=-1)

        for row, domain in enumerate(batch["domain_id"].tolist()):
            out[domain][0].append(features[row])
            out[domain][1].append(delta[row])
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_run", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained_step", type=int, default=None)
    parser.add_argument("--clips", required=True, help="glob for the derived *.zarr stores")
    parser.add_argument("--config", default="mot_jepa_pilot")
    parser.add_argument("--batches-per-domain", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_module.CONFIGS[args.config]
    backbone, step = load_backbone(args.pretrained_run, args.pretrained_step, cfg, device)

    stores = sorted(glob.glob(args.clips))
    names = sorted({pathlib.Path(p).parent.name for p in stores})
    domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]
    dataset = MotJepaClipDataset(
        stores, cfg.layout, domain_ids=domain_ids, strides=(1,), index_step=97, with_conditioning=True
    )
    logger.info("%d stores / %d domains / %d clips", len(stores), len(names), len(dataset))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_clips
    )
    collected = collect(backbone, loader, cfg.layout, device, args.batches_per_domain * len(names))

    print(f"\n{'domain':24s} {'clips':>7s} {'R^2 (held out)':>15s}")
    rows, all_x, all_y = [], [], []
    for domain in sorted(collected):
        features = np.concatenate(collected[domain][0], axis=0)
        targets = np.concatenate(collected[domain][1], axis=0)
        score, count = r_squared(features, targets)
        rows.append((names[domain], score))
        all_x.append(features)
        all_y.append(targets)
        print(f"{names[domain]:24s} {count:7d} {score:15.4f}")

    pooled, pooled_n = r_squared(np.concatenate(all_x), np.concatenate(all_y))
    print(f"\n{'POOLED':24s} {pooled_n:7d} {pooled:15.4f}")

    best = max((s for _, s in rows if np.isfinite(s)), default=float("nan"))
    print(f"\nbackbone step {step}; best single domain R^2 = {best:.4f}")
    print(
        "\nverdict: "
        + (
            "the latent does respond to the action -- an action-conditioned predictor has "
            "something to learn from."
            if best > 0.05
            else "the latent barely moves with the action at this timescale. No predictor "
            "architecture can condition on information that is not there; Stage 4 needs a "
            "longer horizon or a different corpus, not a better model."
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
