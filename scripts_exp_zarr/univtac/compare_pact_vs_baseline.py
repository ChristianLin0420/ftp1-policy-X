#!/usr/bin/env python
"""Overlay a PACT run against the FTP-1 baseline and publish the comparison to W&B.

Both arms must come from the same dataset, split, and normalization statistics for the
comparison to mean anything; the only intended difference is the architecture.

Usage:
    uv run python scripts_exp_zarr/univtac/compare_pact_vs_baseline.py \
        --baseline 4tdhuz7r --pact 2b3520wj
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# The evaluator writes -1 for an action group that is inactive for the embodiment. UniVTAC
# lift_bottle is a right-arm joint-control task, so left-arm and Cartesian pose fields are
# inactive and must be dropped rather than averaged in.
INACTIVE_SENTINEL = -1.0

# Groups that are actually driven by this embodiment, most specific first.
ACTIVE_GROUPS = ("total", "right-arm-joints", "right-hand-joint")

TRAIN_METRICS = ("Trainning/loss",)
GRAD_METRICS = (
    "GradNorm/tactile_total",
    "GradNorm/vision_tower",
    "GradNorm/vlm_language",
    "GradNorm/action_expert",
)
# Ratios are logged only when their denominator branch has a gradient, so a missing series
# means that branch is frozen -- or that its parameter prefix matched nothing.
GRAD_RATIOS = (
    "GradNorm/tactile_over_vision",
    "GradNorm/tactile_over_vlm",
    "GradNorm/tactile_over_action",
)


def fetch_history(run, keys: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Return {key: [(step, value), ...]} with inactive-sentinel samples removed."""
    series: dict[str, list[tuple[int, float]]] = {key: [] for key in keys}
    for row in run.scan_history(keys=["_step", *keys]):
        step = row.get("_step")
        if step is None:
            continue
        for key in keys:
            value = row.get(key)
            if value is None or value == INACTIVE_SENTINEL:
                continue
            series[key].append((int(step), float(value)))
    return {key: sorted(points) for key, points in series.items() if points}


def plot_overlay(title: str, ylabel: str, arms: dict[str, list[tuple[int, float]]], logy: bool):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    for label, points in arms.items():
        if not points:
            continue
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        ax.plot(steps, values, label=label, linewidth=1.8, marker="o", markersize=3)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def final_value(points: list[tuple[int, float]]) -> float | None:
    return points[-1][1] if points else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="W&B run id of the FTP-1 baseline arm")
    parser.add_argument("--pact", required=True, help="W&B run id of the PACT arm")
    parser.add_argument("--project", default="crlc112358/openpi")
    parser.add_argument("--name", default="pact_vs_baseline")
    args = parser.parse_args()

    api = wandb.Api()
    baseline = api.run(f"{args.project}/{args.baseline}")
    pact = api.run(f"{args.project}/{args.pact}")

    val_keys = [f"Validation/{stat}_{group}" for stat in ("rmse", "mape") for group in ACTIVE_GROUPS]
    val_keys += [f"Validation/jitter_rms_{group}" for group in ACTIVE_GROUPS]

    baseline_hist = fetch_history(baseline, [*val_keys, *TRAIN_METRICS])
    pact_hist = fetch_history(pact, [*val_keys, *TRAIN_METRICS, *GRAD_METRICS, *GRAD_RATIOS])

    run = wandb.init(
        project=args.project.split("/")[-1],
        entity=args.project.split("/")[0] if "/" in args.project else None,
        name=args.name,
        job_type="comparison",
        config={
            "baseline_run": baseline.id,
            "baseline_name": baseline.name,
            "pact_run": pact.id,
            "pact_name": pact.name,
            "pact_tokens_per_area": pact.config.get("model", {}).get("tactile_tokenizer_config", {}).get("tokens_per_area"),
            "pact_tactile_reads_vision": pact.config.get("model", {}).get("tactile_reads_vision"),
        },
    )

    panels = {}
    for key in [*val_keys, *TRAIN_METRICS]:
        arms = {}
        if key in baseline_hist:
            arms["FTP-1 baseline"] = baseline_hist[key]
        if key in pact_hist:
            arms["PACT"] = pact_hist[key]
        if len(arms) < 1:
            continue
        pretty = key.replace("Validation/", "").replace("Trainning/", "train ")
        fig = plot_overlay(pretty, pretty, arms, logy=key in TRAIN_METRICS)
        panels[f"compare/{pretty}"] = wandb.Image(fig)
        plt.close(fig)

    # The modality-dominance monitor only exists on the PACT arm.
    grad_arms = {key.split("/")[-1]: pact_hist[key] for key in GRAD_METRICS if key in pact_hist}
    if grad_arms:
        fig = plot_overlay(
            "PACT gradient norms by branch", "pre-clip grad norm", grad_arms, logy=True
        )
        panels["compare/pact_grad_norms"] = wandb.Image(fig)
        plt.close(fig)

    ratio_arms = {key.split("/")[-1]: pact_hist[key] for key in GRAD_RATIOS if key in pact_hist}
    if ratio_arms:
        fig = plot_overlay(
            "PACT tactile gradient share", "tactile / branch", ratio_arms, logy=True
        )
        panels["compare/pact_tactile_share"] = wandb.Image(fig)
        plt.close(fig)

    table = wandb.Table(columns=["metric", "baseline", "pact", "delta", "pct_change"])
    for key in val_keys:
        b = final_value(baseline_hist.get(key, []))
        p = final_value(pact_hist.get(key, []))
        if b is None or p is None:
            continue
        delta = p - b
        pct = (delta / b * 100.0) if b else float("nan")
        table.add_data(key.replace("Validation/", ""), b, p, delta, pct)

    run.log({**panels, "compare/final_metrics": table})
    run.summary["baseline_url"] = baseline.url
    run.summary["pact_url"] = pact.url
    print(f"comparison published: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
