#!/usr/bin/env python
"""Train a DiT action head on a frozen MoT-JEPA encoder.

Two arms behind one script, differing in exactly one config field:

``mot_jepa_policy_drifting``   Implicit Drifting Policy -- one-step generation, 1 NFE.
``mot_jepa_policy_flowmatch``  Conditional flow matching -- the control, 10 Euler steps.

The control exists because this pipeline changes two things at once relative to the FTP-1 policy:
the conditioning encoder (frozen MoT-JEPA rather than PaliGemma) *and* the generative objective.
An arm that changes only the encoder is what makes a bad result attributable.

The encoder is frozen and runs under ``no_grad``; only the head trains. Requeue, checkpointing and
sample-order continuity are inherited from the pretraining trainer.

    torchrun ... scripts/mot_jepa_policy_train.py mot_jepa_policy_drifting \
        --pretrained_run .../backbones/probe2_s50000 --pretrained_step 50000 \
        --data.store_glob '.../ftp1-clips/*/*.zarr'
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from torch.nn.parallel import DistributedDataParallel

from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.action_dit import ActionDiT
from openpi.mot_jepa.action_dit import flow_matching_loss
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import load_domain_config
from openpi.mot_jepa.drifting import drifting_loss
from openpi.shared.wandb_compat import wandb
from scripts.mot_jepa_train import InfiniteBatchSampler
from scripts.mot_jepa_train import lr_at
from scripts.mot_jepa_train import reduce_metrics
from scripts.mot_jepa_train import to_inputs

logger = logging.getLogger("mot_jepa.policy")


def build_dataset(cfg: config_module.PolicyConfig) -> MotJepaClipDataset:
    """Same discovery rules as pretraining, plus the future-action horizon.

    The horizon shrinks the index -- every clip must now have ``horizon`` frames of future inside
    its own episode -- so short episodes drop out here rather than producing padded targets.
    """
    if cfg.data.domain_config:
        pairs = load_domain_config(cfg.data.domain_config)
        names = sorted({name for name, _ in pairs})
        stores = [path for _, path in pairs]
        domain_ids = [names.index(name) for name, _ in pairs]
    elif cfg.data.store_glob:
        stores = sorted(glob.glob(cfg.data.store_glob))
        names = sorted({pathlib.Path(p).parent.name for p in stores})
        domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]
        logger.info("discovered %d domains from %d stores", len(names), len(stores))
    else:
        raise ValueError("set either data.domain_config or data.store_glob")
    if not stores:
        raise ValueError("no *.zarr stores matched the data configuration")

    return MotJepaClipDataset(
        stores,
        cfg.layout,
        domain_ids=domain_ids,
        strides=cfg.data.strides,
        index_step=cfg.data.index_step,
        lowdim_channels=cfg.data.lowdim_channels,
        with_conditioning=True,
        action_horizon=cfg.head.horizon,
    )


def init_tracking(cfg: config_module.PolicyConfig, run_dir: pathlib.Path, *, resuming: bool) -> None:
    """Fixed W&B run id so a requeue chain appends to one run instead of forking."""
    if not cfg.wandb_enabled or not runtime.is_main_process():
        return
    id_path = run_dir / "wandb_id.txt"
    if resuming and id_path.exists():
        wandb.init(id=id_path.read_text().strip(), resume="must", project=cfg.project_name)
    else:
        wandb.init(name=cfg.exp_name, project=cfg.project_name, config=json.loads(cfg.to_json()))
        id_path.write_text(str(wandb.run.id))


def train(cfg: config_module.PolicyConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    use_ddp, local_rank, device = runtime.setup_ddp()
    rank, world_size = runtime.get_rank(), runtime.get_world_size()

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed + rank)

    if not cfg.pretrained_run:
        raise ValueError("--pretrained_run is required; the head is meaningless without an encoder")
    backbone, backbone_step = runtime.load_frozen_backbone(
        pathlib.Path(cfg.pretrained_run), cfg.pretrained_step, cfg, device
    )
    head = ActionDiT(cfg.head, cfg.layout).to(device)
    logger.info(
        "frozen backbone step %d; head %s, %.1fM trainable params",
        backbone_step,
        cfg.head.objective,
        sum(p.numel() for p in head.parameters()) / 1e6,
    )

    dataset = build_dataset(cfg)
    logger.info("%d clips at horizon %d", len(dataset), cfg.head.horizon)
    # Domain-pure batches, which pretraining does not need but IDP does. Its neighbour geometry
    # and its reference variance are both computed WITHIN a batch, so a batch mixing embodiments
    # would compare a 16-slot arm against a 40-slot bimanual rig and read the difference in
    # *layout* as a difference in *action*. Same domain => same action_mask => the statistics
    # are over comparable quantities.
    domain_of_sample = np.asarray(dataset.domain_ids, dtype=np.int64)[dataset.clip_index.entries[:, 0]]
    sampler = InfiniteBatchSampler(
        len(dataset),
        cfg.local_batch_size,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
        domain_of_sample=domain_of_sample,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_clips,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.lr_peak, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
    )

    resume_step = runtime.find_latest_step(cfg.checkpoint_dir)
    global_step = 0
    if resume_step is not None:
        global_step = runtime.load_checkpoint(
            cfg.checkpoint_dir, resume_step, student=head, teacher=None, optimizer=optimizer, device=device
        )
        logger.info("resumed from step %d", global_step)
    sampler.set_start_step(global_step)
    init_tracking(cfg, run_dir, resuming=resume_step is not None)

    model = head
    if use_ddp:
        model = DistributedDataParallel(
            head, device_ids=[local_rank], find_unused_parameters=cfg.find_unused_parameters
        )

    monitor = runtime.PreemptionMonitor(run_dir / "PREEMPT_REQUEST", device)
    records: list[dict[str, float]] = []
    window_clips, window_start = 0, time.time()
    iterator = iter(loader)
    preempted = False

    while global_step < cfg.num_train_steps:
        if monitor.should_stop(global_step):
            logger.info("preemption requested at step %d; checkpointing", global_step)
            preempted = True
            break

        batch = next(iterator)
        global_step += 1

        lr = lr_at(
            global_step,
            peak=cfg.lr_peak,
            end=cfg.lr_end,
            warmup=cfg.lr_warmup_steps,
            total=cfg.num_train_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs = to_inputs(batch, device)
        actions = batch["action_chunk"].to(device, non_blocking=True).float()
        action_mask = batch["action_mask"].to(device, non_blocking=True).float()
        chunk_mask = batch["chunk_mask"].to(device, non_blocking=True).float()

        # The encoder never trains, so no activations are kept for it. That is what makes the
        # step affordable: only the 61.9M-parameter head holds a graph.
        with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            encoded = backbone.encode_full(inputs)
        encoded = type(encoded)(
            tokens=[t.float() for t in encoded.tokens],
            sync_readout=[t.float() for t in encoded.sync_readout],
        )

        if cfg.head.objective == "drifting":
            loss, extras = drifting_loss(
                model, encoded, actions, action_mask, chunk_mask, config=cfg.drifting
            )
        else:
            loss, extras = flow_matching_loss(model, encoded, actions, action_mask, chunk_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_grad_norm)
        optimizer.step()

        records.append({"loss": float(loss.detach()), "learning_rate": lr, "grad_norm": float(grad_norm), **extras})
        window_clips += cfg.local_batch_size * world_size

        if global_step % cfg.log_interval == 0:
            metrics = reduce_metrics(records, device)
            records.clear()
            elapsed = max(time.time() - window_start, 1e-6)
            metrics["clips_per_second"] = window_clips / elapsed
            window_clips, window_start = 0, time.time()
            if runtime.is_main_process():
                logger.info(
                    "step %d loss %.4f grad %.3f lr %.2e %.1f clips/s %s",
                    global_step,
                    metrics["loss"],
                    metrics["grad_norm"],
                    lr,
                    metrics["clips_per_second"],
                    {k: round(v, 4) for k, v in metrics.items() if k not in ("loss", "grad_norm", "learning_rate")},
                )
                if cfg.wandb_enabled:
                    wandb.log({f"policy/{k}": v for k, v in metrics.items()}, step=global_step)

        if global_step % cfg.save_interval == 0 and runtime.is_main_process():
            runtime.save_checkpoint(
                cfg.checkpoint_dir,
                global_step,
                student=head,
                teacher=None,
                optimizer=optimizer,
                config_json=cfg.to_json(),
                keep_last=cfg.keep_last,
                keep_period=cfg.keep_period,
            )
        if preempted.is_set():
            break

    if runtime.is_main_process():
        runtime.save_checkpoint(
            cfg.checkpoint_dir,
            global_step,
            student=head,
            teacher=None,
            optimizer=optimizer,
            config_json=cfg.to_json(),
            keep_last=cfg.keep_last,
            keep_period=cfg.keep_period,
        )
        if not preempted:
            # Stops the requeue chain; the batch script checks for this before resubmitting.
            (run_dir / "DONE").write_text(str(global_step))
            logger.info("training complete at step %d; not requeueing", global_step)

    runtime.barrier()
    if runtime.is_main_process() and cfg.wandb_enabled:
        wandb.finish()
    runtime.cleanup_ddp()
    # Exit 0 either way: the batch script decides whether to requeue, not the exit code.
    sys.exit(0)


def main() -> None:
    train(config_module.policy_cli())


if __name__ == "__main__":
    main()
