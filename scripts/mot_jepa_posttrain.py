"""MoT-JEPA post-training: Stage 3 (instruction) and Stage 4 (action).

Both stages are the same loop -- frozen backbone forward, one small trainable head, one loss,
one falsifier probe -- so they share a script and differ by a branch rather than by a fork.

The backbone is **frozen** in both. That is not only a cost decision: it means a binding
result measured after pretraining is still true after post-training, so re-running P1 and P3
afterwards is a *check* (they must be identical) rather than a re-measurement. It also removes
the collapse mode entirely, which is why there is no EMA teacher here -- targets come from the
same frozen weights under ``no_grad``.

Everything about surviving a 4-hour walltime is inherited unchanged from ``runtime.py``: the
frozen config, the fixed W&B run id, the preemption flag and the requeue chain.

Usage::

    torchrun ... scripts/mot_jepa_posttrain.py mot_jepa_stage3 \
        --pretrained_run .cache/mot_jepa/runs/mot_jepa_pilot/pilot02 \
        --data.store_glob '/lustre/.../ftp1-clips/*/*.zarr'
"""

from __future__ import annotations

import collections
import glob
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import zarr

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("USE_SWANLAB", "false")

from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import load_domain_config
from openpi.mot_jepa.condition import ActionEmbed
from openpi.mot_jepa.condition import InstructionHead
from openpi.mot_jepa.condition import info_nce
from openpi.mot_jepa.condition import pool_experts
from openpi.mot_jepa.losses import jepa_regression_loss
from openpi.mot_jepa.losses import normalize_targets
from openpi.mot_jepa.masking import build_rollout_masks
from openpi.mot_jepa.model import MotJepaStudent
from openpi.mot_jepa.mot_encoder import split_index_by_expert
from openpi.mot_jepa.predictor import MoTPredictor
from openpi.mot_jepa.predictor import target_index_by_expert
from openpi.shared.wandb_compat import wandb
from scripts.mot_jepa_train import InfiniteBatchSampler
from scripts.mot_jepa_train import init_tracking
from scripts.mot_jepa_train import lr_at
from scripts.mot_jepa_train import reduce_metrics
from scripts.mot_jepa_train import to_inputs

logger = logging.getLogger("mot_jepa.posttrain")

STAGE_INSTRUCTION = "instruction"
STAGE_ACTION = "action"


# ======================================================================================
# Data
# ======================================================================================


def build_dataset(
    cfg: config_module.MotJepaPosttrainConfig,
) -> tuple[MotJepaClipDataset, np.ndarray, np.ndarray]:
    """Dataset plus per-store domain and task ids.

    ``task_of_store`` is the Stage 3 label, and it is neither the store nor the instruction
    string. Two constraints have to hold at once:

    * Within a store, episodes carry different *paraphrases* of one task. Labelling by string
      would push those apart and teach the model that rewording changes the goal.
    * Across stores, the same instruction can repeat verbatim -- sharpa has **164 stores
      sharing 10 instruction strings**. Labelling by store would push apart two clips whose
      instructions are literally identical, which is the same mistake one level up.

    So the label is the store's *set* of instruction ids: stores with identical sets collapse
    to one task. sharpa becomes 10 tasks, FreeTacMan stays 44.
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
    else:
        raise ValueError("set either data.domain_config or data.store_glob")
    if not stores:
        raise ValueError("no *.zarr stores matched the data configuration")

    task_ids = task_ids_for_stores(stores)

    if cfg.stage == STAGE_INSTRUCTION and cfg.stage3.min_stores_per_domain > 1:
        # A domain-pure batch must be able to see several distinct TASKS or the retrieval is
        # trivial. Counting tasks rather than stores is what makes this honest on sharpa,
        # where 164 stores carry 10 tasks. Undersized domains are dropped rather than mixed
        # with another domain, which would reintroduce the appearance shortcut.
        tasks_per_domain = collections.defaultdict(set)
        for domain, task in zip(domain_ids, task_ids, strict=True):
            tasks_per_domain[domain].add(task)
        keep = [i for i, d in enumerate(domain_ids) if len(tasks_per_domain[d]) >= cfg.stage3.min_stores_per_domain]
        dropped = sorted({names[domain_ids[i]] for i in set(range(len(stores))) - set(keep)})
        if dropped:
            logger.info(
                "Stage 3 drops %d domain(s) with <%d distinct tasks: %s",
                len(dropped),
                cfg.stage3.min_stores_per_domain,
                dropped,
            )
        stores = [stores[i] for i in keep]
        domain_ids = [domain_ids[i] for i in keep]
        task_ids = [task_ids[i] for i in keep]
    logger.info("using %d stores / %d tasks across %d domains", len(stores), len(set(task_ids)), len(set(domain_ids)))

    dataset = MotJepaClipDataset(
        stores,
        cfg.layout,
        domain_ids=domain_ids,
        strides=cfg.data.strides,
        index_step=cfg.data.index_step,
        lowdim_channels=cfg.data.lowdim_channels,
        with_conditioning=True,
    )
    entries = dataset.clip_index.entries
    domain_of_sample = np.asarray(domain_ids, dtype=np.int64)[entries[:, 0]]
    return dataset, domain_of_sample, np.asarray(task_ids, dtype=np.int64)


def task_ids_for_stores(stores: list[str]) -> list[int]:
    """Map each store to a task id: stores with the same instruction-id set share one.

    Falls back to a per-store id for any store missing ``meta/instruction_id``, which keeps a
    half-migrated corpus loadable instead of collapsing every unlabelled store into one task.
    """
    by_signature: dict[tuple, int] = {}
    out: list[int] = []
    for index, path in enumerate(stores):
        try:
            ids = np.asarray(zarr.open(path, mode="r")["meta/instruction_id"][:])
            signature: tuple = tuple(sorted({int(i) for i in ids}))
        except Exception:
            logger.warning("no instruction_id in %s; treating it as its own task", path)
            signature = ("unlabelled", index)
        out.append(by_signature.setdefault(signature, len(by_signature)))
    return out


def load_instruction_table(path: str, device: torch.device) -> torch.Tensor:
    payload = np.load(path, allow_pickle=True)
    table = torch.from_numpy(np.asarray(payload["embeddings"], dtype=np.float32)).to(device)
    logger.info("instruction table %s from %s", tuple(table.shape), payload["model"])
    return table


# ======================================================================================
# Frozen backbone
# ======================================================================================


def load_frozen_backbone(cfg: config_module.MotJepaPosttrainConfig, device: torch.device) -> MotJepaStudent:
    """Rebuild the pretrained student and freeze its backbone.

    The **teacher** shadow is loaded, not the student: the EMA is the artifact pretraining was
    selecting for, and it is what every probe number was measured on.

    ``EmaTeacher.state_dict`` serializes the fp32 shadow as a **positional** list keyed
    ``shadow.0 ... shadow.N-1``, ordered by ``backbone.parameters()`` (``ema.py:74,144``) --
    not as named parameters. Loading it with ``load_state_dict(..., strict=False)`` therefore
    matches nothing at all and leaves the backbone at its random init, with the run looking
    entirely healthy. Hence the positional copy and the hard count check.
    """
    student = MotJepaStudent(cfg.layout, cfg.encoder, cfg.predictor, lowdim_channels=cfg.data.lowdim_channels)
    if not cfg.pretrained_run:
        logger.warning("no pretrained_run set; the backbone is RANDOM. Debug only.")
    else:
        checkpoint_dir = pathlib.Path(cfg.pretrained_run) / "checkpoints"
        step = cfg.pretrained_step or runtime.find_latest_step(checkpoint_dir)
        if step is None:
            raise FileNotFoundError(f"no checkpoint under {checkpoint_dir}")
        shadow = torch.load(checkpoint_dir / str(step) / "teacher_ema.pt", map_location="cpu", weights_only=True)
        params = list(student.backbone.parameters())
        if len(shadow) != len(params):
            raise RuntimeError(
                f"checkpoint has {len(shadow)} shadow tensors but this backbone has {len(params)} "
                f"parameters; the encoder/layout config does not match {checkpoint_dir}"
            )
        with torch.no_grad():
            for index, param in enumerate(params):
                tensor = shadow[f"shadow.{index}"]
                if tensor.shape != param.shape:
                    raise RuntimeError(f"shadow.{index} is {tuple(tensor.shape)}, expected {tuple(param.shape)}")
                param.copy_(tensor.to(param.dtype))
        logger.info("loaded frozen backbone from %s step %d (%d tensors)", checkpoint_dir, step, len(params))

    student = student.to(device)
    student.backbone.requires_grad_(requires_grad=False)
    student.backbone.eval()
    return student


# ======================================================================================
# Stage 3 -- clip <-> language alignment
# ======================================================================================


class InstructionStage(torch.nn.Module):
    """Wraps the only trainable module so DDP builds one reducer over it."""

    def __init__(self, cfg: config_module.MotJepaPosttrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head = InstructionHead(
            cfg.layout,
            text_dim=cfg.stage3.text_dim,
            hidden=cfg.stage3.hidden,
            projector_dim=cfg.stage3.projector_dim,
        )

    def forward(self, encoded, text: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        clip, projected = self.head(encoded, text)
        result = info_nce(clip, projected, labels, temperature=self.cfg.stage3.temperature)
        return result["loss"], {
            "instruction_top1": float(result["top1"]),
            "instruction_chance": float(result["chance"]),
            "instruction_gap": float(result["top1"] - result["chance"]),
            "distinct_tasks": float(result["distinct_tasks"]),
        }


def surrogate_table(num_stores: int, text_dim: int, device: torch.device) -> torch.Tensor:
    """One fixed random vector per store, carrying no language at all.

    Built once, from a fixed seed, so every rank and every step sees the same table -- a
    control that varied per step would be measuring noise rather than the head.
    """
    generator = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(num_stores, text_dim, generator=generator).to(device)


def store_identity_control(stage: InstructionStage, encoded, labels: torch.Tensor, surrogate: torch.Tensor) -> dict:
    """Stage 3's falsifier: swap each instruction for its store's random surrogate.

    If top-1 is unchanged the head learned store identity, not language, and the stage has
    failed. Passing retrieval *and* passing this is a failure, not a partial success -- the
    same reading error that made a mixed-domain pretraining run look like it had learned
    binding when the control had reached the identical top-1.
    """
    with torch.no_grad():
        loss, metrics = stage(encoded, surrogate[labels], labels)
    return {"control_" + key: value for key, value in metrics.items()} | {"control_loss": float(loss)}


# ======================================================================================
# Stage 4 -- action-conditioned latent rollout
# ======================================================================================


class ActionStage(torch.nn.Module):
    """A fresh predictor plus the action embedder. The encoder is not part of this."""

    def __init__(self, cfg: config_module.MotJepaPosttrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout = cfg.layout
        self.predictor = MoTPredictor(cfg.predictor, cfg.layout)
        self.action_embed = ActionEmbed(cfg.predictor.width)
        widths = (cfg.layout.video_width, cfg.layout.tactile_width)
        self.transition_proj = torch.nn.Linear(sum(widths), cfg.stage4.projector_dim)
        self.action_proj = torch.nn.Linear(cfg.predictor.width, cfg.stage4.projector_dim)

    def forward(self, encoded, masks, action, action_mask):
        context_index, _ = split_index_by_expert(self.layout, masks.ctx_index, masks.ctx_expert_bounds)
        target_index = target_index_by_expert(self.layout, masks.tgt_index, masks.tgt_expert_bounds)
        cond = self.action_embed(action, action_mask)
        predictions = self.predictor(encoded.tokens, context_index, target_index, masks.mode, cond=cond)
        return predictions, cond


def rollout_targets(teacher_tokens: list[torch.Tensor], masks, layout) -> list[torch.Tensor]:
    """Teacher latents at the target positions. Same slicing as pretraining's gather_targets."""
    targets = []
    for expert, tokens in enumerate(teacher_tokens):
        offset = 0 if expert == 0 else layout.num_video_tokens
        lo, hi = (int(v) for v in masks.tgt_expert_bounds[expert])
        index = (masks.tgt_index[:, lo:hi] - offset).to(tokens.device)
        gathered = tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1]))
        targets.append(normalize_targets(gathered))
    return targets


def action_donor_ratio(stage, encoded, masks, action, action_mask, targets) -> float:
    """Prediction error under a donor clip's actions divided by error under the true ones.

    At 1.0 the predictor ignores the action entirely and has learned an unconditional
    dynamics prior -- the Stage 4 analogue of ``shuffle == real``.
    """
    with torch.no_grad():
        donor = torch.roll(action, shifts=1, dims=0)
        true_pred, _ = stage(encoded, masks, action, action_mask)
        donor_pred, _ = stage(encoded, masks, donor, action_mask)
        true_error = sum(jepa_regression_loss(p, t) for p, t in zip(true_pred, targets, strict=True))
        donor_error = sum(jepa_regression_loss(p, t) for p, t in zip(donor_pred, targets, strict=True))
    return float(true_error / donor_error.clamp(min=1e-6))


# ======================================================================================
# Loop
# ======================================================================================


def train(cfg: config_module.MotJepaPosttrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if cfg.stage not in (STAGE_INSTRUCTION, STAGE_ACTION):
        raise ValueError(f"stage must be {STAGE_INSTRUCTION!r} or {STAGE_ACTION!r}, got {cfg.stage!r}")

    use_ddp, local_rank, device = runtime.setup_ddp()
    rank, world_size = runtime.get_rank(), runtime.get_world_size()

    run_dir = cfg.run_dir
    if runtime.is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
    runtime.barrier()

    frozen_json, drift = runtime.resolve_run_config(run_dir, cfg.to_json())
    if drift and runtime.is_main_process():
        logger.warning("run_config.json differs from the CLI config; the FROZEN one wins:")
        for key, (frozen_value, incoming) in drift.items():
            logger.warning("  %s: frozen=%r cli=%r", key, frozen_value, incoming)
    cfg = config_module.MotJepaPosttrainConfig.from_json(frozen_json)

    layout = cfg.layout
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    resume_step = runtime.find_latest_step(cfg.checkpoint_dir)
    init_tracking(cfg, run_dir, resuming=resume_step is not None)

    dataset, domain_of_sample, task_of_store = build_dataset(cfg)
    task_of_store_t = torch.from_numpy(task_of_store)
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
        collate_fn=collate_clips,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
    )

    frozen = load_frozen_backbone(cfg, device)
    backbone = frozen.backbone

    if cfg.stage == STAGE_INSTRUCTION:
        stage: torch.nn.Module = InstructionStage(cfg).to(device)
        table = (
            load_instruction_table(cfg.stage3.instruction_emb, device)
            if cfg.stage3.instruction_emb
            else torch.randn(4096, cfg.stage3.text_dim, device=device)
        )
        surrogate = surrogate_table(int(task_of_store.max()) + 1, cfg.stage3.text_dim, device)
    else:
        stage = ActionStage(cfg).to(device)
        table = surrogate = None

    trainable = [p for p in stage.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr_peak, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
    )
    logger.info(
        "trainable %.2fM parameters; backbone frozen at %.2fM",
        sum(p.numel() for p in trainable) / 1e6,
        sum(p.numel() for p in backbone.parameters()) / 1e6,
    )

    global_step = 0
    if resume_step is not None:
        global_step = runtime.load_checkpoint(
            cfg.checkpoint_dir, resume_step, student=stage, teacher=None, optimizer=optimizer, device=device
        )
        logger.info("resumed from step %d", global_step)
    sampler.set_start_step(global_step)

    model = stage
    if use_ddp:
        model = DistributedDataParallel(
            stage,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=cfg.find_unused_parameters,
            gradient_as_bucket_view=True,
        )

    monitor = runtime.PreemptionMonitor(run_dir / "PREEMPT_REQUEST", device)
    records: list[dict[str, float]] = []
    loader_iter = iter(loader)
    preempted = False
    window_start = time.time()

    while global_step < cfg.num_train_steps:
        if monitor.should_stop(global_step):
            logger.info("preemption requested at step %d; checkpointing", global_step)
            preempted = True
            break

        batch = next(loader_iter)
        inputs = to_inputs(batch, device)
        lr = lr_at(global_step, peak=cfg.lr_peak, end=cfg.lr_end, warmup=cfg.lr_warmup_steps, total=cfg.num_train_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        extras: dict[str, float] = {}
        if cfg.stage == STAGE_INSTRUCTION:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                encoded = backbone.encode_full(inputs)
            # Label by TASK. Not by string (a store's episodes are paraphrases of one task)
            # and not by store (sharpa's 164 stores carry 10 tasks, so store labels would
            # push apart clips whose instructions are word-for-word identical).
            labels = task_of_store_t[batch["store_idx"]].to(device)
            text = table[batch["instruction_id"].clamp(min=0).to(device)]
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                loss, extras = model(encoded, text, labels)
            if global_step % cfg.stage3.control_interval == 0:
                extras |= store_identity_control(stage, encoded, labels, surrogate)
        else:
            masks = build_rollout_masks(layout, split_step=cfg.stage4.split_step, batch_size=cfg.local_batch_size).to(
                device
            )
            action_mask = batch["action_mask"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                teacher_out = backbone.encode_full(inputs)
                encoded = backbone(inputs, masks.ctx_index, masks.ctx_expert_bounds)
            targets = rollout_targets(teacher_out.tokens, masks, layout)
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                predictions, cond = model(encoded, masks, action, action_mask)
                latent = sum(jepa_regression_loss(p, t) for p, t in zip(predictions, targets, strict=True))
                sync = action_sync_loss(stage, encoded, cond, cfg)
                loss = cfg.stage4.weight_latent * latent + cfg.stage4.weight_action_sync * sync
            extras = {"latent": float(latent.detach()), "action_sync": float(sync.detach())}
            if global_step % cfg.stage4.donor_interval == 0:
                extras["action_donor_ratio"] = action_donor_ratio(stage, encoded, masks, action, action_mask, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {global_step}: {loss.item()}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.clip_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        records.append({"loss": float(loss.detach()), "learning_rate": lr, "grad_norm": float(grad_norm), **extras})
        global_step += 1

        if global_step % cfg.log_interval == 0:
            metrics = reduce_metrics(records, device)
            elapsed = time.time() - window_start
            metrics["clips_per_second"] = cfg.log_interval * cfg.local_batch_size * world_size / max(elapsed, 1e-6)
            if runtime.is_main_process():
                logger.info(
                    "step %d loss %.4f grad %.3f lr %.2e %.1f clips/s %s",
                    global_step,
                    metrics.get("loss", float("nan")),
                    metrics.get("grad_norm", float("nan")),
                    lr,
                    metrics["clips_per_second"],
                    {k: round(v, 4) for k, v in metrics.items() if k.startswith(("instruction", "action", "control"))},
                )
                if cfg.wandb_enabled:
                    wandb.log({f"{cfg.stage}/{k}": v for k, v in metrics.items()}, step=global_step)
            records.clear()
            window_start = time.time()

        if global_step % cfg.save_interval == 0 and runtime.is_main_process():
            runtime.save_checkpoint(
                cfg.checkpoint_dir,
                global_step,
                student=stage,
                teacher=None,
                optimizer=optimizer,
                config_json=frozen_json,
                keep_last=cfg.keep_last,
                keep_period=cfg.keep_period,
            )

    if runtime.is_main_process():
        runtime.save_checkpoint(
            cfg.checkpoint_dir,
            global_step,
            student=stage,
            teacher=None,
            optimizer=optimizer,
            config_json=frozen_json,
            keep_last=cfg.keep_last,
            keep_period=cfg.keep_period,
        )
        if not preempted:
            (run_dir / "DONE").write_text(str(global_step))

    runtime.barrier()
    if runtime.is_main_process() and cfg.wandb_enabled:
        wandb.finish()
    runtime.cleanup_ddp()
    sys.exit(0)


def action_sync_loss(stage, encoded, cond: torch.Tensor, cfg) -> torch.Tensor:
    """InfoNCE matching a clip's observed transition to its own action sequence.

    The latent regression alone has a solution that ignores the action entirely -- predict the
    unconditional mean future. This term is the one that is provably lower for the matched
    (clip, action) pair, exactly the role ``L_sync`` plays during pretraining.
    """
    core = stage.module if hasattr(stage, "module") else stage
    transition = torch.nn.functional.normalize(core.transition_proj(pool_experts(encoded)).float(), dim=-1)
    summary = torch.nn.functional.normalize(core.action_proj(cond.mean(dim=1)).float(), dim=-1)
    labels = torch.arange(transition.shape[0], device=transition.device)
    return info_nce(transition, summary, labels, temperature=cfg.stage4.temperature)["loss"]


def main() -> None:
    train(config_module.posttrain_cli())


if __name__ == "__main__":
    main()
