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
    *,
    partition: str = "all",
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

    if partition != "all" and cfg.stage3.holdout_frac > 0:
        heldout = split_tasks(task_ids, cfg.stage3.holdout_frac)
        want = heldout if partition == "heldout" else ~heldout
        keep = [i for i in range(len(stores)) if want[i]]
        stores = [stores[i] for i in keep]
        domain_ids = [domain_ids[i] for i in keep]
        task_ids = [task_ids[i] for i in keep]
        if not stores:
            raise ValueError(f"the {partition!r} partition is empty; lower stage3.holdout_frac")
    logger.info(
        "%s: %d stores / %d tasks across %d domains",
        partition,
        len(stores),
        len(set(task_ids)),
        len(set(domain_ids)),
    )

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


def split_tasks(task_ids: list[int], holdout_frac: float) -> np.ndarray:
    """Deterministic per-TASK train/eval split, returned as a boolean "is held out" mask.

    Split by task rather than by clip or by store: holding out clips of a task the head also
    trains on measures memorisation, not generalisation. Deterministic in the task id so every
    rank and every requeue agree without communicating.
    """
    unique = sorted(set(task_ids))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(unique))
    cutoff = round(len(unique) * holdout_frac)
    heldout = {unique[i] for i in order[:cutoff]}
    return np.asarray([task in heldout for task in task_ids], dtype=bool)


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


def _pin_backbone_step(checkpoint_dir: pathlib.Path, run_dir: pathlib.Path) -> int | None:
    """Resolve "latest" ONCE and record it, so a requeue cannot swap the backbone.

    A post-training run is a chain of ~40 requeued jobs, and the pretraining run it freezes
    may still be writing checkpoints. Re-resolving "latest" on every attempt would silently
    change the frozen backbone partway through the chain, splicing two different experiments
    into one W&B history -- the same hazard ``resolve_run_config`` exists to prevent for
    hyperparameters. Pinning at launch with ``--pretrained_step`` is preferred; this is the
    backstop for when it is omitted.
    """
    pin = run_dir / "backbone_step.txt"
    if pin.exists():
        step = int(pin.read_text().strip())
        logger.info("reusing pinned backbone step %d from %s", step, pin)
        return step
    step = runtime.find_latest_step(checkpoint_dir)
    if step is None:
        return None
    if runtime.is_main_process():
        staging = pin.with_suffix(f".{os.getpid()}.tmp")
        staging.write_text(str(step))
        os.replace(staging, pin)
    logger.warning(
        "pretrained_step was not set; pinning the backbone at step %d for the whole run. "
        "Pass --pretrained_step explicitly to make this reproducible from the launcher.",
        step,
    )
    return step


def load_frozen_backbone(
    cfg: config_module.MotJepaPosttrainConfig, device: torch.device, run_dir: pathlib.Path
) -> MotJepaStudent:
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
        step = cfg.pretrained_step or _pin_backbone_step(checkpoint_dir, run_dir)
        if step is None:
            raise FileNotFoundError(f"no checkpoint under {checkpoint_dir}")
        source = checkpoint_dir / str(step) / "teacher_ema.pt"
        if not source.exists():
            # A *live* pretraining run prunes its own history: keep_last=3 plus every
            # keep_period. Pinning to a step it has since deleted fails here, minutes after
            # the same step loaded fine. Snapshot the backbone somewhere the source run
            # cannot reach before depending on it.
            available = sorted((int(p.name) for p in checkpoint_dir.iterdir() if p.name.isdigit()), reverse=True)
            raise FileNotFoundError(
                f"{source} is missing. The source run has probably pruned it -- it retains only "
                f"keep_last plus every keep_period. Available now: {available[:6]}. Copy the "
                f"checkpoint to a stable directory and point --pretrained_run at that instead."
            )
        shadow = torch.load(source, map_location="cpu", weights_only=True)
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


def surrogate_table(vocabulary_size: int, text_dim: int, device: torch.device) -> torch.Tensor:
    """One fixed random vector per INSTRUCTION, carrying no language at all.

    Per instruction rather than per task, so it destroys paraphrase grouping as well as
    semantics: a task's prototype becomes the mean of unrelated random vectors. That makes it
    a strictly harder control than a per-task surrogate would be.

    This is the **control arm**, used for a whole run via ``stage3.surrogate_text`` -- not as
    an inline probe. An inline version would feed random vectors through a ``text_proj``
    fitted to SigLIP's space and sit at chance no matter what the head learned, so it tests
    only that the text input is used at all, never that *language* is.

    Read the control against the real arm as a RATIO, not by whether either clears chance.
    Both do: the held-out gallery spans ten domains, so recognising the domain narrows fifty
    candidates to about five without understanding anything, and that shortcut is available to
    both arms equally. The language contribution is what remains after dividing it out.
    """
    generator = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(vocabulary_size, text_dim, generator=generator).to(device)


def build_task_gallery(dataset, task_of_store: np.ndarray, table: torch.Tensor, device: torch.device):
    """One prototype embedding per held-out task, plus the task id of each gallery row.

    A prototype is the L2-normalized mean of that task's paraphrase embeddings, so retrieval
    is against *the task*, not against whichever wording a particular episode happened to use.
    """
    by_task: dict[int, list[int]] = collections.defaultdict(list)
    for store_index, path in enumerate(dataset.store_paths):
        try:
            ids = np.asarray(zarr.open(path, mode="r")["meta/instruction_id"][:])
        except Exception:
            continue
        by_task[int(task_of_store[store_index])].extend(int(i) for i in ids)

    task_order = sorted(by_task)
    prototypes = torch.stack(
        [torch.nn.functional.normalize(table[sorted(set(by_task[t]))].float().mean(0), dim=-1) for t in task_order]
    ).to(device)
    return prototypes, torch.tensor(task_order, dtype=torch.int64, device=device)


def fixed_eval_indices(dataset, task_of_store: np.ndarray, *, per_task: int = 8) -> list[int]:
    """A fixed, task-balanced set of clip indices, chosen once and reused at every eval.

    Drawing a fresh batch each time makes the metric track *which domain got sampled* rather
    than the model: eval batches are domain-pure, so consecutive evaluations of one run swung
    4.5x -> 4.3x -> 1.0x with the weights barely moving. Evenly spaced picks per task, same
    clips every time, so a change in the number is a change in the head.
    """
    by_task: dict[int, list[int]] = collections.defaultdict(list)
    for row, entry in enumerate(dataset.clip_index.entries):
        by_task[int(task_of_store[int(entry[0])])].append(row)

    chosen: list[int] = []
    for task in sorted(by_task):
        rows = by_task[task]
        take = min(per_task, len(rows))
        chosen.extend(rows[i] for i in np.linspace(0, len(rows) - 1, take).astype(int))
    return chosen


@torch.no_grad()
def heldout_retrieval(
    stage, backbone, loader, gallery, gallery_tasks, task_of_store, device, *, batches: int | None = None
) -> dict:
    """Retrieval against a gallery of EVERY held-out task, not just the ones in this batch.

    This is Stage 3's real falsifier. Within this corpus instruction and task are bijective --
    each task is a distinct store with its own objects and lighting -- so on *seen* tasks
    "recognise the store and emit its vector" and "understand the instruction" are
    behaviourally identical, and no within-batch control can tell them apart. On unseen tasks
    a lookup has nothing to look up.

    Scoring against the full gallery matters. Eval batches are domain-pure, so the tasks that
    happen to co-occur in one batch number only three or four and chance sits near 0.35 -- a
    ratio measured that way is mostly noise. Against every held-out task the pool is ~50 and
    chance is ~0.02, which is what makes the number worth reading.
    """
    core = stage.module if hasattr(stage, "module") else stage
    was_training = core.training
    core.eval()

    projected_gallery = torch.nn.functional.normalize(core.head.text_proj(gallery).float(), dim=-1)
    correct = total = 0
    for seen, batch in enumerate(loader):
        if batches is not None and seen >= batches:
            break
        inputs = to_inputs(batch, device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            encoded = backbone.encode_full(inputs)
            clip = torch.nn.functional.normalize(core.head.clip_proj(pool_experts(encoded)).float(), dim=-1)
        predicted = gallery_tasks[(clip @ projected_gallery.t()).argmax(dim=-1)]
        correct += int((predicted == task_of_store[batch["store_idx"]].to(device)).sum())
        total += predicted.numel()
    core.train(was_training)

    mean_top1 = correct / max(total, 1)
    mean_chance = 1.0 / max(gallery_tasks.numel(), 1)
    return {
        "heldout_top1": mean_top1,
        "heldout_chance": mean_chance,
        "heldout_gap": mean_top1 - mean_chance,
        "heldout_ratio_to_chance": mean_top1 / max(mean_chance, 1e-9),
        "heldout_gallery_tasks": float(gallery_tasks.numel()),
    }


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

    # Stage 3 trains on one task partition and evaluates on the other; Stage 4 has no
    # held-out notion and uses everything.
    train_partition = "train" if cfg.stage == STAGE_INSTRUCTION else "all"
    dataset, domain_of_sample, task_of_store = build_dataset(cfg, partition=train_partition)
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

    def make_loader(ds, sam):
        return torch.utils.data.DataLoader(
            ds,
            batch_sampler=sam,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_clips,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.data.num_workers > 0,
            prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        )

    frozen = load_frozen_backbone(cfg, device, run_dir)
    backbone = frozen.backbone

    if cfg.stage == STAGE_INSTRUCTION:
        stage: torch.nn.Module = InstructionStage(cfg).to(device)
        table = (
            load_instruction_table(cfg.stage3.instruction_emb, device)
            if cfg.stage3.instruction_emb
            else torch.randn(4096, cfg.stage3.text_dim, device=device)
        )
        if cfg.stage3.surrogate_text:
            # Control arm: every instruction becomes a fixed random vector for the WHOLE run.
            # If held-out top-1 matches the real-embedding arm, language contributed nothing.
            logger.warning("SURROGATE TEXT: instructions replaced by random per-task vectors")
            table = surrogate_table(table.shape[0], cfg.stage3.text_dim, device)

        eval_dataset, _, eval_task_of_store = build_dataset(cfg, partition="heldout")
        eval_indices = fixed_eval_indices(eval_dataset, eval_task_of_store)
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(eval_dataset, eval_indices),
            batch_size=cfg.local_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_clips,
            pin_memory=torch.cuda.is_available(),
        )
        eval_task_of_store_t = torch.from_numpy(eval_task_of_store)
        gallery, gallery_tasks = build_task_gallery(eval_dataset, eval_task_of_store, table, device)
        logger.info(
            "held-out gallery: %d task prototypes (chance %.4f)", gallery_tasks.numel(), 1 / gallery_tasks.numel()
        )
    else:
        stage = ActionStage(cfg).to(device)
        table = None

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
        # Interval-sampled falsifier metrics are kept OUT of `records`. reduce_metrics fills a
        # missing key with 0.0 -- deliberate for the mode-frequency counters in pretraining,
        # but it silently divides an interval-sampled scalar by the window length, so a
        # control at chance reads as 50x below the number it is meant to be compared against.
        sparse: dict[str, float] = {}
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
            if global_step % cfg.stage3.eval_interval == 0:
                sparse = heldout_retrieval(
                    stage, backbone, eval_loader, gallery, gallery_tasks, eval_task_of_store_t, device
                )

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
                sparse = {"action_donor_ratio": action_donor_ratio(stage, encoded, masks, action, action_mask, targets)}

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

        if sparse and runtime.is_main_process():
            logger.info("falsifier @%d: %s", global_step, {k: round(v, 4) for k, v in sparse.items()})
            if cfg.wandb_enabled:
                wandb.log({f"{cfg.stage}/{k}": v for k, v in sparse.items()}, step=global_step)

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
