#!/usr/bin/env python
"""Train an action policy from a pretrained MoT-JEPA encoder.

Two arms behind one script, differing in exactly one config field:

``mot_jepa_policy_drifting``   Implicit Drifting Policy -- one-step generation, 1 NFE.
``mot_jepa_policy_flowmatch``  Conditional flow matching -- the control, 10 Euler steps.

The control exists because this pipeline changes two things at once relative to the FTP-1 policy:
the conditioning encoder (frozen MoT-JEPA rather than PaliGemma) *and* the generative objective.
An arm that changes only the encoder is what makes a bad result attributable.

The historical default keeps the encoder frozen. ``backbone_train_mode`` can instead adapt its
last blocks or every reachable parameter; in those modes the encoder and head sit behind one DDP
reducer and the adapted backbone is an explicit checkpoint artifact. Requeue, checkpointing and
sample-order continuity are inherited from the pretraining trainer.

    torchrun ... scripts/mot_jepa_policy_train.py mot_jepa_policy_drifting \
        --pretrained_run .../backbones/probe2_s50000 --pretrained_step 50000 \
        --data.store_glob '.../ftp1-clips/*/*.zarr'
"""

from __future__ import annotations

import glob
import hashlib
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
from openpi.mot_jepa.action_dit import ActionNormalizer
from openpi.mot_jepa.action_dit import LinearHead
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import load_domain_config
from openpi.mot_jepa.clip_dataset import split_clip_index
from openpi.mot_jepa.policy_finetune import BackboneSelection
from openpi.mot_jepa.policy_finetune import PolicyTrainModel
from openpi.mot_jepa.policy_finetune import configure_backbone_trainability
from openpi.mot_jepa.policy_finetune import grad_norm
from openpi.mot_jepa.policy_finetune import make_step_generator
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

    dataset = MotJepaClipDataset(
        stores,
        cfg.layout,
        domain_ids=domain_ids,
        strides=cfg.data.strides,
        index_step=cfg.data.index_step,
        lowdim_channels=cfg.data.lowdim_channels,
        with_conditioning=True,
        action_horizon=cfg.head.horizon,
        action_stride=cfg.data.action_stride,
    )
    # Domain ids are only meaningful together with this exact ordering. Persist the names in
    # action_stats.npz so adding or renaming a store cannot silently route a checkpoint through a
    # different task-specific embedding/output row at evaluation time.
    dataset.domain_names = tuple(names)
    if cfg.holdout_mod > 1:
        before = len(dataset)
        dataset.clip_index = split_clip_index(dataset.clip_index, holdout_mod=cfg.holdout_mod, want="train")
        logger.info("holdout_mod=%d: training on %d of %d clips", cfg.holdout_mod, len(dataset), before)
    return dataset


def fit_action_stats(
    dataset: MotJepaClipDataset,
    num_domains: int,
    path: pathlib.Path,
    *,
    observation_strides: tuple[int, ...],
    action_stride: int,
    domain_names: tuple[str, ...],
    clips_per_domain: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-domain, per-dimension mean/std of the action chunk, cached to ``path``.

    Computed once by rank 0 and reused across the whole requeue chain: recomputing per job would
    make the target distribution drift between checkpoints, so a resumed head would be predicting
    in a slightly different space than the one it was trained in.

    Only live slots contribute. A masked-out slot is structurally absent, and letting its zeros
    into the mean would drag every live statistic toward zero in proportion to how many slots the
    embodiment happens to leave empty.
    """
    if path.exists():
        blob = np.load(path)
        required = {"observation_strides", "action_stride", "horizon", "domains"}
        missing = sorted(required - set(blob.files))
        if missing:
            raise ValueError(f"{path} lacks sampling metadata {missing}; refuse to reuse stale action statistics")
        stored_observation = tuple(int(v) for v in np.asarray(blob["observation_strides"]).reshape(-1))
        stored_action = int(np.asarray(blob["action_stride"]).item())
        stored_horizon = int(np.asarray(blob["horizon"]).item())
        stored_domains = tuple(str(name) for name in blob["domains"])
        expected = (observation_strides, action_stride, dataset.action_horizon)
        actual = (stored_observation, stored_action, stored_horizon)
        if actual != expected:
            raise ValueError(f"{path} sampling metadata is {actual}, expected {expected}; refuse stale statistics")
        if stored_domains != domain_names:
            raise ValueError(f"{path} domain mapping is {stored_domains}, expected {domain_names}")
        return torch.from_numpy(blob["mean"]), torch.from_numpy(blob["scale"])

    total = np.zeros((num_domains, 120), dtype=np.float64)
    total_sq = np.zeros((num_domains, 120), dtype=np.float64)
    count = np.zeros((num_domains, 120), dtype=np.float64)

    domain_of_sample = np.asarray(dataset.domain_ids, dtype=np.int64)[dataset.clip_index.entries[:, 0]]
    rng = np.random.default_rng(0)
    picks: list[int] = []
    for domain in range(num_domains):
        pool = np.flatnonzero(domain_of_sample == domain)
        if pool.size:
            picks.extend(rng.choice(pool, size=min(clips_per_domain, pool.size), replace=False).tolist())

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, picks), batch_size=64, num_workers=8, collate_fn=collate_clips
    )
    for batch in loader:
        chunk = batch["action_chunk"].numpy().astype(np.float64)
        mask = batch["chunk_mask"].numpy().astype(np.float64)
        for row, domain in enumerate(batch["domain_id"].tolist()):
            total[domain] += (chunk[row] * mask[row]).sum(axis=0)
            total_sq[domain] += ((chunk[row] ** 2) * mask[row]).sum(axis=0)
            count[domain] += mask[row].sum(axis=0)

    safe = np.maximum(count, 1.0)
    mean = total / safe
    scale = np.sqrt(np.maximum(total_sq / safe - mean**2, 0.0))
    mean[count == 0] = 0.0
    scale[count == 0] = 1.0

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        observation_strides=np.asarray(observation_strides, dtype=np.int64),
        action_stride=np.asarray(action_stride, dtype=np.int64),
        horizon=np.asarray(dataset.action_horizon, dtype=np.int64),
        domains=np.asarray(domain_names),
    )
    logger.info(
        "action stats over %d clips: median live scale %.5f (min %.5f, max %.5f)",
        len(picks),
        float(np.median(scale[count > 0])),
        float(scale[count > 0].min()),
        float(scale[count > 0].max()),
    )
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(scale.astype(np.float32))


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


def optimizer_step_bounds(optimizer: torch.optim.Optimizer) -> tuple[int, int] | None:
    """Min/max Adam counter across parameters that have entered the executed graph."""
    steps = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        steps.append(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
    return (min(steps), max(steps)) if steps else None


def gather_peak_gpu_memory(device: torch.device) -> list[dict[str, int]]:
    """Collect allocated/reserved CUDA peaks from every rank in stable rank order."""
    if device.type != "cuda":
        return [{"allocated": 0, "reserved": 0}]
    local = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.int64,
        device=device,
    )
    if runtime.get_world_size() > 1:
        gathered = [torch.zeros_like(local) for _ in range(runtime.get_world_size())]
        torch.distributed.all_gather(gathered, local)
    else:
        gathered = [local]
    return [
        {"allocated": int(values[0].item()), "reserved": int(values[1].item())}
        for values in gathered
    ]


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def head_initialization_provenance(cfg: config_module.PolicyConfig) -> dict:
    """Validate and stamp the completed policy checkpoint used for a fresh adaptation run."""
    if not cfg.init_head_from:
        return {
            "init_head_path": None,
            "init_head_sha256": None,
            "init_head_source_run": None,
            "init_head_source_step": None,
            "init_head_source_config_sha256": None,
        }

    init_path = pathlib.Path(cfg.init_head_from).resolve()
    if init_path.name != "student.pt" or init_path.parent.parent.name != "checkpoints":
        raise ValueError(f"init_head_from must be <completed_run>/checkpoints/<step>/student.pt, got {init_path}")
    try:
        source_step = int(init_path.parent.name)
    except ValueError as exc:
        raise ValueError(f"head source checkpoint directory is not an integer step: {init_path.parent}") from exc
    source_run = init_path.parent.parent.parent
    expected_path = source_run / "checkpoints" / str(source_step) / "student.pt"
    if init_path != expected_path.resolve() or not init_path.is_file():
        raise FileNotFoundError(f"missing canonical head source {expected_path}")
    done_path = source_run / "DONE"
    latest_path = source_run / "checkpoints" / "latest"
    if not done_path.is_file() or int(done_path.read_text().strip()) != source_step:
        raise ValueError(f"head source is not a completed run at step {source_step}: {done_path}")
    if not latest_path.is_file() or int(latest_path.read_text().strip()) != source_step:
        raise ValueError(f"head source latest pointer does not equal completed step {source_step}: {latest_path}")

    source_config_path = source_run / "run_config.json"
    source_payload = json.loads(source_config_path.read_text())
    target_payload = json.loads(cfg.to_json())
    exact_fields = ("name", "layout_preset", "encoder", "predictor", "head", "holdout_mod", "seed")
    mismatches = {
        field: (source_payload.get(field), target_payload.get(field))
        for field in exact_fields
        if source_payload.get(field) != target_payload.get(field)
    }
    data_fields = (
        "store_glob",
        "domain_config",
        "strides",
        "index_step",
        "lowdim_channels",
        "lowdim_log_compress",
        "action_stride",
    )
    for field in data_fields:
        source_value = source_payload.get("data", {}).get(field)
        target_value = target_payload.get("data", {}).get(field)
        if source_value != target_value:
            mismatches[f"data.{field}"] = (source_value, target_value)
    source_pretrained_run = pathlib.Path(str(source_payload.get("pretrained_run", ""))).resolve()
    target_pretrained_run = pathlib.Path(cfg.pretrained_run).resolve()
    if source_pretrained_run != target_pretrained_run:
        mismatches["pretrained_run"] = (str(source_pretrained_run), str(target_pretrained_run))
    if source_payload.get("pretrained_step") != cfg.pretrained_step:
        mismatches["pretrained_step"] = (source_payload.get("pretrained_step"), cfg.pretrained_step)
    if mismatches:
        raise ValueError(f"head source/target policy contract differs: {mismatches}")

    return {
        "init_head_path": str(init_path),
        "init_head_sha256": file_sha256(init_path),
        "init_head_source_run": str(source_run.resolve()),
        "init_head_source_step": source_step,
        "init_head_source_config_sha256": file_sha256(source_config_path),
    }


def policy_checkpoint_metadata(
    *,
    cfg: config_module.PolicyConfig,
    source_backbone_step: int,
    backbone_selection: BackboneSelection,
    optimizer: torch.optim.Optimizer,
    head_parameters: tuple[torch.nn.Parameter, ...],
    world_size: int,
    last_record: dict[str, float] | None,
    initialization_provenance: dict,
    peak_gpu_memory_by_rank: list[dict[str, int]],
) -> dict:
    """Self-describing provenance for a policy/backbone/optimizer tuple."""
    bounds = optimizer_step_bounds(optimizer)
    return {
        "policy_checkpoint_schema": 2,
        "backbone_train_mode": cfg.backbone_train_mode,
        "backbone_last_n_blocks": cfg.backbone_last_n_blocks,
        "backbone_lr_multiplier": cfg.backbone_lr_multiplier,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "source_backbone_run": str(pathlib.Path(cfg.pretrained_run).resolve()),
        "source_backbone_step": source_backbone_step,
        "trainable_backbone_tensors": len(backbone_selection.parameters),
        "trainable_backbone_parameters": backbone_selection.num_parameters,
        "trainable_backbone_name_sha256": backbone_selection.name_digest,
        "excluded_unreachable_backbone_parameters": list(backbone_selection.excluded_unreachable),
        "trainable_head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "optimizer_step_min": bounds[0] if bounds else 0,
        "optimizer_step_max": bounds[1] if bounds else 0,
        "optimizer_groups": [
            {
                "name": str(group.get("group_name", f"group_{index}")),
                "lr": float(group["lr"]),
                "lr_multiplier": float(group.get("lr_multiplier", 1.0)),
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
        "world_size": world_size,
        "last_train_metrics": dict(last_record or {}),
        "peak_gpu_memory_bytes_by_rank": peak_gpu_memory_by_rank,
        **initialization_provenance,
    }


def train(cfg: config_module.PolicyConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
    cfg = config_module.PolicyConfig.from_json(frozen_json)
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    if not cfg.pretrained_run:
        raise ValueError("--pretrained_run is required; the head is meaningless without an encoder")
    initialization_provenance = head_initialization_provenance(cfg)
    backbone, backbone_step = runtime.load_frozen_backbone(
        pathlib.Path(cfg.pretrained_run), cfg.pretrained_step, cfg, device
    )
    backbone_selection = configure_backbone_trainability(
        backbone,
        mode=cfg.backbone_train_mode,
        last_n_blocks=cfg.backbone_last_n_blocks,
    )
    backbone.set_gradient_checkpointing(
        enabled=cfg.gradient_checkpointing and cfg.backbone_train_mode != "frozen"
    )
    dataset = build_dataset(cfg)
    logger.info("%d clips at horizon %d", len(dataset), cfg.head.horizon)

    # The head predicts NORMALISED actions. Raw FTP-1 deltas are ~0.01 and vary 8x across
    # domains, which makes flow matching degenerate (the action is 1% of x_t, so echoing the
    # noise scores near-zero loss) and lets large-motion domains dominate a shared head.
    # num_domains also sizes the head's domain embedding, so the dataset must be built first.
    num_domains = int(max(dataset.domain_ids)) + 1

    head_cls = LinearHead if cfg.head.objective == "linear" else ActionDiT
    head = head_cls(cfg.head, cfg.layout, num_domains=num_domains).to(device)
    logger.info(
        "source backbone step %d; mode=%s (%d blocks, %.1fM trainable%s); "
        "head %s over %d domains, %.1fM trainable params",
        backbone_step,
        cfg.backbone_train_mode,
        cfg.backbone_last_n_blocks if cfg.backbone_train_mode == "last_blocks" else 0,
        backbone_selection.num_parameters / 1e6,
        f", excluded={backbone_selection.excluded_unreachable}" if backbone_selection.excluded_unreachable else "",
        cfg.head.objective,
        num_domains,
        sum(p.numel() for p in head.parameters()) / 1e6,
    )

    stats_path = run_dir / "action_stats.npz"
    if cfg.data.action_stride is None:
        raise ValueError("policy training requires data.action_stride")
    if runtime.is_main_process():
        fit_action_stats(
            dataset,
            num_domains,
            stats_path,
            observation_strides=cfg.data.strides,
            action_stride=cfg.data.action_stride,
            domain_names=dataset.domain_names,
        )
    runtime.barrier()
    normalizer = ActionNormalizer(num_domains).to(device)
    normalizer.load_stats(
        *[
            t.to(device)
            for t in fit_action_stats(
                dataset,
                num_domains,
                stats_path,
                observation_strides=cfg.data.strides,
                action_stride=cfg.data.action_stride,
                domain_names=dataset.domain_names,
            )
        ]
    )
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

    head_parameters = tuple(head.parameters())
    optimizer_groups: list[dict] = [
        {
            "params": head_parameters,
            "lr": cfg.lr_peak,
            "lr_multiplier": 1.0,
            "group_name": "head",
        }
    ]
    if backbone_selection.parameters:
        optimizer_groups.append(
            {
                "params": backbone_selection.parameters,
                "lr": cfg.lr_peak * cfg.backbone_lr_multiplier,
                "lr_multiplier": cfg.backbone_lr_multiplier,
                "group_name": "backbone",
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
    )

    resume_step = runtime.find_latest_step(cfg.checkpoint_dir)
    global_step = 0
    if resume_step is not None:
        resume_path = cfg.checkpoint_dir / str(resume_step)
        resume_metadata = torch.load(resume_path / "metadata.pt", map_location="cpu", weights_only=True)
        expected_resume_metadata = {
            "policy_checkpoint_schema": 2,
            "global_step": resume_step,
            "backbone_train_mode": cfg.backbone_train_mode,
            "backbone_last_n_blocks": cfg.backbone_last_n_blocks,
            "source_backbone_run": str(pathlib.Path(cfg.pretrained_run).resolve()),
            "source_backbone_step": backbone_step,
            "trainable_backbone_name_sha256": backbone_selection.name_digest,
            "world_size": world_size,
            **initialization_provenance,
        }
        resume_mismatches = {
            key: (resume_metadata.get(key), expected)
            for key, expected in expected_resume_metadata.items()
            if resume_metadata.get(key) != expected
        }
        if resume_mismatches:
            raise RuntimeError(
                "resume provenance/topology differs from the checkpoint; refuse a discontinuous "
                f"sample/RNG/weight stream: {resume_mismatches}"
            )
        resume_modules = (
            {"backbone": backbone}
            if (resume_path / "backbone.pt").exists() or cfg.backbone_train_mode != "frozen"
            else None
        )
        global_step = runtime.load_checkpoint(
            cfg.checkpoint_dir,
            resume_step,
            student=head,
            teacher=None,
            optimizer=optimizer,
            device=device,
            loss_fn=normalizer,
            extra_modules=resume_modules,
        )
        bounds = optimizer_step_bounds(optimizer)
        if bounds is not None and bounds != (global_step, global_step):
            raise RuntimeError(
                f"optimizer counters {bounds} do not match resumed global step {global_step}; "
                "refuse a discontinuous LR/noise sequence"
            )
        logger.info("resumed from step %d", global_step)
    elif cfg.init_head_from:
        # Fine-tuning: start from a head trained on another dataset, at step 0 with a fresh
        # optimizer and a fresh LR schedule. Deliberately only on a FRESH run -- a requeue must
        # continue from its own checkpoint, not silently rewind to the initialisation.
        #
        # The head only. The normalizer is NOT loaded: its buffers are (num_domains, 120) and the
        # fine-tuning corpus has different domains, so the shapes need not even match. Refitting
        # is also what we want -- the head predicts in normalised units, which is precisely what
        # lets it transfer across datasets whose raw action scales differ ~8x.
        init_path = pathlib.Path(initialization_provenance["init_head_path"])
        weights = torch.load(init_path, map_location=device, weights_only=True)
        missing, unexpected = head.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"head init from {init_path} mismatched: {len(missing)} missing, "
                f"{len(unexpected)} unexpected. Refusing to start on a partly-loaded head."
            )
        logger.info("initialised head from %s (%d tensors); optimizer and step start fresh", init_path, len(weights))
    sampler.set_start_step(global_step)
    init_tracking(cfg, run_dir, resuming=resume_step is not None)

    train_model = PolicyTrainModel(
        backbone,
        head,
        objective=cfg.head.objective,
        drifting_config=cfg.drifting,
        train_backbone=bool(backbone_selection.parameters),
    )
    model = train_model
    if use_ddp:
        model = DistributedDataParallel(
            train_model,
            device_ids=[local_rank],
            find_unused_parameters=cfg.find_unused_parameters,
            gradient_as_bucket_view=True,
        )

    monitor = runtime.PreemptionMonitor(run_dir / "PREEMPT_REQUEST", device)
    records: list[dict[str, float]] = []
    last_record: dict[str, float] | None = None
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
            # Historical frozen checkpoints have no custom multiplier in their one optimizer
            # group; defaulting to 1.0 makes them resume bit-for-bit instead of becoming unloadable.
            group["lr"] = lr * float(group.get("lr_multiplier", 1.0))

        inputs = to_inputs(batch, device)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True).float()
        chunk_mask = batch["chunk_mask"].to(device, non_blocking=True).float()
        # Normalise, then re-apply the mask: z-scoring a dead slot would turn its structural zero
        # into -mean/scale, which is not zero and would be regressed as if it were a real target.
        actions = (
            normalizer.normalize(batch["action_chunk"].to(device, non_blocking=True).float(), domain_id) * chunk_mask
        )

        optimizer.zero_grad(set_to_none=True)
        generator = make_step_generator(device, base_seed=cfg.seed, step=global_step, rank=rank)
        loss, extras = model(
            inputs,
            actions,
            action_mask,
            chunk_mask,
            domain_id,
            generator=generator,
        )
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(f"non-finite policy loss at step {global_step}: {float(loss.detach())}")
        loss.backward()

        missing_head = [name for name, parameter in head.named_parameters() if parameter.requires_grad and parameter.grad is None]
        missing_backbone = [
            name
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing_head or missing_backbone:
            raise RuntimeError(
                f"trainable parameters missing gradients at step {global_step}: "
                f"head={missing_head[:8]}, backbone={missing_backbone[:8]}"
            )

        head_grad_norm = grad_norm(head_parameters, device=device)
        backbone_grad_norm = grad_norm(backbone_selection.parameters, device=device)
        trainable_parameters = (*head_parameters, *backbone_selection.parameters)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg.clip_grad_norm)
        norms = torch.stack([head_grad_norm, backbone_grad_norm, total_grad_norm])
        if not bool(torch.isfinite(norms).all()):
            raise FloatingPointError(
                f"non-finite gradient norm at step {global_step}: "
                f"head={float(head_grad_norm)}, backbone={float(backbone_grad_norm)}, total={float(total_grad_norm)}"
            )
        optimizer.step()

        last_record = {
            "loss": float(loss.detach()),
            "learning_rate": lr,
            "grad_norm": float(total_grad_norm),
            "head_grad_norm": float(head_grad_norm),
            "backbone_grad_norm": float(backbone_grad_norm),
            **extras,
        }
        records.append(last_record)
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

        if global_step % cfg.save_interval == 0:
            peak_gpu_memory_by_rank = gather_peak_gpu_memory(device)
            if runtime.is_main_process():
                runtime.save_checkpoint(
                    cfg.checkpoint_dir,
                    global_step,
                    student=head,
                    teacher=None,
                    optimizer=optimizer,
                    config_json=cfg.to_json(),
                    loss_fn=normalizer,
                    extra_modules={"backbone": backbone},
                    keep_last=cfg.keep_last,
                    keep_period=cfg.keep_period,
                    extra=policy_checkpoint_metadata(
                        cfg=cfg,
                        source_backbone_step=backbone_step,
                        backbone_selection=backbone_selection,
                        optimizer=optimizer,
                        head_parameters=head_parameters,
                        world_size=world_size,
                        last_record=last_record,
                        initialization_provenance=initialization_provenance,
                        peak_gpu_memory_by_rank=peak_gpu_memory_by_rank,
                    ),
                )

    peak_gpu_memory_by_rank = gather_peak_gpu_memory(device)
    if runtime.is_main_process():
        runtime.save_checkpoint(
            cfg.checkpoint_dir,
            global_step,
            student=head,
            teacher=None,
            optimizer=optimizer,
            config_json=cfg.to_json(),
            loss_fn=normalizer,
            extra_modules={"backbone": backbone},
            keep_last=cfg.keep_last,
            keep_period=cfg.keep_period,
            extra=policy_checkpoint_metadata(
                cfg=cfg,
                source_backbone_step=backbone_step,
                backbone_selection=backbone_selection,
                optimizer=optimizer,
                head_parameters=head_parameters,
                world_size=world_size,
                last_record=last_record,
                initialization_provenance=initialization_provenance,
                peak_gpu_memory_by_rank=peak_gpu_memory_by_rank,
            ),
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
