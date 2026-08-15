"""MoT-JEPA pretraining entrypoint.

Designed for a chain of 4-hour SLURM jobs rather than one long process, which drives three
structural choices:

* The loop is ``while global_step < N`` over an **infinite** sampler, not
  ``for batch in loader``. That designs out the unequal-batches-per-rank hang that
  ``DomainBatchSampler.__len__`` (``data_loader.py:715-719``) admits, and it is required once
  the synchrony loss all-gathers, since every rank must enter that collective every step.
* The sampler is a pure function of ``global_step``, so resuming reproduces the *same* data
  order rather than restarting an epoch.
* Both the learning rate and the EMA decay are stateless closures over ``global_step``, so a
  requeue restores them by restoring the step counter alone.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

# jax[cuda12] preallocates GPU memory on import; openpi pulls it in transitively.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# The wandb shim defaults to SwanLab and silently drops `id`/`resume`, which would turn a
# 40-job requeue chain into 40 disjoint runs (`shared/wandb_compat.py:22,137-145`).
os.environ.setdefault("USE_SWANLAB", "false")

from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa import viz
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import load_domain_config
from openpi.mot_jepa.ema import EmaTeacher
from openpi.mot_jepa.losses import MotJepaLoss
from openpi.mot_jepa.losses import normalize_targets
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import assert_mask_invariants
from openpi.mot_jepa.masking import build_batch_masks
from openpi.mot_jepa.model import ClipInputs
from openpi.mot_jepa.model import MotJepaStudent
from openpi.mot_jepa.probes import ProbeSuite
from openpi.shared.wandb_compat import BACKEND_NAME
from openpi.shared.wandb_compat import wandb

logger = logging.getLogger("mot_jepa")


class InfiniteBatchSampler(torch.utils.data.Sampler):
    """Yields index lists forever, as a pure function of the optimizer step.

    Resume sets ``start_step`` and the stream continues exactly where it left off, so a
    40-job chain sees the same sample order an uninterrupted run would.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        domain_of_sample: np.ndarray | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.start_step = 0
        self.global_batch = batch_size * world_size
        if self.global_batch > num_samples:
            raise ValueError(f"global batch {self.global_batch} exceeds dataset size {num_samples}")

        # Domain-pure batches. Without this, Level-A InfoNCE is solvable by DATASET IDENTITY:
        # a GelSight clip and a uSkin clip differ so obviously in appearance that matching
        # video to touch needs no temporal correspondence at all. Measured on a mixed-domain
        # run, the positional-shortcut control reached the SAME top-1 as real tactile
        # (retrieval_gap = 0.0 at step 6000, ratio-to-chance 14.0) -- the shuffle-equals-real
        # failure this whole design exists to prevent, reproduced in our own model.
        #
        # Keeping every batch single-domain makes sensor, embodiment, appearance and lighting
        # constant across the positive and all negatives, so they carry zero discriminative
        # signal and only *when things happen* can solve it.
        self.domain_pools: list[np.ndarray] | None = None
        if domain_of_sample is not None:
            pools = [np.flatnonzero(domain_of_sample == d) for d in np.unique(domain_of_sample)]
            # A domain smaller than one global batch cannot fill a pure batch; drop it rather
            # than silently padding with another domain's clips.
            self.domain_pools = [p for p in pools if p.size >= self.global_batch]
            dropped = len(pools) - len(self.domain_pools)
            if dropped:
                logger.warning("%d domain(s) smaller than one global batch were dropped", dropped)
            if not self.domain_pools:
                raise ValueError(
                    f"no domain has >= {self.global_batch} clips; lower local_batch_size or "
                    "disable domain-pure batching"
                )

    def set_start_step(self, step: int) -> None:
        self.start_step = int(step)

    def _epoch_permutation(self, epoch: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(self.seed * 1_000_003 + epoch))
        return rng.permutation(self.num_samples)

    def indices_for_step(self, step: int) -> list[int]:
        if self.domain_pools is None:
            batches_per_epoch = self.num_samples // self.global_batch
            epoch, offset = divmod(step, batches_per_epoch)
            permutation = self._epoch_permutation(epoch)
            begin = offset * self.global_batch + self.rank * self.batch_size
            return permutation[begin : begin + self.batch_size].tolist()

        # Domain choice is a pure function of the step, so every rank picks the SAME domain --
        # ranks disagreeing here would put different datasets in one all-gathered InfoNCE
        # batch and quietly reintroduce the very shortcut this removes.
        pool = self.domain_pools[self._domain_for_step(step)]
        batches_per_epoch = max(pool.size // self.global_batch, 1)
        epoch, offset = divmod(step, batches_per_epoch)
        rng = np.random.Generator(np.random.PCG64(self.seed * 7_919 + step - offset))
        permutation = pool[rng.permutation(pool.size)]
        begin = offset * self.global_batch + self.rank * self.batch_size
        return permutation[begin : begin + self.batch_size].tolist()

    def _domain_for_step(self, step: int) -> int:
        """Sample a domain in proportion to its size, identically on every rank."""
        sizes = np.array([p.size for p in self.domain_pools], dtype=np.float64)
        rng = np.random.Generator(np.random.PCG64(self.seed * 104_729 + step))
        return int(rng.choice(len(self.domain_pools), p=sizes / sizes.sum()))

    def __iter__(self):
        step = self.start_step
        while True:
            yield self.indices_for_step(step)
            step += 1


def lr_at(step: int, *, peak: float, end: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay. Stateless, so resume needs no scheduler state."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * progress))


def build_dataset(cfg: config_module.MotJepaTrainConfig) -> MotJepaClipDataset:
    if cfg.data.domain_config:
        pairs = load_domain_config(cfg.data.domain_config)
        names = sorted({name for name, _ in pairs})
        stores = [path for _, path in pairs]
        domain_ids = [names.index(name) for name, _ in pairs]
    elif cfg.data.store_glob:
        # Domain = the parent directory (<clips>/<domain>/<store>.zarr). Assigning every
        # store id 0 here would silently make domain-pure batching a no-op, which is exactly
        # the shortcut it exists to remove.
        stores = sorted(glob.glob(cfg.data.store_glob))
        names = sorted({pathlib.Path(p).parent.name for p in stores})
        domain_ids = [names.index(pathlib.Path(p).parent.name) for p in stores]
        logger.info("discovered %d domains from %d stores: %s", len(names), len(stores), names)
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
    )


def to_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> ClipInputs:
    """Move to device and convert uint8 images to [-1, 1] there, not in the worker."""

    def image(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)

    return ClipInputs(
        video=image(batch["video"]),
        gel=image(batch["gel"]),
        lowdim=batch["lowdim"].to(device, non_blocking=True).float(),
    )


def gather_targets(teacher_tokens: list[torch.Tensor], masks, layout) -> list[torch.Tensor]:
    """Slice the teacher's full-clip output down to the target positions, then LayerNorm."""
    targets = []
    for expert, tokens in enumerate(teacher_tokens):
        offset = 0 if expert == 0 else layout.num_video_tokens
        lo, hi = (int(v) for v in masks.tgt_expert_bounds[expert])
        index = (masks.tgt_index[:, lo:hi] - offset).to(tokens.device)
        gathered = tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1]))
        targets.append(normalize_targets(gathered))
    return targets


def reduce_metrics(records: list[dict[str, float]], device: torch.device) -> dict[str, float]:
    """Average per-step metrics across ranks as one stacked tensor.

    The key list is the sorted **union**, not a hardcoded set: the FTP-1 trainer copies
    ``loss_extras`` into ``info`` (``zarr_train_ftp1_pytorch.py:836-838``) and then builds
    its log payload from five fixed keys (``:875-893``), so every auxiliary term it computes
    is silently discarded. Sorting keeps the ordering identical on every rank, which is what
    makes the single all-reduce safe.
    """
    if not records:
        return {}
    keys = sorted(set().union(*(record.keys() for record in records)))
    means = torch.tensor(
        [float(np.mean([record.get(key, 0.0) for record in records])) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    if runtime.get_world_size() > 1:
        torch.distributed.all_reduce(means)
        means /= runtime.get_world_size()
    return dict(zip(keys, means.tolist(), strict=True))


def init_tracking(cfg: config_module.MotJepaTrainConfig, run_dir: pathlib.Path, *, resuming: bool) -> None:
    """Fixed W&B run id so a requeue chain appends to one run instead of forking."""
    if not cfg.wandb_enabled or not runtime.is_main_process():
        return
    if BACKEND_NAME != "wandb":
        raise RuntimeError(
            f"tracking backend is {BACKEND_NAME!r}, not 'wandb'. The SwanLab shim drops the "
            "run id and resume flag, so a requeue chain would fork a new run every job. "
            "Export USE_SWANLAB=false."
        )
    id_path = run_dir / "wandb_id.txt"
    if resuming and id_path.exists():
        wandb.init(id=id_path.read_text().strip(), resume="must", project=cfg.project_name)
    else:
        wandb.init(name=cfg.exp_name, project=cfg.project_name, config=json.loads(cfg.to_json()))
        # Written before the first checkpoint: a crash three minutes in must still rejoin.
        id_path.write_text(str(wandb.run.id))


def train(cfg: config_module.MotJepaTrainConfig) -> None:
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
    cfg = config_module.MotJepaTrainConfig.from_json(frozen_json)

    layout = cfg.layout
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    resume_step = runtime.find_latest_step(cfg.checkpoint_dir)
    init_tracking(cfg, run_dir, resuming=resume_step is not None)

    dataset = build_dataset(cfg)
    sampler = InfiniteBatchSampler(len(dataset), cfg.local_batch_size, rank=rank, world_size=world_size, seed=cfg.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_clips,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
    )

    student = MotJepaStudent(
        layout,
        cfg.encoder,
        cfg.predictor,
        lowdim_channels=cfg.data.lowdim_channels,
        lowdim_log_compress=cfg.data.lowdim_log_compress,
    ).to(device)
    student.set_gradient_checkpointing(enabled=cfg.gradient_checkpointing)
    # Teacher is built from the *unwrapped* backbone, never DDP-wrapped: it has no gradients
    # and registering it would make the reducer wait for buckets that never fire.
    teacher = EmaTeacher(
        student.backbone,
        decay_start=cfg.ema.decay_start,
        decay_end=cfg.ema.decay_end,
        # Resolved, not raw: `None` means "ramp across the whole run". See
        # MotJepaTrainConfig.ema_warmup_steps for the three runs that established why.
        warmup_steps=cfg.ema_warmup_steps,
        device=device,
        # bf16 runtime weights only pay off under CUDA autocast; on CPU there is no autocast
        # to promote them, so bf16 parameters would meet fp32 activations. The fp32 *shadow*
        # is unaffected either way -- that is what must never be bf16.
        runtime_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    loss_fn = MotJepaLoss(cfg.loss, layout).to(device)

    parameters = list(student.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.lr_peak, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
    )

    global_step = 0
    if resume_step is not None:
        global_step = runtime.load_checkpoint(
            cfg.checkpoint_dir,
            resume_step,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
        )
        logger.info("resumed from step %d", global_step)
    sampler.set_start_step(global_step)

    model = student
    if use_ddp:
        # T_HARD removes tactile entirely by design, so the tactile encoder gets no gradient
        # on ~10% of steps. That is the sole reason this flag is on.
        model = DistributedDataParallel(
            student,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=cfg.find_unused_parameters,
            gradient_as_bucket_view=True,
        )

    mask_spec = MaskSpec(
        layout=layout,
        mode_probs=cfg.masking.mode_probs,
        video_mask_frac=cfg.masking.video_mask_frac,
        x_video_mask_frac=cfg.masking.x_video_mask_frac,
        tactile_window_steps=cfg.masking.tactile_window_steps,
        video_window_steps=cfg.masking.video_window_steps,
        min_targets_per_stream=cfg.masking.min_targets_per_stream,
        forecast_horizon_steps=cfg.masking.forecast_horizon_steps,
    )
    probes = ProbeSuite(layout, max_batches=cfg.probe_batches)
    monitor = runtime.PreemptionMonitor(run_dir / "PREEMPT_REQUEST", device)

    records: list[dict[str, float]] = []
    # Last probe result, carried so the step log can print a fixed-reference signal. Empty
    # until the first probe, which prints as nan rather than a misleading zero.
    last_probe: dict[str, float] = {}
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
        cpu_masks = build_batch_masks(
            mask_spec,
            step=global_step,
            batch_size=cfg.local_batch_size,
            base_seed=cfg.seed,
            rank=rank,
            world_size=world_size,
        )
        if global_step == 0:
            # Checked in the run that depends on it, not only in the test suite. The whole premise
            # of MaskMode.F is that its context is strictly earlier than its targets, and that is
            # enforced only by which indices reach `ctx_index` -- if it silently broke, F would
            # quietly become another interpolation mode and the entire pretrain would be void.
            # One call over one batch; free relative to a 100k-step run.
            for probe_step in range(len(MaskMode) * 8):
                assert_mask_invariants(
                    build_batch_masks(
                        mask_spec,
                        step=probe_step,
                        batch_size=min(2, cfg.local_batch_size),
                        base_seed=cfg.seed,
                        rank=rank,
                        world_size=world_size,
                    ),
                    mask_spec,
                )
            logger.info("mask invariants hold for every mode, including F's causal split")
        masks = cpu_masks.to(device)

        lr = lr_at(global_step, peak=cfg.lr_peak, end=cfg.lr_end, warmup=cfg.lr_warmup_steps, total=cfg.num_train_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            teacher_out = teacher.module.encode_full(inputs)
        targets = gather_targets(teacher_out.tokens, masks, layout)

        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            out = model(inputs, masks)
            loss, extras = loss_fn(out.predictions, targets, out.sync_readout, masks, global_step)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {global_step}: {loss.item()}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # The teacher shadows the backbone only -- the predictor is discarded after
        # pretraining and must not appear in the target network.
        decay = teacher.update(student.backbone, global_step)

        drift_abs, drift_rel = (0.0, 0.0)
        if global_step % cfg.log_interval == 0:
            drift_abs, drift_rel = teacher.drift_norm(student.backbone)
        records.append(
            {
                "loss": float(loss.detach()),
                "learning_rate": lr,
                "grad_norm": float(grad_norm),
                "ema_decay": decay,
                "ema_drift_l2": drift_abs,
                "ema_drift_rel": drift_rel,
                f"mode/{masks.mode_enum.name}": 1.0,
                **{key: float(value) for key, value in extras.items()},
            }
        )

        global_step += 1

        if global_step % cfg.log_interval == 0:
            metrics = reduce_metrics(records, device)
            elapsed = time.time() - window_start
            metrics["clips_per_second"] = cfg.log_interval * cfg.local_batch_size * world_size / max(elapsed, 1e-6)
            if runtime.is_main_process():
                # `loss` RISES for the first ~10k steps of a healthy run and that is not a fault.
                # The JEPA target is an EMA of the student, so the loss measures the distance to a
                # LAGGING COPY OF ITSELF, and that distance grows mechanically as lr and the
                # teacher's lag grow. Measured across three runs (probe2, probe3, forecast100k) the
                # minimum lands at step 1050 every time and is never revisited -- probe2's global
                # minimum over all 50k steps is at step 1050, and probe2 is the backbone that went
                # on to produce a working policy.
                #
                # `rank` is the fixed-reference signal to read instead: RankMe over the video
                # stream, carried from the last probe. It rises monotonically while the loss
                # doubles. Printing it on every log line means nobody has to infer training health
                # from a quantity that cannot show it.
                logger.info(
                    "step %d loss %.4f grad %.3f lr %.2e drift %.3e rank %.0f/%.0f "
                    "disp %.2f logit %.3g %.1f clips/s",
                    global_step,
                    metrics.get("loss", float("nan")),
                    metrics.get("grad_norm", float("nan")),
                    lr,
                    metrics.get("ema_drift_rel", 0.0),
                    last_probe.get("rankme_video", float("nan")),
                    last_probe.get("rankme_tactile", float("nan")),
                    # The two quantities that caught the last two failures, carried from the most
                    # recent probe. `disp` is student/teacher dispersion -- it falls when the
                    # student drifts off a teacher that stopped tracking. `logit` is the largest
                    # attention logit -- it grows for thousands of steps before anything else
                    # reacts, and neither is visible in the loss, which FELL during both failures.
                    last_probe.get("student_dispersion", float("nan"))
                    / max(last_probe.get("teacher_dispersion", float("nan")), 1e-9),
                    last_probe.get("attn_logit_max", float("nan")),
                    metrics["clips_per_second"],
                )
                if cfg.wandb_enabled:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)
            records.clear()
            window_start = time.time()

        if global_step % cfg.probe_interval == 0:
            want_panels = cfg.wandb_enabled and runtime.is_main_process()
            # A probe is a measurement, not a training step, and must never be able to end a
            # run. A 50k-step chain represents most of a day; losing it to a bug in an
            # instrument that only reports numbers would be absurd. The panel rendering below
            # has always been wrapped for this reason -- the probe itself was not, which left
            # every requeue one probe bug away from ending the run.
            #
            # Note the batch is consumed as the first statement inside ``ProbeSuite.run``, so a
            # later failure has already advanced every rank's loader identically and cannot
            # desync the data stream.
            probe_metrics: dict = {}
            try:
                probe_metrics = probes.run(
                    student,
                    teacher,
                    loader_iter,
                    device,
                    layout=layout,
                    projectors=loss_fn.projectors,
                    masks=masks,
                    collect_panels=want_panels,
                )
            except Exception:
                logger.exception("probes failed at step %d; continuing", global_step)

            if probe_metrics:
                last_probe = dict(probe_metrics)
            if probe_metrics and runtime.is_main_process():
                logger.info("probes @%d: %s", global_step, probe_metrics)
                if cfg.wandb_enabled:
                    wandb.log({f"probe/{k}": v for k, v in probe_metrics.items()}, step=global_step)
                    try:
                        panels = viz.training_panels(
                            probes.panels, layout, masks.tgt_index[0].cpu(), masks.mode_enum.name
                        )
                        if panels:
                            wandb.log({f"panel/{k}": v for k, v in panels.items()}, step=global_step)
                    except Exception:  # a plot must never kill a training run
                        logger.exception("panel rendering failed at step %d; continuing", global_step)

        if global_step % cfg.save_interval == 0 and runtime.is_main_process():
            runtime.save_checkpoint(
                cfg.checkpoint_dir,
                global_step,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                config_json=frozen_json,
                loss_fn=loss_fn,
                keep_last=cfg.keep_last,
                keep_period=cfg.keep_period,
            )

    if runtime.is_main_process():
        runtime.save_checkpoint(
            cfg.checkpoint_dir,
            global_step,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            config_json=frozen_json,
            loss_fn=loss_fn,
            keep_last=cfg.keep_last,
            keep_period=cfg.keep_period,
        )
        if not preempted:
            # Stops the requeue chain; the batch script checks for this before resubmitting.
            (run_dir / "DONE").write_text(str(global_step))

    runtime.barrier()
    if runtime.is_main_process() and cfg.wandb_enabled:
        wandb.finish()
    runtime.cleanup_ddp()
    # Exit 0 either way: the batch script decides whether to requeue, not the exit code, so
    # a genuine crash (non-zero) cannot be mistaken for a preemption.
    sys.exit(0)


def main() -> None:
    train(config_module.cli())


if __name__ == "__main__":
    main()
