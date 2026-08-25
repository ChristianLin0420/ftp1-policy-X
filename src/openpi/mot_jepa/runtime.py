"""Distributed setup, preemption handling, and checkpointing.

All three exist because a long run here is a *chain* of 4-hour jobs, not one process.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import time
import types

import torch
from torch import nn
import torch.distributed as dist

from openpi.mot_jepa.config import DataConfig
from openpi.mot_jepa.config import _from_dict
from openpi.mot_jepa.model import MotJepaStudent
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.predictor import MoTPredictorConfig

logger = logging.getLogger(__name__)

_STEP_DIR_RE = re.compile(r"^\d+$")


# --------------------------------------------------------------------------------------
# Distributed
# --------------------------------------------------------------------------------------


def setup_ddp() -> tuple[bool, int, torch.device]:
    """Initialize NCCL from the environment torchrun provides.

    Deliberately *not* imported from ``scripts/train_pytorch.py``. That module does
    ``import jax`` at file scope (``:33``), and jax[cuda12] preallocates GPU memory on
    import; pulling it into 32 torch-only ranks costs real memory for nothing.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        # Set the device and pass device_id BEFORE init_process_group. Initializing first and
        # binding after makes NCCL guess the rank-to-GPU mapping, which it warns about
        # explicitly ("device used by this process is currently unknown ... can potentially
        # cause a hang"). On one node it usually guesses right; across nodes a wrong guess is
        # a deadlock that only surfaces as a 30-minute watchdog timeout.
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=30),
            device_id=device,
        )
        return True, local_rank, device
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return False, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# --------------------------------------------------------------------------------------
# Preemption
# --------------------------------------------------------------------------------------


class PreemptionMonitor:
    """Decides, collectively, when to checkpoint and exit.

    Two independent triggers feed one decision:

    1. A signal handler whose entire body sets a boolean. Nothing else is async-signal-safe.
    2. Rank 0 stats a flag file that the batch script touches on ``SIGUSR1``.

    The file is the *primary* path. ``--signal=B:USR1@600`` delivers only to the batch shell,
    and ``scancel --signal`` reaches the job step -- neither reliably reaches torchrun's
    worker children, since torchrun has historically not forwarded SIGUSR1.

    The result is then **broadcast from rank 0**, so every rank makes the identical decision.
    A per-rank ``os.path.exists`` race would let ranks disagree and hang on the next
    collective, which is a far worse failure than a missed checkpoint.
    """

    def __init__(
        self,
        flag_path: pathlib.Path,
        device: torch.device,
        *,
        check_interval: int = 10,
        signals: tuple[int, ...] = (signal.SIGUSR1, signal.SIGTERM),
    ) -> None:
        self.flag_path = pathlib.Path(flag_path)
        self.device = device
        self.check_interval = check_interval
        self._signalled = False
        self._decided = False
        for signum in signals:
            try:
                signal.signal(signum, self._handle)
            except ValueError:  # pragma: no cover - not the main thread
                logger.warning("could not install handler for signal %s", signum)

    def _handle(self, signum: int, frame: types.FrameType | None) -> None:
        self._signalled = True

    def should_stop(self, step: int) -> bool:
        if self._decided:
            return True
        local = self._signalled
        if is_main_process() and step % self.check_interval == 0 and self.flag_path.exists():
            local = True

        if get_world_size() > 1:
            flag = torch.tensor([1 if local else 0], dtype=torch.uint8, device=self.device)
            dist.broadcast(flag, src=0)
            local = bool(flag.item())

        self._decided = local
        return local


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


def find_latest_step(checkpoint_dir: pathlib.Path) -> int | None:
    """Latest completed step, or ``None`` on a fresh run.

    Returns ``None`` rather than raising -- unlike ``get_ftp1_latest_checkpoint_step``
    (``zarr_train_ftp1_utils.py:1483-1512``), whose raise-on-empty behaviour left its own
    caller's ``None`` check dead code. Idempotent auto-resume needs a real ``None``, because
    the very first job in a chain has no checkpoint yet.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    pointer = checkpoint_dir / "latest"
    if pointer.exists():
        try:
            step = int(pointer.read_text().strip())
        except ValueError:
            step = -1
        if step >= 0 and (checkpoint_dir / str(step)).is_dir():
            return step

    steps = [int(p.name) for p in checkpoint_dir.iterdir() if p.is_dir() and _STEP_DIR_RE.match(p.name)]
    return max(steps) if steps else None


def prune_checkpoints(checkpoint_dir: pathlib.Path, *, keep_last: int, keep_period: int | None) -> list[int]:
    """Delete old checkpoints, keeping the newest ``keep_last`` plus every ``keep_period``.

    Runs only *after* a new checkpoint has been committed, so there is never a window with
    zero checkpoints on disk.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    steps = sorted(int(p.name) for p in checkpoint_dir.iterdir() if p.is_dir() and _STEP_DIR_RE.match(p.name))
    keep = set(steps[-keep_last:]) if keep_last > 0 else set()
    if keep_period:
        keep |= {step for step in steps if step % keep_period == 0}

    removed = []
    for step in steps:
        if step in keep:
            continue
        shutil.rmtree(checkpoint_dir / str(step), ignore_errors=True)
        removed.append(step)
    return removed


def save_checkpoint(
    checkpoint_dir: pathlib.Path,
    step: int,
    *,
    student: nn.Module,
    teacher,  # EmaTeacher; untyped to avoid a circular import
    optimizer: torch.optim.Optimizer,
    config_json: str,
    loss_fn: nn.Module | None = None,
    extra_modules: dict[str, nn.Module] | None = None,
    keep_last: int = 3,
    keep_period: int | None = None,
    extra: dict | None = None,
) -> pathlib.Path:
    """Write atomically: staging directory, then rename, then advance the pointer.

    ``os.replace`` is atomic within a filesystem, so a job killed mid-write leaves the
    previous checkpoint intact rather than a half-written directory that resume would load.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    staging = checkpoint_dir / f"tmp_{step}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    module = student.module if hasattr(student, "module") else student
    torch.save(module.state_dict(), staging / "student.pt")
    for name, extra_module in _checkpoint_modules(extra_modules).items():
        module_to_save = extra_module.module if hasattr(extra_module, "module") else extra_module
        torch.save(module_to_save.state_dict(), staging / f"{name}.pt")
    # Post-training freezes the backbone, so there is no EMA to shadow and no teacher to save.
    if teacher is not None:
        torch.save(teacher.state_dict(), staging / "teacher_ema.pt")
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    # The loss module owns trainable state that lives OUTSIDE the student: the two synchrony
    # projectors and the running ``lowdim_scale``. They are in the optimizer's parameter list
    # (mot_jepa_train.py:306), so omitting them here does not merely lose them -- the restored
    # optimizer reapplies the OLD Adam moments to freshly random projectors, which is worse than
    # a clean restart. Observed on probe3's requeue at step 37970: retrieval 0.586 -> 0.176 and
    # loss 0.66 -> 1.04 in one interval.
    if loss_fn is not None:
        torch.save(loss_fn.state_dict(), staging / "loss.pt")
    torch.save({"global_step": step, **(extra or {})}, staging / "metadata.pt")
    (staging / "train_config.json").write_text(config_json)

    final = checkpoint_dir / str(step)
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)

    pointer = checkpoint_dir / "latest"
    pointer.write_text(str(step))
    _fsync_dir(checkpoint_dir)

    prune_checkpoints(checkpoint_dir, keep_last=keep_last, keep_period=keep_period)
    return final


def _fsync_dir(path: pathlib.Path) -> None:
    """Force the rename and pointer to reach stable storage before we claim success."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - unusual filesystems
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - Lustre sometimes refuses directory fsync
        pass
    finally:
        os.close(fd)


def load_checkpoint(
    checkpoint_dir: pathlib.Path,
    step: int,
    *,
    student: nn.Module,
    teacher,  # EmaTeacher
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    extra_modules: dict[str, nn.Module] | None = None,
) -> int:
    """Restore student, teacher shadow, optimizer, loss state and ``global_step``.

    Nothing scheduler-shaped is restored because nothing scheduler-shaped is stored: both the
    learning rate and the EMA decay are pure functions of ``global_step``.

    ``loss.pt`` is absent from checkpoints written before it was saved at all, so a missing file
    is tolerated -- those runs resume with the behaviour they already had rather than failing to
    start. It is logged, because silently keeping a random projector is the failure this exists
    to remove.
    """
    path = pathlib.Path(checkpoint_dir) / str(step)
    module = student.module if hasattr(student, "module") else student
    module.load_state_dict(torch.load(path / "student.pt", map_location=device))
    for name, extra_module in _checkpoint_modules(extra_modules).items():
        extra_path = path / f"{name}.pt"
        if not extra_path.exists():
            raise FileNotFoundError(f"{extra_path} missing; cannot resume module {name!r}")
        module_to_load = extra_module.module if hasattr(extra_module, "module") else extra_module
        module_to_load.load_state_dict(torch.load(extra_path, map_location=device, weights_only=True), strict=True)
    if teacher is not None:
        teacher.load_state_dict(torch.load(path / "teacher_ema.pt", map_location=device))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=device))
    if loss_fn is not None:
        loss_path = path / "loss.pt"
        if loss_path.exists():
            loss_fn.load_state_dict(torch.load(loss_path, map_location=device))
        else:
            logger.warning(
                "%s has no loss.pt: synchrony projectors resume from random init and the "
                "restored optimizer moments no longer match them. Expect a retrieval dip.",
                path,
            )
    metadata = torch.load(path / "metadata.pt", map_location="cpu")
    return int(metadata["global_step"])


def _checkpoint_modules(extra_modules: dict[str, nn.Module] | None) -> dict[str, nn.Module]:
    """Validate optional named module artifacts before they become checkpoint filenames."""
    modules = extra_modules or {}
    reserved = {"student", "teacher_ema", "optimizer", "loss", "metadata"}
    for name, module in modules.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in reserved:
            raise ValueError(f"invalid or reserved checkpoint module name {name!r}")
        if not isinstance(module, nn.Module):
            raise TypeError(f"checkpoint module {name!r} is {type(module).__name__}, expected nn.Module")
    return modules


def architecture_from_run(run: pathlib.Path, cfg, step: int | None = None):
    """Rebuild ``cfg``'s encoder/predictor shape from the config stored with the checkpoint.

    A checkpoint can only be loaded into the architecture it was trained with, and the live
    dataclass defaults drift away from old runs over time. ``qk_norm`` is the case that forced
    this: it defaults to ``True`` now, but every config written before it existed describes a
    model trained WITHOUT it, and ``_from_dict`` skips absent keys -- so an old run would be
    rebuilt with q/k LayerNorms its checkpoint cannot fill. Absent means ``False`` here, which is
    what those checkpoints actually are.

    Two locations are tried, because a *run* directory and an extracted *backbone snapshot* are
    laid out differently. Snapshots under ``ftp1-runs/backbones/`` carry only
    ``checkpoints/<step>/train_config.json`` -- ``save_checkpoint`` writes it there -- and no
    top-level ``run_config.json``. Reading only the latter made all three released backbones fail
    with "254 shadow tensors for 350 parameters" the moment qk_norm defaulted on, which would
    have blocked every downstream evaluation.

    Returns ``cfg`` unchanged when neither file exists.
    """
    run = pathlib.Path(run)
    candidates = [run / "run_config.json"]
    resolved = step if step is not None else find_latest_step(run / "checkpoints")
    if resolved is not None:
        candidates.append(run / "checkpoints" / str(resolved) / "train_config.json")
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return cfg
    payload = json.loads(path.read_text())
    encoder, predictor = dict(payload.get("encoder", {})), dict(payload.get("predictor", {}))
    if not encoder and not predictor:
        return cfg
    encoder.setdefault("qk_norm", False)
    predictor.setdefault("qk_norm", False)
    # Same rule for the lowdim input transform: absent means the run predates it.
    data = dict(payload.get("data", {}))
    data.setdefault("lowdim_log_compress", False)
    return dataclasses.replace(
        cfg,
        encoder=_from_dict(MoTEncoderConfig, encoder),
        predictor=_from_dict(MoTPredictorConfig, predictor),
        data=_from_dict(DataConfig, data) if data else cfg.data,
    )


def load_frozen_backbone(run: pathlib.Path, step: int | None, cfg, device: torch.device):
    """Load a pretrained EMA teacher as a frozen backbone. Returns ``(backbone, step)``.

    The **teacher** shadow is loaded, not the student: the EMA is what pretraining selects for
    and what every probe number was measured on.

    ``EmaTeacher.state_dict`` serialises the fp32 shadow as a **positional** list keyed
    ``shadow.0 ... shadow.N-1``, ordered by ``backbone.parameters()`` -- not as named parameters.
    Loading it with ``load_state_dict(..., strict=False)`` therefore matches nothing at all and
    leaves the backbone at its random init, with the run looking entirely healthy. That happened
    once and cost a day, hence the positional copy and the hard count check below.
    """
    checkpoint_dir = pathlib.Path(run) / "checkpoints"
    step = step or find_latest_step(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint under {checkpoint_dir}")
    shadow = torch.load(checkpoint_dir / str(step) / "teacher_ema.pt", map_location="cpu", weights_only=True)
    # Build the architecture the checkpoint was TRAINED with, not today's defaults.
    cfg = architecture_from_run(run, cfg, step)
    student = MotJepaStudent(
        cfg.layout,
        cfg.encoder,
        cfg.predictor,
        lowdim_channels=cfg.data.lowdim_channels,
        lowdim_log_compress=cfg.data.lowdim_log_compress,
    )
    params = list(student.backbone.parameters())
    if len(shadow) != len(params):
        raise RuntimeError(f"{len(shadow)} shadow tensors for {len(params)} parameters; config mismatch")
    with torch.no_grad():
        for index, param in enumerate(params):
            param.copy_(shadow[f"shadow.{index}"].to(param.dtype))
    backbone = student.to(device).eval().backbone
    backbone.requires_grad_(requires_grad=False)
    logger.info("loaded frozen backbone from %s step %d (%d params)", checkpoint_dir, step, len(params))
    return backbone, step


@dataclasses.dataclass(frozen=True)
class LoadedPolicyBackbone:
    """A policy's executable backbone plus unambiguous source/adaptation provenance."""

    backbone: nn.Module
    source_step: int
    checkpoint: pathlib.Path | None
    mode: str
    adapted_step: int | None


def load_policy_backbone(
    policy_run: pathlib.Path,
    policy_step: int,
    cfg,
    device: torch.device,
) -> LoadedPolicyBackbone:
    """Load the exact backbone paired with one supervised policy checkpoint.

    Historical policy runs contain only ``student.pt`` because their encoder was frozen. Those
    retain the legacy source-EMA path. New runs also save ``backbone.pt``; adapted modes require it
    so evaluation can never silently fall back to the pretraining snapshot.
    """
    mode = getattr(cfg, "backbone_train_mode", "frozen")
    if mode not in {"frozen", "last_blocks", "full"}:
        raise ValueError(f"unknown saved backbone_train_mode {mode!r}")
    if not cfg.pretrained_run:
        raise ValueError("policy config must set pretrained_run")
    if mode != "frozen" and cfg.pretrained_step is None:
        raise ValueError("adapted policy config must pin pretrained_step")

    backbone, source_step = load_frozen_backbone(
        pathlib.Path(cfg.pretrained_run), cfg.pretrained_step, cfg, device
    )
    if cfg.pretrained_step is not None and source_step != cfg.pretrained_step:
        raise ValueError(f"loaded source backbone step {source_step} != configured {cfg.pretrained_step}")

    checkpoint_dir = pathlib.Path(policy_run) / "checkpoints" / str(policy_step)
    backbone_path = checkpoint_dir / "backbone.pt"
    if backbone_path.exists():
        metadata_path = checkpoint_dir / "metadata.pt"
        if not metadata_path.exists():
            raise FileNotFoundError(f"{metadata_path} missing beside saved policy backbone")
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        stored_policy_step = int(metadata.get("global_step", -1))
        if stored_policy_step != policy_step:
            raise ValueError(f"checkpoint metadata step {stored_policy_step} != requested policy step {policy_step}")
        stored_mode = metadata.get("backbone_train_mode")
        if stored_mode != mode:
            raise ValueError(f"checkpoint backbone mode {stored_mode!r} != run config mode {mode!r}")
        stored_source_run = metadata.get("source_backbone_run")
        configured_source_run = str(pathlib.Path(cfg.pretrained_run).resolve())
        if stored_source_run != configured_source_run:
            raise ValueError(
                f"checkpoint source run {stored_source_run!r} != configured source run {configured_source_run!r}"
            )
        stored_source = int(metadata.get("source_backbone_step", -1))
        if stored_source != source_step:
            raise ValueError(f"checkpoint source step {stored_source} != loaded source step {source_step}")
        state = torch.load(backbone_path, map_location=device, weights_only=True)
        backbone.load_state_dict(state, strict=True)
        selected_path: pathlib.Path | None = backbone_path
    elif mode != "frozen":
        raise FileNotFoundError(
            f"{backbone_path} missing for {mode!r} policy; refuse to evaluate the frozen source backbone"
        )
    else:
        # Compatibility path for every policy checkpoint written before backbone adaptation.
        selected_path = None

    backbone.eval().requires_grad_(requires_grad=False)
    return LoadedPolicyBackbone(
        backbone=backbone,
        source_step=source_step,
        checkpoint=selected_path,
        mode=mode,
        adapted_step=policy_step if mode != "frozen" else None,
    )


def resolve_run_config(
    run_dir: pathlib.Path, config_json: str, *, timeout_s: float = 120.0, poll_s: float = 0.2
) -> tuple[str, dict]:
    """Freeze the config on first launch; re-read it on every later launch.

    Returns the authoritative JSON plus any drift against what was passed in, so the caller
    can warn rather than silently switching hyperparameters mid-chain.

    **Every rank calls this simultaneously, so the content must appear atomically, not just
    the file.** An earlier version claimed ``O_CREAT | O_EXCL`` on the destination and then
    wrote into that descriptor. The *claim* is atomic but the *content* is not: losing ranks
    caught ``FileExistsError`` and immediately read a file that existed but was still empty,
    so ``json.loads("")`` raised. A single node survived on page-cache timing; two nodes on
    Lustre failed every rank but one.

    The fix is two-part. The winner writes a temporary file and ``os.replace``s it, so the
    destination never exists in a partially-written state. Losers poll until it is present
    and parses, because on a shared filesystem another node's rename is not instantly visible.
    """
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_config.json"
    claim = run_dir / "run_config.claim"

    try:
        fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        frozen = _await_config(path, timeout_s=timeout_s, poll_s=poll_s)
    else:
        os.close(fd)
        staging = run_dir / f"run_config.{os.getpid()}.tmp"
        staging.write_text(config_json)
        os.replace(staging, path)  # atomic: readers see either nothing or the whole file
        _fsync_dir(run_dir)
        return config_json, {}

    incoming, existing = json.loads(config_json), json.loads(frozen)
    drift = {key: (existing.get(key), incoming.get(key)) for key in incoming if existing.get(key) != incoming.get(key)}
    return frozen, drift


def _await_config(path: pathlib.Path, *, timeout_s: float, poll_s: float) -> str:
    """Wait for a config written by another rank to become visible and parseable."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            text = path.read_text()
            if text.strip():
                json.loads(text)
                return text
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(poll_s)
    raise TimeoutError(f"{path} did not become readable within {timeout_s}s (last error: {last_error})")
