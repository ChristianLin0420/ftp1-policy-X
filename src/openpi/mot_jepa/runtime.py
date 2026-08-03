"""Distributed setup, preemption handling, and checkpointing.

All three exist because a long run here is a *chain* of 4-hour jobs, not one process.
"""

from __future__ import annotations

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
    torch.save(teacher.state_dict(), staging / "teacher_ema.pt")
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
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
) -> int:
    """Restore student, teacher shadow, optimizer and ``global_step``.

    Nothing scheduler-shaped is restored because nothing scheduler-shaped is stored: both the
    learning rate and the EMA decay are pure functions of ``global_step``.
    """
    path = pathlib.Path(checkpoint_dir) / str(step)
    module = student.module if hasattr(student, "module") else student
    module.load_state_dict(torch.load(path / "student.pt", map_location=device))
    teacher.load_state_dict(torch.load(path / "teacher_ema.pt", map_location=device))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=device))
    metadata = torch.load(path / "metadata.pt", map_location="cpu")
    return int(metadata["global_step"])


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
