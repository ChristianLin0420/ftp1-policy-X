"""Bounded, atomic per-episode trajectory artifacts for UniVTAC evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
MAX_TRAJECTORY_ACTIONS = 4096
MAX_RAW_ACTION_DIM = 512
TERMINATION_REASONS = frozenset({"success", "task_early_stop", "action_limit", "sim_step_limit", "error"})
ACTION_REPRESENTATIONS = frozenset({"relative", "absolute", "mix"})


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeTraceRecorder:
    """Collect a fixed-width action trace without retaining images or whole action chunks."""

    def __init__(self, *, seed: int, output_path: str | Path, max_actions: int = 500) -> None:
        if not 1 <= int(max_actions) <= MAX_TRAJECTORY_ACTIONS:
            raise ValueError(f"max_actions must be in [1, {MAX_TRAJECTORY_ACTIONS}], got {max_actions}")
        self.seed = int(seed)
        self.output_path = Path(output_path)
        self.max_actions = int(max_actions)
        self.truncated = False
        self._raw_action_dim: int | None = None
        self._raw_action_rep: str | None = None
        self._rows: list[dict[str, Any]] = []
        self._total_actions_seen = 0

    def append(
        self,
        *,
        action_index: int,
        sim_step_before: int,
        sim_step_after: int,
        qpos8_before: Any,
        qpos8_after: Any,
        sent_action8: Any,
        raw_first_vector: Any,
        raw_first_action8: Any,
        resolved_first_action8: Any,
        first_executable_index: int,
        raw_action_rep: str,
        exec_success: bool,
        eval_success_after: bool,
        early_stop_after: bool,
    ) -> bool:
        """Append one executed action; return false when the configured trace cap was reached."""
        expected_action_index = self._total_actions_seen + 1
        if int(action_index) != expected_action_index:
            raise ValueError(f"action_index must be contiguous and one-based: expected {expected_action_index}")
        if int(sim_step_after) < int(sim_step_before):
            raise ValueError("sim_step_after precedes sim_step_before")
        if len(self._rows) >= self.max_actions:
            self._total_actions_seen += 1
            self.truncated = True
            return False

        raw = np.asarray(raw_first_vector, dtype=np.float32).reshape(-1)
        if raw.size < 1 or raw.size > MAX_RAW_ACTION_DIM:
            raise ValueError(f"raw_first_vector size must be in [1, {MAX_RAW_ACTION_DIM}], got {raw.size}")
        if not np.isfinite(raw).all():
            raise ValueError("raw_first_vector contains non-finite values")
        if self._raw_action_dim is not None and raw.size != self._raw_action_dim:
            raise ValueError(f"raw action dimension changed from {self._raw_action_dim} to {raw.size}")
        if raw_action_rep not in ACTION_REPRESENTATIONS:
            raise ValueError(f"invalid raw_action_rep {raw_action_rep!r}")
        if self._raw_action_rep is not None and raw_action_rep != self._raw_action_rep:
            raise ValueError(f"raw action representation changed from {self._raw_action_rep!r} to {raw_action_rep!r}")

        row = {
            "action_index": int(action_index),
            "sim_step_before": int(sim_step_before),
            "sim_step_after": int(sim_step_after),
            "qpos8_before": _vector(qpos8_before, 8, "qpos8_before"),
            "qpos8_after": _vector(qpos8_after, 8, "qpos8_after"),
            "sent_action8": _vector(sent_action8, 8, "sent_action8"),
            "raw_first_vector": raw.copy(),
            "raw_first_action8": _vector(raw_first_action8, 8, "raw_first_action8"),
            "resolved_first_action8": _vector(resolved_first_action8, 8, "resolved_first_action8"),
            "first_executable_index": int(first_executable_index),
            "exec_success": bool(exec_success),
            "eval_success_after": bool(eval_success_after),
            "early_stop_after": bool(early_stop_after),
        }
        if self._raw_action_dim is None:
            self._raw_action_dim = int(raw.size)
        if self._raw_action_rep is None:
            self._raw_action_rep = raw_action_rep
        self._total_actions_seen += 1
        self._rows.append(row)
        return True

    def write(self, *, termination_reason: str, result: str) -> dict[str, Any]:
        """Atomically write the trace and return metadata suitable for worker metadata JSON."""
        if termination_reason not in TERMINATION_REASONS:
            raise ValueError(f"invalid termination_reason {termination_reason!r}")
        if result not in {"success", "failed", "error"}:
            raise ValueError(f"invalid result {result!r}")
        expected_result = {"success": "success", "error": "error"}.get(termination_reason, "failed")
        if result != expected_result:
            raise ValueError(f"termination_reason {termination_reason!r} requires result {expected_result!r}")

        count = len(self._rows)
        raw_dim = int(self._raw_action_dim or 0)
        raw_action_rep = self._raw_action_rep or ""
        if count > 0 and not self.truncated:
            last = self._rows[-1]
            if termination_reason == "success" and not last["eval_success_after"]:
                raise ValueError("success trace does not end with eval_success_after")
            if termination_reason == "task_early_stop" and not last["early_stop_after"]:
                raise ValueError("task_early_stop trace does not end with early_stop_after")

        def column(name: str, dtype: np.dtype[Any]) -> np.ndarray:
            return np.asarray([row[name] for row in self._rows], dtype=dtype)

        def vectors(name: str, width: int) -> np.ndarray:
            if not self._rows:
                return np.empty((0, width), dtype=np.float32)
            return np.stack([row[name] for row in self._rows]).astype(np.float32, copy=False)

        payload = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "num_actions": np.asarray(count, dtype=np.int32),
            "total_actions_seen": np.asarray(self._total_actions_seen, dtype=np.int32),
            "max_actions": np.asarray(self.max_actions, dtype=np.int32),
            "raw_action_dim": np.asarray(raw_dim, dtype=np.int32),
            "raw_action_rep": np.asarray(raw_action_rep),
            "truncated": np.asarray(self.truncated, dtype=np.bool_),
            "termination_reason": np.asarray(termination_reason),
            "result": np.asarray(result),
            "action_index": column("action_index", np.int32),
            "sim_step_before": column("sim_step_before", np.int32),
            "sim_step_after": column("sim_step_after", np.int32),
            "qpos8_before": vectors("qpos8_before", 8),
            "qpos8_after": vectors("qpos8_after", 8),
            "sent_action8": vectors("sent_action8", 8),
            "raw_first_vector": vectors("raw_first_vector", raw_dim),
            "raw_first_action8": vectors("raw_first_action8", 8),
            "resolved_first_action8": vectors("resolved_first_action8", 8),
            "first_executable_index": column("first_executable_index", np.int16),
            "exec_success": column("exec_success", np.bool_),
            "eval_success_after": column("eval_success_after", np.bool_),
            "early_stop_after": column("early_stop_after", np.bool_),
        }

        path = self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        try:
            with open(temporary, "wb") as stream:
                np.savez_compressed(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

        validate_episode_trace(path, expected_seed=self.seed)
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(path),
            "sha256": _sha256(path),
            "seed": self.seed,
            "recorded_actions": count,
            "total_actions_seen": self._total_actions_seen,
            "raw_action_dim": raw_dim,
            "raw_action_rep": raw_action_rep,
            "max_actions": self.max_actions,
            "truncated": self.truncated,
            "termination_reason": termination_reason,
            "result": result,
        }


def validate_episode_trace(path: str | Path, *, expected_seed: int | None = None) -> dict[str, Any]:
    """Validate the seed, shapes, lengths, and finite numeric fields of one trace."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as trace:
        schema_version = int(trace["schema_version"])
        seed = int(trace["seed"])
        count = int(trace["num_actions"])
        total_actions_seen = int(trace["total_actions_seen"])
        max_actions = int(trace["max_actions"])
        raw_dim = int(trace["raw_action_dim"])
        raw_action_rep = str(trace["raw_action_rep"])
        truncated = bool(trace["truncated"])
        termination_reason = str(trace["termination_reason"])
        result = str(trace["result"])

        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema {schema_version}")
        if expected_seed is not None and seed != int(expected_seed):
            raise ValueError(f"trace seed {seed} does not match expected seed {expected_seed}")
        if not 0 <= count <= max_actions <= MAX_TRAJECTORY_ACTIONS or total_actions_seen < count:
            raise ValueError(f"invalid action counts count={count}, total_seen={total_actions_seen}, max={max_actions}")
        if truncated != (total_actions_seen > count):
            raise ValueError(f"truncated={truncated} disagrees with count={count}, total_seen={total_actions_seen}")
        if not 0 <= raw_dim <= MAX_RAW_ACTION_DIM:
            raise ValueError(f"invalid raw_action_dim {raw_dim}")
        if count > 0 and raw_action_rep not in ACTION_REPRESENTATIONS:
            raise ValueError(f"invalid raw_action_rep {raw_action_rep!r}")
        if count == 0 and raw_action_rep:
            raise ValueError("empty trace unexpectedly defines raw_action_rep")
        if termination_reason not in TERMINATION_REASONS:
            raise ValueError(f"invalid termination_reason {termination_reason!r}")
        if result not in {"success", "failed", "error"}:
            raise ValueError(f"invalid result {result!r}")
        expected_result = {"success": "success", "error": "error"}.get(termination_reason, "failed")
        if result != expected_result:
            raise ValueError(f"termination_reason {termination_reason!r} requires result {expected_result!r}")

        widths = {
            "qpos8_before": 8,
            "qpos8_after": 8,
            "sent_action8": 8,
            "raw_first_vector": raw_dim,
            "raw_first_action8": 8,
            "resolved_first_action8": 8,
        }
        for name, width in widths.items():
            values = trace[name]
            if values.shape != (count, width):
                raise ValueError(f"{name} has shape {values.shape}, expected {(count, width)}")
            if values.dtype != np.float32:
                raise ValueError(f"{name} has dtype {values.dtype}, expected float32")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")

        one_dimensional = {
            "action_index": np.dtype(np.int32),
            "sim_step_before": np.dtype(np.int32),
            "sim_step_after": np.dtype(np.int32),
            "first_executable_index": np.dtype(np.int16),
            "exec_success": np.dtype(np.bool_),
            "eval_success_after": np.dtype(np.bool_),
            "early_stop_after": np.dtype(np.bool_),
        }
        for name, expected_dtype in one_dimensional.items():
            if trace[name].shape != (count,):
                raise ValueError(f"{name} has shape {trace[name].shape}, expected {(count,)}")
            if trace[name].dtype != expected_dtype:
                raise ValueError(f"{name} has dtype {trace[name].dtype}, expected {expected_dtype}")

        if count > 0:
            expected_indices = np.arange(1, count + 1, dtype=np.int32)
            if not np.array_equal(trace["action_index"], expected_indices):
                raise ValueError("action_index is not contiguous and one-based")
            if np.any(trace["sim_step_after"] < trace["sim_step_before"]):
                raise ValueError("sim_step_after precedes sim_step_before")
            if not truncated:
                if termination_reason == "success" and not bool(trace["eval_success_after"][-1]):
                    raise ValueError("success trace does not end with eval_success_after")
                if termination_reason == "task_early_stop" and not bool(trace["early_stop_after"][-1]):
                    raise ValueError("task_early_stop trace does not end with early_stop_after")

    return {
        "schema_version": schema_version,
        "seed": seed,
        "recorded_actions": count,
        "total_actions_seen": total_actions_seen,
        "raw_action_dim": raw_dim,
        "raw_action_rep": raw_action_rep,
        "max_actions": max_actions,
        "truncated": truncated,
        "termination_reason": termination_reason,
        "result": result,
        "sha256": _sha256(path),
    }
