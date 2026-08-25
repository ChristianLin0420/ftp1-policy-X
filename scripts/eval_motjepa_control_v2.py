#!/usr/bin/env python
"""Run MoT-Control V2 through the unchanged official UniVTAC episode harness.

The common harness owns simulation, seeds, success criteria, videos, resume behavior, and action
traces.  This sibling entrypoint replaces only its policy class and augments trajectory NPZs with
the V2 safety counters required by the qualification gate.

Environment:
    MOT_CONTROL_V2_RUN: completed V2 run directory (required)
    MOT_CONTROL_V2_STEP: optional held-out-selected step assertion
    TORCH_MEM_FRAC: optional CUDA allocator fraction (default 0.15)
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import time
import types
from typing import Any

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "UniVTAC" / "scripts"))
sys.path.insert(0, str(_REPO / "UniVTAC"))
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

_ACTIVE_POLICY: Any | None = None


def _prepare_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device if isinstance(device, str) else "cuda")
    if dev.type == "cuda":
        fraction = float(os.environ.get("TORCH_MEM_FRAC", "0.15"))
        torch.cuda.set_per_process_memory_fraction(fraction, dev.index or 0)
        print(f"[eval_motjepa_control_v2] torch memory fraction {fraction} on {dev}", flush=True)
    return dev


def _build_policy_class():
    from openpi.mot_jepa.control_v2_deploy import MotJepaControlV2Policy  # noqa: PLC0415
    from openpi.mot_jepa.control_v2_deploy import load_trained_control_v2_artifacts  # noqa: PLC0415

    class MotJepaControlV2ChunkPolicy(MotJepaControlV2Policy):
        """Wear ``FTP1ChunkPolicy``'s constructor while retaining the V2 runtime."""

        def __init__(self, checkpoint_dir, domain_name, device, **kwargs):
            run = pathlib.Path(os.environ["MOT_CONTROL_V2_RUN"])
            if pathlib.Path(checkpoint_dir).resolve() != run.resolve():
                raise ValueError(f"harness checkpoint {checkpoint_dir} does not match V2 run {run}")
            raw_step = os.environ.get("MOT_CONTROL_V2_STEP")
            requested_step = int(raw_step) if raw_step else None
            dev = _prepare_device(device)
            artifacts = load_trained_control_v2_artifacts(run, dev, step=requested_step)
            task = str(domain_name).replace("UniVTAC_", "")
            if task != artifacts.config.task:
                raise KeyError(f"V2 run is for {artifacts.config.task!r}, not {task!r}")
            if int(kwargs.get("chunk_first_n", 20)) != artifacts.artifact.chunk_first_n:
                raise ValueError("harness chunk_first_n differs from the V2 artifact")
            if float(kwargs.get("ensemble_K", 0.01)) != artifacts.artifact.temporal_ensemble_k:
                raise ValueError("harness ensemble_K differs from the V2 artifact")
            if kwargs.get("action_rep") != "absolute":
                raise ValueError("the common harness must receive absolute V2 command semantics")
            if not bool(kwargs.get("use_temporal_ensemble", False)):
                raise ValueError("the common harness metadata must identify V2 temporal ensembling")
            super().__init__(
                artifacts,
                device=dev,
                save_infer_input_dir=kwargs.get("save_infer_input_dir"),
            )
            # eval_ftp1's generic trace extractor needs an explicit map for native 8-D chunks.
            self.mapping = types.SimpleNamespace(arm_slice=(0, 7), gripper_index=7, gripper_slot_idx=7)
            global _ACTIVE_POLICY  # noqa: PLW0603
            _ACTIVE_POLICY = self
            print(
                f"[eval_motjepa_control_v2] task={task!r}, selected_step={artifacts.step}, "
                f"completed_step={artifacts.completed_step}, "
                f"{artifacts.selection_metric}={artifacts.selection_value:.8g}, "
                f"backbone_mode={artifacts.config.backbone_train_mode}, H32/index1/first20/K=0.01",
                flush=True,
            )

    return MotJepaControlV2ChunkPolicy


def _stub_ftp1_model_stack() -> None:
    """Satisfy imports that the replaced FTP-1 policy class never reaches."""
    from openpi.mot_jepa.action_parse import RESERVED_ACTION_DIM  # noqa: PLC0415
    from openpi.mot_jepa.action_parse import SINGLE_ARM_ACTION_REP_DIM  # noqa: PLC0415

    config_module = types.ModuleType("openpi.models_pytorch.ftp1_model_config")
    config_module.FTP1_RESERVED_ACTION_DIM = RESERVED_ACTION_DIM
    config_module.FTP1_SINGLE_ARM_ACTION_REP_DIM = SINGLE_ARM_ACTION_REP_DIM
    sys.modules["openpi.models_pytorch.ftp1_model_config"] = config_module

    wrapper_module = types.ModuleType("openpi.policies.ftp1_inference_wrapper")

    class FTP1InferenceWrapper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FTP1InferenceWrapper is stubbed; the V2 policy replaces it")

    wrapper_module.FTP1InferenceWrapper = FTP1InferenceWrapper
    sys.modules["openpi.policies.ftp1_inference_wrapper"] = wrapper_module


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_trace_counter_extension(eval_ftp1) -> None:
    """Add fail-closed per-episode safety counters without forking the common evaluator."""
    original = eval_ftp1.EpisodeTraceRecorder

    class ControlV2TraceRecorder(original):
        def write(self, *, termination_reason: str, result: str) -> dict[str, Any]:
            metadata = super().write(termination_reason=termination_reason, result=result)
            if _ACTIVE_POLICY is None:
                raise RuntimeError("V2 trace finalized before its policy was constructed")
            counters = _ACTIVE_POLICY.get_safety_counters()
            artifacts = _ACTIVE_POLICY.artifacts
            required = {
                "command_count",
                "joint_clamp_commands",
                "joint_clamped_values",
                "rate_clamp_commands",
                "rate_clamped_values",
                "nonfinite_fallbacks",
            }
            if set(counters) != required:
                raise ValueError(f"unexpected V2 safety counter schema {sorted(counters)}")
            clamp_count = int(counters["joint_clamp_commands"]) + int(counters["rate_clamp_commands"])
            path = pathlib.Path(self.output_path)
            with np.load(path, allow_pickle=False) as trace:
                payload = {name: trace[name] for name in trace.files}
            payload.update(
                {
                    "control_v2_trace_schema": np.asarray(1, dtype=np.int32),
                    "control_v2_clamp_count": np.asarray(clamp_count, dtype=np.int32),
                    "control_v2_nonfinite_fallbacks": np.asarray(counters["nonfinite_fallbacks"], dtype=np.int32),
                    "control_v2_policy_step": np.asarray(artifacts.step, dtype=np.int64),
                    "control_v2_completed_step": np.asarray(artifacts.completed_step, dtype=np.int64),
                    "control_v2_selection_metric": np.asarray(artifacts.selection_metric),
                    "control_v2_selection_value": np.asarray(artifacts.selection_value, dtype=np.float64),
                    "control_v2_artifact_sha256": np.asarray(artifacts.metadata["artifact_sha256"]),
                    **{f"control_v2_{name}": np.asarray(value, dtype=np.int32) for name, value in counters.items()},
                }
            )
            temporary = path.with_name(f".{path.name}.v2.{os.getpid()}.{time.time_ns()}")
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            eval_ftp1.validate_episode_trace(path, expected_seed=self.seed)
            # The common recorder computed its digest before the extension was added.
            metadata["sha256"] = _file_sha256(path)
            metadata["control_v2_safety"] = {
                "schema_version": 1,
                "clamp_count": clamp_count,
                **counters,
            }
            metadata["control_v2_policy"] = {
                "selected_step": artifacts.step,
                "completed_step": artifacts.completed_step,
                "selection_metric": artifacts.selection_metric,
                "selection_value": artifacts.selection_value,
                "artifact_sha256": artifacts.metadata["artifact_sha256"],
            }
            return metadata

    eval_ftp1.EpisodeTraceRecorder = ControlV2TraceRecorder


def main() -> int:
    if not os.environ.get("MOT_CONTROL_V2_RUN"):
        raise ValueError("MOT_CONTROL_V2_RUN must name a completed V2 run")
    _stub_ftp1_model_stack()
    import eval_ftp1  # noqa: PLC0415

    eval_ftp1.FTP1ChunkPolicy = _build_policy_class()
    _install_trace_counter_extension(eval_ftp1)
    # Pin every execution semantic the common CLI would otherwise leave configurable.  The V2
    # class performs the ensemble itself; --temporal_ensemble keeps the harness metadata honest.
    sys.argv += [
        "--action_rep",
        "absolute",
        "--workers",
        "1",
        "--temporal_ensemble",
        "--save_trajectory",
        "--chunk_first_n",
        "20",
        "--ensemble_K",
        "0.01",
    ]
    return eval_ftp1.main()


if __name__ == "__main__":
    raise SystemExit(main())
