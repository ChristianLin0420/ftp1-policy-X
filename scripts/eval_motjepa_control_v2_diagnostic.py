#!/usr/bin/env python
"""Run the exact unqualified final V3 head as an explicitly non-official diagnostic."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import types

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "UniVTAC" / "scripts"))
sys.path.insert(0, str(_REPO / "UniVTAC"))
sys.path.insert(0, str(_REPO / "src"))


_DATASET_MODULE = "openpi.mot_jepa.control_v2_dataset"
_EXPECTED_SPLIT_IDENTITY = "source_episode_seed_v1"
_OUTER_STORE_VERIFICATION = "MOT_CONTROL_V2_DIAGNOSTIC_OUTER_STORE_VERIFIED"


def _install_runtime_dataset_contract_stub() -> None:
    """Expose the training-only dataset identity without importing Zarr in Isaac Python."""

    if os.environ.get("MOT_CONTROL_V2_DIAGNOSTIC_UNQUALIFIED_FINAL") != "1":
        raise ValueError("refusing diagnostic runtime shim without explicit diagnostic opt-in")
    if os.environ.get(_OUTER_STORE_VERIFICATION) != "1":
        raise ValueError("diagnostic runtime requires outer prepared-store verification")

    run_value = os.environ.get("MOT_CONTROL_V2_RUN")
    if not run_value:
        raise ValueError("diagnostic runtime requires MOT_CONTROL_V2_RUN")
    config_path = pathlib.Path(run_value) / "run_config.json"
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("split_identity") != _EXPECTED_SPLIT_IDENTITY:
        raise ValueError(f"diagnostic run must use split identity {_EXPECTED_SPLIT_IDENTITY!r}")

    existing = sys.modules.get(_DATASET_MODULE)
    if existing is not None:
        if getattr(existing, "CONTROL_V2_SPLIT_IDENTITY", None) != _EXPECTED_SPLIT_IDENTITY:
            raise RuntimeError("loaded control-v2 dataset module has the wrong split identity")
        return

    # Deployment imports only this constant from the training dataset module. The real module
    # imports Zarr, which is intentionally absent from the qualified Isaac runtime. The outer
    # launcher verifies the immutable prepared-store manifest immediately before and after the
    # rollout, so no dataset access is needed inside the simulator process.
    dataset_contract = types.ModuleType(_DATASET_MODULE)
    dataset_contract.CONTROL_V2_SPLIT_IDENTITY = _EXPECTED_SPLIT_IDENTITY
    sys.modules[_DATASET_MODULE] = dataset_contract


_install_runtime_dataset_contract_stub()

import eval_motjepa_control_v2 as base  # noqa: E402
from mot_jepa_control_v2_diagnostic import DIAGNOSTIC_SELECTION_METRIC  # noqa: E402
from mot_jepa_control_v2_diagnostic import load_unqualified_final_control_v2_artifacts  # noqa: E402


def _build_diagnostic_policy_class():
    from openpi.mot_jepa.control_v2_deploy import MotJepaControlV2Policy  # noqa: PLC0415

    class DiagnosticFinalControlV2ChunkPolicy(MotJepaControlV2Policy):
        """Wear the common harness constructor while retaining the diagnostic V3 runtime."""

        def __init__(self, checkpoint_dir, domain_name, device, **kwargs):
            run = pathlib.Path(os.environ["MOT_CONTROL_V2_RUN"])
            if pathlib.Path(checkpoint_dir).resolve() != run.resolve():
                raise ValueError(f"harness checkpoint {checkpoint_dir} does not match diagnostic run {run}")
            raw_step = os.environ.get("MOT_CONTROL_V2_STEP")
            train_job_id = os.environ.get("MOT_CONTROL_V2_TRAIN_JOB_ID", "")
            if not raw_step:
                raise ValueError("diagnostic evaluation requires explicit MOT_CONTROL_V2_STEP")
            if os.environ.get("MOT_CONTROL_V2_DIAGNOSTIC_UNQUALIFIED_FINAL") != "1":
                raise ValueError("diagnostic evaluation requires the literal opt-in value 1")
            dev = base._prepare_device(device)  # noqa: SLF001
            artifacts = load_unqualified_final_control_v2_artifacts(
                run,
                dev,
                step=int(raw_step),
                train_job_id=train_job_id,
                # The provenance-closed outer job verifies the prepared store immediately before
                # this process and again after it exits. The Isaac Python intentionally has no
                # Zarr dependency, so repeating the same content read here is unavailable.
                verify_source_stores=False,
            )
            if artifacts.selection_metric != DIAGNOSTIC_SELECTION_METRIC:
                raise RuntimeError("loader did not return the diagnostic-only checkpoint identity")
            task = str(domain_name).replace("UniVTAC_", "")
            if task != artifacts.config.task:
                raise KeyError(f"diagnostic V3 run is for {artifacts.config.task!r}, not {task!r}")
            if int(kwargs.get("chunk_first_n", 20)) != artifacts.artifact.chunk_first_n:
                raise ValueError("harness chunk_first_n differs from the diagnostic V3 artifact")
            if float(kwargs.get("ensemble_K", 0.01)) != artifacts.artifact.temporal_ensemble_k:
                raise ValueError("harness ensemble_K differs from the diagnostic V3 artifact")
            if kwargs.get("action_rep") != "absolute":
                raise ValueError("the common harness must receive absolute V3 command semantics")
            if not bool(kwargs.get("use_temporal_ensemble", False)):
                raise ValueError("the common harness metadata must identify V3 temporal ensembling")
            super().__init__(
                artifacts,
                device=dev,
                save_infer_input_dir=kwargs.get("save_infer_input_dir"),
            )
            self.mapping = types.SimpleNamespace(arm_slice=(0, 7), gripper_index=7, gripper_slot_idx=7)
            base._ACTIVE_POLICY = self  # noqa: SLF001
            print(
                "[DIAGNOSTIC_ONLY NOT_OFFICIAL] "
                f"task={task!r}, final_unqualified_step={artifacts.step}, "
                f"{artifacts.selection_metric}={artifacts.selection_value:.8g}, "
                "held-out safety gate bypassed for rollout measurement only",
                flush=True,
            )

    return DiagnosticFinalControlV2ChunkPolicy


def main() -> int:
    if os.environ.get("MOT_CONTROL_V2_DIAGNOSTIC_UNQUALIFIED_FINAL") != "1":
        raise ValueError("refusing diagnostic final-checkpoint loader without explicit opt-in")
    if not os.environ.get("MOT_CONTROL_V2_STEP") or not os.environ.get("MOT_CONTROL_V2_TRAIN_JOB_ID"):
        raise ValueError("diagnostic evaluation requires explicit step and originating train job")
    base._build_policy_class = _build_diagnostic_policy_class  # type: ignore[attr-defined]  # noqa: SLF001
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
