#!/usr/bin/env python
"""Closed-loop UniVTAC evaluation of a frozen MoT-JEPA encoder plus an action head.

A SIBLING of ``UniVTAC/scripts/eval_ftp1.py``, not a fork of it. That file is 1400 lines of
episode loop, seeding, resume logic, video writing and result accounting, all of which is exactly
what we want unchanged -- reimplementing it would mean our number and the baseline's came from
different harnesses, which is the one thing that would make the comparison meaningless.

So this substitutes the single thing that differs. ``eval_ftp1.main()`` constructs
``FTP1ChunkPolicy`` at :1111 and hands it to the episode driver at :1149; we bind that name to a
MoT-JEPA-backed class with the same constructor signature and the same methods, then call
``main()``. Same seeds, same task config, same success criterion, same recorded artefacts.

``MOTJEPA_POLICY_KIND=ridge`` loads a fitted raw-unit ridge. ``flow`` loads a completed supervised
ActionDiT run and its checkpointed ActionNormalizer. Both execute MoT-JEPA chunk index zero and
re-plan every control step; unlike FTP-1, their chunks contain no current-timestamp placeholder.

    RIDGE=.../ridge_policy.npz SNAP=.../probe2_s50000 \
      python scripts/eval_motjepa.py --task_list lift_bottle --task_config demo \
        --total_num 20 --start_seed 1000000
"""

from __future__ import annotations

import os
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "UniVTAC" / "scripts"))
sys.path.insert(0, str(_REPO / "UniVTAC"))
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402


def _build_policy_class():
    """Defer every heavy import until after Isaac's app is up, as eval_ftp1 itself does."""
    from openpi.mot_jepa import config as config_module  # noqa: PLC0415
    from openpi.mot_jepa import runtime  # noqa: PLC0415
    from openpi.mot_jepa.action_dit import RidgeHead  # noqa: PLC0415
    from openpi.mot_jepa.action_parse import ACTION_DIM  # noqa: PLC0415
    from openpi.mot_jepa.deploy import MotJepaFlowPolicy  # noqa: PLC0415
    from openpi.mot_jepa.deploy import MotJepaRidgePolicy  # noqa: PLC0415
    from openpi.mot_jepa.deploy import load_trained_policy_artifacts  # noqa: PLC0415

    policy_kind = os.environ.get("MOTJEPA_POLICY_KIND", "ridge").strip().lower()
    if policy_kind not in {"ridge", "flow"}:
        raise ValueError(f"MOTJEPA_POLICY_KIND must be ridge or flow, got {policy_kind!r}")

    def prepare_device(device) -> torch.device:
        dev = torch.device(device if isinstance(device, str) else "cuda")
        if dev.type == "cuda":
            frac = float(os.environ.get("TORCH_MEM_FRAC", "0.15"))
            torch.cuda.set_per_process_memory_fraction(frac, dev.index or 0)
            print(f"[eval_motjepa] torch memory fraction {frac} on {dev}", flush=True)
        return dev

    def configure_harness(policy, mapping) -> None:
        policy.mapping = mapping
        # The MoT-JEPA policy consumes raw observations and returns absolute qpos8 directly. These
        # dimensions exist only because the common harness prints them.
        policy.state_dim = ACTION_DIM
        policy.model_action_dim = ACTION_DIM
        policy.use_temporal_ensemble = False

    if policy_kind == "flow":

        class MotJepaChunkPolicy(MotJepaFlowPolicy):
            """Completed supervised flow head wearing FTP1ChunkPolicy's constructor."""

            def __init__(self, checkpoint_dir, domain_name, device, **kwargs):
                run = pathlib.Path(os.environ["MOTJEPA_POLICY_RUN"])
                if pathlib.Path(checkpoint_dir).resolve() != run.resolve():
                    raise ValueError(f"harness checkpoint {checkpoint_dir} does not match flow run {run}")
                raw_step = os.environ.get("MOTJEPA_POLICY_STEP")
                requested_step = int(raw_step) if raw_step else None
                dev = prepare_device(device)
                artifacts = load_trained_policy_artifacts(run, dev, step=requested_step)
                if artifacts.config.head.objective != "flowmatch":
                    raise ValueError(
                        f"flow deployment requires objective=flowmatch, got {artifacts.config.head.objective}"
                    )

                task = str(domain_name).replace("UniVTAC_", "")
                if task not in artifacts.domain_names:
                    raise KeyError(f"task {task!r} not present in trained domains {artifacts.domain_names}")
                domain_id = artifacts.domain_names.index(task)
                super().__init__(
                    artifacts.backbone,
                    artifacts.head,
                    artifacts.normalizer,
                    artifacts.config.layout,
                    domain_id,
                    dev,
                    observation_stride=artifacts.observation_stride,
                    action_stride=artifacts.action_stride,
                    domain_names=artifacts.domain_names,
                    num_inference_steps=int(kwargs.get("num_inference_steps", 10)),
                    num_samples=int(os.environ.get("MOTJEPA_NUM_SAMPLES", "1")),
                    sample_seed=int(os.environ.get("MOTJEPA_SAMPLE_SEED", "0")),
                    save_infer_input_dir=kwargs.get("save_infer_input_dir"),
                )
                configure_harness(self, kwargs.get("mapping"))
                print(
                    f"[eval_motjepa] flow domain {task!r} -> id {domain_id}, "
                    f"head step {artifacts.head_step}, backbone step {artifacts.backbone_step}, "
                    f"backbone mode={artifacts.backbone_train_mode}, "
                    f"backbone checkpoint={artifacts.backbone_checkpoint}, "
                    f"adapted step={artifacts.adapted_backbone_step}, "
                    f"K={self.num_samples}, Euler={self.num_inference_steps}",
                    flush=True,
                )

        return MotJepaChunkPolicy

    class MotJepaChunkPolicy(MotJepaRidgePolicy):
        """Wears FTP1ChunkPolicy's constructor so eval_ftp1's worker can build it unmodified."""

        def __init__(self, checkpoint_dir, domain_name, device, **kwargs):
            # The ridge is a closed-form single-pass map with no sampler or temporal ensemble.
            ridge_path = os.environ["RIDGE"]
            snapshot = os.environ["SNAP"]
            if pathlib.Path(checkpoint_dir).resolve() != pathlib.Path(snapshot).resolve():
                raise ValueError(f"harness checkpoint {checkpoint_dir} does not match ridge backbone {snapshot}")
            raw_step = os.environ.get("SNAP_STEP")
            step = int(raw_step) if raw_step else None
            cfg = config_module.CONFIGS[os.environ.get("MOTJEPA_CONFIG", "mot_jepa_pilot")]

            dev = prepare_device(device)
            backbone, step = runtime.load_frozen_backbone(pathlib.Path(snapshot), step, cfg, dev)
            head = RidgeHead(ridge_path, cfg.layout).to(dev)

            # Which ridge row to use. The npz stores the domain NAMES it was fitted on, so match on
            # the task rather than on an index -- an off-by-one here would silently evaluate one
            # task's policy on another and still produce a plausible-looking success rate.
            task = str(domain_name).replace("UniVTAC_", "")
            names = [str(n) for n in head.names]
            if task not in names:
                raise KeyError(f"no ridge fitted for {task!r}; have {names}")
            domain_id = names.index(task)
            if domain_id not in head.fitted_domains:
                raise KeyError(f"ridge for {task!r} exists but was skipped at fit time (too few clips)")

            super().__init__(
                backbone,
                head,
                cfg.layout,
                domain_id,
                dev,
                domain_names=tuple(names),
                save_infer_input_dir=kwargs.get("save_infer_input_dir"),
            )
            configure_harness(self, kwargs.get("mapping"))
            print(
                f"[eval_motjepa] ridge domain {task!r} -> id {domain_id}, backbone step {step}",
                flush=True,
            )

    return MotJepaChunkPolicy


def _stub_ftp1_model_stack() -> None:
    """Satisfy eval_ftp1's three FTP-1 imports without pulling in jax/transformers.

    eval_ftp1.py:52-54 imports FTP1_RESERVED_ACTION_DIM, FTP1_SINGLE_ARM_ACTION_REP_DIM and
    FTP1InferenceWrapper. The first two are layout constants; the third is only ever touched
    inside FTP1ChunkPolicy, which we replace wholesale. Importing the real modules would drag the
    entire FTP-1 model stack into Isaac's interpreter -- heavy, and it would risk moving the torch
    and numpy that Gates 1 and 2 validated.

    The constants come from OUR action_parse rather than being written as literals, so if the
    layout ever changes these cannot silently disagree with the policy's own view of it.
    """
    import types  # noqa: PLC0415

    from openpi.mot_jepa.action_parse import RESERVED_ACTION_DIM  # noqa: PLC0415
    from openpi.mot_jepa.action_parse import SINGLE_ARM_ACTION_REP_DIM  # noqa: PLC0415

    cfg = types.ModuleType("openpi.models_pytorch.ftp1_model_config")
    cfg.FTP1_RESERVED_ACTION_DIM = RESERVED_ACTION_DIM
    cfg.FTP1_SINGLE_ARM_ACTION_REP_DIM = SINGLE_ARM_ACTION_REP_DIM
    sys.modules["openpi.models_pytorch.ftp1_model_config"] = cfg

    wrapper = types.ModuleType("openpi.policies.ftp1_inference_wrapper")

    class FTP1InferenceWrapper:  # never instantiated: FTP1ChunkPolicy is replaced below
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FTP1InferenceWrapper is stubbed; eval_motjepa supplies the policy")

    wrapper.FTP1InferenceWrapper = FTP1InferenceWrapper
    sys.modules["openpi.policies.ftp1_inference_wrapper"] = wrapper


def main() -> int:
    _stub_ftp1_model_stack()
    import eval_ftp1  # noqa: PLC0415

    eval_ftp1.FTP1ChunkPolicy = _build_policy_class()
    # eval_ftp1 infers action_rep from the FTP-1 checkpoint's train_config.json, which we have no
    # equivalent of. Ours is fixed by construction: deploy.chunk_to_absolute_qpos8 integrates the
    # arm deltas and passes the gripper through, so the harness must not re-base on top of that.
    # The replacement class is local to this process. eval_ftp1's optional multiprocessing-spawn
    # workers would re-import its original FTP1 class instead of inheriting this binding.
    sys.argv += ["--action_rep", "absolute", "--workers", "1"]
    return eval_ftp1.main()


if __name__ == "__main__":
    raise SystemExit(main())
