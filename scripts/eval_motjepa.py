#!/usr/bin/env python
"""Closed-loop UniVTAC evaluation of the frozen MoT-JEPA encoder + fitted per-domain ridge.

A SIBLING of ``UniVTAC/scripts/eval_ftp1.py``, not a fork of it. That file is 1400 lines of
episode loop, seeding, resume logic, video writing and result accounting, all of which is exactly
what we want unchanged -- reimplementing it would mean our number and the baseline's came from
different harnesses, which is the one thing that would make the comparison meaningless.

So this substitutes the single thing that differs. ``eval_ftp1.main()`` constructs
``FTP1ChunkPolicy`` at :1111 and hands it to the episode driver at :1149; we bind that name to a
MoT-JEPA-backed class with the same constructor signature and the same methods, then call
``main()``. Same seeds, same task config, same success criterion, same recorded artefacts.

Deploys the RIDGE rather than a trained head: on 699 held-out clips the ridge scores 0.00329 RMSE
against a per-domain constant's 0.00435 and wins on all eight domains, while every trained head
lands 0.00454-0.00484 -- worse than the constant. And it re-plans EVERY step, because integrated
open-loop drift is 0.00231 rad at k=1 but 0.101 rad at k=32.

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

import numpy as np
import torch


def _build_policy_class():
    """Defer every heavy import until after Isaac's app is up, as eval_ftp1 itself does."""
    from openpi.mot_jepa import config as config_module
    from openpi.mot_jepa import runtime
    from openpi.mot_jepa.action_parse import ACTION_DIM
    from openpi.mot_jepa.action_dit import RidgeHead
    from openpi.mot_jepa.deploy import MotJepaRidgePolicy

    class MotJepaChunkPolicy(MotJepaRidgePolicy):
        """Wears FTP1ChunkPolicy's constructor so eval_ftp1's worker can build it unmodified."""

        def __init__(self, checkpoint_dir, domain_name, device, **kwargs):
            # eval_ftp1 passes num_inference_steps, tactile_key, tactile_sensor, chunk_first_n,
            # save_infer_input_dir, temporal ensembling. None apply: the ridge is a closed-form
            # single-pass map with no sampler and no ensembling, and it consumes the raw
            # observation dict rather than FTP-1's tokenised inputs.
            mapping = kwargs.get("mapping")
            ridge_path = os.environ["RIDGE"]
            snapshot = os.environ["SNAP"]
            step = int(os.environ.get("SNAP_STEP", "50000"))
            cfg = config_module.CONFIGS[os.environ.get("MOTJEPA_CONFIG", "mot_jepa_pilot")]

            dev = torch.device(device if isinstance(device, str) else "cuda")
            backbone, _ = runtime.load_frozen_backbone(pathlib.Path(snapshot), step, cfg, dev)
            head = RidgeHead(ridge_path, cfg.layout, horizon=32).to(dev)

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

            super().__init__(backbone, head, cfg.layout, domain_id, dev)
            self.mapping = mapping
            # Every attribute eval_ftp1's episode loop reads off the policy object, enumerated from
            # the source (`grep -o 'rtac\.[a-z_]*'`) rather than discovered one crashed job at a
            # time: act, reset, set_task, get_last_action_debug, action_dim, action_rep, mapping,
            # model_action_dim, state_dim, use_temporal_ensemble, _chunk_history.
            #
            # state_dim and model_action_dim exist only so the harness can size FTP-1's tokenised
            # state vector. This policy consumes the raw observation dict and never uses either, so
            # they are reported as our own layout's width instead of being invented.
            self.state_dim = ACTION_DIM
            self.model_action_dim = ACTION_DIM
            # No temporal ensembling: the ridge is a single closed-form pass, and the drift
            # measurement makes per-step re-planning the point rather than chunk blending.
            self.use_temporal_ensemble = False
            print(f"[eval_motjepa] ridge domain {task!r} -> id {domain_id}, backbone step {step}",
                  flush=True)

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
    import types

    from openpi.mot_jepa.action_parse import RESERVED_ACTION_DIM
    from openpi.mot_jepa.action_parse import SINGLE_ARM_ACTION_REP_DIM

    cfg = types.ModuleType("openpi.models_pytorch.ftp1_model_config")
    cfg.FTP1_RESERVED_ACTION_DIM = RESERVED_ACTION_DIM
    cfg.FTP1_SINGLE_ARM_ACTION_REP_DIM = SINGLE_ARM_ACTION_REP_DIM
    sys.modules["openpi.models_pytorch.ftp1_model_config"] = cfg

    wrapper = types.ModuleType("openpi.policies.ftp1_inference_wrapper")

    class FTP1InferenceWrapper:  # never instantiated: FTP1ChunkPolicy is replaced below
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FTP1InferenceWrapper is stubbed; eval_motjepa uses the ridge policy")

    wrapper.FTP1InferenceWrapper = FTP1InferenceWrapper
    sys.modules["openpi.policies.ftp1_inference_wrapper"] = wrapper


def main() -> int:
    _stub_ftp1_model_stack()
    import eval_ftp1

    eval_ftp1.FTP1ChunkPolicy = _build_policy_class()
    # eval_ftp1 infers action_rep from the FTP-1 checkpoint's train_config.json, which we have no
    # equivalent of. Ours is fixed by construction: deploy.chunk_to_absolute_qpos8 integrates the
    # arm deltas and passes the gripper through, so the harness must not re-base on top of that.
    sys.argv += ["--action_rep", "absolute"]
    return eval_ftp1.main()


if __name__ == "__main__":
    raise SystemExit(main())
