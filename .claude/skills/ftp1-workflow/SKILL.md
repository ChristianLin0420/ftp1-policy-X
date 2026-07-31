---
name: ftp1-workflow
description: Prepare, validate, train, evaluate, and deploy FTP-1 tactile policies in this repository. Use when working with FTP-1 Zarr datasets, heterogeneous tactile pretraining, UniVTAC fine-tuning, FTP1 PyTorch checkpoints, normalization or tactile tokenizer assets, FTP1InferenceWrapper, or Isaac Sim evaluation.
---

# FTP-1 Workflow

Follow a gated path: validate artifacts, smoke-test one task, fine-tune, evaluate offline, then evaluate closed-loop in UniVTAC. Do not start a multi-terabyte corpus download until storage is explicitly verified.

## Select the workflow

- For Zarr schema, sensor identity, action layout, and checkpoint requirements, read [data-contract.md](references/data-contract.md).
- For commands covering environment setup, fine-tuning, evaluation, deployment, and subset pretraining, read [runbook.md](references/runbook.md).
- For a new real-robot embodiment, use the inference contract in `src/openpi/policies/ftp1_inference_wrapper.py`, then add robot-specific observation packing, action mapping, calibration, limits, and an emergency-stop path.

## Apply the gates

1. Run `uv run python scripts/ftp1_preflight.py --dataset-config <config>` before normalization.
2. Add `--checkpoint <checkpoint> --domain-name <domain>` before inference or deployment.
3. Compute normalization with the exact dataset config, split, pose representations, and joint representations used for training.
4. Run a 50-100-step one-GPU smoke test before a distributed training launch.
5. Require `model.safetensors`, model/train configs, normalization, and tactile tokenizers in a deployable checkpoint.
6. Capture exact model inputs during the first UniVTAC rollout and inspect pad order, colors, state packing, and gripper slot 28.
7. Use fixed seeds and identical runtime settings when comparing checkpoints.

## Preserve experiment integrity

- Split by source episode, never by timestep.
- Keep `repo_id`, dataset domain name, and normalization domain aligned.
- Keep the FTP-1 action representation consistent end to end: relative poses, absolute proprioceptive joints, and `mix` action joints for the UniVTAC reference path.
- On NVIDIA Blackwell (`sm_120`), install the repository's CUDA 12.8 PyTorch override after `uv sync` and use `FTP1_UV_NO_SYNC=true` for every launcher so uv does not restore the locked CUDA 12.6 wheel.
- Treat multi-GPU training as a separate acceptance gate. On the Precision 7960 reference host, basic NCCL collectives pass but DDP's initial synchronization of the 4.3B model triggers an illegal memory access; use the verified one-GPU path until that PyTorch/NCCL stack is upgraded and requalified.
- Treat offline RMSE as a diagnostic, not a substitute for closed-loop success rate.
- Record repository revision, model/dataset revision, effective CLI, GPU/driver versions, checkpoint step, seeds, and success metadata.
- Add newly discovered failure modes and verified remedies to this skill rather than creating a standalone report.
