# FTP-1 Operational Runbook

## Training environment

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --dev
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
uv run python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

For an NVIDIA Blackwell GPU (`sm_120`), the CUDA 12.6 wheel in the lock file cannot execute kernels. Apply the supported CUDA 12.8 wheel override after the normal sync, then prevent later launcher invocations from reconciling it away:

```bash
bash scripts/install_pytorch_cuda128.sh
export FTP1_UV_NO_SYNC=true
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability())'
```

Keep checkpoints, assets, and the OpenPI cache on a filesystem with adequate free space. Portable launchers default to `.cache/ftp1`, but accept `FTP1_CACHE_ROOT`, `FTP1_CHECKPOINT_BASE_DIR`, `FTP1_ASSETS_BASE_DIR`, and `OPENPI_DATA_HOME`.

## UniVTAC fine-tuning

Prepare a JSON config whose enabled domain is `UniVTAC_lift_bottle` and whose path contains immediate `*.zarr` stores. Then run:

```bash
export FTP1_DATASET_CONFIG=/absolute/path/dataset_univtac_lift_bottle.json
export FTP1_PRETRAINED_CHECKPOINT=/absolute/path/ftp1_pretrain_v0426_50kstep
export FTP1_CACHE_ROOT=/absolute/path/ftp1_cache
export FTP1_REPO_ID=univtac_lift_bottle  # Use exactly the same value for norm stats and training.
export CUDA_VISIBLE_DEVICES=0
export FTP1_UV_NO_SYNC=true  # Required when using the CUDA 12.8 override.

uv run python scripts/ftp1_preflight.py \
  --dataset-config "$FTP1_DATASET_CONFIG" \
  --checkpoint "$FTP1_PRETRAINED_CHECKPOINT"
bash scripts_exp_zarr/univtac/compute_norm_stats_univtac_example.sh

# Smoke test; checkpoints remain compatible with the full run.
FTP1_NUM_TRAIN_STEPS=100 FTP1_SAVE_INTERVAL=100 FTP1_VAL_INTERVAL=50 \
CUDA_VISIBLE_DEVICES=0 bash scripts_exp_zarr/univtac/train_univtac_example.sh

# Full verified one-GPU run on the Precision 7960 reference host.
FTP1_LOCAL_BATCH_SIZE=2 bash scripts_exp_zarr/univtac/train_univtac_example.sh
```

The reference launcher uses a 90/10 episode split, Z-score normalization, channel-wise image-tactile statistics, BF16, a Gemma-small tactile expert, and 20,000 steps. It uses `action_joint_rep=mix`; normalization must be computed with that same representation. It also skips FoundationTactile's T3-large initialization by default because the released FTP-1 checkpoint already includes the matching 768-wide HPT tokenizer, while T3-large is shape-incompatible. Set `FTP1_ENABLE_TRACKING=true` for W&B-compatible logging. Use the SwanLab wrapper only after exporting `SWANLAB_API_KEY`.

The Precision 7960 host has a verified local batch size of 2 on one GPU (one optimizer step in about 27 seconds). Three-rank NCCL `all_reduce` succeeds, but DDP's initial parameter synchronization for the 4.3B model fails with an illegal memory access even with peer-to-peer transport disabled. Do not schedule a long DDP job on this stack until a newer PyTorch/NCCL combination passes a full-model one-step gate.

The parser accepts both historical `UniVTAC/<task>/demo/hdf5/*.hdf5` inputs and the published `UniVTAC/<task>/clean/*.hdf5` layout:

```bash
BASE_DIR=/absolute/path/UniVTAC SAVE_DIR=/absolute/path/processed/UniVTAC \
EPISODES_PER_TASK=100 TASK_LIST=lift_bottle \
bash data_processing/parse_data_scripts/parse_data_univtac.sh
```

## Offline and UniVTAC evaluation

Run held-out action evaluation from the training environment:

```bash
uv run python scripts/zarr_eval_ftp1_pytorch.py \
  --checkpoint_dir /absolute/path/to/checkpoint/19999 \
  --device cuda
```

The evaluator writes `evaluation_aggregated_metrics.json` inside the selected step directory. A metric value of `-1` denotes an action group that is inactive for that embodiment (for example, left-arm or Cartesian-pose fields in the UniVTAC right-arm joint-control task); use `rmse_total` and the active right-arm/right-hand fields for comparisons. The entrypoint must remain non-interactive—`scripts/test_zarr_eval_ftp1.py` guards against accidental debugger traps.

Install UniVTAC in its Python 3.10 Isaac Sim environment by following `UniVTAC/Installation_FTP1.md`, then install the inference subset without replacing Isaac's PyTorch:

```bash
conda activate uni
bash UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh
```

Run a ten-episode gate, inspect `infer_input_sample`, then run 100 episodes:

```bash
cd UniVTAC
CHECKPOINT_DIR=/absolute/path/to/checkpoint/19999 \
DOMAIN_NAME=UniVTAC_lift_bottle TOTAL_NUM=10 GPU=0 \
bash scripts/shell/eval_ftp1.sh lift_bottle

CHECKPOINT_DIR=/absolute/path/to/checkpoint/19999 \
DOMAIN_NAME=UniVTAC_lift_bottle TOTAL_NUM=100 GPU=0 \
bash scripts/shell/eval_ftp1.sh lift_bottle
```

Use fixed seeds, `action_rep=auto`, ten diffusion steps, temporal ensembling, and the same chunk prefix for checkpoint comparisons. Results live under `UniVTAC/eval_results/FTP1` with videos and `metadata.json`.

## Released pretraining corpus

The release is roughly 7.5 TB compressed. Provision at least 16 TB before attempting the full corpus. Download individual archives with the ModelScope or Hugging Face CLI rather than cloning everything. Validate and extract one archive without materializing a combined tar:

```bash
bash scripts/stage_ftp1_archive.sh --check-only /archive/download/MotionTrans /data/ftp1/MotionTrans
bash scripts/stage_ftp1_archive.sh /archive/download/MotionTrans /data/ftp1/MotionTrans
```

Point `scripts_exp_zarr/pretrain_small/data_config_pretrain_small.json` at the extracted Zarr domain, then run its normalization and training launchers. Qualify a subset-pretrained checkpoint by fine-tuning and evaluating it on the same UniVTAC split; do not use pretraining loss alone as the transfer result.
