#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=motjepa-tsctl
#SBATCH --output=logs/%x-%j.log
#
# Positive control for timeshuffle_gap. Read-only: loads a frozen EMA snapshot under no_grad
# and writes nothing but its log. See scripts/mot_jepa_timeshuffle_control.py for what the
# pooled/unpooled split decides.
#
#   REPO_ROOT=... SNAP=.../backbones/probe2_s50000 STEP=50000 CLIPS=... \
#     sbatch scripts_exp_zarr/mot_jepa/timeshuffle_control.sbatch
set -uo pipefail
cd "${REPO_ROOT}"
export JAX_PLATFORMS=cpu USE_SWANLAB=false OMP_NUM_THREADS=8
srun "${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_timeshuffle_control.py \
  --pretrained_run "${SNAP}" --pretrained_step "${STEP}" \
  --clips "${CLIPS}/*/*.zarr" --batches "${BATCHES:-12}"
