#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=motjepa-actiongate
#SBATCH --output=logs/%x-%j.log
set -uo pipefail
cd "${REPO_ROOT}"
export JAX_PLATFORMS=cpu USE_SWANLAB=false OMP_NUM_THREADS=8
srun "${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_action_probe.py \
  --pretrained_run "${SNAP}" --pretrained_step 18500 \
  --clips "${CLIPS}/*/*.zarr" --batches-per-domain 6
