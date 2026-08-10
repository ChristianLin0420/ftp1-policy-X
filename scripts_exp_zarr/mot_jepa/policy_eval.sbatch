#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=motjepa-polieval
#SBATCH --output=logs/%x-%j.log
#
# Held-out action-chunk RMSE for one policy arm. Read-only.
#
#   REPO_ROOT=... RUN=<policy run dir> SNAP=<backbone> CLIPS=<univtac clips> OUT=<json> \
#     sbatch scripts_exp_zarr/mot_jepa/policy_eval.sbatch
set -uo pipefail
cd "${REPO_ROOT}"
export JAX_PLATFORMS=cpu USE_SWANLAB=false OMP_NUM_THREADS=8
srun "${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_policy_eval.py \
  --run "${RUN}" \
  --pretrained_run "${SNAP}" --pretrained_step "${STEP:-50000}" \
  --clips "${CLIPS}/*/*.zarr" \
  --holdout_mod "${HOLDOUT_MOD:-10}" \
  --index-step "${INDEX_STEP:-17}" \
  --out "${OUT}"
