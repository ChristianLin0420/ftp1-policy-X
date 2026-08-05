#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=motjepa-readout
#SBATCH --output=logs/%x-%j.log
#
# Viability gate for the action policy: held-out R^2(future action chunk <- frozen latent).
# Read-only. See scripts/mot_jepa_action_readout.py for the pre-agreed interpretation.
#
#   REPO_ROOT=... SNAP=.../backbones/probe2_s50000 STEP=50000 CLIPS=... \
#     sbatch scripts_exp_zarr/mot_jepa/action_readout.sbatch
set -uo pipefail
cd "${REPO_ROOT}"
export JAX_PLATFORMS=cpu USE_SWANLAB=false OMP_NUM_THREADS=8
srun "${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_action_readout.py \
  --pretrained_run "${SNAP}" --pretrained_step "${STEP}" \
  --clips "${CLIPS}/*/*.zarr" --horizon "${HORIZON:-15}" --batches "${BATCHES:-192}"
