#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
# cpu nodes have 251904 MB / 96 CPUs; asking for 256G is rejected outright.
#SBATCH --job-name=univtac-convert
#SBATCH --output=logs/%x-%j.log
#
# UniVTAC HDF5 -> FTP-1 zarr. Stage 1 of 3; mot_jepa_build_clips and
# mot_jepa_add_conditioning follow.
#
# EPISODES_PER_TASK is set to 100 explicitly. The wrapper script defaults it to 50, which would
# silently discard half of every task's episodes -- there are 100 per task on disk.
#
#   REPO_ROOT=... ASSETS=.../ftp1-assets/UniVTAC OUT=.../univtac-zarr \
#     sbatch scripts_exp_zarr/mot_jepa/univtac_convert.sbatch
set -uo pipefail
cd "${REPO_ROOT}/data_processing"
export JAX_PLATFORMS=cpu USE_SWANLAB=false

echo "[$(date -Is)] converting ${ASSETS} -> ${OUT}"
"${REPO_ROOT}/.venv/bin/python" -m parse_data_module.parse_data_univtac \
  --base_dir "${ASSETS}" \
  --save_dir "${OUT}" \
  --image_size "${IMAGE_SIZE:-224}" \
  --episodes_per_task "${EPISODES_PER_TASK:-100}"
RC=$?
echo "[$(date -Is)] parse rc=${RC}"
ls -la "${OUT}" 2>/dev/null
exit ${RC}
