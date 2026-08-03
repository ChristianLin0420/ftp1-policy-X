#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=motjepa-buildclips
#SBATCH --output=logs/%x.%A_%a.log
#SBATCH --error=logs/%x.%A_%a.log
#SBATCH --open-mode=append
#
# Build the derived clip store, one array task per domain.
#
# Extraction alone does not make a domain trainable. The released stores are the wrong SHAPE
# for this workload: gel is stored at 224 squared but the model consumes 112, and RGB is
# chunked at 14 frames so a 16-frame clip straddles two chunks and decompresses ~28 frames to
# use 16. Measured on RDP_Bimanual, the derived store reads 4.1x faster (1520 -> 367 ms/clip)
# and is 4.8x smaller (14 G -> 2.9 G).
#
# Submit with --dependency=afterany:<extract-array> so it starts as extraction lands. `afterany`
# rather than `afterok`: one domain failing to extract should not block building the rest.
#
#   DOMAINS="a b c" sbatch --array=0-$((n-1)) --dependency=afterany:<jobid> \
#       scripts_exp_zarr/mot_jepa/build_clips.sbatch

set -uo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
CLIPS_ROOT="${CLIPS_ROOT:?set CLIPS_ROOT}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
VIDEO_SIZE="${VIDEO_SIZE:-224}"
GEL_SIZE="${GEL_SIZE:-112}"
NUM_FRAMES="${NUM_FRAMES:-16}"
read -r -a DOMAIN_LIST <<< "${DOMAINS:?set DOMAINS}"

DOMAIN="${DOMAIN_LIST[${SLURM_ARRAY_TASK_ID}]}"
echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID} domain=${DOMAIN} node=$(hostname)"

if [[ -z "$(find "${DATASET_ROOT}/${DOMAIN}" -maxdepth 3 -name '*.zarr' -not -path '*extract*' 2>/dev/null | head -1)" ]]; then
  echo "[$(date -Is)] ${DOMAIN}: no extracted zarr stores; skipping"
  exit 0
fi

cd "${REPO_ROOT}"
mkdir -p "${CLIPS_ROOT}"

START=${SECONDS}
"${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_build_clip_zarr.py \
  --source "${DATASET_ROOT}" \
  --output "${CLIPS_ROOT}" \
  --domains "${DOMAIN}" \
  --video-size "${VIDEO_SIZE}" \
  --gel-size "${GEL_SIZE}" \
  --num-frames "${NUM_FRAMES}"
RC=$?

BUILT=$(find "${CLIPS_ROOT}/${DOMAIN}" -maxdepth 2 -name "*.zarr" 2>/dev/null | wc -l)
echo "[$(date -Is)] ${DOMAIN}: rc=${RC} in $((SECONDS - START))s, ${BUILT} derived store(s), $(du -sh "${CLIPS_ROOT}/${DOMAIN}" 2>/dev/null | cut -f1)"
exit ${RC}
