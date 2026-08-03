#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=motjepa-extract
#SBATCH --output=logs/%x.%A_%a.log
#SBATCH --error=logs/%x.%A_%a.log
#SBATCH --open-mode=append
#
# Extract FTP-1 domain archives in parallel, one array task per domain.
#
# Sequential extraction of the ~5.8 TB that remains would take most of a day; the domains are
# independent, so an array turns that into the wall time of the single largest one
# (VisuoTactile_D-WHEEL, 3.1 TB). Runs on cpu_dataprocessing, which has no walltime cap, so
# no GPU is held while tar runs. cpu_dataprocessing is restricted to other accounts here;
# cpu_dataprocessing is restricted to other accounts and cpu_long sits behind the shorter
# queues in priority, so `cpu` (24h cap, same 36 nodes) schedules fastest. tar is single-
# threaded, so a small CPU/memory ask backfills sooner and costs nothing in throughput.
#
# Idempotent: a domain that is already extracted, or whose archives are gone, is skipped. It
# is therefore safe to resubmit over a partially-finished corpus, and safe to run alongside
# another extraction loop as long as DOMAINS excludes whatever that loop holds.
#
# Archives are NOT deleted. Re-downloading 7.6 TB to recover from a bad extract is a far
# worse outcome than holding both copies on a 250 TB quota.
#
#   DOMAINS="a b c" sbatch --array=0-$((n-1)) scripts_exp_zarr/mot_jepa/extract_domains.sbatch

set -uo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
read -r -a DOMAIN_LIST <<< "${DOMAINS:?set DOMAINS}"

DOMAIN="${DOMAIN_LIST[${SLURM_ARRAY_TASK_ID}]}"
SRC="${DATASET_ROOT}/${DOMAIN}"
STAGING="${DATASET_ROOT}/.extract_${DOMAIN}_${SLURM_ARRAY_JOB_ID}"

echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID} domain=${DOMAIN} node=$(hostname)"

if [[ -n "$(ls -d "${SRC}"/*/ 2>/dev/null | grep -v extract)" ]]; then
  echo "[$(date -Is)] ${DOMAIN}: already extracted; skipping"
  exit 0
fi
if [[ -z "$(ls "${SRC}"/*.tar* 2>/dev/null)" ]]; then
  echo "[$(date -Is)] ${DOMAIN}: no archives present; skipping"
  exit 0
fi

rm -rf "${STAGING}"
echo "[$(date -Is)] ${DOMAIN}: validating parts and headroom"
if ! bash "${REPO_ROOT}/scripts/stage_ftp1_archive.sh" --check-only "${SRC}" "${STAGING}"; then
  echo "[$(date -Is)] ${DOMAIN}: FAILED validation"
  exit 1
fi

echo "[$(date -Is)] ${DOMAIN}: extracting $(du -sh "${SRC}" | cut -f1)"
START=${SECONDS}
if ! bash "${REPO_ROOT}/scripts/stage_ftp1_archive.sh" "${SRC}" "${STAGING}"; then
  echo "[$(date -Is)] ${DOMAIN}: FAILED extraction"
  rm -rf "${STAGING}"
  exit 1
fi

# The archives carry their own top-level directory, so the released layout is
# <root>/<domain>/<domain>/*.zarr. Preserve it: the survey and clip builder both tolerate the
# extra level, and matching what is already on disk keeps one convention.
moved=0
for inner in "${STAGING}"/*/; do
  [[ -d "${inner}" ]] || continue
  mv "${inner}" "${SRC}/" && moved=$((moved + 1))
done
rmdir "${STAGING}" 2>/dev/null || rm -rf "${STAGING}"

ZARRS=$(find "${SRC}" -maxdepth 3 -name "*.zarr" 2>/dev/null | wc -l)
echo "[$(date -Is)] ${DOMAIN}: DONE in $((SECONDS - START))s, moved ${moved} dir(s), ${ZARRS} zarr store(s)"
[[ ${ZARRS} -gt 0 ]] || { echo "[$(date -Is)] ${DOMAIN}: WARNING no zarr stores found"; exit 1; }
