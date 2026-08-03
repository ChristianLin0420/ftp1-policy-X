#!/bin/bash
# Wait for clip building to finish, verify the corpus, then launch pretraining.
#
# Written to run unattended: it makes no destructive change, it refuses to launch on an
# incomplete corpus, and every decision it takes is written to REPORT so the outcome can be
# read in one place afterwards.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:?}"
DATASET_ROOT="${DATASET_ROOT:?}"
CLIPS_ROOT="${CLIPS_ROOT:?}"
RUN_ROOT="${RUN_ROOT:?}"
WATCH_JOBS="${WATCH_JOBS:?}"          # comma-separated build job ids
EXP_NAME="${EXP_NAME:-pilot01}"
CONFIG_NAME="${CONFIG_NAME:-mot_jepa_pilot}"
NODES="${NODES:-4}"
REPORT="${REPORT:-${REPO_ROOT}/logs/auto_launch_report.txt}"
MIN_DOMAINS="${MIN_DOMAINS:-15}"

mkdir -p "$(dirname "${REPORT}")"
say() { echo "[$(date -Is)] $*" | tee -a "${REPORT}"; }

say "auto-launch started; watching build jobs ${WATCH_JOBS}"

while squeue -h -j "${WATCH_JOBS}" 2>/dev/null | grep -q .; do sleep 120; done
say "all build jobs finished"

# ---- verify the corpus before spending GPUs on it -------------------------------------
say ""
say "=== per-domain verification ==="
COMPLETE=0
INCOMPLETE=""
for d in "${DATASET_ROOT}"/*/; do
  n="$(basename "$d")"
  [[ "${n}" == .* ]] && continue
  src=$(find "$d" -maxdepth 3 -name "*.zarr" -not -path "*extract*" 2>/dev/null | wc -l)
  clip=$(find "${CLIPS_ROOT}/${n}" -maxdepth 2 -name "*.zarr" 2>/dev/null | wc -l)
  if [[ ${clip} -gt 0 ]]; then
    COMPLETE=$((COMPLETE + 1))
    say "$(printf '  OK       %-26s src=%-5d clips=%-5d' "${n}" "${src}" "${clip}")"
  else
    INCOMPLETE="${INCOMPLETE} ${n}"
    say "$(printf '  MISSING  %-26s src=%-5d clips=0' "${n}" "${src}")"
  fi
done
say "  --> ${COMPLETE} domains have derived clip stores"

if [[ ${COMPLETE} -lt ${MIN_DOMAINS} ]]; then
  say ""
  say "REFUSING TO LAUNCH: only ${COMPLETE}/${MIN_DOMAINS} domains built. Missing:${INCOMPLETE}"
  say "Rebuild those domains, then re-run this script."
  exit 1
fi

TOTAL_CLIPS=$(find "${CLIPS_ROOT}" -maxdepth 2 -name "*.zarr" | wc -l)
say ""
say "corpus ready: ${TOTAL_CLIPS} derived stores, $(du -sh "${CLIPS_ROOT}" 2>/dev/null | cut -f1)"

# ---- launch ---------------------------------------------------------------------------
say ""
say "=== launching pretraining ==="
cd "${REPO_ROOT}" || exit 1
mkdir -p logs

export REPO_ROOT RUN_ROOT CONFIG_NAME EXP_NAME NODES
export DATA_GLOB="${CLIPS_ROOT}/*/*.zarr"
export WANDB_MODE=offline

OUT=$(bash scripts_exp_zarr/mot_jepa/submit.sh 2>&1)
echo "${OUT}" | tee -a "${REPORT}"
JOB=$(echo "${OUT}" | grep -oP 'Submitted batch job \K\d+')

if [[ -z "${JOB}" ]]; then
  say "LAUNCH FAILED -- see above"
  exit 1
fi

say ""
say "PRETRAINING SUBMITTED: job ${JOB}"
say "  config   ${CONFIG_NAME} on ${NODES} nodes x 8 H100"
say "  run dir  ${RUN_ROOT}/${CONFIG_NAME}/${EXP_NAME}"
say "  log      ${REPO_ROOT}/logs/motjepa-${EXP_NAME}.${JOB}.log"
say "  W&B      offline; sync with scripts_exp_zarr/mot_jepa/sync_wandb.sh <run_dir>"
say ""
say "watch:  squeue -j ${JOB}"
say "gate :  probe/shortcut_ratio_to_chance > 1.5 means retrieval reads position -- halt"
