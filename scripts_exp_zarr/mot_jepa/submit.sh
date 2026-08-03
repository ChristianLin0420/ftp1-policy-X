#!/bin/bash
# Submit a MoT-JEPA pretraining chain.
#
# Usage:
#   REPO_ROOT=/path/to/ftp1-policy-X \
#   DATA_GLOB='/lustre/.../ftp1-clips/*/*.zarr' \
#   EXP_NAME=pilot01 NODES=2 CONFIG_NAME=mot_jepa_pilot \
#   bash scripts_exp_zarr/mot_jepa/submit.sh
#
# Optional: STAGE_SOURCE=<dir of *.zarr> stages to node-local NVMe and sets DATA_GLOB.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_NAME="${CONFIG_NAME:-mot_jepa_pilot}"
EXP_NAME="${EXP_NAME:-dev}"
NODES="${NODES:-2}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
PARTITION="${PARTITION:-batch,backfill}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/.cache/mot_jepa/runs}"
RUN_DIR="${RUN_ROOT}/${CONFIG_NAME}/${EXP_NAME}"

if [[ -z "${DATA_GLOB:-}" && -z "${STAGE_SOURCE:-}" ]]; then
  echo "error: set DATA_GLOB (or STAGE_SOURCE) to locate the *.zarr clip stores" >&2
  exit 1
fi

cd "${REPO_ROOT}"
# sbatch fails to open its output file if logs/ is missing, and the job then produces no
# output at all -- which looks exactly like a scheduler problem.
mkdir -p logs "${RUN_DIR}"

if [[ -f "${RUN_DIR}/DONE" ]]; then
  echo "run already finished at step $(cat "${RUN_DIR}/DONE"): ${RUN_DIR}" >&2
  echo "remove the DONE marker to extend it, or pick a new EXP_NAME" >&2
  exit 1
fi

export REPO_ROOT RUN_DIR CONFIG_NAME
export DATA_GLOB="${DATA_GLOB:-}"
export STAGE_SOURCE="${STAGE_SOURCE:-}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "repo      ${REPO_ROOT}"
echo "config    ${CONFIG_NAME}"
echo "run dir   ${RUN_DIR}"
echo "nodes     ${NODES} x 8 H100"
echo "data      ${DATA_GLOB:-<staged from ${STAGE_SOURCE}>}"
[[ -f "${RUN_DIR}/checkpoints/latest" ]] && echo "resuming  step $(cat "${RUN_DIR}/checkpoints/latest")"

sbatch \
  --nodes="${NODES}" \
  --time="${TIME_LIMIT}" \
  --partition="${PARTITION}" \
  --job-name="motjepa-${EXP_NAME}" \
  --export=ALL \
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/mot_jepa_pretrain.sbatch"
