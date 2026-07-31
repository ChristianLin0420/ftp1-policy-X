#!/usr/bin/env bash
# Enable SwanLab logging and delegate all training options to the portable base launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exp_name="${FTP1_EXP_NAME:-pretrain_small_train}"

: "${SWANLAB_API_KEY:?Set SWANLAB_API_KEY before enabling SwanLab logging}"
export USE_SWANLAB="true"
export FTP1_ENABLE_TRACKING="true"
export SWANLAB_SAVE_DIR="${SWANLAB_SAVE_DIR:-$HOME/.swanlab}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${REPO_ROOT}/swanlog/${exp_name}}"
mkdir -p "${SWANLAB_SAVE_DIR}" "${SWANLAB_LOG_DIR}"

exec bash "${SCRIPT_DIR}/train_ftp1_pretrain_small.sh"
