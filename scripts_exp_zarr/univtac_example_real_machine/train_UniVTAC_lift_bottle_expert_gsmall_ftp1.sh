#!/usr/bin/env bash
# Generate a lift_bottle dataset config and delegate to the portable UniVTAC launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cache_root="${FTP1_CACHE_ROOT:-${REPO_ROOT}/.cache/ftp1}"
task_name="lift_bottle"

: "${UNIVTAC_DATA_ROOT:?Set UNIVTAC_DATA_ROOT to the directory containing UniVTAC task directories}"
: "${FTP1_PRETRAINED_CHECKPOINT:?Set FTP1_PRETRAINED_CHECKPOINT to the released FTP-1 checkpoint}"

task_dir="${UNIVTAC_DATA_ROOT}/${task_name}"
config_dir="${cache_root}/configs"
dataset_config_path="${FTP1_DATASET_CONFIG:-${config_dir}/dataset_univtac_${task_name}.json}"
mkdir -p "${config_dir}"

TASK_NAME="${task_name}" TASK_DIR="${task_dir}" DATASET_CONFIG_PATH="${dataset_config_path}" python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "datasets": [
        {
            "name": f"UniVTAC_{os.environ['TASK_NAME']}",
            "path": os.environ["TASK_DIR"],
            "use_trajectory_ratio": 1.0,
            "enabled": True,
        }
    ],
    "default_use_trajectory_ratio": 1.0,
    "description": "Generated UniVTAC FTP-1 fine-tuning config.",
}
Path(os.environ["DATASET_CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY

export FTP1_CACHE_ROOT="${cache_root}"
export FTP1_DATASET_CONFIG="${dataset_config_path}"
export FTP1_REPO_ID="${FTP1_REPO_ID:-UniVTAC_${task_name}_zscore_mix_ftp1}"
export FTP1_EXP_NAME="${FTP1_EXP_NAME:-FTP1_UniVTAC_${task_name}_expert_gsmall_ftp1}"

exec bash "${REPO_ROOT}/scripts_exp_zarr/univtac/train_univtac_example.sh"
