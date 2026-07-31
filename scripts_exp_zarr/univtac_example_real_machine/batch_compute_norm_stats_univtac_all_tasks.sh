#!/usr/bin/env bash
# bash scripts_exp_zarr/univtac/rtac50k/batch_compute_norm_stats_univtac_all_tasks.sh
export USE_SWANLAB="${USE_SWANLAB-false}"

#
# Batch compute norm stats for every UniVTAC task directory:
# - baseline_type=ftp1
#
# Repo naming rule:
#   UniVTAC_{task_name}_zscore_mix_ftp1

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
cache_root="${FTP1_CACHE_ROOT:-${repo_root}/.cache/ftp1}"
: "${UNIVTAC_DATA_ROOT:?Set UNIVTAC_DATA_ROOT to the directory containing task Zarr directories}"
data_root="${UNIVTAC_DATA_ROOT}"
checkpoint_base_dir="${FTP1_CHECKPOINT_BASE_DIR:-${cache_root}/checkpoints}"
assets_base_dir="${FTP1_ASSETS_BASE_DIR:-${cache_root}/assets}"
openpi_data_home="${OPENPI_DATA_HOME:-${cache_root}/openpi}"

val_ratio="${FTP1_VAL_RATIO:-0.1}"
batch_size=32
action_down_sample_steps=1
norm_type="zscore"
norm_image_tactile_mode="channel_wise"
proprioception_pose_rep="relative"
action_pose_rep="relative"
proprioception_joint_rep="abs"
action_joint_rep="mix"
norm_sample_ratio=1.0
norm_batch_size=64
norm_num_workers=12
exp_name="default"
contact_detection_threshold_k="${CONTACT_DETECTION_THRESHOLD_K:-3.0}"

export OPENPI_DATA_HOME="$openpi_data_home"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

tmp_cfg_root="$(mktemp -d /tmp/univtac_ftp1_norm_cfgs_XXXXXX)"
trap 'rm -rf "$tmp_cfg_root"' EXIT

if [ ! -d "$data_root" ]; then
    echo "Data root not found: $data_root"
    exit 1
fi

mapfile -t task_dirs < <(find "$data_root" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#task_dirs[@]}" -eq 0 ]; then
    echo "No task directories found under: $data_root"
    exit 1
fi

echo "Found ${#task_dirs[@]} task directories under: $data_root"

for task_dir in "${task_dirs[@]}"; do
    task_name="$(basename "$task_dir")"

    if ! find "$task_dir" -mindepth 1 -maxdepth 1 -type d -name "*.zarr" | grep -q .; then
        echo "[SKIP] $task_name: no *.zarr directory found"
        continue
    fi

    dataset_config_path="${tmp_cfg_root}/dataset_univtac_${task_name}.json"
    TASK_NAME="$task_name" TASK_DIR="$task_dir" DATASET_CONFIG_PATH="$dataset_config_path" python - <<'PY'
import json
import os

task_name = os.environ["TASK_NAME"]
task_dir = os.environ["TASK_DIR"]
dataset_config_path = os.environ["DATASET_CONFIG_PATH"]

payload = {
    "datasets": [
        {
            "name": f"UniVTAC_{task_name}",
            "path": task_dir,
            "use_trajectory_ratio": 1.0,
            "enabled": True,
        }
    ],
    "default_use_trajectory_ratio": 1.0,
    "description": "Auto-generated per-task dataset config for UniVTAC FTP1 norm stats.",
}

with open(dataset_config_path, "w") as f:
    json.dump(payload, f, indent=2)
PY

    repo_id="UniVTAC_${task_name}_zscore_mix_ftp1"
    echo "============================================================"
    echo "[TASK] $task_name | [BASELINE] ftp1 | [REPO] $repo_id"
    echo "============================================================"

    uv run python scripts/ftp1_preflight.py --dataset-config "$dataset_config_path"

    uv run python scripts/zarr_compute_norm_stats.py ftp1 \
        --repo_id="$repo_id" \
        --data.repo-id="$repo_id" \
        --exp_name="$exp_name" \
        --use_val_dataset \
        --create_train_val_split \
        --val_ratio="$val_ratio" \
        --no-wandb_enabled \
        --checkpoint_base_dir="$checkpoint_base_dir" \
        --assets_base_dir="$assets_base_dir" \
        --dataset_config_path="$dataset_config_path" \
        --norm_sample_ratio="$norm_sample_ratio" \
        --batch_size="$batch_size" \
        --norm_batch_size="$norm_batch_size" \
        --norm_num_workers="$norm_num_workers" \
        --action_down_sample_steps="$action_down_sample_steps" \
        --norm_type="$norm_type" \
        --norm_image_tactile_mode="$norm_image_tactile_mode" \
        --contact_detection_threshold_k="$contact_detection_threshold_k" \
        --no-skip_contact_detection \
        --proprioception_pose_rep="$proprioception_pose_rep" \
        --action_pose_rep="$action_pose_rep" \
        --proprioception_joint_rep="$proprioception_joint_rep" \
        --action_joint_rep="$action_joint_rep"
done

echo "All task norm stats finished."
