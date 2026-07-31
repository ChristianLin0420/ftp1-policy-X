#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

UV_RUN=(uv run)
if [ "${FTP1_UV_NO_SYNC:-false}" = "true" ]; then
  UV_RUN+=(--no-sync)
fi

cache_root="${FTP1_CACHE_ROOT:-${REPO_ROOT}/.cache/ftp1}"
repo_id="${FTP1_REPO_ID:-univtac_lift_bottle}"
exp_name="${FTP1_EXP_NAME:-univtac_lift_bottle_norm}"
dataset_config_path="${FTP1_DATASET_CONFIG:-scripts_exp_zarr/univtac/dataset_univtac.json}"

checkpoint_base_dir="${FTP1_CHECKPOINT_BASE_DIR:-${cache_root}/checkpoints}"
assets_base_dir="${FTP1_ASSETS_BASE_DIR:-${cache_root}/assets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${cache_root}/openpi}"

val_ratio="${FTP1_VAL_RATIO:-0.1}"
batch_size="${FTP1_NORM_GLOBAL_BATCH_SIZE:-256}"
action_down_sample_steps=1
norm_type="zscore"
norm_image_tactile_mode="channel_wise"
state_input_mode="adarms"
model_tactile_expert_variant="gemma_small"
proprioception_pose_rep="relative"
action_pose_rep="relative"
proprioception_joint_rep="abs"
action_joint_rep="mix"
norm_sample_ratio="${FTP1_NORM_SAMPLE_RATIO:-1.0}"
norm_batch_size="${FTP1_NORM_BATCH_SIZE:-64}"
norm_num_workers="${FTP1_NORM_NUM_WORKERS:-12}"

mkdir -p "${checkpoint_base_dir}" "${assets_base_dir}" "${OPENPI_DATA_HOME}"
"${UV_RUN[@]}" python scripts/ftp1_preflight.py --dataset-config "${dataset_config_path}"

COMPUTE_ARGS=(
  ftp1
  --repo_id=${repo_id}
  --data.repo-id=${repo_id}
  --exp_name=${exp_name}
  --use_val_dataset
  --create_train_val_split
  --val_ratio=${val_ratio}
  --no-wandb_enabled
  --checkpoint_base_dir=${checkpoint_base_dir}
  --assets_base_dir=${assets_base_dir}
  --dataset_config_path=${dataset_config_path}
  --batch_size=${batch_size}
  --action_down_sample_steps=${action_down_sample_steps}
  --norm_type=${norm_type}
  --norm_image_tactile_mode=${norm_image_tactile_mode}
  --model.state_input_mode=${state_input_mode}
  --model.tactile_expert_variant=${model_tactile_expert_variant}
  --model.use_tactile_input
  --proprioception_pose_rep=${proprioception_pose_rep}
  --action_pose_rep=${action_pose_rep}
  --proprioception_joint_rep=${proprioception_joint_rep}
  --action_joint_rep=${action_joint_rep}
  --norm_sample_ratio=${norm_sample_ratio}
  --norm_batch_size=${norm_batch_size}
  --norm_num_workers=${norm_num_workers}
)

"${UV_RUN[@]}" python scripts/zarr_compute_norm_stats.py "${COMPUTE_ARGS[@]}"
