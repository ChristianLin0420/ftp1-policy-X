#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

cache_root="${FTP1_CACHE_ROOT:-${REPO_ROOT}/.cache/ftp1}"
repo_id="${FTP1_REPO_ID:-pretrain_small_example}"
exp_name="${FTP1_EXP_NAME:-pretrain_small_norm}"
dataset_config_path="${FTP1_DATASET_CONFIG:-scripts_exp_zarr/pretrain_small/data_config_pretrain_small.json}"

checkpoint_base_dir="${FTP1_CHECKPOINT_BASE_DIR:-${cache_root}/checkpoints}"
assets_base_dir="${FTP1_ASSETS_BASE_DIR:-${cache_root}/assets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${cache_root}/openpi}"

val_ratio=0.021
batch_size=256
action_down_sample_steps=1
norm_type="zscore"
norm_sample_ratio=0.1
norm_batch_size=64
norm_num_workers=12

mkdir -p "${checkpoint_base_dir}" "${assets_base_dir}" "${OPENPI_DATA_HOME}"
uv run python scripts/ftp1_preflight.py --dataset-config "${dataset_config_path}"

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
  --norm_sample_ratio=${norm_sample_ratio}
  --norm_batch_size=${norm_batch_size}
  --norm_num_workers=${norm_num_workers}
)

uv run python scripts/zarr_compute_norm_stats.py "${COMPUTE_ARGS[@]}"
