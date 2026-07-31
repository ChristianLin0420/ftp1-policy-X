#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ================================ Core parameters ================================
cache_root="${FTP1_CACHE_ROOT:-${REPO_ROOT}/.cache/ftp1}"
repo_id="${FTP1_REPO_ID:-univtac_lift_bottle}"
exp_name="${FTP1_EXP_NAME:-univtac_lift_bottle_train}"
dataset_config_path="${FTP1_DATASET_CONFIG:-scripts_exp_zarr/univtac/dataset_univtac.json}"

checkpoint_base_dir="${FTP1_CHECKPOINT_BASE_DIR:-${cache_root}/checkpoints}"
assets_base_dir="${FTP1_ASSETS_BASE_DIR:-${cache_root}/assets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${cache_root}/openpi}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HYDRA_FULL_ERROR=1
# This launcher is the W&B variant; the sibling *_swanlab.sh launcher opts in
# to SwanLab explicitly.
export USE_SWANLAB="${USE_SWANLAB:-false}"

UV_RUN=(uv run)
if [ "${FTP1_UV_NO_SYNC:-false}" = "true" ]; then
  # Preserve local platform-specific wheel overrides (for example CUDA 12.8
  # wheels on Blackwell) instead of reconciling them back to uv.lock.
  UV_RUN+=(--no-sync)
fi

local_batch_size="${FTP1_LOCAL_BATCH_SIZE:-16}"
num_train_steps="${FTP1_NUM_TRAIN_STEPS:-20000}"
log_interval="${FTP1_LOG_INTERVAL:-100}"
val_interval="${FTP1_VAL_INTERVAL:-2000}"
val_ratio="${FTP1_VAL_RATIO:-0.1}"
save_interval="${FTP1_SAVE_INTERVAL:-10000}"
keep_period="${FTP1_KEEP_PERIOD:-10000}"
num_workers="${FTP1_NUM_WORKERS:-12}"
val_num_workers="${FTP1_VAL_NUM_WORKERS:-2}"
use_val_dataset="${FTP1_USE_VAL_DATASET:-true}"

action_down_sample_steps=1
norm_type="zscore"
norm_image_tactile_mode="channel_wise"
state_input_mode="adarms"
model_tactile_expert_variant="gemma_small"
proprioception_pose_rep="relative"
action_pose_rep="relative"
proprioception_joint_rep="abs"
action_joint_rep="mix"

pytorch_training_precision="bfloat16"
use_torch_compile="${FTP1_USE_TORCH_COMPILE:-true}"
load_t3_pretrained_checkpoint="${FTP1_LOAD_T3_PRETRAINED:-false}"
check_norm_params_snapshot="${FTP1_CHECK_NORM_SNAPSHOT:-true}"
lr_warmup_steps=500
lr_peak_lr="5e-5"
lr_decay_lr="5e-6"

optimizer_b1=0.9
optimizer_b2=0.95
optimizer_eps="1e-8"
optimizer_weight_decay="1e-10"
optimizer_clip_gradient_norm=1.0

pytorch_weight_path="${FTP1_PRETRAINED_CHECKPOINT:-${REPO_ROOT}/checkpoints/pretrain_ckpt/ftp1_pretrain_v0426_50kstep}"
use_wandb="${FTP1_ENABLE_TRACKING:-false}"

num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c . || echo "1")
if [ "${num_gpus}" -lt 1 ]; then
  num_gpus=1
fi
batch_size=$((local_batch_size * num_gpus))

mkdir -p "${checkpoint_base_dir}" "${assets_base_dir}" "${OPENPI_DATA_HOME}"
"${UV_RUN[@]}" python scripts/ftp1_preflight.py \
  --dataset-config "${dataset_config_path}" \
  --checkpoint "${pytorch_weight_path}"

norm_snapshot_path="${assets_base_dir}/ftp1/${repo_id}/norm_params_snapshot.json"
if [ "${check_norm_params_snapshot}" = "true" ] && [ ! -f "${norm_snapshot_path}" ]; then
  echo "Missing normalization snapshot: ${norm_snapshot_path}" >&2
  echo "Run compute_norm_stats_univtac_example.sh with the same FTP1_REPO_ID and FTP1_CACHE_ROOT first." >&2
  exit 1
fi

TRAIN_ARGS=(
  ftp1
  --exp_name=${exp_name}
  --repo_id=${repo_id}
  --data.repo-id=${repo_id}
  --checkpoint_base_dir=${checkpoint_base_dir}
  --assets_base_dir=${assets_base_dir}
  --dataset_config_path=${dataset_config_path}
  --batch_size=${batch_size}
  --action_down_sample_steps=${action_down_sample_steps}
  --num_train_steps=${num_train_steps}
  --log_interval=${log_interval}
  --val_interval=${val_interval}
  --val_ratio=${val_ratio}
  --save_interval=${save_interval}
  --keep_period=${keep_period}
  --num_workers=${num_workers}
  --val_num_workers=${val_num_workers}
  --norm_type=${norm_type}
  --norm_image_tactile_mode=${norm_image_tactile_mode}
  --pytorch_training_precision=${pytorch_training_precision}
  --model.state_input_mode=${state_input_mode}
  --model.tactile_expert_variant=${model_tactile_expert_variant}
  --model.use_tactile_input
  --proprioception_pose_rep=${proprioception_pose_rep}
  --action_pose_rep=${action_pose_rep}
  --proprioception_joint_rep=${proprioception_joint_rep}
  --action_joint_rep=${action_joint_rep}
  --lr_schedule.warmup_steps=${lr_warmup_steps}
  --lr_schedule.peak_lr=${lr_peak_lr}
  --lr_schedule.decay_steps=${num_train_steps}
  --lr_schedule.decay_lr=${lr_decay_lr}
  --optimizer.b1=${optimizer_b1}
  --optimizer.b2=${optimizer_b2}
  --optimizer.eps=${optimizer_eps}
  --optimizer.weight_decay=${optimizer_weight_decay}
  --optimizer.clip_gradient_norm=${optimizer_clip_gradient_norm}
  --lr_decay_till_end
  --ema_decay=0.99
)

if [ "${use_val_dataset}" = "true" ]; then
  TRAIN_ARGS+=(--use_val_dataset)
  TRAIN_ARGS+=(--create_train_val_split)
else
  TRAIN_ARGS+=(--no-use_val_dataset)
  TRAIN_ARGS+=(--no-create_train_val_split)
fi

if [ "${check_norm_params_snapshot}" = "true" ]; then
  TRAIN_ARGS+=(--check_norm_params_snapshot)
else
  TRAIN_ARGS+=(--no-check_norm_params_snapshot)
fi

if [ "${use_wandb}" = "true" ]; then
  TRAIN_ARGS+=(--wandb_enabled)
else
  TRAIN_ARGS+=(--no-wandb_enabled)
fi

if [ "${use_torch_compile}" = "true" ]; then
  TRAIN_ARGS+=(--use_torch_compile)
else
  TRAIN_ARGS+=(--no-use_torch_compile)
fi

# Fine-tuning checkpoints already contain the matching HPT tokenizer weights.
# Loading FoundationTactile's T3-large initialization first is both redundant
# and shape-incompatible with FTP-1's 768-wide image tokenizer.
if [ "${load_t3_pretrained_checkpoint}" = "true" ]; then
  TRAIN_ARGS+=(--model.tactile-tokenizer-config.load-t3-pretrained-checkpoint)
else
  TRAIN_ARGS+=(--model.tactile-tokenizer-config.no-load-t3-pretrained-checkpoint)
fi

TRAIN_ARGS+=(--no-gradient_checkpointing_enable)

if [ -n "${pytorch_weight_path}" ]; then
  TRAIN_ARGS+=(--pytorch_weight_path=${pytorch_weight_path})
fi

if [ "${num_gpus}" -gt 1 ]; then
  "${UV_RUN[@]}" python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
else
  "${UV_RUN[@]}" python scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
fi
