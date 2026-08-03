#!/bin/bash
# Per-node launcher: stage data to local NVMe, compute the rendezvous, exec torchrun.
# One task per node (ntasks-per-node=1), so this runs exactly once per node.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?}"
RUN_DIR="${RUN_DIR:?}"
CONFIG_NAME="${CONFIG_NAME:-mot_jepa_pilot}"
SCRIPT_DIR="${SCRIPT_DIR:-${REPO_ROOT}/scripts_exp_zarr/mot_jepa}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
export MASTER_ADDR="${NODES[0]}"
# Derived from the JobID: stable within a job (so it survives requeue) and collision-free
# against other concurrent jobs on the same node.
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TOKENIZERS_PARALLELISM=false
# jax[cuda12] preallocates GPU memory on import and openpi pulls it in transitively.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# MANDATORY. The tracking shim defaults to SwanLab and pops `id`/`resume`
# (src/openpi/shared/wandb_compat.py:22,137-145), which would turn this requeue chain into
# one disjoint W&B run per job. The trainer also asserts on this.
export USE_SWANLAB=false
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${WANDB_DIR}"

echo "[$(hostname)] node ${SLURM_NODEID}/${SLURM_NNODES} cpus=$(nproc) master=${MASTER_ADDR}:${MASTER_PORT}"

# Sourced, not executed: it exports DATA_GLOB, which a subshell could not hand back.
if [[ -n "${STAGE_SOURCE:-}" ]]; then
  # shellcheck source=stage_to_raid.sh
  source "${SCRIPT_DIR}/stage_to_raid.sh"
fi
DATA_GLOB="${DATA_GLOB:?set DATA_GLOB (or STAGE_SOURCE, which sets it)}"

cd "${REPO_ROOT}"

# Rendezvous notes, since this repository has only ever used --standalone --nnodes=1:
#   c10d over the static store, because a slow node start (container pull, /raid staging)
#     blows through the static TCPStore's fixed timeout; c10d retries.
#   rdzv_id includes SLURM_RESTART_COUNT because requeue PRESERVES the JobID, so a bare
#     JobID would reuse the rendezvous namespace across attempts.
#   max_restarts=0 so a crashed rank fails the job fast into our requeue logic rather than
#     torchelastic restarting into a half-torn-down NCCL state.
exec "${REPO_ROOT}/.venv/bin/python" -m torch.distributed.run \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}" \
  --max_restarts=0 \
  scripts/mot_jepa_train.py "${CONFIG_NAME}" \
  --run_root "$(dirname "$(dirname "${RUN_DIR}")")" \
  --exp_name "$(basename "${RUN_DIR}")" \
  --data.store_glob "${DATA_GLOB}" \
  ${EXTRA_ARGS:-}
