#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --job-name=motjepa-pretrain
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#
# MoT-JEPA pretraining, as a self-requeueing chain of 4-hour jobs.
#
# Differences from the plain template, and why:
#   --cpus-per-task=128  With ntasks-per-node=1 and torchrun forking 8 local ranks, a
#                        16-CPU cgroup pins 8 ranks plus ~100 dataloader workers onto a
#                        fraction of a core. It shows up as "GPUs at 15% util", not as an
#                        error. These nodes have 128 CPUs; take them.
#   --mem=0              Claims all 2 TB. zarr decompression plus page cache wants it, and
#                        the default per-node limit OOM-kills workers silently.
#   --exclusive          Guarantees /raid and the NIC are ours.
#
# %j is correct across requeues: scontrol requeue preserves the JobID, so with
# --open-mode=append every attempt appends to one continuous log.
#
# Submit through submit.sh, which creates the run directory and logs/ first.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
CONFIG_NAME="${CONFIG_NAME:-mot_jepa_pilot}"
MAX_REQUEUES="${MAX_REQUEUES:-60}"
SCRIPT_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"

export REPO_ROOT RUN_DIR CONFIG_NAME SCRIPT_DIR

echo "[$(date -Is)] job=${SLURM_JOB_ID} attempt=${SLURM_RESTART_COUNT:-0} nodes=${SLURM_NNODES} run=${RUN_DIR}"

if [[ -f "${RUN_DIR}/DONE" ]]; then
  echo "[$(date -Is)] DONE marker present ($(cat "${RUN_DIR}/DONE")); nothing to do"
  exit 0
fi

REQUEUE_ARMED=0

_on_usr1() {
  echo "[$(date -Is)] SIGUSR1 received on the batch shell; arming requeue"
  # The file flag is the PRIMARY mechanism. --signal=B:USR1@600 delivers only to this batch
  # shell, and `scancel --signal` reaches the job step (bash srun_node.sh) rather than
  # torchrun's eight worker children -- torchrun has historically not forwarded SIGUSR1.
  # Rank 0 polls this path every few steps and broadcasts the decision to every rank.
  touch "${RUN_DIR}/PREEMPT_REQUEST"
  # Secondary path, harmless if it does not reach the workers.
  scancel --signal=USR1 "${SLURM_JOB_ID}" 2>/dev/null || true
  REQUEUE_ARMED=1
}
trap _on_usr1 USR1

# Clear a stale flag left by an attempt that died before it could clean up.
rm -f "${RUN_DIR}/PREEMPT_REQUEST"

# SLURM >= 22.05 stopped propagating --cpus-per-task from sbatch to srun. Without this the
# CPU cliff described above happens anyway, even with the header correct.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-128}"

# srun must be BACKGROUNDED. A bash trap only runs between commands, so with srun in the
# foreground the shell is blocked and the handler would fire only after the walltime kill --
# far too late to checkpoint. Backgrounding makes `wait` the interruptible point.
srun --kill-on-bad-exit=1 --wait=120 bash "${SCRIPT_DIR}/srun_node.sh" &
SRUN_PID=$!

wait "${SRUN_PID}"
RC=$?
# An interrupted `wait` returns 128+signum without the child having exited, so re-wait.
while [[ ${RC} -gt 128 ]]; do
  wait "${SRUN_PID}"
  RC=$?
done

echo "[$(date -Is)] srun exited rc=${RC} requeue_armed=${REQUEUE_ARMED}"

if [[ -f "${RUN_DIR}/DONE" ]]; then
  echo "[$(date -Is)] training complete at step $(cat "${RUN_DIR}/DONE"); not requeueing"
  exit 0
fi

# The requeue decision keys on REQUEUE_ARMED, NOT on the exit code. A genuine crash
# (non-zero rc, no signal) must not enter an infinite loop burning 4 GPU-hours per attempt.
if [[ ${REQUEUE_ARMED} -eq 1 ]]; then
  if [[ ${SLURM_RESTART_COUNT:-0} -ge ${MAX_REQUEUES} ]]; then
    echo "[$(date -Is)] requeue cap ${MAX_REQUEUES} reached; stopping"
    exit 1
  fi
  rm -f "${RUN_DIR}/PREEMPT_REQUEST"
  echo "[$(date -Is)] requeueing job ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  # Stay alive until SLURM tears us down; exiting first can race the requeue.
  sleep 120
  exit 0
fi

exit "${RC}"
