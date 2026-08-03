#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=motjepa-signalprobe
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Step (a) of the smoke ladder. Costs three GPU-minutes and answers the one question that
# genuinely varies by SLURM version and site configuration: does --signal=B:USR1@N actually
# reach the batch shell here, and does the backgrounded-srun + wait + trap pattern fire
# before the walltime kill?
#
# Run this BEFORE trusting a 40-job chain. If the trap line never appears in the log, the
# preemption path is not working and the file-flag fallback is the only thing standing
# between you and losing every step since the last periodic checkpoint.
#
#   mkdir -p logs && sbatch scripts_exp_zarr/mot_jepa/signal_probe.sbatch
#   # then, in the log, expect: "TRAP FIRED" well before the 3-minute walltime.

set -uo pipefail

PROBE_FLAG="${SLURM_SUBMIT_DIR:-$PWD}/logs/.signal_probe_${SLURM_JOB_ID}"
rm -f "${PROBE_FLAG}"

TRAP_FIRED=0
_on_usr1() {
  echo "[$(date -Is)] TRAP FIRED: SIGUSR1 reached the batch shell"
  # Exactly what the real launcher does: touch the flag the workers poll.
  touch "${PROBE_FLAG}"
  TRAP_FIRED=1
}
trap _on_usr1 USR1

echo "[$(date -Is)] probe start job=${SLURM_JOB_ID} attempt=${SLURM_RESTART_COUNT:-0}"
echo "[$(date -Is)] nproc=$(nproc) node=$(hostname)"
echo "[$(date -Is)] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"

# Same shape as the real launcher: background, then wait. The worker polls the flag file
# and exits cleanly, standing in for the trainer's checkpoint-and-exit.
export PROBE_FLAG
srun --wait=30 bash -c '
  for i in $(seq 1 200); do
    if [[ -f "${PROBE_FLAG}" ]]; then
      echo "  worker on $(hostname) saw the flag at step $i; exiting cleanly"
      exit 0
    fi
    echo "  step $i on $(hostname)"
    sleep 1
  done' &
SRUN_PID=$!

wait "${SRUN_PID}"
RC=$?
while [[ ${RC} -gt 128 ]]; do
  echo "[$(date -Is)] wait interrupted (rc=${RC}); re-waiting"
  wait "${SRUN_PID}"
  RC=$?
done

rm -f "${PROBE_FLAG}"
echo "[$(date -Is)] probe finished rc=${RC} trap_fired=${TRAP_FIRED}"
if [[ ${TRAP_FIRED} -eq 1 ]]; then
  echo "RESULT: PASS -- SIGUSR1 reached the batch shell, the trap fired, the worker saw the"
  echo "                flag and exited, and the batch script regained control before walltime."
else
  echo "RESULT: FAIL -- SIGUSR1 never reached the batch shell; do not rely on save-on-signal"
fi
