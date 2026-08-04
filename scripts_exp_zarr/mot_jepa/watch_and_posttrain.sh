#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=motjepa-watch
#SBATCH --output=logs/%x-%j.log
#SBATCH --open-mode=append
#
# Wait for a pretraining run to genuinely finish, snapshot its final backbone, and launch
# Stage 3's two arms against it.
#
# Gated on the DONE marker rather than on `--dependency=afterany`. The pretraining runs requeue
# under the SAME JobID, and if a dependency fired on the first requeue this would launch Stage 3
# against a mid-training checkpoint and produce a plausible number for the wrong backbone --
# silent, expensive, and exactly the failure mode that has bitten this project repeatedly. DONE
# is written only by the genuine-completion path in mot_jepa_train (`training complete at step
# N; not requeueing`), so it is the ground truth. A dependency may still be passed at submit
# time as a scheduling hint; the marker remains the correctness gate.
#
#   WATCH_RUN=/lustre/.../ftp1-runs/mot_jepa_pilot/probe2 \
#   TAG=probe2 REPO_ROOT=... CLIPS=... sbatch scripts_exp_zarr/mot_jepa/watch_and_posttrain.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
WATCH_RUN="${WATCH_RUN:?set WATCH_RUN -- the pretraining run directory to wait on}"
CLIPS="${CLIPS:?set CLIPS -- the derived clip store root}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
TAG="${TAG:-$(basename "${WATCH_RUN}")}"
TIMEOUT_MIN="${TIMEOUT_MIN:-900}"
POLL_SEC="${POLL_SEC:-120}"
STAGE3_STEPS="${STAGE3_STEPS:-4000}"
NODES="${NODES:-2}"

cd "${REPO_ROOT}" || exit 1
say() { echo "[$(date -Is)] $*"; }

say "watching ${WATCH_RUN}/DONE (timeout ${TIMEOUT_MIN} min, poll ${POLL_SEC}s)"

WAITED=0
while [[ ! -f "${WATCH_RUN}/DONE" ]]; do
  if (( WAITED >= TIMEOUT_MIN * 60 )); then
    say "TIMEOUT after ${TIMEOUT_MIN} min with no DONE marker."
    say "The run did not complete. NOT launching post-training -- a run that died must never"
    say "silently become a post-training result. Inspect ${WATCH_RUN} and resubmit by hand."
    exit 1
  fi
  sleep "${POLL_SEC}"
  WAITED=$(( WAITED + POLL_SEC ))
done

FINAL_STEP="$(cat "${WATCH_RUN}/DONE")"
say "DONE at step ${FINAL_STEP} after $(( WAITED / 60 )) min of waiting"

# ---- snapshot the backbone out of the source run's reach --------------------------------
# A finished run stops pruning, but pinning post-training to a live run's step is what killed
# three jobs earlier today (keep_last=3 plus every keep_period; the step was deleted underneath
# them). Copying is cheap insurance and makes the provenance explicit in the path.
SNAP="${RUN_ROOT}/backbones/${TAG}_s${FINAL_STEP}"
CKPT="${WATCH_RUN}/checkpoints/${FINAL_STEP}"
if [[ ! -f "${CKPT}/teacher_ema.pt" ]]; then
  say "ERROR: ${CKPT}/teacher_ema.pt missing; available: $(ls "${WATCH_RUN}/checkpoints" | tr '\n' ' ')"
  exit 1
fi
mkdir -p "${SNAP}/checkpoints/${FINAL_STEP}"
cp "${CKPT}/teacher_ema.pt" "${SNAP}/checkpoints/${FINAL_STEP}/"
cp "${CKPT}/metadata.pt" "${SNAP}/checkpoints/${FINAL_STEP}/" 2>/dev/null
cp "${CKPT}/train_config.json" "${SNAP}/checkpoints/${FINAL_STEP}/" 2>/dev/null
echo "${FINAL_STEP}" > "${SNAP}/checkpoints/latest"
say "snapshot -> ${SNAP}"

# ---- launch BOTH Stage 3 arms -----------------------------------------------------------
# One arm alone is uninterpretable. Both clear chance because the held-out gallery spans ten
# domains and recognising the domain is worth ~1.5x for free; the language contribution is the
# RATIO between the real and surrogate arms, so they are always submitted as a pair.
export REPO_ROOT RUN_ROOT
export TRAIN_SCRIPT=scripts/mot_jepa_posttrain.py
export DATA_GLOB="${CLIPS}/*/*.zarr"
export NODES WANDB_MODE="${WANDB_MODE:-online}"

COMMON="--pretrained_run ${SNAP} --pretrained_step ${FINAL_STEP}"
COMMON="${COMMON} --num_train_steps ${STAGE3_STEPS} --save_interval 1000"
COMMON="${COMMON} --stage3.instruction_emb ${CLIPS}/instruction_emb.npz"

for ARM in real surr; do
  EXTRA="${COMMON}"
  [[ "${ARM}" == "surr" ]] && EXTRA="${EXTRA} --stage3.surrogate_text"
  OUT=$(CONFIG_NAME=mot_jepa_stage3 EXP_NAME="s3_${TAG}_${ARM}" EXTRA_ARGS="${EXTRA}" \
        bash scripts_exp_zarr/mot_jepa/submit.sh 2>&1)
  say "arm=${ARM}: $(echo "${OUT}" | grep -oP 'Submitted batch job \K\d+' || echo "SUBMIT FAILED")"
  echo "${OUT}" | grep -vE "^(repo|config|script|run dir|nodes|data) " | head -3
done

say "done. Compare heldout_ratio_to_chance between s3_${TAG}_real and s3_${TAG}_surr;"
say "the real arm alone is not evidence of language grounding."
