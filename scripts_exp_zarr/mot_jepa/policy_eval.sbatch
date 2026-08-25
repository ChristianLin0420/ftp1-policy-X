#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --job-name=motjepa-polieval
#SBATCH --output=logs/%x-%j.log
#
# Held-out action-chunk RMSE for one policy arm. Read-only.
#
#   REPO_ROOT=... RUN=<policy run dir> SNAP=<backbone> CLIPS=<univtac clips> OUT=<json> \
#     sbatch scripts_exp_zarr/mot_jepa/policy_eval.sbatch
#
# Set RIDGE=<ridge_policy.npz> instead of RUN to score the fitted ridge through this same path,
# which is how its number gets re-derived by the canonical metric rather than by the script that
# produced it.
set -uo pipefail
cd "${REPO_ROOT}"
export JAX_PLATFORMS=cpu USE_SWANLAB=false OMP_NUM_THREADS=8
if [[ -n "${RIDGE:-}" && -n "${RUN:-}" ]]; then
  echo "Set exactly one of RUN or RIDGE, not both" >&2
  exit 2
elif [[ -n "${RIDGE:-}" ]]; then
  POLICY=(--ridge "${RIDGE}")
elif [[ -n "${RUN:-}" ]]; then
  POLICY=(--run "${RUN}")
else
  echo "Set exactly one of RUN or RIDGE" >&2
  exit 2
fi

[[ -n "${CONFIG:-}" ]] && POLICY+=(--config "${CONFIG}")
BACKBONE=(--pretrained_run "${SNAP}")
[[ -n "${STEP:-}" ]] && BACKBONE+=(--pretrained_step "${STEP}")
SAMPLING=()
# A trained run derives these from its frozen run_config.json. Only pass an explicit value when
# the submitter supplied one; the Python evaluator rejects a mismatch. Ridge evaluation keeps its
# historical 10/17 defaults when neither value is supplied.
[[ -n "${HOLDOUT_MOD:-}" ]] && SAMPLING+=(--holdout_mod "${HOLDOUT_MOD}")
[[ -n "${INDEX_STEP:-}" ]] && SAMPLING+=(--index-step "${INDEX_STEP}")
[[ -n "${EXPECTED_CLIPS:-}" ]] && SAMPLING+=(--expected-clips "${EXPECTED_CLIPS}")
[[ -n "${NUM_SAMPLES:-}" ]] && SAMPLING+=(--num-samples "${NUM_SAMPLES}")
[[ -n "${EULER_STEPS:-}" ]] && SAMPLING+=(--euler-steps "${EULER_STEPS}")
[[ -n "${SEED:-}" ]] && SAMPLING+=(--seed "${SEED}")

srun "${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_policy_eval.py \
  "${POLICY[@]}" \
  "${BACKBONE[@]}" \
  --clips "${CLIPS}/*/*.zarr" \
  "${SAMPLING[@]}" \
  --out "${OUT}"
