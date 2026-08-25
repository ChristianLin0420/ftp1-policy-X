#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=mot-control-v2-train
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#SBATCH --requeue

# Parameterized four-H100 stage runner.  The head, last-four adaptation, and recovery phases are
# independent immutable runs, so DDP trainability and optimizer topology never change in flight.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs}"
EXP_NAME="${EXP_NAME:?set EXP_NAME}"
RUN_DIR="${RUN_ROOT}/mot_jepa_control_v2/${EXP_NAME}"
PRETRAINED_RUN="${PRETRAINED_RUN:?set PRETRAINED_RUN to the qualified 100k backbone run}"
STORE_GLOB="${STORE_GLOB:?set STORE_GLOB to converted V3 lift_bottle Zarr stores}"
STEPS="${STEPS:?set STEPS (20000 head, 10000 adapt, or 5000 recovery)}"
BACKBONE_MODE="${BACKBONE_MODE:?set BACKBONE_MODE to frozen or last_blocks}"
INIT_RUN="${INIT_RUN:-}"
INIT_STEP="${INIT_STEP:-}"
MAX_REQUEUES="${MAX_REQUEUES:-4}"
SOURCE_PROVENANCE="${RUN_DIR}/TRAIN_SOURCE_PROVENANCE.json"

case "${BACKBONE_MODE}" in
  frozen|last_blocks) ;;
  *) echo "invalid BACKBONE_MODE=${BACKBONE_MODE}" >&2; exit 2 ;;
esac
if [[ ! -s "${RUN_DIR}/STATS_DONE" || ! -s "${RUN_DIR}/run_config.json" \
      || ! -s "${RUN_DIR}/artifact.json" || ! -s "${RUN_DIR}/normalization.json" ]]; then
  echo "stage lacks completed stats contract: ${RUN_DIR}" >&2
  exit 2
fi
if [[ -f "${RUN_DIR}/DONE" ]]; then
  if [[ "$(cat "${RUN_DIR}/DONE")" == "${STEPS}" ]]; then
    echo "[$(date -Is)] DONE already present at ${STEPS}; nothing to do"
    exit 0
  fi
  echo "DONE step differs from requested STEPS=${STEPS}" >&2
  exit 2
fi

export REPO_ROOT RUN_ROOT RUN_DIR EXP_NAME PRETRAINED_RUN STORE_GLOB STEPS BACKBONE_MODE INIT_RUN INIT_STEP
export SOURCE_PROVENANCE
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false USE_SWANLAB=false
export WANDB_DIR="${RUN_DIR}/wandb" WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${REPO_ROOT}/logs" "${WANDB_DIR}"
PYCACHE_ROOT="${RUN_DIR}/.empty_pycache_train_${SLURM_JOB_ID}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --verify "${RUN_DIR}/STATS_SOURCE_PROVENANCE.json"

mapfile -t prepared_data_provenance < <("${REPO_ROOT}/.venv/bin/python" - <<'PY'
import json, os, pathlib

manifest = json.loads((pathlib.Path(os.environ["RUN_DIR"]) / "STATS_SOURCE_PROVENANCE.json").read_text())
markers = sorted(
    pathlib.Path(path)
    for path in manifest.get("files", {})
    if pathlib.Path(path).name == "store_content_provenance.json"
)
if not markers:
    raise SystemExit("stats provenance lacks a derived-store content manifest")
for marker in markers:
    content = json.loads(marker.read_text())
    if not content.get("roots") or not content.get("files"):
        raise SystemExit(f"invalid derived-store content manifest: {marker}")
    print(marker)
PY
)
if [[ "${#prepared_data_provenance[@]}" -eq 0 ]]; then
  echo "stats provenance resolved no derived-store content manifest" >&2
  exit 2
fi
verify_prepared_data() {
  local marker
  for marker in "${prepared_data_provenance[@]}"; do
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
      --verify-stat-only "${marker}"
  done
}
verify_prepared_data

init_source_paths=()
if [[ -n "${INIT_RUN}" ]]; then
  mapfile -t init_source_paths < <("${REPO_ROOT}/.venv/bin/python" - "${INIT_RUN}" "${INIT_STEP}" <<'PY'
import json, pathlib, sys

run = pathlib.Path(sys.argv[1]).resolve()
requested_step = sys.argv[2]
best = json.loads((run / "BEST_VALIDATION.json").read_text())
step = int(best["step"])
if requested_step and int(requested_step) != step:
    raise SystemExit(f"INIT_STEP {requested_step} differs from selected step {step}")
checkpoint = run / "checkpoints" / f"best_validation_{step}"
if pathlib.Path(best["checkpoint"]).resolve() != checkpoint.resolve():
    raise SystemExit("INIT_RUN BEST_VALIDATION checkpoint path is not canonical")
paths = [
    run / "DONE",
    run / "STATS_DONE",
    run / "checkpoints/latest",
    run / "BEST_VALIDATION.json",
    run / "VALIDATION_SAMPLES.json",
    run / "run_config.json",
    run / "artifact.json",
    run / "normalization.json",
    run / "STATS_SOURCE_PROVENANCE.json",
    run / "TRAIN_SOURCE_PROVENANCE.json",
    checkpoint,
    *sorted(run.glob("STATS_JOB_DONE.*.json")),
    *sorted(run.glob("TRAIN_JOB_DONE.*.json")),
]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit(f"INIT_RUN closure is incomplete: {missing}")
for path in paths:
    print(path)
PY
  )
  if [[ "${#init_source_paths[@]}" -eq 0 ]]; then
    echo "failed to resolve immutable INIT_RUN closure: ${INIT_RUN}" >&2
    exit 2
  fi
fi

source_paths=(
  "${REPO_ROOT}/src/openpi"
  "${REPO_ROOT}/scripts/mot_jepa_train.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_train.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_train.sbatch"
  "${RUN_DIR}/run_config.json"
  "${RUN_DIR}/artifact.json"
  "${RUN_DIR}/normalization.json"
  "${RUN_DIR}/STATS_DONE"
  "${RUN_DIR}/STATS_SOURCE_PROVENANCE.json"
  "${PRETRAINED_RUN}/checkpoints/100000/metadata.pt"
  "${PRETRAINED_RUN}/checkpoints/100000/teacher_ema.pt"
  "${PRETRAINED_RUN}/checkpoints/100000/train_config.json"
  "${PRETRAINED_RUN}/checkpoints/latest"
  "${prepared_data_provenance[@]}"
  "${init_source_paths[@]}"
)
if [[ -f "${SOURCE_PROVENANCE}" ]]; then
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
    --verify "${SOURCE_PROVENANCE}"
else
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
    --output "${SOURCE_PROVENANCE}" "${source_paths[@]}"
fi

contract_before="$(sha256sum "${RUN_DIR}/run_config.json" "${RUN_DIR}/artifact.json" "${RUN_DIR}/normalization.json")"
heartbeat="${RUN_DIR}/HEARTBEAT.train.${SLURM_JOB_ID}.json"
heartbeat_pid=""
train_pid=""
completed=0
requeue_armed=0

_usr1() {
  echo "[$(date -Is)] SIGUSR1: requesting synchronized checkpoint"
  touch "${RUN_DIR}/PREEMPT_REQUEST"
  requeue_armed=1
}
trap _usr1 USR1

_finish() {
  rc=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  rm -f "${heartbeat}"
  if [[ "${completed}" -ne 1 && "${requeue_armed}" -ne 1 ]]; then
    printf '{"status":"FAILED","job_id":"%s","exit_code":%d,"updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${rc}" "$(date -Is)" > "${RUN_DIR}/FAILED.train.${SLURM_JOB_ID}.json"
  fi
}
trap _finish EXIT
rm -f "${RUN_DIR}/PREEMPT_REQUEST"

main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] || state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    latest="$(cat "${RUN_DIR}/checkpoints/latest" 2>/dev/null || echo 0)"
    printf '{"status":"RUNNING","job_id":"%s","scheduler_state":"%s","latest_step":%s,"updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${state:-UNKNOWN}" "${latest}" "$(date -Is)" > "${heartbeat}"
    sleep 60
  done
) &
heartbeat_pid=$!

args=(
  --run-root "${RUN_ROOT}"
  --exp-name "${EXP_NAME}"
  --pretrained-run "${PRETRAINED_RUN}"
  --pretrained-step 100000
  --store-glob "${STORE_GLOB}"
  --num-train-steps "${STEPS}"
  --backbone-train-mode "${BACKBONE_MODE}"
)
if [[ -n "${INIT_RUN}" ]]; then
  args+=(--init-run "${INIT_RUN}")
fi
if [[ -n "${INIT_STEP}" ]]; then
  args+=(--init-step "${INIT_STEP}")
fi

echo "[$(date -Is)] train job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} run=${RUN_DIR} mode=${BACKBONE_MODE} steps=${STEPS}"
cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
  scripts/mot_jepa_control_v2_train.py "${args[@]}" &
train_pid=$!

set +e
wait "${train_pid}"
train_rc=$?
drain=0
while [[ "${train_rc}" -gt 128 ]] && kill -0 "${train_pid}" 2>/dev/null && [[ "${drain}" -lt 60 ]]; do
  wait "${train_pid}"
  train_rc=$?
  drain=$((drain + 1))
done
set -e
echo "[$(date -Is)] torchrun rc=${train_rc} requeue_armed=${requeue_armed}"
if ! "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --verify "${SOURCE_PROVENANCE}"; then
  requeue_armed=0
  exit 1
fi
if ! verify_prepared_data; then
  requeue_armed=0
  exit 1
fi

if [[ -f "${RUN_DIR}/DONE" ]]; then
  latest="$(cat "${RUN_DIR}/checkpoints/latest")"
  if [[ "$(cat "${RUN_DIR}/DONE")" != "${STEPS}" || "${latest}" != "${STEPS}" ]]; then
    echo "DONE/latest mismatch: $(cat "${RUN_DIR}/DONE")/${latest}, expected ${STEPS}" >&2
    exit 1
  fi
  contract_after="$(sha256sum "${RUN_DIR}/run_config.json" "${RUN_DIR}/artifact.json" "${RUN_DIR}/normalization.json")"
  if [[ "${contract_before}" != "${contract_after}" ]]; then
    echo "frozen stage contract changed during training" >&2
    exit 1
  fi
  source_digest="$("${REPO_ROOT}/.venv/bin/python" -c 'import json,os; print(json.load(open(os.environ["SOURCE_PROVENANCE"]))["content_sha256"])')"
  selection="$(${REPO_ROOT}/.venv/bin/python - <<'PY'
import json, math, os, pathlib

run = pathlib.Path(os.environ["RUN_DIR"])
marker = json.loads((run / "BEST_VALIDATION.json").read_text())
step = int(marker["step"])
loss = float(marker["validation_action_loss"])
checkpoint = pathlib.Path(marker["checkpoint"]).resolve()
expected = (run / "checkpoints" / f"best_validation_{step}").resolve()
if (
    marker.get("selection_metric") != "validation_action_loss"
    or marker.get("selection_constraint") != "validation_safety_violation_values==0"
    or not math.isfinite(loss)
):
    raise SystemExit("invalid held-out selection metric")
counts = (
    marker.get("validation_joint_violation_values"),
    marker.get("validation_rate_violation_values"),
    marker.get("validation_safety_violation_values"),
)
if counts != (0.0, 0.0, 0.0):
    raise SystemExit(f"held-out selected checkpoint has safety violations: {counts}")
if marker.get("validation_joint_max_excess_ratio") != 0.0:
    raise SystemExit("held-out selected checkpoint exceeds a joint limit")
rate_max = float(marker.get("validation_rate_max_ratio", float("nan")))
if not math.isfinite(rate_max) or rate_max > 1.0:
    raise SystemExit(f"held-out selected checkpoint exceeds the rate limit: {rate_max}")
if checkpoint != expected or not checkpoint.is_dir():
    raise SystemExit(f"invalid held-out checkpoint {checkpoint}, expected {expected}")
print(f"{step}|{loss:.17g}")
PY
)"
  IFS='|' read -r selected_step selected_loss <<< "${selection}"
  printf '{"status":"DONE","job_id":"%s","completed_step":%s,"selected_step":%s,"validation_action_loss":%s,"source_content_sha256":"%s","updated":"%s"}\n' \
    "${SLURM_JOB_ID}" "${STEPS}" "${selected_step}" "${selected_loss}" "${source_digest}" "$(date -Is)" \
    > "${RUN_DIR}/TRAIN_JOB_DONE.${SLURM_JOB_ID}.json"
  completed=1
  echo "[$(date -Is)] DONE train=${RUN_DIR} step=${STEPS}"
  exit 0
fi

if [[ "${requeue_armed}" -eq 1 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge "${MAX_REQUEUES}" ]]; then
    echo "requeue cap ${MAX_REQUEUES} reached" >&2
    requeue_armed=0
    exit 1
  fi
  if [[ ! -s "${RUN_DIR}/checkpoints/latest" ]]; then
    echo "preemption requested but no checkpoint was written" >&2
    requeue_armed=0
    exit 1
  fi
  rm -f "${RUN_DIR}/PREEMPT_REQUEST"
  scontrol requeue "${SLURM_JOB_ID}"
  sleep 60
  exit 0
fi

exit "${train_rc}"
