#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-h100-0044
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=mot-v2-paired20
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# One non-preemptible batch allocation and one exact 20-seed interval for both the student and
# official FTP1. This shard validates raw records, provenance, and student traces, but deliberately
# applies no n=20 task-success threshold. Five completed shards feed the CPU-only n=100 gate.
set -euo pipefail

PYCACHE_ROOT="${SLURM_TMPDIR:-/tmp}/mot_control_v2_empty_pycache_${SLURM_JOB_ID:-$$}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
ASSETS="${ASSETS:-/lustre/fsw/portfolios/edgeai/users/chrislin/tacex-assets}"
FTP1_OVERLAY="${FTP1_OVERLAY:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runtime/univtac_ftp1_py310_v1}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runtime/openpi-data}"
FTP1_CHECKPOINT="${FTP1_CHECKPOINT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-baselines/FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1/19999}"
V2_RUN="${V2_RUN:?set V2_RUN to a completed Control V2 stage}"
APPEND_MANIFEST="${APPEND_MANIFEST:?set APPEND_MANIFEST to the immutable paired submission manifest}"
APPEND_LAUNCHER="${APPEND_LAUNCHER:?set APPEND_LAUNCHER to the versioned paired append launcher}"
V2_FULL_STEP="$(cat "${V2_RUN}/DONE" 2>/dev/null || true)"
V2_STEP="${V2_STEP:-$(python3 -c 'import json,pathlib,sys; print(json.loads((pathlib.Path(sys.argv[1])/"BEST_VALIDATION.json").read_text())["step"])' "${V2_RUN}" 2>/dev/null || true)}"
TOTAL="${TOTAL:-20}"
START_SEED="${START_SEED:-1000000}"
EVAL_MODE=paired_shard
RUN_SECONDS="${RUN_SECONDS:-5400}"
PAIR_PARENT="${PAIR_PARENT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/mot_control_v2_eval}"
PAIR_ROOT="${PAIR_ROOT:-${PAIR_PARENT}/lift_bottle_s${START_SEED}_n${TOTAL}_${SLURM_JOB_ID}}"
PAIR_ROOT="${PAIR_ROOT//%j/${SLURM_JOB_ID}}"
PROVENANCE="${PAIR_ROOT}/provenance.json"
SHARD_MANIFEST="${PAIR_ROOT}/PAIRED_SHARD.json"
EVAL_NODE="${EVAL_NODE:-gpu-h100-0044}"
EVAL_NODE_ACTUAL="${SLURM_JOB_NODELIST:-UNKNOWN}"
EVAL_CONTRACT="${PAIR_ROOT}/evaluation_contract.json"
QUALIFIED_RUNTIME="${PAIR_ROOT}/qualified_runtime.json"
QUALIFIED_RUNTIME_AFTER="${PAIR_ROOT}/qualified_runtime_after.json"
GSMINI_ROBOT_ASSETS="${ASSETS}/Robots/Franka/GelSight_Mini"
GSMINI_SENSOR_ASSETS="${ASSETS}/Sensors/GelSight_Mini"

if [[ "${TOTAL}" -ne 20 ]]; then
  echo "paired evaluation shards are fixed to exactly 20 seeds" >&2
  exit 2
fi
case "${START_SEED}" in
  1000000|1000020|1000040|1000060|1000080) ;;
  *) echo "paired shard START_SEED must be one canonical n100 boundary" >&2; exit 2 ;;
esac
if [[ ! "${RUN_SECONDS}" =~ ^[1-9][0-9]*$ || "${RUN_SECONDS}" -gt 5400 ]]; then
  echo "RUN_SECONDS must be a positive integer no greater than 5400" >&2
  exit 2
fi
if [[ "${PAIR_ROOT}" == *"%"* ]]; then
  echo "PAIR_ROOT contains an unresolved Slurm placeholder: ${PAIR_ROOT}" >&2
  exit 2
fi
for path in "${APPEND_MANIFEST}" "${APPEND_LAUNCHER}"; do
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "missing or symlinked append-submission artifact ${path}" >&2
    exit 2
  fi
done
if [[ "$(realpath "${APPEND_LAUNCHER}")" != "$(realpath "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh")" ]]; then
  echo "APPEND_LAUNCHER is not the versioned paired append launcher" >&2
  exit 2
fi
RUN_OFFICIAL=1
if [[ ! "${START_SEED}" =~ ^[0-9]+$ || ! "${V2_STEP}" =~ ^[0-9]+$ ]]; then
  echo "START_SEED and V2_STEP must be non-negative integers" >&2
  exit 2
fi
if [[ -e "${PAIR_ROOT}" ]]; then
  echo "refuse existing paired evaluation root ${PAIR_ROOT}" >&2
  exit 2
fi
if [[ "${EVAL_NODE_ACTUAL}" != "${EVAL_NODE}" ]]; then
  echo "renderer-qualified evaluation requires node ${EVAL_NODE}, got ${EVAL_NODE_ACTUAL}" >&2
  exit 2
fi
if [[ -z "${V2_FULL_STEP}" || "$(cat "${V2_RUN}/checkpoints/latest" 2>/dev/null || true)" != "${V2_FULL_STEP}" ]]; then
  echo "V2 run is not fully completed (DONE/latest mismatch)" >&2
  exit 2
fi
V2_SELECTED_CHECKPOINT="${V2_RUN}/checkpoints/best_validation_${V2_STEP}"
required_paths=(
  "${V2_RUN}/run_config.json" "${V2_RUN}/artifact.json" "${V2_RUN}/normalization.json" \
  "${V2_RUN}/BEST_VALIDATION.json" "${V2_RUN}/TRAIN_SOURCE_PROVENANCE.json" \
  "${V2_RUN}/VALIDATION_SAMPLES.json" \
  "${V2_SELECTED_CHECKPOINT}" "${IMAGE}" "${GSMINI_ROBOT_ASSETS}" "${GSMINI_SENSOR_ASSETS}"
)
required_paths+=("${FTP1_CHECKPOINT}/model.safetensors" "${FTP1_CHECKPOINT}/train_config.json" "${FTP1_OVERLAY}/READY.json")
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing paired-evaluation artifact ${path}" >&2
    exit 2
  fi
done

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --verify "${V2_RUN}/TRAIN_SOURCE_PROVENANCE.json"

mapfile -t prepared_data_provenance < <(python3 - "${V2_RUN}" <<'PY'
import json, pathlib, sys

manifest = json.loads((pathlib.Path(sys.argv[1]) / "TRAIN_SOURCE_PROVENANCE.json").read_text())
markers = sorted(
    pathlib.Path(path)
    for path in manifest.get("files", {})
    if pathlib.Path(path).name == "store_content_provenance.json"
)
if not markers:
    raise SystemExit("training provenance lacks a derived-store content manifest")
for marker in markers:
    print(marker)
PY
)
if [[ "${#prepared_data_provenance[@]}" -eq 0 ]]; then
  echo "training provenance resolved no derived-store content manifest" >&2
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

mkdir -p "${REPO_ROOT}/logs" "${PAIR_ROOT}"
export REPO_ROOT IMAGE ASSETS FTP1_OVERLAY OPENPI_DATA_HOME FTP1_CHECKPOINT
export V2_RUN V2_STEP V2_FULL_STEP V2_SELECTED_CHECKPOINT TOTAL START_SEED RUN_SECONDS PAIR_ROOT SLURM_JOB_ID
export APPEND_MANIFEST APPEND_LAUNCHER
export EVAL_MODE RUN_OFFICIAL EVAL_NODE EVAL_NODE_ACTUAL EVAL_CONTRACT QUALIFIED_RUNTIME QUALIFIED_RUNTIME_AFTER

qualified_args=(
  --image "${IMAGE}"
  --assets "${ASSETS}"
  --repo-root "${REPO_ROOT}"
  --output "${QUALIFIED_RUNTIME}"
)
qualified_args+=(
  --official
  --overlay "${FTP1_OVERLAY}"
  --checkpoint "${FTP1_CHECKPOINT}"
  --openpi-data-home "${OPENPI_DATA_HOME}"
)
echo "[$(date -Is)] verifying byte-identical qualified runtime"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py" \
  "${qualified_args[@]}"

python3 - <<'PY'
import json, os, pathlib, time

payload = {
    "schema_version": 2,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "requested_node": os.environ["EVAL_NODE"],
    "actual_node_list": os.environ["EVAL_NODE_ACTUAL"],
    "mode": os.environ["EVAL_MODE"],
    "trials": int(os.environ["TOTAL"]),
    "start_seed": int(os.environ["START_SEED"]),
    "selected_step": int(os.environ["V2_STEP"]),
    "completed_step": int(os.environ["V2_FULL_STEP"]),
    "append_manifest": str(pathlib.Path(os.environ["APPEND_MANIFEST"]).resolve()),
    "append_launcher": str(pathlib.Path(os.environ["APPEND_LAUNCHER"]).resolve()),
    "created_unix": time.time(),
}
pathlib.Path(os.environ["EVAL_CONTRACT"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

heartbeat_pid=""
completed=0
_finish() {
  rc=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  rm -f "${PAIR_ROOT}/HEARTBEAT.json"
  if [[ "${completed}" -ne 1 ]]; then
    printf '{"status":"FAILED","job_id":"%s","exit_code":%d,"updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${rc}" "$(date -Is)" > "${PAIR_ROOT}/FAILED.json"
  fi
}
trap _finish EXIT

main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] || state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    printf '{"status":"RUNNING","job_id":"%s","node_list":"%s","scheduler_state":"%s","updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${EVAL_NODE_ACTUAL}" "${state:-UNKNOWN}" "$(date -Is)" > "${PAIR_ROOT}/HEARTBEAT.json"
    sleep 60
  done
) &
heartbeat_pid=$!

echo "[$(date -Is)] snapshotting paired artifact/source closure"
source_paths=(
  "${V2_RUN}/DONE"
  "${V2_RUN}/run_config.json"
  "${V2_RUN}/artifact.json"
  "${V2_RUN}/normalization.json"
  "${V2_RUN}/BEST_VALIDATION.json"
  "${V2_RUN}/TRAIN_SOURCE_PROVENANCE.json"
  "${V2_RUN}/VALIDATION_SAMPLES.json"
  "${V2_RUN}/checkpoints/latest"
  "${V2_SELECTED_CHECKPOINT}"
  "${EVAL_CONTRACT}"
  "${QUALIFIED_RUNTIME}"
  "${REPO_ROOT}/scripts/eval_motjepa_control_v2.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_acceptance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_merge_paired_eval.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop.sbatch"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_paired_shard.sbatch"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_inner.sh"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh"
  "${APPEND_MANIFEST}"
  "${REPO_ROOT}/UniVTAC/scripts"
  "${REPO_ROOT}/UniVTAC/task_config/demo.yml"
  "${REPO_ROOT}/UniVTAC/envs"
  "${REPO_ROOT}/UniVTAC/encoder"
  "${REPO_ROOT}/UniVTAC/policy"
  "${REPO_ROOT}/UniVTAC/assets"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex/tacex"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/robots"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/__init__.py"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/config/extension.toml"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_tasks/tacex_tasks"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc"
  "${REPO_ROOT}/src/openpi"
)
stat_only_paths=("${IMAGE}" "${ASSETS}")
stat_only_paths+=(
  "${FTP1_OVERLAY}"
  "${FTP1_CHECKPOINT}"
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
)
provenance_args=(--output "${PROVENANCE}")
for path in "${stat_only_paths[@]}"; do
  provenance_args+=(--stat-only "${path}")
done
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  "${provenance_args[@]}" "${source_paths[@]}"

echo "[$(date -Is)] evaluation job=${SLURM_JOB_ID} mode=${EVAL_MODE} v2=${V2_RUN}@best:${V2_STEP} completed:${V2_FULL_STEP} seeds=${START_SEED}..$((START_SEED + TOTAL - 1))"
srun --ntasks=1 --gpus-per-task=2 \
  --container-image="${IMAGE}" \
  --container-remap-root --container-writable \
  --container-mounts="/lustre:/lustre,${ASSETS}:${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data" \
  bash "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_inner.sh"

qualified_after_args=(
  --image "${IMAGE}"
  --assets "${ASSETS}"
  --repo-root "${REPO_ROOT}"
  --output "${QUALIFIED_RUNTIME_AFTER}"
)
qualified_after_args+=(
  --official
  --overlay "${FTP1_OVERLAY}"
  --checkpoint "${FTP1_CHECKPOINT}"
  --openpi-data-home "${OPENPI_DATA_HOME}"
)
echo "[$(date -Is)] requalifying byte-identical runtime after evaluation"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py" \
  "${qualified_after_args[@]}"
cmp --silent "${QUALIFIED_RUNTIME}" "${QUALIFIED_RUNTIME_AFTER}" || {
  echo "qualified evaluation runtime changed while policies ran" >&2
  exit 2
}

echo "[$(date -Is)] verifying source/artifact closure after both policies"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" --verify "${PROVENANCE}"
verify_prepared_data

echo "[$(date -Is)] validating exact paired20 raw records and traces without a success threshold"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_merge_paired_eval.py" \
  verify-shard \
  --shard-root "${PAIR_ROOT}" \
  --artifact "${V2_RUN}/artifact.json" \
  --start-seed "${START_SEED}" \
  --output "${SHARD_MANIFEST}" \
  --done "${PAIR_ROOT}/DONE"
completed=1
echo "[$(date -Is)] COMPLETE paired shard=${PAIR_ROOT} seeds=${START_SEED}..$((START_SEED + TOTAL - 1))"
