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
#SBATCH --job-name=mot-v3-diagnostic
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# Diagnostic-only rollout of the exact numeric final head checkpoint after the expected
# no-qualified-validation terminal failure.  This script never runs the official comparator and
# never writes DONE, ACCEPTED, or BEST_VALIDATION markers.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
ASSETS="${ASSETS:-/lustre/fsw/portfolios/edgeai/users/chrislin/tacex-assets}"
V2_RUN="${V2_RUN:?set V2_RUN to the unqualified completed head run}"
V2_STEP="${V2_STEP:?set V2_STEP to the exact numeric final step}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:?set TRAIN_JOB_ID to the originating training job}"
DIAGNOSTIC_ROLE="${DIAGNOSTIC_ROLE:?set DIAGNOSTIC_ROLE to smoke or n20}"
DIAGNOSTIC_SUBMISSION_MANIFEST="${DIAGNOSTIC_SUBMISSION_MANIFEST:?set immutable submission manifest}"
TOTAL="${TOTAL:?set TOTAL to 1 or 20}"
START_SEED="${START_SEED:?set the diagnostic seed interval}"
RUN_SECONDS="${RUN_SECONDS:?set bounded evaluator runtime}"
PAIR_PARENT="${PAIR_PARENT:?set diagnostic output parent}"
PAIR_ROOT="${PAIR_ROOT:?set unique diagnostic output root}"
PAIR_ROOT="${PAIR_ROOT//%j/${SLURM_JOB_ID}}"
EVAL_NODE="${EVAL_NODE:-gpu-h100-0044}"
EVAL_NODE_ACTUAL="${SLURM_JOB_NODELIST:-UNKNOWN}"
CONTRACT="${PAIR_ROOT}/DIAGNOSTIC_CHECKPOINT_CONTRACT.json"
PROVENANCE="${PAIR_ROOT}/provenance.json"
QUALIFIED_RUNTIME="${PAIR_ROOT}/qualified_runtime.json"
QUALIFIED_RUNTIME_AFTER="${PAIR_ROOT}/qualified_runtime_after.json"
RESULT="${PAIR_ROOT}/diagnostic_result.json"
GSMINI_ROBOT_ASSETS="${ASSETS}/Robots/Franka/GelSight_Mini"
GSMINI_SENSOR_ASSETS="${ASSETS}/Sensors/GelSight_Mini"

case "${DIAGNOSTIC_ROLE}:${TOTAL}:${START_SEED}" in
  smoke:1:900000) ;;
  n20:20:900100) ;;
  *) echo "diagnostic roles are fixed to smoke 1@900000 or n20 20@900100" >&2; exit 2 ;;
esac
if [[ ! "${V2_STEP}" =~ ^[1-9][0-9]*$ || ! "${TRAIN_JOB_ID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "V2_STEP and TRAIN_JOB_ID must be positive integers" >&2
  exit 2
fi
if [[ ! "${RUN_SECONDS}" =~ ^[1-9][0-9]*$ || "${RUN_SECONDS}" -gt 10800 ]]; then
  echo "RUN_SECONDS must be a positive integer no greater than 10800" >&2
  exit 2
fi
if [[ "${EVAL_NODE}" != "gpu-h100-0044" || "${EVAL_NODE_ACTUAL}" != "${EVAL_NODE}" ]]; then
  echo "diagnostic renderer evaluation requires qualified node gpu-h100-0044, got ${EVAL_NODE_ACTUAL}" >&2
  exit 2
fi
if [[ "${PAIR_ROOT}" == *"%"* || "${PAIR_ROOT}" != "${PAIR_PARENT}"/* ]]; then
  echo "invalid or unresolved diagnostic output root ${PAIR_ROOT}" >&2
  exit 2
fi
if [[ -e "${PAIR_ROOT}" ]]; then
  echo "refuse existing diagnostic output root ${PAIR_ROOT}" >&2
  exit 2
fi
for path in "${DIAGNOSTIC_SUBMISSION_MANIFEST}" "${IMAGE}" "${GSMINI_ROBOT_ASSETS}" "${GSMINI_SENSOR_ASSETS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing diagnostic prerequisite ${path}" >&2
    exit 2
  fi
done

export REPO_ROOT V2_RUN V2_STEP TRAIN_JOB_ID DIAGNOSTIC_ROLE DIAGNOSTIC_SUBMISSION_MANIFEST
export TOTAL START_SEED RUN_SECONDS PAIR_PARENT PAIR_ROOT EVAL_NODE EVAL_NODE_ACTUAL
export CONTRACT PROVENANCE QUALIFIED_RUNTIME QUALIFIED_RUNTIME_AFTER RESULT IMAGE ASSETS

python3 - <<'PY'
import hashlib
import json
import os
import pathlib

manifest_path = pathlib.Path(os.environ["DIAGNOSTIC_SUBMISSION_MANIFEST"]).resolve()
manifest = json.loads(manifest_path.read_text())
role = os.environ["DIAGNOSTIC_ROLE"]
if manifest.get("schema_version") != 1 or manifest.get("status") != "SUBMITTED_DIAGNOSTIC_ONLY":
    raise SystemExit("invalid diagnostic submission manifest header")
expected = {
    "train_job_id": os.environ["TRAIN_JOB_ID"],
    "v2_run": str(pathlib.Path(os.environ["V2_RUN"]).resolve()),
    "step": int(os.environ["V2_STEP"]),
    "eval_node": os.environ["EVAL_NODE"],
}
for name, value in expected.items():
    if manifest.get(name) != value:
        raise SystemExit(f"diagnostic manifest {name} differs: {manifest.get(name)!r} != {value!r}")
job = manifest.get("jobs", {}).get(role, {})
if (
    job.get("job_id") != os.environ.get("SLURM_JOB_ID")
    or job.get("root") != str(pathlib.Path(os.environ["PAIR_ROOT"]).resolve())
    or job.get("trials") != int(os.environ["TOTAL"])
    or job.get("start_seed") != int(os.environ["START_SEED"])
):
    raise SystemExit("diagnostic manifest does not close this job identity")
for raw_path, expected_sha256 in manifest.get("source_sha256", {}).items():
    path = pathlib.Path(raw_path)
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"diagnostic source is missing or symlinked: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(f"diagnostic source changed after submission: {path}")
PY

mkdir -p "${REPO_ROOT}/logs" "${PAIR_ROOT}"
PYCACHE_ROOT="${PAIR_ROOT}/.empty_pycache_${SLURM_JOB_ID}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"

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
    printf '{"status":"INVALID","diagnostic_only":true,"job_id":"%s","exit_code":%d,"updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${rc}" "$(date -Is)" > "${PAIR_ROOT}/FAILED.json"
  fi
}
trap _finish EXIT

main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] || state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    printf '{"status":"RUNNING_DIAGNOSTIC_ONLY","job_id":"%s","scheduler_state":"%s","updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${state:-UNKNOWN}" "$(date -Is)" > "${PAIR_ROOT}/HEARTBEAT.json"
    sleep 60
  done
) &
heartbeat_pid=$!

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --verify "${V2_RUN}/TRAIN_SOURCE_PROVENANCE.json"

mapfile -t prepared_data_provenance < <(python3 - "${V2_RUN}" <<'PY'
import json
import pathlib
import sys

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
verify_prepared_data() {
  local marker
  for marker in "${prepared_data_provenance[@]}"; do
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
      --verify-stat-only "${marker}"
  done
}
verify_prepared_data

echo "[$(date -Is)] freezing explicit unqualified-final diagnostic contract"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_diagnostic.py" validate \
  --run "${V2_RUN}" --step "${V2_STEP}" --train-job-id "${TRAIN_JOB_ID}" --output "${CONTRACT}"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py" \
  --image "${IMAGE}" --assets "${ASSETS}" --repo-root "${REPO_ROOT}" --output "${QUALIFIED_RUNTIME}"

source_paths=(
  "${V2_RUN}/run_config.json"
  "${V2_RUN}/artifact.json"
  "${V2_RUN}/normalization.json"
  "${V2_RUN}/STATS_DONE"
  "${V2_RUN}/VALIDATION_SAMPLES.json"
  "${V2_RUN}/TRAIN_SOURCE_PROVENANCE.json"
  "${V2_RUN}/FAILED"
  "${V2_RUN}/FAILED.train.${TRAIN_JOB_ID}.json"
  "${V2_RUN}/checkpoints/latest"
  "${V2_RUN}/checkpoints/${V2_STEP}"
  "${CONTRACT}"
  "${QUALIFIED_RUNTIME}"
  "${DIAGNOSTIC_SUBMISSION_MANIFEST}"
  "${REPO_ROOT}/scripts/eval_motjepa_control_v2.py"
  "${REPO_ROOT}/scripts/eval_motjepa_control_v2_diagnostic.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_acceptance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_diagnostic.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_diagnostic.sbatch"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_diagnostic_inner.sh"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_submit_diagnostic.sh"
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
provenance_args=(--output "${PROVENANCE}" --stat-only "${IMAGE}" --stat-only "${ASSETS}")
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  "${provenance_args[@]}" "${source_paths[@]}"

echo "[$(date -Is)] launching DIAGNOSTIC_ONLY rollout role=${DIAGNOSTIC_ROLE}"
export MOT_CONTROL_V2_DIAGNOSTIC_OUTER_STORE_VERIFIED=1
srun --ntasks=1 --gpus-per-task=2 \
  --container-image="${IMAGE}" \
  --container-remap-root --container-writable \
  --container-mounts="/lustre:/lustre,${ASSETS}:${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data" \
  bash "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_diagnostic_inner.sh"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py" \
  --image "${IMAGE}" --assets "${ASSETS}" --repo-root "${REPO_ROOT}" --output "${QUALIFIED_RUNTIME_AFTER}"
cmp --silent "${QUALIFIED_RUNTIME}" "${QUALIFIED_RUNTIME_AFTER}" || {
  echo "qualified diagnostic runtime changed during rollout" >&2
  exit 2
}
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" --verify "${PROVENANCE}"
verify_prepared_data

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_diagnostic.py" aggregate \
  --run "${V2_RUN}" --step "${V2_STEP}" --train-job-id "${TRAIN_JOB_ID}" \
  --student-root "${PAIR_ROOT}/student/lift_bottle/student_diagnostic" \
  --qualified-runtime "${QUALIFIED_RUNTIME_AFTER}" --start-seed "${START_SEED}" --trials "${TOTAL}" \
  --contract "${CONTRACT}" --output "${RESULT}"

result_sha256="$(sha256sum "${RESULT}" | awk '{print $1}')"
printf 'diagnostic_only role=%s n=%s result_sha256=%s at %s\n' \
  "${DIAGNOSTIC_ROLE}" "${TOTAL}" "${result_sha256}" "$(date -Is)" > "${PAIR_ROOT}/DIAGNOSTIC_EVAL_COMPLETE"
completed=1
echo "[$(date -Is)] DIAGNOSTIC_ONLY complete result=${RESULT}"
