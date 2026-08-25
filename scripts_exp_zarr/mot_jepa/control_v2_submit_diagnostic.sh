#!/bin/bash
# Submit a non-official final-head smoke and n=20 rollout after any terminal state of training.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:?set the live V3 head training job ID}"
V2_RUN="${V2_RUN:?set the live V3 head run directory}"
V2_STEP="${V2_STEP:-20000}"
EVAL_NODE="${EVAL_NODE:-gpu-h100-0044}"
PAIR_PARENT="${PAIR_PARENT:?set a dedicated DIAGNOSTIC_ONLY output parent}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/logs/control_v2_diagnostic_after_${TRAIN_JOB_ID}.json}"
SUBMIT="${SUBMIT:-0}"
SBATCH_SCRIPT="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_diagnostic.sbatch"
LAUNCHER="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_submit_diagnostic.sh"

if [[ "${SUBMIT}" != "0" && "${SUBMIT}" != "1" ]]; then
  echo "SUBMIT must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${TRAIN_JOB_ID}" =~ ^[1-9][0-9]*$ || ! "${V2_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_JOB_ID and V2_STEP must be positive integers" >&2
  exit 2
fi
if [[ "${EVAL_NODE}" != "gpu-h100-0044" ]]; then
  echo "diagnostic rollout is pinned to renderer-qualified gpu-h100-0044" >&2
  exit 2
fi
for name in REPO_ROOT V2_RUN PAIR_PARENT MANIFEST; do
  value="${!name}"
  if [[ "${value}" != /* || "${value}" == *","* || "${value}" == *":"* || "${value}" == *$'\n'* ]]; then
    echo "${name} must be an absolute Slurm-export-safe path: ${value}" >&2
    exit 2
  fi
done
if [[ "${PAIR_PARENT}" != *diagnostic* || "${PAIR_PARENT}" != *unqualified* ]]; then
  echo "PAIR_PARENT must visibly contain both diagnostic and unqualified" >&2
  exit 2
fi
if [[ -e "${MANIFEST}" || -e "${MANIFEST}.lock" ]]; then
  echo "refuse existing diagnostic manifest or lock ${MANIFEST}" >&2
  exit 2
fi

source_files=(
  "${REPO_ROOT}/scripts/eval_motjepa_control_v2.py"
  "${REPO_ROOT}/scripts/eval_motjepa_control_v2_diagnostic.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_acceptance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_diagnostic.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py"
  "${SBATCH_SCRIPT}"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_closedloop_diagnostic_inner.sh"
  "${LAUNCHER}"
)
for path in "${source_files[@]}"; do
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "missing or symlinked diagnostic source ${path}" >&2
    exit 2
  fi
done

V2_RUN="$(realpath -m "${V2_RUN}")"
PAIR_PARENT="$(realpath -m "${PAIR_PARENT}")"
MANIFEST="$(realpath -m "${MANIFEST}")"
mkdir -p "${REPO_ROOT}/logs" "$(dirname "${MANIFEST}")"

echo "DIAGNOSTIC_ONLY / NOT_OFFICIAL"
echo "training_dependency=afterany:${TRAIN_JOB_ID}"
echo "v2_run=${V2_RUN}@numeric-final:${V2_STEP}"
echo "smoke=seed900000 n1 node=${EVAL_NODE} time=00:30:00"
echo "n20=seeds900100..900119 dependency=afterok:<smoke> node=${EVAL_NODE} time=02:00:00"
echo "pair_parent=${PAIR_PARENT}"
echo "manifest=${MANIFEST}"
if [[ "${SUBMIT}" != "1" ]]; then
  echo "DRY RUN; set SUBMIT=1 to submit"
  exit 0
fi

mkdir "${MANIFEST}.lock"
job_ids=()
committed=0
cleanup() {
  rc=$?
  if [[ "${committed}" -ne 1 && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
  fi
  if [[ "${committed}" -ne 1 ]]; then
    rm -f "${MANIFEST}"
  fi
  rmdir "${MANIFEST}.lock" 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT

_job_id() {
  local raw=$1
  local job_id="${raw%%;*}"
  if [[ ! "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "sbatch returned a non-numeric job id: ${raw}" >&2
    return 2
  fi
  printf '%s\n' "${job_id}"
}

raw_smoke_job="$(sbatch --parsable --hold --dependency="afterany:${TRAIN_JOB_ID}" --kill-on-invalid-dep=yes \
  --partition=batch --time=00:30:00 --no-requeue --nodelist="${EVAL_NODE}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${V2_RUN},V2_STEP=${V2_STEP},TRAIN_JOB_ID=${TRAIN_JOB_ID},DIAGNOSTIC_ROLE=smoke,TOTAL=1,START_SEED=900000,RUN_SECONDS=1200,PAIR_PARENT=${PAIR_PARENT},PAIR_ROOT=${PAIR_PARENT}/lift_bottle_s900000_n1_diagnostic_unqualified_head20k_%j,EVAL_NODE=${EVAL_NODE},DIAGNOSTIC_SUBMISSION_MANIFEST=${MANIFEST}" \
  "${SBATCH_SCRIPT}")"
smoke_job="$(_job_id "${raw_smoke_job}")"
job_ids+=("${smoke_job}")

raw_n20_job="$(sbatch --parsable --hold --dependency="afterok:${smoke_job}" --kill-on-invalid-dep=yes \
  --partition=batch --time=02:00:00 --no-requeue --nodelist="${EVAL_NODE}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${V2_RUN},V2_STEP=${V2_STEP},TRAIN_JOB_ID=${TRAIN_JOB_ID},DIAGNOSTIC_ROLE=n20,TOTAL=20,START_SEED=900100,RUN_SECONDS=5400,PAIR_PARENT=${PAIR_PARENT},PAIR_ROOT=${PAIR_PARENT}/lift_bottle_s900100_n20_diagnostic_unqualified_head20k_%j,EVAL_NODE=${EVAL_NODE},DIAGNOSTIC_SUBMISSION_MANIFEST=${MANIFEST}" \
  "${SBATCH_SCRIPT}")"
n20_job="$(_job_id "${raw_n20_job}")"
job_ids+=("${n20_job}")

export CONTROL_V2_DIAG_REPO_ROOT="${REPO_ROOT}"
export CONTROL_V2_DIAG_TRAIN_JOB="${TRAIN_JOB_ID}"
export CONTROL_V2_DIAG_V2_RUN="${V2_RUN}"
export CONTROL_V2_DIAG_STEP="${V2_STEP}"
export CONTROL_V2_DIAG_NODE="${EVAL_NODE}"
export CONTROL_V2_DIAG_PAIR_PARENT="${PAIR_PARENT}"
export CONTROL_V2_DIAG_MANIFEST="${MANIFEST}"
export CONTROL_V2_DIAG_SMOKE_JOB="${smoke_job}"
export CONTROL_V2_DIAG_N20_JOB="${n20_job}"
export CONTROL_V2_DIAG_SOURCE_FILES="$(printf '%s\n' "${source_files[@]}")"
python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import tempfile
import time

manifest = pathlib.Path(os.environ["CONTROL_V2_DIAG_MANIFEST"])
source_paths = [pathlib.Path(path) for path in os.environ["CONTROL_V2_DIAG_SOURCE_FILES"].splitlines()]
smoke_job = os.environ["CONTROL_V2_DIAG_SMOKE_JOB"]
n20_job = os.environ["CONTROL_V2_DIAG_N20_JOB"]
parent = pathlib.Path(os.environ["CONTROL_V2_DIAG_PAIR_PARENT"])
payload = {
    "schema_version": 1,
    "status": "SUBMITTED_DIAGNOSTIC_ONLY",
    "warning": "NOT_OFFICIAL: validation qualification is bypassed only for final-head rollout measurement",
    "train_job_id": os.environ["CONTROL_V2_DIAG_TRAIN_JOB"],
    "train_dependency": f"afterany:{os.environ['CONTROL_V2_DIAG_TRAIN_JOB']}",
    "v2_run": str(pathlib.Path(os.environ["CONTROL_V2_DIAG_V2_RUN"]).resolve()),
    "step": int(os.environ["CONTROL_V2_DIAG_STEP"]),
    "eval_node": os.environ["CONTROL_V2_DIAG_NODE"],
    "pair_parent": str(parent.resolve()),
    "jobs": {
        "smoke": {
            "job_id": smoke_job,
            "dependency": f"afterany:{os.environ['CONTROL_V2_DIAG_TRAIN_JOB']}",
            "trials": 1,
            "start_seed": 900000,
            "root": str((parent / f"lift_bottle_s900000_n1_diagnostic_unqualified_head20k_{smoke_job}").resolve()),
        },
        "n20": {
            "job_id": n20_job,
            "dependency": f"afterok:{smoke_job}",
            "trials": 20,
            "start_seed": 900100,
            "root": str((parent / f"lift_bottle_s900100_n20_diagnostic_unqualified_head20k_{n20_job}").resolve()),
        },
    },
    "job_ids": [smoke_job, n20_job],
    "source_sha256": {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
    },
    "created_unix": time.time(),
}
manifest.parent.mkdir(parents=True, exist_ok=True)
fd, raw_temporary = tempfile.mkstemp(prefix=f".{manifest.name}.", dir=manifest.parent)
temporary = pathlib.Path(raw_temporary)
try:
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest)
finally:
    temporary.unlink(missing_ok=True)
PY

scontrol release "${smoke_job},${n20_job}"
committed=1
echo "submitted DIAGNOSTIC_ONLY jobs smoke=${smoke_job} n20=${n20_job}"
