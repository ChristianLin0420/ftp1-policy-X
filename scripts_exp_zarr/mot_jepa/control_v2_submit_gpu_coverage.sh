#!/bin/bash
# Submit eight independent one-GPU smokes plus a distinct-eight-UUID CPU gate.
# Dry-run by default. This launcher never changes production collector jobs.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
QUAL_NODE="${QUAL_NODE:?set QUAL_NODE to the exact H100 node to qualify}"
QUAL_PARENT="${QUAL_PARENT:?set QUAL_PARENT to a fresh absolute output parent}"
REFERENCE_ROOT="${REFERENCE_ROOT:?set REFERENCE_ROOT to a fully verified production shard}"
MANIFEST="${MANIFEST:?set MANIFEST to a fresh absolute submission manifest}"
SUBMIT="${SUBMIT:-0}"
SBATCH_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"
COLLECT_SBATCH="${SBATCH_DIR}/control_v2_collect_batch.sbatch"
GATE_SBATCH="${SBATCH_DIR}/control_v2_gpu_coverage_gate.sbatch"
VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage.py"

if [[ "${SUBMIT}" != 0 && "${SUBMIT}" != 1 ]]; then
  echo "SUBMIT must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${QUAL_NODE}" =~ ^gpu-h100-[0-9]{4}$ ]]; then
  echo "QUAL_NODE must be one exact gpu-h100-NNNN node" >&2
  exit 2
fi
for name in REPO_ROOT QUAL_PARENT REFERENCE_ROOT MANIFEST; do
  value="${!name}"
  if [[ "${value}" != /* || "${value}" == *","* || "${value}" == *":"* || "${value}" == *$'\n'* ]]; then
    echo "${name} must be an absolute Slurm-export-safe path" >&2
    exit 2
  fi
done
REPO_ROOT="$(realpath -m "${REPO_ROOT}")"
QUAL_PARENT="$(realpath -m "${QUAL_PARENT}")"
REFERENCE_ROOT="$(realpath -m "${REFERENCE_ROOT}")"
MANIFEST="$(realpath -m "${MANIFEST}")"
SBATCH_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"
COLLECT_SBATCH="${SBATCH_DIR}/control_v2_collect_batch.sbatch"
GATE_SBATCH="${SBATCH_DIR}/control_v2_gpu_coverage_gate.sbatch"
VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage.py"
LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
for path in "${COLLECT_SBATCH}" "${GATE_SBATCH}" "${VERIFIER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing GPU-coverage source: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${REFERENCE_ROOT}" ]]; then
  echo "missing verified production reference: ${REFERENCE_ROOT}" >&2
  exit 2
fi
if [[ -e "${QUAL_PARENT}" || -e "${MANIFEST}" ]]; then
  echo "qualification parent and manifest must both be fresh" >&2
  exit 2
fi

if [[ "${SUBMIT}" != 1 ]]; then
  echo "DRY RUN: no jobs submitted. Set SUBMIT=1 after inspection."
  echo "node=${QUAL_NODE} topology=8 independent jobs x 1 GPU/16 CPU/125G"
  echo "smokes=partition=batch_short time=00:15:00 no-requeue"
  echo "gate=afterok:<eight-smokes> partition=cpu_short time=00:15:00 no-requeue"
  echo "parent=${QUAL_PARENT}"
  echo "manifest=${MANIFEST}"
  echo "reference=${REFERENCE_ROOT}"
  exit 0
fi

mkdir -p "$(dirname "${MANIFEST}")"
job_ids=()
submission_complete=0
manifest_written=0
manifest_lock="${MANIFEST}.lock"
lock_held=0
_cancel_partial_submission() {
  local rc=$?
  trap - EXIT
  if [[ "${submission_complete}" -ne 1 && "${#job_ids[@]}" -gt 0 ]]; then
    echo "GPU-coverage submission failed; canceling partial jobs: ${job_ids[*]}" >&2
    scancel "${job_ids[@]}" 2>/dev/null || true
  fi
  if [[ "${submission_complete}" -ne 1 && "${manifest_written}" -eq 1 ]]; then
    rm -f "${MANIFEST}"
  fi
  if [[ "${lock_held}" -eq 1 ]]; then
    rmdir "${manifest_lock}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap _cancel_partial_submission EXIT
if ! mkdir "${manifest_lock}" 2>/dev/null; then
  echo "another GPU-coverage submission owns ${manifest_lock}" >&2
  exit 2
fi
lock_held=1
if [[ -e "${MANIFEST}" || -e "${QUAL_PARENT}" ]]; then
  echo "qualification parent or manifest appeared during submission setup" >&2
  exit 2
fi

_job_id() {
  local raw=$1
  local job_id="${raw%%;*}"
  if [[ ! "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "sbatch returned a non-numeric job id: ${raw}" >&2
    return 2
  fi
  printf '%s\n' "${job_id}"
}

qual_jobs=()
qual_roots=()
for index in {0..7}; do
  rank=$((index % 4))
  raw_job="$(sbatch --parsable \
    --hold \
    --job-name=mot-v3-gpuqual \
    --partition=batch_short --time=00:15:00 --no-requeue --nodelist="${QUAL_NODE}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_PARENT=${QUAL_PARENT},COLLECTION_MODE=smoke,EPISODES=1,WORKERS=1,GLOBAL_EPISODES=4,SHARD_RANK=${rank},SHARD_COUNT=4" \
    "${COLLECT_SBATCH}")"
  job_id="$(_job_id "${raw_job}")"
  job_ids+=("${job_id}")
  qual_jobs+=("${job_id}")
  qual_roots+=("${QUAL_PARENT}/lift_bottle_${job_id}")
done

job_list="$(IFS=:; echo "${qual_jobs[*]}")"
root_list="$(IFS=:; echo "${qual_roots[*]}")"
qual_output="${QUAL_PARENT}/GPU_COVERAGE.json"
raw_gate="$(sbatch --parsable \
  --hold \
  --dependency="afterok:${job_list}" \
  --partition=cpu_short --time=00:15:00 --no-requeue \
  --export="ALL,REPO_ROOT=${REPO_ROOT},QUAL_NODE=${QUAL_NODE},QUAL_ROOTS=${root_list},QUAL_JOB_IDS=${job_list},REFERENCE_ROOT=${REFERENCE_ROOT},QUAL_OUTPUT=${qual_output},QUAL_MANIFEST=${MANIFEST},QUAL_LAUNCHER=${LAUNCHER}" \
  "${GATE_SBATCH}")"
gate_job="$(_job_id "${raw_gate}")"
job_ids+=("${gate_job}")

export COVERAGE_MANIFEST="${MANIFEST}" COVERAGE_REPO_ROOT="${REPO_ROOT}"
export COVERAGE_PARENT="${QUAL_PARENT}" COVERAGE_NODE="${QUAL_NODE}"
export COVERAGE_REFERENCE_ROOT="${REFERENCE_ROOT}" COVERAGE_OUTPUT="${qual_output}"
export COVERAGE_JOBS="${job_list}" COVERAGE_ROOTS="${root_list}" COVERAGE_GATE_JOB="${gate_job}"
export COVERAGE_LAUNCHER="${LAUNCHER}" COVERAGE_COLLECT_SBATCH="${COLLECT_SBATCH}"
export COVERAGE_GATE_SBATCH="${GATE_SBATCH}" COVERAGE_VERIFIER="${VERIFIER}"
python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import tempfile
import time


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = pathlib.Path(os.environ["COVERAGE_MANIFEST"])
jobs = os.environ["COVERAGE_JOBS"].split(":")
roots = os.environ["COVERAGE_ROOTS"].split(":")
sources = {
    name: pathlib.Path(os.environ[env])
    for name, env in {
        "launcher": "COVERAGE_LAUNCHER",
        "collect_sbatch": "COVERAGE_COLLECT_SBATCH",
        "gate_sbatch": "COVERAGE_GATE_SBATCH",
        "verifier": "COVERAGE_VERIFIER",
    }.items()
}
payload = {
    "schema_version": 1,
    "status": "SUBMITTED",
    "node": os.environ["COVERAGE_NODE"],
    "parent": os.environ["COVERAGE_PARENT"],
    "reference_root": os.environ["COVERAGE_REFERENCE_ROOT"],
    "smoke": {
        "partition": "batch_short",
        "time_limit": "00:15:00",
        "job_ids": jobs,
        "roots": roots,
        "shard_ranks": [index % 4 for index in range(8)],
        "episodes_each": 1,
    },
    "gate": {
        "partition": "cpu_short",
        "time_limit": "00:15:00",
        "dependency": f"afterok:{':'.join(jobs)}",
        "job_id": os.environ["COVERAGE_GATE_JOB"],
        "output": os.environ["COVERAGE_OUTPUT"],
    },
    "job_ids": [*jobs, os.environ["COVERAGE_GATE_JOB"]],
    "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
    "submitted_unix": time.time(),
}
fd, raw = tempfile.mkstemp(prefix=f".{manifest.name}.", dir=manifest.parent)
temporary = pathlib.Path(raw)
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
manifest_written=1

release_list="$(IFS=,; echo "${job_ids[*]}")"
scontrol release "${release_list}"
submission_complete=1
rmdir "${manifest_lock}"
lock_held=0
echo "submitted GPU coverage node=${QUAL_NODE}: ${job_ids[*]}"
echo "manifest=${MANIFEST}"
echo "output=${qual_output}"
