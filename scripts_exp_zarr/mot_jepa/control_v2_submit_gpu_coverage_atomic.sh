#!/bin/bash
# Submit one atomic eight-GPU smoke allocation plus its independent CPU verification gate.
# Dry-run by default. This launcher never changes production collector jobs.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
QUAL_NODE="${QUAL_NODE:?set QUAL_NODE to the exact H100 node to qualify}"
QUAL_PARENT="${QUAL_PARENT:?set QUAL_PARENT to a fresh absolute output parent}"
REFERENCE_ROOT="${REFERENCE_ROOT:?set REFERENCE_ROOT to a fully verified production shard}"
MANIFEST="${MANIFEST:?set MANIFEST to a fresh absolute submission manifest}"
SUBMIT="${SUBMIT:-0}"
SBATCH_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"
ATOMIC_SBATCH="${SBATCH_DIR}/control_v2_gpu_coverage_atomic.sbatch"
GATE_SBATCH="${SBATCH_DIR}/control_v2_gpu_coverage_atomic_gate.sbatch"
COLLECT_SBATCH="${SBATCH_DIR}/control_v2_collect_batch.sbatch"
VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage_atomic.py"
BASE_VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage.py"

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
ATOMIC_SBATCH="${SBATCH_DIR}/control_v2_gpu_coverage_atomic.sbatch"
GATE_SBATCH="${SBATCH_DIR}/control_v2_gpu_coverage_atomic_gate.sbatch"
COLLECT_SBATCH="${SBATCH_DIR}/control_v2_collect_batch.sbatch"
VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage_atomic.py"
BASE_VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage.py"
LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
for path in "${ATOMIC_SBATCH}" "${GATE_SBATCH}" "${COLLECT_SBATCH}" "${VERIFIER}" "${BASE_VERIFIER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing atomic GPU-coverage source: ${path}" >&2
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
  echo "node=${QUAL_NODE} topology=one atomic allocation with 8 GPUs/128 CPU/1000G"
  echo "smokes=partition=batch_short time=00:30:00 no-requeue"
  echo "gate=afterok:<atomic-allocation> partition=cpu_short time=00:15:00 no-requeue"
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
    echo "atomic GPU-coverage submission failed; canceling partial jobs: ${job_ids[*]}" >&2
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
  echo "another atomic GPU-coverage submission owns ${manifest_lock}" >&2
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

raw_atomic="$(sbatch --parsable \
  --hold \
  --job-name=mot-v3-gpuqual-atomic \
  --partition=batch_short --time=00:30:00 --no-requeue --nodelist="${QUAL_NODE}" \
  --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-node=8 --mem=1000G \
  --export="ALL,REPO_ROOT=${REPO_ROOT},QUAL_NODE=${QUAL_NODE},QUAL_PARENT=${QUAL_PARENT},REFERENCE_ROOT=${REFERENCE_ROOT},QUAL_MANIFEST=${MANIFEST}" \
  "${ATOMIC_SBATCH}")"
atomic_job="$(_job_id "${raw_atomic}")"
job_ids+=("${atomic_job}")

port_base=$((12000 + (atomic_job % 5000) * 8))
slot_roots=()
slot_ranks=()
slot_ports=()
for slot in {0..7}; do
  slot_roots+=("${QUAL_PARENT}/lift_bottle_${atomic_job}_slot${slot}")
  slot_ranks+=("$((slot % 4))")
  slot_ports+=("$((port_base + slot))")
done
root_list="$(IFS=:; echo "${slot_roots[*]}")"
rank_list="$(IFS=:; echo "${slot_ranks[*]}")"
port_list="$(IFS=:; echo "${slot_ports[*]}")"
barrier="${QUAL_PARENT}/CONCURRENCY_BARRIER.json"
qual_output="${QUAL_PARENT}/GPU_COVERAGE.json"

raw_gate="$(sbatch --parsable \
  --hold \
  --dependency="afterok:${atomic_job}" \
  --partition=cpu_short --time=00:15:00 --no-requeue \
  --export="ALL,REPO_ROOT=${REPO_ROOT},QUAL_NODE=${QUAL_NODE},QUAL_PARENT=${QUAL_PARENT},REFERENCE_ROOT=${REFERENCE_ROOT},QUAL_MANIFEST=${MANIFEST},QUAL_ROOTS=${root_list},QUAL_RANKS=${rank_list},QUAL_PORTS=${port_list},QUAL_BARRIER=${barrier},QUAL_OUTPUT=${qual_output},QUAL_ATOMIC_JOB=${atomic_job},QUAL_LAUNCHER=${LAUNCHER}" \
  "${GATE_SBATCH}")"
gate_job="$(_job_id "${raw_gate}")"
job_ids+=("${gate_job}")

export COVERAGE_MANIFEST="${MANIFEST}" COVERAGE_REPO_ROOT="${REPO_ROOT}"
export COVERAGE_PARENT="${QUAL_PARENT}" COVERAGE_NODE="${QUAL_NODE}"
export COVERAGE_REFERENCE_ROOT="${REFERENCE_ROOT}" COVERAGE_OUTPUT="${qual_output}"
export COVERAGE_BARRIER="${barrier}" COVERAGE_ATOMIC_JOB="${atomic_job}" COVERAGE_GATE_JOB="${gate_job}"
export COVERAGE_ROOTS="${root_list}" COVERAGE_RANKS="${rank_list}" COVERAGE_PORTS="${port_list}"
export COVERAGE_LAUNCHER="${LAUNCHER}" COVERAGE_ATOMIC_SBATCH="${ATOMIC_SBATCH}"
export COVERAGE_GATE_SBATCH="${GATE_SBATCH}" COVERAGE_COLLECT_SBATCH="${COLLECT_SBATCH}"
export COVERAGE_VERIFIER="${VERIFIER}" COVERAGE_BASE_VERIFIER="${BASE_VERIFIER}"
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
roots = os.environ["COVERAGE_ROOTS"].split(":")
ranks = [int(value) for value in os.environ["COVERAGE_RANKS"].split(":")]
ports = [int(value) for value in os.environ["COVERAGE_PORTS"].split(":")]
job_id = os.environ["COVERAGE_ATOMIC_JOB"]
sources = {
    name: pathlib.Path(os.environ[env])
    for name, env in {
        "launcher": "COVERAGE_LAUNCHER",
        "atomic_sbatch": "COVERAGE_ATOMIC_SBATCH",
        "gate_sbatch": "COVERAGE_GATE_SBATCH",
        "collect_sbatch": "COVERAGE_COLLECT_SBATCH",
        "verifier": "COVERAGE_VERIFIER",
        "base_verifier": "COVERAGE_BASE_VERIFIER",
    }.items()
}
slots = [
    {"slot": slot, "root": root, "shard_rank": rank, "http_port": port}
    for slot, (root, rank, port) in enumerate(zip(roots, ranks, ports, strict=True))
]
payload = {
    "schema_version": 1,
    "status": "SUBMITTED",
    "mode": "atomic_8gpu",
    "node": os.environ["COVERAGE_NODE"],
    "parent": os.environ["COVERAGE_PARENT"],
    "reference_root": os.environ["COVERAGE_REFERENCE_ROOT"],
    "allocation": {
        "job_id": job_id,
        "partition": "batch_short",
        "time_limit": "00:30:00",
        "nodes": 1,
        "gpus": 8,
        "cpus": 128,
        "memory": "1000G",
    },
    "slots": slots,
    "barrier": os.environ["COVERAGE_BARRIER"],
    "gate": {
        "partition": "cpu_short",
        "time_limit": "00:15:00",
        "dependency": f"afterok:{job_id}",
        "job_id": os.environ["COVERAGE_GATE_JOB"],
        "output": os.environ["COVERAGE_OUTPUT"],
    },
    "job_ids": [job_id, os.environ["COVERAGE_GATE_JOB"]],
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
echo "submitted atomic GPU coverage node=${QUAL_NODE}: ${job_ids[*]}"
echo "manifest=${MANIFEST}"
echo "output=${qual_output}"
