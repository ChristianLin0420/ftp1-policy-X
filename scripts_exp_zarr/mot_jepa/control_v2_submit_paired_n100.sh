#!/bin/bash
# Append the exact paired n=100 evaluation to an existing n=20 gate. Dry-run unless SUBMIT=1.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
V2_RUN="${V2_RUN:?set V2_RUN to the Control V2 run consumed by N20_JOB}"
N20_JOB="${N20_JOB:?set N20_JOB to the existing student n=20 gate job}"
PAIR_PARENT="${PAIR_PARENT:?set PAIR_PARENT to the dedicated paired n100 output parent}"
EVAL_NODE="${EVAL_NODE:?set EVAL_NODE to the renderer-qualified node}"
SUBMIT="${SUBMIT:-0}"
SBATCH_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"
SHARD_SBATCH="${SBATCH_DIR}/control_v2_closedloop_paired_shard.sbatch"
MERGE_SBATCH="${SBATCH_DIR}/control_v2_merge_paired_eval.sbatch"
STARTS=(1000000 1000020 1000040 1000060 1000080)

if [[ "${SUBMIT}" != "0" && "${SUBMIT}" != "1" ]]; then
  echo "SUBMIT must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${N20_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N20_JOB must be one positive numeric Slurm job ID" >&2
  exit 2
fi
if [[ "${EVAL_NODE}" != "gpu-h100-0044" ]]; then
  echo "paired n100 is preregistered on renderer-qualified gpu-h100-0044, got ${EVAL_NODE}" >&2
  exit 2
fi
for name in REPO_ROOT V2_RUN PAIR_PARENT; do
  value="${!name}"
  if [[ "${value}" != /* || "${value}" == *","* || "${value}" == *":"* || "${value}" == *$'\n'* ]]; then
    echo "${name} must be an absolute Slurm-export-safe path" >&2
    exit 2
  fi
done
REPO_ROOT="$(realpath -m "${REPO_ROOT}")"
V2_RUN="$(realpath -m "${V2_RUN}")"
PAIR_PARENT="$(realpath -m "${PAIR_PARENT}")"
SBATCH_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"
SHARD_SBATCH="${SBATCH_DIR}/control_v2_closedloop_paired_shard.sbatch"
MERGE_SBATCH="${SBATCH_DIR}/control_v2_merge_paired_eval.sbatch"
for script in "${SHARD_SBATCH}" "${MERGE_SBATCH}"; do
  if [[ ! -f "${script}" ]]; then
    echo "missing paired n100 scheduler script: ${script}" >&2
    exit 2
  fi
done

MANIFEST="${MANIFEST:-${REPO_ROOT}/logs/control_v2_paired_n100_after_${N20_JOB}.json}"
if [[ "${MANIFEST}" != /* || "${MANIFEST}" == *$'\n'* ]]; then
  echo "MANIFEST must be an absolute path" >&2
  exit 2
fi
MANIFEST="$(realpath -m "${MANIFEST}")"
APPEND_LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
if [[ "${MANIFEST}" == *","* || "${MANIFEST}" == *":"* ]]; then
  echo "MANIFEST must be a Slurm-export-safe path without comma or colon" >&2
  exit 2
fi

if [[ "${SUBMIT}" != "1" ]]; then
  echo "DRY RUN: no jobs submitted. Set SUBMIT=1 to append paired n100."
  echo "v2_run=${V2_RUN}"
  echo "dependency=afterok:${N20_JOB}"
  echo "shards=5x20 starts=${STARTS[*]} node=${EVAL_NODE} partition=batch time=04:00:00"
  echo "merge=afterok:<five-shards> partition=cpu time=02:00:00 no-requeue"
  echo "pair_parent=${PAIR_PARENT}"
  echo "manifest=${MANIFEST}"
  exit 0
fi

mkdir -p "$(dirname "${MANIFEST}")"
if [[ -e "${MANIFEST}" ]]; then
  echo "refuse existing paired n100 submission manifest: ${MANIFEST}" >&2
  exit 2
fi

job_ids=()
submission_complete=0
manifest_written=0
manifest_lock="${MANIFEST}.lock"
lock_held=0
_cancel_partial_submission() {
  local rc=$?
  trap - EXIT
  if [[ "${submission_complete}" -ne 1 && "${#job_ids[@]}" -gt 0 ]]; then
    echo "paired n100 submission failed; canceling partial jobs: ${job_ids[*]}" >&2
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
  echo "another paired n100 submission owns ${manifest_lock}" >&2
  exit 2
fi
lock_held=1
if [[ -e "${MANIFEST}" ]]; then
  echo "refuse existing paired n100 submission manifest: ${MANIFEST}" >&2
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

shard_jobs=()
shard_roots=()
for start_seed in "${STARTS[@]}"; do
  raw_job="$(sbatch --parsable \
    --hold \
    --dependency="afterok:${N20_JOB}" \
    --partition=batch --time=04:00:00 --no-requeue --nodelist="${EVAL_NODE}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${V2_RUN},START_SEED=${start_seed},PAIR_PARENT=${PAIR_PARENT},PAIR_ROOT=${PAIR_PARENT}/lift_bottle_s${start_seed}_n20_%j,EVAL_NODE=${EVAL_NODE},RUN_SECONDS=5400,APPEND_MANIFEST=${MANIFEST},APPEND_LAUNCHER=${APPEND_LAUNCHER}" \
    "${SHARD_SBATCH}")"
  shard_job="$(_job_id "${raw_job}")"
  job_ids+=("${shard_job}")
  shard_jobs+=("${shard_job}")
  shard_roots+=("${PAIR_PARENT}/lift_bottle_s${start_seed}_n20_${shard_job}")
done

shard_dependency="$(IFS=:; echo "${shard_jobs[*]}")"
shard_root_list="$(IFS=:; echo "${shard_roots[*]}")"
merge_root_template="${PAIR_PARENT}/lift_bottle_s1000000_n100_%j"
raw_merge="$(sbatch --parsable \
  --hold \
  --dependency="afterok:${shard_dependency}" \
  --partition=cpu --time=02:00:00 --no-requeue \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${V2_RUN},SHARD_ROOTS=${shard_root_list},PAIR_PARENT=${PAIR_PARENT},PAIR_ROOT=${merge_root_template},APPEND_MANIFEST=${MANIFEST},APPEND_LAUNCHER=${APPEND_LAUNCHER}" \
  "${MERGE_SBATCH}")"
merge_job="$(_job_id "${raw_merge}")"
job_ids+=("${merge_job}")
merge_root="${PAIR_PARENT}/lift_bottle_s1000000_n100_${merge_job}"

export CONTROL_V2_APPEND_MANIFEST="${MANIFEST}"
export CONTROL_V2_APPEND_LAUNCHER="${APPEND_LAUNCHER}"
export CONTROL_V2_APPEND_REPO_ROOT="${REPO_ROOT}" CONTROL_V2_APPEND_V2_RUN="${V2_RUN}"
export CONTROL_V2_APPEND_N20_JOB="${N20_JOB}" CONTROL_V2_APPEND_PAIR_PARENT="${PAIR_PARENT}"
export CONTROL_V2_APPEND_EVAL_NODE="${EVAL_NODE}" CONTROL_V2_APPEND_STARTS="${STARTS[*]}"
export CONTROL_V2_APPEND_SHARD_JOBS="${shard_dependency}" CONTROL_V2_APPEND_SHARD_ROOTS="${shard_root_list}"
export CONTROL_V2_APPEND_MERGE_JOB="${merge_job}" CONTROL_V2_APPEND_MERGE_ROOT="${merge_root}"
export CONTROL_V2_APPEND_SHARD_SBATCH="${SHARD_SBATCH}" CONTROL_V2_APPEND_MERGE_SBATCH="${MERGE_SBATCH}"
python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import tempfile
import time


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split(name: str, separator: str) -> list[str]:
    return [value for value in os.environ[name].split(separator) if value]


manifest = pathlib.Path(os.environ["CONTROL_V2_APPEND_MANIFEST"])
launcher = pathlib.Path(os.environ["CONTROL_V2_APPEND_LAUNCHER"])
shard_script = pathlib.Path(os.environ["CONTROL_V2_APPEND_SHARD_SBATCH"])
merge_script = pathlib.Path(os.environ["CONTROL_V2_APPEND_MERGE_SBATCH"])
shard_jobs = split("CONTROL_V2_APPEND_SHARD_JOBS", ":")
shard_roots = split("CONTROL_V2_APPEND_SHARD_ROOTS", ":")
starts = [int(value) for value in split("CONTROL_V2_APPEND_STARTS", " ")]
payload = {
    "schema_version": 1,
    "status": "SUBMITTED",
    "v2_run": os.environ["CONTROL_V2_APPEND_V2_RUN"],
    "n20_job_id": os.environ["CONTROL_V2_APPEND_N20_JOB"],
    "n20_dependency": f"afterok:{os.environ['CONTROL_V2_APPEND_N20_JOB']}",
    "pair_parent": os.environ["CONTROL_V2_APPEND_PAIR_PARENT"],
    "eval_node": os.environ["CONTROL_V2_APPEND_EVAL_NODE"],
    "shard": {
        "partition": "batch",
        "time_limit": "04:00:00",
        "run_seconds_per_policy": 5400,
        "trials_each": 20,
        "start_seeds": starts,
        "job_ids": shard_jobs,
        "roots": shard_roots,
    },
    "merge": {
        "partition": "cpu",
        "time_limit": "02:00:00",
        "dependency": f"afterok:{':'.join(shard_jobs)}",
        "job_id": os.environ["CONTROL_V2_APPEND_MERGE_JOB"],
        "root": os.environ["CONTROL_V2_APPEND_MERGE_ROOT"],
    },
    "job_ids": [*shard_jobs, os.environ["CONTROL_V2_APPEND_MERGE_JOB"]],
    "sources": {
        "launcher": {"path": str(launcher), "sha256": sha256(launcher)},
        "shard_sbatch": {"path": str(shard_script), "sha256": sha256(shard_script)},
        "merge_sbatch": {"path": str(merge_script), "sha256": sha256(merge_script)},
    },
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
echo "submitted paired n100 after n20=${N20_JOB}: ${job_ids[*]}"
echo "manifest=${MANIFEST}"
echo "monitor: bash ${SBATCH_DIR}/control_v2_status.sh ${job_ids[*]}"
