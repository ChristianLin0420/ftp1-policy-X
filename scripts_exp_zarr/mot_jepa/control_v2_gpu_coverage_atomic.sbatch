#!/bin/bash
# Run eight one-GPU Control V2 smokes inside one atomic full-node allocation.
# The independent CPU gate verifies every committed artifact after this job exits.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch_short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --mem=1000G
#SBATCH --job-name=mot-v3-gpuqual-atomic
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
QUAL_NODE="${QUAL_NODE:?set QUAL_NODE to the exact node under qualification}"
QUAL_PARENT="${QUAL_PARENT:?set QUAL_PARENT to the fresh atomic output parent}"
REFERENCE_ROOT="${REFERENCE_ROOT:?set REFERENCE_ROOT to a fully verified production shard}"
QUAL_MANIFEST="${QUAL_MANIFEST:?set QUAL_MANIFEST to the immutable submission manifest}"
COLLECT_SBATCH="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch"
ATOMIC_SBATCH="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_gpu_coverage_atomic.sbatch"
GATE_SBATCH="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_gpu_coverage_atomic_gate.sbatch"
LAUNCHER="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_submit_gpu_coverage_atomic.sh"
VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage_atomic.py"
BASE_VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage.py"

for path in "${REPO_ROOT}" "${REFERENCE_ROOT}"; do
  if [[ "${path}" != /* || ! -d "${path}" ]]; then
    echo "required directory is missing or non-absolute: ${path}" >&2
    exit 2
  fi
done
for path in "${QUAL_PARENT}" "${QUAL_MANIFEST}"; do
  if [[ "${path}" != /* ]]; then
    echo "required path is non-absolute: ${path}" >&2
    exit 2
  fi
done
for path in "${COLLECT_SBATCH}" "${ATOMIC_SBATCH}" "${GATE_SBATCH}" "${LAUNCHER}" "${VERIFIER}" \
  "${BASE_VERIFIER}" "${QUAL_MANIFEST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required atomic qualification source is missing: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${QUAL_PARENT}" ]]; then
  echo "refuse existing atomic qualification parent: ${QUAL_PARENT}" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "${QUAL_NODE}" || "${SLURM_JOB_NUM_NODES:-}" != 1 ]]; then
  echo "atomic qualification did not receive the exact requested node" >&2
  exit 2
fi

port_base=$((12000 + (SLURM_JOB_ID % 5000) * 8))
roots=()
ranks=()
ports=()
for slot in {0..7}; do
  roots+=("${QUAL_PARENT}/lift_bottle_${SLURM_JOB_ID}_slot${slot}")
  ranks+=("$((slot % 4))")
  ports+=("$((port_base + slot))")
done
root_list="$(IFS=:; echo "${roots[*]}")"
rank_list="$(IFS=:; echo "${ranks[*]}")"
port_list="$(IFS=:; echo "${ports[*]}")"
barrier="${QUAL_PARENT}/CONCURRENCY_BARRIER.json"

"${REPO_ROOT}/.venv/bin/python" - \
  "${QUAL_MANIFEST}" "${SLURM_JOB_ID}" "${QUAL_NODE}" "${QUAL_PARENT}" "${REFERENCE_ROOT}" \
  "${root_list}" "${rank_list}" "${port_list}" "${barrier}" \
  "${LAUNCHER}" "${ATOMIC_SBATCH}" "${GATE_SBATCH}" "${COLLECT_SBATCH}" "${VERIFIER}" "${BASE_VERIFIER}" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    manifest_arg,
    allocation_job,
    node,
    parent,
    reference,
    roots_arg,
    ranks_arg,
    ports_arg,
    barrier,
    launcher_arg,
    atomic_arg,
    gate_arg,
    collect_arg,
    verifier_arg,
    base_verifier_arg,
) = sys.argv[1:]
manifest_path = pathlib.Path(manifest_arg)
payload = json.loads(manifest_path.read_text())
roots = roots_arg.split(":")
ranks = [int(value) for value in ranks_arg.split(":")]
ports = [int(value) for value in ports_arg.split(":")]
expected_slots = [
    {"slot": slot, "root": root, "shard_rank": rank, "http_port": port}
    for slot, (root, rank, port) in enumerate(zip(roots, ranks, ports, strict=True))
]
expected_sources = {
    "launcher": pathlib.Path(launcher_arg).resolve(),
    "atomic_sbatch": pathlib.Path(atomic_arg).resolve(),
    "gate_sbatch": pathlib.Path(gate_arg).resolve(),
    "collect_sbatch": pathlib.Path(collect_arg).resolve(),
    "verifier": pathlib.Path(verifier_arg).resolve(),
    "base_verifier": pathlib.Path(base_verifier_arg).resolve(),
}
if payload.get("schema_version") != 1 or payload.get("status") != "SUBMITTED":
    raise SystemExit("atomic submission manifest has an invalid schema/status")
if payload.get("mode") != "atomic_8gpu" or payload.get("node") != node:
    raise SystemExit("atomic submission manifest mode/node changed")
if payload.get("parent") != parent or payload.get("reference_root") != reference:
    raise SystemExit("atomic submission manifest parent/reference changed")
allocation = payload.get("allocation", {})
expected_allocation = {
    "job_id": allocation_job,
    "partition": "batch_short",
    "time_limit": "00:30:00",
    "nodes": 1,
    "gpus": 8,
    "cpus": 128,
    "memory": "1000G",
}
if allocation != expected_allocation or payload.get("slots") != expected_slots:
    raise SystemExit("atomic allocation or slot mapping changed")
if payload.get("barrier") != barrier:
    raise SystemExit("atomic concurrency barrier path changed")
sources = payload.get("sources", {})
if set(sources) != set(expected_sources):
    raise SystemExit("atomic submission manifest source set changed")
for name, path in expected_sources.items():
    record = sources[name]
    if pathlib.Path(record.get("path", "")).resolve() != path or record.get("sha256") != sha256(path):
        raise SystemExit(f"atomic qualification source changed: {name}")
print(json.dumps({"manifest_verified": True, "allocation_job": allocation_job, "node": node}))
PY

visible_uuids="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | awk 'NF {print}')"
visible_count="$(printf '%s\n' "${visible_uuids}" | awk 'NF {count++} END {print count+0}')"
unique_count="$(printf '%s\n' "${visible_uuids}" | sort -u | awk 'NF {count++} END {print count+0}')"
if [[ "${visible_count}" -ne 8 || "${unique_count}" -ne 8 ]]; then
  echo "atomic allocation must expose exactly eight distinct physical GPUs; found ${visible_count}/${unique_count}" >&2
  exit 2
fi

mkdir -p "$(dirname "${QUAL_PARENT}")"
mkdir "${QUAL_PARENT}"
export QUAL_PARENT QUAL_NODE QUAL_MANIFEST
child_pids=()
completed=0
_finish() {
  rc=$?
  trap - EXIT
  if [[ "${completed}" -ne 1 ]]; then
    for pid in "${child_pids[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    for pid in "${child_pids[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
    export ATOMIC_EXIT_CODE="${rc}"
    "${REPO_ROOT}/.venv/bin/python" - <<'PY'
import json
import os
import pathlib
import tempfile
import time

parent = pathlib.Path(os.environ["QUAL_PARENT"])
target = parent / "ATOMIC_FAILED.json"
payload = {
    "status": "FAILED",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": os.environ.get("QUAL_NODE"),
    "exit_code": int(os.environ["ATOMIC_EXIT_CODE"]),
    "updated_unix": time.time(),
}
fd, raw = tempfile.mkstemp(prefix=".atomic_failed.", dir=parent)
with os.fdopen(fd, "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(raw, target)
PY
  fi
  exit "${rc}"
}
trap _finish EXIT

for slot in {0..7}; do
  (
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
    unset SLURM_CPUS_PER_GPU SLURM_CPUS_PER_TASK SLURM_GRES SLURM_GPUS SLURM_GPUS_PER_NODE
    unset SLURM_GPUS_PER_SOCKET SLURM_GPUS_PER_TASK SLURM_GPU_BIND SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
    unset SLURM_NTASKS SLURM_NTASKS_PER_CORE SLURM_NTASKS_PER_GPU SLURM_NTASKS_PER_NODE
    unset SLURM_NTASKS_PER_SOCKET SLURM_OVERCOMMIT SLURM_OVERLAP SLURM_STEP_GRES SLURM_TRES_PER_TASK
    export SLURM_EXCLUSIVE=user SLURM_MEM_PER_NODE=125G SLURM_EXPORT_ENV=ALL
    export COLLECTION_MODE=smoke EPISODES=1 WORKERS=1 GLOBAL_EPISODES=4 SHARD_COUNT=4
    export SHARD_RANK="${ranks[slot]}" HTTP_PORT_BASE="${ports[slot]}" COLLECTION_ROOT="${roots[slot]}"
    bash "${COLLECT_SBATCH}"
  ) >"${QUAL_PARENT}/slot_${slot}.log" 2>&1 &
  child_pids+=("$!")
done

record_args=(
  record-barrier
  --expected-node "${QUAL_NODE}"
  --expected-allocation-job "${SLURM_JOB_ID}"
  --output "${barrier}"
  --timeout-seconds 600
)
for slot in {0..7}; do
  record_args+=(
    --root "${roots[slot]}"
    --expected-slot "${slot}"
    --expected-rank "${ranks[slot]}"
    --expected-port "${ports[slot]}"
  )
done
"${REPO_ROOT}/.venv/bin/python" "${VERIFIER}" "${record_args[@]}"

child_failure=0
for pid in "${child_pids[@]}"; do
  if ! wait "${pid}"; then
    child_failure=1
  fi
done
if [[ "${child_failure}" -ne 0 ]]; then
  echo "one or more atomic smoke collectors failed" >&2
  exit 1
fi
for root in "${roots[@]}"; do
  if [[ ! -f "${root}/DONE" || "$(tr -d '[:space:]' < "${root}/DONE")" != 1 || \
        -e "${root}/FAILED.json" || -e "${root}/HEARTBEAT.json" ]]; then
    echo "atomic smoke root did not commit cleanly: ${root}" >&2
    exit 1
  fi
done
printf '8\n' > "${QUAL_PARENT}/ATOMIC_DONE.tmp"
mv "${QUAL_PARENT}/ATOMIC_DONE.tmp" "${QUAL_PARENT}/ATOMIC_DONE"
completed=1
echo "atomic eight-GPU smoke collection committed job=${SLURM_JOB_ID} node=${QUAL_NODE}"
