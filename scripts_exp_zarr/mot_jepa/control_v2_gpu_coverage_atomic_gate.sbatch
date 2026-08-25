#!/bin/bash
# Independently verify the atomic eight-GPU qualification after the allocation commits.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu_short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=mot-v3-gpu-cover-atomic-gate
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
QUAL_NODE="${QUAL_NODE:?set QUAL_NODE to the exact node under qualification}"
QUAL_PARENT="${QUAL_PARENT:?set QUAL_PARENT to the atomic output parent}"
REFERENCE_ROOT="${REFERENCE_ROOT:?set REFERENCE_ROOT to a fully verified production shard}"
QUAL_MANIFEST="${QUAL_MANIFEST:?set QUAL_MANIFEST to the immutable submission manifest}"
QUAL_ROOTS="${QUAL_ROOTS:?set QUAL_ROOTS to eight colon-separated smoke roots}"
QUAL_RANKS="${QUAL_RANKS:?set QUAL_RANKS to eight colon-separated ranks}"
QUAL_PORTS="${QUAL_PORTS:?set QUAL_PORTS to eight colon-separated ports}"
QUAL_BARRIER="${QUAL_BARRIER:?set QUAL_BARRIER to the concurrency barrier}"
QUAL_OUTPUT="${QUAL_OUTPUT:?set QUAL_OUTPUT to the fresh atomic verification output}"
QUAL_ATOMIC_JOB="${QUAL_ATOMIC_JOB:?set QUAL_ATOMIC_JOB to the allocation job ID}"
QUAL_LAUNCHER="${QUAL_LAUNCHER:?set QUAL_LAUNCHER to the submitted launcher}"
VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage_atomic.py"
BASE_VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage.py"
COLLECT_SBATCH="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch"
ATOMIC_SBATCH="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_gpu_coverage_atomic.sbatch"
GATE_SBATCH="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_gpu_coverage_atomic_gate.sbatch"

for path in "${REPO_ROOT}" "${QUAL_PARENT}" "${REFERENCE_ROOT}"; do
  if [[ "${path}" != /* || ! -d "${path}" ]]; then
    echo "required directory is missing or non-absolute: ${path}" >&2
    exit 2
  fi
done
for path in "${QUAL_MANIFEST}" "${QUAL_BARRIER}" "${QUAL_OUTPUT}" "${QUAL_LAUNCHER}"; do
  if [[ "${path}" != /* ]]; then
    echo "required path is non-absolute: ${path}" >&2
    exit 2
  fi
done
for path in "${VERIFIER}" "${BASE_VERIFIER}" "${COLLECT_SBATCH}" "${ATOMIC_SBATCH}" "${GATE_SBATCH}" \
  "${QUAL_MANIFEST}" "${QUAL_LAUNCHER}" "${QUAL_BARRIER}" "${QUAL_PARENT}/ATOMIC_DONE"; do
  if [[ ! -f "${path}" ]]; then
    echo "required atomic qualification artifact is missing: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${QUAL_OUTPUT}" ]]; then
  echo "refuse existing atomic GPU-coverage output: ${QUAL_OUTPUT}" >&2
  exit 2
fi
candidate_output="${QUAL_OUTPUT}.candidate.${SLURM_JOB_ID}"
if [[ -e "${candidate_output}" ]]; then
  echo "refuse existing atomic GPU-coverage candidate: ${candidate_output}" >&2
  exit 2
fi
committed=0
_finish() {
  rc=$?
  trap - EXIT
  if [[ "${committed}" -ne 1 ]]; then
    rm -f "${candidate_output}"
  fi
  exit "${rc}"
}
trap _finish EXIT

IFS=: read -r -a roots <<< "${QUAL_ROOTS}"
IFS=: read -r -a ranks <<< "${QUAL_RANKS}"
IFS=: read -r -a ports <<< "${QUAL_PORTS}"
if [[ "${#roots[@]}" -ne 8 || "${#ranks[@]}" -ne 8 || "${#ports[@]}" -ne 8 ]]; then
  echo "atomic GPU-coverage gate requires exactly eight roots/ranks/ports" >&2
  exit 2
fi

"${REPO_ROOT}/.venv/bin/python" - \
  "${QUAL_MANIFEST}" "${QUAL_ATOMIC_JOB}" "${SLURM_JOB_ID}" "${QUAL_NODE}" "${QUAL_PARENT}" \
  "${REFERENCE_ROOT}" "${QUAL_ROOTS}" "${QUAL_RANKS}" "${QUAL_PORTS}" "${QUAL_BARRIER}" "${QUAL_OUTPUT}" \
  "${QUAL_LAUNCHER}" "${ATOMIC_SBATCH}" "${GATE_SBATCH}" "${COLLECT_SBATCH}" "${VERIFIER}" "${BASE_VERIFIER}" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    manifest_arg,
    allocation_job,
    gate_job,
    node,
    parent,
    reference,
    roots_arg,
    ranks_arg,
    ports_arg,
    barrier,
    output,
    launcher_arg,
    atomic_arg,
    gate_arg,
    collect_arg,
    verifier_arg,
    base_verifier_arg,
) = sys.argv[1:]
payload = json.loads(pathlib.Path(manifest_arg).read_text())
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
if allocation.get("job_id") != allocation_job or payload.get("slots") != expected_slots:
    raise SystemExit("atomic allocation identity or slot mapping changed")
gate = payload.get("gate", {})
if gate.get("job_id") != gate_job or gate.get("dependency") != f"afterok:{allocation_job}":
    raise SystemExit("atomic gate identity or dependency changed")
if gate.get("output") != output or payload.get("barrier") != barrier:
    raise SystemExit("atomic gate output or barrier path changed")
sources = payload.get("sources", {})
if set(sources) != set(expected_sources):
    raise SystemExit("atomic submission manifest source set changed")
for name, path in expected_sources.items():
    record = sources[name]
    if pathlib.Path(record.get("path", "")).resolve() != path or record.get("sha256") != sha256(path):
        raise SystemExit(f"atomic qualification source changed: {name}")
print(json.dumps({"manifest_verified": True, "allocation_job": allocation_job, "gate_job": gate_job}))
PY

args=(
  verify
  --reference-root "${REFERENCE_ROOT}"
  --expected-node "${QUAL_NODE}"
  --expected-allocation-job "${QUAL_ATOMIC_JOB}"
  --concurrency-barrier "${QUAL_BARRIER}"
  --output "${candidate_output}"
)
for slot in {0..7}; do
  root="${roots[slot]}"
  rank="${ranks[slot]}"
  port="${ports[slot]}"
  if [[ "${root}" != /* || ! "${rank}" =~ ^[0-3]$ || ! "${port}" =~ ^[0-9]+$ ]]; then
    echo "invalid atomic qualification slot mapping: ${slot} ${root} ${rank} ${port}" >&2
    exit 2
  fi
  args+=(
    --root "${root}"
    --expected-slot "${slot}"
    --expected-rank "${rank}"
    --expected-port "${port}"
  )
done

before_verifier="$(sha256sum "${VERIFIER}" | awk '{print $1}')"
before_base="$(sha256sum "${BASE_VERIFIER}" | awk '{print $1}')"
before_manifest="$(sha256sum "${QUAL_MANIFEST}" | awk '{print $1}')"
before_launcher="$(sha256sum "${QUAL_LAUNCHER}" | awk '{print $1}')"
before_collect="$(sha256sum "${COLLECT_SBATCH}" | awk '{print $1}')"
before_atomic="$(sha256sum "${ATOMIC_SBATCH}" | awk '{print $1}')"
before_gate="$(sha256sum "${GATE_SBATCH}" | awk '{print $1}')"
"${REPO_ROOT}/.venv/bin/python" "${VERIFIER}" "${args[@]}"
after_verifier="$(sha256sum "${VERIFIER}" | awk '{print $1}')"
after_base="$(sha256sum "${BASE_VERIFIER}" | awk '{print $1}')"
after_manifest="$(sha256sum "${QUAL_MANIFEST}" | awk '{print $1}')"
after_launcher="$(sha256sum "${QUAL_LAUNCHER}" | awk '{print $1}')"
after_collect="$(sha256sum "${COLLECT_SBATCH}" | awk '{print $1}')"
after_atomic="$(sha256sum "${ATOMIC_SBATCH}" | awk '{print $1}')"
after_gate="$(sha256sum "${GATE_SBATCH}" | awk '{print $1}')"
if [[ "${before_verifier}" != "${after_verifier}" || "${before_base}" != "${after_base}" || \
      "${before_manifest}" != "${after_manifest}" || "${before_launcher}" != "${after_launcher}" || \
      "${before_collect}" != "${after_collect}" || "${before_atomic}" != "${after_atomic}" || \
      "${before_gate}" != "${after_gate}" ]]; then
  echo "atomic GPU-coverage sources drifted during verification" >&2
  exit 2
fi

mv "${candidate_output}" "${QUAL_OUTPUT}"
committed=1

echo "atomic GPU coverage qualified node=${QUAL_NODE} output=${QUAL_OUTPUT}"
