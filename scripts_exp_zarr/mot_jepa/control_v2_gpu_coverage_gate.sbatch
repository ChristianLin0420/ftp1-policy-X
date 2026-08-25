#!/bin/bash
# Verify eight independent one-GPU collection smokes before a node is eligible for production.
# This gate is deliberately separate from the production chain: a failed qualification leaves the
# existing collector placement and dependencies untouched.
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu_short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=mot-v3-gpu-cover-gate
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
QUAL_NODE="${QUAL_NODE:?set QUAL_NODE to the exact node under qualification}"
QUAL_ROOTS="${QUAL_ROOTS:?set QUAL_ROOTS to eight colon-separated smoke roots}"
QUAL_JOB_IDS="${QUAL_JOB_IDS:?set QUAL_JOB_IDS to eight colon-separated smoke job IDs}"
REFERENCE_ROOT="${REFERENCE_ROOT:?set REFERENCE_ROOT to a fully verified production shard}"
QUAL_OUTPUT="${QUAL_OUTPUT:?set QUAL_OUTPUT to the atomic GPU-coverage result path}"
QUAL_MANIFEST="${QUAL_MANIFEST:?set QUAL_MANIFEST to the immutable submission manifest}"
QUAL_LAUNCHER="${QUAL_LAUNCHER:?set QUAL_LAUNCHER to the submitted launcher}"
VERIFIER="${REPO_ROOT}/scripts/mot_jepa_control_v2_verify_gpu_coverage.py"
COLLECT_SBATCH="${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch"
GATE_SBATCH="$(realpath "${BASH_SOURCE[0]}")"

for path in "${REPO_ROOT}" "${REFERENCE_ROOT}"; do
  if [[ "${path}" != /* || ! -d "${path}" ]]; then
    echo "required directory is missing or non-absolute: ${path}" >&2
    exit 2
  fi
done
for path in "${QUAL_OUTPUT}" "${QUAL_MANIFEST}" "${QUAL_LAUNCHER}"; do
  if [[ "${path}" != /* ]]; then
    echo "required path is non-absolute: ${path}" >&2
    exit 2
  fi
done
for path in "${VERIFIER}" "${COLLECT_SBATCH}" "${GATE_SBATCH}" "${QUAL_MANIFEST}" "${QUAL_LAUNCHER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required qualification source is missing: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${QUAL_OUTPUT}" ]]; then
  echo "refuse existing GPU-coverage output: ${QUAL_OUTPUT}" >&2
  exit 2
fi

IFS=: read -r -a roots <<< "${QUAL_ROOTS}"
IFS=: read -r -a job_ids <<< "${QUAL_JOB_IDS}"
if [[ "${#roots[@]}" -ne 8 || "${#job_ids[@]}" -ne 8 ]]; then
  echo "GPU-coverage gate requires exactly eight roots and eight job IDs" >&2
  exit 2
fi

"${REPO_ROOT}/.venv/bin/python" - \
  "${QUAL_MANIFEST}" "${QUAL_NODE}" "${QUAL_ROOTS}" "${QUAL_JOB_IDS}" "${QUAL_OUTPUT}" \
  "${QUAL_LAUNCHER}" "${COLLECT_SBATCH}" "${GATE_SBATCH}" "${VERIFIER}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    manifest_arg,
    node,
    roots_arg,
    jobs_arg,
    output_arg,
    launcher_arg,
    collect_arg,
    gate_arg,
    verifier_arg,
    gate_job,
) = sys.argv[1:]
manifest_path = pathlib.Path(manifest_arg)
payload = json.loads(manifest_path.read_text())
roots = roots_arg.split(":")
jobs = jobs_arg.split(":")
expected_sources = {
    "launcher": pathlib.Path(launcher_arg).resolve(),
    "collect_sbatch": pathlib.Path(collect_arg).resolve(),
    "gate_sbatch": pathlib.Path(gate_arg).resolve(),
    "verifier": pathlib.Path(verifier_arg).resolve(),
}
if payload.get("schema_version") != 1 or payload.get("status") != "SUBMITTED":
    raise SystemExit("GPU-coverage submission manifest has an invalid schema/status")
if payload.get("node") != node:
    raise SystemExit("GPU-coverage manifest node changed")
smoke = payload.get("smoke", {})
gate = payload.get("gate", {})
if smoke.get("job_ids") != jobs or smoke.get("roots") != roots:
    raise SystemExit("GPU-coverage manifest smoke jobs/roots changed")
if len(set(jobs)) != 8 or len(set(roots)) != 8:
    raise SystemExit("GPU-coverage manifest does not contain eight distinct jobs/roots")
if gate.get("job_id") != gate_job or gate.get("output") != output_arg:
    raise SystemExit("GPU-coverage manifest gate identity changed")
if gate.get("dependency") != f"afterok:{':'.join(jobs)}":
    raise SystemExit("GPU-coverage manifest dependency changed")
sources = payload.get("sources", {})
if set(sources) != set(expected_sources):
    raise SystemExit("GPU-coverage manifest source set changed")
for name, path in expected_sources.items():
    record = sources[name]
    if pathlib.Path(record.get("path", "")).resolve() != path or record.get("sha256") != sha256(path):
        raise SystemExit(f"GPU-coverage source changed: {name}")
print(json.dumps({"manifest_verified": True, "node": node, "smoke_jobs": jobs, "gate_job": gate_job}))
PY

args=(
  --expected-count 8
  --expected-node "${QUAL_NODE}"
  --reference-root "${REFERENCE_ROOT}"
  --output "${QUAL_OUTPUT}"
)
for index in "${!roots[@]}"; do
  root="${roots[index]}"
  job_id="${job_ids[index]}"
  if [[ "${root}" != /* || ! "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid qualification root/job pair: ${root} ${job_id}" >&2
    exit 2
  fi
  args+=(--root "${root}" --expected-job "${job_id}")
done

before_verifier="$(sha256sum "${VERIFIER}" | awk '{print $1}')"
before_manifest="$(sha256sum "${QUAL_MANIFEST}" | awk '{print $1}')"
before_launcher="$(sha256sum "${QUAL_LAUNCHER}" | awk '{print $1}')"
before_collect="$(sha256sum "${COLLECT_SBATCH}" | awk '{print $1}')"
before_gate="$(sha256sum "${GATE_SBATCH}" | awk '{print $1}')"
"${REPO_ROOT}/.venv/bin/python" "${VERIFIER}" "${args[@]}"
after_verifier="$(sha256sum "${VERIFIER}" | awk '{print $1}')"
after_manifest="$(sha256sum "${QUAL_MANIFEST}" | awk '{print $1}')"
after_launcher="$(sha256sum "${QUAL_LAUNCHER}" | awk '{print $1}')"
after_collect="$(sha256sum "${COLLECT_SBATCH}" | awk '{print $1}')"
after_gate="$(sha256sum "${GATE_SBATCH}" | awk '{print $1}')"
if [[ "${before_verifier}" != "${after_verifier}" || \
      "${before_manifest}" != "${after_manifest}" || \
      "${before_launcher}" != "${after_launcher}" || \
      "${before_collect}" != "${after_collect}" || \
      "${before_gate}" != "${after_gate}" ]]; then
  echo "GPU-coverage sources drifted during verification" >&2
  exit 2
fi

echo "GPU coverage qualified node=${QUAL_NODE} output=${QUAL_OUTPUT}"
