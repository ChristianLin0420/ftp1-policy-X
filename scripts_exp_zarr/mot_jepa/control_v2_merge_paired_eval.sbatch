#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --job-name=mot-v2-paired100-merge
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# CPU-only exact merge of five completed paired n=20 roots. The launcher supplies afterok on all
# five GPU shards. This job reconstructs canonical raw n=100 metadata/traces and delegates the only
# statistical verdict to the unchanged acceptance implementation.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
V2_RUN="${V2_RUN:?set V2_RUN to the completed Control V2 stage evaluated by every shard}"
APPEND_MANIFEST="${APPEND_MANIFEST:?set APPEND_MANIFEST to the immutable paired submission manifest}"
APPEND_LAUNCHER="${APPEND_LAUNCHER:?set APPEND_LAUNCHER to the versioned paired append launcher}"
SHARD_ROOTS="${SHARD_ROOTS:?set colon-separated exact five paired shard roots}"
PAIR_PARENT="${PAIR_PARENT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/mot_control_v2_eval}"
PAIR_ROOT="${PAIR_ROOT:-${PAIR_PARENT}/lift_bottle_s1000000_n100_${SLURM_JOB_ID}}"
PAIR_ROOT="${PAIR_ROOT//%j/${SLURM_JOB_ID}}"
MERGE_MANIFEST="${PAIR_ROOT}/PAIRED_N100.json"
PROVENANCE="${PAIR_ROOT}/provenance.json"
VERDICT="${PAIR_ROOT}/acceptance.json"
SIDECAR_HEARTBEAT="${PAIR_ROOT}.HEARTBEAT.json"
SIDECAR_FAILED="${PAIR_ROOT}.FAILED.json"

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

if [[ "${SHARD_ROOTS}" == :* || "${SHARD_ROOTS}" == *: || "${SHARD_ROOTS}" == *::* ]]; then
  echo "SHARD_ROOTS contains an empty root" >&2
  exit 2
fi
IFS=: read -r -a shard_roots <<< "${SHARD_ROOTS}"
if [[ "${#shard_roots[@]}" -ne 5 ]]; then
  echo "paired n100 merge requires exactly five shard roots" >&2
  exit 2
fi
if [[ -e "${PAIR_ROOT}" || -e "${SIDECAR_HEARTBEAT}" || -e "${SIDECAR_FAILED}" ]]; then
  echo "refuse existing paired n100 root or sidecar: ${PAIR_ROOT}" >&2
  exit 2
fi
for path in "${V2_RUN}/artifact.json" "${V2_RUN}/DONE"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing deployed Control V2 artifact ${path}" >&2
    exit 2
  fi
done

mkdir -p "${REPO_ROOT}/logs" "${PAIR_PARENT}"
PYCACHE_ROOT="${SLURM_TMPDIR:-/tmp}/mot_control_v2_paired_merge_pycache_${SLURM_JOB_ID:-$$}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"

completed=0
heartbeat_pid=""
_finish() {
  rc=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  rm -f "${SIDECAR_HEARTBEAT}"
  if [[ "${completed}" -ne 1 ]]; then
    if [[ -d "${PAIR_ROOT}" ]]; then
      printf '{"status":"FAILED","job_id":"%s","exit_code":%d}\n' "${SLURM_JOB_ID}" "${rc}" > "${PAIR_ROOT}/FAILED.json"
    else
      printf '{"status":"FAILED","job_id":"%s","exit_code":%d}\n' "${SLURM_JOB_ID}" "${rc}" > "${SIDECAR_FAILED}"
    fi
  fi
}
trap _finish EXIT

main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] || state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    printf '{"status":"RUNNING","job_id":"%s","scheduler_state":"%s"}\n' \
      "${SLURM_JOB_ID}" "${state:-UNKNOWN}" > "${SIDECAR_HEARTBEAT}"
    sleep 60
  done
) &
heartbeat_pid=$!

echo "[$(date -Is)] validating and synthesizing exact paired n100 roots"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_merge_paired_eval.py" \
  merge --shard-roots "${SHARD_ROOTS}" --output-root "${PAIR_ROOT}"

if [[ ! -s "${MERGE_MANIFEST}" || ! -s "${PAIR_ROOT}/qualified_runtime.json" ]]; then
  echo "paired n100 synthesis did not commit its canonical artifacts" >&2
  exit 2
fi

source_paths=(
  "${V2_RUN}/DONE"
  "${V2_RUN}/artifact.json"
  "${V2_RUN}/normalization.json"
  "${V2_RUN}/BEST_VALIDATION.json"
  "${V2_RUN}/VALIDATION_SAMPLES.json"
  "${PAIR_ROOT}/qualified_runtime.json"
  "${PAIR_ROOT}/student/lift_bottle/student/metadata.json"
  "${PAIR_ROOT}/student/lift_bottle/student/metadata"
  "${PAIR_ROOT}/student/lift_bottle/student/trajectory"
  "${PAIR_ROOT}/official/lift_bottle/official/metadata.json"
  "${PAIR_ROOT}/official/lift_bottle/official/metadata"
  "${MERGE_MANIFEST}"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_acceptance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_merge_paired_eval.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_merge_paired_eval.sbatch"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh"
  "${APPEND_MANIFEST}"
)
for shard_root in "${shard_roots[@]}"; do
  source_paths+=(
    "${shard_root}/DONE"
    "${shard_root}/PAIRED_SHARD.json"
    "${shard_root}/evaluation_contract.json"
    "${shard_root}/provenance.json"
    "${shard_root}/qualified_runtime.json"
    "${shard_root}/qualified_runtime_after.json"
  )
done
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --output "${PROVENANCE}" "${source_paths[@]}"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --verify "${PROVENANCE}"

echo "[$(date -Is)] invoking unchanged authoritative paired n100 acceptance"
set +e
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_acceptance.py" \
  --official-root "${PAIR_ROOT}/official/lift_bottle/official" \
  --student-root "${PAIR_ROOT}/student/lift_bottle/student" \
  --artifact "${V2_RUN}/artifact.json" \
  --qualified-runtime "${PAIR_ROOT}/qualified_runtime.json" \
  --start-seed 1000000 \
  --output "${VERDICT}"
acceptance_rc=$?
set -e
if [[ ! -s "${VERDICT}" ]]; then
  echo "authoritative acceptance produced no verdict" >&2
  exit 2
fi
echo "[$(date -Is)] post-verifying combined raw records and every shard source closure"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --verify "${PROVENANCE}"
for shard_root in "${shard_roots[@]}"; do
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
    --verify "${shard_root}/provenance.json"
done
acceptance_decision="$(python3 - "${VERDICT}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload.get("decision", ""))
PY
)"

verdict_sha256="$(sha256sum "${VERDICT}" | awk '{print $1}')"
merge_sha256="$(sha256sum "${MERGE_MANIFEST}" | awk '{print $1}')"
provenance_sha256="$(sha256sum "${PROVENANCE}" | awk '{print $1}')"
if [[ "${acceptance_rc}" -eq 0 ]]; then
  if [[ "${acceptance_decision}" != "ACCEPT" ]]; then
    echo "authoritative acceptance returned rc=0 with decision=${acceptance_decision:-missing}" >&2
    exit 2
  fi
  printf '{"status":"ACCEPTED","trials":100,"acceptance_sha256":"%s","merge_sha256":"%s","provenance_sha256":"%s"}\n' \
    "${verdict_sha256}" "${merge_sha256}" "${provenance_sha256}" > "${PAIR_ROOT}/DONE"
  completed=1
  echo "[$(date -Is)] ACCEPTED authoritative paired n100 verdict=${VERDICT}"
  exit 0
fi
if [[ "${acceptance_decision}" != "REJECT" ]]; then
  echo "authoritative acceptance is invalid: decision=${acceptance_decision:-missing}" >&2
  exit 2
fi
printf '{"status":"REJECTED","trials":100,"acceptance_sha256":"%s","merge_sha256":"%s","provenance_sha256":"%s"}\n' \
  "${verdict_sha256}" "${merge_sha256}" "${provenance_sha256}" > "${PAIR_ROOT}/REJECTED"
completed=1
echo "[$(date -Is)] REJECTED authoritative paired n100 verdict=${VERDICT}" >&2
exit 3
