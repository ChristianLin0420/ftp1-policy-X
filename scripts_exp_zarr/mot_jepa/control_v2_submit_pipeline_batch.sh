#!/bin/bash
# Submit the immutable Control V2 chain. This script is a dry run unless SUBMIT=1 is explicit.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs}"
PRETRAINED_RUN="${PRETRAINED_RUN:?set PRETRAINED_RUN to the qualified 100k MoT run}"
PIPELINE_ID="${PIPELINE_ID:-v3_$(date -u +%Y%m%dT%H%M%SZ)}"
SUBMIT="${SUBMIT:-0}"
COLLECTION_PARENT="${COLLECTION_PARENT:-${RUN_ROOT}/mot_control_v2_data}"
PREP_PARENT="${PREP_PARENT:-${RUN_ROOT}/mot_control_v2_data_prepared}"
PAIR_PARENT="${PAIR_PARENT:-${RUN_ROOT}/mot_control_v2_eval}"
EVAL_NODE="${EVAL_NODE:-gpu-h100-0044}"
COLLECTION_NODE="${COLLECTION_NODE:-${EVAL_NODE}}"
COLLECTION_NODES="${COLLECTION_NODES:-${COLLECTION_NODE}}"
PRODUCTION_SHARD_COUNT="${PRODUCTION_SHARD_COUNT:-20}"
PRODUCTION_EPISODES_PER_SHARD="${PRODUCTION_EPISODES_PER_SHARD:-50}"
PRODUCTION_TIME_LIMIT="${PRODUCTION_TIME_LIMIT:-04:00:00}"
SUBMIT_PAIRED_N100="${SUBMIT_PAIRED_N100:-0}"
HEAD_EXP="lift_bottle_head20k_${PIPELINE_ID}"
ADAPT_EXP="lift_bottle_adapt10k_${PIPELINE_ID}"
HEAD_RUN="${RUN_ROOT}/mot_jepa_control_v2/${HEAD_EXP}"
ADAPT_RUN="${RUN_ROOT}/mot_jepa_control_v2/${ADAPT_EXP}"
SBATCH_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"
COLLECT_SBATCH="${SBATCH_DIR}/control_v2_collect_batch.sbatch"
MERGE_SBATCH="${SBATCH_DIR}/control_v2_merge_collection_batch.sbatch"
PAIRED_SHARD_SBATCH="${SBATCH_DIR}/control_v2_closedloop_paired_shard.sbatch"
PAIRED_MERGE_SBATCH="${SBATCH_DIR}/control_v2_merge_paired_eval.sbatch"

for value_name in PRODUCTION_SHARD_COUNT PRODUCTION_EPISODES_PER_SHARD; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer" >&2
    exit 2
  fi
done
PRODUCTION_GLOBAL_EPISODES=$((PRODUCTION_SHARD_COUNT * PRODUCTION_EPISODES_PER_SHARD))
if [[ "${PRODUCTION_GLOBAL_EPISODES}" -ne 1000 ]]; then
  echo "production shard topology must close exactly 1000 episodes; got ${PRODUCTION_SHARD_COUNT}x${PRODUCTION_EPISODES_PER_SHARD}=${PRODUCTION_GLOBAL_EPISODES}" >&2
  exit 2
fi
if [[ ! "${PRODUCTION_TIME_LIMIT}" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
  echo "PRODUCTION_TIME_LIMIT must use HH:MM:SS" >&2
  exit 2
fi
production_time_seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
if [[ "${production_time_seconds}" -lt 1 || "${production_time_seconds}" -gt 14400 ]]; then
  echo "PRODUCTION_TIME_LIMIT must be greater than zero and no more than 04:00:00" >&2
  exit 2
fi
IFS=: read -r -a production_nodes <<< "${COLLECTION_NODES}"
if [[ "${#production_nodes[@]}" -lt 1 ]]; then
  echo "COLLECTION_NODES must contain at least one qualified node" >&2
  exit 2
fi
for node in "${production_nodes[@]}"; do
  if [[ -z "${node}" || ! "${node}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "COLLECTION_NODES must be a colon-separated list of explicit qualified node names" >&2
    exit 2
  fi
done
if [[ "${SUBMIT_PAIRED_N100}" != "0" && "${SUBMIT_PAIRED_N100}" != "1" ]]; then
  echo "SUBMIT_PAIRED_N100 must be 0 or 1" >&2
  exit 2
fi
if [[ "${SUBMIT_PAIRED_N100}" == "1" ]]; then
  if [[ ! -f "${PAIRED_SHARD_SBATCH}" || ! -f "${PAIRED_MERGE_SBATCH}" ]]; then
    echo "paired n100 requires both ${PAIRED_SHARD_SBATCH} and ${PAIRED_MERGE_SBATCH}" >&2
    exit 2
  fi
fi

if [[ "${SUBMIT}" != "1" ]]; then
  echo "DRY RUN: no jobs submitted. Set SUBMIT=1 to launch."
  echo "pipeline=${PIPELINE_ID}"
  echo "source_backbone=${PRETRAINED_RUN}@100000"
  echo "chain: 4x isolated one-GPU smoke shards -> merge4 -> ${PRODUCTION_SHARD_COUNT}x${PRODUCTION_EPISODES_PER_SHARD} production shards -> merge${PRODUCTION_GLOBAL_EPISODES}"
  echo "       -> prepare V3 Zarr -> head20k(frozen) -> adapt10k(last4)"
  echo "       -> n1@900000(trace-only) -> n20@900100 >=19"
  if [[ "${SUBMIT_PAIRED_N100}" == "1" ]]; then
    echo "       -> 5x paired n20@1000000..1000099 -> merge paired n100 -> acceptance"
  else
    echo "       -> stop after n20; append paired n100 from its recorded gate job"
  fi
  echo "production: partition=batch time=${PRODUCTION_TIME_LIMIT} nodes=${COLLECTION_NODES} (round-robin)"
  echo "recovery is deliberately outside this preregistered chain; mine/correct n20 failures before defining it"
  echo "Optional: PREPARED_STORE=/absolute/.../lift_bottle_head.zarr skips collection/preparation."
  echo "Optional: SMOKE_SHARD_ROOTS=/abs/shard0:/abs/shard1:/abs/shard2:/abs/shard3 reuses batch-versioned smoke shards."
  echo "          Legacy smoke shards must use their completed SMOKE_MERGED_ROOT instead."
  echo "Optional: SMOKE_MERGED_ROOT=/abs/merged_smoke reuses its completed Slurm job as the production dependency gate."
  echo "Optional: PRODUCTION_SHARD_COUNT and PRODUCTION_EPISODES_PER_SHARD may change the split while preserving exactly 1000 episodes."
  exit 0
fi
if [[ "${RUN_RECOVERY:-0}" != "0" || -n "${RECOVERY_STORE_GLOB:-}" ]]; then
  echo "recovery is not part of this pipeline: first run n20, collect expert-corrected failures, then freeze a new contract" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs"
job_ids=()
dependency=""
submission_complete=0

_cancel_partial_submission() {
  local rc=$?
  trap - EXIT
  if [[ "${submission_complete}" -ne 1 && "${#job_ids[@]}" -gt 0 ]]; then
    echo "submission failed; canceling partial job chain: ${job_ids[*]}" >&2
    scancel "${job_ids[@]}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap _cancel_partial_submission EXIT

submit_collection_group() {
  local mode=$1
  local local_episodes=$2
  local global_episodes=$3
  local shard_count=$4
  local shard_parent=$5
  local merge_parent=$6
  local incoming=$7
  local shard_ids=()
  local shard_roots=()
  local shard_nodes=()
  local rank
  for ((rank = 0; rank < shard_count; rank++)); do
    local collection_node="${COLLECTION_NODE}"
    local shard_args=(--parsable)
    if [[ "${mode}" == "production" ]]; then
      local node_index=$((rank % ${#production_nodes[@]}))
      collection_node="${production_nodes[node_index]}"
      shard_args+=(--partition=batch --time="${PRODUCTION_TIME_LIMIT}")
    else
      shard_args+=(--time=01:00:00)
    fi
    shard_args+=(--nodelist="${collection_node}")
    if [[ -n "${incoming}" ]]; then
      shard_args+=(--dependency="${incoming}")
    fi
    local exports="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_PARENT=${shard_parent},COLLECTION_MODE=${mode},EPISODES=${local_episodes},WORKERS=1,GLOBAL_EPISODES=${global_episodes},SHARD_RANK=${rank},SHARD_COUNT=${shard_count}"
    local shard_job
    shard_job="$(sbatch "${shard_args[@]}" --export="${exports}" "${COLLECT_SBATCH}")"
    shard_job="${shard_job%%;*}"
    job_ids+=("${shard_job}")
    shard_ids+=("${shard_job}")
    shard_roots+=("${shard_parent}/lift_bottle_${shard_job}")
    shard_nodes+=("${collection_node}")
  done

  local dependency_ids shard_root_list shard_node_list
  dependency_ids="$(IFS=:; echo "${shard_ids[*]}")"
  shard_root_list="$(IFS=:; echo "${shard_roots[*]}")"
  shard_node_list="$(IFS=:; echo "${shard_nodes[*]}")"
  local merge_job
  merge_job="$(sbatch --parsable --dependency="afterok:${dependency_ids}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_PARENT=${merge_parent},COLLECTION_MODE=${mode},SHARD_ROOTS=${shard_root_list},EXPECTED_SHARD_COUNT=${shard_count},EXPECTED_PER_SHARD=${local_episodes}" \
    "${MERGE_SBATCH}")"
  merge_job="${merge_job%%;*}"
  job_ids+=("${merge_job}")
  COLLECTION_GROUP_JOB="${merge_job}"
  COLLECTION_GROUP_ROOT="${merge_parent}/lift_bottle_${merge_job}"
  COLLECTION_GROUP_SHARD_JOBS="${dependency_ids}"
  COLLECTION_GROUP_SHARD_ROOTS="${shard_root_list}"
  COLLECTION_GROUP_SHARD_NODES="${shard_node_list}"
}

submit_existing_smoke_merge() {
  local raw_roots=$1
  local merge_parent=$2
  local roots=()
  IFS=: read -r -a roots <<< "${raw_roots}"
  if [[ "${#roots[@]}" -ne 4 ]]; then
    echo "SMOKE_SHARD_ROOTS must contain exactly four colon-separated roots" >&2
    exit 2
  fi
  local root
  for root in "${roots[@]}"; do
    if [[ ! -s "${root}/DONE" || "$(cat "${root}/DONE")" != "1" || ! -s "${root}/manifest.json" || \
      ! -s "${root}/source_provenance.json" ]]; then
      echo "reused smoke shard is incomplete: ${root}" >&2
      exit 2
    fi
    if ! python3 - "${root}/source_provenance.json" "${REPO_ROOT}" <<'PY'
import json
import pathlib
import sys

source_path = pathlib.Path(sys.argv[1])
repo_root = pathlib.Path(sys.argv[2]).resolve()
payload = json.loads(source_path.read_text())
batch_collector = (repo_root / "scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch").resolve()
matching_roots = [
    record
    for record in payload.get("roots", [])
    if isinstance(record, dict)
    and isinstance(record.get("path"), str)
    and pathlib.Path(record["path"]).resolve() == batch_collector
    and record.get("content_hashed") is True
]
if len(matching_roots) != 1:
    raise SystemExit(
        "SMOKE_SHARD_ROOTS accepts only batch-versioned shards; "
        "use the completed SMOKE_MERGED_ROOT for legacy smoke shards"
    )
PY
    then
      echo "refusing legacy or unclosed SMOKE_SHARD_ROOTS before submission: ${root}" >&2
      exit 2
    fi
  done
  local canonical_roots
  canonical_roots="$(IFS=:; echo "${roots[*]}")"
  local merge_job
  merge_job="$(sbatch --parsable \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_PARENT=${merge_parent},COLLECTION_MODE=smoke,SHARD_ROOTS=${canonical_roots}" \
    "${MERGE_SBATCH}")"
  merge_job="${merge_job%%;*}"
  job_ids+=("${merge_job}")
  COLLECTION_GROUP_JOB="${merge_job}"
  COLLECTION_GROUP_ROOT="${merge_parent}/lift_bottle_${merge_job}"
  COLLECTION_GROUP_SHARD_JOBS=""
  COLLECTION_GROUP_SHARD_ROOTS="${canonical_roots}"
  COLLECTION_GROUP_SHARD_NODES=""
}

reuse_completed_smoke_merge() {
  local root=$1
  if [[ ! -d "${root}" || ! -s "${root}/DONE" || ! -s "${root}/manifest.json" ]]; then
    echo "reused merged smoke collection is incomplete: ${root}" >&2
    exit 2
  fi
  local merge_job
  merge_job="$(python3 - "${root}" <<'PY'
import json
import pathlib
import sys

declared_root = pathlib.Path(sys.argv[1])
if not declared_root.is_absolute() or ".." in declared_root.parts:
    raise SystemExit("SMOKE_MERGED_ROOT must be an absolute path without traversal")
root = declared_root.resolve()
if (root / "DONE").read_text().strip() != "4":
    raise SystemExit("reused merged smoke DONE marker must be exactly 4")
payload = json.loads((root / "manifest.json").read_text())
required = {
    "schema_version": 3,
    "status": "DONE",
    "collection_mode": "smoke",
    "requested_episodes": 4,
    "saved_episodes": 4,
    "global_episodes": 4,
    "shard_rank": -1,
    "shard_count": 4,
}
for field, expected in required.items():
    if payload.get(field) != expected:
        raise SystemExit(f"reused merged smoke manifest has {field}={payload.get(field)!r}; expected {expected!r}")
job_id = payload.get("job_id")
if not isinstance(job_id, str) or not job_id.isdigit():
    raise SystemExit("reused merged smoke manifest lacks a numeric job_id")
print(job_id)
PY
)"
  local merge_state
  merge_state="$(sacct -X -n -j "${merge_job}" --format=State | awk 'NF {print $1; exit}')"
  merge_state="${merge_state%%+*}"
  if [[ "${merge_state}" != "COMPLETED" ]]; then
    echo "reused merged smoke job ${merge_job} is not a completed dependency gate (state=${merge_state:-missing})" >&2
    exit 2
  fi
  COLLECTION_GROUP_JOB="${merge_job}"
  COLLECTION_GROUP_ROOT="${root}"
  COLLECTION_GROUP_SHARD_JOBS=""
  COLLECTION_GROUP_SHARD_ROOTS=""
  COLLECTION_GROUP_SHARD_NODES=""
}

smoke_shard_jobs=""
smoke_shard_roots=""
smoke_shard_nodes=""
smoke_merge_job=""
smoke_merge_root=""
reused_smoke_merge_root=""
production_shard_jobs=""
production_shard_roots=""
production_shard_nodes=""
production_merge_job=""
production_merge_root=""
if [[ -n "${PREPARED_STORE:-}" ]]; then
  if [[ -n "${SMOKE_SHARD_ROOTS:-}" || -n "${SMOKE_MERGED_ROOT:-}" ]]; then
    echo "PREPARED_STORE, SMOKE_SHARD_ROOTS, and SMOKE_MERGED_ROOT are mutually exclusive" >&2
    exit 2
  fi
  store="${PREPARED_STORE}"
  if [[ ! -d "${store}" ]]; then
    echo "PREPARED_STORE is not a Zarr directory: ${store}" >&2
    exit 2
  fi
  echo "using pre-qualified V3 store ${store}"
else
  if [[ -n "${SMOKE_SHARD_ROOTS:-}" && -n "${SMOKE_MERGED_ROOT:-}" ]]; then
    echo "SMOKE_SHARD_ROOTS and SMOKE_MERGED_ROOT are mutually exclusive" >&2
    exit 2
  fi
  for batch_script in "${COLLECT_SBATCH}" "${MERGE_SBATCH}"; do
    if [[ ! -f "${batch_script}" ]]; then
      echo "missing versioned batch collection script: ${batch_script}" >&2
      exit 2
    fi
  done
  COLLECTION_GROUP_JOB=""
  COLLECTION_GROUP_ROOT=""
  if [[ -n "${SMOKE_MERGED_ROOT:-}" ]]; then
    reuse_completed_smoke_merge "${SMOKE_MERGED_ROOT}"
    reused_smoke_merge_root="${COLLECTION_GROUP_ROOT}"
  elif [[ -n "${SMOKE_SHARD_ROOTS:-}" ]]; then
    submit_existing_smoke_merge "${SMOKE_SHARD_ROOTS}" "${COLLECTION_PARENT}/smoke"
  else
    submit_collection_group \
      smoke 1 4 4 "${COLLECTION_PARENT}/smoke_shards" "${COLLECTION_PARENT}/smoke" ""
  fi
  collect_smoke_job="${COLLECTION_GROUP_JOB}"
  smoke_merge_job="${COLLECTION_GROUP_JOB}"
  smoke_merge_root="${COLLECTION_GROUP_ROOT}"
  smoke_shard_jobs="${COLLECTION_GROUP_SHARD_JOBS}"
  smoke_shard_roots="${COLLECTION_GROUP_SHARD_ROOTS}"
  smoke_shard_nodes="${COLLECTION_GROUP_SHARD_NODES}"
  production_dependency="afterok:${collect_smoke_job}"
  if [[ -n "${reused_smoke_merge_root}" ]]; then
    # A completed smoke job may already be purged from slurmctld even while sacct retains its
    # terminal record.  The immutable root and COMPLETED accounting record were validated above,
    # so no live scheduler dependency is needed (and Slurm rejects one once the job is purged).
    production_dependency=""
  fi
  submit_collection_group \
    production "${PRODUCTION_EPISODES_PER_SHARD}" "${PRODUCTION_GLOBAL_EPISODES}" "${PRODUCTION_SHARD_COUNT}" \
    "${COLLECTION_PARENT}/shards" "${COLLECTION_PARENT}" "${production_dependency}"
  collect_job="${COLLECTION_GROUP_JOB}"
  collection_root="${COLLECTION_GROUP_ROOT}"
  production_merge_job="${COLLECTION_GROUP_JOB}"
  production_merge_root="${COLLECTION_GROUP_ROOT}"
  production_shard_jobs="${COLLECTION_GROUP_SHARD_JOBS}"
  production_shard_roots="${COLLECTION_GROUP_SHARD_ROOTS}"
  production_shard_nodes="${COLLECTION_GROUP_SHARD_NODES}"

  prep_job="$(sbatch --parsable --dependency="afterok:${collect_job}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_ROOT=${collection_root},PREP_PARENT=${PREP_PARENT},COLLECTION_PROVENANCE_SCRIPT=${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
    "${SBATCH_DIR}/control_v2_prepare_data.sbatch")"
  prep_job="${prep_job%%;*}"
  job_ids+=("${prep_job}")
  prep_root="${PREP_PARENT}/lift_bottle_${prep_job}"
  store="${prep_root}/clips/UniVTAC_lift_bottle/lift_bottle_head.zarr"
  dependency="afterok:${prep_job}"
fi

submit_stage() {
  local exp_name=$1
  local steps=$2
  local mode=$3
  local init_run=$4
  local init_step=$5
  local incoming=$6
  local stage_store=$7
  local stats_args=(--parsable)
  if [[ -n "${incoming}" ]]; then
    stats_args+=(--dependency="${incoming}")
  fi
  local exports="ALL,REPO_ROOT=${REPO_ROOT},RUN_ROOT=${RUN_ROOT},EXP_NAME=${exp_name},PRETRAINED_RUN=${PRETRAINED_RUN},STORE_GLOB=${stage_store},STEPS=${steps},BACKBONE_MODE=${mode},INIT_RUN=${init_run},INIT_STEP=${init_step}"
  local stats_job
  stats_job="$(sbatch "${stats_args[@]}" --export="${exports}" "${SBATCH_DIR}/control_v2_stats.sbatch")"
  stats_job="${stats_job%%;*}"
  job_ids+=("${stats_job}")
  local train_job
  train_job="$(sbatch --parsable --partition=batch --time=04:00:00 --dependency="afterok:${stats_job}" \
    --export="${exports}" "${SBATCH_DIR}/control_v2_train.sbatch")"
  train_job="${train_job%%;*}"
  job_ids+=("${train_job}")
  STAGE_JOB="${train_job}"
}

STAGE_JOB=""
submit_stage "${HEAD_EXP}" 20000 frozen "" "" "${dependency}" "${store}"
head_job="${STAGE_JOB}"
submit_stage "${ADAPT_EXP}" 10000 last_blocks "${HEAD_RUN}" "" "afterok:${head_job}" "${store}"
adapt_job="${STAGE_JOB}"
final_job="${adapt_job}"
final_run="${ADAPT_RUN}"
final_step=10000

smoke_job="$(sbatch --parsable --dependency="afterok:${final_job}" \
  --partition=batch --time=00:30:00 --no-requeue \
  --nodelist="${EVAL_NODE}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${final_run},V2_STEP=,PAIR_PARENT=${PAIR_PARENT}/smoke,EVAL_MODE=student_gate,TOTAL=1,START_SEED=900000,MIN_SUCCESSES=0,EVAL_NODE=${EVAL_NODE}" \
  "${SBATCH_DIR}/control_v2_closedloop.sbatch")"
smoke_job="${smoke_job%%;*}"
job_ids+=("${smoke_job}")

gate20_job="$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  --partition=batch --time=02:00:00 --no-requeue \
  --nodelist="${EVAL_NODE}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${final_run},V2_STEP=,PAIR_PARENT=${PAIR_PARENT}/gate20,EVAL_MODE=student_gate,TOTAL=20,START_SEED=900100,MIN_SUCCESSES=19,EVAL_NODE=${EVAL_NODE}" \
  "${SBATCH_DIR}/control_v2_closedloop.sbatch")"
gate20_job="${gate20_job%%;*}"
job_ids+=("${gate20_job}")

paired_n100_status="deferred_after_n20"
paired_shard_jobs=""
paired_shard_roots=""
paired_shard_starts=""
paired_merge_job=""
paired_merge_root=""
submit_paired_n100() {
  local incoming=$1
  local paired_parent=$2
  local starts=(1000000 1000020 1000040 1000060 1000080)
  local shard_ids=()
  local shard_roots=()
  local start_seed
  for start_seed in "${starts[@]}"; do
    local shard_job
    shard_job="$(sbatch --parsable --dependency="${incoming}" \
      --partition=batch --time=02:00:00 --no-requeue --nodelist="${EVAL_NODE}" \
      --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${final_run},V2_STEP=,START_SEED=${start_seed},PAIR_PARENT=${paired_parent},EVAL_NODE=${EVAL_NODE}" \
      "${PAIRED_SHARD_SBATCH}")"
    shard_job="${shard_job%%;*}"
    job_ids+=("${shard_job}")
    shard_ids+=("${shard_job}")
    shard_roots+=("${paired_parent}/lift_bottle_s${start_seed}_n20_${shard_job}")
  done

  local dependency_ids shard_root_list shard_start_list
  dependency_ids="$(IFS=:; echo "${shard_ids[*]}")"
  shard_root_list="$(IFS=:; echo "${shard_roots[*]}")"
  shard_start_list="$(IFS=:; echo "${starts[*]}")"
  local merge_job
  merge_job="$(sbatch --parsable --dependency="afterok:${dependency_ids}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${final_run},V2_STEP=,SHARD_ROOTS=${shard_root_list},PAIR_PARENT=${paired_parent}" \
    "${PAIRED_MERGE_SBATCH}")"
  merge_job="${merge_job%%;*}"
  job_ids+=("${merge_job}")
  paired_n100_status="submitted"
  paired_shard_jobs="${dependency_ids}"
  paired_shard_roots="${shard_root_list}"
  paired_shard_starts="${shard_start_list}"
  paired_merge_job="${merge_job}"
  paired_merge_root="${paired_parent}/lift_bottle_s1000000_n100_${merge_job}"
}

if [[ "${SUBMIT_PAIRED_N100}" == "1" ]]; then
  submit_paired_n100 "afterok:${gate20_job}" "${PAIR_PARENT}/paired100"
else
  echo "paired n100 deferred; extend the chain from n20 gate job ${gate20_job}" >&2
fi

manifest="${REPO_ROOT}/logs/control_v2_pipeline_${PIPELINE_ID}.json"
export CONTROL_V2_PIPELINE_ID="${PIPELINE_ID}" CONTROL_V2_PIPELINE_JOBS="${job_ids[*]}"
export CONTROL_V2_PIPELINE_STORE="${store}" CONTROL_V2_PIPELINE_FINAL_RUN="${final_run}"
export CONTROL_V2_PIPELINE_FINAL_STEP="${final_step}" CONTROL_V2_PIPELINE_MANIFEST="${manifest}"
export CONTROL_V2_PIPELINE_SMOKE_SHARDS="${SMOKE_SHARD_ROOTS:-}"
export CONTROL_V2_PIPELINE_SMOKE_JOBS="${smoke_shard_jobs}" CONTROL_V2_PIPELINE_SMOKE_ROOTS="${smoke_shard_roots}"
export CONTROL_V2_PIPELINE_SMOKE_NODES="${smoke_shard_nodes}" CONTROL_V2_PIPELINE_SMOKE_MERGE_JOB="${smoke_merge_job}"
export CONTROL_V2_PIPELINE_SMOKE_MERGE_ROOT="${smoke_merge_root}"
export CONTROL_V2_PIPELINE_REUSED_SMOKE_MERGE_ROOT="${reused_smoke_merge_root}"
export CONTROL_V2_PIPELINE_PRODUCTION_JOBS="${production_shard_jobs}"
export CONTROL_V2_PIPELINE_PRODUCTION_ROOTS="${production_shard_roots}"
export CONTROL_V2_PIPELINE_PRODUCTION_NODES="${production_shard_nodes}"
export CONTROL_V2_PIPELINE_PRODUCTION_MERGE_JOB="${production_merge_job}"
export CONTROL_V2_PIPELINE_PRODUCTION_MERGE_ROOT="${production_merge_root}"
export CONTROL_V2_PIPELINE_PRODUCTION_SHARD_COUNT="${PRODUCTION_SHARD_COUNT}"
export CONTROL_V2_PIPELINE_PRODUCTION_EPISODES_PER_SHARD="${PRODUCTION_EPISODES_PER_SHARD}"
export CONTROL_V2_PIPELINE_PRODUCTION_GLOBAL_EPISODES="${PRODUCTION_GLOBAL_EPISODES}"
export CONTROL_V2_PIPELINE_PRODUCTION_TIME_LIMIT="${PRODUCTION_TIME_LIMIT}"
export CONTROL_V2_PIPELINE_PAIRED_STATUS="${paired_n100_status}"
export CONTROL_V2_PIPELINE_PAIRED_JOBS="${paired_shard_jobs}"
export CONTROL_V2_PIPELINE_PAIRED_ROOTS="${paired_shard_roots}"
export CONTROL_V2_PIPELINE_PAIRED_STARTS="${paired_shard_starts}"
export CONTROL_V2_PIPELINE_PAIRED_MERGE_JOB="${paired_merge_job}"
export CONTROL_V2_PIPELINE_PAIRED_MERGE_ROOT="${paired_merge_root}"
export CONTROL_V2_PIPELINE_PAIRED_NODE="${EVAL_NODE}"
export CONTROL_V2_PIPELINE_N1_JOB="${smoke_job}" CONTROL_V2_PIPELINE_N20_JOB="${gate20_job}"
python3 - <<'PY'
import json, os, pathlib, time


def split(value):
    return [item for item in value.split(":") if item]


payload = {
    "schema_version": 1,
    "pipeline_id": os.environ["CONTROL_V2_PIPELINE_ID"],
    "job_ids": os.environ["CONTROL_V2_PIPELINE_JOBS"].split(),
    "store": os.environ["CONTROL_V2_PIPELINE_STORE"],
    "final_run": os.environ["CONTROL_V2_PIPELINE_FINAL_RUN"],
    "final_step": int(os.environ["CONTROL_V2_PIPELINE_FINAL_STEP"]),
    "reused_smoke_shards": [
        item for item in os.environ["CONTROL_V2_PIPELINE_SMOKE_SHARDS"].split(":") if item
    ],
    "reused_smoke_merge_root": os.environ["CONTROL_V2_PIPELINE_REUSED_SMOKE_MERGE_ROOT"] or None,
    "collection": {
        "smoke": {
            "shard_job_ids": split(os.environ["CONTROL_V2_PIPELINE_SMOKE_JOBS"]),
            "shard_roots": split(os.environ["CONTROL_V2_PIPELINE_SMOKE_ROOTS"]),
            "assigned_nodes": split(os.environ["CONTROL_V2_PIPELINE_SMOKE_NODES"]),
            "merge_job_id": os.environ["CONTROL_V2_PIPELINE_SMOKE_MERGE_JOB"] or None,
            "merge_root": os.environ["CONTROL_V2_PIPELINE_SMOKE_MERGE_ROOT"] or None,
        },
        "production": {
            "shard_count": int(os.environ["CONTROL_V2_PIPELINE_PRODUCTION_SHARD_COUNT"]),
            "episodes_per_shard": int(os.environ["CONTROL_V2_PIPELINE_PRODUCTION_EPISODES_PER_SHARD"]),
            "global_episodes": int(os.environ["CONTROL_V2_PIPELINE_PRODUCTION_GLOBAL_EPISODES"]),
            "partition": "batch",
            "time_limit": os.environ["CONTROL_V2_PIPELINE_PRODUCTION_TIME_LIMIT"],
            "shard_job_ids": split(os.environ["CONTROL_V2_PIPELINE_PRODUCTION_JOBS"]),
            "shard_roots": split(os.environ["CONTROL_V2_PIPELINE_PRODUCTION_ROOTS"]),
            "assigned_nodes": split(os.environ["CONTROL_V2_PIPELINE_PRODUCTION_NODES"]),
            "merge_job_id": os.environ["CONTROL_V2_PIPELINE_PRODUCTION_MERGE_JOB"] or None,
            "merge_root": os.environ["CONTROL_V2_PIPELINE_PRODUCTION_MERGE_ROOT"] or None,
        },
    },
    "paired_n100": {
        "status": os.environ["CONTROL_V2_PIPELINE_PAIRED_STATUS"],
        "partition": "batch",
        "time_limit_per_shard": "02:00:00",
        "trials_per_shard": 20,
        "start_seeds": [int(item) for item in split(os.environ["CONTROL_V2_PIPELINE_PAIRED_STARTS"])],
        "shard_job_ids": split(os.environ["CONTROL_V2_PIPELINE_PAIRED_JOBS"]),
        "shard_roots": split(os.environ["CONTROL_V2_PIPELINE_PAIRED_ROOTS"]),
        "assigned_node": os.environ["CONTROL_V2_PIPELINE_PAIRED_NODE"],
        "merge_job_id": os.environ["CONTROL_V2_PIPELINE_PAIRED_MERGE_JOB"] or None,
        "merge_root": os.environ["CONTROL_V2_PIPELINE_PAIRED_MERGE_ROOT"] or None,
    },
    "downstream_gates": {
        "training_partition": "batch",
        "training_time_limit": "04:00:00",
        "n1_job_id": os.environ["CONTROL_V2_PIPELINE_N1_JOB"],
        "n1_partition": "batch",
        "n1_time_limit": "00:30:00",
        "n20_job_id": os.environ["CONTROL_V2_PIPELINE_N20_JOB"],
        "n20_partition": "batch",
        "n20_time_limit": "02:00:00",
    },
    "submitted_unix": time.time(),
}
path = pathlib.Path(os.environ["CONTROL_V2_PIPELINE_MANIFEST"])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

submission_complete=1
echo "submitted pipeline=${PIPELINE_ID} jobs=${job_ids[*]}"
echo "manifest=${manifest}"
echo "monitor: bash ${SBATCH_DIR}/control_v2_status.sh ${job_ids[*]}"
