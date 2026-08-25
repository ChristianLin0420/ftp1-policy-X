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
HEAD_EXP="lift_bottle_head20k_${PIPELINE_ID}"
ADAPT_EXP="lift_bottle_adapt10k_${PIPELINE_ID}"
HEAD_RUN="${RUN_ROOT}/mot_jepa_control_v2/${HEAD_EXP}"
ADAPT_RUN="${RUN_ROOT}/mot_jepa_control_v2/${ADAPT_EXP}"
SBATCH_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"

if [[ "${SUBMIT}" != "1" ]]; then
  echo "DRY RUN: no jobs submitted. Set SUBMIT=1 to launch."
  echo "pipeline=${PIPELINE_ID}"
  echo "source_backbone=${PRETRAINED_RUN}@100000"
  echo "chain: 4x isolated one-GPU smoke shards -> merge4 -> 4x250 production shards -> merge1000"
  echo "       -> prepare V3 Zarr -> head20k(frozen) -> adapt10k(last4)"
  echo "       -> n1@900000(trace-only) -> n20@900100 >=19 -> paired n100@1000000 -> acceptance"
  echo "recovery is deliberately outside this preregistered chain; mine/correct n20 failures before defining it"
  echo "Optional: PREPARED_STORE=/absolute/.../lift_bottle_head.zarr skips collection/preparation."
  echo "Optional: SMOKE_SHARD_ROOTS=/abs/shard0:/abs/shard1:/abs/shard2:/abs/shard3 reuses qualified smoke shards."
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
  local shard_parent=$4
  local merge_parent=$5
  local incoming=$6
  local shard_ids=()
  local shard_roots=()
  local rank
  for rank in 0 1 2 3; do
    local shard_args=(--parsable --nodelist="${COLLECTION_NODE}")
    if [[ -n "${incoming}" ]]; then
      shard_args+=(--dependency="${incoming}")
    fi
    if [[ "${mode}" == "smoke" ]]; then
      shard_args+=(--time=01:00:00)
    fi
    local exports="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_PARENT=${shard_parent},COLLECTION_MODE=${mode},EPISODES=${local_episodes},WORKERS=1,GLOBAL_EPISODES=${global_episodes},SHARD_RANK=${rank},SHARD_COUNT=4"
    local shard_job
    shard_job="$(sbatch "${shard_args[@]}" --export="${exports}" "${SBATCH_DIR}/control_v2_collect.sbatch")"
    shard_job="${shard_job%%;*}"
    job_ids+=("${shard_job}")
    shard_ids+=("${shard_job}")
    shard_roots+=("${shard_parent}/lift_bottle_${shard_job}")
  done

  local dependency_ids="${shard_ids[0]}"
  local shard_root_list="${shard_roots[0]}"
  local index
  for index in 1 2 3; do
    dependency_ids+=":${shard_ids[index]}"
    shard_root_list+=":${shard_roots[index]}"
  done
  local merge_job
  merge_job="$(sbatch --parsable --dependency="afterok:${dependency_ids}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_PARENT=${merge_parent},COLLECTION_MODE=${mode},SHARD_ROOTS=${shard_root_list}" \
    "${SBATCH_DIR}/control_v2_merge_collection.sbatch")"
  merge_job="${merge_job%%;*}"
  job_ids+=("${merge_job}")
  COLLECTION_GROUP_JOB="${merge_job}"
  COLLECTION_GROUP_ROOT="${merge_parent}/lift_bottle_${merge_job}"
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
    if [[ ! -s "${root}/DONE" || "$(cat "${root}/DONE")" != "1" || ! -s "${root}/manifest.json" ]]; then
      echo "reused smoke shard is incomplete: ${root}" >&2
      exit 2
    fi
  done
  local canonical_roots
  canonical_roots="$(IFS=:; echo "${roots[*]}")"
  local merge_job
  merge_job="$(sbatch --parsable \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_PARENT=${merge_parent},COLLECTION_MODE=smoke,SHARD_ROOTS=${canonical_roots}" \
    "${SBATCH_DIR}/control_v2_merge_collection.sbatch")"
  merge_job="${merge_job%%;*}"
  job_ids+=("${merge_job}")
  COLLECTION_GROUP_JOB="${merge_job}"
  COLLECTION_GROUP_ROOT="${merge_parent}/lift_bottle_${merge_job}"
}

if [[ -n "${PREPARED_STORE:-}" ]]; then
  if [[ -n "${SMOKE_SHARD_ROOTS:-}" ]]; then
    echo "PREPARED_STORE and SMOKE_SHARD_ROOTS are mutually exclusive" >&2
    exit 2
  fi
  store="${PREPARED_STORE}"
  if [[ ! -d "${store}" ]]; then
    echo "PREPARED_STORE is not a Zarr directory: ${store}" >&2
    exit 2
  fi
  echo "using pre-qualified V3 store ${store}"
else
  COLLECTION_GROUP_JOB=""
  COLLECTION_GROUP_ROOT=""
  if [[ -n "${SMOKE_SHARD_ROOTS:-}" ]]; then
    submit_existing_smoke_merge "${SMOKE_SHARD_ROOTS}" "${COLLECTION_PARENT}/smoke"
  else
    submit_collection_group \
      smoke 1 4 "${COLLECTION_PARENT}/smoke_shards" "${COLLECTION_PARENT}/smoke" ""
  fi
  collect_smoke_job="${COLLECTION_GROUP_JOB}"
  submit_collection_group \
    production 250 1000 "${COLLECTION_PARENT}/shards" "${COLLECTION_PARENT}" "afterok:${collect_smoke_job}"
  collect_job="${COLLECTION_GROUP_JOB}"
  collection_root="${COLLECTION_GROUP_ROOT}"

  prep_job="$(sbatch --parsable --dependency="afterok:${collect_job}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COLLECTION_ROOT=${collection_root},PREP_PARENT=${PREP_PARENT}" \
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
  train_job="$(sbatch --parsable --dependency="afterok:${stats_job}" \
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
  --nodelist="${EVAL_NODE}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${final_run},V2_STEP=,PAIR_PARENT=${PAIR_PARENT}/smoke,EVAL_MODE=student_gate,TOTAL=1,START_SEED=900000,MIN_SUCCESSES=0,EVAL_NODE=${EVAL_NODE}" \
  "${SBATCH_DIR}/control_v2_closedloop.sbatch")"
smoke_job="${smoke_job%%;*}"
job_ids+=("${smoke_job}")

gate20_job="$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  --nodelist="${EVAL_NODE}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${final_run},V2_STEP=,PAIR_PARENT=${PAIR_PARENT}/gate20,EVAL_MODE=student_gate,TOTAL=20,START_SEED=900100,MIN_SUCCESSES=19,EVAL_NODE=${EVAL_NODE}" \
  "${SBATCH_DIR}/control_v2_closedloop.sbatch")"
gate20_job="${gate20_job%%;*}"
job_ids+=("${gate20_job}")

eval_job="$(sbatch --parsable --dependency="afterok:${gate20_job}" \
  --nodelist="${EVAL_NODE}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},V2_RUN=${final_run},V2_STEP=,PAIR_PARENT=${PAIR_PARENT}/paired100,EVAL_MODE=paired,TOTAL=100,START_SEED=1000000,EVAL_NODE=${EVAL_NODE}" \
  "${SBATCH_DIR}/control_v2_closedloop.sbatch")"
eval_job="${eval_job%%;*}"
job_ids+=("${eval_job}")

manifest="${REPO_ROOT}/logs/control_v2_pipeline_${PIPELINE_ID}.json"
export CONTROL_V2_PIPELINE_ID="${PIPELINE_ID}" CONTROL_V2_PIPELINE_JOBS="${job_ids[*]}"
export CONTROL_V2_PIPELINE_STORE="${store}" CONTROL_V2_PIPELINE_FINAL_RUN="${final_run}"
export CONTROL_V2_PIPELINE_FINAL_STEP="${final_step}" CONTROL_V2_PIPELINE_MANIFEST="${manifest}"
export CONTROL_V2_PIPELINE_SMOKE_SHARDS="${SMOKE_SHARD_ROOTS:-}"
python3 - <<'PY'
import json, os, pathlib, time

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
    "submitted_unix": time.time(),
}
path = pathlib.Path(os.environ["CONTROL_V2_PIPELINE_MANIFEST"])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

submission_complete=1
echo "submitted pipeline=${PIPELINE_ID} jobs=${job_ids[*]}"
echo "manifest=${manifest}"
echo "monitor: bash ${SBATCH_DIR}/control_v2_status.sh ${job_ids[*]}"
