#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
# Stats uses no accelerator; current cluster policy rejects CPU-only jobs from GPU partitions.
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
# CPU nodes expose about 246 GiB, so keep the request within their schedulable envelope.
#SBATCH --mem=220G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=mot-control-v2-stats
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# Fit train-only state/action statistics and freeze the stage's complete configuration. Run this
# separately for head20k, adapt10k, and recovery5k because each immutable stage has its own run.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs}"
EXP_NAME="${EXP_NAME:?set EXP_NAME}"
RUN_DIR="${RUN_ROOT}/mot_jepa_control_v2/${EXP_NAME}"
PRETRAINED_RUN="${PRETRAINED_RUN:?set PRETRAINED_RUN to the qualified 100k backbone run}"
STORE_GLOB="${STORE_GLOB:?set STORE_GLOB to converted V3 lift_bottle Zarr stores}"
STEPS="${STEPS:?set STEPS (20000 head, 10000 adapt, or 5000 recovery)}"
BACKBONE_MODE="${BACKBONE_MODE:?set BACKBONE_MODE to frozen or last_blocks}"
INIT_RUN="${INIT_RUN:-}"
INIT_STEP="${INIT_STEP:-}"
SOURCE_PROVENANCE="${RUN_DIR}/STATS_SOURCE_PROVENANCE.json"

case "${BACKBONE_MODE}" in
  frozen|last_blocks) ;;
  *) echo "invalid BACKBONE_MODE=${BACKBONE_MODE}" >&2; exit 2 ;;
esac
if [[ "${BACKBONE_MODE}" == "last_blocks" && -z "${INIT_RUN}" ]]; then
  echo "last_blocks stages require a completed INIT_RUN" >&2
  exit 2
fi
if [[ -f "${RUN_DIR}/STATS_DONE" ]]; then
  echo "[$(date -Is)] STATS_DONE already present for ${RUN_DIR}"
  exit 0
fi
if [[ -f "${RUN_DIR}/DONE" ]]; then
  echo "training DONE exists without STATS_DONE; refuse inconsistent run ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${REPO_ROOT}/logs" "${RUN_DIR}"
PYCACHE_ROOT="${RUN_DIR}/.empty_pycache_stats_${SLURM_JOB_ID}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"
export REPO_ROOT RUN_DIR SOURCE_PROVENANCE SLURM_JOB_ID

export STORE_GLOB
prepared_source_provenance=()
prepared_data_provenance=()
prepared_manifests=()
while IFS=$'\t' read -r source_marker data_marker prep_manifest; do
  prepared_source_provenance+=("${source_marker}")
  prepared_data_provenance+=("${data_marker}")
  prepared_manifests+=("${prep_manifest}")
done < <("${REPO_ROOT}/.venv/bin/python" - <<'PY'
import glob, hashlib, json, os, pathlib

roots = set()
for raw in glob.glob(os.environ["STORE_GLOB"]):
    path = pathlib.Path(raw).resolve()
    root = next(
        (
            parent
            for parent in path.parents
            if (parent / "source_provenance.json").is_file()
            and (parent / "store_content_provenance.json").is_file()
            and (parent / "manifest.json").is_file()
        ),
        None,
    )
    if root is None:
        raise SystemExit(f"store lacks prepared-data provenance closure: {path}")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_marker = (root / "store_content_provenance.json").resolve()
    if pathlib.Path(manifest.get("store", "")).resolve() != path:
        raise SystemExit(f"prepared manifest store differs from requested store: {path}")
    if pathlib.Path(manifest.get("store_content_provenance", "")).resolve() != data_marker:
        raise SystemExit(f"prepared manifest points to the wrong content provenance: {path}")
    if hashlib.sha256(data_marker.read_bytes()).hexdigest() != manifest.get("store_content_manifest_sha256"):
        raise SystemExit(f"derived-store content manifest digest differs: {path}")
    content = json.loads(data_marker.read_text())
    if content.get("content_sha256") != manifest.get("store_content_sha256"):
        raise SystemExit(f"derived-store content digest differs: {path}")
    roots.add(root)
for root in sorted(roots):
    print(root / "source_provenance.json", root / "store_content_provenance.json", root / "manifest.json", sep="\t")
PY
)
if [[ "${#prepared_source_provenance[@]}" -eq 0 ]]; then
  echo "STORE_GLOB resolved no prepared-data provenance" >&2
  exit 2
fi
verify_prepared_data() {
  local index
  for index in "${!prepared_source_provenance[@]}"; do
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
      --verify "${prepared_source_provenance[${index}]}"
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
      --verify-stat-only "${prepared_data_provenance[${index}]}"
  done
}
verify_prepared_data

init_source_paths=()
if [[ -n "${INIT_RUN}" ]]; then
  mapfile -t init_source_paths < <("${REPO_ROOT}/.venv/bin/python" - "${INIT_RUN}" "${INIT_STEP}" <<'PY'
import json, pathlib, sys

run = pathlib.Path(sys.argv[1]).resolve()
requested_step = sys.argv[2]
best = json.loads((run / "BEST_VALIDATION.json").read_text())
step = int(best["step"])
if requested_step and int(requested_step) != step:
    raise SystemExit(f"INIT_STEP {requested_step} differs from selected step {step}")
checkpoint = run / "checkpoints" / f"best_validation_{step}"
if pathlib.Path(best["checkpoint"]).resolve() != checkpoint.resolve():
    raise SystemExit("INIT_RUN BEST_VALIDATION checkpoint path is not canonical")
paths = [
    run / "DONE",
    run / "STATS_DONE",
    run / "checkpoints/latest",
    run / "BEST_VALIDATION.json",
    run / "VALIDATION_SAMPLES.json",
    run / "run_config.json",
    run / "artifact.json",
    run / "normalization.json",
    run / "STATS_SOURCE_PROVENANCE.json",
    run / "TRAIN_SOURCE_PROVENANCE.json",
    checkpoint,
    *sorted(run.glob("STATS_JOB_DONE.*.json")),
    *sorted(run.glob("TRAIN_JOB_DONE.*.json")),
]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit(f"INIT_RUN closure is incomplete: {missing}")
for path in paths:
    print(path)
PY
  )
  if [[ "${#init_source_paths[@]}" -eq 0 ]]; then
    echo "failed to resolve immutable INIT_RUN closure: ${INIT_RUN}" >&2
    exit 2
  fi
fi

source_paths=(
  "${REPO_ROOT}/src/openpi"
  "${REPO_ROOT}/scripts/mot_jepa_train.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_stats.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_train.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_stats.sbatch"
  "${REPO_ROOT}/data_processing/parse_data_module/parse_data_univtac.py"
  "${REPO_ROOT}/scripts/mot_jepa_build_clip_zarr.py"
  "${REPO_ROOT}/UniVTAC/task_config/mot_control_v2.yml"
  "${PRETRAINED_RUN}/checkpoints/100000/metadata.pt"
  "${PRETRAINED_RUN}/checkpoints/100000/teacher_ema.pt"
  "${PRETRAINED_RUN}/checkpoints/100000/train_config.json"
  "${PRETRAINED_RUN}/checkpoints/latest"
  "${prepared_source_provenance[@]}"
  "${prepared_data_provenance[@]}"
  "${prepared_manifests[@]}"
  "${init_source_paths[@]}"
)
if [[ -f "${SOURCE_PROVENANCE}" ]]; then
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
    --verify "${SOURCE_PROVENANCE}"
else
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
    --output "${SOURCE_PROVENANCE}" "${source_paths[@]}"
fi

heartbeat_pid=""
completed=0
_finish() {
  rc=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  rm -f "${RUN_DIR}/HEARTBEAT.stats.${SLURM_JOB_ID}.json"
  if [[ "${completed}" -ne 1 ]]; then
    printf '{"status":"FAILED","job_id":"%s","exit_code":%d}\n' \
      "${SLURM_JOB_ID}" "${rc}" > "${RUN_DIR}/FAILED.stats.${SLURM_JOB_ID}.json"
  fi
}
trap _finish EXIT

main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] || state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    printf '{"status":"RUNNING","job_id":"%s","scheduler_state":"%s","updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${state:-UNKNOWN}" "$(date -Is)" \
      > "${RUN_DIR}/HEARTBEAT.stats.${SLURM_JOB_ID}.json"
    sleep 60
  done
) &
heartbeat_pid=$!

args=(
  --run-root "${RUN_ROOT}"
  --exp-name "${EXP_NAME}"
  --pretrained-run "${PRETRAINED_RUN}"
  --pretrained-step 100000
  --store-glob "${STORE_GLOB}"
  --num-train-steps "${STEPS}"
  --backbone-train-mode "${BACKBONE_MODE}"
)
if [[ -n "${INIT_RUN}" ]]; then
  args+=(--init-run "${INIT_RUN}")
fi
if [[ -n "${INIT_STEP}" ]]; then
  args+=(--init-step "${INIT_STEP}")
fi

echo "[$(date -Is)] stats job=${SLURM_JOB_ID} run=${RUN_DIR} mode=${BACKBONE_MODE} steps=${STEPS}"
cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_control_v2_stats.py "${args[@]}"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  --verify "${SOURCE_PROVENANCE}"
verify_prepared_data

[[ -s "${RUN_DIR}/STATS_DONE" && -s "${RUN_DIR}/artifact.json" && -s "${RUN_DIR}/normalization.json" ]]
completed=1
source_digest="$("${REPO_ROOT}/.venv/bin/python" -c 'import json,os; print(json.load(open(os.environ["SOURCE_PROVENANCE"]))["content_sha256"])')"
printf '{"status":"DONE","job_id":"%s","source_content_sha256":"%s","updated":"%s"}\n' \
  "${SLURM_JOB_ID}" "${source_digest}" "$(date -Is)" > "${RUN_DIR}/STATS_JOB_DONE.${SLURM_JOB_ID}.json"
echo "[$(date -Is)] DONE stats=${RUN_DIR}"
