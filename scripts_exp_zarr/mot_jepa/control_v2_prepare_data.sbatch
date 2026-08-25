#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
# The CPU partition's qualified nodes expose about 246 GiB each. Conversion is sequential and
# peaks well below this request; 500G is unschedulable there and must not be copied from GPU jobs.
#SBATCH --mem=220G
#SBATCH --job-name=mot-control-v2-data-prep
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# Convert the authoritative V3 HDF5 collection to FTP Zarr and then to the high-resolution
# MoT-Control clip layout. Outputs are job-owned and never overwrite an earlier conversion.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
COLLECTION_ROOT="${COLLECTION_ROOT:?set COLLECTION_ROOT from a completed control_v2_collect job}"
PREP_PARENT="${PREP_PARENT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/mot_control_v2_data_prepared}"
PREP_ROOT="${PREP_ROOT:-${PREP_PARENT}/lift_bottle_${SLURM_JOB_ID}}"
SOURCE_ROOT="${PREP_ROOT}/source"
SOURCE_DOMAIN="${SOURCE_ROOT}/UniVTAC_lift_bottle"
CLIPS_ROOT="${PREP_ROOT}/clips"
STORE="${CLIPS_ROOT}/UniVTAC_lift_bottle/lift_bottle_head.zarr"
SOURCE_PROVENANCE="${PREP_ROOT}/source_provenance.json"
DATA_CONTENT_PROVENANCE="${PREP_ROOT}/store_content_provenance.json"
GENERIC_PROVENANCE_SCRIPT="${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
COLLECTION_PROVENANCE_SCRIPT="${COLLECTION_PROVENANCE_SCRIPT:-${GENERIC_PROVENANCE_SCRIPT}}"

generic_provenance_real="$(realpath -e "${GENERIC_PROVENANCE_SCRIPT}")"
batch_provenance_real="$(realpath -e "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py")"
collection_provenance_real="$(realpath -e "${COLLECTION_PROVENANCE_SCRIPT}")"
case "${collection_provenance_real}" in
  "${generic_provenance_real}"|"${batch_provenance_real}") ;;
  *)
    echo "COLLECTION_PROVENANCE_SCRIPT must select a repository Control V2 verifier" >&2
    exit 2
    ;;
esac

if [[ ! -s "${COLLECTION_ROOT}/DONE" || ! -s "${COLLECTION_ROOT}/manifest.json" ]]; then
  echo "collection is not complete: ${COLLECTION_ROOT}" >&2
  exit 2
fi
if [[ -e "${PREP_ROOT}" ]]; then
  echo "refuse existing prepared-data root ${PREP_ROOT}" >&2
  exit 2
fi
mkdir -p "${REPO_ROOT}/logs" "${SOURCE_DOMAIN}" "${CLIPS_ROOT}"
PYCACHE_ROOT="${PREP_ROOT}/.empty_pycache_${SLURM_JOB_ID}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"
export REPO_ROOT COLLECTION_ROOT PREP_ROOT STORE SOURCE_PROVENANCE DATA_CONTENT_PROVENANCE SLURM_JOB_ID

heartbeat_pid=""
completed=0
_finish() {
  rc=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  rm -f "${PREP_ROOT}/HEARTBEAT.json"
  if [[ "${completed}" -ne 1 ]]; then
    printf '{"status":"FAILED","job_id":"%s","exit_code":%d,"updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${rc}" "$(date -Is)" > "${PREP_ROOT}/FAILED.json"
  fi
}
trap _finish EXIT

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_handoff.py" \
  --collection-root "${COLLECTION_ROOT}"
"${REPO_ROOT}/.venv/bin/python" "${GENERIC_PROVENANCE_SCRIPT}" \
  --verify "${COLLECTION_ROOT}/source_provenance.json"
"${REPO_ROOT}/.venv/bin/python" "${COLLECTION_PROVENANCE_SCRIPT}" \
  --verify-collection "${COLLECTION_ROOT}/manifest.json"

"${REPO_ROOT}/.venv/bin/python" "${GENERIC_PROVENANCE_SCRIPT}" \
  --output "${SOURCE_PROVENANCE}" \
  "${COLLECTION_ROOT}/DONE" \
  "${COLLECTION_ROOT}/manifest.json" \
  "${COLLECTION_ROOT}/source_provenance.json" \
  "${REPO_ROOT}/data_processing/parse_data_module/parse_data_univtac.py" \
  "${REPO_ROOT}/scripts/mot_jepa_build_clip_zarr.py" \
  "${REPO_ROOT}/scripts/mot_jepa_add_conditioning.py" \
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_handoff.py" \
  "${REPO_ROOT}/src/openpi" \
  "${REPO_ROOT}/data_processing/common" \
  "${GENERIC_PROVENANCE_SCRIPT}" \
  "${COLLECTION_PROVENANCE_SCRIPT}" \
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_prepare_data.sbatch"

main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] || state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    printf '{"status":"RUNNING","job_id":"%s","scheduler_state":"%s","updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${state:-UNKNOWN}" "$(date -Is)" > "${PREP_ROOT}/HEARTBEAT.json"
    sleep 60
  done
) &
heartbeat_pid=$!

export JAX_PLATFORMS=cpu USE_SWANLAB=false
echo "[$(date -Is)] HDF5 -> FTP Zarr"
cd "${REPO_ROOT}/data_processing"
"${REPO_ROOT}/.venv/bin/python" -m parse_data_module.parse_data_univtac \
  --base_dir "${COLLECTION_ROOT}/parser_input" \
  --save_dir "${SOURCE_DOMAIN}" \
  --task_list lift_bottle \
  --image_size 224

echo "[$(date -Is)] FTP Zarr -> high-resolution MoT clips"
cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_build_clip_zarr.py \
  --source "${SOURCE_ROOT}" \
  --output "${CLIPS_ROOT}" \
  --domains UniVTAC_lift_bottle \
  --video-size 224 \
  --gel-size 112 \
  --num-frames 16

echo "[$(date -Is)] adding proprioception, action mask, and instruction IDs"
"${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_add_conditioning.py \
  --source "${SOURCE_ROOT}" \
  --clips "${CLIPS_ROOT}" \
  --domains UniVTAC_lift_bottle

# This is the single expensive pass over every derived byte.  It includes all compressed video
# and GEL chunks.  Downstream jobs freeze this manifest itself and verify the complete file/stat
# tree, so they do not reread tens of GB at every stage.
echo "[$(date -Is)] hashing complete derived-store payload once"
"${REPO_ROOT}/.venv/bin/python" "${GENERIC_PROVENANCE_SCRIPT}" \
  --output "${DATA_CONTENT_PROVENANCE}" "${STORE}"

"${REPO_ROOT}/.venv/bin/python" "${GENERIC_PROVENANCE_SCRIPT}" \
  --verify "${SOURCE_PROVENANCE}"
"${REPO_ROOT}/.venv/bin/python" "${COLLECTION_PROVENANCE_SCRIPT}" \
  --verify-collection "${COLLECTION_ROOT}/manifest.json"

"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import hashlib, json, os, pathlib, time

import numpy as np
import zarr

store = pathlib.Path(os.environ["STORE"])
group = zarr.open_group(store, mode="r")
arrays = set(group["data"].array_keys())
required = {
    "video",
    "gel",
    "lowdim",
    "state",
    "command8",
    "command_valid",
    "phase_id",
    "contact",
    "contact_valid",
    "control_step",
    "control_step_valid",
}
if missing := sorted(required - arrays):
    raise SystemExit(f"derived V3 store lacks {missing}")
episodes = len(np.asarray(group["meta"]["episode_ends"][:]))
if episodes != 1000:
    raise SystemExit(f"derived store has {episodes} episodes, expected exactly 1000")
episode_ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
episode_lengths = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), episode_ends)))
if np.any(episode_lengths < 62):
    raise SystemExit(f"derived V3 episodes shorter than 62 rows: {episode_lengths[episode_lengths < 62].tolist()}")
collection_manifest = pathlib.Path(os.environ["COLLECTION_ROOT"]) / "manifest.json"
collection_payload = json.loads(collection_manifest.read_text())
collection_episodes = collection_payload.get("episodes")
if not isinstance(collection_episodes, list) or len(collection_episodes) != episodes:
    raise SystemExit("immutable collection manifest must contain exactly one record per derived episode")
manifest_seeds = []
manifest_frames = []
for record in collection_episodes:
    logical_path = record.get("logical_path") if isinstance(record, dict) else None
    if not isinstance(logical_path, str) or pathlib.Path(logical_path).suffix != ".hdf5":
        raise SystemExit(f"collection episode has invalid logical_path: {logical_path!r}")
    stem = pathlib.Path(logical_path).stem
    if not stem.isascii() or not stem.isdecimal():
        raise SystemExit(f"collection episode logical_path must have a numeric stem: {logical_path!r}")
    seed = int(stem)
    if seed > np.iinfo(np.int64).max:
        raise SystemExit(f"collection episode seed exceeds int64: {logical_path!r}")
    frames = record.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 1:
        raise SystemExit(f"collection episode has invalid frame count: {logical_path!r} frames={frames!r}")
    manifest_seeds.append(seed)
    manifest_frames.append(frames)
if len(set(manifest_seeds)) != 1000:
    raise SystemExit("immutable collection manifest must encode exactly 1000 unique episode seeds")
sorted_manifest_seeds = sorted(manifest_seeds)
if manifest_seeds != sorted_manifest_seeds:
    raise SystemExit("immutable collection manifest episodes are not ordered by numeric logical_path stem")
expected_lengths = np.asarray(manifest_frames, dtype=np.int64) - 1
if not np.array_equal(episode_lengths, expected_lengths):
    mismatched = np.flatnonzero(episode_lengths != expected_lengths)
    first = int(mismatched[0])
    raise SystemExit(
        "derived episode order/length does not close against immutable collection: "
        f"episode={first} seed={manifest_seeds[first]} derived_rows={episode_lengths[first]} "
        f"collection_frames={manifest_frames[first]}"
    )
meta_arrays = set(group["meta"].array_keys())
if missing := sorted({"action_mask", "instruction_id", "source_episode_seed"} - meta_arrays):
    raise SystemExit(f"derived V3 store lacks meta arrays {missing}")
raw_source_episode_seeds = group["meta"]["source_episode_seed"]
if np.dtype(raw_source_episode_seeds.dtype).kind not in "iu":
    raise SystemExit("meta/source_episode_seed must have an integer dtype")
source_episode_seeds = np.asarray(raw_source_episode_seeds[:])
if source_episode_seeds.shape != (episodes,):
    raise SystemExit(
        f"meta/source_episode_seed must have shape {(episodes,)}, got {source_episode_seeds.shape}"
    )
if (np.dtype(source_episode_seeds.dtype).kind == "i" and np.any(source_episode_seeds < 0)) or np.any(
    source_episode_seeds > np.iinfo(np.int64).max
):
    raise SystemExit("meta/source_episode_seed values must be non-negative int64-compatible integers")
source_episode_seeds = source_episode_seeds.astype(np.int64, copy=False)
if np.unique(source_episode_seeds).size != 1000:
    raise SystemExit("meta/source_episode_seed must contain exactly 1000 unique values")
expected_source_episode_seeds = np.asarray(sorted_manifest_seeds, dtype=np.int64)
if not np.array_equal(source_episode_seeds, expected_source_episode_seeds):
    mismatch = int(np.flatnonzero(source_episode_seeds != expected_source_episode_seeds)[0])
    raise SystemExit(
        "derived source-episode identity differs from immutable collection order: "
        f"episode={mismatch} derived={source_episode_seeds[mismatch]} "
        f"expected={expected_source_episode_seeds[mismatch]}"
    )
rows = int(group["data"]["state"].shape[0])
if tuple(group["data"]["state"].shape) != (rows, 120):
    raise SystemExit(f"data/state must have shape (T, 120), got {group['data']['state'].shape}")
for name in required - {"state"}:
    if int(group["data"][name].shape[0]) != rows:
        raise SystemExit(f"data/{name} is not time-aligned with data/state")
action_mask = np.asarray(group["meta"]["action_mask"][:], dtype=np.uint8)
if action_mask.shape != (120,) or not action_mask.any() or not set(np.unique(action_mask)) <= {0, 1}:
    raise SystemExit("meta/action_mask must be a non-empty binary length-120 mask")
instruction_ids = np.asarray(group["meta"]["instruction_id"][:], dtype=np.int64)
if instruction_ids.shape != (episodes,) or np.any(instruction_ids < 0):
    raise SystemExit("meta/instruction_id must contain one non-negative ID per episode")
if not np.asarray(group["data"]["command_valid"][:], dtype=bool).all():
    raise SystemExit("V3 collection contains non-authoritative command rows")
if not np.asarray(group["data"]["contact_valid"][:], dtype=bool).all():
    raise SystemExit("V3 collection contains non-authoritative contact rows")
if not np.asarray(group["data"]["control_step_valid"][:], dtype=bool).all():
    raise SystemExit("V3 collection contains non-authoritative control-step rows")
steps = np.asarray(group["data"]["control_step"][:], dtype=np.int64)
begin = 0
for raw_end in episode_ends:
    end = int(raw_end)
    if not np.all(np.diff(steps[begin:end]) == 1):
        raise SystemExit(f"episode rows [{begin},{end}) are not consecutive control steps")
    begin = end
phase = np.asarray(group["data"]["phase_id"][:], dtype=np.int64)
if set(np.unique(phase)) != {0, 1, 2, 3}:
    raise SystemExit(f"phase coverage differs from 0..3: {np.unique(phase).tolist()}")
source_provenance = pathlib.Path(os.environ["SOURCE_PROVENANCE"])
source_manifest = json.loads(source_provenance.read_text())
data_content_provenance = pathlib.Path(os.environ["DATA_CONTENT_PROVENANCE"])
data_content_manifest = json.loads(data_content_provenance.read_text())
payload = {
    "schema_version": 3,
    "status": "DONE",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "store": str(store),
    "episodes": episodes,
    "source_collection": str(pathlib.Path(os.environ["COLLECTION_ROOT"]).resolve()),
    "source_manifest_sha256": hashlib.sha256(collection_manifest.read_bytes()).hexdigest(),
    "source_provenance": str(source_provenance),
    "source_content_sha256": source_manifest["content_sha256"],
    "store_content_provenance": str(data_content_provenance),
    "store_content_manifest_sha256": hashlib.sha256(data_content_provenance.read_bytes()).hexdigest(),
    "store_content_sha256": data_content_manifest["content_sha256"],
    "store_file_count": data_content_manifest["file_count"],
    "completed_unix": time.time(),
}
root = pathlib.Path(os.environ["PREP_ROOT"])
(root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
(root / "DONE").write_text(str(store.resolve()) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

completed=1
echo "[$(date -Is)] DONE prepared store=${STORE}"
