#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G
#SBATCH --job-name=mot-control-v2-data
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# Collect the independent scripted lift_bottle demonstrations for Control V2. This never reads an
# FTP1 checkpoint or rollout. A job-id-owned output root and fail-closed marker protocol make a
# partial collection impossible to confuse with a qualified input dataset.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
ASSETS="${ASSETS:-/lustre/fsw/portfolios/edgeai/users/chrislin/tacex-assets}"
EPISODES="${EPISODES:-250}"
WORKERS="${WORKERS:-1}"
GLOBAL_EPISODES="${GLOBAL_EPISODES:-1000}"
SHARD_RANK="${SHARD_RANK:-0}"
SHARD_COUNT="${SHARD_COUNT:-4}"
COLLECTION_MODE="${COLLECTION_MODE:-production}" # smoke | production
COLLECTION_PARENT="${COLLECTION_PARENT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/mot_control_v2_data}"
COLLECTION_ROOT="${COLLECTION_ROOT:-${COLLECTION_PARENT}/lift_bottle_${SLURM_JOB_ID}}"
RUNTIME_CONFIG="${COLLECTION_ROOT}/mot_control_v2.yml"
DATA_ROOT="${COLLECTION_ROOT}/raw/lift_bottle/mot_control_v2"
RAW_SHARD_ROOT="${COLLECTION_ROOT}/raw_shards"
SHARD_CONFIG_ROOT="${COLLECTION_ROOT}/shard_configs"
HEARTBEAT="${COLLECTION_ROOT}/HEARTBEAT.json"
SOURCE_PROVENANCE="${COLLECTION_ROOT}/source_provenance.json"
QUALIFIED_RUNTIME="${COLLECTION_ROOT}/qualified_runtime.json"
QUALIFIED_RUNTIME_AFTER="${COLLECTION_ROOT}/qualified_runtime_after.json"
GSMINI_ROBOT_ASSETS="${ASSETS}/Robots/Franka/GelSight_Mini"
GSMINI_SENSOR_ASSETS="${ASSETS}/Sensors/GelSight_Mini"

case "${COLLECTION_MODE}" in
  production)
    if [[ "${GLOBAL_EPISODES}" -ne 1000 || "${SHARD_COUNT}" -lt 1 || "${EPISODES}" -lt 1 || \
      $((EPISODES * SHARD_COUNT)) -ne 1000 ]]; then
      echo "production V2 sharding requires GLOBAL_EPISODES=1000, SHARD_COUNT>=1," \
        "EPISODES>=1, and EPISODES*SHARD_COUNT=1000" >&2
      exit 2
    fi
    ;;
  smoke)
    if [[ "${EPISODES}" -ne 1 || "${GLOBAL_EPISODES}" -ne 4 || "${SHARD_COUNT}" -ne 4 ]]; then
      echo "topology smoke sharding requires EPISODES=1, GLOBAL_EPISODES=4, and SHARD_COUNT=4" >&2
      exit 2
    fi
    ;;
  *) echo "COLLECTION_MODE must be smoke or production" >&2; exit 2 ;;
esac
if [[ "${WORKERS}" -ne 1 ]]; then
  echo "each qualified collection shard requires exactly WORKERS=1" >&2
  exit 2
fi
if [[ "${SHARD_RANK}" -lt 0 || "${SHARD_RANK}" -ge "${SHARD_COUNT}" ]]; then
  echo "qualified collection requires SHARD_RANK in [0, SHARD_COUNT)" >&2
  exit 2
fi
if [[ $((EPISODES * SHARD_COUNT)) -ne "${GLOBAL_EPISODES}" ]]; then
  echo "local shard quota does not close the global episode target" >&2
  exit 2
fi
HTTP_PORT_BASE="${HTTP_PORT_BASE:-$((10000 + (SLURM_JOB_ID % 10000) * 4))}"
if [[ "${HTTP_PORT_BASE}" -lt 1024 || $((HTTP_PORT_BASE + WORKERS - 1)) -gt 65535 ]]; then
  echo "invalid per-job Isaac HTTP port range starting at ${HTTP_PORT_BASE}" >&2
  exit 2
fi
if [[ -e "${COLLECTION_ROOT}" ]]; then
  echo "refuse existing collection root ${COLLECTION_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/UniVTAC/task_config/mot_control_v2.yml" ]]; then
  echo "missing committed collection config" >&2
  exit 2
fi
for path in "${IMAGE}" "${GSMINI_ROBOT_ASSETS}" "${GSMINI_SENSOR_ASSETS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing qualified collection runtime artifact ${path}" >&2
    exit 2
  fi
done

mkdir -p "${REPO_ROOT}/logs" "${COLLECTION_ROOT}" "${SHARD_CONFIG_ROOT}"
PYCACHE_ROOT="${COLLECTION_ROOT}/.empty_pycache_${SLURM_JOB_ID}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"
export REPO_ROOT COLLECTION_ROOT RUNTIME_CONFIG DATA_ROOT RAW_SHARD_ROOT SHARD_CONFIG_ROOT
export SOURCE_PROVENANCE QUALIFIED_RUNTIME QUALIFIED_RUNTIME_AFTER EPISODES WORKERS COLLECTION_MODE
export HTTP_PORT_BASE SLURM_JOB_ID GLOBAL_EPISODES SHARD_RANK SHARD_COUNT

python3 - <<'PY'
import os, pathlib, yaml

source = pathlib.Path(os.environ["REPO_ROOT"]) / "UniVTAC/task_config/mot_control_v2.yml"
payload = yaml.safe_load(source.read_text())
payload["episode_num"] = int(os.environ["EPISODES"])
payload["save_dir"] = str(pathlib.Path(os.environ["COLLECTION_ROOT"]) / "raw")
target = pathlib.Path(os.environ["RUNTIME_CONFIG"])
target.write_text(yaml.safe_dump(payload, sort_keys=True))

workers = int(os.environ["WORKERS"])
episodes = int(os.environ["EPISODES"])
base_seed = int(payload["start_seed"])
shard_root = pathlib.Path(os.environ["SHARD_CONFIG_ROOT"])
raw_shard_root = pathlib.Path(os.environ["RAW_SHARD_ROOT"])
for rank in range(workers):
    count = episodes
    if count < 1:
        raise SystemExit("every qualified collector task must receive at least one episode")
    shard = dict(payload)
    shard["episode_num"] = count
    # Attempts, not only successes, consume simulator seeds.  Interleaved stride-SHARD_COUNT streams
    # remain disjoint no matter how many failures any one task encounters.
    shard["start_seed"] = base_seed + int(os.environ["SHARD_RANK"])
    shard["seed_step"] = int(os.environ["SHARD_COUNT"])
    shard["save_dir"] = str(raw_shard_root / f"worker_{rank}")
    shard_dir = shard_root / f"worker_{rank}"
    shard_dir.mkdir(parents=True, exist_ok=False)
    (shard_dir / "mot_control_v2.yml").write_text(yaml.safe_dump(shard, sort_keys=True))
PY

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py" \
  --image "${IMAGE}" \
  --assets "${ASSETS}" \
  --repo-root "${REPO_ROOT}" \
  --output "${QUALIFIED_RUNTIME}"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
  --output "${SOURCE_PROVENANCE}" \
  --stat-only "${IMAGE}" \
  --stat-only "${ASSETS}" \
  "${QUALIFIED_RUNTIME}" \
  "${RUNTIME_CONFIG}" \
  "${SHARD_CONFIG_ROOT}" \
  "${REPO_ROOT}/UniVTAC/task_config/mot_control_v2.yml" \
  "${REPO_ROOT}/UniVTAC/scripts/parallel_collect_data.py" \
  "${REPO_ROOT}/UniVTAC/envs" \
  "${REPO_ROOT}/UniVTAC/assets" \
  "${REPO_ROOT}/UniVTAC/policy/task_settings.json" \
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex/tacex" \
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/__init__.py" \
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors" \
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/robots" \
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/config/extension.toml" \
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_tasks/tacex_tasks" \
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc" \
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py" \
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py" \
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch" \
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch"

main_pid=$$
heartbeat_pid=""
completed=0
_finish() {
  rc=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  if [[ "${completed}" -ne 1 ]]; then
    export CONTROL_V2_EXIT_CODE="${rc}"
    python3 - <<'PY'
import json, os, pathlib, time

root = pathlib.Path(os.environ["COLLECTION_ROOT"])
payload = {
    "status": "FAILED",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "exit_code": int(os.environ["CONTROL_V2_EXIT_CODE"]),
    "updated_unix": time.time(),
}
(root / "FAILED.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  fi
}
trap _finish EXIT

(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    if [[ -z "${state}" ]]; then
      state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    fi
    export CONTROL_V2_JOB_STATE="${state:-UNKNOWN}"
    python3 - <<'PY'
import json, os, pathlib, tempfile, time

target = pathlib.Path(os.environ["COLLECTION_ROOT"]) / "HEARTBEAT.json"
payload = {
    "status": "RUNNING",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "scheduler_state": os.environ.get("CONTROL_V2_JOB_STATE", "UNKNOWN"),
    "updated_unix": time.time(),
}
fd, raw = tempfile.mkstemp(prefix=".heartbeat.", dir=target.parent)
with os.fdopen(fd, "w") as stream:
    json.dump(payload, stream, sort_keys=True)
    stream.write("\n")
os.replace(raw, target)
PY
    sleep 60
  done
) &
heartbeat_pid=$!

echo "[$(date -Is)] collect shard job=${SLURM_JOB_ID} rank=${SHARD_RANK}/${SHARD_COUNT} root=${COLLECTION_ROOT} episodes=${EPISODES} http_base=${HTTP_PORT_BASE}"
# A collection shard owns an independent one-GPU Slurm allocation and one Pyxis container.  This
# is the smallest topology on which both CUDA and Vulkan receive the same mounted physical GPU.
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=16 --kill-on-bad-exit=1 \
  --container-image="${IMAGE}" \
  --container-remap-root --container-writable \
  --container-mounts="/lustre:/lustre,${ASSETS}:${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data" \
  bash -lc '
    set -euo pipefail
    rank="${SLURM_PROCID}"
    config="${SHARD_CONFIG_ROOT}/worker_${rank}/mot_control_v2.yml"
    count="${EPISODES}"
    port=$((HTTP_PORT_BASE + rank))
    visible_gpus="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | awk "NF {count++} END {print count+0}")"
    if [[ "${visible_gpus}" -ne 1 ]]; then
      echo "collector task ${rank} must see exactly one mounted GPU; found ${visible_gpus}" >&2
      exit 2
    fi
    gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | awk "NF {print; exit}")"
    if [[ -z "${gpu_uuid}" ]]; then
      echo "collector task ${rank} could not resolve its physical GPU UUID" >&2
      exit 2
    fi
    topology_record="${COLLECTION_ROOT}/GPU_TOPOLOGY.txt"
    printf "%s %s %s\n" "${SHARD_RANK}" "${gpu_uuid}" "${port}" > "${topology_record}.tmp"
    mv "${topology_record}.tmp" "${topology_record}"
    echo "collector task=${rank} shard_rank=${SHARD_RANK} gpu_uuid=${gpu_uuid} port=${port} config=${config} seed_step=${SHARD_COUNT}"
    cd "${REPO_ROOT}/UniVTAC"
    /isaac-sim/python.sh scripts/parallel_collect_data.py lift_bottle "${config}" \
      --workers=1 --episodes="${count}" --gpu="" --http-port-base="${port}" --seed-step="${SHARD_COUNT}"
  '

echo "[$(date -Is)] committing independently isolated collection shard"
python3 - <<'PY'
import json, os, pathlib

import yaml

root = pathlib.Path(os.environ["COLLECTION_ROOT"])
shard_config_root = pathlib.Path(os.environ["SHARD_CONFIG_ROOT"])
raw_shard_root = pathlib.Path(os.environ["RAW_SHARD_ROOT"])
data_root = pathlib.Path(os.environ["DATA_ROOT"])
data_root.mkdir(parents=True, exist_ok=False)
records = []
seen_names = set()
base_seed = int(yaml.safe_load((root / "mot_control_v2.yml").read_text())["start_seed"])
global_rank = int(os.environ["SHARD_RANK"])
global_count = int(os.environ["SHARD_COUNT"])
topology_fields = (root / "GPU_TOPOLOGY.txt").read_text().split()
if len(topology_fields) != 3 or int(topology_fields[0]) != global_rank:
    raise SystemExit("GPU topology record does not match the requested shard rank")
gpu_uuid, http_port = topology_fields[1], int(topology_fields[2])
if not gpu_uuid.startswith("GPU-") or http_port != int(os.environ["HTTP_PORT_BASE"]):
    raise SystemExit("GPU topology record has an invalid UUID or HTTP port")
gpu_topology = {
    "schema_version": 1,
    "job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_JOB_NODELIST"),
    "shard_rank": global_rank,
    "gpu_uuid": gpu_uuid,
    "http_port": http_port,
}
(root / "GPU_TOPOLOGY.json").write_text(json.dumps(gpu_topology, indent=2, sort_keys=True) + "\n")
for rank in range(int(os.environ["WORKERS"])):
    config_path = shard_config_root / f"worker_{rank}/mot_control_v2.yml"
    config = yaml.safe_load(config_path.read_text())
    source_root = raw_shard_root / f"worker_{rank}/lift_bottle/mot_control_v2"
    paths = sorted(source_root.rglob("*.hdf5"))
    expected = int(config["episode_num"])
    if int(config["start_seed"]) != base_seed + global_rank:
        raise SystemExit("shard config start_seed differs from base_seed + SHARD_RANK")
    if int(config.get("seed_step", -1)) != global_count:
        raise SystemExit("shard config seed_step differs from SHARD_COUNT")
    if len(paths) != expected:
        raise SystemExit(f"worker_{rank} produced {len(paths)} HDF5 episodes; expected {expected}")
    for source in paths:
        if source.name in seen_names:
            raise SystemExit(f"duplicate seed filename across collection shards: {source.name}")
        try:
            seed = int(source.stem)
        except ValueError as error:
            raise SystemExit(f"collected filename is not its integer simulator seed: {source.name}") from error
        if seed < base_seed + global_rank or (seed - base_seed) % global_count != global_rank:
            raise SystemExit(f"shard {global_rank} produced seed {seed} outside its disjoint stride stream")
        seen_names.add(source.name)
        target = data_root / source.name
        os.link(source, target)
        records.append(
            {
                "rank": global_rank,
                "source": str(source.resolve()),
                "target": str(target.resolve()),
                "start_seed": int(config["start_seed"]),
                "seed": seed,
                "seed_step": int(config["seed_step"]),
            }
        )
if len(records) != int(os.environ["EPISODES"]):
    raise SystemExit(f"merged {len(records)} episodes; expected {os.environ['EPISODES']}")
(root / "SHARD_COLLECTION.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "workers": int(os.environ["WORKERS"]),
            "shard_rank": global_rank,
            "shard_count": global_count,
            "global_episodes": int(os.environ["GLOBAL_EPISODES"]),
            "gpu_topology": gpu_topology,
            "http_port_base": int(os.environ["HTTP_PORT_BASE"]),
            "episodes": records,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

echo "[$(date -Is)] requalifying runtime after collection"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py" \
  --image "${IMAGE}" \
  --assets "${ASSETS}" \
  --repo-root "${REPO_ROOT}" \
  --output "${QUALIFIED_RUNTIME_AFTER}"
cmp --silent "${QUALIFIED_RUNTIME}" "${QUALIFIED_RUNTIME_AFTER}" || {
  echo "qualified collection runtime changed while data was generated" >&2
  exit 2
}

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
  --verify "${SOURCE_PROVENANCE}"

python3 - <<'PY'
import hashlib, json, os, pathlib, time

import h5py

root = pathlib.Path(os.environ["COLLECTION_ROOT"])
data = pathlib.Path(os.environ["DATA_ROOT"])
paths = sorted(data.rglob("*.hdf5"))
if len(paths) != int(os.environ["EPISODES"]):
    raise SystemExit(f"found {len(paths)} completed HDF5 episodes; expected exactly {os.environ['EPISODES']}")
required = {
    "embodiment/joint",
    "embodiment/command",
    "control/phase",
    "control/contact",
    "step",
    "observation/head/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
}
rows = []
parser_hdf5 = root / "parser_input/lift_bottle/demo/hdf5"
parser_hdf5.mkdir(parents=True, exist_ok=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


for path in paths:
    with h5py.File(path, "r") as episode:
        missing = sorted(name for name in required if name not in episode)
        if missing:
            raise SystemExit(f"{path} lacks V2 datasets {missing}")
        length = int(episode["embodiment/joint"].shape[0])
        if length < 63 or any(int(episode[name].shape[0]) != length for name in required):
            raise SystemExit(f"{path} has inconsistent arrays or fewer than 63 raw rows")
    stat = path.stat()
    link = parser_hdf5 / path.name
    if link.exists():
        raise SystemExit(f"duplicate collected seed filename {path.name}")
    os.link(path, link)
    rows.append(
        {
            "path": str(path.resolve()),
            "parser_path": str(link.resolve()),
            "logical_path": path.relative_to(data).as_posix(),
            "bytes": stat.st_size,
            "frames": length,
            "sha256": file_sha256(path),
        }
    )
config = pathlib.Path(os.environ["RUNTIME_CONFIG"])
source_provenance = pathlib.Path(os.environ["SOURCE_PROVENANCE"])
source_manifest = json.loads(source_provenance.read_text())
qualified_runtime = pathlib.Path(os.environ["QUALIFIED_RUNTIME"])
qualified_runtime_after = pathlib.Path(os.environ["QUALIFIED_RUNTIME_AFTER"])
shard_collection = root / "SHARD_COLLECTION.json"
gpu_topology = root / "GPU_TOPOLOGY.json"
manifest = {
    "schema_version": 3,
    "status": "DONE",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "requested_episodes": int(os.environ["EPISODES"]),
    "collection_mode": os.environ["COLLECTION_MODE"],
    "global_episodes": int(os.environ["GLOBAL_EPISODES"]),
    "shard_rank": int(os.environ["SHARD_RANK"]),
    "shard_count": int(os.environ["SHARD_COUNT"]),
    "saved_episodes": len(paths),
    "runtime_config": str(config),
    "runtime_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "source_provenance": str(source_provenance),
    "source_provenance_sha256": hashlib.sha256(source_provenance.read_bytes()).hexdigest(),
    "source_content_sha256": source_manifest["content_sha256"],
    "qualified_runtime_sha256": hashlib.sha256(qualified_runtime.read_bytes()).hexdigest(),
    "qualified_runtime_after_sha256": hashlib.sha256(qualified_runtime_after.read_bytes()).hexdigest(),
    "shard_collection": str(shard_collection),
    "shard_collection_sha256": hashlib.sha256(shard_collection.read_bytes()).hexdigest(),
    "gpu_topology": str(gpu_topology),
    "gpu_topology_sha256": hashlib.sha256(gpu_topology.read_bytes()).hexdigest(),
    "data_root": str(data),
    "parser_input_root": str(root / "parser_input"),
    "episodes": rows,
    "episode_content_sha256": hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
    "completed_unix": time.time(),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({key: manifest[key] for key in ("status", "job_id", "saved_episodes", "data_root")}, sort_keys=True))
PY

echo "[$(date -Is)] rehashing committed HDF5 collection and parser hardlinks"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
  --verify-collection "${COLLECTION_ROOT}/manifest.json"
printf '%s\n' "${EPISODES}" > "${COLLECTION_ROOT}/DONE"

completed=1
rm -f "${HEARTBEAT}"
echo "[$(date -Is)] DONE collection=${COLLECTION_ROOT} episodes=$(cat "${COLLECTION_ROOT}/DONE")"
