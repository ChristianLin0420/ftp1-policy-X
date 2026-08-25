#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --job-name=mot-control-v2-merge
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# Merge independently allocated one-GPU collection shards into the one immutable collection
# consumed by V3 preparation. Smoke remains exactly four shards of one episode. Production accepts
# a validated shard count and equal per-shard quota whose product is exactly 1,000. The merge never
# accepts a partial shard, overlapping seed stream, changed source closure, or unqualified
# HDF5/parser hardlink.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
ASSETS="${ASSETS:-/lustre/fsw/portfolios/edgeai/users/chrislin/tacex-assets}"
COLLECTION_MODE="${COLLECTION_MODE:?set COLLECTION_MODE to smoke or production}"
SHARD_ROOTS="${SHARD_ROOTS:?set colon-separated SHARD_ROOTS from completed shard jobs}"
COLLECTION_PARENT="${COLLECTION_PARENT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/mot_control_v2_data}"
COLLECTION_ROOT="${COLLECTION_ROOT:-${COLLECTION_PARENT}/lift_bottle_${SLURM_JOB_ID}}"
DATA_ROOT="${COLLECTION_ROOT}/raw/lift_bottle/mot_control_v2"
PARSER_ROOT="${COLLECTION_ROOT}/parser_input/lift_bottle/demo/hdf5"
RUNTIME_CONFIG="${COLLECTION_ROOT}/mot_control_v2.yml"
SOURCE_PROVENANCE="${COLLECTION_ROOT}/source_provenance.json"
QUALIFIED_RUNTIME="${COLLECTION_ROOT}/qualified_runtime.json"
QUALIFIED_RUNTIME_AFTER="${COLLECTION_ROOT}/qualified_runtime_after.json"

case "${COLLECTION_MODE}" in
  smoke) EXPECTED_EPISODES=4 ;;
  production) EXPECTED_EPISODES=1000 ;;
  *) echo "COLLECTION_MODE must be smoke or production" >&2; exit 2 ;;
esac
if [[ "${SHARD_ROOTS}" == :* || "${SHARD_ROOTS}" == *: || "${SHARD_ROOTS}" == *::* ]]; then
  echo "SHARD_ROOTS contains an empty shard root" >&2
  exit 2
fi
IFS=: read -r -a shard_roots <<< "${SHARD_ROOTS}"
ACTUAL_SHARD_COUNT="${#shard_roots[@]}"
EXPECTED_SHARD_COUNT="${EXPECTED_SHARD_COUNT:-${ACTUAL_SHARD_COUNT}}"
if [[ ! "${EXPECTED_SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_SHARD_COUNT must be one positive integer" >&2
  exit 2
fi
if [[ "${ACTUAL_SHARD_COUNT}" -ne "${EXPECTED_SHARD_COUNT}" ]]; then
  echo "received ${ACTUAL_SHARD_COUNT} shard roots; expected ${EXPECTED_SHARD_COUNT}" >&2
  exit 2
fi
if [[ "${COLLECTION_MODE}" == smoke ]]; then
  EXPECTED_PER_SHARD="${EXPECTED_PER_SHARD:-1}"
  if [[ "${EXPECTED_SHARD_COUNT}" -ne 4 || "${EXPECTED_PER_SHARD}" != 1 ]]; then
    echo "smoke merge requires exactly EXPECTED_SHARD_COUNT=4 and EXPECTED_PER_SHARD=1" >&2
    exit 2
  fi
else
  if [[ "${EXPECTED_EPISODES}" -lt "${EXPECTED_SHARD_COUNT}" ]]; then
    echo "production merge cannot have more shards than episodes" >&2
    exit 2
  fi
  if [[ -z "${EXPECTED_PER_SHARD:-}" ]]; then
    if (( EXPECTED_EPISODES % EXPECTED_SHARD_COUNT != 0 )); then
      echo "production shard count ${EXPECTED_SHARD_COUNT} does not divide ${EXPECTED_EPISODES}" >&2
      exit 2
    fi
    EXPECTED_PER_SHARD=$((EXPECTED_EPISODES / EXPECTED_SHARD_COUNT))
  fi
fi
if [[ ! "${EXPECTED_PER_SHARD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_PER_SHARD must be one positive integer" >&2
  exit 2
fi
if (( EXPECTED_SHARD_COUNT * EXPECTED_PER_SHARD != EXPECTED_EPISODES )); then
  echo "shard count/quota product differs from exactly ${EXPECTED_EPISODES}" >&2
  exit 2
fi
if [[ -e "${COLLECTION_ROOT}" ]]; then
  echo "refuse existing merged collection root ${COLLECTION_ROOT}" >&2
  exit 2
fi
declare -A seen_shard_roots=()
for shard_root in "${shard_roots[@]}"; do
  if [[ "${shard_root}" != /* || ! -d "${shard_root}" ]]; then
    echo "shard root must be an existing absolute directory: ${shard_root}" >&2
    exit 2
  fi
  canonical_shard_root="$(readlink -f -- "${shard_root}")"
  if [[ -n "${seen_shard_roots[${canonical_shard_root}]+x}" ]]; then
    echo "duplicate shard root: ${canonical_shard_root}" >&2
    exit 2
  fi
  seen_shard_roots["${canonical_shard_root}"]=1
  for required in DONE manifest.json source_provenance.json SHARD_COLLECTION.json GPU_TOPOLOGY.json qualified_runtime.json qualified_runtime_after.json; do
    if [[ ! -s "${shard_root}/${required}" ]]; then
      echo "incomplete collection shard: ${shard_root}/${required}" >&2
      exit 2
    fi
  done
  done_quota="$(<"${shard_root}/DONE")"
  if [[ ! "${done_quota}" =~ ^[1-9][0-9]*$ || "${done_quota}" -ne "${EXPECTED_PER_SHARD}" ]]; then
    echo "shard DONE quota differs from ${EXPECTED_PER_SHARD}: ${shard_root}" >&2
    exit 2
  fi
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
    --verify "${shard_root}/source_provenance.json"
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
    --verify-collection "${shard_root}/manifest.json"
done

mkdir -p "${REPO_ROOT}/logs" "${COLLECTION_ROOT}"
PYCACHE_ROOT="${COLLECTION_ROOT}/.empty_pycache_${SLURM_JOB_ID}"
mkdir -p "${PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"
export REPO_ROOT COLLECTION_ROOT COLLECTION_MODE SHARD_ROOTS DATA_ROOT PARSER_ROOT RUNTIME_CONFIG
export SOURCE_PROVENANCE QUALIFIED_RUNTIME QUALIFIED_RUNTIME_AFTER EXPECTED_EPISODES EXPECTED_PER_SHARD
export EXPECTED_SHARD_COUNT SLURM_JOB_ID

python3 - <<'PY'
import os, pathlib, yaml

repo = pathlib.Path(os.environ["REPO_ROOT"])
root = pathlib.Path(os.environ["COLLECTION_ROOT"])
shards = [pathlib.Path(item) for item in os.environ["SHARD_ROOTS"].split(":")]
payload = yaml.safe_load((repo / "UniVTAC/task_config/mot_control_v2.yml").read_text())
payload["episode_num"] = int(os.environ["EXPECTED_EPISODES"])
payload["save_dir"] = str(root / "raw")
pathlib.Path(os.environ["RUNTIME_CONFIG"]).write_text(yaml.safe_dump(payload, sort_keys=True))
for name in ("qualified_runtime.json", "qualified_runtime_after.json"):
    source = shards[0] / name
    target = root / name
    os.link(source, target)
PY

provenance_paths=(
  "${RUNTIME_CONFIG}"
  "${QUALIFIED_RUNTIME}"
  "${QUALIFIED_RUNTIME_AFTER}"
  "${REPO_ROOT}/UniVTAC/task_config/mot_control_v2.yml"
  "${REPO_ROOT}/UniVTAC/scripts/parallel_collect_data.py"
  "${REPO_ROOT}/UniVTAC/envs"
  "${REPO_ROOT}/UniVTAC/assets"
  "${REPO_ROOT}/UniVTAC/policy/task_settings.json"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex/tacex"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/__init__.py"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/robots"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_assets/config/extension.toml"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_tasks/tacex_tasks"
  "${REPO_ROOT}/UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py"
  "${REPO_ROOT}/scripts/mot_jepa_control_v2_qualified_runtime.py"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch"
  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_merge_collection_batch.sbatch"
)
for shard_root in "${shard_roots[@]}"; do
  provenance_paths+=(
    "${shard_root}/DONE"
    "${shard_root}/manifest.json"
    "${shard_root}/source_provenance.json"
    "${shard_root}/SHARD_COLLECTION.json"
    "${shard_root}/GPU_TOPOLOGY.json"
  )
done
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
  --output "${SOURCE_PROVENANCE}" \
  --stat-only "${IMAGE}" \
  --stat-only "${ASSETS}" \
  "${provenance_paths[@]}"

completed=0
heartbeat_pid=""
_finish() {
  rc=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  rm -f "${COLLECTION_ROOT}/HEARTBEAT.json"
  if [[ "${completed}" -ne 1 ]]; then
    printf '{"status":"FAILED","job_id":"%s","exit_code":%d,"updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${rc}" "$(date -Is)" > "${COLLECTION_ROOT}/FAILED.json"
  fi
}
trap _finish EXIT

main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    state="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] || state="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    printf '{"status":"RUNNING","job_id":"%s","scheduler_state":"%s","updated":"%s"}\n' \
      "${SLURM_JOB_ID}" "${state:-UNKNOWN}" "$(date -Is)" > "${COLLECTION_ROOT}/HEARTBEAT.json"
    sleep 60
  done
) &
heartbeat_pid=$!

python3 - <<'PY'
import hashlib, json, os, pathlib, time

import h5py

root = pathlib.Path(os.environ["COLLECTION_ROOT"])
data_root = pathlib.Path(os.environ["DATA_ROOT"])
parser_root = pathlib.Path(os.environ["PARSER_ROOT"])
shard_roots = [pathlib.Path(item).resolve() for item in os.environ["SHARD_ROOTS"].split(":")]
manifests = [json.loads((path / "manifest.json").read_text()) for path in shard_roots]
ranked = sorted(zip(shard_roots, manifests, strict=True), key=lambda item: int(item[1]["shard_rank"]))
expected_shard_count = int(os.environ["EXPECTED_SHARD_COUNT"])
expected_ranks = list(range(expected_shard_count))
if [int(manifest["shard_rank"]) for _, manifest in ranked] != expected_ranks:
    raise SystemExit(f"collection shards do not cover ranks 0..{expected_shard_count - 1} exactly once")

mode = os.environ["COLLECTION_MODE"]
expected_total = int(os.environ["EXPECTED_EPISODES"])
expected_local = int(os.environ["EXPECTED_PER_SHARD"])
base_seed = None
static_source_records = None
static_source_roots = None
topologies = []
qualification_digests = set()


def canonical_shard_file(shard_root, manifest, field, filename):
    declared = manifest.get(field)
    if not isinstance(declared, str) or not declared:
        raise SystemExit(f"shard manifest lacks {field}: {shard_root}")
    raw = pathlib.Path(declared)
    if not raw.is_absolute() or ".." in raw.parts:
        raise SystemExit(f"shard manifest {field} contains a relative/traversing path: {raw}")
    expected = shard_root / filename
    if raw.resolve() != expected or not expected.is_file():
        raise SystemExit(f"shard manifest {field} is not canonical: {raw} != {expected}")
    return expected


for shard_root, manifest in ranked:
    required = {
        "schema_version": 3,
        "status": "DONE",
        "collection_mode": mode,
        "global_episodes": expected_total,
        "shard_count": expected_shard_count,
        "saved_episodes": expected_local,
        "requested_episodes": expected_local,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in required.items() if manifest.get(key) != value}
    if mismatches:
        raise SystemExit(f"invalid shard manifest {shard_root}: {mismatches}")
    topology = json.loads((shard_root / "GPU_TOPOLOGY.json").read_text())
    if (
        topology.get("shard_rank") != manifest["shard_rank"]
        or str(topology.get("job_id")) != str(manifest["job_id"])
        or not str(topology.get("gpu_uuid", "")).startswith("GPU-")
        or not isinstance(topology.get("node"), str)
        or not topology["node"]
    ):
        raise SystemExit(f"invalid GPU topology record for shard {shard_root}")
    topologies.append(topology)
    qualification_digests.add(
        (manifest["qualified_runtime_sha256"], manifest["qualified_runtime_after_sha256"])
    )
    runtime_config = canonical_shard_file(shard_root, manifest, "runtime_config", "mot_control_v2.yml")
    source_provenance = canonical_shard_file(
        shard_root, manifest, "source_provenance", "source_provenance.json"
    )
    if hashlib.sha256(runtime_config.read_bytes()).hexdigest() != manifest.get("runtime_config_sha256"):
        raise SystemExit(f"{runtime_config} differs from its shard manifest digest")
    if hashlib.sha256(source_provenance.read_bytes()).hexdigest() != manifest.get("source_provenance_sha256"):
        raise SystemExit(f"{source_provenance} differs from its shard manifest digest")
    for filename, digest_field in (
        ("qualified_runtime.json", "qualified_runtime_sha256"),
        ("qualified_runtime_after.json", "qualified_runtime_after_sha256"),
    ):
        actual = hashlib.sha256((shard_root / filename).read_bytes()).hexdigest()
        if actual != manifest[digest_field]:
            raise SystemExit(f"{shard_root}/{filename} differs from its manifest digest")

    provenance = json.loads(source_provenance.read_text())
    if provenance.get("content_sha256") != manifest.get("source_content_sha256"):
        raise SystemExit(f"{source_provenance} content id differs from its shard manifest")
    for required_source in (runtime_config, shard_root / "qualified_runtime.json"):
        if str(required_source.resolve()) not in provenance.get("files", {}):
            raise SystemExit(f"{source_provenance} does not close {required_source}")
    # Runtime configs and qualifications live under the job-owned shard root. Everything outside
    # that root is the shared source/container/assets closure and must be byte/stat identical.
    static = {path: record for path, record in provenance["files"].items() if not pathlib.Path(path).is_relative_to(shard_root)}
    static_roots = {
        str(pathlib.Path(record["path"]).resolve()): record["content_hashed"]
        for record in provenance["roots"]
        if not pathlib.Path(record["path"]).resolve().is_relative_to(shard_root)
    }
    if static_source_records is None:
        static_source_records = static
        static_source_roots = static_roots
    elif static != static_source_records:
        raise SystemExit("collection shard source/container/assets closures differ")
    elif static_roots != static_source_roots:
        raise SystemExit("collection shard source/container/assets root profiles differ")

    import yaml

    config_payload = yaml.safe_load(runtime_config.read_text())
    candidate_base = int(config_payload["start_seed"])
    base_seed = candidate_base if base_seed is None else base_seed
    if candidate_base != base_seed:
        raise SystemExit("collection shards use different base simulator seeds")

if len(qualification_digests) != 1:
    raise SystemExit("collection shards use different qualified container/assets runtimes")

# The merged collection must retain the exact live source/profile records frozen by its input
# shards. Re-snapshotting only a subset (or a changed tree) would otherwise produce a self-
# consistent merged manifest which no longer identifies the code/assets that generated the data.
merged_provenance = json.loads(pathlib.Path(os.environ["SOURCE_PROVENANCE"]).read_text())
merged_records = merged_provenance.get("files", {})
missing_records = sorted(set(static_source_records) - set(merged_records))
changed_records = sorted(
    path for path, record in static_source_records.items() if merged_records.get(path) != record
)
merged_roots = {
    str(pathlib.Path(record["path"]).resolve()): record["content_hashed"]
    for record in merged_provenance.get("roots", [])
}
missing_roots = sorted(
    path for path, content_hashed in static_source_roots.items() if merged_roots.get(path) != content_hashed
)
if missing_records or changed_records or missing_roots:
    raise SystemExit(
        "merged source provenance differs from its shard-frozen source profile: "
        f"missing_records={missing_records}, changed_records={changed_records}, missing_roots={missing_roots}"
    )

ports = [int(item["http_port"]) for item in topologies]
job_ids = [str(item["job_id"]) for item in topologies]
if len(set(ports)) != expected_shard_count or len(set(job_ids)) != expected_shard_count:
    raise SystemExit("collection shards do not have distinct jobs and HTTP ports")

data_root.mkdir(parents=True, exist_ok=False)
parser_root.mkdir(parents=True, exist_ok=False)
rows = []
seen_seeds = set()
shard_records = []
required_hdf5 = {
    "embodiment/joint",
    "embodiment/command",
    "control/phase",
    "control/contact",
    "step",
    "observation/head/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
}
for shard_root, manifest in ranked:
    rank = int(manifest["shard_rank"])
    shard_records.append(
        {
            "rank": rank,
            "quota": expected_local,
            "root": str(shard_root),
            "manifest": str(shard_root / "manifest.json"),
            "manifest_sha256": hashlib.sha256((shard_root / "manifest.json").read_bytes()).hexdigest(),
            "source_content_sha256": manifest["source_content_sha256"],
            "gpu_topology": topologies[rank],
        }
    )
    for source_row in manifest["episodes"]:
        source = pathlib.Path(source_row["path"]).resolve()
        seed = int(source.stem)
        if seed < base_seed + rank or seed in seen_seeds or (seed - base_seed) % expected_shard_count != rank:
            raise SystemExit(f"duplicate or wrong-residue simulator seed {seed} in shard {rank}")
        seen_seeds.add(seed)
        with h5py.File(source, "r") as episode:
            missing = sorted(name for name in required_hdf5 if name not in episode)
            if missing:
                raise SystemExit(f"{source} lacks V3 datasets {missing}")
            frames = int(episode["embodiment/joint"].shape[0])
            if frames != int(source_row["frames"]):
                raise SystemExit(f"{source} frame count changed during merge")
        raw = data_root / source.name
        parser = parser_root / source.name
        os.link(source, raw)
        os.link(raw, parser)
        stat = raw.stat()
        rows.append(
            {
                "path": str(raw.resolve()),
                "parser_path": str(parser.resolve()),
                "logical_path": raw.name,
                "bytes": stat.st_size,
                "frames": frames,
                "sha256": source_row["sha256"],
            }
        )
if len(rows) != expected_total:
    raise SystemExit(f"merged {len(rows)} episodes; expected exactly {expected_total}")
rows.sort(key=lambda row: int(pathlib.Path(row["path"]).stem))

gpu_topology_path = root / "GPU_TOPOLOGY.json"
gpu_topology_path.write_text(
    json.dumps({"schema_version": 1, "shards": topologies}, indent=2, sort_keys=True) + "\n"
)
shard_collection_path = root / "SHARD_COLLECTION.json"
shard_collection_path.write_text(
    json.dumps(
        {
            "schema_version": 2,
            "collection_mode": mode,
            "global_episodes": expected_total,
            "base_seed": base_seed,
            "shard_count": expected_shard_count,
            "seed_step": expected_shard_count,
            "shards": shard_records,
            "saved_seeds": sorted(seen_seeds),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
runtime_config = pathlib.Path(os.environ["RUNTIME_CONFIG"])
source_provenance = pathlib.Path(os.environ["SOURCE_PROVENANCE"])
source_payload = json.loads(source_provenance.read_text())
qualified = pathlib.Path(os.environ["QUALIFIED_RUNTIME"])
qualified_after = pathlib.Path(os.environ["QUALIFIED_RUNTIME_AFTER"])
manifest = {
    "schema_version": 3,
    "status": "DONE",
    "job_id": os.environ["SLURM_JOB_ID"],
    "requested_episodes": expected_total,
    "collection_mode": mode,
    "global_episodes": expected_total,
    "shard_rank": -1,
    "shard_count": expected_shard_count,
    "saved_episodes": len(rows),
    "runtime_config": str(runtime_config),
    "runtime_config_sha256": hashlib.sha256(runtime_config.read_bytes()).hexdigest(),
    "source_provenance": str(source_provenance),
    "source_provenance_sha256": hashlib.sha256(source_provenance.read_bytes()).hexdigest(),
    "source_content_sha256": source_payload["content_sha256"],
    "qualified_runtime_sha256": hashlib.sha256(qualified.read_bytes()).hexdigest(),
    "qualified_runtime_after_sha256": hashlib.sha256(qualified_after.read_bytes()).hexdigest(),
    "shard_collection": str(shard_collection_path),
    "shard_collection_sha256": hashlib.sha256(shard_collection_path.read_bytes()).hexdigest(),
    "gpu_topology": str(gpu_topology_path),
    "gpu_topology_sha256": hashlib.sha256(gpu_topology_path.read_bytes()).hexdigest(),
    "data_root": str(data_root),
    "parser_input_root": str(root / "parser_input"),
    "episodes": rows,
    "episode_content_sha256": hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
    "completed_unix": time.time(),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
  --verify "${SOURCE_PROVENANCE}"
"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/mot_jepa_control_v2_provenance_batch.py" \
  --verify-collection "${COLLECTION_ROOT}/manifest.json"
printf '%s\n' "${EXPECTED_EPISODES}" > "${COLLECTION_ROOT}/DONE"

completed=1
rm -f "${COLLECTION_ROOT}/HEARTBEAT.json"
echo "[$(date -Is)] DONE merged_collection=${COLLECTION_ROOT} episodes=${EXPECTED_EPISODES}"
