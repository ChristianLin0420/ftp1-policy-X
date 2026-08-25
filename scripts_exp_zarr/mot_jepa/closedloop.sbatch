#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=03:00:00
#SBATCH --nodes=1
# Isaac/TacEx renders on cuda:0 while the torch policy runs on cuda:1. Sharing one GPU previously
# caused VkResult ERROR_DEVICE_LOST and 60-second texture-upload stalls.
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --job-name=motjepa-closedloop
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#
# Matched UniVTAC closed-loop success rate for one of:
#   flow  = frozen 100k encoder + completed supervised flow head
#   ridge = frozen 100k encoder + corrected supervised ridge
#   ftp1  = task checkpoint through the original FTP-1 wrapper

set -euo pipefail

IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
ASSETS="${ASSETS:-/lustre/fsw/portfolios/edgeai/users/chrislin/tacex-assets}"
TASK="${TASK:-lift_bottle}"
DOMAIN="${DOMAIN:-UniVTAC_${TASK}}"
TOTAL="${TOTAL:-20}"
START_SEED="${START_SEED:-1000000}"
POLICY="${POLICY:-flow}" # flow | ridge | ftp1
FLOW_RUN="${FLOW_RUN:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/mot_jepa_policy_flowmatch/f100k_univtac_o2a1_h16_flowmatch_s42}"
FLOW_STEP="${FLOW_STEP:-20000}"
RIDGE="${RIDGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/eval/ridge_forecast100k_qknorm_s100000_obs2_act1_h16_v2.npz}"
SNAP="${SNAP:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/backbones/forecast100k_qknorm_s100000}"
SNAP_STEP="${SNAP_STEP:-100000}"
CKPT="${CKPT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-baselines/FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1/19999}"
FTP1_OVERLAY="${FTP1_OVERLAY:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runtime/univtac_ftp1_py310_v1}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runtime/openpi-data}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
EULER_STEPS="${EULER_STEPS:-10}"
ENSEMBLE_K="${ENSEMBLE_K:-0.01}"
POLICY_SAMPLE_SEED="${POLICY_SAMPLE_SEED:-0}"
RUN_SECONDS="${RUN_SECONDS:-9900}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1}"
PAIR_ID="${PAIR_ID:-f100k_${TASK}_s${START_SEED}_n${TOTAL}}"
RUN_ID="${RUN_ID:-${TASK}_s${START_SEED}_n${TOTAL}}"
SAVE_ROOT="${SAVE_ROOT:-${WORK}/UniVTAC/eval_result/motjepa_closedloop/${PAIR_ID}/${POLICY}_${SLURM_JOB_ID}}"
RESULT="${RESULT:-${WORK}/logs/_closedloop_${POLICY}_${SLURM_JOB_ID}.json}"
INPUT_DIR="${INPUT_DIR:-${WORK}/logs/_closedloop_input_${POLICY}_${SLURM_JOB_ID}}"
SOURCE_DIR="${SOURCE_DIR:-${WORK}/logs/_closedloop_source_${POLICY}_${SLURM_JOB_ID}}"
RUN_SH="${WORK}/logs/_closedloop_${SLURM_JOB_ID}.sh"

if [[ "${DOMAIN}" != "UniVTAC_${TASK}" ]]; then
  echo "DOMAIN=${DOMAIN} does not match TASK=${TASK}" >&2
  exit 2
fi
if [[ "${TOTAL}" -le 0 || "${EULER_STEPS}" -le 0 || "${NUM_SAMPLES}" -le 0 ]]; then
  echo "TOTAL, EULER_STEPS, and NUM_SAMPLES must be positive" >&2
  exit 2
fi
if [[ "${EVAL_CUDA_VISIBLE_DEVICES}" != *,* || "${EVAL_CUDA_VISIBLE_DEVICES}" == *,*,* ]]; then
  echo "EVAL_CUDA_VISIBLE_DEVICES must name exactly two logical devices" >&2
  exit 2
fi
case "${POLICY}" in
  flow|ridge|ftp1) ;;
  *) echo "POLICY must be flow, ridge, or ftp1; got ${POLICY}" >&2; exit 2 ;;
esac

mkdir -p "${WORK}/logs"
if [[ -e "${RESULT}" ]]; then
  echo "refuse stale verdict path ${RESULT}" >&2
  exit 2
fi
if [[ -d "${SAVE_ROOT}" ]] && [[ -n "$(find "${SAVE_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "refuse nonempty save root ${SAVE_ROOT}" >&2
  exit 2
fi
if [[ -e "${SOURCE_DIR}" ]]; then
  echo "refuse stale source snapshot path ${SOURCE_DIR}" >&2
  exit 2
fi
mkdir -p "${SAVE_ROOT}"

cat > "${RUN_SH}" <<'INNER'
set -euo pipefail
set -x
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/local/cuda-13.0
# Keep the evaluator on its stable logical renderer/policy pair. This is a guard on an actual
# two-GPU allocation, not a substitute for one: Vulkan ignores this CUDA mask and enumerates every
# physical device mounted by Pyxis, so an eight-GPU batch allocation is not renderer-safe.
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES_IN}"
export HEADLESS=1 ENABLE_CAMERAS=1
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_IN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${WORK_IN:?}" "${RESULT_IN:?}" "${POLICY_IN:?}" "${SAVE_ROOT_IN:?}" "${RUN_ID_IN:?}"
PY=/isaac-sim/python.sh
cd "${WORK_IN}/UniVTAC"

# Snapshot every Python source that can participate in policy inference or the common
# evaluator before importing it. The verifier below also rejects a source change while
# an evaluation is in flight.
"${PY}" - <<'PYEOF'
import hashlib, json, os, pathlib, shutil, time

work = pathlib.Path(os.environ["WORK_IN"]).resolve()
snapshot = pathlib.Path(os.environ["SOURCE_DIR_IN"]).resolve()
run_sh = pathlib.Path(os.environ["RUN_SH_IN"]).resolve()
roots = [
    work / "src" / "openpi",
    work / "UniVTAC" / "scripts",
    work / "UniVTAC" / "envs",
    work / "UniVTAC" / "encoder",
    work / "UniVTAC" / "policy",
    work / "UniVTAC" / "third_party" / "TacEx" / "source" / "tacex" / "tacex",
    work / "UniVTAC" / "third_party" / "TacEx" / "source" / "tacex_assets" / "tacex_assets" / "sensors",
    work / "UniVTAC" / "third_party" / "TacEx" / "source" / "tacex_assets" / "tacex_assets" / "robots",
    work / "UniVTAC" / "third_party" / "TacEx" / "source" / "tacex_tasks" / "tacex_tasks",
    work / "UniVTAC" / "third_party" / "TacEx" / "source" / "tacex_uipc" / "tacex_uipc",
]
files = sorted(
    {path for root in roots for path in root.rglob("*.py")}
    | {
        work / "scripts" / "eval_motjepa.py",
        work / "UniVTAC" / "task_config" / "demo.yml",
        work / "UniVTAC" / "policy" / "task_settings.json",
        work / "scripts_exp_zarr" / "mot_jepa" / "closedloop.sbatch",
        run_sh,
        *(
            work / "UniVTAC" / "third_party" / "TacEx" / "source" / "tacex_assets" / "tacex_assets"
        ).glob("*.py"),
    }
)
snapshot.mkdir(parents=True, exist_ok=False)
manifest = {
    "schema_version": 1,
    "created_unix_ns": time.time_ns(),
    "work": str(work),
    "files": {},
}
for source in files:
    if not source.is_file():
        raise SystemExit(f"missing inference source: {source}")
    source_relative = source.relative_to(work)
    # The generated script bytes are protocol source, but its live filename contains the
    # SLURM job id. Give it a canonical manifest key so matched jobs have the same content digest.
    relative = pathlib.Path("generated_run.sh") if source == run_sh else source_relative
    payload = source.read_bytes()
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest["files"][str(relative)] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "source_path": str(source_relative),
    }
content_index = "".join(
    f"{relative}\0{record['sha256']}\0{record['size']}\n"
    for relative, record in sorted(manifest["files"].items())
)
manifest["content_sha256"] = hashlib.sha256(content_index.encode()).hexdigest()
(snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(f"snapshotted {len(files)} Python sources to {snapshot}", flush=True)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


artifact_records = {}


def record_artifact(logical: str, path: pathlib.Path, expected_size: int, expected_sha256: str) -> None:
    path = path.resolve()
    stat = path.stat()
    digest = file_sha256(path)
    if stat.st_size != expected_size or digest != expected_sha256:
        raise SystemExit(
            f"artifact mismatch {logical}: size/sha={stat.st_size}/{digest}, "
            f"expected {expected_size}/{expected_sha256}"
        )
    artifact_records[logical] = {
        "path": str(path),
        "size": stat.st_size,
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def record_dynamic_artifact(logical: str, path: pathlib.Path) -> None:
    """Record a newly trained artifact whose digest does not predate this launcher."""
    path = path.resolve()
    stat = path.stat()
    digest = file_sha256(path)
    artifact_records[logical] = {
        "path": str(path),
        "size": stat.st_size,
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


record_artifact(
    "container/univtac_full.sqsh",
    pathlib.Path(os.environ["IMAGE_IN"]),
    38_711_164_928,
    "d8a83ddb9cf71fa37f4a39cc84a3244ec3a7f3c44419128a14eec29e39835af0",
)

# The rendered scene and contact simulation depend on both the external TacEx asset mount and
# local UniVTAC assets. Pin their full conservative closures, not only demo.yml.
asset_rows = []
external_assets = pathlib.Path(os.environ["ASSETS_IN"]).resolve()
for path in sorted(candidate for candidate in external_assets.rglob("*") if candidate.is_file()):
    logical = f"tacex-assets/{path.relative_to(external_assets).as_posix()}"
    stat = path.stat()
    digest = file_sha256(path)
    asset_rows.append((logical, digest, stat.st_size))
    artifact_records[logical] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }
local_univtac = work / "UniVTAC"
local_asset_files = [
    path
    for path in (local_univtac / "assets").rglob("*")
    if path.is_file() and not ({".thumbs", ".cache", "__pycache__"} & set(path.parts))
]
local_asset_files += [local_univtac / "task_config" / "demo.yml", local_univtac / "policy" / "task_settings.json"]
for path in sorted(local_asset_files):
    logical = f"UniVTAC/{path.relative_to(local_univtac).as_posix()}"
    stat = path.stat()
    digest = file_sha256(path)
    asset_rows.append((logical, digest, stat.st_size))
    artifact_records[logical] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }
external_rows = [row for row in asset_rows if row[0].startswith("tacex-assets/")]
local_rows = [row for row in asset_rows if row[0].startswith("UniVTAC/")]


def rows_digest(rows) -> str:
    payload = "".join(f"{digest}\t{size}\t{logical}\n" for logical, digest, size in sorted(rows)).encode()
    return hashlib.sha256(payload).hexdigest()


tree_checks = {
    "external_tacex": (external_rows, 237, 429_248_620, "534a909c5c09a21878d0569e52bd0fcf5137b6662511bb51ff9f38c9ae7af368"),
    "local_univtac": (local_rows, 65, 58_445_582, "66be4cda23ac6b11728186103f0b856273e2778f1f61a5bc65fa4a4ff57c1e67"),
    "combined": (asset_rows, 302, 487_694_202, "adf21e585c0ce730e3f3880cdb2521db0a3888b16cd0ed6b55251bfe159c3b54"),
}
asset_tree_digests = {}
for name, (rows, expected_count, expected_bytes, expected_digest) in tree_checks.items():
    actual = (len(rows), sum(size for _, _, size in rows), rows_digest(rows))
    expected = (expected_count, expected_bytes, expected_digest)
    if actual != expected:
        raise SystemExit(f"asset tree {name}={actual}, expected {expected}")
    asset_tree_digests[name] = actual[2]

backbone_root = pathlib.Path(os.environ["SNAP_IN"])
backbone_entries = {
    "backbone/checkpoints/100000/metadata.pt": (1_267, "202764d003c37c2247f60b72978c72ec7732ab7ff128b35470565349308089c7"),
    "backbone/checkpoints/100000/teacher_ema.pt": (119_633_043, "46af86aaafbd7c2b9a9c065653d5de917bc48d7e02726689a0e6aa4a666c461d"),
    "backbone/checkpoints/100000/train_config.json": (2_566, "9ab7df985458799738455dcc5bac85ea9ad05af12b699d88a1cad7a98aa993ea"),
    "backbone/checkpoints/latest": (7, "b80500a01f984c764f1a3b486622d0ef7cc5b13fa9bd57ec9015113eaf875597"),
}
if os.environ["POLICY_IN"] in {"flow", "ridge"}:
    for logical, (size, digest) in backbone_entries.items():
        relative = logical.removeprefix("backbone/")
        record_artifact(logical, backbone_root / relative, size, digest)

if os.environ["POLICY_IN"] == "flow":
    flow_root = pathlib.Path(os.environ["FLOW_RUN_IN"])
    released_flow = pathlib.Path(
        "/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/mot_jepa_policy_flowmatch/"
        "f100k_univtac_o2a1_h16_flowmatch_s42"
    ).resolve()
    if flow_root.resolve() == released_flow and int(os.environ["FLOW_STEP_IN"]) == 20_000:
        flow_entries = {
            "flow/DONE": (5, "876c9b16254e157d1eb645390dcfae6f29b9d3cd394e73a91de8ee5d0e67ee43"),
            "flow/action_stats.npz": (9_836, "227db1c9de15a1c976a6bfeabdd9ddf9280aba751c70fad6126dedadf235f7b7"),
            "flow/checkpoints/20000/metadata.pt": (1_267, "3d00775f07a139d435f733fab46d470a68c8981a3241ce0cbe1dbe67ec5c76c1"),
            "flow/checkpoints/20000/student.pt": (249_296_067, "c50f50ffb4eb5299e69c221ecee1ac60e168e78f77e9feaf61a1697d7a191d9e"),
            "flow/checkpoints/20000/train_config.json": (1_941, "80102462542a908d340857a9dcf447fb4e5dc05e93c7997ff33c813dadf3c792"),
            "flow/checkpoints/latest": (5, "876c9b16254e157d1eb645390dcfae6f29b9d3cd394e73a91de8ee5d0e67ee43"),
            "flow/run_config.json": (1_941, "80102462542a908d340857a9dcf447fb4e5dc05e93c7997ff33c813dadf3c792"),
        }
        for logical, (size, digest) in flow_entries.items():
            record_artifact(logical, flow_root / logical.removeprefix("flow/"), size, digest)
    else:
        flow_step = int(os.environ["FLOW_STEP_IN"])
        if int((flow_root / "DONE").read_text().strip()) != flow_step:
            raise SystemExit(f"adapted flow run is not DONE at requested step {flow_step}: {flow_root}")
        if int((flow_root / "checkpoints" / "latest").read_text().strip()) != flow_step:
            raise SystemExit(f"adapted flow latest pointer is not {flow_step}: {flow_root}")
        dynamic_paths = [
            flow_root / "DONE",
            flow_root / "action_stats.npz",
            flow_root / "run_config.json",
            flow_root / "checkpoints" / "latest",
            *sorted((flow_root / "checkpoints" / str(flow_step)).glob("*.pt")),
            flow_root / "checkpoints" / str(flow_step) / "train_config.json",
        ]
        for path in dynamic_paths:
            record_dynamic_artifact(f"flow/{path.relative_to(flow_root).as_posix()}", path)
elif os.environ["POLICY_IN"] == "ridge":
    record_artifact(
        "ridge/ridge_forecast100k_qknorm_s100000_obs2_act1_h16_v2.npz",
        pathlib.Path(os.environ["RIDGE_IN"]),
        566_656_722,
        "4a8969392cf937d30c959bb8ced0aa672a09f093b5bdce2b43b0ded319bd9d19",
    )
else:
    ftp1_root = pathlib.Path(os.environ["CKPT_IN"])
    ftp1_entries = {
        "hpt_tokenizer/GelSightMini_image_224_224_3.safetensors": (88_029_208, "657955e80a6888d0dd7a5731a5a3a279dcf65157fac165e82fa320f43b94c207"),
        "hpt_tokenizer/shared_image_chunk_encoder.safetensors": (256_757_104, "9453d7b49b9c3ff65858629c9d375aa37516801ad954fe2d28a2efd7794ab29e"),
        "metadata.pt": (6_131, "250240d490567c63b6ddc64d23e57cc3fe7e2c81253ea0bdd5e9e0c78e5ca658"),
        "model.safetensors": (7_904_019_864, "87534dd13263481b12daa303c12291cf2e72e07266f7d3056aad58f9426fe35a"),
        "model_config.json": (851, "157393b0680bcbf3d8da9c145c93178cc72beb0ad727ec0c9e316223cceace1a"),
        "normalization/UniVTAC_lift_bottle/contact_detection_thresholds.json": (1_868, "381ea08fe779444032e3f1a7650ef40b98256ee9089108237031e529be540d99"),
        "normalization/UniVTAC_lift_bottle/independent_norm_stats_all_t0_zscore.json": (10_056, "a109951bcaf284fe874ccb9f964602775a313229c7a033d48a9666e6f3e0ca4e"),
        "normalization/UniVTAC_lift_bottle/train_val_split.json": (548, "1f66069eaf37f0b23d43e166edc8667bd3c68a0c2c4f5811e8cf03a0fcf2c8ad"),
        "normalization/action_group_frequency_stats_train.json": (2_245, "8e1d66f57d4ec75b369de09a410bdcef4e74e6a6be023088ba2173b0b2a6c858"),
        "normalization/dataset_stats.json": (6_343, "ddfe26b2c5ec53027dea13402426e7a3bc8451839821b0c40a892ab245991d2a"),
        "normalization/norm_params_snapshot.json": (710, "23c775b63aebf20bea73718acca77dc23c0dd5192b606871fa1ee6bcc4dce99a"),
        "normalization/share_norm_stats_all_t0_zscore.json": (106, "9d8893778ea1e4928b0c62f4af1aaf4d26875762d83efa68c2361da5493f8f04"),
        "tactile_input_config_file.json": (266, "08c3ca3b6a75765e16a05a2ebefa91b324a4e68abdc73191d99823b9ed2223a9"),
        "train_config.json": (2_503, "7acd19a540397c390a1c1c332b33280e1c1aefa6700d6dbe99a7587ee405bed8"),
    }
    for relative, (size, digest) in ftp1_entries.items():
        record_artifact(f"ftp1/{relative}", ftp1_root / relative, size, digest)
    overlay_root = pathlib.Path(os.environ["FTP1_OVERLAY_IN"]).resolve()
    overlay_rows = []
    for path in sorted(candidate for candidate in overlay_root.rglob("*") if candidate.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        logical = f"ftp1-overlay/{path.relative_to(overlay_root).as_posix()}"
        stat = path.stat()
        digest = file_sha256(path)
        overlay_rows.append((logical, digest, stat.st_size))
        artifact_records[logical] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "sha256": digest,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
        }
    overlay_actual = (len(overlay_rows), sum(size for _, _, size in overlay_rows), rows_digest(overlay_rows))
    overlay_expected = (9_459, 1_064_700_713, "c29aa0062293776a44fe584feed0b8ccf34be6afebf702241f87a724c69448ea")
    if overlay_actual != overlay_expected:
        raise SystemExit(f"FTP1 overlay tree={overlay_actual}, expected {overlay_expected}")
    asset_tree_digests["ftp1_overlay"] = overlay_actual[2]
    record_artifact(
        "ftp1_runtime/READY.json",
        pathlib.Path(os.environ["FTP1_OVERLAY_IN"]) / "READY.json",
        907,
        "135092bfbd21ef4db2f4fc71b81669acbf7cbe4d729a4abd3d9963db04e82c8b",
    )
    record_artifact(
        "ftp1_runtime/paligemma_tokenizer.model",
        pathlib.Path(os.environ["OPENPI_DATA_HOME_IN"]) / "big_vision" / "paligemma_tokenizer.model",
        4_264_023,
        "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",
    )

artifact_index = "".join(
    f"{logical}\0{record['sha256']}\0{record['size']}\n"
    for logical, record in sorted(artifact_records.items())
)
artifact_manifest = {
    "schema_version": 1,
    "created_unix_ns": time.time_ns(),
    "policy": os.environ["POLICY_IN"],
    "asset_tree_sha256": asset_tree_digests,
    "content_sha256": hashlib.sha256(artifact_index.encode()).hexdigest(),
    "files": artifact_records,
}
(snapshot / "artifacts.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True))
print(
    f"validated {len(artifact_records)} immutable artifacts; "
    f"content_sha256={artifact_manifest['content_sha256']}",
    flush=True,
)
PYEOF

"${PY}" - <<'PYEOF'
import numpy, sys, torch
print(f"preflight numpy={numpy.__version__} torch={torch.__version__} cuda_devices={torch.cuda.device_count()}", flush=True)
if not numpy.__version__.startswith("1."):
    raise SystemExit("Isaac extensions require numpy 1.x")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected exactly two visible GPUs, got {torch.cuda.device_count()}")
for index in range(2):
    value = torch.ones(1, device=f"cuda:{index}") + index
    print(f"cuda:{index} {torch.cuda.get_device_name(index)} value={value.item()}", flush=True)
PYEOF

export PYTHONPATH="${WORK_IN}/src"
if [[ "${POLICY_IN}" == "flow" ]]; then
  "${PY}" -c "import openpi.mot_jepa.deploy; print('openpi flow imports ok', flush=True)"
  SCRIPT="${WORK_IN}/scripts/eval_motjepa.py"
  FLOW_MODE_IN="$("${PY}" -c 'import json, os, pathlib; cfg=json.loads((pathlib.Path(os.environ["FLOW_RUN_IN"])/"run_config.json").read_text()); print(cfg.get("backbone_train_mode", "frozen"))')"
  case "${FLOW_MODE_IN}" in frozen|last_blocks|full) ;; *) echo "invalid flow backbone mode ${FLOW_MODE_IN}" >&2; exit 2 ;; esac
  export FLOW_MODE_IN
  POLICY_LABEL="motjepa_flow_${FLOW_MODE_IN}_p${FLOW_STEP_IN}_k${NUM_SAMPLES_IN}"
  export MOTJEPA_POLICY_KIND=flow
  export MOTJEPA_POLICY_RUN="${FLOW_RUN_IN}"
  export MOTJEPA_POLICY_STEP="${FLOW_STEP_IN}"
  export MOTJEPA_NUM_SAMPLES="${NUM_SAMPLES_IN}"
  export MOTJEPA_SAMPLE_SEED="${POLICY_SAMPLE_SEED_IN}"
  EXTRA=(--checkpoint_dir "${FLOW_RUN_IN}")
elif [[ "${POLICY_IN}" == "ridge" ]]; then
  "${PY}" -c "import openpi.mot_jepa.deploy; print('openpi ridge imports ok', flush=True)"
  SCRIPT="${WORK_IN}/scripts/eval_motjepa.py"
  POLICY_LABEL="motjepa_f100k_ridge"
  export MOTJEPA_POLICY_KIND=ridge
  export RIDGE="${RIDGE_IN}" SNAP="${SNAP_IN}" SNAP_STEP="${SNAP_STEP_IN}"
  EXTRA=(--checkpoint_dir "${SNAP_IN}")
else
  if [[ ! -f "${FTP1_OVERLAY_IN}/READY.json" ]]; then
    echo "missing qualified FTP1 runtime overlay ${FTP1_OVERLAY_IN}/READY.json" >&2
    exit 2
  fi
  export PYTHONPATH="${FTP1_OVERLAY_IN}:${WORK_IN}/src"
  "${PY}" - <<'PYEOF'
import hashlib, json, os, pathlib

overlay = pathlib.Path(os.environ["FTP1_OVERLAY_IN"]).resolve()
ready = json.loads((overlay / "READY.json").read_text())
import numpy
import torch
import transformers
import transformers.models.siglip.check
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

del FTP1InferenceWrapper
torch_path = pathlib.Path(torch.__file__).resolve()
if overlay in torch_path.parents:
    raise SystemExit(f"overlay shadowed Isaac torch: {torch_path}")
if ready.get("torch") != torch.__version__ or pathlib.Path(ready.get("torch_path", "")).resolve() != torch_path:
    raise SystemExit(f"runtime differs from qualified READY torch: ready={ready} live={torch.__version__} {torch_path}")
if transformers.__version__ != "4.53.2":
    raise SystemExit(f"expected patched transformers 4.53.2, got {transformers.__version__}")
if not numpy.__version__.startswith("1."):
    raise SystemExit(f"Isaac requires numpy 1.x, got {numpy.__version__}")
tokenizer_path = pathlib.Path(ready.get("tokenizer_path", ""))
if not tokenizer_path.is_file():
    raise SystemExit(f"qualified tokenizer is missing: {tokenizer_path}")
if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != ready.get("tokenizer_sha256"):
    raise SystemExit(f"qualified tokenizer digest changed: {tokenizer_path}")
for key in ("augmax_path", "orbax_path"):
    qualified_path = overlay / ready.get(key, "")
    if not qualified_path.is_file():
        raise SystemExit(f"qualified overlay path {key} is missing: {qualified_path}")
print(
    f"FTP1 runtime ok numpy={numpy.__version__} torch={torch.__version__} "
    f"transformers={transformers.__version__}",
    flush=True,
)
PYEOF
  SCRIPT="${WORK_IN}/UniVTAC/scripts/eval_ftp1.py"
  POLICY_LABEL="ftp1"
  EXTRA=(--checkpoint_dir "${CKPT_IN}" --chunk_first_n 20 --temporal_ensemble --ensemble_K "${ENSEMBLE_K_IN}")
fi

set +e
timeout --signal=TERM --kill-after=120s "${RUN_SECONDS_IN}" "${PY}" "${SCRIPT}" \
  --task_list "${TASK_IN}" \
  --domain_name "${DOMAIN_IN}" \
  --policy_name "${POLICY_LABEL}" \
  --ftp1_device "${POLICY_DEVICE_IN}" \
  --gpu "" \
  --task_config demo.yml \
  --total_num "${TOTAL_IN}" \
  --start_seed "${START_SEED_IN}" \
  --workers 1 \
  --num_inference_steps "${EULER_STEPS_IN}" \
  --tactile_key right_tactile_gripper \
  --tactile_sensor GelSightMini \
  --policy_sample_seed "${POLICY_SAMPLE_SEED_IN}" \
  --arm_slice 9:16 \
  --gripper_index 44 \
  --save_trajectory \
  --trajectory_max_actions 500 \
  --save_root "${SAVE_ROOT_IN}" \
  --run_id "${RUN_ID_IN}" \
  --save_infer_input_dir "${INPUT_DIR_IN}" \
  "${EXTRA[@]}"
eval_rc=$?
set -e
echo "eval rc=${eval_rc}"

export EVAL_RC_IN="${eval_rc}" POLICY_LABEL_IN="${POLICY_LABEL}"
set +e
"${PY}" - <<'PYEOF'
import hashlib, json, os, pathlib, sys, time

import numpy as np
import torch

task = os.environ["TASK_IN"]
total_expected = int(os.environ["TOTAL_IN"])
seed0 = int(os.environ["START_SEED_IN"])
run_root = pathlib.Path(os.environ["SAVE_ROOT_IN"]) / task / os.environ["RUN_ID_IN"]
metadata_path = run_root / "metadata.json"
worker_path = run_root / "metadata" / "worker_0.json"
out = {
    "ok": False,
    "policy": os.environ["POLICY_IN"],
    "policy_label": os.environ["POLICY_LABEL_IN"],
    "task": task,
    "domain": os.environ["DOMAIN_IN"],
    "task_config": "demo.yml",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "pair_id": os.environ["PAIR_ID_IN"],
    "run_id": os.environ["RUN_ID_IN"],
    "run_root": str(run_root),
    "eval_rc": int(os.environ["EVAL_RC_IN"]),
    "start_seed": seed0,
    "expected_total": total_expected,
    "num_samples": int(os.environ["NUM_SAMPLES_IN"]),
    "euler_steps": int(os.environ["EULER_STEPS_IN"]),
    "ensemble_K": float(os.environ["ENSEMBLE_K_IN"]),
    "policy_sample_seed": int(os.environ["POLICY_SAMPLE_SEED_IN"]),
    "flow_run": os.environ["FLOW_RUN_IN"] if os.environ["POLICY_IN"] == "flow" else None,
    "flow_step": int(os.environ["FLOW_STEP_IN"]) if os.environ["POLICY_IN"] == "flow" else None,
    "ridge": os.environ["RIDGE_IN"] if os.environ["POLICY_IN"] == "ridge" else None,
    "ftp1_checkpoint": os.environ["CKPT_IN"] if os.environ["POLICY_IN"] == "ftp1" else None,
}
errors = []
try:
    metadata = json.loads(metadata_path.read_text())
    worker = json.loads(worker_path.read_text())
except Exception as exc:
    errors.append(f"missing or invalid owned metadata: {exc}")
    metadata = {}
    worker = {}
if not isinstance(metadata, dict):
    errors.append(f"root metadata is {type(metadata).__name__}, expected object")
    metadata = {}
if not isinstance(worker, dict):
    errors.append(f"worker metadata is {type(worker).__name__}, expected object")
    worker = {}

expected_seeds = {str(seed) for seed in range(seed0, seed0 + total_expected)}
actual_seeds = set(worker)
if actual_seeds != expected_seeds:
    errors.append(f"seed keys differ: missing={sorted(expected_seeds-actual_seeds)} extra={sorted(actual_seeds-expected_seeds)}")
results = [entry.get("result") for entry in worker.values() if isinstance(entry, dict)]
if len(results) != total_expected or any(result not in {"success", "failed"} for result in results):
    errors.append(f"worker results are incomplete or invalid: {results}")
worker_success = sum(result == "success" for result in results)
worker_failed = sum(result == "failed" for result in results)

success = int(metadata.get("success", -1))
failed = int(metadata.get("failed", -1))
error_count = int(metadata.get("error", -1))
reported_total = int(metadata.get("total_episodes", -1))
if metadata.get("task_name") != task or metadata.get("run_id") != os.environ["RUN_ID_IN"]:
    errors.append("root metadata task/run ownership mismatch")
if metadata.get("policy_name") != os.environ["POLICY_LABEL_IN"]:
    errors.append(f"policy label mismatch: {metadata.get('policy_name')!r}")
if metadata.get("domain_name") != os.environ["DOMAIN_IN"]:
    errors.append(f"domain mismatch: {metadata.get('domain_name')!r}")
expected_checkpoint = {
    "flow": os.environ["FLOW_RUN_IN"],
    "ridge": os.environ["SNAP_IN"],
    "ftp1": os.environ["CKPT_IN"],
}[os.environ["POLICY_IN"]]
if metadata.get("checkpoint_dir") != expected_checkpoint:
    errors.append(f"checkpoint mismatch: {metadata.get('checkpoint_dir')!r}")
if metadata.get("worker_metadata_files") != ["metadata/worker_0.json"]:
    errors.append(f"unexpected worker metadata files: {metadata.get('worker_metadata_files')!r}")
eval_args = metadata.get("eval_args")
if not isinstance(eval_args, dict):
    errors.append("root metadata lacks eval_args object")
    eval_args = {}
expected_eval_args = {
    "total_num": total_expected,
    "start_seed": seed0,
    "workers": 1,
    "action_rep": "absolute" if os.environ["POLICY_IN"] != "ftp1" else "mix",
    "arm_slice": "9:16",
    "gripper_index": 44,
    "policy_sample_seed": int(os.environ["POLICY_SAMPLE_SEED_IN"]),
    "save_trajectory": True,
    "trajectory_max_actions": 500,
    "num_inference_steps": int(os.environ["EULER_STEPS_IN"]),
    "ftp1_device": os.environ["POLICY_DEVICE_IN"],
    "tactile_key": "right_tactile_gripper",
    "tactile_sensor": "GelSightMini",
}
if os.environ["POLICY_IN"] == "ftp1":
    expected_eval_args.update(
        chunk_first_n=20,
        temporal_ensemble=True,
        ensemble_K=float(os.environ["ENSEMBLE_K_IN"]),
    )
for key, expected in expected_eval_args.items():
    if eval_args.get(key) != expected:
        errors.append(f"eval arg {key}={eval_args.get(key)!r}, expected {expected!r}")
task_config_path = pathlib.Path(str(metadata.get("task_config_file", "")))
if task_config_path.name != "demo.yml" or not task_config_path.is_file():
    errors.append(f"invalid task config provenance: {task_config_path}")
else:
    out["task_config_path"] = str(task_config_path)
    out["task_config_sha256"] = hashlib.sha256(task_config_path.read_bytes()).hexdigest()
if reported_total != total_expected or success + failed != total_expected or error_count != 0:
    errors.append(
        f"episode accounting mismatch success={success} failed={failed} error={error_count} total={reported_total}"
    )
if success != worker_success or failed != worker_failed:
    errors.append(
        f"root/worker count mismatch root={success}/{failed} worker={worker_success}/{worker_failed}"
    )
if int(os.environ["EVAL_RC_IN"]) != 0:
    errors.append(f"evaluator exited {os.environ['EVAL_RC_IN']}")

trace_dir = run_root / "trajectory" / "worker_0"
trace_paths = sorted(trace_dir.glob("*.npz")) if trace_dir.is_dir() else []
trace_seeds = {path.stem for path in trace_paths}
if trace_seeds != expected_seeds:
    errors.append(
        f"trajectory seed files differ: missing={sorted(expected_seeds-trace_seeds)} "
        f"extra={sorted(trace_seeds-expected_seeds)}"
    )
expected_first_index = 1 if os.environ["POLICY_IN"] == "ftp1" else 0
trace_hashes = {}
for trace_path in trace_paths:
    try:
        with np.load(trace_path, allow_pickle=False) as trace:
            seed = int(trace["seed"])
            count = int(trace["num_actions"])
            result = str(trace["result"])
            reason = str(trace["termination_reason"])
            raw_dim = int(trace["raw_action_dim"])
            truncated = bool(trace["truncated"])
            if trace_path.stem != str(seed) or str(seed) not in expected_seeds:
                raise ValueError(f"seed ownership mismatch path={trace_path.name} payload={seed}")
            worker_entry = worker.get(str(seed), {})
            if result != worker_entry.get("result"):
                raise ValueError(f"result {result!r} differs from worker {worker_entry.get('result')!r}")
            expected_reason_result = "success" if reason == "success" else "failed"
            if result != expected_reason_result or reason == "error":
                raise ValueError(f"invalid production termination reason/result {reason!r}/{result!r}")
            if count <= 0 or count > 500 or truncated:
                raise ValueError(f"invalid action count/truncation count={count} truncated={truncated}")
            if raw_dim != 120:
                raise ValueError(f"raw_action_dim={raw_dim}, expected 120")
            if trace["sent_action8"].shape != (count, 8) or not np.isfinite(trace["sent_action8"]).all():
                raise ValueError("sent_action8 is malformed or non-finite")
            if trace["qpos8_before"].shape != (count, 8) or not np.isfinite(trace["qpos8_before"]).all():
                raise ValueError("qpos8_before is malformed or non-finite")
            if trace["qpos8_after"].shape != (count, 8) or not np.isfinite(trace["qpos8_after"]).all():
                raise ValueError("qpos8_after is malformed or non-finite")
            if trace["raw_first_vector"].shape != (count, raw_dim) or not np.isfinite(trace["raw_first_vector"]).all():
                raise ValueError("raw_first_vector is malformed or non-finite")
            if not np.all(trace["first_executable_index"] == expected_first_index):
                raise ValueError(f"unexpected first executable index; expected {expected_first_index}")
            trace_meta = worker_entry.get("trajectory")
            if not isinstance(trace_meta, dict) or trace_meta.get("recorded_actions") != count:
                raise ValueError("worker metadata lacks matching trajectory record")
        digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        if trace_meta.get("sha256") != digest:
            raise ValueError("worker trajectory SHA-256 mismatch")
        trace_hashes[str(seed)] = digest
    except Exception as exc:
        errors.append(f"invalid trajectory {trace_path}: {exc}")
out["trajectory_sha256"] = trace_hashes

input_root = pathlib.Path(os.environ["INPUT_DIR_IN"])
input_metadata_path = input_root / "metadata.json"
input_npz_path = input_root / ("ftp1_input.npz" if os.environ["POLICY_IN"] == "ftp1" else "motjepa_input.npz")
try:
    input_metadata = json.loads(input_metadata_path.read_text())
except Exception as exc:
    errors.append(f"missing or invalid captured-input metadata: {exc}")
    input_metadata = {}
if input_metadata.get("channel_order") != "univtac_native_cv2_decoded":
    errors.append(f"captured-input channel contract mismatch: {input_metadata.get('channel_order')!r}")
if not input_npz_path.is_file() or input_npz_path.stat().st_size <= 0:
    errors.append(f"missing exact captured input: {input_npz_path}")
else:
    out["input_npz_path"] = str(input_npz_path)
    out["input_npz_sha256"] = hashlib.sha256(input_npz_path.read_bytes()).hexdigest()

if os.environ["POLICY_IN"] == "flow":
    flow_cfg_path = pathlib.Path(os.environ["FLOW_RUN_IN"]) / "run_config.json"
    try:
        flow_cfg = json.loads(flow_cfg_path.read_text())
        out["backbone"] = flow_cfg["pretrained_run"]
        out["backbone_step"] = int(flow_cfg["pretrained_step"])
        flow_mode = flow_cfg.get("backbone_train_mode", "frozen")
        out["backbone_train_mode"] = flow_mode
        flow_step = int(os.environ["FLOW_STEP_IN"])
        flow_checkpoint = pathlib.Path(os.environ["FLOW_RUN_IN"]) / "checkpoints" / str(flow_step)
        backbone_checkpoint = flow_checkpoint / "backbone.pt"
        out["adapted_backbone_step"] = flow_step if flow_mode != "frozen" else None
        out["backbone_checkpoint"] = str(backbone_checkpoint) if backbone_checkpoint.is_file() else None
        out["backbone_checkpoint_sha256"] = (
            hashlib.sha256(backbone_checkpoint.read_bytes()).hexdigest() if backbone_checkpoint.is_file() else None
        )
        if flow_mode != "frozen" and not backbone_checkpoint.is_file():
            errors.append(f"adapted flow lacks backbone checkpoint: {backbone_checkpoint}")
        flow_metadata = torch.load(flow_checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
        if int(flow_metadata.get("global_step", -1)) != flow_step:
            errors.append(f"flow checkpoint metadata step differs from {flow_step}")
        if flow_metadata.get("backbone_train_mode", "frozen") != flow_mode:
            errors.append("flow checkpoint/run backbone modes differ")
    except Exception as exc:
        errors.append(f"cannot resolve flow backbone provenance: {exc}")
elif os.environ["POLICY_IN"] == "ridge":
    out["backbone"] = os.environ["SNAP_IN"]
    out["backbone_step"] = int(os.environ["SNAP_STEP_IN"])
    out["backbone_train_mode"] = "frozen"
    out["backbone_checkpoint"] = None
    out["backbone_checkpoint_sha256"] = None
    out["adapted_backbone_step"] = None
else:
    out["backbone"] = None
    out["backbone_step"] = None
    out["backbone_train_mode"] = None
    out["backbone_checkpoint"] = None
    out["backbone_checkpoint_sha256"] = None
    out["adapted_backbone_step"] = None

source_dir = pathlib.Path(os.environ["SOURCE_DIR_IN"])
source_manifest_path = source_dir / "manifest.json"
try:
    source_manifest = json.loads(source_manifest_path.read_text())
    source_hashes = source_manifest["files"]
    for relative, recorded in source_hashes.items():
        live = pathlib.Path(os.environ["WORK_IN"]) / recorded.get("source_path", relative)
        snapshot_file = source_dir / relative
        live_digest = hashlib.sha256(live.read_bytes()).hexdigest()
        snapshot_digest = hashlib.sha256(snapshot_file.read_bytes()).hexdigest()
        if live_digest != recorded["sha256"] or snapshot_digest != recorded["sha256"]:
            errors.append(f"source changed during evaluation: {relative}")
    out["source_manifest_path"] = str(source_manifest_path)
    out["source_manifest_sha256"] = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    out["source_content_sha256"] = source_manifest["content_sha256"]
    out["source_file_count"] = len(source_hashes)
except Exception as exc:
    errors.append(f"invalid source snapshot: {exc}")

artifact_manifest_path = source_dir / "artifacts.json"
try:
    artifact_manifest = json.loads(artifact_manifest_path.read_text())
    if artifact_manifest.get("policy") != os.environ["POLICY_IN"]:
        raise ValueError(f"artifact policy is {artifact_manifest.get('policy')!r}")
    for logical, recorded in artifact_manifest["files"].items():
        stat = pathlib.Path(recorded["path"]).stat()
        actual = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        expected = (recorded["size"], recorded["mtime_ns"], recorded["inode"])
        if actual != expected:
            errors.append(f"artifact changed during evaluation: {logical} actual={actual} expected={expected}")
    out["artifact_manifest_path"] = str(artifact_manifest_path)
    out["artifact_manifest_sha256"] = hashlib.sha256(artifact_manifest_path.read_bytes()).hexdigest()
    out["artifact_content_sha256"] = artifact_manifest["content_sha256"]
    out["asset_tree_sha256"] = artifact_manifest["asset_tree_sha256"]
    out["container_sha256"] = artifact_manifest["files"]["container/univtac_full.sqsh"]["sha256"]
    out["artifact_file_count"] = len(artifact_manifest["files"])
except Exception as exc:
    errors.append(f"invalid artifact manifest: {exc}")

out.update(
    success=success,
    failed=failed,
    error=error_count,
    total=reported_total,
    success_rate=(worker_success / total_expected if not errors else None),
    metadata_path=str(metadata_path),
    worker_metadata_path=str(worker_path),
    validation_errors=errors,
    ok=not errors,
)
result = pathlib.Path(os.environ["RESULT_IN"])
result.parent.mkdir(parents=True, exist_ok=True)
tmp = result.with_name(f"{result.name}.tmp.{os.getpid()}.{time.time_ns()}")
tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
os.replace(tmp, result)
print("CLOSEDLOOP", json.dumps(out, sort_keys=True), flush=True)
raise SystemExit(0 if out["ok"] else 1)
PYEOF
verify_rc=$?
set -e

if [[ "${eval_rc}" -ne 0 || "${verify_rc}" -ne 0 ]]; then
  exit 1
fi
INNER

export WORK_IN="${WORK}" RESULT_IN="${RESULT}" POLICY_IN="${POLICY}" TASK_IN="${TASK}"
export DOMAIN_IN="${DOMAIN}" TOTAL_IN="${TOTAL}" START_SEED_IN="${START_SEED}"
export FLOW_RUN_IN="${FLOW_RUN}" FLOW_STEP_IN="${FLOW_STEP}" RIDGE_IN="${RIDGE}"
export SNAP_IN="${SNAP}" SNAP_STEP_IN="${SNAP_STEP}" CKPT_IN="${CKPT}"
export FTP1_OVERLAY_IN="${FTP1_OVERLAY}"
export OPENPI_DATA_HOME_IN="${OPENPI_DATA_HOME}"
export IMAGE_IN="${IMAGE}" ASSETS_IN="${ASSETS}"
export NUM_SAMPLES_IN="${NUM_SAMPLES}" EULER_STEPS_IN="${EULER_STEPS}" ENSEMBLE_K_IN="${ENSEMBLE_K}"
export POLICY_SAMPLE_SEED_IN="${POLICY_SAMPLE_SEED}" RUN_SECONDS_IN="${RUN_SECONDS}"
export EVAL_CUDA_VISIBLE_DEVICES_IN="${EVAL_CUDA_VISIBLE_DEVICES}"
export PAIR_ID_IN="${PAIR_ID}" RUN_ID_IN="${RUN_ID}" SAVE_ROOT_IN="${SAVE_ROOT}"
export INPUT_DIR_IN="${INPUT_DIR}" SOURCE_DIR_IN="${SOURCE_DIR}" POLICY_DEVICE_IN="${POLICY_DEVICE:-cuda:1}"
export RUN_SH_IN="${RUN_SH}"

echo "[$(date -Is)] closed-loop policy=${POLICY} task=${TASK} n=${TOTAL} seed0=${START_SEED} samples=${NUM_SAMPLES} ensemble_K=${ENSEMBLE_K} euler_steps=${EULER_STEPS}"
set +e
# Preserve the exact two-GPU contract at both allocation and step scope (renderer cuda:0, policy
# cuda:1). Do not use an eight-GPU allocation plus masking: CUDA respects the mask but Vulkan does
# not, which causes ERROR_DEVICE_LOST on this cluster.
srun --ntasks=1 --gpus-per-task=2 \
     --container-image="${IMAGE}" \
     --container-remap-root --container-writable \
     --container-mounts=/lustre:/lustre,${ASSETS}:${WORK}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data \
     bash "${RUN_SH}"
srun_rc=$?
set -e

echo "[$(date -Is)] srun rc=${srun_rc}"
if [[ -f "${RESULT}" ]]; then
  echo "===== verdict ====="
  cat "${RESULT}"
else
  echo "missing verdict ${RESULT}" >&2
  exit 1
fi
if [[ "${srun_rc}" -ne 0 ]]; then
  exit "${srun_rc}"
fi
