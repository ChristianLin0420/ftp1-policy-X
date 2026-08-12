#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill,batch_long
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --job-name=uipc-gate3
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#
# GATE 3: prove the sim loop closes, with NO model at all.
#
# 10 episodes of lift_bottle driven by UniVTAC's own scripted expert via collect_data.py. This is
# the decisive cheap test and it deliberately comes before any bridging code: if sim + tactile +
# FEM + video does not close, nothing above it matters and no policy work helps.
#
# Uses task_config/motjepa_gate3.yml rather than demo.yml for one reason. demo.yml sets
# render_frequency: 0, and collect_data.py:85-86 turns exactly that value into livestream = 2 --
# WebRTC streaming, which has no place in a batch job and is the hang the plan warned about. The
# key only governs the GUI window; cameras are forced on at :58 and video writing follows
# video_frequency, so a non-zero value keeps rendering while skipping the livestream branch.
#
# VERDICT IS READ FROM ARTIFACTS, NEVER THE EXIT CODE. Isaac runs with --/app/fastShutdown=True,
# which can _exit() and discard buffered stdout; an earlier probe in this project exited 0 with
# every decisive print missing. Success is: episode files on disk, a video that is not black, and
# at least one reported success.

set -uo pipefail

IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
TASK="${TASK:-lift_bottle}"
CONFIG="${CONFIG:-motjepa_gate3}"
EPISODES="${EPISODES:-10}"
OUTDIR="${OUTDIR:-/lustre/fsw/portfolios/edgeai/users/chrislin/univtac-gate3}"
RESULT="${WORK}/logs/_uipc_gate3_${SLURM_JOB_ID}.json"
RUN_SH="${WORK}/logs/_uipc_gate3_${SLURM_JOB_ID}.sh"

mkdir -p "${OUTDIR}" "${WORK}/logs"

cat > "${RUN_SH}" <<'INNER'
set -x
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/local/cuda-12.4
: "${UNIVTAC:?}"; : "${RESULT_IN:?}"; : "${TASK_IN:?}"; : "${CONFIG_IN:?}"; : "${EPISODES_IN:?}"; : "${OUTDIR_IN:?}"

PY=/isaac-sim/python.sh

# REPAIR numpy before anything else. Isaac Sim's extensions are prebuilt against numpy 1.x, and
# the first Gate 3 run died with
#   "numpy.dtype size changed ... Expected 96 from C header, got 88 from PyObject"
#   -> isaacsim.core.simulation_manager failed to startup
# i.e. a numpy 2.x ABI under modules compiled for 1.x. The culprit is Gate 1's
# `pip install -e .` for tacex_uipc, the ONE install in this stack that omitted --no-deps, which
# let wildmeshing's dependency chain upgrade numpy underneath Isaac.
#
# Nothing here notices until the app boots: pip is happy, every import outside Kit is happy, and
# Gate 2 passed because AppLauncher tolerated the broken extensions. Pin it back.
"${PY}" -c "import numpy; print('numpy_before', numpy.__version__)"
"${PY}" -m pip install --no-deps "numpy<2" || { echo '{"gate3": false, "stage": "numpy_pin"}' > "${RESULT_IN}"; exit 1; }
"${PY}" -c "import numpy; print('numpy_after', numpy.__version__)"

# UniVTAC's own third-party imports. NOT requirements.txt: that file is a frozen full environment
# pinning torch_scatter==2.1.2+pt28cu128, torchvision==0.22.0+cu128 and numpy==1.23.5, which would
# tear out the torch stack Gates 1 and 2 validated. These are the packages the sim import chain
# actually reaches, scanned from envs/ and collect_data.py; transforms3d is the one that stopped
# the previous attempt. --no-deps throughout so nothing re-upgrades numpy behind Isaac's back.
"${PY}" -m pip install --no-deps transforms3d trimesh warp-lang tqdm scipy pydantic decorator \
  || { echo '{"gate3": false, "stage": "univtac_deps"}' > "${RESULT_IN}"; exit 1; }
# torchvision must MATCH Isaac's torch; never install the pinned +cu128 build over it.
"${PY}" -c "import torchvision; print('torchvision', torchvision.__version__)" \
  || echo "WARNING: torchvision missing; leaving Isaac's stack alone rather than installing a pinned build"
# Re-assert after the installs above -- the whole point of this block is that one careless
# dependency already upgraded numpy once and broke Isaac's core extensions silently.
"${PY}" -c "import numpy,sys; v=numpy.__version__; print('numpy_final',v); sys.exit(0 if v.startswith('1.') else 1)" \
  || { echo '{"gate3": false, "stage": "numpy_regressed"}' > "${RESULT_IN}"; exit 1; }

cd "${UNIVTAC}" || { echo '{"gate3": false, "stage": "cd_univtac"}' > "${RESULT_IN}"; exit 1; }

# NO pre-flight import check here. tacex_uipc pulls in isaacsim.core, a Kit EXTENSION that only
# exists once the Isaac app has booted, so `python -c "import tacex_uipc"` raises
# ModuleNotFoundError however correctly everything is installed. A check like that killed the
# first Gate 3 run before collect_data.py started -- the same mistake Gate 2 had just proven, and
# the imports are Gate 2's job anyway. collect_data.py launches the app itself.

# Headless via ENVIRONMENT, not a flag. collect_data.py calls parser.parse_args() at :50 and only
# then calls AppLauncher.add_app_launcher_args(parser) at :55, so the launcher flags are never
# registered and `--headless` is rejected outright -- "unrecognized arguments: --headless" ended
# the previous attempt in 45 seconds. It sets enable_cameras itself at :58. AppLauncher reads
# these env vars, which is the only channel left open.
export HEADLESS=1
export LIVESTREAM=0
export ENABLE_CAMERAS=1
timeout 10800 "${PY}" scripts/collect_data.py "${TASK_IN}" "${CONFIG_IN}" \
  --episode_num "${EPISODES_IN}"
echo "collect_data rc=$?"

# Artifacts, not exit codes.
"${PY}" - <<'PYEOF'
import glob
import json
import os
import pathlib

out = {"stage": "artifacts"}
root = pathlib.Path(os.environ["UNIVTAC"]) / "data"
videos = sorted(glob.glob(str(root / "**" / "*.mp4"), recursive=True))
episodes = sorted(glob.glob(str(root / "**" / "*.hdf5"), recursive=True))
out["video_count"] = len(videos)
out["episode_count"] = len(episodes)
out["videos"] = videos[:5]

# A rendered-but-black frame is the failure mode that passes every other check: the app starts,
# the file is written, the episode "succeeds", and every frame is zeros because the renderer never
# attached. Measure the pixels.
if videos:
    try:
        import cv2
        cap = cv2.VideoCapture(videos[0])
        best = {"nonblack_fraction": 0.0, "pixel_std": 0.0}
        read = 0
        while read < 40:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            nz = float((frame > 8).mean())
            if nz > best["nonblack_fraction"]:
                best = {"nonblack_fraction": nz, "pixel_std": float(frame.std())}
        cap.release()
        out.update(best)
        out["frames_read"] = read
    except Exception as exc:  # noqa: BLE001
        out["video_error"] = f"{type(exc).__name__}: {exc}"

out["gate3"] = (
    out["episode_count"] > 0
    and out.get("nonblack_fraction", 0.0) > 0.05
    and out.get("pixel_std", 0.0) > 5.0
)
with open(os.environ["RESULT_IN"], "w") as handle:
    json.dump(out, handle, indent=2)
print("GATE3_PASS" if out["gate3"] else "GATE3_FAIL", out)
PYEOF

# Keep the artifacts off the container's ephemeral layer.
cp -r "${UNIVTAC}/data" "${OUTDIR_IN}/" 2>/dev/null || true
INNER

export UNIVTAC="${WORK}/UniVTAC"
export RESULT_IN="${RESULT}" TASK_IN="${TASK}" CONFIG_IN="${CONFIG}"
export EPISODES_IN="${EPISODES}" OUTDIR_IN="${OUTDIR}"

echo "[$(date -Is)] gate3: ${TASK} x ${EPISODES} episodes, image=${IMAGE}"
srun --container-image="${IMAGE}" \
     --container-remap-root --container-writable \
     --container-save="${IMAGE}" \
     --container-mounts=/lustre:/lustre \
     bash "${RUN_SH}"

echo "[$(date -Is)] srun rc=$?"
if [[ -f "${RESULT}" ]]; then
  echo "===== GATE 3 verdict ====="
  cat "${RESULT}"
else
  echo "===== GATE 3: no verdict file at ${RESULT} ====="
fi
