#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill,batch_long
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --job-name=uipc-build
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#
# RESUMABLE, and sized to SHARE a node rather than own one. The first attempt asked --mem=0 and
# 8 hours: --mem=0 resolves to the node's full 2010G, so Slurm had to wait for an entirely empty
# node (473 of 491 allocated, none idle) and reserved a start five hours out, while 4-hour jobs
# submitted later ran immediately. 8 hours also excludes `batch`, whose cap is 4h.
#
# Nothing here needs a whole node. Under 4 hours it fits every partition, and both expensive
# stages resume: vcpkg is cloned/bootstrapped only if absent, and `cmake --build` is incremental.
# Both live on Lustre, so a cut-off attempt resumes on resubmit and only the apt/CUDA step (a few
# minutes) repeats -- that runs in the container layer, which is lost unless --container-save is
# reached.
#
# GATE 1 of the tacex_uipc plan: nvcc present, libuipc built, `import uipc` succeeds.
#
# Four decisions, each forced by a measurement rather than by the install guide:
#
#   CUDA 12.4, not 13.0. The image ships torch 2.13.0+cu130 and NO nvcc at all (probe 6547440).
#     libuipc's reference build is nvidia/cuda:12.4.0-devel-ubuntu22.04 (dev.dockerfile:1) and
#     there is no evidence of anyone building it against CUDA 13. Installing the 12.4 toolkit
#     alongside is the pre-agreed fallback done in-place: cudart is soname-versioned, so
#     libuipc's libcudart.so.12 and torch's .13 coexist in one process.
#
#   sm_90. dev.dockerfile:5 makes the architecture an ARG defaulting to 89 (Ada); this cluster is
#     H100. torch already lists sm_90 in its arch list, so the driver is not in question.
#
#   Build libuipc BEFORE pip. setup.py hardcodes `-j4` (line ~97) and never disables
#     UIPC_BUILD_TESTS / EXAMPLES / BENCHMARKS, all of which default ON in CMakeLists.txt:19-21.
#     Rather than patch vendored third_party, configure and build here with the options set and
#     all cores; setup.py's later cmake re-run inherits the CMakeCache and its -j4 pass finds
#     nothing left to compile.
#
#   vcpkg on Lustre, not /raid. Its ~13 ports build from source and take hours; /raid is
#     node-local and wiped between jobs, so a requeue would rebuild them every time.
#
# Verdict goes to a FILE. Read the artifact, never the exit code.
#
#   mkdir -p logs && sbatch scripts_exp_zarr/mot_jepa/uipc_build.sbatch

set -uo pipefail

IMAGE_IN="${IMAGE_IN:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac.sqsh}"
IMAGE_OUT="${IMAGE_OUT:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_uipc.sqsh}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
VCPKG_DIR="${VCPKG_DIR:-/lustre/fsw/portfolios/edgeai/users/chrislin/build/vcpkg}"
UIPC_SRC="${UIPC_SRC:-${WORK}/UniVTAC/third_party/TacEx/source/tacex_uipc}"
RESULT="${WORK}/logs/_uipc_gate1_${SLURM_JOB_ID}.json"
BUILD_SH="${WORK}/logs/_uipc_build_${SLURM_JOB_ID}.sh"

mkdir -p "$(dirname "${IMAGE_OUT}")" "$(dirname "${VCPKG_DIR}")" "${WORK}/logs"

cat > "${BUILD_SH}" <<'INNER'
set -x
# bash -c, NOT -lc. A login shell sources the profile, which puts the Lustre conda (Python 3.13)
# ahead of Isaac Sim's 3.10 on PATH -- and /lustre is mounted here. That exact shadowing is what
# broke the earlier image build with "rl-games requires a different Python".
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export DEBIAN_FRONTEND=noninteractive

: "${VCPKG_DIR:?}"; : "${UIPC_SRC:?}"; : "${RESULT_IN:?}"
FAIL() { echo "GATE1_FAIL: $1"; echo "{\"gate1\": false, \"stage\": \"$1\"}" > "${RESULT_IN}"; exit 1; }

ISAACSIM_PATH="${ISAACSIM_PATH:-/isaac-sim}"
PY="${ISAACSIM_PATH}/python.sh"
# The interpreter libuipc's pybind11 module must match. Isaac Sim 4.5 is 3.10 while libuipc's docs
# say >=3.11; letting CMake pick would build for the wrong ABI and fail only at `import uipc`.
PY_EXE="$("${PY}" -c 'import sys; print(sys.executable)')" || FAIL "isaac_python"
echo "PY_EXE=${PY_EXE}"

. /etc/os-release; echo "DISTRO=${ID}${VERSION_ID}"
UBU="$(echo "${VERSION_ID}" | tr -d '.')"

apt-get update -y || FAIL "apt_update"
apt-get install -y --no-install-recommends git wget curl zip unzip tar pkg-config \
  build-essential ca-certificates gnupg || FAIL "apt_base"

# --- CUDA 12.4 toolkit (nvcc) -------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBU}/x86_64/cuda-keyring_1.1-1_all.deb" \
    -O /tmp/cuda-keyring.deb || FAIL "cuda_keyring_download"
  dpkg -i /tmp/cuda-keyring.deb || FAIL "cuda_keyring_install"
  apt-get update -y || FAIL "apt_update_cuda"
  # Toolkit only. NOT the driver: enroot injects the host driver, and installing another would
  # break every CUDA process on the node.
  apt-get install -y --no-install-recommends cuda-toolkit-12-4 || FAIL "cuda_toolkit"
fi
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"
command -v nvcc || FAIL "nvcc_missing_after_install"
nvcc --version

# --- vcpkg --------------------------------------------------------------------------------
if [[ ! -f "${VCPKG_DIR}/vcpkg" ]]; then
  git clone https://github.com/microsoft/vcpkg.git "${VCPKG_DIR}" || FAIL "vcpkg_clone"
  git -C "${VCPKG_DIR}" checkout 2024.11.16 || FAIL "vcpkg_checkout"
  "${VCPKG_DIR}/bootstrap-vcpkg.sh" -disableMetrics || FAIL "vcpkg_bootstrap"
fi
export VCPKG_ROOT="${VCPKG_DIR}"
export CMAKE_TOOLCHAIN_FILE="${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake"
[[ -f "${CMAKE_TOOLCHAIN_FILE}" ]] || FAIL "vcpkg_toolchain_missing"

# --- configure + build libuipc with all cores ---------------------------------------------
BUILD="${UIPC_SRC}/build"
mkdir -p "${BUILD}"
cmake "${UIPC_SRC}/libuipc" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DUIPC_BUILD_GUI=0 \
  -DUIPC_BUILD_TESTS=0 \
  -DUIPC_BUILD_EXAMPLES=0 \
  -DUIPC_BUILD_BENCHMARKS=0 \
  -DUIPC_BUILD_PYBIND=1 \
  -DUIPC_DEV_MODE=1 \
  -DUIPC_PYTHON_EXECUTABLE_PATH="${PY_EXE}" || FAIL "cmake_configure"

cmake --build "${BUILD}" -j "$(nproc)" || FAIL "cmake_build"

# --- pip install; its cmake re-run inherits the cache above --------------------------------
export CMAKE_CUDA_ARCHITECTURES=90
cd "${UIPC_SRC}" || FAIL "cd_src"
"${PY}" -m pip install --no-build-isolation -e . || FAIL "pip_install"

# --- GATE ----------------------------------------------------------------------------------
"${PY}" - <<'PYEOF' || FAIL "import_uipc"
import json
import os
out = {}
import uipc
out["uipc"] = getattr(uipc, "__file__", "?")
try:
    import pyuipc
    out["pyuipc"] = getattr(pyuipc, "__file__", "?")
except Exception as exc:
    out["pyuipc"] = f"{type(exc).__name__}: {exc}"
out["gate1"] = True
with open(os.environ["RESULT_IN"], "w") as handle:
    json.dump(out, handle, indent=2)
print("GATE1_PASS", out)
PYEOF
INNER

export VCPKG_DIR UIPC_SRC
export RESULT_IN="${RESULT}"

echo "[$(date -Is)] gate1: ${IMAGE_IN} -> ${IMAGE_OUT}"
srun --container-image="${IMAGE_IN}" \
     --container-remap-root --container-writable \
     --container-save="${IMAGE_OUT}" \
     --container-mounts=/lustre:/lustre \
     bash "${BUILD_SH}"

echo "[$(date -Is)] srun rc=$?"
if [[ -f "${RESULT}" ]]; then
  echo "===== GATE 1 verdict ====="
  cat "${RESULT}"
else
  echo "===== GATE 1: no verdict file at ${RESULT} -- the build did not reach its write ====="
fi
