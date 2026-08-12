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

# Attempt 1 (job 6550554) failed at cmake_configure but its container WAS exported, so this
# image already carries the CUDA 12.4 toolkit and re-running skips the apt stage entirely.
# Falls back to the pre-CUDA image if that export is ever missing.
DEFAULT_IN=/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_uipc.sqsh
[[ -f "${DEFAULT_IN}" ]] || DEFAULT_IN=/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac.sqsh
IMAGE_IN="${IMAGE_IN:-${DEFAULT_IN}}"
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

# urdfdom must come from VCPKG at 3.1.1, and the apt copy must NOT be present.
#
# tacex_uipc/overlay-ports/{urdfdom,urdfdom-headers} are one-line stubs -- VCPKG_POLICY_EMPTY_PACKAGE
# -- that make vcpkg install nothing, so libuipc's find_package(urdfdom CONFIG REQUIRED) has to be
# satisfied elsewhere. Attempt 9 tried apt: the config was found, cmake configured, and it still
# failed at src/io/CMakeLists.txt:12 because libuipc links the NAMESPACED targets
# urdfdom::urdf_parser / urdfdom_model / urdfdom_world / urdfdom_sensor. Ubuntu 22.04 ships
# urdfdom 3.0.0, which exports them un-namespaced. That is exactly why the manifest pins 3.1.1 --
# the version where the urdfdom:: namespace appears.
#
# So the stubs are dropped from VCPKG_OVERLAY_PORTS below and vcpkg builds urdfdom itself. Purge
# the apt copy here so find_package cannot resolve against 3.0.0 and reintroduce the same error
# one layer deeper; a previous attempt baked it into the saved image, so this must be explicit
# rather than merely omitted.
apt-get purge -y liburdfdom-dev liburdfdom-headers-dev 2>/dev/null || true

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

# --- cmake >= 3.26 ---------------------------------------------------------------------------
# Ubuntu 22.04 ships 3.22.1 and libuipc's CMakeLists.txt:1 demands 3.26, which is exactly how
# attempt 1 died. Use the version and prefix libuipc's own installer uses
# (artifacts/docker/installers/install_cmake.sh, mirrored by dev.dockerfile:29-31) rather than a
# newer release: 3.26 is what this project is known to configure under.
if ! /opt/cmake/bin/cmake --version >/dev/null 2>&1; then
  wget -q https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh \
    -O /tmp/cmake.sh || FAIL "cmake_download"
  mkdir -p /opt/cmake
  bash /tmp/cmake.sh --prefix=/opt/cmake --skip-license || FAIL "cmake_install"
fi
export PATH="/opt/cmake/bin:${PATH}"
cmake --version | head -1
case "$(cmake --version | head -1)" in *" 3.2"[0-5]*|*" 3.1"*) FAIL "cmake_too_old";; esac

# --- vcpkg, pinned to the baseline the MANIFEST asks for -------------------------------------
# Attempt 2 died here with "no version database entry for cpptrace at 0.8.3". The tag 2024.11.16
# was wrong, and libuipc never asked for it: install_vcpkg.sh clones plain HEAD with no tag, while
# gen_vcpkg_json.py writes `builtin-baseline` into the generated vcpkg.json. vcpkg resolves every
# version against the registry AT THAT COMMIT, so the checkout must match the baseline or ports
# newer than the tag simply do not exist.
#
# Read the commit out of the vendored script rather than copying it here, so the two cannot drift.
BASELINE="$(grep -oE "VCPKG_BASE_LINE *= *'[0-9a-f]{40}'" "${UIPC_SRC}/libuipc/scripts/gen_vcpkg_json.py" \
            | grep -oE "[0-9a-f]{40}")"
[[ -n "${BASELINE}" ]] || FAIL "vcpkg_baseline_unreadable"
echo "VCPKG_BASELINE=${BASELINE}"

# Do NOT check out the baseline. It is dated 2024-11-15 and its version database tops out at
# cpptrace 0.7.2, while the generated manifest demands version>=0.8.3 -- the manifest asks for a
# version that did not exist when its own baseline was cut. Attempt 4 pinned the checkout there
# and died on exactly that.
#
# vcpkg reads AVAILABLE versions from the `versions/` tree on disk and uses builtin-baseline only
# to pin defaults, so the tree must be NEWER than the baseline. That is why install_vcpkg.sh
# clones plain HEAD with no checkout at all. Follow it, and record the resolved SHA in the verdict
# so the run stays reproducible despite tracking a moving branch.
if [[ ! -d "${VCPKG_DIR}/.git" ]]; then
  git clone https://github.com/microsoft/vcpkg.git "${VCPKG_DIR}" || FAIL "vcpkg_clone"
fi
git -C "${VCPKG_DIR}" fetch -q origin || FAIL "vcpkg_fetch"
if [[ "$(git -C "${VCPKG_DIR}" rev-parse HEAD)" != "$(git -C "${VCPKG_DIR}" rev-parse origin/master)" ]]; then
  git -C "${VCPKG_DIR}" checkout -q --detach origin/master || FAIL "vcpkg_checkout_master"
  rm -f "${VCPKG_DIR}/vcpkg"  # the bootstrapped tool is tied to the tree it came from
fi
VCPKG_SHA="$(git -C "${VCPKG_DIR}" rev-parse HEAD)"
echo "VCPKG_SHA=${VCPKG_SHA}"
[[ -f "${VCPKG_DIR}/vcpkg" ]] || "${VCPKG_DIR}/bootstrap-vcpkg.sh" -disableMetrics || FAIL "vcpkg_bootstrap"

# Assert the precondition rather than discovering it 20 lines into a cmake cascade: the port whose
# absence killed attempts 2 and 4 must be resolvable at the version the manifest requires.
WANT="$(grep -A2 "'name' *: *'cpptrace'" "${UIPC_SRC}/libuipc/scripts/gen_vcpkg_json.py" \
        | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)"
HAVE="$(git -C "${VCPKG_DIR}" show HEAD:versions/c-/cpptrace.json 2>/dev/null \
        | grep -oE '"version"[^0-9]*[0-9]+\.[0-9]+\.[0-9]+' | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" \
        | sort -V | tail -1)"
echo "cpptrace wanted>=${WANT} available<=${HAVE}"
[[ -n "${HAVE}" && "$(printf '%s\n%s\n' "${WANT}" "${HAVE}" | sort -V | tail -1)" == "${HAVE}" ]] \
  || FAIL "vcpkg_tree_older_than_manifest"
export VCPKG_ROOT="${VCPKG_DIR}"
export CMAKE_TOOLCHAIN_FILE="${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake"
[[ -f "${CMAKE_TOOLCHAIN_FILE}" ]] || FAIL "vcpkg_toolchain_missing"

# --- overlay ports --------------------------------------------------------------------------
# TWO sets, and neither was being applied before. tacex_uipc ships overlay-ports/ for urdfdom and
# urdfdom-headers -- upstream's are incompatible -- but nothing in libuipc sets VCPKG_OVERLAY_PORTS,
# so they were silently unused and urdfdom would have failed or built wrong later.
#
# Ours adds tinygltf 2.9.3 with a corrected source hash; see that portfile for why, and for the
# verification that the content is genuine.
# COLON, not semicolon. VCPKG_OVERLAY_PORTS uses the PLATFORM path-list separator, so ";" is
# Windows-only; on Linux vcpkg read "pathA;pathB" as a single directory name and reported
# `Overlay path "..." must be an existing directory`.
# OURS ONLY. Including ${UIPC_SRC}/overlay-ports would reinstate the urdfdom stubs explained
# above, which is what left find_package with nothing to link.
export VCPKG_OVERLAY_PORTS="${WORK_IN}/scripts_exp_zarr/mot_jepa/vcpkg-overlays"
echo "VCPKG_OVERLAY_PORTS=${VCPKG_OVERLAY_PORTS}"
for d in ${VCPKG_OVERLAY_PORTS//:/ }; do [[ -d "$d" ]] || FAIL "overlay_missing:$d"; done

# --- configure + build libuipc with all cores ---------------------------------------------
# UIPC_GITHUB_ACTIONS=1 forces VCPKG_MANIFEST_INSTALL ON (uipc_utils.cmake:107-109), and it is the
# only reliable way to run this repeatedly. libuipc otherwise decides whether to invoke vcpkg from
# whether vcpkg.json CHANGED -- gen_vcpkg_json.py returns 1 for needs-install, 0 for no-change --
# which assumes a configure never dies partway through installing. Attempts 2 and 6 both did, and
# attempt 6 then skipped vcpkg entirely: 0 ports installed in 75 seconds, failing at
# find_package(urdfdom) with 34 of 39 ports on disk.
#
# The earlier repair -- discard the build dir when a sentinel port is absent -- was wrong twice
# over. eigen3 survived attempt 5's PARTIAL install, so it proved "some ports" rather than "all
# ports" and never triggered; and vcpkg_installed lives inside the build dir, so triggering would
# have thrown away 34 working ports to recover 5. Forcing the install on costs a few seconds of
# ABI hashing when everything is already present and is correct in both states.
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
  -DUIPC_GITHUB_ACTIONS=1 \
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
export WORK_IN="${WORK}"

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
