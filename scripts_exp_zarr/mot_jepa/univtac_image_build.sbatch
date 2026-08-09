#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=univtac-image
#SBATCH --output=logs/%x-%j.log
#
# Phase 1: build the UniVTAC simulation image on top of the ONE component with verified evidence
# on this cluster -- nvcr.io#nvidia/isaac-sim:4.5.0, which job 6442243 rendered from headless on
# an H100 (nonblack_fraction=0.9996).
#
# Root inside the container comes from /etc/enroot/enroot.conf (ENROOT_REMAP_ROOT yes,
# ENROOT_ROOTFS_WRITABLE yes), so every `sudo apt-get` in Installation_FTP1.md runs as-is with
# the `sudo` dropped. No step in that document is blocked by cluster permissions.
#
# The result MUST be saved to Lustre. The enroot cache lives on node-local /raid and is wiped
# between jobs, so an image built on one node is invisible to the next.
#
#   REPO_ROOT=... IMAGE_OUT=/lustre/.../images/univtac.sqsh \
#     sbatch scripts_exp_zarr/mot_jepa/univtac_image_build.sbatch
set -uo pipefail

IMAGE_IN="${IMAGE_IN:-nvcr.io#nvidia/isaac-sim:4.5.0}"
IMAGE_OUT="${IMAGE_OUT:?set IMAGE_OUT -- an absolute .sqsh path on Lustre}"
ISAACLAB_TAG="${ISAACLAB_TAG:-v2.1.1}"
MARKER="${MARKER:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/BUILD_OK}"

# EXPORT, not just assign. srun starts the container with a fresh shell, so a plain assignment in
# this outer script is invisible inside and the `${VAR:?}` guards abort with exit 127.
export ISAACLAB_TAG MARKER

mkdir -p "$(dirname "${IMAGE_OUT}")"
rm -f "${MARKER}"
echo "[$(date -Is)] base=${IMAGE_IN} -> ${IMAGE_OUT}  isaaclab=${ISAACLAB_TAG}"

srun --container-image="${IMAGE_IN}" \
     --container-remap-root --container-writable \
     --container-save="${IMAGE_OUT}" \
     --container-mounts=/lustre:/lustre \
     bash -c '
set -x
: "${ISAACLAB_TAG:?}"; : "${MARKER:?}"

# `bash -c`, NOT `bash -lc`. A login shell sources the profile, which runs conda init and puts
# the Lustre miniconda (Python 3.13) first on PATH -- and /lustre is bind-mounted here. That is
# exactly what broke the previous attempt: isaaclab.sh resolved against 3.13 and pip refused
# rl-games with "requires a different Python: 3.13.11 not in <3.11,>=3.7.1". Isaac Sim 4.5 is
# Python 3.10 and Isaac Lab must install into ITS interpreter.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# The Isaac Sim image ships its own python; find it rather than assuming a path.
ISAACSIM_PATH="${ISAACSIM_PATH:-/isaac-sim}"
ls -d "${ISAACSIM_PATH}" || ISAACSIM_PATH=$(dirname "$(readlink -f "$(command -v isaacsim 2>/dev/null)")" 2>/dev/null)
echo "ISAACSIM_PATH=${ISAACSIM_PATH}"
# Fail loudly here rather than 20 minutes later inside pip: without this interpreter there is
# nothing correct to install into.
"${ISAACSIM_PATH}/python.sh" -c "import sys; print(\"ISAACSIM_PYTHON\", sys.version)" \
  || { echo "NO_ISAACSIM_PYTHON"; exit 20; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# Installation_FTP1.md:82 (gcc-11), :138 (cmake/build-essential), :238 (vcpkg deps), :331 (ffmpeg).
# NOT installing the Vulkan ICD from :104-120 -- job 6442243 proved /usr/share/vulkan/icd.d is
# empty and rendering works anyway, because enroot injects libnvidia-glcore from the host driver.
# Creating it is a documented way to break a working setup.
apt-get install -y --no-install-recommends \
    git git-lfs curl wget zip unzip tar pkg-config \
    build-essential gcc-11 g++-11 cmake ninja-build ffmpeg \
  || { echo "APT_FAILED"; exit 21; }

gcc-11 --version | head -1
cmake --version | head -1

# ---- Isaac Lab -------------------------------------------------------------------------
git clone --depth 1 --branch "${ISAACLAB_TAG}" https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab \
  || { echo "ISAACLAB_CLONE_FAILED"; exit 22; }
cd /opt/IsaacLab

# Documented fix (Installation_FTP1.md): the pinned flatdict 4.0.1 has no installable sdist.
sed -i "s/flatdict==4\.0\.1/flatdict==4.1.0/" source/isaaclab/setup.py || true
grep -n flatdict source/isaaclab/setup.py || true

ln -sfn "${ISAACSIM_PATH}" /opt/IsaacLab/_isaac_sim || true
./isaaclab.sh --install || { echo "ISAACLAB_INSTALL_FAILED"; exit 23; }

# ---- verify by IMPORT, not by exit code ------------------------------------------------
# And assert the interpreter is 3.10: a successful-looking install into the wrong python is
# precisely the failure this build already hit once.
"${ISAACSIM_PATH}/python.sh" -c "
import sys, isaaclab, isaacsim
assert sys.version_info[:2] == (3, 10), f\"installed into python {sys.version_info[:2]}, expected 3.10\"
print(\"ISAACLAB_IMPORT_OK\", isaaclab.__file__)
" || { echo "ISAACLAB_IMPORT_FAILED"; exit 24; }

touch "${MARKER}"
echo "BUILD_OK"
'
RC=$?
echo "[$(date -Is)] srun rc=${RC}"

# Artifact, not exit code: Isaac can _exit() and discard buffered stdout.
if [[ -f "${MARKER}" ]]; then
  echo "[$(date -Is)] VERDICT=PASS marker present"
else
  echo "[$(date -Is)] VERDICT=FAIL no marker at ${MARKER}"
fi
ls -lah "${IMAGE_OUT}" 2>/dev/null || echo "no image written"
exit ${RC}
