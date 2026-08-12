#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill,batch_long
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --job-name=uipc-gate2
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#
# GATE 2: `import tacex_uipc` and `import curobo` succeed.
#
# Starts from the Gate 1 image, which already carries CUDA 12.4, cmake 3.26.4 and a libuipc built
# for sm_90 against Isaac's Python 3.10.
#
# The one real obstacle is torch_scatter. TacEx's setup.py:23-26 pins a DIRECT WHEEL URL for
# torch-2.8.0+cu128, and a direct-URL requirement is unconditional -- pip will fetch and install it
# over anything already present. Isaac ships torch 2.13.0+cu130, so that wheel's compiled extension
# is built against the wrong ABI and would import-error at best, or corrupt Isaac's torch at worst.
#
# Rather than patch vendored third_party, every TacEx package installs with --no-deps and its
# declared dependencies are supplied explicitly here, with torch_scatter compiled from source
# against the torch that is actually installed. Same end state, no diff against the vendored tree,
# and Isaac's torch is never a candidate for replacement.
#
# Verdict goes to a FILE. Read the artifact, never the exit code.

set -uo pipefail

IMAGE_IN="${IMAGE_IN:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_uipc.sqsh}"
IMAGE_OUT="${IMAGE_OUT:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
TACEX="${WORK}/UniVTAC/third_party/TacEx"
CUROBO_DIR="${CUROBO_DIR:-/lustre/fsw/portfolios/edgeai/users/chrislin/build/curobo}"
RESULT="${WORK}/logs/_uipc_gate2_${SLURM_JOB_ID}.json"
BUILD_SH="${WORK}/logs/_uipc_gate2_${SLURM_JOB_ID}.sh"

mkdir -p "$(dirname "${CUROBO_DIR}")" "${WORK}/logs"

cat > "${BUILD_SH}" <<'INNER'
set -x
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda-12.4
# H100. Without this each extension builds for the build host's native arch or a default list,
# which silently produces kernels that will not launch here.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="$(nproc)"

: "${TACEX:?}"; : "${CUROBO_DIR:?}"; : "${RESULT_IN:?}"
FAIL() { echo "GATE2_FAIL: $1"; echo "{\"gate2\": false, \"stage\": \"$1\"}" > "${RESULT_IN}"; exit 1; }

PY=/isaac-sim/python.sh
command -v nvcc || FAIL "nvcc_missing"
"${PY}" -c "import torch; print('TORCH', torch.__version__, torch.version.cuda)" || FAIL "torch_missing"
TORCH_BEFORE="$("${PY}" -c 'import torch; print(torch.__version__)')"

# --- torch_scatter from source ---------------------------------------------------------------
if ! "${PY}" -c "import torch_scatter" >/dev/null 2>&1; then
  "${PY}" -m pip install --no-build-isolation --no-deps torch_scatter || FAIL "torch_scatter_build"
fi
"${PY}" -c "import torch_scatter; print('torch_scatter', torch_scatter.__version__)" || FAIL "torch_scatter_import"

# --- TacEx packages, --no-deps so the pinned wheel URL is never fetched -----------------------
# Declared deps supplied by hand: tacex wants debugpy/psutil/nvidia-ml-py (pre-commit is a dev
# tool and is skipped), tacex_assets declares none, tacex_tasks wants pytorch-kinematics.
"${PY}" -m pip install --no-deps debugpy psutil nvidia-ml-py pytorch-kinematics || FAIL "tacex_deps"
for pkg in tacex tacex_assets tacex_tasks; do
  "${PY}" -m pip install --no-build-isolation --no-deps -e "${TACEX}/source/${pkg}" || FAIL "install_${pkg}"
done

# --- cuRobo from source ----------------------------------------------------------------------
# Imported unconditionally at envs/robot/robot.py:17, so it is mandatory even for a replay rollout.
if ! "${PY}" -c "import curobo" >/dev/null 2>&1; then
  [[ -d "${CUROBO_DIR}/.git" ]] || git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}" \
    || FAIL "curobo_clone"
  "${PY}" -m pip install --no-build-isolation --no-deps -e "${CUROBO_DIR}" || FAIL "curobo_build"
fi

# Isaac's torch must be untouched. If a transitive install swapped it, every result afterwards is
# measured against a different runtime than Gate 1 validated.
TORCH_AFTER="$("${PY}" -c 'import torch; print(torch.__version__)')"
[[ "${TORCH_BEFORE}" == "${TORCH_AFTER}" ]] || FAIL "torch_was_replaced_${TORCH_BEFORE}_to_${TORCH_AFTER}"

# --- GATE --------------------------------------------------------------------------------------
"${PY}" - <<'PYEOF' || FAIL "gate_imports"
import json
import os
out = {}
for name in ("torch", "torch_scatter", "uipc", "tacex", "tacex_uipc", "curobo", "isaaclab"):
    try:
        mod = __import__(name)
        out[name] = {"ok": True, "path": getattr(mod, "__file__", "?")}
    except Exception as exc:  # noqa: BLE001 -- reporting the failure IS the gate
        out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
out["gate2"] = all(out[n]["ok"] for n in ("tacex_uipc", "curobo"))
with open(os.environ["RESULT_IN"], "w") as handle:
    json.dump(out, handle, indent=2)
print("GATE2_PASS" if out["gate2"] else "GATE2_FAIL: imports", out)
PYEOF
grep -q '"gate2": true' "${RESULT_IN}" || FAIL "gate_imports_false"
INNER

export TACEX CUROBO_DIR
export RESULT_IN="${RESULT}"

echo "[$(date -Is)] gate2: ${IMAGE_IN} -> ${IMAGE_OUT}"
srun --container-image="${IMAGE_IN}" \
     --container-remap-root --container-writable \
     --container-save="${IMAGE_OUT}" \
     --container-mounts=/lustre:/lustre \
     bash "${BUILD_SH}"

echo "[$(date -Is)] srun rc=$?"
if [[ -f "${RESULT}" ]]; then
  echo "===== GATE 2 verdict ====="
  cat "${RESULT}"
else
  echo "===== GATE 2: no verdict file at ${RESULT} ====="
fi
