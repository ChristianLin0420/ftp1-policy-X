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
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export DEBIAN_FRONTEND=noninteractive
# H100. Without this each extension builds for the build host's native arch or a default list,
# which silently produces kernels that will not launch here.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="$(nproc)"

: "${TACEX:?}"; : "${CUROBO_DIR:?}"; : "${RESULT_IN:?}"
FAIL() { echo "GATE2_FAIL: $1"; echo "{\"gate2\": false, \"stage\": \"$1\"}" > "${RESULT_IN}"; exit 1; }

PY=/isaac-sim/python.sh

# CUDA 13.0, NOT the 12.4 that Gate 1 used. torch refuses to build an extension with a different
# nvcc than it was compiled with -- cpp_extension._check_cuda_version raised
#   "The detected CUDA version (12.4) mismatches the version that was used to compile PyTorch
#    (13.0)"
# and killed the first Gate 2 attempt. So the two halves of this stack genuinely need different
# toolkits: libuipc is a standalone C++/CUDA library and built against 12.4, its tested pairing,
# while torch_scatter and cuRobo are torch extensions and must match Isaac's torch 2.13.0+cu130.
# Both toolkits coexist; only PATH and CUDA_HOME differ per build. libuipc is already built and
# installed by Gate 1, so nothing here needs 12.4.
if [[ ! -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  apt-get update -y || FAIL "apt_update"
  apt-get install -y --no-install-recommends cuda-toolkit-13-0 || FAIL "cuda13_toolkit"
fi
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
command -v nvcc || FAIL "nvcc_missing"
nvcc --version | tail -2
# Assert the pairing rather than discovering it inside a 97-line pip traceback.
"${PY}" - <<'PYCHK' || FAIL "cuda_torch_mismatch"
import subprocess, sys, torch
out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
have = out.split("release ")[1].split(",")[0].strip()
want = torch.version.cuda
print("nvcc", have, "torch.cuda", want)
sys.exit(0 if have.split(".")[0] == want.split(".")[0] else 1)
PYCHK
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
# PINNED to v0.7.6. Cloning HEAD gave a cuRobo whose package was restructured to
# curobo/_src/geom, while UniVTAC does `from curobo.geom.transform import pose_multiply`
# (envs/robot/curobo_planner.py:1). `import curobo` succeeded, so Gate 2 passed, and the mismatch
# only surfaced when the sim actually imported the planner. v0.7.6 is the newest tag still using
# the classic src/curobo/geom layout that UniVTAC was written against.
[[ -d "${CUROBO_DIR}/.git" ]] || git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}" \
  || FAIL "curobo_clone"
if [[ "$(git -C "${CUROBO_DIR}" describe --tags --exact-match 2>/dev/null)" != "v0.7.6" ]]; then
  git -C "${CUROBO_DIR}" fetch --tags -q || FAIL "curobo_fetch"
  git -C "${CUROBO_DIR}" checkout -q v0.7.6 || FAIL "curobo_checkout"
  "${PY}" -m pip uninstall -y curobo >/dev/null 2>&1 || true
fi
# Assert the layout the caller needs BEFORE a 20-minute CUDA build.
[[ -d "${CUROBO_DIR}/src/curobo/geom" ]] || FAIL "curobo_layout_missing_geom"

# cuRobo v0.7.6 predates CUDA 13, which declares lerp(float,float,float) in its own math headers.
# The build dies with "'float lerp(float, float, float)' conflicts with a previous declaration".
# Guard ONLY the scalar overload -- CUDA provides that one, the float2/3/4 versions are still
# needed. Idempotent, so a fresh clone is patched the same way rather than depending on a manual
# edit surviving on Lustre.
HM="${CUROBO_DIR}/src/curobo/curobolib/cpp/helper_math.h"
if ! grep -q "CUDART_VERSION < 13000" "${HM}"; then
  "${PY}" - "${HM}" <<'PATCH' || FAIL "curobo_lerp_patch"
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = "inline __device__ __host__ float lerp(float a, float b, float t)\n{\n    return a + t*(b-a);\n}"
if needle not in text:
    raise SystemExit("scalar lerp overload not found; cuRobo layout changed")
text = text.replace(
    needle,
    "#if !defined(CUDART_VERSION) || (CUDART_VERSION < 13000)\n" + needle + "\n#endif",
    1,
)
path.write_text(text)
print("patched helper_math.h for CUDA 13")
PATCH
fi
if ! "${PY}" -c "import curobo.geom" >/dev/null 2>&1; then
  "${PY}" -m pip install --no-build-isolation --no-deps -e "${CUROBO_DIR}" || FAIL "curobo_build"
fi

# Isaac's torch must be untouched. If a transitive install swapped it, every result afterwards is
# measured against a different runtime than Gate 1 validated.
TORCH_AFTER="$("${PY}" -c 'import torch; print(torch.__version__)')"
[[ "${TORCH_BEFORE}" == "${TORCH_AFTER}" ]] || FAIL "torch_was_replaced_${TORCH_BEFORE}_to_${TORCH_AFTER}"

# --- GATE ----------------------------------------------------------------------------------
# The imports must be attempted the way the HARNESS does them: after the Isaac Sim app starts.
# `omni.kit.commands` and `isaacsim.core` are Kit EXTENSIONS, registered when the app boots, not
# ordinary site-packages. Checking them from a bare interpreter reported
#   tacex      -> ModuleNotFoundError: No module named 'omni.kit.commands'
#   tacex_uipc -> ModuleNotFoundError: No module named 'isaacsim.core'
# while every package was in fact installed correctly. That was a defective test, not a defective
# build, and it is exactly the failure mode where a check and the thing it checks disagree about
# what "installed" means.
"${PY}" - <<'PYEOF'
import json
import os

out = {}
try:
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = launcher.app
    out["app_launched"] = True
except Exception as exc:  # noqa: BLE001
    out["app_launched"] = False
    out["app_error"] = f"{type(exc).__name__}: {exc}"

if out["app_launched"]:
    for name in ("torch", "torch_scatter", "uipc", "tacex", "tacex_uipc", "curobo", "isaaclab"):
        try:
            mod = __import__(name)
            out[name] = {"ok": True, "path": getattr(mod, "__file__", "?")}
        except Exception as exc:  # noqa: BLE001 -- reporting the failure IS the gate
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

# curobo.geom explicitly, not just curobo: the top-level package imports fine even when the
# subpackage UniVTAC actually calls is absent, which is exactly how the wrong version passed here.
for name in ("curobo.geom.transform",):
    try:
        __import__(name)
        out[name] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

out["gate2"] = bool(out.get("app_launched")) and all(
    out.get(n, {}).get("ok") for n in ("tacex", "tacex_uipc", "curobo", "curobo.geom.transform")
)
# Write BEFORE shutting the app down. Isaac runs with fastShutdown, which can _exit() and discard
# anything still buffered -- an earlier probe in this project exited 0 with every print missing.
with open(os.environ["RESULT_IN"], "w") as handle:
    json.dump(out, handle, indent=2)
print("GATE2_PASS" if out["gate2"] else "GATE2_FAIL", json.dumps(out)[:400])
try:
    simulation_app.close()
except Exception:  # noqa: BLE001
    pass
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
