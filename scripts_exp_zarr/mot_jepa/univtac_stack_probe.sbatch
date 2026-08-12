#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=univtac-stackprobe
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#
# What is actually inside univtac.sqsh, and what does the closed-loop eval still need?
#
# The build job installed Isaac Sim + Isaac Lab. TacEx, tacex_uipc and cuRobo were never
# attempted, and tacex_uipc is the one with real risk: no wheel, no prebuilt .so in the vendored
# tree, a C++20 + CUDA + vcpkg build whose reference dockerfile targets sm_89 while this cluster
# is sm_90. Before spending hours on that build this answers, in 25 GPU-minutes, which of the
# four imports are missing and which are already satisfied -- so the build targets exactly the
# gap rather than re-installing what is there.
#
# Verdict goes to a FILE. Isaac Sim runs with --/app/fastShutdown=True, which can _exit() and
# discard buffered stdout; an earlier probe exited 0 with every decisive print missing.
#
#   mkdir -p logs && sbatch scripts_exp_zarr/mot_jepa/univtac_stack_probe.sbatch

set -uo pipefail

IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac.sqsh}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
RESULT="${WORK}/logs/_stack_probe_${SLURM_JOB_ID}.json"
PROBE_PY="${WORK}/logs/_stack_probe_${SLURM_JOB_ID}.py"

cat > "${PROBE_PY}" <<'PYEOF'
import importlib
import json
import os
import subprocess
import sys

out = {"python": sys.version, "executable": sys.executable}

# The four the closed-loop harness needs. eval_ftp1.py reaches them transitively:
# envs/_base_task.py makes every task a UipcRLEnv (tacex_uipc), and envs/robot/robot.py imports
# cuRobo unconditionally at module scope.
for name in ("isaacsim", "isaaclab", "tacex", "tacex_uipc", "curobo", "torch", "torch_scatter"):
    try:
        module = importlib.import_module(name)
        out[name] = {"ok": True, "version": getattr(module, "__version__", "?"),
                     "path": getattr(module, "__file__", "?")}
    except Exception as exc:  # noqa: BLE001 -- reporting the failure IS the probe
        out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

if out.get("torch", {}).get("ok"):
    import torch
    out["torch_detail"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        # torch_scatter is pinned in TacEx's setup.py against a DIFFERENT torch build than the
        # install guide provides; the exact pair decides whether that wheel can be reused.
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

# Toolchain the libuipc build needs; missing entries are what the build job must provision.
for tool in ("cmake", "gcc-11", "g++-11", "nvcc", "ninja"):
    try:
        which = subprocess.run(["which", tool], capture_output=True, text=True, timeout=30)
        out[f"tool_{tool}"] = which.stdout.strip() or "MISSING"
    except Exception as exc:  # noqa: BLE001
        out[f"tool_{tool}"] = f"ERROR: {exc}"

with open(os.environ["RESULT_IN_CONTAINER"], "w") as handle:
    json.dump(out, handle, indent=2)
print("WROTE", os.environ["RESULT_IN_CONTAINER"])
PYEOF

export RESULT_IN_CONTAINER="/probe/$(basename "${RESULT}")"

srun --container-image="${IMAGE}" \
     --container-remap-root --container-writable \
     --container-mounts="${WORK}/logs:/probe,/lustre:/lustre" \
     bash -c '
set -x
# bash -c, NOT -lc: a login shell sources the profile, which puts the Lustre conda (3.13) ahead
# of Isaac Sim'"'"'s 3.10 on PATH. /lustre is mounted here, so that is a live hazard.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ISAACSIM_PATH="${ISAACSIM_PATH:-/isaac-sim}"
"${ISAACSIM_PATH}/python.sh" "/probe/'"$(basename "${PROBE_PY}")"'"
'

echo "[$(date -Is)] srun rc=$?"
if [[ -f "${RESULT}" ]]; then
  echo "===== stack inventory ====="
  cat "${RESULT}"
else
  echo "NO RESULT FILE at ${RESULT} -- the probe did not reach its write"
fi
