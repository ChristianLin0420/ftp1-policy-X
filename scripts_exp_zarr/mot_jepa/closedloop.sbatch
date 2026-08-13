#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill,batch_long
#SBATCH --time=03:55:00
#SBATCH --nodes=1
# TWO GPUs. Isaac's RTX renderer and a torch policy sharing one device died with
# VkResult ERROR_DEVICE_LOST and 60 s waits on "Failure to upload Texture", and bf16 did not
# help -- it is a context collision, not compute time. eval_ftp1 separates them by design via
# --ftp1_device (:70) and --sim_device, so give it two devices to separate onto.
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --job-name=motjepa-closedloop
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#
# Closed-loop UniVTAC success rate for the frozen MoT-JEPA encoder + per-domain ridge.
#
# Runs through eval_ftp1.py's own episode loop with only the policy class swapped, so this and the
# FTP-1 baseline share seeds, task config, success criterion and result accounting.
#
# TASK CONFIG IS `demo`, THE AUTHORS'. Do not substitute a custom one. A config that differed only
# in render_frequency (0 -> 2) took the scripted expert from 2/3 to 0/284: render_frequency 0 is
# not "no rendering", and tactile.py:370 has get_min_depth() call _update_render() itself, so extra
# renders desynchronise the gel depth from the UIPC FEM state and no grasp ever registers.

set -uo pipefail

IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
ASSETS="${ASSETS:-/lustre/fsw/portfolios/edgeai/users/chrislin/tacex-assets}"
TASK="${TASK:-lift_bottle}"
DOMAIN="${DOMAIN:-UniVTAC_lift_bottle}"
TOTAL="${TOTAL:-20}"
START_SEED="${START_SEED:-1000000}"
POLICY="${POLICY:-motjepa}"          # motjepa | ftp1
CKPT="${CKPT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-baselines/FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1/19999}"
RIDGE="${RIDGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/eval/ridge_policy.npz}"
SNAP="${SNAP:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runs/backbones/probe2_s50000}"
RUN_SECONDS="${RUN_SECONDS:-12600}"
RESULT="${WORK}/logs/_closedloop_${POLICY}_${SLURM_JOB_ID}.json"
RUN_SH="${WORK}/logs/_closedloop_${SLURM_JOB_ID}.sh"

mkdir -p "${WORK}/logs"

cat > "${RUN_SH}" <<'INNER'
set -x
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/local/cuda-13.0
export HEADLESS=1 ENABLE_CAMERAS=1
# Isaac's RTX renderer shares this GPU with the policy. Cap torch's allocator so it cannot
# starve texture uploads -- the previous run died with ERROR_DEVICE_LOST and 60 s semaphore
# waits on "Failure to upload Texture".
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${WORK_IN:?}"; : "${RESULT_IN:?}"; : "${POLICY_IN:?}"
PY=/isaac-sim/python.sh
cd "${WORK_IN}/UniVTAC" || { echo '{"ok": false, "stage": "cd"}' > "${RESULT_IN}"; exit 1; }

# numpy stays 1.x -- Isaac's prebuilt extensions are compiled against it and numpy 2 breaks
# isaacsim.core silently, with no error until the app boots.
"${PY}" -c "import numpy,sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)" \
  || { echo '{"ok": false, "stage": "numpy"}' > "${RESULT_IN}"; exit 1; }

if [[ "${POLICY_IN}" == "motjepa" ]]; then
  # openpi must be importable by ISAAC's interpreter, not our uv venv. PYTHONPATH rather than
  # `pip install -e`: the editable install failed silently behind a redirect, and a path entry
  # cannot drag in dependencies that would disturb Isaac's numpy or torch.
  export PYTHONPATH="${WORK_IN}/src"
  "${PY}" -c "import openpi.mot_jepa.deploy; print('openpi ok')" \
    || { echo '{"ok": false, "stage": "import_openpi"}' > "${RESULT_IN}"; exit 1; }
  SCRIPT="${WORK_IN}/scripts/eval_motjepa.py"
  # eval_ftp1 requires --checkpoint_dir even though our policy loads from RIDGE/SNAP. Point it at
  # the frozen backbone, which is the honest analogue: it IS the checkpoint this policy depends on.
  # _infer_action_rep_from_checkpoint is bypassed because eval_motjepa forces --action_rep absolute.
  EXTRA=(--checkpoint_dir "${SNAP}")
else
  export PYTHONPATH="${WORK_IN}/src"
  SCRIPT="${WORK_IN}/UniVTAC/scripts/eval_ftp1.py"
  EXTRA=(--checkpoint_dir "${CKPT_IN}")
fi

timeout "${RUN_SECONDS_IN}" "${PY}" "${SCRIPT}" \
  --task_list "${TASK_IN}" \
  --domain_name "${DOMAIN_IN}" \
  --ftp1_device "${POLICY_DEVICE_IN}" \
  --task_config demo.yml \
  --total_num "${TOTAL_IN}" \
  --start_seed "${START_SEED_IN}" \
  "${EXTRA[@]}"
echo "eval rc=$?"

# Verdict from ARTIFACTS. Isaac's fastShutdown can _exit() and discard buffered stdout.
"${PY}" - <<'PYEOF'
import glob, json, os, pathlib, re
out = {"policy": os.environ["POLICY_IN"], "task": os.environ["TASK_IN"]}
# Newest run directory only. A wall-clock window over eval_result/ picked up videos from
# earlier runs and would have reported another run's episodes as this one's.
# The run directory must have been created BY THIS JOB. Falling back to "newest on disk" made a
# failed FTP-1 run report the MoT-JEPA run's directory -- mid-write, at 2/2 -- as its own baseline.
# A fabricated number is worse than a missing one, so a job that produced no directory reports null.
base = pathlib.Path(os.environ["WORK_IN"]) / "UniVTAC" / "eval_result"
started = float(os.environ["JOB_START_EPOCH_IN"])
runs = sorted((d for d in base.glob("*/*/*") if d.is_dir() and d.stat().st_mtime >= started),
              key=lambda d: d.stat().st_mtime)
out["run_dir"] = str(runs[-1]) if runs else None
out["own_run_dir"] = bool(runs)
succ = fail = err = 0
for p in (runs[-1].rglob("*.mp4") if runs else []):
    b = p.name
    if "success" in b: succ += 1
    elif "error" in b: err += 1
    elif "fail" in b: fail += 1
out.update(success=succ, fail=fail, error=err, total=succ + fail + err)
# Rate over COMPLETED episodes; harness "error" episodes are crashes, not policy failures,
# and counting them as failures would understate the policy for the wrong reason.
out["success_rate"] = (succ / (succ + fail)) if (succ + fail) else None
out["ok"] = bool(runs) and out["total"] > 0
with open(os.environ["RESULT_IN"], "w") as h:
    json.dump(out, h, indent=2)
print("CLOSEDLOOP", json.dumps(out), flush=True)
PYEOF
INNER

export WORK_IN="${WORK}" RESULT_IN="${RESULT}" POLICY_IN="${POLICY}" TASK_IN="${TASK}"
export TOTAL_IN="${TOTAL}" START_SEED_IN="${START_SEED}" CKPT_IN="${CKPT}" RUN_SECONDS_IN="${RUN_SECONDS}"
# eval_ftp1.main() enforces exactly three required args (:1285, :1287, :1314):
# --checkpoint_dir, --domain_name and --task_list. All three are now supplied for both arms.
export DOMAIN_IN="${DOMAIN}"
export JOB_START_EPOCH_IN="$(date +%s)"
# --gpu "" is REQUIRED to keep both GPUs visible. It defaults to $CUDA_VISIBLE_DEVICES, which
# SLURM sets to "0,1", and _split_devices(:1006) then hands the worker exactly ONE device and
# rewrites CUDA_VISIBLE_DEVICES to it -- so cuda:1 disappears and the policy load dies with
# "CUDA error: invalid device ordinal". Empty leaves the env untouched (:1077 guard).
# Only --ftp1_device. --sim_device is rewritten to --device by _rewrite_cli_device_flags(:68)
# and this parser does not accept it; Isaac takes cuda:0 by default, which is what we want.
export POLICY_DEVICE_IN="${POLICY_DEVICE:-cuda:1}"
export RIDGE SNAP

echo "[$(date -Is)] closed-loop: policy=${POLICY} task=${TASK} n=${TOTAL} seed0=${START_SEED}"
srun --container-image="${IMAGE}" \
     --container-remap-root --container-writable \
     --container-mounts=/lustre:/lustre,${ASSETS}:${WORK}/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data \
     bash "${RUN_SH}"

echo "[$(date -Is)] srun rc=$?"
[[ -f "${RESULT}" ]] && { echo "===== verdict ====="; cat "${RESULT}"; } || echo "no verdict file"
