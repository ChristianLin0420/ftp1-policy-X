#!/bin/bash
# Runs only the explicitly unqualified final V3 head inside the qualified UniVTAC container.
set -euo pipefail

: "${PYTHONPYCACHEPREFIX:?outer launcher must provide an empty job-owned bytecode cache}"
: "${REPO_ROOT:?}" "${PAIR_ROOT:?}" "${V2_RUN:?}" "${V2_STEP:?}" "${TRAIN_JOB_ID:?}"
: "${START_SEED:?}" "${TOTAL:?}" "${RUN_SECONDS:?}"
if [[ "${MOT_CONTROL_V2_DIAGNOSTIC_OUTER_STORE_VERIFIED:-}" != "1" ]]; then
  echo "diagnostic rollout requires outer prepared-store verification" >&2
  exit 2
fi

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONHOME
export PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_VISIBLE_DEVICES=0,1
export HEADLESS=1 ENABLE_CAMERAS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX
PY=/isaac-sim/python.sh
cd "${REPO_ROOT}/UniVTAC"

"${PY}" - <<'PY'
import numpy
import torch

print(
    f"diagnostic preflight numpy={numpy.__version__} torch={torch.__version__} "
    f"cuda_devices={torch.cuda.device_count()}",
    flush=True,
)
if not numpy.__version__.startswith("1."):
    raise SystemExit("Isaac extensions require numpy 1.x")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected exactly two visible GPUs, got {torch.cuda.device_count()}")
for index in range(2):
    print(f"cuda:{index}={torch.cuda.get_device_name(index)}", flush=True)
PY

export PYTHONPATH="${REPO_ROOT}/src"
export MOT_CONTROL_V2_RUN="${V2_RUN}"
export MOT_CONTROL_V2_STEP="${V2_STEP}"
export MOT_CONTROL_V2_TRAIN_JOB_ID="${TRAIN_JOB_ID}"
export MOT_CONTROL_V2_DIAGNOSTIC_UNQUALIFIED_FINAL=1

echo "[$(date -Is)] DIAGNOSTIC_ONLY student evaluation start"
timeout --signal=TERM --kill-after=120s "${RUN_SECONDS}" \
  "${PY}" "${REPO_ROOT}/scripts/eval_motjepa_control_v2_diagnostic.py" \
  --task_list lift_bottle \
  --domain_name UniVTAC_lift_bottle \
  --task_config demo.yml \
  --total_num "${TOTAL}" \
  --start_seed "${START_SEED}" \
  --workers 1 \
  --gpu "" \
  --ftp1_device cuda:1 \
  --tactile_key right_tactile_gripper \
  --tactile_sensor GelSightMini \
  --policy_sample_seed 0 \
  --save_trajectory \
  --trajectory_max_actions 500 \
  --no_video \
  --policy_name mot_control_v2_diagnostic_unqualified_final \
  --checkpoint_dir "${V2_RUN}" \
  --arm_slice 0:7 \
  --gripper_index 7 \
  --save_root "${PAIR_ROOT}/student" \
  --run_id student_diagnostic

echo "[$(date -Is)] DIAGNOSTIC_ONLY evaluator complete"
