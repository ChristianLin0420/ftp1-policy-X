#!/bin/bash
# Runs inside the qualified UniVTAC container; see control_v2_closedloop.sbatch.
set -euo pipefail

: "${PYTHONPYCACHEPREFIX:?outer launcher must provide an empty job-owned bytecode cache}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX

: "${REPO_ROOT:?}" "${PAIR_ROOT:?}" "${V2_RUN:?}" "${V2_STEP:?}" "${FTP1_CHECKPOINT:?}"
: "${START_SEED:?}" "${TOTAL:?}" "${FTP1_OVERLAY:?}" "${OPENPI_DATA_HOME:?}"
RUN_OFFICIAL="${RUN_OFFICIAL:-1}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONHOME
export PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_VISIBLE_DEVICES=0,1
export HEADLESS=1 ENABLE_CAMERAS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/isaac-sim/python.sh
cd "${REPO_ROOT}/UniVTAC"

"${PY}" - <<'PY'
import numpy, torch

print(
    f"preflight numpy={numpy.__version__} torch={torch.__version__} "
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

common=(
  --task_list lift_bottle
  --domain_name UniVTAC_lift_bottle
  --task_config demo.yml
  --total_num "${TOTAL}"
  --start_seed "${START_SEED}"
  --workers 1
  --gpu ""
  --ftp1_device cuda:1
  --tactile_key right_tactile_gripper
  --tactile_sensor GelSightMini
  --policy_sample_seed 0
  --save_trajectory
  --trajectory_max_actions 500
  --no_video
)

echo "[$(date -Is)] student evaluation start"
export PYTHONPATH="${REPO_ROOT}/src"
export MOT_CONTROL_V2_RUN="${V2_RUN}" MOT_CONTROL_V2_STEP="${V2_STEP}"
timeout --signal=TERM --kill-after=120s "${RUN_SECONDS}" \
  "${PY}" "${REPO_ROOT}/scripts/eval_motjepa_control_v2.py" \
  "${common[@]}" \
  --policy_name mot_control_v2 \
  --checkpoint_dir "${V2_RUN}" \
  --arm_slice 0:7 \
  --gripper_index 7 \
  --save_root "${PAIR_ROOT}/student" \
  --run_id student

if [[ "${RUN_OFFICIAL}" == "1" ]]; then
  if [[ ! -f "${FTP1_OVERLAY}/READY.json" ]]; then
    echo "missing qualified FTP1 overlay ${FTP1_OVERLAY}/READY.json" >&2
    exit 2
  fi
  echo "[$(date -Is)] official FTP1 evaluation start"
  export PYTHONPATH="${FTP1_OVERLAY}:${REPO_ROOT}/src"
  "${PY}" - <<'PY'
import hashlib
import json
import os
import pathlib

import numpy
import torch
import transformers

from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

del FTP1InferenceWrapper
overlay = pathlib.Path(os.environ["FTP1_OVERLAY"]).resolve()
ready = json.loads((overlay / "READY.json").read_text())
torch_path = pathlib.Path(torch.__file__).resolve()
if overlay in pathlib.Path(torch.__file__).resolve().parents:
    raise SystemExit("FTP1 overlay must not shadow the qualified Isaac torch")
if ready.get("torch") != torch.__version__ or pathlib.Path(ready.get("torch_path", "")).resolve() != torch_path:
    raise SystemExit(f"runtime differs from qualified READY torch: ready={ready} live={torch.__version__} {torch_path}")
if transformers.__version__ != "4.53.2":
    raise SystemExit(f"expected patched transformers 4.53.2, got {transformers.__version__}")
if not numpy.__version__.startswith("1."):
    raise SystemExit(f"Isaac requires NumPy 1.x, got {numpy.__version__}")
tokenizer = pathlib.Path(ready.get("tokenizer_path", ""))
if not tokenizer.is_file() or hashlib.sha256(tokenizer.read_bytes()).hexdigest() != ready.get("tokenizer_sha256"):
    raise SystemExit(f"qualified tokenizer is missing or changed: {tokenizer}")
for key in ("augmax_path", "orbax_path"):
    qualified_path = overlay / ready.get(key, "")
    if not qualified_path.is_file():
        raise SystemExit(f"qualified overlay path {key} is missing: {qualified_path}")
print(
    f"official runtime numpy={numpy.__version__} torch={torch.__version__} "
    f"transformers={transformers.__version__}",
    flush=True,
)
PY

  timeout --signal=TERM --kill-after=120s "${RUN_SECONDS}" \
    "${PY}" "${REPO_ROOT}/UniVTAC/scripts/eval_ftp1.py" \
    "${common[@]}" \
    --policy_name official_ftp1 \
    --checkpoint_dir "${FTP1_CHECKPOINT}" \
    --arm_slice 9:16 \
    --gripper_index 44 \
    --num_inference_steps 10 \
    --chunk_first_n 20 \
    --action_rep mix \
    --temporal_ensemble \
    --ensemble_K 0.01 \
    --save_root "${PAIR_ROOT}/official" \
    --run_id official
fi

echo "[$(date -Is)] paired evaluators complete"
