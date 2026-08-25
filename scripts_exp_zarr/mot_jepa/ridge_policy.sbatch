#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --job-name=motjepa-ridge
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

# Frozen-backbone UniVTAC linear probe. Required environment:
#   REPO_ROOT, SNAP, CLIPS (a quoted *.zarr glob), OUT
# Optional: STEP, HORIZON, OBSERVATION_STRIDE, ACTION_STRIDE, HOLDOUT_MOD,
# INDEX_STEP, BATCHES, BATCH_SIZE.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
SNAP="${SNAP:?set SNAP}"
CLIPS="${CLIPS:?set CLIPS to a quoted *.zarr glob}"
OUT="${OUT:?set OUT}"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${OUT}")"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false USE_SWANLAB=false OMP_NUM_THREADS=8

echo "[$(date -Is)] host=$(hostname) snap=${SNAP} step=${STEP:-latest} out=${OUT}"
srun "${REPO_ROOT}/.venv/bin/python" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(1, device="cuda"); print(torch.cuda.get_device_name(), x.device, flush=True)'

ARGS=(
  --pretrained_run "${SNAP}"
  --clips "${CLIPS}"
  --horizon "${HORIZON:-16}"
  --observation-stride "${OBSERVATION_STRIDE:-2}"
  --action-stride "${ACTION_STRIDE:-1}"
  --holdout_mod "${HOLDOUT_MOD:-10}"
  --index-step "${INDEX_STEP:-4}"
  --batches "${BATCHES:-1000}"
  --batch-size "${BATCH_SIZE:-32}"
  --out "${OUT}"
)
[[ -n "${STEP:-}" ]] && ARGS+=(--pretrained_step "${STEP}")

srun "${REPO_ROOT}/.venv/bin/python" scripts/mot_jepa_ridge_policy.py "${ARGS[@]}"

"${REPO_ROOT}/.venv/bin/python" - "${OUT}" "${OBSERVATION_STRIDE:-2}" "${ACTION_STRIDE:-1}" "${HORIZON:-16}" <<'PY'
import pathlib
import sys

import numpy as np

path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"missing result artifact: {path}")
with np.load(path, allow_pickle=False) as blob:
    actual = tuple(int(np.asarray(blob[key]).item()) for key in ("observation_stride", "action_stride", "horizon"))
    domains = [str(name) for name in blob["domains"]]
    fitted = {int(key.removeprefix("weights_")) for key in blob.files if key.startswith("weights_")}
expected = tuple(map(int, sys.argv[2:]))
if actual != expected:
    raise SystemExit(f"artifact sampling metadata {actual} != expected {expected}")
missing = [name for index, name in enumerate(domains) if index not in fitted]
if missing:
    raise SystemExit(f"ridge did not fit every domain: {missing}")
path.with_suffix(path.suffix + ".DONE").write_text("ok\n")
print(f"validated {path} ({path.stat().st_size} bytes), sampling={actual}, domains={len(domains)}", flush=True)
PY
echo "[$(date -Is)] DONE"
