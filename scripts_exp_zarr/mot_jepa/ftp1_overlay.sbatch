#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=backfill,batch
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=ftp1-overlay
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append

# Build a versioned Python 3.10 dependency overlay for FTP1 inside the already-qualified UniVTAC
# Isaac image. The overlay intentionally does not contain torch: Isaac Sim owns that binary stack.

set -euo pipefail

IMAGE="${IMAGE:-/lustre/fsw/portfolios/edgeai/users/chrislin/images/univtac_full.sqsh}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
OVERLAY_OUT="${OVERLAY_OUT:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runtime/univtac_ftp1_py310_v1}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-runtime/openpi-data}"
BUILD="${OVERLAY_OUT}.building.${SLURM_JOB_ID}"

mkdir -p "${WORK}/logs" "$(dirname "${OVERLAY_OUT}")"
if [[ -e "${OVERLAY_OUT}" ]]; then
  echo "refuse existing overlay target ${OVERLAY_OUT}" >&2
  exit 2
fi
if [[ -e "${BUILD}" ]]; then
  echo "refuse existing build target ${BUILD}" >&2
  exit 2
fi

export WORK_IN="${WORK}" OVERLAY_OUT_IN="${OVERLAY_OUT}" BUILD_IN="${BUILD}" OPENPI_DATA_HOME
srun --container-image="${IMAGE}" \
     --container-remap-root --container-writable \
     --container-mounts=/lustre:/lustre \
     bash -c '
set -euo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH PYTHONHOME
export PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export JAX_PLATFORMS=cpu
PY=/isaac-sim/python.sh

"${PY}" - <<'"'"'PYEOF'"'"'
import sys
import torch

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Isaac Python 3.10, got {sys.version}")
print(f"base python={sys.version.split()[0]} torch={torch.__version__} path={torch.__file__}", flush=True)
PYEOF

mkdir "${BUILD_IN}"
"${PY}" -m pip install --no-cache-dir --target "${BUILD_IN}" \
  -c "${WORK_IN}/UniVTAC/requirements-univtac-infer-overlay-constraints.txt" \
  -r <(sed -e "s/^jax\[cuda12\]==0\.5\.3$/jax==0.5.3/" -e "/^timm$/d" \
    "${WORK_IN}/UniVTAC/requirements-univtac-infer-into-isaacsim.txt")
# `pip --target` resolves dependencies as though the target were empty, so an ordinary timm
# install tries to copy a second torch into it even though Isaac already supplies a compatible
# torch/torchvision pair. Install only timm itself, then prove its import against Isaac torch.
"${PY}" -m pip install --no-cache-dir --no-deps --target "${BUILD_IN}" timm==1.0.28

if find "${BUILD_IN}" -maxdepth 1 \( -name torch -o -name "torch-*.dist-info" \) -print -quit | grep -q .; then
  echo "overlay unexpectedly contains torch" >&2
  exit 3
fi
cp -a "${WORK_IN}/src/openpi/models_pytorch/transformers_replace/." "${BUILD_IN}/transformers/"

PYTHONPATH="${BUILD_IN}:${WORK_IN}/src" "${PY}" - <<'"'"'PYEOF'"'"'
import hashlib
import json
import os
import pathlib
import sys

import augmax
import flax
import jax
import numpy
import orbax.checkpoint
import sentencepiece
import timm
import torch
import transformers
import transformers.models.siglip.check
import isaaclab
import osqp
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.ftp1_pytorch import FTP1Pytorch
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

del FTP1InferenceWrapper, FTP1Pytorch
tokenizer = PaligemmaTokenizer()
del tokenizer
build = pathlib.Path(os.environ["BUILD_IN"])
if build in pathlib.Path(torch.__file__).resolve().parents:
    raise SystemExit(f"overlay shadowed Isaac torch: {torch.__file__}")
payload = {
    "python": sys.version,
    "torch": torch.__version__,
    "torch_path": torch.__file__,
    "numpy": numpy.__version__,
    "transformers": transformers.__version__,
    "jax": jax.__version__,
    "flax": flax.__version__,
    "timm": timm.__version__,
    "sentencepiece": sentencepiece.__version__,
    # The staging directory is atomically renamed after qualification. Store paths relative to
    # the overlay so READY.json remains truthful after that rename.
    "augmax_path": str(pathlib.Path(augmax.__file__).resolve().relative_to(build)),
    "orbax_path": str(pathlib.Path(orbax.checkpoint.__file__).resolve().relative_to(build)),
    "isaaclab_path": isaaclab.__file__,
    "osqp": osqp.__version__,
    "openpi_root": os.environ["WORK_IN"],
}
if payload["transformers"] != "4.53.2":
    raise SystemExit(f"wrong transformers version: {payload['"'"'transformers'"'"']}")
tokenizer_path = pathlib.Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision" / "paligemma_tokenizer.model"
if not tokenizer_path.is_file() or tokenizer_path.stat().st_size <= 0:
    raise SystemExit(f"missing cached tokenizer {tokenizer_path}")
payload["tokenizer_path"] = str(tokenizer_path)
payload["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
(build / "READY.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, sort_keys=True), flush=True)
PYEOF

mv "${BUILD_IN}" "${OVERLAY_OUT_IN}"
test -s "${OVERLAY_OUT_IN}/READY.json"
echo "FTP1_OVERLAY_READY=${OVERLAY_OUT_IN}"
'

test -s "${OVERLAY_OUT}/READY.json"
cat "${OVERLAY_OUT}/READY.json"
