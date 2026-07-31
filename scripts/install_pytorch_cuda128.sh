#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python_path="${FTP1_PYTHON_PATH:-${REPO_ROOT}/.venv/bin/python}"
if [ ! -x "${python_path}" ]; then
  echo "Python environment not found at ${python_path}; run 'uv sync --all-extras --dev' first." >&2
  exit 1
fi

# PyTorch 2.7.1's CUDA 12.8 wheels include sm_120 kernels for Blackwell while
# retaining the version pinned by this repository. Keep NumPy within the
# project's <2 constraint after the platform-wheel dependency resolution.
uv pip install --python "${python_path}" --reinstall \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${python_path}" numpy==1.26.4

"${python_path}" - <<'PY'
import torch

if "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError(f"Installed wheel lacks sm_120 support: {torch.cuda.get_arch_list()}")
result = (torch.arange(4, device="cuda") * 2).cpu().tolist()
print(f"Validated torch={torch.__version__}, CUDA={torch.version.cuda}, result={result}")
PY

echo "Use FTP1_UV_NO_SYNC=true with FTP-1 launchers to preserve this platform override."
