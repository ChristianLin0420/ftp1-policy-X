#!/bin/bash
# Copy the derived clip store to node-local NVMe once per node.
#
# Worth it because the training read pattern is thousands of small chunk files, which is
# exactly the workload that saturates a Lustre MDS when 32 ranks do it concurrently. A
# sentinel makes node reuse free, which matters when a requeue lands on the same node.
#
# Sets DATA_GLOB to point at the staged copy.
set -euo pipefail

STAGE_SOURCE="${STAGE_SOURCE:?}"     # directory of *.zarr stores, or of domain dirs

# The clip tree is <root>/<domain>/*.zarr, but a single staged domain is <dir>/*.zarr.
# Pick whichever actually matches so both a whole corpus and one domain can be staged.
_glob_for() {
  if compgen -G "$1/*.zarr" > /dev/null; then echo "$1/*.zarr"; else echo "$1/*/*.zarr"; fi
}
STAGE_ROOT="${STAGE_ROOT:-/raid/${USER}/mot_jepa}"
NAME="$(basename "${STAGE_SOURCE}")"
DEST="${STAGE_ROOT}/${NAME}"

if [[ ! -d /raid ]]; then
  echo "[stage] no /raid on $(hostname); reading directly from ${STAGE_SOURCE}"
  export DATA_GLOB="$(_glob_for "${STAGE_SOURCE}")"
  return 0 2>/dev/null || exit 0
fi

# Content key: if the source changes, the sentinel stops matching and we re-stage.
KEY="$(du -sb "${STAGE_SOURCE}" 2>/dev/null | cut -f1)-$(find "${STAGE_SOURCE}" -name '*.zarr' | wc -l)"

if [[ -f "${DEST}/.stage_complete" && "$(cat "${DEST}/.stage_key" 2>/dev/null)" == "${KEY}" ]]; then
  echo "[stage] $(hostname): reusing ${DEST}"
else
  echo "[stage] $(hostname): copying ${STAGE_SOURCE} -> ${DEST}"
  rm -rf "${DEST}"
  mkdir -p "${DEST}"
  # -a preserves structure; this is a bulk sequential read, far kinder to Lustre than
  # thousands of random small reads during training.
  cp -a "${STAGE_SOURCE}/." "${DEST}/"
  echo "${KEY}" > "${DEST}/.stage_key"
  touch "${DEST}/.stage_complete"
  echo "[stage] $(hostname): done ($(du -sh "${DEST}" | cut -f1))"
fi

export DATA_GLOB="$(_glob_for "${DEST}")"
