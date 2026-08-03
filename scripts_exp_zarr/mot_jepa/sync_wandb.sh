#!/bin/bash
# Push offline W&B runs from a login node.
#
# Compute nodes generally have no outbound internet, so the trainer writes offline and this
# syncs afterwards. Because every job in the chain reuses the same run id (run_dir/wandb_id.txt
# plus resume="must"), all of the offline directories belong to ONE W&B run and can be synced
# in any order.
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:?usage: sync_wandb.sh <run_dir>}}"
WANDB_BIN="${WANDB_BIN:-$(dirname "$(dirname "${RUN_DIR}")")/../../.venv/bin/wandb}"
[[ -x "${WANDB_BIN}" ]] || WANDB_BIN="$(command -v wandb)"

if [[ ! -d "${RUN_DIR}/wandb" ]]; then
  echo "no offline runs under ${RUN_DIR}/wandb" >&2
  exit 1
fi

echo "run id: $(cat "${RUN_DIR}/wandb_id.txt" 2>/dev/null || echo '<none recorded yet>')"
shopt -s nullglob
for offline in "${RUN_DIR}"/wandb/offline-run-*; do
  echo "syncing ${offline}"
  "${WANDB_BIN}" sync "${offline}"
done
echo "done"
