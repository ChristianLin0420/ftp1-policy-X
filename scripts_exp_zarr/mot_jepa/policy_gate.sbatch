#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=motjepa-policy-gate
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

# Short, non-exclusive DDP topology gate. Unlike the production/requeue launcher this can use
# four free GPUs on a mixed node, avoiding a wait for an entirely idle eight-GPU node.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
CONFIG_NAME="${CONFIG_NAME:?set CONFIG_NAME}"
SOURCE_HEAD="${SOURCE_HEAD:?set SOURCE_HEAD to the completed policy student.pt}"
SOURCE_ACTION_STATS="${SOURCE_ACTION_STATS:?set SOURCE_ACTION_STATS to the completed policy action_stats.npz}"
SOURCE_BACKBONE="${SOURCE_BACKBONE:?set SOURCE_BACKBONE to the exact 100k backbone snapshot}"
EXPECTED_MODE="${EXPECTED_MODE:?set EXPECTED_MODE to frozen, last_blocks, or full}"
EXPECTED_STEPS="${EXPECTED_STEPS:-100}"
GATE_PREEMPT_AT="${GATE_PREEMPT_AT:-0}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/mot_jepa_policy_train.py}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
SCRIPT_DIR="${REPO_ROOT}/scripts_exp_zarr/mot_jepa"
export REPO_ROOT RUN_DIR CONFIG_NAME SCRIPT_DIR TRAIN_SCRIPT GPUS_PER_NODE SOURCE_HEAD SOURCE_ACTION_STATS SOURCE_BACKBONE
export EXPECTED_MODE EXPECTED_STEPS GATE_PREEMPT_AT ALLOW_RESUME
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}"

if [[ "${ALLOW_RESUME}" == "1" ]]; then
  if [[ ! -f "${RUN_DIR}/run_config.json" || ! -f "${RUN_DIR}/checkpoints/latest" ]]; then
    echo "resume gate lacks an existing run/checkpoint at ${RUN_DIR}" >&2
    exit 2
  fi
elif [[ -e "${RUN_DIR}/run_config.json" || -e "${RUN_DIR}/checkpoints" || -e "${RUN_DIR}/DONE" ]]; then
  echo "fresh gate refuses existing run state at ${RUN_DIR}" >&2
  exit 2
fi
rm -f "${RUN_DIR}/PREEMPT_REQUEST"
echo "[$(date -Is)] gate job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} run=${RUN_DIR}"

watch_pid=""
if [[ "${GATE_PREEMPT_AT}" -gt 0 ]]; then
  if [[ "${ALLOW_RESUME}" == "1" ]]; then
    echo "GATE_PREEMPT_AT is only valid on a fresh gate phase" >&2
    exit 2
  fi
  (
    deadline=$((SECONDS + 1500))
    while (( SECONDS < deadline )); do
      if [[ -f "${RUN_DIR}/checkpoints/latest" ]]; then
        latest="$(cat "${RUN_DIR}/checkpoints/latest")"
        if [[ "${latest}" -ge "${GATE_PREEMPT_AT}" ]]; then
          touch "${RUN_DIR}/PREEMPT_REQUEST"
          echo "[$(date -Is)] requested controlled preemption after checkpoint ${latest}"
          exit 0
        fi
      fi
      sleep 1
    done
    echo "timed out waiting to request controlled preemption" >&2
    exit 1
  ) &
  watch_pid=$!
fi

set +e
srun --kill-on-bad-exit=1 bash "${SCRIPT_DIR}/srun_node.sh"
srun_rc=$?
set -e
if [[ -n "${watch_pid}" ]]; then
  kill "${watch_pid}" 2>/dev/null || true
  wait "${watch_pid}" 2>/dev/null || true
fi
if [[ "${srun_rc}" -ne 0 ]]; then
  echo "training srun failed with rc=${srun_rc}" >&2
  exit "${srun_rc}"
fi

if [[ ! -f "${RUN_DIR}/checkpoints/latest" ]]; then
  echo "missing checkpoint latest pointer" >&2
  exit 1
fi
FINAL_STEP="$(cat "${RUN_DIR}/checkpoints/latest")"
export FINAL_STEP

if [[ "${GATE_PREEMPT_AT}" -gt 0 ]]; then
  if [[ -f "${RUN_DIR}/DONE" ]]; then
    echo "controlled-preemption phase unexpectedly reached DONE" >&2
    exit 1
  fi
  if [[ "${FINAL_STEP}" -lt "${GATE_PREEMPT_AT}" || "${FINAL_STEP}" -ge "${EXPECTED_STEPS}" ]]; then
    echo "preemption checkpoint ${FINAL_STEP} is outside [${GATE_PREEMPT_AT}, ${EXPECTED_STEPS})" >&2
    exit 1
  fi
else
  if [[ ! -f "${RUN_DIR}/DONE" ]]; then
    echo "missing DONE marker" >&2
    exit 1
  fi
  if [[ "$(cat "${RUN_DIR}/DONE")" != "${EXPECTED_STEPS}" || "${FINAL_STEP}" != "${EXPECTED_STEPS}" ]]; then
    echo "unexpected DONE/latest $(cat "${RUN_DIR}/DONE")/${FINAL_STEP}, expected ${EXPECTED_STEPS}" >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import hashlib
import json
import math
import os
import pathlib

import numpy as np
import torch

from openpi.mot_jepa import config as config_module
from openpi.mot_jepa import runtime
from openpi.mot_jepa.policy_finetune import configure_backbone_trainability


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


run = pathlib.Path(os.environ["RUN_DIR"])
step = int(os.environ["FINAL_STEP"])
expected_mode = os.environ["EXPECTED_MODE"]
checkpoint = run / "checkpoints" / str(step)
cfg = config_module.PolicyConfig.from_json((run / "run_config.json").read_text())
if cfg.backbone_train_mode != expected_mode:
    raise SystemExit(f"mode {cfg.backbone_train_mode!r} != expected {expected_mode!r}")
if pathlib.Path(cfg.init_head_from).resolve() != pathlib.Path(os.environ["SOURCE_HEAD"]).resolve():
    raise SystemExit("run did not pin the requested completed source head")
if pathlib.Path(cfg.pretrained_run).resolve() != pathlib.Path(os.environ["SOURCE_BACKBONE"]).resolve():
    raise SystemExit("run did not pin the requested source backbone snapshot")
if cfg.pretrained_step != 100000:
    raise SystemExit(f"source backbone step is {cfg.pretrained_step}, expected 100000")
if cfg.num_train_steps != int(os.environ["EXPECTED_STEPS"]):
    raise SystemExit(f"frozen target step is {cfg.num_train_steps}, expected {os.environ['EXPECTED_STEPS']}")

source_head_path = pathlib.Path(os.environ["SOURCE_HEAD"]).resolve()
source_run = source_head_path.parent.parent.parent
source_step = int(source_head_path.parent.name)
source_config_path = source_run / "run_config.json"
expected_initialization = {
    "init_head_path": str(source_head_path),
    "init_head_sha256": file_digest(source_head_path),
    "init_head_source_run": str(source_run.resolve()),
    "init_head_source_step": source_step,
    "init_head_source_config_sha256": file_digest(source_config_path),
}

metadata = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
required_metadata = {
    "global_step": step,
    "backbone_train_mode": expected_mode,
    "source_backbone_step": 100000,
    "optimizer_step_min": step,
    "optimizer_step_max": step,
    "world_size": 4,
}
for key, expected in required_metadata.items():
    if metadata.get(key) != expected:
        raise SystemExit(f"metadata {key}={metadata.get(key)!r}, expected {expected!r}")
for key, expected in expected_initialization.items():
    if metadata.get(key) != expected:
        raise SystemExit(f"initialization metadata {key}={metadata.get(key)!r}, expected {expected!r}")
memory = metadata.get("peak_gpu_memory_bytes_by_rank")
if not isinstance(memory, list) or len(memory) != 4:
    raise SystemExit(f"expected four rank memory records, got {memory!r}")
allocated = [int(record.get("allocated", 0)) for record in memory]
reserved = [int(record.get("reserved", 0)) for record in memory]
if min(allocated) <= 0 or any(reserve < alloc for alloc, reserve in zip(allocated, reserved, strict=True)):
    raise SystemExit(f"invalid per-rank CUDA memory records: {memory!r}")
if max(reserved) >= 85 * 1024**3:
    raise SystemExit(f"peak reserved memory is not safe on an 80-GiB H100: {reserved}")
if max(allocated) / min(allocated) > 1.5:
    raise SystemExit(f"asymmetric per-rank allocated-memory peaks: {allocated}")

metrics = metadata.get("last_train_metrics", {})
for key in ("loss", "head_grad_norm", "backbone_grad_norm"):
    value = float(metrics.get(key, float("nan")))
    if not math.isfinite(value):
        raise SystemExit(f"non-finite/missing final metric {key}={value}")
if float(metrics["head_grad_norm"]) <= 0:
    raise SystemExit("head gradient norm did not become positive")
if expected_mode == "frozen":
    if float(metrics["backbone_grad_norm"]) != 0.0:
        raise SystemExit(f"frozen backbone gradient norm is {metrics['backbone_grad_norm']}")
elif float(metrics["backbone_grad_norm"]) <= 0:
    raise SystemExit("adapted backbone gradient norm did not become positive")

optimizer_state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
optimizer_steps = {
    int(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"])
    for value in optimizer_state["state"].values()
    if "step" in value
}
if optimizer_steps != {step}:
    raise SystemExit(f"optimizer tensor counters are {sorted(optimizer_steps)}, expected only {step}")
optimizer_parameter_ids = {
    parameter_id for group in optimizer_state["param_groups"] for parameter_id in group["params"]
}
optimizer_state_ids = set(optimizer_state["state"])
if optimizer_state_ids != optimizer_parameter_ids:
    raise SystemExit(
        f"optimizer state coverage differs: missing={sorted(optimizer_parameter_ids-optimizer_state_ids)[:8]} "
        f"extra={sorted(optimizer_state_ids-optimizer_parameter_ids)[:8]}"
    )

source_head = torch.load(os.environ["SOURCE_HEAD"], map_location="cpu", weights_only=True)
saved_head = torch.load(checkpoint / "student.pt", map_location="cpu", weights_only=True)
if source_head.keys() != saved_head.keys():
    raise SystemExit("source/saved head keys differ")
head_changed = [name for name in source_head if not torch.equal(source_head[name], saved_head[name])]
if not head_changed:
    raise SystemExit("head remained bit-identical after optimizer steps")

with np.load(os.environ["SOURCE_ACTION_STATS"], allow_pickle=False) as source_stats:
    with np.load(run / "action_stats.npz", allow_pickle=False) as saved_stats:
        if set(source_stats.files) != set(saved_stats.files):
            raise SystemExit("source/saved action-stat fields differ")
        for name in source_stats.files:
            if not np.array_equal(source_stats[name], saved_stats[name]):
                raise SystemExit(f"action stat {name} differs from the completed source run")

source_backbone, source_step = runtime.load_frozen_backbone(
    pathlib.Path(cfg.pretrained_run), cfg.pretrained_step, cfg, torch.device("cpu")
)
if source_step != 100000:
    raise SystemExit(f"loaded source step {source_step}, expected 100000")
selection = configure_backbone_trainability(
    source_backbone, mode=expected_mode, last_n_blocks=cfg.backbone_last_n_blocks
)
if metadata.get("trainable_backbone_name_sha256") != selection.name_digest:
    raise SystemExit("trainable backbone manifest digest differs from checkpoint metadata")
source_state = source_backbone.state_dict()
saved_state = torch.load(checkpoint / "backbone.pt", map_location="cpu", weights_only=True)
if source_state.keys() != saved_state.keys():
    raise SystemExit("source/saved backbone keys differ")
selected = set(selection.names)
changed_selected = []
for name in source_state:
    changed = not torch.equal(source_state[name], saved_state[name])
    if name in selected:
        if changed:
            changed_selected.append(name)
    elif changed:
        raise SystemExit(f"non-selected backbone tensor changed: {name}")
if expected_mode == "frozen" and changed_selected:
    raise SystemExit(f"frozen backbone changed: {changed_selected[:4]}")
if expected_mode != "frozen" and not changed_selected:
    raise SystemExit("no selected backbone tensor changed")

loaded = runtime.load_policy_backbone(run, step, cfg, torch.device("cpu"))
if tensor_digest(loaded.backbone.state_dict()) != tensor_digest(saved_state):
    raise SystemExit("deployment loader did not reproduce the saved backbone")

print(json.dumps({
    "ok": True,
    "step": step,
    "mode": expected_mode,
    "head_changed_tensors": len(head_changed),
    "backbone_changed_selected_tensors": len(changed_selected),
    "backbone_state_sha256": tensor_digest(saved_state),
    "optimizer_steps": sorted(optimizer_steps),
    "peak_gpu_memory_gib_by_rank": [round(value / 1024**3, 3) for value in allocated],
}, sort_keys=True))
PY

echo "[$(date -Is)] gate PASS mode=${EXPECTED_MODE} step=${FINAL_STEP} preempt_at=${GATE_PREEMPT_AT}"
