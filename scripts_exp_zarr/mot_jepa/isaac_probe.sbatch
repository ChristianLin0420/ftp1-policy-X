#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=motjepa-isaacprobe
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --open-mode=append
#
# Eval Track B gate: can UniVTAC's closed-loop evaluation run on THIS cluster?
#
# The question is not about silicon. H100 has no RT cores, but UniVTAC's own code carries an
# "A800 headless" render workaround (UniVTAC/envs/_base_task.py:114) and A800 is equally
# RT-core-less, so the pipeline has demonstrably run on a datacenter GPU. The blocker is that
# Installation_FTP1.md provisions libnvidia-gl and writes /usr/share/vulkan/icd.d/nvidia_icd.json
# with sudo apt-get, and we have no root on compute nodes.
#
# So the real question is whether the NGC container supplies that userspace stack and whether
# the RTX renderer initializes headless. This probe answers it in under 40 GPU-minutes:
#   1. Is there an NVIDIA Vulkan ICD inside the container?
#   2. Does SimulationApp start headless with cameras enabled?
#   3. Does a camera actually return a non-trivial rendered frame?
#
# Step 3 is the decisive one. Steps 1-2 can pass while rendering silently returns black.
#
#   mkdir -p logs && sbatch scripts_exp_zarr/mot_jepa/isaac_probe.sbatch

set -uo pipefail

IMAGE="${IMAGE:-nvcr.io#nvidia/isaac-sim:4.5.0}"
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
PROBE_PY="${WORK}/logs/_isaac_probe_${SLURM_JOB_ID}.py"
RESULT_JSON="${WORK}/logs/_isaac_probe_${SLURM_JOB_ID}.json"

cat > "${PROBE_PY}" <<'PYEOF'
import json
import sys

# Results go to a FILE, not stdout. Isaac Sim runs with --/app/fastShutdown=True, which can
# _exit() and discard buffered stdout -- an earlier run of this probe exited 0 with every
# decisive print missing from the log, which would have been trivially mistaken for a pass.
RESULT_PATH = "/probe/result.json"
result = {"stage": "start"}


def record(**kwargs):
    result.update(kwargs)
    with open(RESULT_PATH, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.flush()


record(stage="init")

print("=== 1. Vulkan / GL userspace ===", flush=True)
import glob
import os
icds = sorted(glob.glob("/usr/share/vulkan/icd.d/*.json"))
print("vulkan ICDs:", icds or "NONE")
print("nvidia ICD present:", any("nvidia" in p.lower() for p in icds))
print("libnvidia-glcore:", bool(glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-glcore*")))
print("libnvoptix:", bool(glob.glob("/usr/lib/x86_64-linux-gnu/libnvoptix*")))
record(
    stage="userspace",
    vulkan_icds=icds,
    nvidia_icd=any("nvidia" in p.lower() for p in icds),
    libnvidia_glcore=bool(glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-glcore*")),
    libnvoptix=bool(glob.glob("/usr/lib/x86_64-linux-gnu/libnvoptix*")),
)

print("\n=== 2. SimulationApp headless ===", flush=True)
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "enable_cameras": True, "renderer": "RayTracedLighting"})

try:
    import numpy as np
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.0, 0.0, 0.5]),
                                  scale=np.array([0.3, 0.3, 0.3]), color=np.array([0.9, 0.2, 0.2])))
    world.reset()

    print("\n=== 3. Render a frame ===", flush=True)
    camera = rep.create.camera(position=(2.0, 2.0, 2.0), look_at=(0.0, 0.0, 0.5))
    product = rep.create.render_product(camera, (320, 240))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])

    for _ in range(40):
        world.step(render=True)

    array = np.asarray(annotator.get_data())
    record(stage="rendered", frame_shape=list(array.shape), frame_dtype=str(array.dtype))
    if array.size == 0:
        record(verdict="FAIL", reason="renderer returned an empty frame")
        app.close()
        sys.exit(2)
    nonzero = float((array[..., :3] > 0).mean())
    spread = float(array[..., :3].std())
    passed = bool(nonzero > 0.10 and spread > 1.0)
    record(
        verdict="PASS" if passed else "FAIL",
        nonblack_fraction=nonzero,
        pixel_std=spread,
        reason=None if passed else "frame is blank; the RTX path initialized but rendered nothing",
    )
    print(f"frame {array.shape} nonblack={nonzero:.3f} std={spread:.2f}", flush=True)
    app.close()
    sys.exit(0 if passed else 3)
except Exception as exc:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    record(verdict="FAIL", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    app.close()
    sys.exit(4)
PYEOF

echo "[$(date -Is)] isaac probe job=${SLURM_JOB_ID} image=${IMAGE} node=$(hostname)"

srun --container-image="${IMAGE}" \
     --container-mounts="${WORK}/logs:/probe" \
     --no-container-mount-home \
     bash -lc "nvidia-smi --query-gpu=name --format=csv,noheader; \
               /isaac-sim/python.sh /probe/$(basename "${PROBE_PY}")"
RC=$?

echo "[$(date -Is)] probe exit rc=${RC}"
echo "=== result.json ==="
cat "${WORK}/logs/result.json" 2>/dev/null || echo "(no result.json written)"

# The exit code alone is not evidence: fastShutdown can exit 0 having rendered nothing, and
# it discards buffered stdout. The verdict must come from the file.
VERDICT=$(python3 -c "import json;print(json.load(open('${WORK}/logs/result.json'))['verdict'])" 2>/dev/null || echo MISSING)
echo "[$(date -Is)] verdict=${VERDICT}"
case "${VERDICT}" in
  PASS) echo "GATE: PASS -- build the closed-loop UniVTAC eval track on this container"; exit 0 ;;
  FAIL) echo "GATE: FAIL -- stay on Track A offline probes; see result.json"; exit 1 ;;
  *)    echo "GATE: INCONCLUSIVE -- no verdict recorded; do not treat rc=${RC} as a pass"; exit 1 ;;
esac
