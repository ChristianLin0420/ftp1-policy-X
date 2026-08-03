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

cat > "${PROBE_PY}" <<'PYEOF'
import sys

print("=== 1. Vulkan / GL userspace ===", flush=True)
import glob
import os
icds = sorted(glob.glob("/usr/share/vulkan/icd.d/*.json"))
print("vulkan ICDs:", icds or "NONE")
print("nvidia ICD present:", any("nvidia" in p.lower() for p in icds))
print("libnvidia-glcore:", bool(glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-glcore*")))
print("libnvoptix:", bool(glob.glob("/usr/lib/x86_64-linux-gnu/libnvoptix*")))

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

    frame = annotator.get_data()
    array = np.asarray(frame)
    print("frame shape:", array.shape, "dtype:", array.dtype)
    if array.size == 0:
        print("RESULT: FAIL -- renderer returned an empty frame")
        sys.exit(2)
    nonzero = float((array[..., :3] > 0).mean())
    spread = float(array[..., :3].std())
    print(f"non-black pixel fraction: {nonzero:.3f}   pixel std: {spread:.2f}")
    if nonzero > 0.10 and spread > 1.0:
        print("RESULT: PASS -- RTX rendering works headless on this GPU inside the container")
    else:
        print("RESULT: FAIL -- frame is blank; the RTX path initialized but rendered nothing")
        sys.exit(3)
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print(f"RESULT: FAIL -- {type(exc).__name__}: {exc}")
    sys.exit(4)
finally:
    app.close()
PYEOF

echo "[$(date -Is)] isaac probe job=${SLURM_JOB_ID} image=${IMAGE} node=$(hostname)"

srun --container-image="${IMAGE}" \
     --container-mounts="${WORK}/logs:/probe" \
     --no-container-mount-home \
     bash -lc "nvidia-smi --query-gpu=name --format=csv,noheader; \
               /isaac-sim/python.sh /probe/$(basename "${PROBE_PY}")"
RC=$?

echo "[$(date -Is)] probe exit rc=${RC}"
case ${RC} in
  0) echo "GATE: PASS -- build the closed-loop UniVTAC eval track on this container" ;;
  *) echo "GATE: FAIL (rc=${RC}) -- stay on Track A offline probes; see the traceback above" ;;
esac
exit ${RC}
