"""Build a derived, clip-shaped Zarr store for MoT-JEPA pretraining.

The released FTP-1 stores are the wrong *shape* for this workload, in two measured ways.

**Resolution.** Gel is stored at 224 squared but the model consumes 112 squared, so every
clip read decompresses 4x more gel bytes than it needs and then spends CPU shrinking them.
On the staged RDP domain a 16-frame clip costs ~682 ms end to end, of which roughly 360 ms
is zarr reads and the remaining ~320 ms is the resize.

**Chunk alignment.** RGB chunks are 14 frames, so a 16-frame clip straddles two chunks and
decompresses ~28 frames to use 16. Chunking at exactly ``num_frames`` makes one clip one
chunk.

Note the compressor is *not* the problem here: the shipped stores already use Blosc-lz4
(``cname='lz4'``, NOSHUFFLE), not the zstd-bitshuffle setting used elsewhere in the
repository, so this writer keeps lz4 rather than "switching" to it.

The output is written once, offline, and then staged to node-local NVMe per job.

Usage::

    uv run python scripts/mot_jepa_build_clip_zarr.py \
        --source /lustre/.../ftp1-zarr --output /lustre/.../ftp1-clips \
        --video-size 224 --gel-size 112
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import cv2
import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs import BloscShuffle

from openpi.mot_jepa import tactile_parse as tp
from openpi.mot_jepa.clip_dataset import discover_keys

cv2.setNumThreads(0)

# Keep lz4/NOSHUFFLE, matching what the released stores already use. Level 1 rather than 5:
# this store is read far more often than it is written, and decode speed is the constraint.
COMPRESSOR = BloscCodec(cname="lz4", clevel=1, shuffle=BloscShuffle.noshuffle)


def _resize_batch(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in frames])


#: Widest low-dimensional unit in the release is uSkin at 4x4x3 = 48.
DEFAULT_LOWDIM_WIDTH = 48


def build_store(
    source_path: str,
    dest_path: pathlib.Path,
    *,
    video_size: int,
    gel_size: int,
    num_frames: int,
    max_gel_pads: int,
    lowdim_width: int = DEFAULT_LOWDIM_WIDTH,
    max_lowdim_slots: int = 12,
    batch_frames: int = 512,
) -> dict:
    """Rewrite one store at model resolution with clip-aligned chunks.

    Every tactile stream is routed by :mod:`openpi.mot_jepa.tactile_parse`, which is what
    makes all three declared types survive the conversion: image keys contribute **every**
    pad (not just pad 0), channels-first images are transposed rather than reinterpreted as
    extra pads, ``matrix`` taxel grids keep all of their channels instead of being read as a
    scalar, and wide ``state`` wrenches keep all six components.
    """
    source = zarr.open(source_path, mode="r")
    data = source["data"]
    keys = discover_keys(source)
    specs = tp.specs_for_store(data)
    episode_ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
    total = int(episode_ends[-1])

    gel_specs = [spec for spec in specs if spec.route == tp.GEL]
    lowdim_specs = [spec for spec in specs if spec.route == tp.LOWDIM]

    # Flatten specs into concrete output slots, honouring the caps.
    gel_slots: list[tuple[tp.TactileSpec, int]] = []
    for spec in gel_specs:
        for unit in range(spec.num_units):
            if len(gel_slots) < max_gel_pads:
                gel_slots.append((spec, unit))
    lowdim_slots: list[tuple[tp.TactileSpec, int]] = []
    for spec in lowdim_specs:
        for unit in range(spec.num_units):
            if len(lowdim_slots) < max_lowdim_slots:
                lowdim_slots.append((spec, unit))

    dest = zarr.open(str(dest_path), mode="w")
    dest_data = dest.create_group("data")
    dest_meta = dest.create_group("meta")
    dest_meta.create_array("episode_ends", shape=episode_ends.shape, dtype="int64")
    dest_meta["episode_ends"][:] = episode_ends

    video_out = dest_data.create_array(
        "video",
        shape=(total, video_size, video_size, 3),
        chunks=(num_frames, video_size, video_size, 3),
        dtype="uint8",
        compressors=[COMPRESSOR],
    )
    num_pads = max(len(gel_slots), 1)
    gel_out = dest_data.create_array(
        "gel",
        shape=(total, num_pads, gel_size, gel_size, 3),
        chunks=(num_frames, num_pads, gel_size, gel_size, 3),
        dtype="uint8",
        compressors=[COMPRESSOR],
    )
    num_slots = max(len(lowdim_slots), 1)
    lowdim_out = dest_data.create_array(
        "lowdim",
        shape=(total, num_slots, lowdim_width),
        chunks=(num_frames * 64, num_slots, lowdim_width),
        dtype="float32",
        compressors=[COMPRESSOR],
    )

    start_time = time.time()
    for begin in range(0, total, batch_frames):
        end = min(begin + batch_frames, total)
        video_out[begin:end] = _resize_batch(np.asarray(data[keys.rgb][begin:end]), video_size)

        cache: dict[str, np.ndarray] = {}
        for slot, (spec, unit) in enumerate(gel_slots):
            if spec.key not in cache:
                cache[spec.key] = tp.read_gel(np.asarray(data[spec.key][begin:end]), spec)
            gel_out[begin:end, slot] = _resize_batch(cache[spec.key][:, unit], gel_size)

        cache.clear()
        for slot, (spec, unit) in enumerate(lowdim_slots):
            if spec.key not in cache:
                cache[spec.key] = tp.read_lowdim(np.asarray(data[spec.key][begin:end]), spec)
            width = min(spec.width, lowdim_width)
            lowdim_out[begin:end, slot, :width] = cache[spec.key][:, unit, :width]

        print(f"    {end}/{total} frames ({end / total:.0%})", end="\r", flush=True)

    elapsed = time.time() - start_time
    print(f"    {total}/{total} frames in {elapsed:.0f}s")
    truncated = [spec.key for spec, _ in lowdim_slots if spec.width > lowdim_width]
    if truncated:
        print(f"    WARNING truncated to {lowdim_width} channels: {sorted(set(truncated))}")
    gel_wanted = sum(spec.num_units for spec in gel_specs)
    low_wanted = sum(spec.num_units for spec in lowdim_specs)
    if gel_wanted > len(gel_slots):
        print(f"    WARNING dropped {gel_wanted - len(gel_slots)} gel pad(s): cap is {max_gel_pads}")
    if low_wanted > len(lowdim_slots):
        print(f"    WARNING dropped {low_wanted - len(lowdim_slots)} low-dim unit(s): cap is {max_lowdim_slots}")
    return {
        "source": source_path,
        "frames": total,
        "episodes": int(episode_ends.size),
        "video_size": video_size,
        "gel_size": gel_size,
        "gel_pads": len(gel_slots),
        "lowdim_slots": len(lowdim_slots),
        "lowdim_width": lowdim_width,
        "source_rgb_key": keys.rgb,
        "tactile_streams": [
            {
                "key": spec.key,
                "type": spec.tactile_type,
                "sensor": spec.sensor,
                "route": spec.route,
                "units": spec.num_units,
                "unit_shape": list(spec.unit_shape),
            }
            for spec in specs
        ],
        "truncated_streams": sorted(set(truncated)),
        "dropped_gel_units": gel_wanted - len(gel_slots),
        "dropped_lowdim_units": low_wanted - len(lowdim_slots),
        "chunk_frames": num_frames,
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=pathlib.Path, required=True, help="Root of staged domain directories.")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Destination root for derived stores.")
    parser.add_argument("--video-size", type=int, default=224)
    parser.add_argument("--gel-size", type=int, default=112)
    parser.add_argument("--num-frames", type=int, default=16, help="Clip length; sets the chunk size.")
    parser.add_argument("--max-gel-pads", type=int, default=2)
    parser.add_argument("--lowdim-width", type=int, default=DEFAULT_LOWDIM_WIDTH)
    parser.add_argument("--max-lowdim-slots", type=int, default=12)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources: list[tuple[str, pathlib.Path]] = []
    for domain_dir in sorted(p for p in args.source.iterdir() if p.is_dir()):
        if args.domains and domain_dir.name not in args.domains:
            continue
        stores = sorted(domain_dir.glob("*.zarr")) or sorted(domain_dir.glob("*/*.zarr"))
        sources.extend((domain_dir.name, store) for store in stores)
    if not sources:
        parser.error(f"no *.zarr stores found under {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"stores": [], "video_size": args.video_size, "gel_size": args.gel_size}
    skipped: list[str] = []

    for domain, store in sources:
        dest = args.output / domain / store.name
        if dest.exists() and not args.overwrite:
            print(f"[skip] {dest} exists (use --overwrite)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[build] {domain}/{store.name} -> {dest}")
        try:
            entry = build_store(
                str(store),
                dest,
                video_size=args.video_size,
                gel_size=args.gel_size,
                num_frames=args.num_frames,
                max_gel_pads=args.max_gel_pads,
                lowdim_width=args.lowdim_width,
                max_lowdim_slots=args.max_lowdim_slots,
            )
        except Exception as exc:
            # One unusable store must not abort a whole domain. Three FreeTacMan stores
            # (FragileCup, Stamp, Write) are tactile-only recordings with no camera array at
            # all; aborting on the first of them cost 36 of 44 stores on the corpus's largest
            # image-tactile domain. A video-tactile model cannot use a camera-less store, so
            # skipping is the correct outcome -- it just has to be loud rather than fatal.
            print(f"[SKIP] {domain}/{store.name}: {type(exc).__name__}: {exc}")
            shutil.rmtree(dest, ignore_errors=True)
            skipped.append(f"{domain}/{store.name}")
            continue
        entry["domain"] = domain
        entry["dest"] = str(dest)
        manifest["stores"].append(entry)

    # Per-domain manifests, never one shared file. This script is run as a SLURM array with
    # one task per domain, and a shared manifest would be a read-modify-write race: tasks
    # would clobber each other's entries and the record of what was built would be wrong in a
    # way nothing downstream would notice. Nothing reads the manifest to locate data -- the
    # dataset globs for *.zarr -- so per-domain files lose nothing.
    for entry in manifest["stores"]:
        domain_dir = args.output / entry["domain"]
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / "_manifest.json").write_text(
            json.dumps(
                {"video_size": args.video_size, "gel_size": args.gel_size, "stores": [entry]},
                indent=2,
            )
        )
    print(f"\nBuilt {len(manifest['stores'])} store(s); wrote per-domain _manifest.json")
    if skipped:
        print(f"Skipped {len(skipped)} unusable store(s): {skipped}")
    if not manifest["stores"] and skipped:
        print("ERROR: every store was skipped")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
