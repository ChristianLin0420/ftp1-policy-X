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
import sys
import time

import cv2
import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs import BloscShuffle

from openpi.mot_jepa.clip_dataset import discover_keys

cv2.setNumThreads(0)

# Keep lz4/NOSHUFFLE, matching what the released stores already use. Level 1 rather than 5:
# this store is read far more often than it is written, and decode speed is the constraint.
COMPRESSOR = BloscCodec(cname="lz4", clevel=1, shuffle=BloscShuffle.noshuffle)


def _resize_batch(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in frames])


def _normalize_gel(raw: np.ndarray) -> np.ndarray:
    """Collapse the source pad axis and force 3 channels."""
    if raw.ndim == 5:
        raw = raw[:, 0]
    if raw.ndim == 3:
        raw = raw[..., None]
    if raw.shape[-1] == 1:
        raw = np.repeat(raw, 3, axis=-1)
    return raw


def build_store(
    source_path: str,
    dest_path: pathlib.Path,
    *,
    video_size: int,
    gel_size: int,
    num_frames: int,
    max_gel_pads: int,
    batch_frames: int = 512,
) -> dict:
    """Rewrite one store at model resolution with clip-aligned chunks."""
    source = zarr.open(source_path, mode="r")
    data = source["data"]
    keys = discover_keys(source)
    episode_ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
    total = int(episode_ends[-1])

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
    num_pads = min(len(keys.gel), max_gel_pads)
    gel_out = dest_data.create_array(
        "gel",
        shape=(total, num_pads, gel_size, gel_size, 3),
        chunks=(num_frames, num_pads, gel_size, gel_size, 3),
        dtype="uint8",
        compressors=[COMPRESSOR],
    )
    num_lowdim = max(len(keys.lowdim), 1)
    lowdim_out = dest_data.create_array(
        "lowdim",
        shape=(total, num_lowdim, 1),
        chunks=(num_frames * 64, num_lowdim, 1),
        dtype="float32",
        compressors=[COMPRESSOR],
    )

    start_time = time.time()
    for begin in range(0, total, batch_frames):
        end = min(begin + batch_frames, total)
        video_out[begin:end] = _resize_batch(np.asarray(data[keys.rgb][begin:end]), video_size)
        for pad, key in enumerate(keys.gel[:num_pads]):
            gel_out[begin:end, pad] = _resize_batch(_normalize_gel(np.asarray(data[key][begin:end])), gel_size)
        for slot, key in enumerate(keys.lowdim[:num_lowdim]):
            raw = np.asarray(data[key][begin:end], dtype=np.float32).reshape(end - begin, -1)
            lowdim_out[begin:end, slot, 0] = raw[:, 0]
        print(f"    {end}/{total} frames ({end / total:.0%})", end="\r", flush=True)

    elapsed = time.time() - start_time
    print(f"    {total}/{total} frames in {elapsed:.0f}s")
    return {
        "source": source_path,
        "frames": total,
        "episodes": int(episode_ends.size),
        "video_size": video_size,
        "gel_size": gel_size,
        "gel_pads": num_pads,
        "lowdim_slots": num_lowdim,
        "source_rgb_key": keys.rgb,
        "source_gel_keys": list(keys.gel[:num_pads]),
        "source_lowdim_keys": list(keys.lowdim[:num_lowdim]),
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

    for domain, store in sources:
        dest = args.output / domain / store.name
        if dest.exists() and not args.overwrite:
            print(f"[skip] {dest} exists (use --overwrite)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[build] {domain}/{store.name} -> {dest}")
        entry = build_store(
            str(store),
            dest,
            video_size=args.video_size,
            gel_size=args.gel_size,
            num_frames=args.num_frames,
            max_gel_pads=args.max_gel_pads,
        )
        entry["domain"] = domain
        entry["dest"] = str(dest)
        manifest["stores"].append(entry)

    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        seen = {entry["dest"] for entry in manifest["stores"]}
        manifest["stores"] += [entry for entry in existing.get("stores", []) if entry["dest"] not in seen]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path} ({len(manifest['stores'])} stores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
