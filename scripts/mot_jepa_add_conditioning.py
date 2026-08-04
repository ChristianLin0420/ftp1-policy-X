"""Add instruction and proprioception to an existing derived clip store, in place.

The derived stores built by ``mot_jepa_build_clip_zarr.py`` are self-supervised only: they
carry ``video``, ``gel`` and ``lowdim`` and nothing a post-training stage can condition on.
Stage 3 needs an instruction label per episode; Stage 4 needs proprioception per frame.

This is an **additive** pass, not a rebuild, for two measured reasons. Rebuilding 527 stores
costs about nine hours of cluster time, and the Lustre inode quota is at 24.3 M of 26.2 M --
only 1.9 M free. So the pass opens each derived store ``mode="r+"``, reads only the cheap
1-D and 2-D arrays from its *source* store (no image decode anywhere), and appends:

===========================  =====================  =============================
array                        shape                  why it lives there
===========================  =====================  =============================
``data/state``               ``(T, 120)`` float32   per-frame, FTP-1 slot order
``meta/action_mask``         ``(120,)`` uint8       constant: embodiment is fixed
``meta/instruction_id``      ``(E,)`` int32         constant: fixed per episode
===========================  =====================  =============================

Roughly 9 GB and 5.5 k inodes across the corpus. ``video`` and ``gel`` are never touched.

**Actions are deliberately not stored.** An action is a function of two states *and the
clip's stride*, and the dataset samples stride in {1, 2}; a stored per-frame action would be
silently wrong for every stride-2 clip. :func:`openpi.mot_jepa.action_parse.states_to_actions`
derives them at read time from whichever frames the clip actually used.

Usage::

    uv run python scripts/mot_jepa_add_conditioning.py \
        --source /lustre/.../FTP-1-Dataset --clips /lustre/.../ftp1-clips \
        --domains RDP sharpa
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs import BloscShuffle

from openpi.mot_jepa import action_parse as ap

# Matches the derived store's existing choice (lz4/NOSHUFFLE level 1): this array is read far
# more often than written, so decode speed is the constraint.
COMPRESSOR = BloscCodec(cname="lz4", clevel=1, shuffle=BloscShuffle.noshuffle)

#: 120 float32 is 480 B per frame, so 4096 frames is a ~1.9 MB chunk. Coarser than the clip
#: chunking used for video because this array is small enough that a whole chunk is cheap,
#: and every chunk is an inode we do not have much room for.
STATE_CHUNK_FRAMES = 4096

INSTRUCTION_KEY = "sub_task_instruction"


def episode_instructions(source: zarr.Group) -> list[str]:
    """One instruction string per episode.

    ``sub_task_instruction`` is stored per frame but is constant within an episode across
    every domain in the release (measured: zero episodes carry more than one string), so the
    first frame of each episode is the episode's label. Reading one frame per episode instead
    of the whole array is what keeps this pass cheap.
    """
    ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    raw = source["data"][INSTRUCTION_KEY]
    return [str(np.asarray(raw[int(start)]).item()).strip() for start in starts]


def add_conditioning(
    source_path: str,
    dest_path: pathlib.Path,
    vocabulary: dict[str, int],
    *,
    batch_frames: int = 65_536,
    overwrite: bool = False,
) -> dict:
    """Append the conditioning arrays to one derived store, reading from its source."""
    dest = zarr.open(str(dest_path), mode="r+")
    dest_data, dest_meta = dest["data"], dest["meta"]
    if "state" in set(dest_data.array_keys()) and not overwrite:
        return {"status": "skip", "reason": "already has data/state"}

    source = zarr.open(source_path, mode="r")
    total = int(np.asarray(dest_meta["episode_ends"][:])[-1])
    source_total = int(np.asarray(source["meta/episode_ends"][:])[-1])
    if total != source_total:
        # The derived store is a frame-for-frame copy; a mismatch means they are not a pair
        # and writing state anyway would misalign proprioception against video by an unknown
        # offset -- the exact class of defect that produces a plausible but meaningless model.
        raise ValueError(f"frame count mismatch: derived has {total}, source has {source_total}")

    spec = ap.specs_for_store(source["data"])
    if spec.is_empty:
        raise ValueError("no proprioceptive streams; a store with no hand joints cannot be posed")

    state_out = dest_data.require_array(
        "state",
        shape=(total, ap.ACTION_DIM),
        chunks=(STATE_CHUNK_FRAMES, ap.ACTION_DIM),
        dtype="float32",
        compressors=[COMPRESSOR],
    )
    start_time = time.time()
    for begin in range(0, total, batch_frames):
        end = min(begin + batch_frames, total)
        state_out[begin:end] = ap.read_state(source["data"], spec, begin, end)

    # Tighten against the data, not just the declared streams: a static camera leaves the
    # head block exactly constant, and nine dead slots marked "present" dilute every masked
    # mean and let the action embedder spend capacity on a constant.
    sample = np.asarray(state_out[:: max(1, total // 4096)], dtype=np.float32)
    mask = ap.drop_constant_columns(sample, ap.state_mask(spec))
    mask_out = dest_meta.require_array("action_mask", shape=mask.shape, dtype="uint8")
    mask_out[:] = mask

    instructions = episode_instructions(source)
    ids = np.asarray([vocabulary[text] for text in instructions], dtype=np.int32)
    id_out = dest_meta.require_array("instruction_id", shape=ids.shape, dtype="int32")
    id_out[:] = ids

    return {
        "status": "ok",
        "frames": total,
        "episodes": int(ids.size),
        "mask_slots": int(mask.sum()),
        "groups": sorted(name for name, (lo, hi) in ap.SLICES.items() if mask[lo:hi].any()),
        "unique_instructions": int(np.unique(ids).size),
        "seconds": round(time.time() - start_time, 1),
    }


def refresh_mask(dest_path: pathlib.Path) -> dict:
    """Recompute ``meta/action_mask`` from the state already written, without touching it.

    Cheap enough to run over the whole corpus: it reads a strided sample of ``data/state``
    and rewrites 120 bytes. Separate from the full pass so an existing corpus can be
    tightened without re-deriving anything.
    """
    dest = zarr.open(str(dest_path), mode="r+")
    if "state" not in set(dest["data"].array_keys()):
        return {"status": "skip", "reason": "no data/state"}
    state = dest["data"]["state"]
    stride = max(1, state.shape[0] // 4096)
    sample = np.asarray(state[::stride], dtype=np.float32)
    before = np.asarray(dest["meta"]["action_mask"][:], dtype=np.uint8)
    after = ap.drop_constant_columns(sample, before)
    dest["meta"]["action_mask"][:] = after
    return {"status": "ok", "before": int(before.sum()), "after": int(after.sum())}


def pair_stores(source_root: pathlib.Path, clips_root: pathlib.Path, domains: list[str] | None) -> list[tuple]:
    """Match each derived store to the source store it was built from, by name.

    ``mot_jepa_build_clip_zarr`` writes ``<clips>/<domain>/<name>.zarr`` from
    ``<source>/<domain>/**/<name>.zarr``, so the basename is the join key.
    """
    pairs = []
    for domain_dir in sorted(p for p in clips_root.iterdir() if p.is_dir()):
        if domains and domain_dir.name not in domains:
            continue
        source_by_name = {
            path.name: path
            for path in sorted((source_root / domain_dir.name).rglob("*.zarr"))
            if "extract" not in str(path)
        }
        pairs.extend(
            (domain_dir.name, source_by_name.get(derived.name), derived)
            for derived in sorted(domain_dir.glob("*.zarr"))
        )
    return pairs


def build_vocabulary(pairs: list[tuple]) -> tuple[dict[str, int], list[str]]:
    """Corpus-wide instruction vocabulary, assigned in sorted order so it is reproducible.

    A run-order-dependent vocabulary would give the same string a different id on a rerun,
    silently invalidating every ``meta/instruction_id`` written by an earlier pass.
    """
    texts: set[str] = set()
    for _, source, _ in pairs:
        if source is None:
            continue
        try:
            texts.update(episode_instructions(zarr.open(str(source), mode="r")))
        except Exception as exc:  # one unreadable store must not stop the sweep
            print(f"[warn] vocabulary: cannot read {source}: {type(exc).__name__}: {exc}")
    ordered = sorted(texts)
    return {text: index for index, text in enumerate(ordered)}, ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=pathlib.Path, required=True, help="Root of extracted source domains.")
    parser.add_argument("--clips", type=pathlib.Path, required=True, help="Root of derived clip stores.")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--vocabulary", type=pathlib.Path, default=None, help="Defaults to <clips>/instructions.json.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written; write nothing.")
    parser.add_argument(
        "--refresh-mask",
        action="store_true",
        help="Only recompute meta/action_mask from the state already present; write nothing else.",
    )
    args = parser.parse_args()

    pairs = pair_stores(args.source, args.clips, args.domains)
    if not pairs:
        parser.error(f"no derived stores found under {args.clips}")

    # The vocabulary is ALWAYS corpus-wide, never restricted to --domains. This script runs as
    # a SLURM array with one task per domain; a domain-local vocabulary would assign id 0 to a
    # different string in every task, and the resulting `meta/instruction_id` values would be
    # mutually meaningless with nothing downstream to notice.
    vocabulary_path = args.vocabulary or (args.clips / "instructions.json")
    if vocabulary_path.exists():
        ordered = json.loads(vocabulary_path.read_text())["instructions"]
        vocabulary = {text: index for index, text in enumerate(ordered)}
        print(f"[vocab] reusing {len(ordered)} instructions from {vocabulary_path}")
    else:
        vocabulary, ordered = build_vocabulary(pair_stores(args.source, args.clips, None))
        print(f"[vocab] built {len(ordered)} unique instructions corpus-wide")
        if not args.dry_run:
            # Content is a deterministic sorted list, so concurrent array tasks racing here all
            # write identical bytes; tmp + replace makes the destination never partially visible.
            vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
            staging = vocabulary_path.with_suffix(f".{os.getpid()}.tmp")
            staging.write_text(json.dumps({"instructions": ordered}, indent=2))
            os.replace(staging, vocabulary_path)

    if args.refresh_mask:
        tightened = 0
        for domain, _, derived in pairs:
            result = refresh_mask(derived)
            if result["status"] == "ok" and result["after"] < result["before"]:
                tightened += 1
                print(f"[tighten] {domain}/{derived.name}: {result['before']} -> {result['after']} slots")
        print(f"\ntightened {tightened} of {len(pairs)} store(s)")
        return 0

    written, skipped, failed = 0, 0, []
    for domain, source, derived in pairs:
        label = f"{domain}/{derived.name}"
        if source is None:
            print(f"[SKIP] {label}: no matching source store")
            failed.append(label)
            continue
        if args.dry_run:
            print(f"[dry-run] would write {label}")
            continue
        try:
            result = add_conditioning(str(source), derived, vocabulary, overwrite=args.overwrite)
        except Exception as exc:  # one bad store must not abort a whole domain
            print(f"[SKIP] {label}: {type(exc).__name__}: {exc}")
            failed.append(label)
            continue
        if result["status"] == "skip":
            skipped += 1
            continue
        written += 1
        print(
            f"[ok]   {label}: {result['frames']} frames, {result['episodes']} eps, "
            f"{result['mask_slots']} slots, {result['unique_instructions']} instr, {result['seconds']}s"
        )

    print(f"\nwrote {written}, skipped {skipped} already-done, {len(failed)} unusable")
    if failed:
        print(f"unusable: {failed}")
    return 0 if written or skipped or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
