"""Measure the FTP-1 release before committing GPU hours to it.

Answers the questions the MoT-JEPA design flags as unmeasured, chiefly R4: *what fraction
of corpus hours carry image-type tactile?* The strong binding terms (modes ``T``,
``T_HARD``, ``X`` and ``L_sync``) only fire on image-tactile batches, so a corpus that is
mostly force-scalars would silently reduce the run to video-only pretraining.

Reads only metadata and small arrays -- never the RGB or gel payloads -- so it runs in
seconds per store on a login node.

Usage::

    uv run python scripts/mot_jepa_survey_corpus.py --data-root /path/to/ftp1-zarr
    uv run python scripts/mot_jepa_survey_corpus.py --data-root ... --output survey.json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import re
import sys

import numpy as np
import zarr

#: RGB keys the FTP-1 loader recognises (``dataset_zarr.py:1618-1624``).
SUPPORTED_RGB_KEYS = (
    "camera_main_rgb",
    "camera_ego_rgb",
    "right_wrist_camera_rgb",
    "left_wrist_camera_rgb",
)

#: Tactile types that the JEPA image path can consume as gel tokens.
IMAGE_TACTILE_TYPES = frozenset({"image"})

#: Frame rates outside this band indicate corrupt timestamps, not an unusual capture rate.
MIN_PLAUSIBLE_FPS = 1.0
MAX_PLAUSIBLE_FPS = 240.0

#: Placeholder instruction written by parsers for episodes with no real annotation.
UNANNOTATED_INSTRUCTIONS = frozenset({"", "finish tasks.", "finish tasks"})

_TACTILE_DATA_RE = re.compile(r"^(?P<side>left|right)_tactile_data_(?P<detail>.+)$")


@dataclasses.dataclass
class TactileStream:
    key: str
    side: str
    detail: str
    tactile_type: str
    sensor: str
    shape: tuple[int, ...]
    dtype: str
    degenerate: bool = False

    @property
    def is_image(self) -> bool:
        """Declared image type AND actually carrying signal.

        The type label is not evidence that a sensor was recording: the released
        ``RDP_Bimanual`` store labels two identically-zero streams as ``image``.
        """
        return self.tactile_type in IMAGE_TACTILE_TYPES and not self.degenerate

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "shape": list(self.shape), "is_image": self.is_image}


@dataclasses.dataclass
class StoreSurvey:
    domain: str
    store: str
    num_episodes: int
    num_frames: int
    fps: float
    hours: float
    fps_quality: str
    fps_assumed: bool
    rgb_keys: list[str]
    rgb_shape: list[int] | None
    rgb_chunks: list[int] | None
    tactile: list[TactileStream]
    instruction_coverage: float
    unique_instructions: int
    min_episode_len: int
    max_episode_len: int

    @property
    def has_image_tactile(self) -> bool:
        return any(stream.is_image for stream in self.tactile)

    def to_dict(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["tactile"] = [stream.to_dict() for stream in self.tactile]
        payload["has_image_tactile"] = self.has_image_tactile
        return payload


def _is_degenerate(array: zarr.Array, num_samples: int = 24) -> bool:
    """True when a tactile stream is constant, i.e. the sensor recorded nothing."""
    length = array.shape[0]
    if length == 0:
        return True
    index = np.unique(np.linspace(0, length - 1, min(num_samples, length)).astype(np.int64))
    return bool(np.asarray(array[index]).astype(np.float32).std() == 0.0)


def _scalar_str(array: zarr.Array) -> str:
    """Read the last element of a per-frame string array (type/sensor labels are constant)."""
    if array.shape[0] == 0:
        return ""
    return str(array[-1])


def _estimate_fps(timestamps: zarr.Array, episode_ends: np.ndarray) -> tuple[float, str]:
    """Median inter-frame rate, computed within episodes so boundaries do not skew it.

    Returns ``(fps, quality)``. Quality is ``"degenerate"`` when the stored timestamps
    cannot resolve a single frame interval, in which case ``fps`` is NaN and the caller
    must fall back to an assumed rate.

    This is not hypothetical: the released RDP stores hold absolute Unix epoch seconds in
    **float32**. At ~1.74e9 the float32 ulp is 128 s, so every frame in a clip rounds to the
    same value and consecutive differences are exactly 0 or 128. Deriving a rate from that
    array yields ~0.008 Hz and inflates an 0.8-hour corpus into 3000 hours. Detect it and
    say so rather than reporting a fabricated number.
    """
    raw = np.asarray(timestamps[:])
    values = raw.astype(np.float64)

    # A float32 array whose spacing is below its own representable resolution has already
    # lost the information; no amount of downstream arithmetic recovers it.
    if raw.dtype == np.float32 and values.size:
        magnitude = float(np.max(np.abs(values)))
        ulp = float(np.spacing(np.float32(magnitude))) if magnitude > 0 else 0.0
    else:
        ulp = 0.0

    starts = np.concatenate([[0], episode_ends[:-1]]) if episode_ends.size else np.zeros(0, dtype=np.int64)
    deltas = [np.diff(values[start:end]) for start, end in zip(starts, episode_ends, strict=True) if end - start >= 2]
    if not deltas:
        return float("nan"), "no-episodes"

    all_deltas = np.concatenate(deltas)
    zero_fraction = float(np.mean(all_deltas <= 0.0))
    positive = all_deltas[all_deltas > 0]
    if positive.size == 0:
        return float("nan"), "degenerate: all inter-frame deltas are zero"

    median_delta = float(np.median(positive))
    implied_fps = 1.0 / median_delta if median_delta > 0 else 0.0
    # Plausibility band. Some released stores have timestamps that parse cleanly but imply
    # ~0.014 Hz -- 320k frames over "6,472 hours" -- which silently inflates the hours
    # denominator. That alone drove an image-tactile fraction of 0.1% and a spurious R4
    # FALSIFIER TRIPPED, when the exact frame-based fraction is above 50%. A rate outside a
    # physically sensible band is not a measurement, it is corrupt metadata.
    if not (MIN_PLAUSIBLE_FPS <= implied_fps <= MAX_PLAUSIBLE_FPS):
        return float(
            "nan"
        ), f"implausible: implied {implied_fps:.4g} Hz outside [{MIN_PLAUSIBLE_FPS}, {MAX_PLAUSIBLE_FPS}]"
    if zero_fraction > 0.5:
        return float("nan"), (
            f"degenerate: {zero_fraction:.0%} of inter-frame deltas are zero (float32 epoch timestamps, ulp={ulp:g}s)"
        )
    if ulp > 0 and median_delta <= ulp:
        return float("nan"), f"degenerate: median delta {median_delta:g}s is at the float32 ulp ({ulp:g}s)"
    return float(1.0 / median_delta), "ok"


def survey_store(domain: str, store_path: pathlib.Path, *, assumed_fps: float) -> StoreSurvey:
    root = zarr.open(str(store_path), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    num_frames = int(episode_ends[-1]) if episode_ends.size else 0
    episode_lengths = np.diff(np.concatenate([[0], episode_ends])) if episode_ends.size else np.zeros(0, dtype=np.int64)

    keys = set(data.array_keys())

    rgb_keys = [key for key in SUPPORTED_RGB_KEYS if key in keys]
    rgb_shape = rgb_chunks = None
    if rgb_keys:
        first = data[rgb_keys[0]]
        rgb_shape = list(first.shape)
        rgb_chunks = list(first.chunks)

    tactile: list[TactileStream] = []
    for key in sorted(keys):
        match = _TACTILE_DATA_RE.match(key)
        if match is None:
            continue
        side, detail = match["side"], match["detail"]
        type_key = f"{side}_tactile_type_{detail}"
        sensor_key = f"{side}_tactile_sensor_{detail}"
        tactile.append(
            TactileStream(
                key=key,
                side=side,
                detail=detail,
                tactile_type=_scalar_str(data[type_key]) if type_key in keys else "<missing>",
                sensor=_scalar_str(data[sensor_key]) if sensor_key in keys else "<missing>",
                shape=tuple(int(dim) for dim in data[key].shape),
                dtype=str(data[key].dtype),
                degenerate=_is_degenerate(data[key]),
            )
        )

    if "timestamps" in keys:
        fps, fps_quality = _estimate_fps(data["timestamps"], episode_ends)
    else:
        fps, fps_quality = float("nan"), "missing: no timestamps array"

    fps_assumed = not (np.isfinite(fps) and fps > 0)
    effective_fps = assumed_fps if fps_assumed else fps
    hours = num_frames / effective_fps / 3600.0 if effective_fps > 0 else float("nan")

    coverage, unique = 0.0, 0
    if "sub_task_instruction" in keys:
        instructions = np.asarray(data["sub_task_instruction"][:]).astype(str)
        annotated = np.array([text.strip().lower() not in UNANNOTATED_INSTRUCTIONS for text in instructions])
        coverage = float(annotated.mean()) if annotated.size else 0.0
        unique = len({text for text in instructions if text.strip().lower() not in UNANNOTATED_INSTRUCTIONS})

    return StoreSurvey(
        domain=domain,
        store=store_path.name,
        num_episodes=int(episode_ends.size),
        num_frames=num_frames,
        fps=effective_fps,
        hours=hours,
        fps_quality=fps_quality,
        fps_assumed=fps_assumed,
        rgb_keys=rgb_keys,
        rgb_shape=rgb_shape,
        rgb_chunks=rgb_chunks,
        tactile=tactile,
        instruction_coverage=coverage,
        unique_instructions=unique,
        min_episode_len=int(episode_lengths.min()) if episode_lengths.size else 0,
        max_episode_len=int(episode_lengths.max()) if episode_lengths.size else 0,
    )


def find_stores(data_root: pathlib.Path, domains: list[str] | None) -> list[tuple[str, pathlib.Path]]:
    """Locate ``*.zarr`` stores under ``data_root``, tolerating an extra nesting level.

    ``stage_ftp1_archive.sh`` preserves the archive's own top-level directory, so a staged
    domain is usually ``<root>/<domain>/<domain>/*.zarr`` rather than ``<root>/<domain>/*.zarr``.
    """
    found: list[tuple[str, pathlib.Path]] = []
    for domain_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if domains and domain_dir.name not in domains:
            continue
        stores = sorted(domain_dir.glob("*.zarr")) or sorted(domain_dir.glob("*/*.zarr"))
        found.extend((domain_dir.name, store) for store in stores)
    return found


def aggregate(surveys: list[StoreSurvey]) -> dict:
    by_domain: dict[str, dict] = {}
    for survey in surveys:
        entry = by_domain.setdefault(
            survey.domain,
            {
                "stores": 0,
                "episodes": 0,
                "frames": 0,
                "hours": 0.0,
                "image_tactile_hours": 0.0,
                "image_tactile_frames": 0,
                "sensors": set(),
            },
        )
        entry["stores"] += 1
        entry["episodes"] += survey.num_episodes
        entry["frames"] += survey.num_frames
        hours = 0.0 if not np.isfinite(survey.hours) else survey.hours
        entry["hours"] += hours
        if survey.has_image_tactile:
            entry["image_tactile_hours"] += hours
            entry["image_tactile_frames"] += survey.num_frames
        entry["sensors"].update(stream.sensor for stream in survey.tactile)

    for entry in by_domain.values():
        entry["sensors"] = sorted(entry["sensors"])
        entry["image_tactile_fraction"] = entry["image_tactile_frames"] / entry["frames"] if entry["frames"] else 0.0

    total_hours = sum(entry["hours"] for entry in by_domain.values())
    image_hours = sum(entry["image_tactile_hours"] for entry in by_domain.values())
    total_frames = sum(entry["frames"] for entry in by_domain.values())
    image_frames = sum(entry["image_tactile_frames"] for entry in by_domain.values())
    return {
        "domains": by_domain,
        "total_stores": len(surveys),
        "total_episodes": sum(entry["episodes"] for entry in by_domain.values()),
        "total_frames": total_frames,
        "total_hours": total_hours,
        "image_tactile_hours": image_hours,
        "image_tactile_frames": image_frames,
        # Frames are counted exactly; hours depend on timestamps that are not always
        # trustworthy, so the gate reads frames.
        "image_tactile_fraction": image_frames / total_frames if total_frames else 0.0,
        "image_tactile_fraction_hours": image_hours / total_hours if total_hours else 0.0,
    }


def print_report(surveys: list[StoreSurvey], totals: dict, *, threshold: float) -> None:
    print("=" * 100)
    print("FTP-1 corpus survey")
    print("=" * 100)
    for survey in surveys:
        fps_note = f"{survey.fps:.2f}" + (" (ASSUMED)" if survey.fps_assumed else "")
        print(f"\n[{survey.domain}] {survey.store}")
        print(
            f"  episodes={survey.num_episodes}  frames={survey.num_frames}  "
            f"fps={fps_note}  hours={survey.hours:.2f}  "
            f"episode_len=[{survey.min_episode_len}, {survey.max_episode_len}]"
        )
        if survey.fps_assumed:
            print(f"  !! timestamps unusable -- {survey.fps_quality}")
        print(f"  rgb={survey.rgb_keys} shape={survey.rgb_shape} chunks={survey.rgb_chunks}")
        print(f"  instruction_coverage={survey.instruction_coverage:.1%} unique={survey.unique_instructions}")
        for stream in survey.tactile:
            if stream.degenerate:
                flag = "DEAD"
            elif stream.is_image:
                flag = "IMAGE"
            else:
                flag = stream.tactile_type.upper()
            note = "  <-- constant, no signal" if stream.degenerate else ""
            print(f"    [{flag:6s}] {stream.key:48s} {stream.sensor:20s} {tuple(stream.shape)}{note}")

    print("\n" + "=" * 100)
    print("Per-domain totals")
    print("=" * 100)
    header = f"{'domain':24s} {'stores':>6s} {'episodes':>9s} {'frames':>10s} {'hours':>8s} {'img-tac':>8s}"
    print(header)
    for name, entry in sorted(totals["domains"].items()):
        print(
            f"{name:24s} {entry['stores']:6d} {entry['episodes']:9d} {entry['frames']:10d} "
            f"{entry['hours']:8.2f} {entry['image_tactile_fraction']:7.1%}"
        )

    fraction = totals["image_tactile_fraction"]
    print(
        f"\nTOTAL: {totals['total_episodes']} episodes, {totals['total_frames']} frames, "
        f"{totals['total_hours']:.2f} hours"
    )
    print(
        f"Image-tactile: {totals['image_tactile_frames']:,} / {totals['total_frames']:,} frames "
        f"({fraction:.1%} of corpus) -- frames are exact; hours depend on timestamps"
    )

    assumed = [survey for survey in surveys if survey.fps_assumed]
    if assumed:
        print(
            f"\nNOTE: {len(assumed)}/{len(surveys)} stores have unusable `timestamps`; hours above are "
            "derived from --assumed-fps, not measured. Frame counts and the image-tactile fraction are exact."
        )

    # R4 gate from the design document.
    if fraction < threshold:
        print(
            f"\n*** R4 FALSIFIER TRIPPED: image-tactile is {fraction:.1%} of hours, below the "
            f"{threshold:.0%} threshold. The strong binding terms would fire on too few batches; "
            "oversample the image-tactile domains or reconsider the corpus. ***"
        )
    else:
        print(f"\nR4 gate PASSED: image-tactile {fraction:.1%} >= {threshold:.0%} threshold.")

    dead = [(s.domain, t.key) for s in surveys for t in s.tactile if t.degenerate]
    if dead:
        print(f"\nDEAD tactile streams ({len(dead)}) -- labelled but constant, excluded from the fraction above:")
        for domain, key in dead:
            print(f"  {domain}: {key}")

    sensor_counts = collections.Counter(
        stream.sensor for survey in surveys for stream in survey.tactile if stream.is_image
    )
    if sensor_counts:
        print(
            "\nImage-tactile sensors: " + ", ".join(f"{name} x{count}" for name, count in sensor_counts.most_common())
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=pathlib.Path, required=True, help="Directory of staged domain folders.")
    parser.add_argument("--domains", nargs="*", default=None, help="Restrict to these domain directory names.")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Write the full survey as JSON here.")
    parser.add_argument(
        "--image-tactile-threshold",
        type=float,
        default=0.05,
        help="R4 gate: minimum image-tactile fraction of corpus hours.",
    )
    parser.add_argument(
        "--assumed-fps",
        type=float,
        default=30.0,
        help="Frame rate used for hour estimates when a store's timestamps are unusable.",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        parser.error(f"--data-root does not exist: {args.data_root}")

    stores = find_stores(args.data_root, args.domains)
    if not stores:
        parser.error(f"no *.zarr stores found under {args.data_root}")

    surveys = [survey_store(domain, path, assumed_fps=args.assumed_fps) for domain, path in stores]
    totals = aggregate(surveys)
    print_report(surveys, totals, threshold=args.image_tactile_threshold)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"stores": [survey.to_dict() for survey in surveys], "totals": totals}, indent=2, default=str)
        )
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
