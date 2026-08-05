"""Clip sampling from FTP-1 Zarr stores.

Reads 16 contiguous frames at stride 1 or 2. The released stores chunk RGB along time at 14
frames (~2 MB), so one clip costs about two chunk decompressions -- cheap. *Strided* sampling
is what costs, which is why the stride stays small.

Three deliberate departures from ``dataset_zarr.py``:

**Clips that would cross an episode boundary are rejected, never clamped.** The FTP-1 loader
``np.clip``s indices, which silently repeats the final frame. Repeated frames make a
synchrony target degenerate -- two "different" instants become bit-identical, so the
objective can be satisfied without reading anything -- and they teach the model that motion
stops at episode ends.

**No ``scipy.interp1d`` and no per-frame Python resize loop.** The FTP-1 path resamples
through ``interp1d`` (returning float64) and resizes image tactile with a Python double loop
over ``T x N`` (``dataset_zarr.py:1493-1515``). Here indices are integers by construction and
resizes are batched over the frame axis.

**No actions.** ``data_loader.py:886`` raises unless a batch carries ``actions``, and
``ftp1_pytorch.py:435`` takes them positionally. A self-supervised objective has none, so
this is a separate loader rather than a patch of that one.

Images stay ``uint8`` until the GPU: the host-to-device copy is 4x cheaper than float32 and
normalization is a trivial elementwise op on device.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import re

import cv2
import numpy as np
import torch
import zarr

from openpi.mot_jepa import action_parse as ap
from openpi.mot_jepa import tactile_parse as tp
from openpi.mot_jepa.layout import TokenLayout

SUPPORTED_RGB_KEYS = (
    "camera_main_rgb",
    "camera_ego_rgb",
    "right_wrist_camera_rgb",
    "left_wrist_camera_rgb",
)

_TACTILE_DATA_RE = re.compile(r"^(?P<side>left|right)_tactile_data_(?P<detail>.+)$")

# cv2 is faster single-threaded here: the DataLoader already provides parallelism, and
# letting each worker spawn its own OpenCV pool oversubscribes the node badly.
cv2.setNumThreads(0)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class StoreKeys:
    """Which arrays of one store feed which stream."""

    rgb: str
    gel: tuple[str, ...]
    lowdim: tuple[str, ...]


def is_derived_store(store: zarr.Group) -> bool:
    """True for a store written by ``scripts/mot_jepa_build_clip_zarr.py``.

    Derived stores are already at model resolution with clip-aligned chunks, so reads skip
    both the resize and the straddled second chunk.
    """
    return "video" in set(store["data"].array_keys())


def is_degenerate(array: zarr.Array, *, num_samples: int = 32) -> bool:
    """True when a stream carries no signal at all (constant across space and time).

    Not hypothetical: in the released ``RDP_Bimanual`` store each hand has exactly one
    all-zero gel stream, and which one differs by hand -- ``left_gelsightmini`` and
    ``right_mctac`` are identically zero while their partners are live. The *type label*
    still says ``image``, so key presence alone reports them as usable tactile.

    Feeding such a stream to the model is worse than dropping it: a constant target is
    trivially predictable, so modes T and T_HARD score well on it while learning nothing,
    and it contributes no discriminative signal to the synchrony loss.
    """
    length = array.shape[0]
    if length == 0:
        return True
    index = np.unique(np.linspace(0, length - 1, min(num_samples, length)).astype(np.int64))
    sample = np.asarray(array[index]).astype(np.float32)
    return bool(sample.std() == 0.0)


def discover_keys(store: zarr.Group, *, prefer_rgb: str | None = None, drop_degenerate: bool = True) -> StoreKeys:
    """Pick the RGB, gel and low-dimensional arrays of a store by inspecting its metadata.

    ``drop_degenerate`` additionally checks *content*, not just the declared type, because a
    label of ``image`` is not evidence that a sensor was recording.
    """
    data = store["data"]
    keys = set(data.array_keys())
    if is_derived_store(store):
        return StoreKeys(rgb="video", gel=("gel",), lowdim=("lowdim",))

    rgb_candidates = [key for key in SUPPORTED_RGB_KEYS if key in keys]
    if not rgb_candidates:
        raise ValueError(f"store has none of the supported RGB keys {SUPPORTED_RGB_KEYS}")
    rgb = prefer_rgb if prefer_rgb in rgb_candidates else rgb_candidates[0]

    gel, lowdim = [], []
    for key in sorted(keys):
        match = _TACTILE_DATA_RE.match(key)
        if match is None:
            continue
        type_key = f"{match['side']}_tactile_type_{match['detail']}"
        tactile_type = str(data[type_key][-1]) if type_key in keys else ""
        if drop_degenerate and is_degenerate(data[key]):
            continue
        (gel if tactile_type == "image" else lowdim).append(key)
    return StoreKeys(rgb=rgb, gel=tuple(gel), lowdim=tuple(lowdim))


@dataclasses.dataclass(frozen=True)
class ClipIndexEntry:
    store_idx: int
    start: int
    stride: int
    episode_idx: int = 0
    """Which episode this clip lies in. Free here -- ``build`` already walks episodes -- and
    it is how a clip reaches its instruction, which is stored per episode rather than per
    frame because it is constant within an episode in every domain of the release."""


class ClipIndex:
    """Every clip start that fits entirely inside one episode."""

    def __init__(self, entries: np.ndarray, store_paths: list[str]) -> None:
        self.entries = entries  # (N, 4) int64 = (store_idx, start, stride, episode_idx)
        self.store_paths = store_paths

    def __len__(self) -> int:
        return int(self.entries.shape[0])

    def __getitem__(self, index: int) -> ClipIndexEntry:
        store_idx, start, stride, episode_idx = (int(v) for v in self.entries[index])
        return ClipIndexEntry(store_idx=store_idx, start=start, stride=stride, episode_idx=episode_idx)

    @classmethod
    def build(
        cls,
        store_paths: list[str],
        *,
        num_frames: int,
        strides: tuple[int, ...] = (1, 2),
        step: int = 1,
        action_horizon: int = 0,
    ) -> ClipIndex:
        """Enumerate valid clips by scanning only ``meta/episode_ends``.

        A clip beginning at ``s`` with stride ``r`` is valid when
        ``s + (num_frames - 1) * r < episode_end``. Starts that would run past the end are
        dropped outright rather than clamped.

        ``action_horizon > 0`` additionally requires the *future* window to fit:
        ``s + (num_frames - 1 + horizon) * r < episode_end``. A policy conditioned on a clip
        must predict actions that genuinely follow it, and an edge-padded chunk would be a
        chunk of repeated rows -- perfectly predictable, and it would inflate any metric
        computed over it. Same contract as the observation window: reject, never clamp.
        """
        rows = []
        unreadable: list[str] = []
        for store_idx, path in enumerate(store_paths):
            # One corrupt store must never take down a whole run. A build that dies partway
            # (we hit the Lustre INODE quota at 23.8M of 26.2M files) leaves directories that
            # look like stores but contain no zarr group; without this guard every rank
            # raises GroupNotFoundError during dataset construction and the job is dead
            # before step 0.
            try:
                ends = np.asarray(zarr.open(path, mode="r")["meta/episode_ends"][:], dtype=np.int64)
            except Exception as exc:
                logger.warning("skipping unreadable store %s: %s", path, type(exc).__name__)
                unreadable.append(path)
                continue
            if ends.size == 0:
                logger.warning("skipping store with no episodes: %s", path)
                unreadable.append(path)
                continue
            starts = np.concatenate([[0], ends[:-1]])
            for episode_idx, (episode_start, episode_end) in enumerate(zip(starts, ends, strict=True)):
                for stride in strides:
                    span = (num_frames - 1 + action_horizon) * stride
                    last_start = episode_end - 1 - span
                    if last_start < episode_start:
                        continue
                    candidates = np.arange(episode_start, last_start + 1, step, dtype=np.int64)
                    rows.append(
                        np.stack(
                            [
                                np.full(candidates.shape, store_idx, dtype=np.int64),
                                candidates,
                                np.full(candidates.shape, stride, dtype=np.int64),
                                np.full(candidates.shape, episode_idx, dtype=np.int64),
                            ],
                            axis=1,
                        )
                    )
        entries = np.concatenate(rows, axis=0) if rows else np.zeros((0, 4), dtype=np.int64)
        if unreadable:
            logger.warning("%d of %d stores were unreadable and excluded", len(unreadable), len(store_paths))
        return cls(entries, store_paths)

    def save(self, path: str | pathlib.Path) -> None:
        np.savez(path, entries=self.entries, store_paths=np.asarray(self.store_paths, dtype=object))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> ClipIndex:
        payload = np.load(path, allow_pickle=True)
        entries = payload["entries"]
        if entries.shape[1] == 3:
            # An index saved before episode_idx existed. Upgrading in place keeps a cached
            # index usable rather than silently mis-indexing instructions, which would pair
            # every clip with episode 0's text.
            logger.warning("upgrading a 3-column clip index from %s; episode_idx will be 0", path)
            entries = np.concatenate([entries, np.zeros((entries.shape[0], 1), dtype=np.int64)], axis=1)
        return cls(entries, [str(p) for p in payload["store_paths"]])


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize ``(T, H, W, C)`` uint8 frames to ``(T, size, size, C)``.

    Loops over the frame axis only. The FTP-1 path loops over ``T x N`` in Python, which for
    2 pads over a 16-frame clip is 32 interpreter round-trips per clip per sensor.
    """
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in frames])


@dataclasses.dataclass
class ClipSample:
    """One clip. Images are uint8; conversion happens on device."""

    video: torch.Tensor  # (T, 3, H, W) uint8
    gel: torch.Tensor  # (T, N, 3, h, w) uint8
    lowdim: torch.Tensor  # (T, S, C) float32
    gel_valid: torch.Tensor  # (N,) bool
    lowdim_valid: torch.Tensor  # (S,) bool
    domain_id: torch.Tensor  # () int64
    store_idx: torch.Tensor  # () int64
    start: torch.Tensor  # () int64
    stride: torch.Tensor  # () int64
    # Post-training conditioning. Always present so the batch dict has a stable key set --
    # a shape that varied with the store would deadlock DDP's all-reduce -- but only filled
    # when the dataset is built with ``with_conditioning=True``.
    state: torch.Tensor  # (T, 120) float32
    action: torch.Tensor  # (num_steps, 120) float32, derived from state at this clip's stride
    action_mask: torch.Tensor  # (120,) float32
    instruction_id: torch.Tensor  # () int64
    episode_idx: torch.Tensor  # () int64
    # Future action chunk for policy training. Zero-filled unless the dataset was built with
    # ``action_horizon > 0``; ``chunk_mask`` is the per-clip mask broadcast over the horizon,
    # matching what ``dataset_zarr.py`` emits so the two action pipelines agree.
    action_chunk: torch.Tensor  # (H, 120) float32
    chunk_mask: torch.Tensor  # (H, 120) float32


class MotJepaClipDataset(torch.utils.data.Dataset):
    """Clips from one or more Zarr stores, shaped for the MoT-JEPA token layout."""

    def __init__(
        self,
        store_paths: list[str],
        layout: TokenLayout,
        *,
        domain_ids: list[int] | None = None,
        strides: tuple[int, ...] = (1, 2),
        index_step: int = 1,
        clip_index: ClipIndex | None = None,
        lowdim_channels: int | None = None,
        prefer_rgb: str | None = None,
        with_conditioning: bool = False,
        action_horizon: int = 0,
    ) -> None:
        self.store_paths = list(store_paths)
        self.layout = layout
        self.domain_ids = domain_ids or [0] * len(store_paths)
        self.lowdim_channels = layout.lowdim_channels if lowdim_channels is None else lowdim_channels
        self.prefer_rgb = prefer_rgb
        # Off by default: pretraining has no use for proprioception, and reading it would add
        # a zarr access per clip for nothing.
        self.with_conditioning = with_conditioning
        # A horizon shrinks the index -- every clip must now have `horizon` frames of future
        # inside its own episode -- so it belongs in the index build, not just the read path.
        self.action_horizon = action_horizon
        self.clip_index = clip_index or ClipIndex.build(
            self.store_paths,
            num_frames=layout.num_frames,
            strides=strides,
            step=index_step,
            action_horizon=action_horizon,
        )
        if len(self.clip_index) == 0:
            raise ValueError(
                f"no usable clips across {len(self.store_paths)} store(s); every store was "
                "unreadable or had episodes shorter than one clip"
            )
        # Zarr handles are opened lazily per worker: an open handle is not fork-safe.
        self._stores: dict[int, zarr.Group] = {}
        self._keys: dict[int, StoreKeys] = {}
        self._derived: dict[int, bool] = {}
        self._spec_cache: dict[int, list[tp.TactileSpec]] = {}
        self._conditioned: dict[int, bool] = {}

    def __len__(self) -> int:
        return len(self.clip_index)

    def _store(self, store_idx: int) -> tuple[zarr.Group, StoreKeys]:
        if store_idx not in self._stores:
            group = zarr.open(self.store_paths[store_idx], mode="r")
            self._stores[store_idx] = group
            self._keys[store_idx] = discover_keys(group, prefer_rgb=self.prefer_rgb)
            self._derived[store_idx] = is_derived_store(group)
            specs = [] if self._derived[store_idx] else tp.specs_for_store(group["data"])
            self._spec_cache[store_idx] = specs
            self._warn_on_overflow(store_idx, specs)
            has_state = "state" in set(group["data"].array_keys())
            self._conditioned[store_idx] = has_state
            if self.with_conditioning and not has_state:
                logger.warning(
                    "%s has no data/state; run scripts/mot_jepa_add_conditioning.py. "
                    "Its clips will carry zeros and instruction_id -1",
                    self.store_paths[store_idx],
                )
        return self._stores[store_idx], self._keys[store_idx]

    def _conditioning(self, entry: ClipIndexEntry, frames: np.ndarray) -> dict[str, torch.Tensor]:
        """State, action mask and instruction id for one clip.

        Absent conditioning yields zeros and ``instruction_id = -1`` rather than ``0``: zero
        is a real vocabulary entry, so defaulting to it would quietly pair unlabelled clips
        with whichever instruction sorted first.
        """
        horizon = max(self.action_horizon, 1)
        out = {
            "state": torch.zeros(self.layout.num_frames, ap.ACTION_DIM, dtype=torch.float32),
            "action": torch.zeros(self.layout.num_steps, ap.ACTION_DIM, dtype=torch.float32),
            "action_mask": torch.zeros(ap.ACTION_DIM, dtype=torch.float32),
            "instruction_id": torch.tensor(-1, dtype=torch.int64),
            "episode_idx": torch.tensor(entry.episode_idx, dtype=torch.int64),
            "action_chunk": torch.zeros(horizon, ap.ACTION_DIM, dtype=torch.float32),
            "chunk_mask": torch.zeros(horizon, ap.ACTION_DIM, dtype=torch.float32),
        }
        if not (self.with_conditioning and self._conditioned[entry.store_idx]):
            return out
        group = self._stores[entry.store_idx]
        state = np.asarray(group["data"]["state"][frames], dtype=np.float32)
        mask = np.asarray(group["meta"]["action_mask"][:], dtype=np.float32)
        out["state"] = torch.from_numpy(state)
        out["action_mask"] = torch.from_numpy(mask)
        out["instruction_id"] = torch.tensor(int(group["meta"]["instruction_id"][entry.episode_idx]), dtype=torch.int64)

        # Actions are derived here, on the clip's OWN frames, which is the whole reason they
        # are not stored: they depend on the stride, and this dataset samples stride in {1,2}.
        # Taken at tubelet granularity so it lines up with the token time axis and no rotation
        # has to be composed inside a tubelet. The final step has no successor, so it repeats
        # rather than inventing one.
        anchors = state[:: self.layout.tubelet_t]
        actions = ap.actions_from_state(anchors, mask)
        out["action"] = torch.from_numpy(np.concatenate([actions, actions[-1:]], axis=0))

        if self.action_horizon > 0:
            # The chunk starts at the clip's LAST observed frame and runs forward. Reading
            # horizon+1 states gives horizon actions, and the index guarantees the window is
            # inside the episode, so nothing is padded and nothing crosses a boundary.
            last = int(frames[-1])
            future = np.arange(last, last + (self.action_horizon + 1) * entry.stride, entry.stride, dtype=np.int64)
            future_state = np.asarray(group["data"]["state"][future], dtype=np.float32)
            chunk = ap.actions_from_state(future_state, mask)
            out["action_chunk"] = torch.from_numpy(np.ascontiguousarray(chunk))
            out["chunk_mask"] = torch.from_numpy(mask).unsqueeze(0).expand(self.action_horizon, -1).contiguous()
        return out

    def _specs(self, store_idx: int) -> list[tp.TactileSpec]:
        return self._spec_cache[store_idx]

    def _warn_on_overflow(self, store_idx: int, specs: list[tp.TactileSpec]) -> None:
        """Say so when a store carries more tactile units than the layout has slots.

        Dropping a sensor because the layout is too small is a legitimate configuration
        choice, but it must never be invisible -- otherwise a domain quietly trains without
        half its touch data and nothing in the metrics says why.
        """
        gel_units = sum(spec.num_units for spec in specs if spec.route == tp.GEL)
        low_units = sum(spec.num_units for spec in specs if spec.route == tp.LOWDIM)
        if gel_units > self.layout.num_gel_pads:
            logger.warning(
                "%s: %d gel pads present but layout has %d; dropping %d",
                self.store_paths[store_idx],
                gel_units,
                self.layout.num_gel_pads,
                gel_units - self.layout.num_gel_pads,
            )
        if low_units > self.layout.lowdim_slots:
            logger.warning(
                "%s: %d low-dim units present but layout has %d slots; dropping %d",
                self.store_paths[store_idx],
                low_units,
                self.layout.lowdim_slots,
                low_units - self.layout.lowdim_slots,
            )

    def _read_derived(self, entry: ClipIndexEntry, frames: np.ndarray) -> ClipSample:
        """Fast path: arrays are already at model resolution and chunk-aligned to a clip.

        A derived store built for one layout can still be read by another (for example the
        base-resolution store feeding a pilot run), in which case the resize is reinstated for
        the mismatched stream only. Matching resolutions skip it entirely, which is the point.
        """
        data = self._stores[entry.store_idx]["data"]
        video = np.asarray(data["video"][frames])
        gel_src = np.asarray(data["gel"][frames])
        lowdim_src = np.asarray(data["lowdim"][frames], dtype=np.float32)

        video = _resize_frames(video, self.layout.video_size)

        gel = np.zeros(
            (self.layout.num_frames, self.layout.num_gel_pads, self.layout.gel_size, self.layout.gel_size, 3),
            dtype=np.uint8,
        )
        pads = min(gel_src.shape[1], self.layout.num_gel_pads)
        for pad in range(pads):
            gel[:, pad] = _resize_frames(gel_src[:, pad], self.layout.gel_size)
        gel_valid = torch.zeros(self.layout.num_gel_pads, dtype=torch.bool)
        gel_valid[:pads] = True

        lowdim = np.zeros((self.layout.num_frames, self.layout.lowdim_slots, self.lowdim_channels), dtype=np.float32)
        slots = min(lowdim_src.shape[1], self.layout.lowdim_slots)
        width = min(lowdim_src.shape[2], self.lowdim_channels)
        lowdim[:, :slots, :width] = lowdim_src[:, :slots, :width]
        lowdim_valid = torch.zeros(self.layout.lowdim_slots, dtype=torch.bool)
        lowdim_valid[:slots] = True

        return ClipSample(
            video=torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2))),
            gel=torch.from_numpy(np.ascontiguousarray(gel.transpose(0, 1, 4, 2, 3))),
            lowdim=torch.from_numpy(lowdim),
            gel_valid=gel_valid,
            lowdim_valid=lowdim_valid,
            domain_id=torch.tensor(self.domain_ids[entry.store_idx], dtype=torch.int64),
            store_idx=torch.tensor(entry.store_idx, dtype=torch.int64),
            start=torch.tensor(entry.start, dtype=torch.int64),
            stride=torch.tensor(entry.stride, dtype=torch.int64),
            **self._conditioning(entry, frames),
        )

    def __getitem__(self, index: int) -> ClipSample:
        entry = self.clip_index[index]
        group, keys = self._store(entry.store_idx)
        data = group["data"]
        frames = entry.start + np.arange(self.layout.num_frames, dtype=np.int64) * entry.stride
        if self._derived[entry.store_idx]:
            return self._read_derived(entry, frames)

        video = _resize_frames(np.asarray(data[keys.rgb][frames]), self.layout.video_size)
        video_t = torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2)))

        gel_stack = np.zeros(
            (self.layout.num_frames, self.layout.num_gel_pads, self.layout.gel_size, self.layout.gel_size, 3),
            dtype=np.uint8,
        )
        gel_valid = torch.zeros(self.layout.num_gel_pads, dtype=torch.bool)
        specs = self._specs(entry.store_idx)
        slot = 0
        for spec in (sp for sp in specs if sp.route == tp.GEL):
            units = tp.read_gel(np.asarray(data[spec.key][frames]), spec)
            for unit in range(spec.num_units):
                if slot >= self.layout.num_gel_pads:
                    break
                gel_stack[:, slot] = _resize_frames(units[:, unit], self.layout.gel_size)
                gel_valid[slot] = True
                slot += 1
        gel_t = torch.from_numpy(np.ascontiguousarray(gel_stack.transpose(0, 1, 4, 2, 3)))

        lowdim = np.zeros((self.layout.num_frames, self.layout.lowdim_slots, self.lowdim_channels), dtype=np.float32)
        lowdim_valid = torch.zeros(self.layout.lowdim_slots, dtype=torch.bool)
        slot = 0
        for spec in (sp for sp in specs if sp.route == tp.LOWDIM):
            units = tp.read_lowdim(np.asarray(data[spec.key][frames]), spec)
            width = min(spec.width, self.lowdim_channels)
            for unit in range(spec.num_units):
                if slot >= self.layout.lowdim_slots:
                    break
                lowdim[:, slot, :width] = units[:, unit, :width]
                lowdim_valid[slot] = True
                slot += 1

        return ClipSample(
            video=video_t,
            gel=gel_t,
            lowdim=torch.from_numpy(lowdim),
            gel_valid=gel_valid,
            lowdim_valid=lowdim_valid,
            domain_id=torch.tensor(self.domain_ids[entry.store_idx], dtype=torch.int64),
            store_idx=torch.tensor(entry.store_idx, dtype=torch.int64),
            start=torch.tensor(entry.start, dtype=torch.int64),
            stride=torch.tensor(entry.stride, dtype=torch.int64),
            **self._conditioning(entry, frames),
        )


def collate_clips(samples: list[ClipSample]) -> dict[str, torch.Tensor]:
    """Stack samples. No randomness here -- masks are built in the training loop."""
    return {
        field.name: torch.stack([getattr(sample, field.name) for sample in samples])
        for field in dataclasses.fields(ClipSample)
    }


def load_domain_config(path: str | pathlib.Path) -> list[tuple[str, str]]:
    """Read the existing FTP-1 domain-config JSON schema (``dataset_zarr.py:2409-2427``).

    Reusing the schema verbatim keeps these configs drop-in compatible with the repository's
    preflight tooling. Returns ``(domain_name, store_path)`` pairs, one per ``*.zarr`` found.
    """
    config = json.loads(pathlib.Path(path).read_text())
    pairs: list[tuple[str, str]] = []
    for entry in config.get("datasets", []):
        if not entry.get("enabled", True):
            continue
        root = pathlib.Path(entry["path"])
        name = entry.get("name", root.name)
        stores = sorted(root.glob("*.zarr")) or sorted(root.glob("*/*.zarr"))
        pairs.extend((name, str(store)) for store in stores)
    if not pairs:
        raise ValueError(f"no enabled *.zarr stores found via {path}")
    return pairs
