from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr

from openpi.mot_jepa.clip_dataset import ClipIndex
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import discover_keys
from openpi.mot_jepa.layout import TokenLayout

LAYOUT = TokenLayout(
    num_frames=4,
    tubelet_t=2,
    video_size=32,
    gel_size=32,
    num_gel_pads=2,
    lowdim_slots=2,
    video_width=32,
    tactile_width=16,
)


def make_store(path, episode_lengths: list[int], *, size: int = 48) -> str:
    """Synthetic store matching the FTP-1 Zarr contract."""
    total = int(sum(episode_lengths))
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    meta.create_array("episode_ends", shape=(len(episode_lengths),), dtype="int64")
    meta["episode_ends"][:] = np.cumsum(episode_lengths)

    # Frame index painted into pixel 0 so tests can verify exactly which frames were read.
    rgb = np.zeros((total, size, size, 3), dtype=np.uint8)
    rgb[:, 0, 0, 0] = np.arange(total, dtype=np.uint8)
    data.create_array("camera_ego_rgb", shape=rgb.shape, dtype="uint8")
    data["camera_ego_rgb"][:] = rgb

    for name in ("gelsightmini", "mctac"):
        key = f"left_tactile_data_gripper_{name}"
        data.create_array(key, shape=(total, 1, size, size, 3), dtype="uint8")
        data[key][:] = np.zeros((total, 1, size, size, 3), dtype=np.uint8)
        type_key = f"left_tactile_type_gripper_{name}"
        data.create_array(type_key, shape=(total,), dtype="<U5")
        data[type_key][:] = np.array(["image"] * total)

    force_key = "left_tactile_data_gripperforce_flexivgripper"
    data.create_array(force_key, shape=(total, 1, 1), dtype="float32")
    data[force_key][:] = np.arange(total, dtype=np.float32).reshape(total, 1, 1)
    force_type = "left_tactile_type_gripperforce_flexivgripper"
    data.create_array(force_type, shape=(total,), dtype="<U5")
    data[force_type][:] = np.array(["state"] * total)
    return str(path)


@pytest.fixture(name="store")
def store_fixture(tmp_path) -> str:
    return make_store(tmp_path / "toy.zarr", [10, 12, 9])


def test_discover_keys_splits_image_from_lowdim_tactile(store):
    keys = discover_keys(zarr.open(store, mode="r"))
    assert keys.rgb == "camera_ego_rgb"
    assert set(keys.gel) == {
        "left_tactile_data_gripper_gelsightmini",
        "left_tactile_data_gripper_mctac",
    }
    assert keys.lowdim == ("left_tactile_data_gripperforce_flexivgripper",)


def test_no_clip_crosses_an_episode_boundary(store):
    """The must-have property. Clamping instead would repeat the final frame, which makes a
    synchrony target degenerate and teaches the model that motion stops at episode ends."""
    ends = np.asarray(zarr.open(store, mode="r")["meta/episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    index = ClipIndex.build([store], num_frames=LAYOUT.num_frames, strides=(1, 2))
    assert len(index) > 0

    for i in range(len(index)):
        entry = index[i]
        last = entry.start + (LAYOUT.num_frames - 1) * entry.stride
        episode = int(np.searchsorted(ends, entry.start, side="right"))
        assert starts[episode] <= entry.start
        assert last < ends[episode], f"clip {entry} spans past episode end {ends[episode]}"


def test_clip_index_rejects_episodes_shorter_than_the_clip(tmp_path):
    path = make_store(tmp_path / "short.zarr", [3, 3])
    index = ClipIndex.build([path], num_frames=LAYOUT.num_frames, strides=(1,))
    assert len(index) == 0


def test_dataset_returns_layout_shaped_tensors(store):
    dataset = MotJepaClipDataset([store], LAYOUT, strides=(1,))
    sample = dataset[0]
    assert sample.video.shape == (LAYOUT.num_frames, 3, LAYOUT.video_size, LAYOUT.video_size)
    assert sample.gel.shape == (LAYOUT.num_frames, LAYOUT.num_gel_pads, 3, LAYOUT.gel_size, LAYOUT.gel_size)
    assert sample.lowdim.shape == (LAYOUT.num_frames, LAYOUT.lowdim_slots, 1)
    assert sample.video.dtype is torch.uint8, "images must stay uint8 until the GPU"
    assert sample.lowdim.dtype is torch.float32


def test_dataset_reads_exactly_the_requested_frames(store):
    """Frame identity is painted into pixel 0, so a stride bug is directly visible."""
    dataset = MotJepaClipDataset([store], LAYOUT, strides=(2,))
    entry = dataset.clip_index[0]
    sample = dataset[0]
    expected = entry.start + np.arange(LAYOUT.num_frames) * entry.stride
    # Pixel (0,0) survives INTER_AREA only approximately, so compare the low-dim channel,
    # which is stored exactly.
    np.testing.assert_allclose(sample.lowdim[:, 0, 0].numpy(), expected.astype(np.float32))


def test_lowdim_valid_flags_track_present_sensors(store):
    dataset = MotJepaClipDataset([store], LAYOUT, strides=(1,))
    sample = dataset[0]
    assert bool(sample.gel_valid.all()), "both gel pads are present in this store"
    assert bool(sample.lowdim_valid[0])
    assert not bool(sample.lowdim_valid[1]), "only one low-dim sensor exists"


def test_missing_gel_pads_are_zero_filled_and_marked_invalid(tmp_path):
    path = make_store(tmp_path / "one_pad.zarr", [12])
    root = zarr.open(path, mode="a")
    del root["data"]["left_tactile_data_gripper_mctac"]
    del root["data"]["left_tactile_type_gripper_mctac"]

    dataset = MotJepaClipDataset([path], LAYOUT, strides=(1,))
    sample = dataset[0]
    assert bool(sample.gel_valid[0])
    assert not bool(sample.gel_valid[1])
    assert torch.equal(sample.gel[:, 1], torch.zeros_like(sample.gel[:, 1]))


def test_collate_stacks_without_randomness(store):
    dataset = MotJepaClipDataset([store], LAYOUT, strides=(1,))
    batch = collate_clips([dataset[0], dataset[1]])
    assert batch["video"].shape[0] == 2
    assert batch["gel_valid"].shape == (2, LAYOUT.num_gel_pads)
    # Collation must be a pure function of its inputs.
    again = collate_clips([dataset[0], dataset[1]])
    assert torch.equal(batch["video"], again["video"])


def test_clip_index_roundtrips_through_disk(store, tmp_path):
    index = ClipIndex.build([store], num_frames=LAYOUT.num_frames, strides=(1, 2))
    path = tmp_path / "index.npz"
    index.save(path)
    restored = ClipIndex.load(path)
    assert len(restored) == len(index)
    np.testing.assert_array_equal(restored.entries, index.entries)
    assert restored.store_paths == index.store_paths


def test_dataset_is_index_stable_across_repeated_reads(store):
    dataset = MotJepaClipDataset([store], LAYOUT, strides=(1,))
    first = dataset[5]
    second = dataset[5]
    assert torch.equal(first.video, second.video)
    assert torch.equal(first.lowdim, second.lowdim)
