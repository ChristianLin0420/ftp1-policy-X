from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import zarr

from openpi.mot_jepa import action_parse as ap
from openpi.mot_jepa.clip_dataset import ClipIndex
from openpi.mot_jepa.clip_dataset import MotJepaClipDataset
from openpi.mot_jepa.clip_dataset import collate_clips
from openpi.mot_jepa.clip_dataset import discover_keys
from openpi.mot_jepa.clip_dataset import is_degenerate
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

    rng = np.random.default_rng(0)
    for name in ("gelsightmini", "mctac"):
        key = f"left_tactile_data_gripper_{name}"
        data.create_array(key, shape=(total, 1, size, size, 3), dtype="uint8")
        # Must actually vary: a constant stream is now correctly treated as a dead sensor.
        data[key][:] = rng.integers(0, 255, (total, 1, size, size, 3), dtype=np.uint8)
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


def test_degenerate_streams_are_dropped_by_content_not_by_label(tmp_path):
    """A type label of ``image`` is not evidence that the sensor was recording.

    The released RDP_Bimanual store labels two identically-zero gel streams as ``image``;
    feeding those to the model is worse than dropping them, since a constant target is
    trivially predictable and contributes nothing to the synchrony loss.
    """
    path = make_store(tmp_path / "dead.zarr", [12])
    root = zarr.open(path, mode="a")
    dead_key = "left_tactile_data_gripper_gelsightmini"
    root["data"][dead_key][:] = 0  # sensor present in metadata, recording nothing

    assert is_degenerate(root["data"][dead_key])
    assert not is_degenerate(root["data"]["left_tactile_data_gripper_mctac"])

    keys = discover_keys(zarr.open(path, mode="r"))
    assert dead_key not in keys.gel
    assert "left_tactile_data_gripper_mctac" in keys.gel

    # Opting out keeps the declared-type behaviour, for surveying rather than training.
    labelled = discover_keys(zarr.open(path, mode="r"), drop_degenerate=False)
    assert dead_key in labelled.gel


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
    assert sample.lowdim.shape == (LAYOUT.num_frames, LAYOUT.lowdim_slots, LAYOUT.lowdim_channels)
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


def test_all_three_tactile_types_survive_the_source_read(tmp_path):
    """image, matrix and state together, which RH20TCfg7Tactile actually contains.

    Guards the four defects the corpus survey exposed: dropped second pads, channels-first
    images read as fake pads, ``matrix`` treated as a scalar, and wide ``state`` truncated to
    its first channel.
    """
    path = tmp_path / "mixed.zarr"
    total = 40
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    meta.create_array("episode_ends", shape=(1,), dtype="int64")
    meta["episode_ends"][:] = [total]
    rng = np.random.default_rng(0)

    data.create_array("camera_ego_rgb", shape=(total, 48, 48, 3), dtype="uint8")
    data["camera_ego_rgb"][:] = rng.integers(0, 255, (total, 48, 48, 3), dtype=np.uint8)

    def add(key, arr, ttype, sensor):
        data.create_array(key, shape=arr.shape, dtype=str(arr.dtype))
        data[key][:] = arr
        side, detail = key.split("_", 1)[0], key.split("_tactile_data_")[1]
        for name, value in ((f"{side}_tactile_type_{detail}", ttype), (f"{side}_tactile_sensor_{detail}", sensor)):
            a = data.create_array(name, shape=(total,), dtype=f"<U{max(len(value), 4)}")
            a[:] = np.array([value] * total)

    add("right_tactile_data_gel", rng.integers(1, 255, (total, 2, 24, 24, 3), dtype=np.uint8), "image", "FreeTacMan")
    add("right_tactile_data_uskin", rng.random((total, 2, 4, 4, 3)).astype("float32"), "matrix", "uSkin")
    add("right_tactile_data_ft", rng.random((total, 1, 6)).astype("float32"), "state", "ATIAxia80M20")

    # 2 uSkin pads + 1 wrench needs 3 low-dim slots; the default LAYOUT here has 2.
    layout = dataclasses.replace(LAYOUT, lowdim_slots=4)
    dataset = MotJepaClipDataset([str(path)], layout, strides=(1,))
    sample = dataset[0]

    # Both gel pads present and distinct -- not one pad duplicated or dropped.
    assert bool(sample.gel_valid[0])
    assert bool(sample.gel_valid[1])
    assert not torch.equal(sample.gel[:, 0], sample.gel[:, 1])

    # 2 uSkin pads + 1 wrench = 3 low-dim slots, none of them all-zero.
    assert int(sample.lowdim_valid.sum()) == 3
    assert float(sample.lowdim[:, 0].abs().sum()) > 0
    # uSkin carries 48 channels; the wrench carries 6. Both must exceed one.
    assert int((sample.lowdim[:, 0].abs().sum(dim=0) > 0).sum()) > 1
    assert int((sample.lowdim[:, 2].abs().sum(dim=0) > 0).sum()) > 1


def test_one_corrupt_store_does_not_kill_the_dataset(tmp_path):
    """A build that dies partway leaves a directory that looks like a store but has no group.

    That is not hypothetical: hitting the Lustre inode quota (23.8M of 26.2M files) left 14
    such shells among 511 derived stores, and every rank raised GroupNotFoundError during
    dataset construction -- the job was dead before step 0. One bad store out of hundreds
    must cost only that store.
    """
    good = make_store(tmp_path / "good.zarr", [20])
    broken = tmp_path / "broken.zarr"
    broken.mkdir()  # exists, looks like a store, contains no zarr group

    index = ClipIndex.build([good, str(broken)], num_frames=LAYOUT.num_frames, strides=(1,))
    assert len(index) > 0
    assert {int(index[i].store_idx) for i in range(len(index))} == {0}, "only the good store contributes"

    dataset = MotJepaClipDataset([good, str(broken)], LAYOUT, strides=(1,))
    assert dataset[0].video.shape[0] == LAYOUT.num_frames


def test_all_stores_unreadable_raises_rather_than_training_on_nothing(tmp_path):
    """build() may legitimately return empty (short episodes); the DATASET must refuse."""
    broken = tmp_path / "broken.zarr"
    broken.mkdir()
    assert len(ClipIndex.build([str(broken)], num_frames=LAYOUT.num_frames, strides=(1,))) == 0
    with pytest.raises(ValueError, match="no usable clips"):
        MotJepaClipDataset([str(broken)], LAYOUT, strides=(1,))


# --------------------------------------------------------------------------------------
# Post-training conditioning (data/state, meta/action_mask, meta/instruction_id)
# --------------------------------------------------------------------------------------


def add_conditioning_arrays(path: str, episode_lengths: list[int]) -> None:
    """What ``scripts/mot_jepa_add_conditioning.py`` appends to a derived store."""
    root = zarr.open(path, mode="r+")
    total = int(sum(episode_lengths))
    state = np.tile(np.arange(total, dtype=np.float32)[:, None], (1, ap.ACTION_DIM))
    root["data"].create_array("state", shape=state.shape, dtype="float32")
    root["data"]["state"][:] = state

    mask = np.zeros(ap.ACTION_DIM, dtype=np.uint8)
    mask[:10] = 1
    root["meta"].create_array("action_mask", shape=mask.shape, dtype="uint8")
    root["meta"]["action_mask"][:] = mask

    ids = np.arange(len(episode_lengths), dtype=np.int32) * 7
    root["meta"].create_array("instruction_id", shape=ids.shape, dtype="int32")
    root["meta"]["instruction_id"][:] = ids


def test_episode_index_identifies_the_clip_s_own_episode(tmp_path):
    """The instruction lives per episode, so a wrong episode_idx pairs a clip with the wrong text."""
    lengths = [10, 12, 9]
    path = make_store(tmp_path / "eps.zarr", lengths)
    index = ClipIndex.build([path], num_frames=LAYOUT.num_frames, strides=(1,))
    ends = np.cumsum(lengths)
    starts = np.concatenate([[0], ends[:-1]])
    for position in range(len(index)):
        entry = index[position]
        lo, hi = starts[entry.episode_idx], ends[entry.episode_idx]
        assert lo <= entry.start < hi, f"clip at {entry.start} tagged episode {entry.episode_idx} = [{lo},{hi})"


def test_conditioning_is_off_by_default_and_costs_nothing(tmp_path):
    """Pretraining has no use for proprioception and must not pay a zarr read for it.

    ``instruction_id`` defaults to -1, not 0: zero is a real vocabulary entry, so defaulting
    to it would silently pair every unlabelled clip with whichever instruction sorted first.
    """
    path = make_store(tmp_path / "plain.zarr", [20])
    sample = MotJepaClipDataset([path], LAYOUT, strides=(1,))[0]
    assert sample.state.shape == (LAYOUT.num_frames, ap.ACTION_DIM)
    assert float(sample.state.abs().sum()) == 0.0
    assert float(sample.action_mask.sum()) == 0.0
    assert int(sample.instruction_id) == -1


def test_conditioning_reads_the_clip_s_own_frames_and_episode(tmp_path):
    lengths = [20, 20]
    path = make_store(tmp_path / "cond.zarr", lengths)
    add_conditioning_arrays(path, lengths)
    dataset = MotJepaClipDataset([path], LAYOUT, strides=(1, 2), with_conditioning=True)

    for position in range(len(dataset)):
        entry = dataset.clip_index[position]
        sample = dataset[position]
        expected = entry.start + np.arange(LAYOUT.num_frames) * entry.stride
        # state was painted with the frame index, so this checks the stride was honoured.
        np.testing.assert_allclose(sample.state[:, 0].numpy(), expected)
        assert int(sample.instruction_id) == entry.episode_idx * 7
        assert float(sample.action_mask.sum()) == 10.0


def test_a_store_without_conditioning_still_loads_when_it_is_requested(tmp_path):
    """A half-migrated corpus must degrade to zeros, not crash a 32-rank job at step 0."""
    plain = make_store(tmp_path / "plain.zarr", [20])
    dataset = MotJepaClipDataset([plain], LAYOUT, strides=(1,), with_conditioning=True)
    sample = dataset[0]
    assert int(sample.instruction_id) == -1
    assert float(sample.state.abs().sum()) == 0.0


def test_collate_carries_the_conditioning_keys(tmp_path):
    lengths = [20]
    path = make_store(tmp_path / "c.zarr", lengths)
    add_conditioning_arrays(path, lengths)
    dataset = MotJepaClipDataset([path], LAYOUT, strides=(1,), with_conditioning=True)
    batch = collate_clips([dataset[0], dataset[1]])
    assert batch["state"].shape == (2, LAYOUT.num_frames, ap.ACTION_DIM)
    assert batch["action_mask"].shape == (2, ap.ACTION_DIM)
    assert batch["instruction_id"].shape == (2,)


def test_an_old_three_column_index_upgrades_rather_than_mis_indexing(tmp_path):
    path = make_store(tmp_path / "s.zarr", [20])
    index = ClipIndex.build([path], num_frames=LAYOUT.num_frames, strides=(1,))
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, entries=index.entries[:, :3], store_paths=np.asarray([path], dtype=object))

    upgraded = ClipIndex.load(legacy)
    assert upgraded.entries.shape[1] == 4
    assert upgraded[0].episode_idx == 0


# --------------------------------------------------------------------------------------
# Future action chunks for policy training
# --------------------------------------------------------------------------------------


def test_action_horizon_shrinks_the_index_and_never_crosses_an_episode(tmp_path):
    """Every chunk must lie inside its clip's own episode.

    A chunk that ran past the end would splice the next episode's motion onto this one's
    observation -- a target no policy could ever be right about, and one that looks like noise
    rather than a bug. Same reject-never-clamp contract the observation window already has.
    """
    lengths = [40, 45, 38]
    path = make_store(tmp_path / "chunk.zarr", lengths)
    add_conditioning_arrays(path, lengths)

    horizon = 8
    plain = ClipIndex.build([path], num_frames=LAYOUT.num_frames, strides=(1,))
    gated = ClipIndex.build([path], num_frames=LAYOUT.num_frames, strides=(1,), action_horizon=horizon)
    assert len(gated) < len(plain), "a horizon must remove clips that have no future left"

    ends = np.cumsum(lengths)
    starts = np.concatenate([[0], ends[:-1]])
    for store_idx, start, stride, episode_idx in gated.entries:
        del store_idx
        last_observed = start + (LAYOUT.num_frames - 1) * stride
        assert last_observed + horizon * stride < ends[episode_idx]
        assert start >= starts[episode_idx]


def test_action_chunk_is_the_future_not_the_observed_clip(tmp_path):
    """The chunk must start where the observation ends, or the policy is predicting the past.

    ``state`` here is frame_index broadcast over all 120 columns, so a first difference is
    exactly 1.0 on every live column -- which makes an off-by-one in the window visible as a
    wrong *value*, not merely a wrong shape.
    """
    lengths = [60]
    path = make_store(tmp_path / "future.zarr", lengths)
    add_conditioning_arrays(path, lengths)

    horizon = 6
    dataset = MotJepaClipDataset(
        [path], LAYOUT, strides=(1,), with_conditioning=True, action_horizon=horizon
    )
    sample = dataset[0]
    assert sample.action_chunk.shape == (horizon, ap.ACTION_DIM)
    assert sample.chunk_mask.shape == (horizon, ap.ACTION_DIM)

    live = sample.chunk_mask[0].bool()
    # Consecutive states differ by exactly 1 on live columns, and the pose blocks are excluded
    # from that because they go through relative_pose rather than a first difference.
    plain = live.clone()
    for block in ap.POSE_BLOCKS:
        plain[block] = False
    assert plain.any()
    assert torch.allclose(sample.action_chunk[:, plain], torch.ones_like(sample.action_chunk[:, plain]))
    # Masked-out columns must be exactly zero, not a small number.
    assert torch.equal(sample.action_chunk[:, ~live], torch.zeros_like(sample.action_chunk[:, ~live]))


def test_action_chunk_is_absent_without_a_horizon(tmp_path):
    """Pretraining must keep paying nothing: no horizon, no extra zarr read, zeros in the batch."""
    lengths = [40]
    path = make_store(tmp_path / "nohorizon.zarr", lengths)
    add_conditioning_arrays(path, lengths)

    dataset = MotJepaClipDataset([path], LAYOUT, strides=(1,), with_conditioning=True)
    sample = dataset[0]
    assert torch.equal(sample.action_chunk, torch.zeros_like(sample.action_chunk))
    assert torch.equal(sample.chunk_mask, torch.zeros_like(sample.chunk_mask))
