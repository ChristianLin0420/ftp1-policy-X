from __future__ import annotations

import numpy as np
import pytest
import zarr

from scripts import mot_jepa_build_clip_zarr as builder


def _write_array(group: zarr.Group, name: str, value: np.ndarray) -> None:
    array = group.create_array(name, shape=value.shape, dtype=value.dtype)
    array[:] = value


def _source_store(path, *, seed_values: np.ndarray) -> str:
    root = zarr.open_group(path, mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    _write_array(meta, "episode_ends", np.array([1, 3], dtype=np.int64))
    _write_array(meta, builder.SOURCE_EPISODE_SEED_KEY, seed_values)
    video = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
    _write_array(data, "camera_ego_rgb", video)
    return str(path)


def test_build_store_copies_source_episode_seed_values_and_dtype_unchanged(tmp_path) -> None:
    source = _source_store(tmp_path / "source.zarr", seed_values=np.array([2_000_003, 2_000_011], dtype=np.int64))
    destination = tmp_path / "derived.zarr"

    result = builder.build_store(
        source,
        destination,
        video_size=4,
        gel_size=2,
        num_frames=2,
        max_gel_pads=2,
        batch_frames=2,
    )

    copied = zarr.open_group(destination, mode="r")["meta"][builder.SOURCE_EPISODE_SEED_KEY]
    assert copied.dtype == np.dtype("int64")
    np.testing.assert_array_equal(copied[:], np.array([2_000_003, 2_000_011], dtype=np.int64))
    assert result["episode_meta_arrays"] == [builder.SOURCE_EPISODE_SEED_KEY]


def test_build_store_rejects_source_episode_seed_count_mismatch(tmp_path) -> None:
    source = _source_store(tmp_path / "source.zarr", seed_values=np.array([2_000_003], dtype=np.int64))

    with pytest.raises(ValueError, match="one value per episode"):
        builder.build_store(
            source,
            tmp_path / "derived.zarr",
            video_size=4,
            gel_size=2,
            num_frames=2,
            max_gel_pads=2,
        )
