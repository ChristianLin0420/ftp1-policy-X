from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from data_processing.parse_data_module import parse_data_univtac
from data_processing.parse_data_module.parse_data_univtac import _infer_control_phase
from data_processing.parse_data_module.parse_data_univtac import _source_episode_seed_array


def test_infer_control_phase_follows_settle_close_lift_release() -> None:
    gripper = np.array([0.020, 0.020, 0.018, 0.014, 0.010, 0.010, 0.010, 0.014, 0.019])
    phase = _infer_control_phase(gripper)

    np.testing.assert_array_equal(phase, np.array([0, 0, 1, 1, 1, 1, 2, 3, 3]))


def test_infer_control_phase_handles_empty_and_constant_sequences() -> None:
    assert _infer_control_phase(np.array([], dtype=np.float32)).shape == (0,)
    np.testing.assert_array_equal(_infer_control_phase(np.ones(4)), np.array([0, 2, 2, 2]))


def test_v3_episode_preserves_authoritative_command_phase_contact_and_control_step(tmp_path, monkeypatch) -> None:
    frames = 6
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as file:
        embodiment = file.create_group("embodiment")
        embodiment.create_dataset("joint", data=np.zeros((frames, 9), dtype=np.float32))
        embodiment.create_dataset("command", data=np.ones((frames, 8), dtype=np.float32))
        file.create_dataset("step", data=np.arange(10, 10 + frames, dtype=np.int64))
        control = file.create_group("control")
        control.create_dataset("phase", data=np.array([0, 0, 1, 2, 3, 3], dtype=np.int64))
        control.create_dataset("contact", data=np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8))
        observation = file.create_group("observation").create_group("head")
        observation.create_dataset("rgb", data=np.zeros(frames, dtype=np.uint8))
        tactile = file.create_group("tactile")
        tactile.create_group("left_tactile").create_dataset("rgb_marker", data=np.zeros(frames, dtype=np.uint8))
        tactile.create_group("right_tactile").create_dataset("rgb_marker", data=np.zeros(frames, dtype=np.uint8))

    monkeypatch.setattr(
        parse_data_univtac,
        "_stream_to_img",
        lambda raw, out_size: np.zeros((len(raw), *out_size, 3), dtype=np.uint8),
    )
    episode = parse_data_univtac._load_univtac_episode(path, image_size=4, use_wrist=False)  # noqa: SLF001
    assert episode is not None
    assert episode["command_valid"].tolist() == [1] * (frames - 1)
    assert episode["contact_valid"].tolist() == [1] * (frames - 1)
    assert episode["control_step_valid"].tolist() == [1] * (frames - 1)
    assert episode["control_step"].tolist() == [10, 11, 12, 13, 14]
    assert episode["phase_id"].tolist() == [0, 0, 1, 2, 3]


def test_source_episode_seeds_preserve_numeric_filename_identity_in_parser_order() -> None:
    seeds = _source_episode_seed_array([(Path("3.hdf5"), True), (Path("00011.hdf5"), True)])

    assert seeds is not None
    assert seeds.dtype == np.int64
    np.testing.assert_array_equal(seeds, np.array([3, 11], dtype=np.int64))


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([(Path("episode.hdf5"), True)], "numeric int64 stems"),
        ([(Path("1.hdf5"), True), (Path("0001.hdf5"), True)], "duplicate episode seeds"),
    ],
)
def test_authoritative_v3_source_episode_seed_rejects_nonnumeric_or_duplicate_stems(
    records: list[tuple[Path, bool]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _source_episode_seed_array(records)


def test_legacy_descriptive_filenames_remain_supported_without_source_seed_metadata() -> None:
    assert _source_episode_seed_array([(Path("episode_a.hdf5"), False)]) is None


def test_run_writes_source_episode_seed_as_episode_metadata(tmp_path, monkeypatch) -> None:
    hdf5_dir = tmp_path / "input" / "lift_bottle" / "demo" / "hdf5"
    hdf5_dir.mkdir(parents=True)
    for filename in ("00011.hdf5", "3.hdf5"):
        (hdf5_dir / filename).touch()

    def authoritative_episode(*_args, **_kwargs) -> dict[str, np.ndarray]:
        rows = 2
        return {
            "timestamps": np.arange(rows, dtype=np.int64),
            "command_valid": np.ones(rows, dtype=np.uint8),
            "contact_valid": np.ones(rows, dtype=np.uint8),
            "control_step_valid": np.ones(rows, dtype=np.uint8),
        }

    monkeypatch.setattr(parse_data_univtac, "_load_univtac_episode", authoritative_episode)
    output = tmp_path / "output"
    parse_data_univtac._run(  # noqa: SLF001
        str(tmp_path / "input"),
        str(output),
        task_list=["lift_bottle"],
    )

    store = zarr.open_group(output / "lift_bottle_head.zarr", mode="r")
    np.testing.assert_array_equal(store["meta/source_episode_seed"][:], np.array([3, 11], dtype=np.int64))
