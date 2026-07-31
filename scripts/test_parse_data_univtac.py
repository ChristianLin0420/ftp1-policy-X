from pathlib import Path

from data_processing.parse_data_module.parse_data_univtac import _discover_task_dirs


def test_discover_task_dirs_accepts_released_clean_layout(tmp_path: Path):
    hdf5_dir = tmp_path / "lift_bottle" / "clean"
    hdf5_dir.mkdir(parents=True)
    (hdf5_dir / "0.hdf5").touch()

    discovered = _discover_task_dirs(tmp_path, ["lift_bottle"])

    assert discovered == [("lift_bottle", tmp_path / "lift_bottle", hdf5_dir)]


def test_discover_task_dirs_prefers_legacy_layout(tmp_path: Path):
    legacy = tmp_path / "lift_bottle" / "demo" / "hdf5"
    released = tmp_path / "lift_bottle" / "clean"
    legacy.mkdir(parents=True)
    released.mkdir(parents=True)
    (legacy / "0.hdf5").touch()
    (released / "0.hdf5").touch()

    discovered = _discover_task_dirs(tmp_path, None)

    assert discovered[0][2] == legacy
