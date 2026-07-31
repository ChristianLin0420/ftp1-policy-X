import json

import numpy as np
import zarr

from scripts import ftp1_preflight


def _write_valid_zarr(path):
    root = zarr.open_group(store=str(path), mode="w", zarr_format=2)
    data = root.create_group("data")
    meta = root.create_group("meta")
    meta.create_array("episode_ends", data=np.array([2, 5], dtype=np.int64))
    data.create_array("timestamps", data=np.arange(5, dtype=np.int64))
    data.create_array("sub_task_instruction", data=np.array(["lift"] * 5, dtype="U8"))
    data.create_array("camera_ego_rgb", data=np.zeros((5, 8, 8, 3), dtype=np.uint8))
    data.create_array("right_arm_joints", data=np.zeros((5, 7), dtype=np.float32))
    data.create_array("right_hand_joints", data=np.zeros((5, 1), dtype=np.float32))
    data.create_array("right_hand_joints_idx", data=np.full((5, 1), 28, dtype=np.int32))
    data.create_array("right_tactile_data_gripper", data=np.zeros((5, 2, 8, 8, 3), dtype=np.uint8))
    data.create_array("right_tactile_area_gripper", data=np.tile(np.array([[0, 1]]), (5, 1)))
    data.create_array("right_tactile_sensor_gripper", data=np.array(["GelSightMini"] * 5, dtype="U16"))
    data.create_array("right_tactile_type_gripper", data=np.array(["image"] * 5, dtype="U8"))


def test_validate_dataset_config_accepts_ftp1_zarr(tmp_path):
    dataset_dir = tmp_path / "lift_bottle"
    dataset_dir.mkdir()
    _write_valid_zarr(dataset_dir / "episodes.zarr")
    config_path = tmp_path / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "UniVTAC_lift_bottle", "path": str(dataset_dir), "enabled": True}
                ]
            }
        )
    )

    findings, summaries = ftp1_preflight.validate_dataset_config(config_path)

    assert not [finding for finding in findings if finding.severity == "error"]
    assert summaries[0]["episodes"] == 2
    assert summaries[0]["steps"] == 5


def test_validate_zarr_reports_missing_tactile_companion(tmp_path):
    zarr_path = tmp_path / "episodes.zarr"
    _write_valid_zarr(zarr_path)
    root = zarr.open_group(store=str(zarr_path), mode="a")
    del root["data"]["right_tactile_sensor_gripper"]

    findings, _ = ftp1_preflight.validate_zarr(zarr_path)

    assert any("missing tactile sensor" in finding.message for finding in findings)


def test_validate_checkpoint_requires_tactile_assets_and_domain(tmp_path):
    checkpoint = tmp_path / "19999"
    checkpoint.mkdir()
    for filename in ("model.safetensors", "train_config.json"):
        (checkpoint / filename).write_text("{}")
    (checkpoint / "model_config.json").write_text(json.dumps({"use_tactile_input": True}))
    (checkpoint / "normalization").mkdir()

    findings, _ = ftp1_preflight.validate_checkpoint(checkpoint, "UniVTAC_lift_bottle")

    messages = {finding.message for finding in findings}
    assert "tactile checkpoint is missing hpt_tokenizer/" in messages
    assert "missing normalization domain 'UniVTAC_lift_bottle'" in messages
