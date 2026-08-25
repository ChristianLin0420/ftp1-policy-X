import dataclasses
import json

import pytest

from openpi.mot_jepa.control_v2_config import ControlV2Artifact
from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.runtime import architecture_from_run


def test_train_config_json_round_trip_and_high_resolution_layout() -> None:
    config = ControlV2TrainConfig(
        pretrained_run="/tmp/backbone",
        store_glob="/tmp/*.zarr",
        safety_tube_loss_weight=0.075,
    )
    restored = ControlV2TrainConfig.from_json(config.to_json())
    assert restored == config
    assert restored.safety_tube_loss_weight == 0.075
    assert restored.layout.video_size == 224
    assert restored.layout.gel_size == 112
    assert restored.layout.video_width == 384


def test_source_architecture_resolution_does_not_replace_control_sampling(tmp_path) -> None:
    config = ControlV2TrainConfig(pretrained_run=str(tmp_path), store_glob="/tmp/*.zarr")
    checkpoint = tmp_path / "checkpoints" / "100000"
    checkpoint.mkdir(parents=True)
    payload = json.loads(config.to_json())
    payload["data"].update({"strides": [2, 4], "action_stride": None, "index_step": 1})
    (checkpoint / "train_config.json").write_text(json.dumps(payload))

    resolved = architecture_from_run(tmp_path, config, 100000)
    assert resolved.data.strides == (2, 4)
    assert resolved.observation_stride == 2
    assert resolved.action_stride == 1
    assert resolved.index_step == 1


def _artifact() -> ControlV2Artifact:
    return ControlV2Artifact(
        schema_version=4,
        data_schema="mot_jepa_control_v2_data_v2",
        task="lift_bottle",
        context_mode="dense_video_gel",
        state_mode="qpos8_history_command",
        target_mode="next_command_chunk_relative_mix8",
        horizon=32,
        first_executable_index=1,
        chunk_first_n=20,
        temporal_ensemble_k=0.01,
        observation_stride=2,
        action_stride=1,
        control_step_stride=1,
        phase_names=("approach", "close", "lift", "release"),
        action_scale=(1.0,) * 8,
        joint_lower=(-3.0,) * 8,
        joint_upper=(3.0,) * 8,
        max_command_delta=(0.1,) * 8,
        authoritative_control=True,
        total_episodes=1000,
        train_episodes=850,
        validation_episodes=150,
        phase_counts=(100, 100, 100, 100),
        contact_counts=(200, 200),
        source_stores=("/tmp/data.zarr",),
        source_store_sha256="a" * 64,
    )


def test_artifact_round_trip_and_contract_rejection() -> None:
    artifact = _artifact()
    assert ControlV2Artifact.from_json(artifact.to_json()) == artifact
    with pytest.raises(ValueError, match="horizon"):
        dataclasses.replace(artifact, horizon=16)
    assert artifact.production_data_qualified
    with pytest.raises(ValueError, match="exactly 1000"):
        dataclasses.replace(artifact, authoritative_control=False).require_production_data()
    overfull = dataclasses.replace(artifact, total_episodes=1001, train_episodes=851)
    assert not overfull.production_data_qualified
    with pytest.raises(ValueError, match="exactly 1000"):
        overfull.require_production_data()


def test_authoritative_train_config_requires_exact_production_episode_count() -> None:
    with pytest.raises(ValueError, match="exactly 1000"):
        ControlV2TrainConfig(minimum_total_episodes=1001)

    nonproduction = ControlV2TrainConfig(require_authoritative_control=False, minimum_total_episodes=1)
    assert nonproduction.minimum_total_episodes == 1


def test_train_config_rejects_negative_safety_tube_loss_weight() -> None:
    with pytest.raises(ValueError, match="auxiliary loss weights must be non-negative"):
        ControlV2TrainConfig(safety_tube_loss_weight=-0.01)
