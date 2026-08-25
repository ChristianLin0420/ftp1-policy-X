from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import runpy
import shutil
import types

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from data_processing.parse_data_module import parse_data_univtac
from openpi.mot_jepa.control_v2 import ControlV2
from openpi.mot_jepa.control_v2 import ControlV2Config
from openpi.mot_jepa.control_v2_config import ControlV2Artifact
from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.control_v2_data import ControlV2Normalizer
from openpi.mot_jepa.control_v2_data import fit_control_v2_normalization
from openpi.mot_jepa.control_v2_deploy import MotJepaControlV2Policy
from openpi.mot_jepa.control_v2_deploy import TrainedControlV2Artifacts
from openpi.mot_jepa.control_v2_deploy import load_trained_control_v2_artifacts
from openpi.mot_jepa.model import MotJepaBackbone
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: pathlib.Path) -> ControlV2TrainConfig:
    return ControlV2TrainConfig(
        exp_name="unit",
        run_root=str(tmp_path),
        pretrained_run=str(tmp_path / "source"),
        store_glob=str(tmp_path / "no-stores" / "*.zarr"),
        encoder=MoTEncoderConfig(depth=1, num_local_layers=0, num_heads=1),
        head=ControlV2Config(width=16, depth=1, num_heads=2, fourier_dim=2),
        backbone_last_n_blocks=1,
        num_train_steps=1,
        save_interval=1,
        warmup_steps=0,
        wandb_enabled=False,
    )


def _artifact(*, action_scale: tuple[float, ...] = (1.0,) * 8) -> ControlV2Artifact:
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
        action_scale=action_scale,
        joint_lower=(-3.0,) * 8,
        joint_upper=(3.0,) * 8,
        max_command_delta=(0.2,) * 8,
        authoritative_control=True,
        total_episodes=1_000,
        train_episodes=800,
        validation_episodes=200,
        phase_counts=(100, 100, 100, 100),
        contact_counts=(900, 100),
        source_stores=("/unavailable/unit.zarr",),
        source_store_sha256="a" * 64,
    )


def _normalizer() -> ControlV2Normalizer:
    qpos = torch.zeros(2, 16, 8)
    command = torch.zeros_like(qpos)
    present = torch.ones(2, 16, dtype=torch.bool)
    stats = fit_control_v2_normalization(
        qpos,
        previous_command=command,
        previous_command_present=present,
        dt=2.0,
    )
    return ControlV2Normalizer(stats)


def _write_completed_run(tmp_path: pathlib.Path) -> pathlib.Path:
    config = _config(tmp_path)
    run = config.run_dir
    checkpoint = run / "checkpoints" / "1"
    checkpoint.mkdir(parents=True)
    config_text = config.to_json()
    (run / "run_config.json").write_text(config_text)
    artifact = _artifact()
    (run / "artifact.json").write_text(artifact.to_json())
    normalizer = _normalizer()
    (run / "normalization.json").write_text(
        json.dumps(normalizer.to_stats().to_artifact(), indent=2, sort_keys=True) + "\n"
    )
    (run / "DONE").write_text("1\n")
    (run / "checkpoints" / "latest").write_text("1\n")

    backbone = MotJepaBackbone(
        config.layout,
        config.encoder,
        lowdim_channels=config.data.lowdim_channels,
        lowdim_log_compress=config.data.lowdim_log_compress,
    )
    head = ControlV2(config.head, config.layout)
    head.load_action_scale(torch.ones(8))
    torch.save(head.state_dict(), checkpoint / "student.pt")
    torch.save(backbone.state_dict(), checkpoint / "backbone.pt")
    torch.save(normalizer.state_dict(), checkpoint / "loss.pt")
    torch.save({}, checkpoint / "optimizer.pt")
    torch.save(
        {
            "global_step": 1,
            "control_v2_checkpoint_schema": 4,
            "backbone_train_mode": config.backbone_train_mode,
            "backbone_last_n_blocks": config.backbone_last_n_blocks,
            "source_backbone_run": str(pathlib.Path(config.pretrained_run).resolve()),
            "source_backbone_step": config.pretrained_step,
            "artifact_sha256": _sha256(run / "artifact.json"),
            "normalization_sha256": _sha256(run / "normalization.json"),
            "source_store_sha256": artifact.source_store_sha256,
            "normalization_counts": {
                "qpos": normalizer.qpos_count,
                "velocity": normalizer.velocity_count,
                "previous_command": normalizer.previous_command_count,
            },
            "best_validation_step": 1,
            "best_validation_chunk_loss": 0.125,
        },
        checkpoint / "metadata.pt",
    )
    (checkpoint / "train_config.json").write_text(config_text)
    selected = run / "checkpoints" / "best_validation_1"
    shutil.copytree(checkpoint, selected)
    identities = []
    for index in range(64):
        scenario = index % 32
        identities.append(
            f"/unavailable/unit.zarr\0episode={index // 32}\0start={index}\0stride=2"
            f"\0dataset_index={index}\0time_bin={('early', 'middle', 'late')[index % 3]}"
            f"\0phase={index % 4}\0contact={index % 2}\0history_length={scenario // 2 + 1}"
            f"\0oldest_command_present={scenario % 2}"
        )
    sample_sha256 = hashlib.sha256(
        json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    (run / "VALIDATION_SAMPLES.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sampling": "all_episode_time_phase_contact_cold_start_stratified_v2",
                "requested_examples": 64,
                "sample_count": 64,
                "episode_count": 2,
                "sample_sha256": sample_sha256,
                "episodes": [
                    {"identity": "/unavailable/unit.zarr\0episode=0", "sample_count": 32},
                    {"identity": "/unavailable/unit.zarr\0episode=1", "sample_count": 32},
                ],
                "samples": [
                    {"dataset_index": index, "identity": identity} for index, identity in enumerate(identities)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (run / "STATS_DONE").write_text(
        json.dumps(
            {
                "schema": 4,
                "split_identity": "source_episode_seed_v1",
                "train_episodes": artifact.train_episodes,
                "validation_episodes": artifact.validation_episodes,
                "source_store_sha256": artifact.source_store_sha256,
                "validation_sampling": "all_episode_time_phase_contact_cold_start_stratified_v2",
                "validation_sample_sha256": sample_sha256,
                "validation_examples": 64,
                "validation_episode_count": 2,
                "validation_oracle_safety_violation_values": 0,
                "validation_oracle_min_slack": 0.01,
                "validation_oracle_max_rate_ratio": 0.9,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (run / "BEST_VALIDATION.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "selection_metric": "validation_action_loss",
                "selection_constraint": "validation_safety_violation_values==0",
                "selection_mode": "min",
                "tie_break": "earliest_step",
                "step": 1,
                "checkpoint": str(selected.resolve()),
                "artifact_sha256": _sha256(run / "artifact.json"),
                "normalization_sha256": _sha256(run / "normalization.json"),
                "source_store_sha256": artifact.source_store_sha256,
                "validation_action_loss": 0.125,
                "validation_action_mae": 0.25,
                "validation_phase_loss": 0.5,
                "validation_contact_loss": 0.5,
                "validation_contact_count": 32.0,
                "validation_clean_order_loss": 0.5,
                "validation_overlap_loss": 0.1,
                "validation_safety_loss": 0.01,
                "validation_safety_tube_loss": 0.02,
                "validation_safety_tube_mean_loss": 0.005,
                "validation_safety_tube_tail_loss": 0.015,
                "validation_safety_tube_max_ratio": 0.8,
                "validation_safety_tube_min_slack": 0.01,
                "validation_joint_violation_values": 0.0,
                "validation_rate_violation_values": 0.0,
                "validation_safety_violation_values": 0.0,
                "validation_joint_max_excess_ratio": 0.0,
                "validation_rate_max_ratio": 0.95,
                "validation_examples": 64.0,
                "validation_episode_count": 2.0,
                "validation_objective": 0.5,
                "validation_sampling": "all_episode_time_phase_contact_cold_start_stratified_v2",
                "validation_sample_sha256": sample_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return run


def test_strict_loader_restores_exact_closed_artifacts(tmp_path: pathlib.Path) -> None:
    run = _write_completed_run(tmp_path)

    loaded = load_trained_control_v2_artifacts(run, "cpu", verify_source_stores=False)

    assert loaded.step == 1
    assert loaded.completed_step == 1
    assert loaded.selection_metric == "validation_action_loss"
    assert loaded.selection_value == pytest.approx(0.125)
    assert loaded.config.task == "lift_bottle"
    assert not loaded.head.training
    assert not loaded.backbone.training
    assert torch.equal(loaded.head.action_scale, torch.ones(8))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection_constraint", "none", "selection_constraint"),
        ("validation_joint_violation_values", 1.0, "zero held-out safety constraint"),
        ("validation_rate_violation_values", 1.0, "zero held-out safety constraint"),
        ("validation_safety_violation_values", 1.0, "zero held-out safety constraint"),
        ("validation_joint_max_excess_ratio", 0.01, "joint_max_excess_ratio must be zero"),
        ("validation_rate_max_ratio", 1.0001, "rate_max_ratio must be at most 1"),
    ],
)
def test_strict_loader_rejects_unsafe_heldout_selection(
    tmp_path: pathlib.Path,
    field: str,
    value: str | float,
    message: str,
) -> None:
    run = _write_completed_run(tmp_path)
    marker_path = run / "BEST_VALIDATION.json"
    marker = json.loads(marker_path.read_text())
    marker[field] = value
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=message):
        load_trained_control_v2_artifacts(run, "cpu", verify_source_stores=False)


def test_strict_loader_accepts_initialization_as_heldout_winner(tmp_path: pathlib.Path) -> None:
    run = _write_completed_run(tmp_path)
    selected_one = run / "checkpoints" / "best_validation_1"
    selected_zero = run / "checkpoints" / "best_validation_0"
    selected_one.rename(selected_zero)
    metadata_path = selected_zero / "metadata.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    metadata["global_step"] = 0
    metadata["best_validation_step"] = 0
    torch.save(metadata, metadata_path)
    marker_path = run / "BEST_VALIDATION.json"
    marker = json.loads(marker_path.read_text())
    marker["step"] = 0
    marker["checkpoint"] = str(selected_zero.resolve())
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    loaded = load_trained_control_v2_artifacts(run, "cpu", verify_source_stores=False)

    assert loaded.step == 0
    assert loaded.completed_step == 1


def test_strict_loader_rejects_artifact_changed_after_checkpoint(tmp_path: pathlib.Path) -> None:
    run = _write_completed_run(tmp_path)
    with (run / "artifact.json").open("a") as stream:
        stream.write(" \n")

    with pytest.raises(ValueError, match="artifact_sha256"):
        load_trained_control_v2_artifacts(run, "cpu", verify_source_stores=False)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("\0time_bin=late", "\0time_bin=early"),
        ("\0phase=3", "\0phase=0"),
        ("\0contact=1", "\0contact=0"),
    ],
)
def test_strict_loader_rejects_validation_manifest_missing_required_stratum(
    tmp_path: pathlib.Path,
    old: str,
    new: str,
) -> None:
    run = _write_completed_run(tmp_path)
    manifest_path = run / "VALIDATION_SAMPLES.json"
    manifest = json.loads(manifest_path.read_text())
    for sample in manifest["samples"]:
        sample["identity"] = sample["identity"].replace(old, new)
    identities = [sample["identity"] for sample in manifest["samples"]]
    manifest["sample_sha256"] = hashlib.sha256(
        json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="required validation strata"):
        load_trained_control_v2_artifacts(run, "cpu", verify_source_stores=False)


class _FakeBackbone(nn.Module):
    def forward(self, *args, **kwargs):  # pragma: no cover - encode_full is the deploy interface
        raise AssertionError

    def encode_full(self, inputs):
        return types.SimpleNamespace(inputs=inputs)


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.features: list[torch.Tensor] = []

    def forward(self, encoded, state_features, current_qpos):
        del encoded
        self.features.append(state_features.detach().cpu())
        chunk = torch.zeros(current_qpos.shape[0], 32, 8, device=current_qpos.device)
        chunk[..., 7] = current_qpos[:, None, 7]
        base = 0.01 if self.calls == 0 else 0.05
        chunk[:, 1:, :7] = base
        if self.calls == 0:
            chunk[:, 2:, :7] = 0.02
        self.calls += 1
        return types.SimpleNamespace(action_chunk=chunk)


def _observation(qpos: np.ndarray, layout) -> dict:
    return {
        "observation": {"head": {"rgb": np.zeros((layout.video_size, layout.video_size, 3), dtype=np.uint8)}},
        "tactile": {
            "left_tactile": {"rgb_marker": np.zeros((layout.gel_size, layout.gel_size, 3), dtype=np.uint8)},
            "right_tactile": {"rgb_marker": np.zeros((layout.gel_size, layout.gel_size, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": qpos},
    }


def _fake_artifacts(tmp_path: pathlib.Path) -> TrainedControlV2Artifacts:
    config = _config(tmp_path)
    return TrainedControlV2Artifacts(
        config=config,
        artifact=_artifact(),
        backbone=_FakeBackbone(),
        head=_FakeHead(),
        normalizer=_normalizer(),
        step=1,
        completed_step=1,
        selection_metric="validation_action_loss",
        selection_value=0.125,
        checkpoint=tmp_path,
        metadata={"artifact_sha256": "b" * 64},
    )


def test_policy_uses_cold_start_masks_previous_command_and_exact_ensemble(tmp_path: pathlib.Path) -> None:
    artifacts = _fake_artifacts(tmp_path)
    policy = MotJepaControlV2Policy(artifacts, device="cpu")
    policy.reset(seed=7)
    qpos = np.zeros(8, dtype=np.float32)

    first = policy.act(_observation(qpos, policy.layout))
    np.testing.assert_allclose(first[:7], 0.01)
    first_presence = artifacts.head.features[0][0, :, -3:]
    assert first_presence[:-1].eq(0).all()
    assert torch.equal(first_presence[-1], torch.tensor([1.0, 0.0, 0.0]))

    second_qpos = first.copy()
    second = policy.act(_observation(second_qpos, policy.layout))
    weights = np.exp(-0.01 * np.arange(2, dtype=np.float32))
    weights /= weights.sum()
    expected = 0.02 * weights[0] + 0.06 * weights[1]
    np.testing.assert_allclose(second[:7], expected, rtol=1e-6)
    assert artifacts.head.features[1][0, -1, -1] == 1  # previous command is present
    assert policy.get_last_action_debug()["ensemble_candidates"] == 2
    assert policy.get_safety_counters()["joint_clamp_commands"] == 0
    assert policy.first_executable_index == 1
    assert policy.action_dim == 8
    assert policy._chunk_history[-1][0].shape == (32, 8)  # noqa: SLF001


def test_policy_preprocessing_matches_offline_two_stage_resize(tmp_path: pathlib.Path) -> None:
    policy = MotJepaControlV2Policy(_fake_artifacts(tmp_path), device="cpu")
    rng = np.random.default_rng(17)
    raw = rng.integers(0, 256, size=(197, 301, 3), dtype=np.uint8)

    writer_path = pathlib.Path(__file__).parents[3] / "UniVTAC/envs/utils/data.py"
    writer = runpy.run_path(str(writer_path))["HDF5Handler"]
    streams, _max_len = writer.img_to_stream(np.stack([raw]))
    # The parser's BGR->RGB conversion and its caller's final reversal intentionally cancel.
    offline_224 = parse_data_univtac._stream_to_img(  # noqa: SLF001
        np.asarray(streams, dtype=object),
        (policy.config.video_size, policy.config.video_size),
    )[0][..., ::-1]
    offline_gel = cv2.resize(
        offline_224,
        (policy.layout.gel_size, policy.layout.gel_size),
        interpolation=cv2.INTER_AREA,
    )

    np.testing.assert_array_equal(policy._prepare_video(raw, name="test video"), offline_224)  # noqa: SLF001
    np.testing.assert_array_equal(policy._prepare_gel(raw, name="test gel"), offline_gel)  # noqa: SLF001
    one_hop = cv2.resize(raw, (policy.layout.gel_size, policy.layout.gel_size), interpolation=cv2.INTER_AREA)
    assert not np.array_equal(offline_gel, one_hop)


def test_policy_nonfinite_model_output_holds_and_records_fallback(tmp_path: pathlib.Path) -> None:
    artifacts = _fake_artifacts(tmp_path)

    class NonfiniteHead(_FakeHead):
        def forward(self, encoded, state_features, current_qpos):
            output = super().forward(encoded, state_features, current_qpos)
            output.action_chunk[:, 4, 2] = torch.nan
            return output

    artifacts = dataclasses.replace(artifacts, head=NonfiniteHead())
    policy = MotJepaControlV2Policy(artifacts, device="cpu")
    qpos = np.zeros(8, dtype=np.float32)

    action = policy.act(_observation(qpos, policy.layout))

    np.testing.assert_array_equal(action, qpos)
    assert policy.get_safety_counters()["nonfinite_fallbacks"] == 1
    assert policy.get_last_action_debug()["used_fallback"]
    assert np.isfinite(policy._chunk_history[-1][0]).all()  # noqa: SLF001
