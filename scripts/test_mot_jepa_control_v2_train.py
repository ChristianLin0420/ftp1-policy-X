from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.mot_jepa.control_v2 import ControlV2
from openpi.mot_jepa.control_v2 import ControlV2Config
from openpi.mot_jepa.control_v2_config import ControlV2TrainConfig
from openpi.mot_jepa.control_v2_data import ControlV2Normalizer
from openpi.mot_jepa.control_v2_data import fit_control_v2_normalization
from openpi.mot_jepa.control_v2_data import scatter_qpos8
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.model import ClipInputs
from openpi.mot_jepa.mot_encoder import EncoderOutput
from scripts import mot_jepa_control_v2_train as train_script
from scripts.mot_jepa_control_v2_train import ArtifactBundle
from scripts.mot_jepa_control_v2_train import ControlV2TrainModel
from scripts.mot_jepa_control_v2_train import _apply_validation_history_scenarios
from scripts.mot_jepa_control_v2_train import _gather_time
from scripts.mot_jepa_control_v2_train import _history_indices
from scripts.mot_jepa_control_v2_train import _order_indices
from scripts.mot_jepa_control_v2_train import _prefix_command_presence
from scripts.mot_jepa_control_v2_train import _stage_initialization
from scripts.mot_jepa_control_v2_train import build_validation_selection
from scripts.mot_jepa_control_v2_train import control_safety_terms
from scripts.mot_jepa_control_v2_train import control_safety_tube_terms
from scripts.mot_jepa_control_v2_train import file_sha256
from scripts.mot_jepa_control_v2_train import freeze_validation_selection
from scripts.mot_jepa_control_v2_train import load_and_validate_artifacts
from scripts.mot_jepa_control_v2_train import load_validation_manifest
from scripts.mot_jepa_control_v2_train import snapshot_best_validation

TINY_LAYOUT = TokenLayout(
    num_frames=4,
    tubelet_t=2,
    video_size=16,
    video_patch=16,
    gel_size=16,
    gel_patch=16,
    num_gel_pads=1,
    lowdim_slots=1,
    lowdim_channels=2,
    video_width=8,
    tactile_width=8,
)


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video = nn.Linear(3, TINY_LAYOUT.video_width)
        self.tactile = nn.Linear(3, TINY_LAYOUT.tactile_width)

    def encode_full(self, inputs: ClipInputs) -> EncoderOutput:
        video = self.video(inputs.video.mean(dim=(1, 3, 4)))
        gel = self.tactile(inputs.gel.mean(dim=(1, 2, 4, 5)))
        video = video[:, None].expand(-1, TINY_LAYOUT.num_video_tokens, -1)
        tactile = gel[:, None].expand(-1, TINY_LAYOUT.num_gel_tokens + TINY_LAYOUT.num_lowdim_tokens, -1)
        readout = [
            video[:, : TINY_LAYOUT.num_steps],
            tactile[:, : TINY_LAYOUT.num_steps],
        ]
        return EncoderOutput(tokens=[video, tactile], sync_readout=readout, final_readout=readout)


def _validation_dataset(*, episodes: int = 10, clips_per_episode: int = 24) -> SimpleNamespace:
    entries = torch.tensor(
        [[0, episode * 1_000 + clip, 2, episode] for episode in range(episodes) for clip in range(clips_per_episode)],
        dtype=torch.long,
    ).numpy()

    def stratum(dataset_index: int, _store_idx: int, _current: int) -> tuple[int, int]:
        clip = int(entries[dataset_index, 1] % 1_000)
        phase = min(3, clip * 4 // clips_per_episode)
        contact = clip % 2
        return phase, contact

    return SimpleNamespace(
        clip_index=SimpleNamespace(entries=entries),
        store_paths=["/data/lift_bottle.zarr"],
        base=SimpleNamespace(layout=SimpleNamespace(num_frames=16)),
        validation_stratum=stratum,
    )


def test_history_padding_reuses_one_index_for_every_modality() -> None:
    lengths = torch.tensor([1, 3, 4], dtype=torch.long)
    indices, valid = _history_indices(lengths, 4)

    torch.testing.assert_close(
        indices,
        torch.tensor([[3, 3, 3, 3], [1, 1, 2, 3], [0, 1, 2, 3]]),
    )
    torch.testing.assert_close(
        valid,
        torch.tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [True, True, True, True],
            ]
        ),
    )
    values = torch.arange(3 * 4 * 2).reshape(3, 4, 2)
    torch.testing.assert_close(_gather_time(values, indices)[0], values[0, 3:].expand(4, -1))
    command_present = _prefix_command_presence(
        torch.ones_like(valid),
        valid,
        lengths,
        torch.tensor([False, True, False]),
    )
    # Both stride-2 rollout parities are representable: an odd-age prefix has no predecessor for
    # its oldest real frame, while the next even-age prefix of the same length does.
    torch.testing.assert_close(command_present[0], torch.zeros(4, dtype=torch.bool))
    torch.testing.assert_close(command_present[1], torch.tensor([False, True, True, True]))
    torch.testing.assert_close(command_present[2], torch.tensor([False, True, True, True]))


def test_order_corruption_never_moves_left_padding_into_the_valid_suffix() -> None:
    valid = torch.tensor([[False, False, True, True, True, True]] * 3)
    labels = torch.tensor([1, 2, 3], dtype=torch.long)
    generator = torch.Generator().manual_seed(7)

    indices = _order_indices(valid, labels, generator=generator)

    torch.testing.assert_close(indices[:, :2], torch.tensor([[0, 1]] * 3))
    assert bool((indices[:, 2:] >= 2).all())
    assert all(torch.unique(row).numel() == 4 for row in indices[:, 2:])
    torch.testing.assert_close(indices[0, 2:], torch.tensor([5, 4, 3, 2]))
    torch.testing.assert_close(indices[1, 2:], torch.tensor([4, 5, 2, 3]))


def test_validation_selection_covers_every_episode_and_all_available_strata() -> None:
    dataset = _validation_dataset()
    entries = dataset.clip_index.entries

    sparse = build_validation_selection(dataset, 4)
    sparse_rows = entries[list(sparse.indices)]
    assert len(sparse.indices) == 32  # floor covers ten episodes and all 32 cold-start scenarios
    assert sparse.episode_count == 10
    assert sorted(set(sparse_rows[:, 3].tolist())) == list(range(10))
    assert set(sparse.time_bins) == {"early", "middle", "late"}
    assert set(sparse.phase_ids) == {0, 1, 2, 3}
    assert set(sparse.contact_classes) == {0, 1}
    assert {
        (length, int(oldest))
        for length, oldest in zip(
            sparse.history_lengths,
            sparse.oldest_command_present,
            strict=True,
        )
    } == {(length, oldest) for length in range(1, 17) for oldest in (0, 1)}
    assert all("\0dataset_index=" in identity and "\0history_length=" in identity for identity in sparse.identities)

    dense = build_validation_selection(dataset, 23)
    counts = [count for _identity, count in dense.episode_sample_counts]
    assert len(dense.indices) == 32
    assert len(counts) == 10
    assert max(counts) - min(counts) == 1
    first_episode_bins = [
        dense.time_bins[offset] for offset, index in enumerate(dense.indices) if entries[index, 3] == 0
    ]
    assert set(first_episode_bins) == {"early", "middle", "late"}
    assert dense == build_validation_selection(dataset, 23)

    with pytest.raises(ValueError, match="all 32 V3 cold-start scenarios"):
        build_validation_selection(_validation_dataset(episodes=2, clips_per_episode=15), 4)


def test_validation_manifest_rejects_index_and_episode_count_drift(tmp_path: pathlib.Path) -> None:
    selection = build_validation_selection(_validation_dataset(), 23)
    freeze_validation_selection(tmp_path, selection)
    manifest_path = tmp_path / "VALIDATION_SAMPLES.json"
    assert load_validation_manifest(tmp_path)["sample_sha256"] == selection.sample_sha256

    manifest = json.loads(manifest_path.read_text())
    manifest["samples"][0]["dataset_index"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from its immutable identity"):
        load_validation_manifest(tmp_path)

    manifest = selection.to_manifest()
    manifest["episodes"][0]["sample_count"] += 1
    manifest["episodes"][1]["sample_count"] -= 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="episode counts differ"):
        load_validation_manifest(tmp_path)

    manifest = selection.to_manifest()
    manifest["samples"][-1]["identity"] = manifest["samples"][-1]["identity"].replace(
        "oldest_command_present=1",
        "oldest_command_present=0",
    )
    manifest["sample_sha256"] = hashlib.sha256(
        json.dumps(
            [item["identity"] for item in manifest["samples"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not cover the fixed V3 cold-start scenarios"):
        load_validation_manifest(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("time_bin=late", "time_bin=early"),
        ("phase=3", "phase=2"),
        ("contact=1", "contact=0"),
    ],
)
def test_validation_manifest_rejects_missing_advertised_strata(
    tmp_path: pathlib.Path,
    old: str,
    new: str,
) -> None:
    selection = build_validation_selection(_validation_dataset(), 64)
    manifest = selection.to_manifest()
    for sample in manifest["samples"]:
        sample["identity"] = sample["identity"].replace(old, new)
    manifest["sample_sha256"] = hashlib.sha256(
        json.dumps(
            [item["identity"] for item in manifest["samples"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (tmp_path / "VALIDATION_SAMPLES.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not cover required validation strata"):
        load_validation_manifest(tmp_path)


def test_artifact_preflight_requires_frozen_validation_manifest(tmp_path) -> None:
    cfg = ControlV2TrainConfig(
        exp_name="missing_validation_manifest",
        run_root=str(tmp_path),
        pretrained_run=str(tmp_path / "pretrained"),
        store_glob=str(tmp_path / "*.zarr"),
    )
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "run_config.json").write_text(cfg.to_json())
    (cfg.run_dir / "artifact.json").write_text("{}")
    (cfg.run_dir / "normalization.json").write_text("{}")
    (cfg.run_dir / "STATS_DONE").write_text("{}")

    with pytest.raises(FileNotFoundError, match=r"VALIDATION_SAMPLES\.json"):
        load_and_validate_artifacts(cfg, validate_store_content=False)


def test_fixed_validation_history_scenarios_match_online_cold_start() -> None:
    batch, history = 2, 4
    inputs = ClipInputs(
        video=torch.arange(batch * history).reshape(batch, history, 1, 1, 1),
        gel=torch.arange(batch * history).reshape(batch, history, 1, 1, 1, 1),
        lowdim=torch.arange(batch * history).reshape(batch, history, 1, 1),
    )
    state = torch.arange(batch * history * 2).reshape(batch, history, 2)
    command = state + 100
    present = torch.ones(batch, history, dtype=torch.bool)

    output = _apply_validation_history_scenarios(
        inputs,
        state,
        command,
        present,
        torch.tensor([1, 3], dtype=torch.long),
        torch.tensor([False, True]),
    )
    selected_inputs, selected_state, selected_command, selected_present, history_valid = output

    torch.testing.assert_close(selected_inputs.video[0], inputs.video[0, -1:].expand_as(inputs.video[0]))
    torch.testing.assert_close(selected_state[0], state[0, -1:].expand_as(state[0]))
    torch.testing.assert_close(selected_command[0], command[0, -1:].expand_as(command[0]))
    torch.testing.assert_close(
        history_valid,
        torch.tensor([[False, False, False, True], [False, True, True, True]]),
    )
    torch.testing.assert_close(
        selected_present,
        torch.tensor([[False, False, False, False], [False, True, True, True]]),
    )


def test_source_backbone_contract_rejects_byte_drift(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    checkpoint = source / "checkpoints/100000"
    checkpoint.mkdir(parents=True)
    torch.save({"global_step": 100_000}, checkpoint / "metadata.pt")
    (checkpoint / "teacher_ema.pt").write_bytes(b"teacher")
    (checkpoint / "train_config.json").write_text("{}")
    (source / "checkpoints/latest").write_text("100000\n")
    relative_paths = (
        "checkpoints/100000/metadata.pt",
        "checkpoints/100000/teacher_ema.pt",
        "checkpoints/100000/train_config.json",
        "checkpoints/latest",
    )
    qualified = {
        relative: ((source / relative).stat().st_size, file_sha256(source / relative)) for relative in relative_paths
    }
    monkeypatch.setattr(train_script, "_QUALIFIED_BACKBONE_FILES", qualified)
    cfg = ControlV2TrainConfig(pretrained_run=str(source), store_glob=str(tmp_path / "*.zarr"))

    assert train_script._source_backbone_contract(cfg) == source.resolve()  # noqa: SLF001

    (checkpoint / "teacher_ema.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="not the qualified"):
        train_script._source_backbone_contract(cfg)  # noqa: SLF001


def test_train_wrapper_backpropagates_all_supervised_heads() -> None:
    torch.manual_seed(5)
    batch, history, horizon = 4, 4, 5
    qpos = torch.linspace(-0.4, 0.4, batch * history * 8).reshape(batch, history, 8)
    command = qpos + 0.01
    normalizer = ControlV2Normalizer(fit_control_v2_normalization(qpos, previous_command=command, dt=2.0))
    backbone = _TinyBackbone()
    head = ControlV2(
        ControlV2Config(
            width=16,
            depth=1,
            num_heads=2,
            horizon=horizon,
            qpos_history=history,
            mlp_ratio=2,
            fourier_dim=4,
        ),
        TINY_LAYOUT,
    )
    model = ControlV2TrainModel(
        backbone,
        head,
        normalizer,
        train_backbone=True,
        anchor_backbone=None,
        overlap_loss_weight=0.5,
        anchor_loss_weight=0.0,
        order_batch_fraction=1.0,
        safety_loss_weight=0.25,
        safety_tube_loss_weight=0.05,
        safety_rate_margin=0.9,
        joint_lower=(-1.0,) * 8,
        joint_upper=(1.0,) * 8,
        max_command_delta=(0.2,) * 8,
        chunk_first_n=horizon - 1,
    )
    inputs = ClipInputs(
        video=torch.randn(batch, history, 3, 16, 16),
        gel=torch.randn(batch, history, 1, 3, 16, 16),
        lowdim=torch.randn(batch, history, 1, 2),
    )
    absolute = qpos[:, -1, None].expand(-1, horizon, -1).clone()
    absolute[:, 1:, :7] += torch.linspace(0.02, 0.08, horizon - 1)[None, :, None]
    absolute[:, 1:, 7] -= 0.01
    mixed = torch.cat((absolute[..., :7] - qpos[:, -1, None, :7], absolute[..., 7:]), dim=-1)
    mixed[:, 0, :7] = 0
    mixed[:, 0, 7] = qpos[:, -1, 7]
    rate_reference = qpos[:, -1, None].expand(-1, horizon - 1, -1).clone()

    validation_a = model.validation_batch(
        inputs,
        scatter_qpos8(qpos),
        command,
        torch.ones(batch, history, dtype=torch.bool),
        mixed,
        absolute,
        rate_reference,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        torch.ones(batch, dtype=torch.bool),
    )
    validation_b = model.validation_batch(
        inputs,
        scatter_qpos8(qpos),
        command,
        torch.ones(batch, history, dtype=torch.bool),
        mixed,
        absolute,
        rate_reference,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        torch.ones(batch, dtype=torch.bool),
    )
    for name in validation_a:
        torch.testing.assert_close(validation_a[name], validation_b[name], rtol=0, atol=0)

    loss, metrics = model(
        inputs,
        scatter_qpos8(qpos),
        command,
        torch.ones(batch, history, dtype=torch.bool),
        mixed,
        absolute,
        rate_reference,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        torch.ones(batch, dtype=torch.bool),
        torch.ones(batch),
        generator=torch.Generator().manual_seed(11),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(metrics) >= {"chunk_loss", "phase_loss", "contact_loss", "corrupt_order_loss", "overlap_loss"}
    assert head.action_output.weight.grad is not None
    assert head.phase_output.weight.grad is not None
    assert head.contact_output.weight.grad is not None
    assert head.order_output.weight.grad is not None
    assert any(parameter.grad is not None for parameter in backbone.parameters())


def test_control_safety_terms_match_runtime_rows_and_limits() -> None:
    predicted = torch.zeros(1, 5, 8, requires_grad=True)
    with torch.no_grad():
        predicted[0, 1, 0] = 0.11  # rate-only violation
        predicted[0, 2, 1] = 1.2  # joint and rate violation
        predicted[0, 3, 2] = 99.0  # outside the retained first_n slice
    terms = control_safety_terms(
        predicted,
        torch.zeros(1, 4, 8),
        torch.full((8,), -1.0),
        torch.full((8,), 1.0),
        torch.full((8,), 0.1),
        rate_margin=0.9,
        first_n=2,
    )

    assert int(terms["joint_violation_values"]) == 1
    assert int(terms["rate_violation_values"]) == 2
    assert float(terms["joint_max_excess_ratio"]) == pytest.approx(0.1)
    assert float(terms["rate_max_ratio"]) == pytest.approx(12.0)
    assert float(terms["loss"]) > 0
    terms["loss"].backward()
    assert bool((predicted.grad[:, 0] == 0).all())
    assert bool((predicted.grad[:, 3:] == 0).all())


def test_control_safety_tube_terms_only_backpropagate_through_deployed_rows() -> None:
    predicted = torch.zeros(1, 5, 8, requires_grad=True)
    with torch.no_grad():
        predicted[0, 0, 0] = 100.0  # observation-time placeholder
        predicted[0, 1, 0] = 0.01
        predicted[0, 2, 1] = -0.02
        predicted[0, 3, 2] = 100.0  # forecast-only rows
        predicted[0, 4, 3] = -100.0

    terms = control_safety_tube_terms(
        predicted,
        torch.zeros_like(predicted),
        torch.zeros(1, 4, 8),
        torch.full((8,), 0.1),
        first_n=2,
    )
    terms["loss"].backward()

    assert predicted.grad is not None
    assert bool((predicted.grad[:, 0] == 0).all())
    assert bool((predicted.grad[:, 1:3].abs().sum(dim=-1) > 0).all())
    assert bool((predicted.grad[:, 3:] == 0).all())


def test_control_safety_tube_terms_penalize_tighter_expert_slack_more() -> None:
    expert = torch.zeros(1, 3, 8)
    expert[0, 1, 0] = 0.04
    predicted = expert.clone()
    predicted[0, 1, 0] = 0.06
    reference = torch.zeros(1, 2, 8)

    loose = control_safety_tube_terms(
        predicted,
        expert,
        reference,
        torch.full((8,), 0.1),
        first_n=1,
    )
    tight = control_safety_tube_terms(
        predicted,
        expert,
        reference,
        torch.full((8,), 0.07),
        first_n=1,
    )

    assert float(tight["min_slack"]) == pytest.approx(0.03)
    assert float(loose["min_slack"]) == pytest.approx(0.06)
    assert float(tight["loss"]) > float(loose["loss"])


def test_control_safety_tube_terms_reject_unsafe_expert_target() -> None:
    expert = torch.zeros(1, 3, 8)
    expert[0, 1, 0] = 0.1

    with pytest.raises(ValueError, match="slack"):
        control_safety_tube_terms(
            expert.clone(),
            expert,
            torch.zeros(1, 2, 8),
            torch.full((8,), 0.1),
            first_n=1,
        )


def test_control_safety_tube_tail_loss_backpropagates_per_example_maxima() -> None:
    predicted = torch.zeros(2, 3, 8, requires_grad=True)
    with torch.no_grad():
        predicted[0, 1, 0] = 0.2
        predicted[0, 1, 1] = 0.6
        predicted[1, 1, 2] = 0.4
        predicted[1, 1, 3] = 0.1

    terms = control_safety_tube_terms(
        predicted,
        torch.zeros_like(predicted),
        torch.zeros(2, 2, 8),
        torch.ones(8),
        first_n=1,
    )

    assert float(terms["mean_loss"]) == pytest.approx(0.285 / 16)
    assert float(terms["tail_loss"]) == pytest.approx(0.13)
    assert float(terms["loss"]) == pytest.approx(0.285 / 16 + 0.13)
    assert float(terms["max_ratio"]) == pytest.approx(0.6)
    assert float(terms["min_slack"]) == pytest.approx(1.0)

    terms["tail_loss"].backward()
    assert predicted.grad is not None
    expected = torch.zeros_like(predicted)
    expected[0, 1, 1] = 0.3
    expected[1, 1, 2] = 0.2
    torch.testing.assert_close(predicted.grad, expected)


def test_best_validation_snapshot_is_durable_and_atomically_closed(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    for step in (4, 8):
        source = checkpoints / str(step)
        source.mkdir(parents=True)
        (source / "student.pt").write_bytes(f"head-{step}".encode())
        (source / "backbone.pt").write_bytes(f"backbone-{step}".encode())
        (source / "metadata.pt").write_bytes(f"metadata-{step}".encode())
    bundle = SimpleNamespace(
        artifact_sha256="a" * 64,
        normalization_sha256="b" * 64,
        artifact=SimpleNamespace(source_store_sha256="c" * 64),
    )
    metrics = {
        "validation_action_loss": 0.25,
        "validation_examples": 16.0,
        "validation_safety_loss": 0.01,
        "validation_joint_violation_values": 0.0,
        "validation_rate_violation_values": 0.0,
        "validation_safety_violation_values": 0.0,
        "validation_joint_max_excess_ratio": 0.0,
        "validation_rate_max_ratio": 0.95,
    }

    first = snapshot_best_validation(tmp_path, 4, metrics, bundle)
    second = snapshot_best_validation(tmp_path, 8, {**metrics, "validation_action_loss": 0.2}, bundle)

    assert not first.exists()
    assert second.is_dir()
    assert (second / "student.pt").read_bytes() == b"head-8"
    marker = json.loads((tmp_path / "BEST_VALIDATION.json").read_text())
    assert marker["step"] == 8
    assert marker["selection_metric"] == "validation_action_loss"
    assert marker["selection_constraint"] == "validation_safety_violation_values==0"
    assert marker["selection_mode"] == "min"
    assert marker["tie_break"] == "earliest_step"
    assert marker["artifact_sha256"] == "a" * 64


def test_stage_initialization_uses_completed_runs_heldout_selection(tmp_path) -> None:
    source_cfg = ControlV2TrainConfig(
        exp_name="source",
        run_root=str(tmp_path),
        pretrained_run=str(tmp_path / "pretrained"),
        store_glob=str(tmp_path / "data" / "*.zarr"),
        num_train_steps=2,
        save_interval=1,
        validation_interval=1,
        wandb_enabled=False,
    )
    source = source_cfg.run_dir
    selected = source / "checkpoints" / "best_validation_0"
    selected.mkdir(parents=True)
    (source / "run_config.json").write_text(source_cfg.to_json())
    (source / "artifact.json").write_text("artifact-v3")
    (source / "normalization.json").write_text("normalization-v3")
    (source / "DONE").write_text("2\n")
    (source / "checkpoints" / "latest").write_text("2\n")

    qpos = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    normalizer = ControlV2Normalizer(fit_control_v2_normalization(qpos, previous_command=qpos + 1))
    source_head = nn.Linear(2, 2)
    source_backbone = nn.Linear(2, 2)
    with torch.no_grad():
        source_head.weight.fill_(1.5)
        source_backbone.weight.fill_(2.5)
    torch.save(source_head.state_dict(), selected / "student.pt")
    torch.save(source_backbone.state_dict(), selected / "backbone.pt")
    torch.save(normalizer.state_dict(), selected / "loss.pt")
    torch.save({}, selected / "optimizer.pt")
    (selected / "train_config.json").write_text(source_cfg.to_json())
    artifact_sha = file_sha256(source / "artifact.json")
    normalization_sha = file_sha256(source / "normalization.json")
    store_sha = "c" * 64
    prefix_scenarios = tuple((length, oldest) for length in range(1, 17) for oldest in (0, 1))
    time_bins = ("early", "middle", "late")
    validation_identities = []
    for index in range(32):
        history_length, oldest_present = prefix_scenarios[index % len(prefix_scenarios)]
        validation_identities.append(
            f"/data/lift_bottle.zarr\0episode={index // 16}\0start={index}\0stride=2"
            f"\0dataset_index={index}\0time_bin={time_bins[index % len(time_bins)]}"
            f"\0phase={index % 4}\0contact={index % 2}\0history_length={history_length}"
            f"\0oldest_command_present={int(oldest_present)}"
        )
    validation_sha = hashlib.sha256(
        json.dumps(validation_identities, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    (source / "VALIDATION_SAMPLES.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sampling": "all_episode_time_phase_contact_cold_start_stratified_v2",
                "requested_examples": 32,
                "sample_count": 32,
                "episode_count": 2,
                "sample_sha256": validation_sha,
                "episodes": [
                    {"identity": "/data/lift_bottle.zarr\0episode=0", "sample_count": 16},
                    {"identity": "/data/lift_bottle.zarr\0episode=1", "sample_count": 16},
                ],
                "samples": [
                    {"dataset_index": index, "identity": identity}
                    for index, identity in enumerate(validation_identities)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    torch.save(
        {
            "control_v2_checkpoint_schema": 4,
            "global_step": 0,
            "backbone_train_mode": source_cfg.backbone_train_mode,
            "backbone_last_n_blocks": source_cfg.backbone_last_n_blocks,
            "source_backbone_run": str(pathlib.Path(source_cfg.pretrained_run).resolve()),
            "source_backbone_step": source_cfg.pretrained_step,
            "artifact_sha256": artifact_sha,
            "normalization_sha256": normalization_sha,
            "source_store_sha256": store_sha,
            "normalization_counts": {
                "qpos": normalizer.qpos_count,
                "velocity": normalizer.velocity_count,
                "previous_command": normalizer.previous_command_count,
            },
            "best_validation_step": 0,
            "best_validation_chunk_loss": 0.2,
        },
        selected / "metadata.pt",
    )
    (source / "BEST_VALIDATION.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "selection_metric": "validation_action_loss",
                "selection_constraint": "validation_safety_violation_values==0",
                "selection_mode": "min",
                "tie_break": "earliest_step",
                "step": 0,
                "checkpoint": str(selected.resolve()),
                "artifact_sha256": artifact_sha,
                "normalization_sha256": normalization_sha,
                "source_store_sha256": store_sha,
                "validation_action_loss": 0.2,
                "validation_action_mae": 0.1,
                "validation_phase_loss": 0.3,
                "validation_contact_loss": 0.4,
                "validation_contact_count": 8.0,
                "validation_clean_order_loss": 0.5,
                "validation_overlap_loss": 0.6,
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
                "validation_examples": 32.0,
                "validation_episode_count": 2.0,
                "validation_objective": 0.7,
                "validation_sampling": "all_episode_time_phase_contact_cold_start_stratified_v2",
                "validation_sample_sha256": validation_sha,
            }
        )
    )
    target_cfg = dataclasses.replace(
        source_cfg,
        exp_name="target",
        init_run=str(source),
        backbone_train_mode="last_blocks",
    )
    target_head = nn.Linear(2, 2)
    target_backbone = nn.Linear(2, 2)
    bundle = ArtifactBundle(
        artifact=SimpleNamespace(source_store_sha256=store_sha),
        normalizer=normalizer,
        artifact_sha256=artifact_sha,
        normalization_sha256=normalization_sha,
        normalization_counts={
            "qpos": normalizer.qpos_count,
            "velocity": normalizer.velocity_count,
            "previous_command": normalizer.previous_command_count,
        },
    )

    provenance = _stage_initialization(
        target_cfg,
        bundle,
        head=target_head,
        backbone=target_backbone,
        device=torch.device("cpu"),
    )

    assert provenance["init_step"] == 0
    torch.testing.assert_close(target_head.weight, source_head.weight)
    torch.testing.assert_close(target_backbone.weight, source_backbone.weight)
