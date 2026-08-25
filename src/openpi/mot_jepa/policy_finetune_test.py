from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.mot_jepa.action_dit import ActionDiTConfig
from openpi.mot_jepa.action_dit import LinearHead
from openpi.mot_jepa.action_dit import flow_matching_loss
from openpi.mot_jepa.model import ClipInputs
from openpi.mot_jepa.model import MotJepaBackbone
from openpi.mot_jepa.mot_encoder import EncoderOutput
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.mot_encoder_test import TINY_LAYOUT
from openpi.mot_jepa.mot_encoder_test import TINY_ROPE
from openpi.mot_jepa.policy_finetune import UNREACHABLE_SENSOR_PARAMETERS
from openpi.mot_jepa.policy_finetune import PolicyTrainModel
from openpi.mot_jepa.policy_finetune import configure_backbone_trainability
from openpi.mot_jepa.policy_finetune import grad_norm
from openpi.mot_jepa.policy_finetune import make_step_generator


class _SensorEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 3)
        self.sensor_embed = nn.Embedding(4, 3)


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(3), nn.LayerNorm(3)])


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Module()
        self.embed.video = nn.Linear(3, 3)
        self.embed.gel = _SensorEmbed()
        self.embed.lowdim = _SensorEmbed()
        self.encoder = _FakeEncoder()


def test_frozen_mode_preserves_the_legacy_no_grad_backbone() -> None:
    backbone = _FakeBackbone()
    selection = configure_backbone_trainability(backbone, mode="frozen", last_n_blocks=2)

    assert selection.parameters == ()
    assert not backbone.training
    assert not any(parameter.requires_grad for parameter in backbone.parameters())


def test_empty_frozen_backbone_grad_norm_stays_on_training_device() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm = grad_norm((), device=device)

    assert norm.device == device
    assert norm.item() == 0.0


def test_last_blocks_selects_only_the_tail_and_final_norms() -> None:
    backbone = _FakeBackbone()
    selection = configure_backbone_trainability(backbone, mode="last_blocks", last_n_blocks=2)

    assert selection.names
    assert all(
        name.startswith(("encoder.blocks.2.", "encoder.blocks.3.", "encoder.norms.")) for name in selection.names
    )
    assert not backbone.encoder.blocks[1].weight.requires_grad
    assert backbone.encoder.blocks[2].weight.requires_grad
    assert backbone.encoder.norms[0].weight.requires_grad


def test_full_mode_freezes_only_sensor_embeddings_outside_the_input_contract() -> None:
    backbone = _FakeBackbone()
    selection = configure_backbone_trainability(backbone, mode="full", last_n_blocks=2)
    named = dict(backbone.named_parameters())

    assert selection.excluded_unreachable == UNREACHABLE_SENSOR_PARAMETERS
    assert all(not named[name].requires_grad for name in UNREACHABLE_SENSOR_PARAMETERS)
    assert all(
        parameter.requires_grad for name, parameter in named.items() if name not in UNREACHABLE_SENSOR_PARAMETERS
    )


def test_last_blocks_rejects_an_impossible_depth() -> None:
    with pytest.raises(ValueError, match="must be in"):
        configure_backbone_trainability(_FakeBackbone(), mode="last_blocks", last_n_blocks=5)


class _DifferentiableBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def encode_full(self, inputs) -> EncoderOutput:
        pooled = inputs.video.flatten(start_dim=1).mean(dim=1) * self.scale
        readout = pooled[:, None, None].expand(-1, 2, 3)
        return EncoderOutput(tokens=[], sync_readout=[readout, readout], final_readout=[readout, readout])


class _TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))

    def forward(self, encoded, action_mask, noisy_actions, timestep, domain_id):
        del action_mask, timestep, domain_id
        value = sum(readout.mean(dim=(1, 2)) for readout in encoded.final_readout) * self.scale
        return value[:, None, None].expand_as(noisy_actions)


@pytest.mark.parametrize("mode", ["frozen", "adapted"])
def test_composite_forward_routes_backbone_gradients_only_when_selected(mode: str) -> None:
    train_backbone = mode == "adapted"
    backbone = _DifferentiableBackbone()
    backbone.requires_grad_(requires_grad=train_backbone)
    head = _TinyHead()
    model = PolicyTrainModel(
        backbone,
        head,
        objective="linear",
        drifting_config=None,
        train_backbone=train_backbone,
    )
    batch, horizon, dim = 3, 2, 4
    inputs = SimpleNamespace(video=torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 1, 2, 2))
    actions = torch.zeros(batch, horizon, dim)
    action_mask = torch.ones(batch, dim)
    chunk_mask = torch.ones_like(actions)
    domain_id = torch.zeros(batch, dtype=torch.long)

    loss, _ = model(inputs, actions, action_mask, chunk_mask, domain_id)
    loss.backward()

    assert torch.isfinite(loss)
    assert head.scale.grad is not None
    assert torch.isfinite(head.scale.grad)
    assert (backbone.scale.grad is not None) is train_backbone


class _RecordingFlowHead:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, encoded, action_mask, noisy_actions, timestep, domain_id):
        del encoded, action_mask, domain_id
        self.calls.append((noisy_actions.detach().clone(), timestep.detach().clone()))
        return torch.zeros_like(noisy_actions)


def _flow_draw(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head = _RecordingFlowHead()
    actions = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 24
    mask = torch.ones_like(actions)
    loss, _ = flow_matching_loss(
        head,
        encoded=None,
        actions=actions,
        action_mask=torch.ones(2, 4),
        chunk_mask=mask,
        domain_id=torch.zeros(2, dtype=torch.long),
        generator=make_step_generator(torch.device("cpu"), base_seed=17, step=seed, rank=0),
    )
    noisy_actions, timestep = head.calls[0]
    return loss, noisy_actions, timestep


def test_flow_noise_and_beta_time_are_stateless_per_step() -> None:
    first = _flow_draw(11)
    repeated = _flow_draw(11)
    different = _flow_draw(12)

    for actual, expected in zip(first, repeated, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(first[1], different[1])
    assert not torch.equal(first[2], different[2])
    assert bool(((first[2] >= 0.001) & (first[2] <= 1.0)).all())


def test_real_tiny_backbone_and_head_execute_two_adaptation_steps() -> None:
    """Exercise the actual embed/encoder/readout/head graph, including full-mode exclusions."""
    torch.manual_seed(23)
    backbone = MotJepaBackbone(
        TINY_LAYOUT,
        MoTEncoderConfig(
            depth=2,
            num_local_layers=1,
            num_heads=2,
            head_dim=8,
            mlp_ratio=2.0,
            rope=TINY_ROPE,
        ),
    )
    selection = configure_backbone_trainability(backbone, mode="full", last_n_blocks=2)
    head = LinearHead(
        ActionDiTConfig(width=16, depth=1, num_heads=2, horizon=2, objective="linear"),
        TINY_LAYOUT,
        num_domains=2,
        action_dim=4,
    )
    model = PolicyTrainModel(backbone, head, objective="linear", drifting_config=None, train_backbone=True)
    parameters = (*head.parameters(), *selection.parameters)
    optimizer = torch.optim.AdamW(parameters, lr=1e-2)
    batch = 2
    generator = torch.Generator().manual_seed(31)
    inputs = ClipInputs(
        video=torch.randn(batch, TINY_LAYOUT.num_frames, 3, 32, 32, generator=generator),
        gel=torch.randn(batch, TINY_LAYOUT.num_frames, 2, 3, 32, 32, generator=generator),
        lowdim=torch.randn(
            batch,
            TINY_LAYOUT.num_frames,
            TINY_LAYOUT.lowdim_slots,
            1,
            generator=generator,
        ),
    )
    actions = torch.randn(batch, 2, 4, generator=generator)
    action_mask = torch.ones(batch, 4)
    chunk_mask = torch.ones_like(actions)
    domain_id = torch.tensor([0, 1])

    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(
            inputs,
            actions,
            action_mask,
            chunk_mask,
            domain_id,
            generator=make_step_generator(torch.device("cpu"), base_seed=5, step=step, rank=0),
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
        optimizer.step()

    assert any(parameter.grad.abs().sum() > 0 for parameter in selection.parameters)
