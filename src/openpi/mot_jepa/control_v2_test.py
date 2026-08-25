from __future__ import annotations

import dataclasses

import pytest
import torch

from openpi.mot_jepa.control_v2 import ACTIVE_ACTION_DIM
from openpi.mot_jepa.control_v2 import ARM_DIM
from openpi.mot_jepa.control_v2 import STATE_FEATURE_DIM
from openpi.mot_jepa.control_v2 import ControlV2
from openpi.mot_jepa.control_v2 import ControlV2Config
from openpi.mot_jepa.control_v2 import make_chunk_relative_target
from openpi.mot_jepa.control_v2_data import build_state_features
from openpi.mot_jepa.control_v2_data import scatter_qpos8
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.mot_encoder import EncoderOutput

TINY_LAYOUT = TokenLayout(
    num_frames=4,
    tubelet_t=2,
    video_size=32,
    video_patch=16,
    gel_size=16,
    gel_patch=16,
    num_gel_pads=2,
    lowdim_slots=3,
    lowdim_channels=4,
    video_width=12,
    tactile_width=8,
)


def _config(**kwargs) -> ControlV2Config:
    values = {
        "width": 32,
        "depth": 2,
        "num_heads": 4,
        "horizon": 5,
        "qpos_history": 4,
        "mlp_ratio": 2.0,
        "fourier_dim": 4,
    }
    values.update(kwargs)
    return ControlV2Config(**values)


def _encoded(batch: int = 2) -> EncoderOutput:
    video = torch.randn(batch, TINY_LAYOUT.num_video_tokens, TINY_LAYOUT.video_width)
    tactile = torch.randn(
        batch,
        TINY_LAYOUT.num_gel_tokens + TINY_LAYOUT.num_lowdim_tokens,
        TINY_LAYOUT.tactile_width,
    )
    readout = [
        torch.empty(batch, TINY_LAYOUT.num_steps, TINY_LAYOUT.video_width),
        torch.empty(batch, TINY_LAYOUT.num_steps, TINY_LAYOUT.tactile_width),
    ]
    return EncoderOutput(tokens=[video, tactile], sync_readout=readout, final_readout=readout)


def _state(batch: int = 2, history: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = torch.linspace(-0.2, 0.3, batch * history * ACTIVE_ACTION_DIM).reshape(batch, history, -1)
    command = (qpos[:, -1] + 0.01)[:, None].expand_as(qpos)
    state_features = build_state_features(scatter_qpos8(qpos), previous_command=command).tensor
    assert state_features.shape[-1] == STATE_FEATURE_DIM
    return state_features, qpos[:, -1]


def test_default_config_is_the_production_query_decoder_shape() -> None:
    config = ControlV2Config()

    assert (config.width, config.depth, config.num_heads, config.horizon) == (512, 8, 8, 32)


def test_zero_initialization_is_an_exact_hold_current_policy() -> None:
    torch.manual_seed(3)
    config = _config()
    model = ControlV2(config, TINY_LAYOUT).eval()
    state_features, current_qpos = _state()

    first = model(_encoded(), state_features, current_qpos)

    expected_mixed = torch.cat(
        (
            torch.zeros(current_qpos.shape[0], config.horizon, ARM_DIM),
            current_qpos[:, None, ARM_DIM:].expand(-1, config.horizon, -1),
        ),
        dim=-1,
    )
    expected_absolute = current_qpos[:, None, :].expand(-1, config.horizon, -1)
    torch.testing.assert_close(first.action_chunk, expected_mixed, rtol=0, atol=0)
    torch.testing.assert_close(first.absolute_qpos_chunk, expected_absolute, rtol=0, atol=0)
    torch.testing.assert_close(first.residual_chunk, torch.zeros_like(first.residual_chunk), rtol=0, atol=0)
    torch.testing.assert_close(
        first.normalized_residual_chunk, torch.zeros_like(first.normalized_residual_chunk), rtol=0, atol=0
    )
    # No noise input or sampling path exists; repeated use of identical inputs is deterministic.
    encoded = _encoded()
    repeated_a = model(encoded, state_features, current_qpos)
    repeated_b = model(encoded, state_features, current_qpos)
    torch.testing.assert_close(repeated_a.action_chunk, repeated_b.action_chunk, rtol=0, atol=0)
    assert first.phase_logits.shape == (current_qpos.shape[0], 4)
    assert first.contact_logits.shape == (current_qpos.shape[0], 1)
    assert first.order_logits.shape == (current_qpos.shape[0], 4)


def test_row_zero_remains_placeholder_after_the_head_learns() -> None:
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    with torch.no_grad():
        model.action_output.bias.fill_(0.25)
    state_features, current_qpos = _state()
    output = model(_encoded(), state_features, current_qpos)

    torch.testing.assert_close(output.residual_chunk[:, 0], torch.zeros_like(output.residual_chunk[:, 0]))
    torch.testing.assert_close(output.action_chunk[:, 0, :ARM_DIM], torch.zeros_like(current_qpos[:, :ARM_DIM]))
    torch.testing.assert_close(output.action_chunk[:, 0, ARM_DIM:], current_qpos[:, ARM_DIM:])
    torch.testing.assert_close(output.absolute_qpos_chunk[:, 0], current_qpos)
    torch.testing.assert_close(
        output.action_chunk[:, 1:, :ARM_DIM], torch.full_like(output.action_chunk[:, 1:, :ARM_DIM], 0.25)
    )


def test_action_scale_reconstructs_physical_residuals_and_is_checkpointed() -> None:
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    scale = torch.linspace(0.1, 0.8, ACTIVE_ACTION_DIM)
    model.load_action_scale(scale)
    with torch.no_grad():
        model.action_output.bias.fill_(1)
    state_features, current_qpos = _state()

    output = model(_encoded(), state_features, current_qpos)

    torch.testing.assert_close(output.normalized_residual_chunk[:, 1:], torch.ones_like(output.residual_chunk[:, 1:]))
    torch.testing.assert_close(output.residual_chunk[:, 1:], scale[None, None].expand_as(output.residual_chunk[:, 1:]))
    torch.testing.assert_close(output.action_chunk[:, 1:, :ARM_DIM], output.residual_chunk[:, 1:, :ARM_DIM])
    torch.testing.assert_close(
        output.action_chunk[:, 1:, ARM_DIM:],
        (current_qpos[:, None, ARM_DIM:] + scale[ARM_DIM]).expand_as(output.action_chunk[:, 1:, ARM_DIM:]),
    )
    torch.testing.assert_close(model.state_dict()["action_scale"], scale)

    with pytest.raises(ValueError, match="strictly positive"):
        model.load_action_scale(torch.zeros(ACTIVE_ACTION_DIM))


def test_memory_uses_dense_video_and_gel_but_excludes_lowdim_tail() -> None:
    torch.manual_seed(4)
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    encoded = _encoded()
    state_features, current_qpos = _state()
    original = model.build_memory(encoded, state_features, current_qpos)

    changed = dataclasses.replace(encoded, tokens=[encoded.tokens[0], encoded.tokens[1].clone()])
    changed.tokens[1][:, TINY_LAYOUT.num_gel_tokens :] = 1e6
    lowdim_changed = model.build_memory(changed, state_features, current_qpos)
    torch.testing.assert_close(original, lowdim_changed, rtol=0, atol=0)

    expected_tokens = TINY_LAYOUT.num_video_tokens + TINY_LAYOUT.num_gel_tokens + state_features.shape[1]
    assert original.shape == (state_features.shape[0], expected_tokens, model.config.width)

    changed_video = dataclasses.replace(encoded, tokens=[encoded.tokens[0] + 1, encoded.tokens[1]])
    assert not torch.equal(original, model.build_memory(changed_video, state_features, current_qpos))


def test_state_memory_preserves_normalized_values_and_presence_bits() -> None:
    torch.manual_seed(5)
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    encoded = _encoded()
    state_features, current_qpos = _state()
    baseline = model.build_memory(encoded, state_features, current_qpos)

    command_missing = state_features.clone()
    command_missing[:, -1, 16:24] = 0
    command_missing[:, -1, 26] = 0
    changed = model.build_memory(encoded, command_missing, current_qpos)

    assert not torch.equal(baseline, changed)


def test_short_history_is_right_aligned_and_supported() -> None:
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    state_features, current_qpos = _state(history=1)
    output = model(_encoded(), state_features, current_qpos)

    assert output.action_chunk.shape == (state_features.shape[0], model.config.horizon, ACTIVE_ACTION_DIM)
    assert torch.isfinite(output.phase_logits).all()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda encoded, state, current: (
                dataclasses.replace(encoded, tokens=[encoded.tokens[0][:, :-1], encoded.tokens[1]]),
                state,
                current,
            ),
            "video tokens",
        ),
        (lambda encoded, state, current: (encoded, state[..., :-1], current), "state_features"),
        (lambda encoded, state, current: (encoded, state, current[:, :-1]), "current_qpos"),
        (lambda encoded, state, current: (encoded, state.clone().fill_(float("nan")), current), "finite"),
    ],
)
def test_invalid_inputs_fail_loudly(mutate, message: str) -> None:
    model = ControlV2(_config(), TINY_LAYOUT)
    inputs = mutate(_encoded(), *_state())

    with pytest.raises(ValueError, match=message):
        model(*inputs)


def test_absolute_demo_conversion_enforces_mixed_row_zero_contract() -> None:
    current = torch.arange(2 * ACTIVE_ACTION_DIM, dtype=torch.float32).reshape(2, -1) / 10
    absolute = current[:, None].expand(-1, 5, -1).clone()
    absolute[:, 1:, :ARM_DIM] += torch.arange(4, dtype=torch.float32)[None, :, None] / 20
    absolute[:, 1:, ARM_DIM] -= 0.03

    target = make_chunk_relative_target(absolute, current)

    torch.testing.assert_close(target[:, 0, :ARM_DIM], torch.zeros_like(target[:, 0, :ARM_DIM]))
    torch.testing.assert_close(target[:, 0, ARM_DIM], current[:, ARM_DIM])
    torch.testing.assert_close(target[:, 1:, :ARM_DIM], absolute[:, 1:, :ARM_DIM] - current[:, None, :ARM_DIM])
    torch.testing.assert_close(target[:, 1:, ARM_DIM], absolute[:, 1:, ARM_DIM])


def test_weighted_loss_trains_action_and_all_auxiliary_heads() -> None:
    torch.manual_seed(6)
    model = ControlV2(_config(), TINY_LAYOUT)
    state_features, current_qpos = _state()
    output = model(_encoded(), state_features, current_qpos)
    target = output.action_chunk.detach().clone()
    target[:, 1:, :ARM_DIM] += 0.1
    target[:, 1:, ARM_DIM] -= 0.01

    losses = model.loss(
        output,
        target,
        sample_weight=torch.tensor([3.0, 1.0]),
        phase_labels=torch.tensor([0, 3]),
        contact_targets=torch.tensor([0.0, 1.0]),
        order_labels=torch.tensor([2, 1]),
    )
    losses.total.backward()

    assert all(
        torch.isfinite(value) for value in (losses.total, losses.chunk, losses.phase, losses.contact, losses.order)
    )
    assert losses.total > losses.chunk
    assert model.action_output.weight.grad is not None
    assert model.phase_output.weight.grad is not None
    assert model.contact_output.weight.grad is not None
    assert model.order_output.weight.grad is not None


def test_loss_downweights_far_horizon_and_upweights_gripper() -> None:
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    state_features, current_qpos = _state()
    output = model(_encoded(), state_features, current_qpos)

    def loss_at(step: int, dimension: int) -> torch.Tensor:
        target = output.action_chunk.detach().clone()
        target[:, step, dimension] += 0.1
        return model.loss(output, target).chunk

    assert loss_at(1, 0) > loss_at(model.config.horizon - 1, 0)
    torch.testing.assert_close(loss_at(1, ARM_DIM), 2 * loss_at(1, 0))


def test_loss_compares_decoder_outputs_in_action_normalized_space() -> None:
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    scale = torch.linspace(0.1, 0.8, ACTIVE_ACTION_DIM)
    model.load_action_scale(scale)
    state_features, current_qpos = _state()
    output = model(_encoded(), state_features, current_qpos)

    def loss_for_arm(dimension: int) -> torch.Tensor:
        target = output.action_chunk.detach().clone()
        target[:, 1, dimension] += 0.2 * scale[dimension]
        return model.loss(output, target).chunk

    torch.testing.assert_close(loss_for_arm(0), loss_for_arm(ARM_DIM - 1))


def test_loss_rejects_a_non_placeholder_row_zero() -> None:
    model = ControlV2(_config(), TINY_LAYOUT).eval()
    state_features, current_qpos = _state()
    output = model(_encoded(), state_features, current_qpos)
    target = output.action_chunk.detach().clone()
    target[:, 0, 0] = 0.1

    with pytest.raises(ValueError, match="row zero"):
        model.loss(output, target)
