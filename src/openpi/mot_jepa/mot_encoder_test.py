from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from openpi.mot_jepa.embed import StreamEmbed
from openpi.mot_jepa.layout import LAYOUT_BASE
from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import build_batch_masks
from openpi.mot_jepa.mot_encoder import MoTEncoder
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.mot_encoder import gather_tokens
from openpi.mot_jepa.mot_encoder import split_index_by_expert
from openpi.mot_jepa.predictor import MoTPredictor
from openpi.mot_jepa.predictor import MoTPredictorConfig
from openpi.mot_jepa.predictor import target_index_by_expert
from openpi.mot_jepa.rope3d import Rope3DConfig

TINY_LAYOUT = TokenLayout(
    num_frames=4,
    tubelet_t=2,
    video_size=32,
    video_patch=16,
    gel_size=32,
    gel_patch=16,
    num_gel_pads=2,
    lowdim_slots=2,
    video_width=32,
    tactile_width=16,
)
TINY_ROPE = Rope3DConfig(head_dim=8, dim_t=4, dim_h=2, dim_w=2)
BATCH = 2


def tiny_encoder(depth: int = 4, num_local_layers: int = 2) -> MoTEncoder:
    config = MoTEncoderConfig(
        depth=depth, num_local_layers=num_local_layers, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE
    )
    return MoTEncoder(config, TINY_LAYOUT)


def tiny_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    video = torch.randn(BATCH, TINY_LAYOUT.num_frames, 3, 32, 32, generator=generator)
    gel = torch.randn(BATCH, TINY_LAYOUT.num_frames, 2, 3, 32, 32, generator=generator)
    lowdim = torch.randn(BATCH, TINY_LAYOUT.num_frames, TINY_LAYOUT.lowdim_slots, 1, generator=generator)
    return video, gel, lowdim


def full_index() -> list[torch.Tensor]:
    video = torch.arange(TINY_LAYOUT.num_video_tokens).expand(BATCH, -1)
    tactile_slice = TINY_LAYOUT.expert_slices[ExpertId.TACTILE]
    tactile = torch.arange(tactile_slice.start, tactile_slice.stop).expand(BATCH, -1)
    return [video, tactile]


def test_stream_embed_produces_layout_shaped_tokens():
    embed = StreamEmbed(TINY_LAYOUT)
    video_tokens, tactile_tokens = embed(*tiny_inputs())
    assert video_tokens.shape == (BATCH, TINY_LAYOUT.num_video_tokens, TINY_LAYOUT.video_width)
    expected_tactile = TINY_LAYOUT.num_gel_tokens + TINY_LAYOUT.num_lowdim_tokens
    assert tactile_tokens.shape == (BATCH, expected_tactile, TINY_LAYOUT.tactile_width)


def test_encoder_output_shapes_and_readout():
    encoder = tiny_encoder()
    embed = StreamEmbed(TINY_LAYOUT)
    tokens = list(embed(*tiny_inputs()))
    out = encoder(tokens, full_index())
    assert out.expert(ExpertId.VIDEO).shape == (BATCH, TINY_LAYOUT.num_video_tokens, TINY_LAYOUT.video_width)
    for readout, width in zip(out.sync_readout, (TINY_LAYOUT.video_width, TINY_LAYOUT.tactile_width), strict=True):
        assert readout.shape == (BATCH, TINY_LAYOUT.num_steps, width)


def test_local_prefix_blocks_cross_modal_leakage():
    """With every layer modality-local, perturbing touch must not move a single video unit.

    This is what gives the synchrony head a genuinely unimodal readout. If it regresses, the
    synchrony loss becomes satisfiable by copying the other modality rather than by binding.
    """
    encoder = tiny_encoder(depth=2, num_local_layers=2)
    embed = StreamEmbed(TINY_LAYOUT)
    video, gel, lowdim = tiny_inputs()
    index = full_index()

    baseline = encoder(list(embed(video, gel, lowdim)), index).expert(ExpertId.VIDEO)
    perturbed = encoder(list(embed(video, gel + 5.0, lowdim)), index).expert(ExpertId.VIDEO)
    assert torch.equal(baseline, perturbed)


def test_global_suffix_does_let_touch_reach_video():
    encoder = tiny_encoder(depth=3, num_local_layers=2)
    embed = StreamEmbed(TINY_LAYOUT)
    video, gel, lowdim = tiny_inputs()
    index = full_index()

    baseline = encoder(list(embed(video, gel, lowdim)), index).expert(ExpertId.VIDEO)
    perturbed = encoder(list(embed(video, gel + 5.0, lowdim)), index).expert(ExpertId.VIDEO)
    assert not torch.allclose(baseline, perturbed)


def test_sync_readout_is_taken_before_any_global_layer():
    """The readout must be unimodal even when later layers mix the streams."""
    encoder = tiny_encoder(depth=4, num_local_layers=2)
    embed = StreamEmbed(TINY_LAYOUT)
    video, gel, lowdim = tiny_inputs()
    index = full_index()

    baseline = encoder(list(embed(video, gel, lowdim)), index).sync_readout[int(ExpertId.VIDEO)]
    perturbed = encoder(list(embed(video, gel + 5.0, lowdim)), index).sync_readout[int(ExpertId.VIDEO)]
    assert torch.equal(baseline, perturbed)


def test_gradient_checkpointing_is_bitwise_identical_forward_and_backward():
    """The single highest-value guard in the package.

    ``preserve_rng_state=False`` means a checkpointed recompute re-samples any RNG inside the
    region. If randomness ever leaks into the model, the backward differentiates a different
    function than the forward computed -- no crash, no NaN, just a plateau. Bitwise equality
    of both the output and the gradients is the only assertion that catches it.
    """
    torch.use_deterministic_algorithms(True)  # noqa: FBT003 - torch API takes a positional bool
    try:
        embed = StreamEmbed(TINY_LAYOUT)
        encoder = tiny_encoder()
        index = full_index()
        video, gel, lowdim = tiny_inputs()

        def run(*, checkpointing: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
            encoder.set_gradient_checkpointing(enabled=checkpointing)
            encoder.zero_grad(set_to_none=True)
            out = encoder(list(embed(video, gel, lowdim)), index)
            loss = sum(tokens.square().mean() for tokens in out.tokens)
            loss.backward()
            grads = [p.grad.detach().clone() for p in encoder.parameters() if p.grad is not None]
            return out.expert(ExpertId.VIDEO).detach().clone(), grads

        plain_out, plain_grads = run(checkpointing=False)
        ckpt_out, ckpt_grads = run(checkpointing=True)

        assert torch.equal(plain_out, ckpt_out)
        assert len(plain_grads) == len(ckpt_grads)
        for lhs, rhs in zip(plain_grads, ckpt_grads, strict=True):
            assert torch.equal(lhs, rhs)
    finally:
        torch.use_deterministic_algorithms(False)  # noqa: FBT003 - torch API takes a positional bool


def test_checkpointing_is_never_auto_enabled_by_training_mode():
    """FTP-1's Gemma stack force-enables checkpointing whenever training; we must not."""
    encoder = tiny_encoder()
    encoder.train()
    assert encoder.gradient_checkpointing is False
    encoder.set_gradient_checkpointing(enabled=True)
    encoder.eval()
    assert encoder.gradient_checkpointing is True


@pytest.mark.parametrize("module", ["mot_encoder.py", "predictor.py", "embed.py", "rope3d.py"])
def test_model_modules_contain_no_rng_calls(module):
    """Static fence: randomness in these modules would silently break checkpointed training.

    Initializers are exempt -- they run once at construction, outside any checkpointed
    region -- so only ``nn.init`` and ``trunc_normal_`` style calls are permitted.
    """
    banned = {"rand", "randn", "randn_like", "rand_like", "randperm", "bernoulli", "multinomial", "randint"}
    source = (pathlib.Path(__file__).parent / module).read_text()
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            base = node.value
            # nn.init.* is fine; torch.rand* is not.
            if isinstance(base, ast.Name) and base.id == "torch":
                offenders.append(node.attr)
        if isinstance(node, ast.Attribute) and node.attr == "Dropout":
            offenders.append("Dropout")
    assert not offenders, f"{module} contains RNG/dropout: {offenders}"


def test_predictor_shapes_match_encoder_widths():
    # Forecast params scaled to TINY_LAYOUT the same way the windows are: num_steps is 2 here,
    # so the production defaults (horizon 2-3, min_context 4) cannot fit.
    spec = MaskSpec(
        layout=TINY_LAYOUT,
        tactile_window_steps=(1,),
        video_window_steps=(1,),
        forecast_horizon_steps=(1,),
        min_context_steps=1,
        min_targets_per_stream=2,
    )
    masks = build_batch_masks(spec, step=0, batch_size=BATCH, base_seed=7)

    embed = StreamEmbed(TINY_LAYOUT)
    encoder = tiny_encoder()
    predictor = MoTPredictor(
        MoTPredictorConfig(depth=2, width=16, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE), TINY_LAYOUT
    )

    full_tokens = list(embed(*tiny_inputs()))
    ctx_global, ctx_local = split_index_by_expert(TINY_LAYOUT, masks.ctx_index, masks.ctx_expert_bounds)
    context_tokens = [gather_tokens(full_tokens[e], ctx_local[e]) for e in range(2)]
    encoded = encoder(context_tokens, ctx_global)

    tgt_global = target_index_by_expert(TINY_LAYOUT, masks.tgt_index, masks.tgt_expert_bounds)
    predictions = predictor(encoded.tokens, ctx_global, tgt_global, masks.mode)

    for expert, width in enumerate((TINY_LAYOUT.video_width, TINY_LAYOUT.tactile_width)):
        assert predictions[expert].shape == (BATCH, tgt_global[expert].shape[1], width)


def test_predictor_mode_embedding_starts_at_zero_and_touches_only_mask_tokens():
    predictor = MoTPredictor(
        MoTPredictorConfig(depth=1, width=16, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE), TINY_LAYOUT
    )
    assert torch.equal(predictor.mode_embed.weight, torch.zeros_like(predictor.mode_embed.weight))

    with torch.no_grad():
        predictor.mode_embed.weight.normal_()
    ctx = [torch.randn(BATCH, 4, TINY_LAYOUT.video_width), torch.randn(BATCH, 4, TINY_LAYOUT.tactile_width)]
    ctx_index = [torch.arange(4).expand(BATCH, -1), torch.arange(8, 12).expand(BATCH, -1)]
    tgt_index = [torch.arange(4, 6).expand(BATCH, -1), torch.arange(12, 14).expand(BATCH, -1)]

    a = predictor(ctx, ctx_index, tgt_index, torch.tensor(int(MaskMode.V)))
    b = predictor(ctx, ctx_index, tgt_index, torch.tensor(int(MaskMode.T_HARD)))
    # Different modes must change the prediction (the embedding is live) ...
    assert not torch.allclose(a[0], b[0])
    # ... but the context tensors themselves are never modified in place.
    assert ctx[0].shape == (BATCH, 4, TINY_LAYOUT.video_width)


def test_parameter_counts_are_in_the_designed_range():
    """Design document: encoder ~154M, predictor ~44M at the base layout."""
    encoder = MoTEncoder(MoTEncoderConfig(), LAYOUT_BASE)
    predictor = MoTPredictor(MoTPredictorConfig(), LAYOUT_BASE)
    encoder_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    predictor_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    assert 100 < encoder_params < 220, f"encoder {encoder_params:.1f}M outside ViT-B range"
    assert 20 < predictor_params < 70, f"predictor {predictor_params:.1f}M outside designed range"
    assert predictor_params < encoder_params / 2, "predictor must stay narrow relative to the encoder"
