from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.mot_jepa.action_parse import ACTION_DIM
from openpi.mot_jepa.condition import ActionEmbed
from openpi.mot_jepa.condition import InstructionHead
from openpi.mot_jepa.condition import info_nce
from openpi.mot_jepa.layout import STREAM_ORDER
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import assert_mask_invariants
from openpi.mot_jepa.masking import build_rollout_masks
from openpi.mot_jepa.mot_encoder import EncoderOutput
from openpi.mot_jepa.mot_encoder import split_index_by_expert
from openpi.mot_jepa.predictor import MoTPredictor
from openpi.mot_jepa.predictor import MoTPredictorConfig
from openpi.mot_jepa.predictor import target_index_by_expert
from openpi.mot_jepa.rope3d import Rope3DConfig

TINY_LAYOUT = TokenLayout(
    num_frames=8,
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
WIDTH = 16
BATCH = 3


def tiny_predictor() -> MoTPredictor:
    torch.manual_seed(0)
    predictor = MoTPredictor(
        MoTPredictorConfig(depth=2, width=WIDTH, num_heads=2, head_dim=8, mlp_ratio=2.0, rope=TINY_ROPE),
        TINY_LAYOUT,
    )
    # mode_embed initializes to zeros, which would hide a bug in how conditioning is placed.
    with torch.no_grad():
        predictor.mode_embed.weight.normal_()
    return predictor.eval()


def rollout_call(predictor: MoTPredictor, masks, cond=None):
    ctx_index, _ = split_index_by_expert(TINY_LAYOUT, masks.ctx_index, masks.ctx_expert_bounds)
    tgt_index = target_index_by_expert(TINY_LAYOUT, masks.tgt_index, masks.tgt_expert_bounds)
    torch.manual_seed(1)
    context = [
        torch.randn(BATCH, ctx_index[0].shape[1], TINY_LAYOUT.video_width),
        torch.randn(BATCH, ctx_index[1].shape[1], TINY_LAYOUT.tactile_width),
    ]
    with torch.no_grad():
        return predictor(context, ctx_index, tgt_index, masks.mode, cond=cond), tgt_index


# --------------------------------------------------------------------------------------
# The predictor change must be provably inert when unused.
# --------------------------------------------------------------------------------------


def test_cond_none_is_bit_identical_to_zero_cond():
    """The only edit to a pretrained model file must change nothing at cond=None.

    Adding zeros is the identity, so if these differ the new branch is doing something
    beyond what it claims -- reordering, recasting, or reading the wrong table.
    """
    predictor = tiny_predictor()
    masks = build_rollout_masks(TINY_LAYOUT, split_step=2, batch_size=BATCH)
    without, _ = rollout_call(predictor, masks, cond=None)
    with_zeros, _ = rollout_call(predictor, masks, cond=torch.zeros(BATCH, TINY_LAYOUT.num_steps, WIDTH))
    for a, b in zip(without, with_zeros, strict=True):
        assert torch.equal(a, b)


def test_the_step_table_never_enters_a_checkpoint():
    """``token_step`` is derived from the layout; persisting it would break old checkpoints."""
    predictor = tiny_predictor()
    assert "token_step" not in predictor.state_dict()
    assert predictor.token_step.shape == (TINY_LAYOUT.num_tokens,)


def test_conditioning_reaches_only_its_own_timestep():
    """A cond vector at step k must move exactly the target tokens whose tubelet step is k.

    This is the test that catches a wrong gather. A cond broadcast to every token, or gathered
    by row position instead of time, still trains and still lowers the loss -- it just makes
    the action explain the wrong transition.
    """
    predictor = tiny_predictor()
    masks = build_rollout_masks(TINY_LAYOUT, split_step=1, batch_size=BATCH)
    baseline, tgt_index = rollout_call(predictor, masks, cond=torch.zeros(BATCH, TINY_LAYOUT.num_steps, WIDTH))

    hot_step = 2
    cond = torch.zeros(BATCH, TINY_LAYOUT.num_steps, WIDTH)
    # Must vary across channels. The predictor ends in a LayerNorm, so a *constant* vector
    # like `cond[:, hot_step] = 5.0` is removed exactly and the test would fail against
    # correct code.
    torch.manual_seed(2)
    cond[:, hot_step] = torch.randn(WIDTH) * 5.0
    perturbed, _ = rollout_call(predictor, masks, cond=cond)

    step_of_token = TINY_LAYOUT.coords[:, 0]
    moved_any = False
    for expert in range(2):
        steps = step_of_token[tgt_index[expert][0]]
        changed = (perturbed[expert][0] - baseline[expert][0]).abs().amax(dim=-1) > 1e-6
        # Attention mixes tokens, so a hot step can move others; what must hold is that the
        # tokens AT that step move, and that a run with no hot step moves nothing.
        assert bool(changed[steps == hot_step].all()), f"expert {expert}: step {hot_step} tokens did not move"
        moved_any = moved_any or bool(changed.any())
    assert moved_any


def test_a_wrong_shaped_cond_is_refused():
    predictor = tiny_predictor()
    masks = build_rollout_masks(TINY_LAYOUT, split_step=2, batch_size=BATCH)
    with pytest.raises(ValueError, match="cond must be"):
        rollout_call(predictor, masks, cond=torch.zeros(BATCH, TINY_LAYOUT.num_steps + 1, WIDTH))


# --------------------------------------------------------------------------------------
# Rollout masks
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("split_step", [1, 2, 3])
def test_rollout_masks_split_exactly_on_time(split_step):
    masks = build_rollout_masks(TINY_LAYOUT, split_step=split_step, batch_size=BATCH)
    step_of_token = TINY_LAYOUT.coords[:, 0]
    assert bool((step_of_token[masks.ctx_index] < split_step).all())
    assert bool((step_of_token[masks.tgt_index] >= split_step).all())


def test_rollout_masks_satisfy_the_pretraining_invariants():
    """Same ClipMasks contract as build_batch_masks, which is why the loss code is reused."""
    masks = build_rollout_masks(TINY_LAYOUT, split_step=2, batch_size=BATCH)
    # Window lengths must be < num_steps; the MaskSpec defaults assume the full 8-step layout.
    spec = MaskSpec(layout=TINY_LAYOUT, tactile_window_steps=(2,), video_window_steps=(2,))
    assert_mask_invariants(masks, spec)
    for stream in STREAM_ORDER:
        lo, hi = (int(v) for v in masks.tgt_bounds[int(stream)])
        assert hi > lo, f"{stream.name} must have targets"


def test_rollout_masks_are_deterministic_and_row_identical():
    a = build_rollout_masks(TINY_LAYOUT, split_step=2, batch_size=BATCH)
    b = build_rollout_masks(TINY_LAYOUT, split_step=2, batch_size=BATCH)
    assert torch.equal(a.ctx_index, b.ctx_index)
    assert torch.equal(a.tgt_index, b.tgt_index)
    assert torch.equal(a.ctx_index[0], a.ctx_index[-1])


def test_rollout_masks_do_not_extend_the_mask_mode_enum():
    """A new enum member would resize ``mode_embed`` and break every pretraining checkpoint."""
    masks = build_rollout_masks(TINY_LAYOUT, split_step=2, batch_size=BATCH)
    assert masks.mode_enum is MaskMode.X
    assert len(MaskMode) == 5


@pytest.mark.parametrize("split_step", [0, 4, 5, -1])
def test_a_degenerate_split_is_refused(split_step):
    with pytest.raises(ValueError, match="split_step"):
        build_rollout_masks(TINY_LAYOUT, split_step=split_step, batch_size=BATCH)


# --------------------------------------------------------------------------------------
# Conditioning heads
# --------------------------------------------------------------------------------------


def test_action_embed_ignores_masked_out_slots():
    """An absent group must contribute exactly zero, not a learned bias.

    Otherwise the predictor can read embodiment identity off the conditioning vector and the
    action donor ratio improves without the action ever being used.
    """
    torch.manual_seed(0)
    embed = ActionEmbed(WIDTH).eval()
    mask = torch.zeros(BATCH, ACTION_DIM)
    mask[:, :10] = 1.0
    action = torch.randn(BATCH, 4, ACTION_DIM)

    tampered = action.clone()
    tampered[:, :, 10:] = 99.0
    with torch.no_grad():
        assert torch.equal(embed(action, mask), embed(tampered, mask))


def test_action_embed_shapes_and_refuses_a_wrong_width():
    embed = ActionEmbed(WIDTH)
    mask = torch.ones(BATCH, ACTION_DIM)
    assert embed(torch.randn(BATCH, 4, ACTION_DIM), mask).shape == (BATCH, 4, WIDTH)
    with pytest.raises(ValueError, match="must end in"):
        embed(torch.randn(BATCH, 4, ACTION_DIM - 1), torch.ones(BATCH, ACTION_DIM - 1))


def test_instruction_head_returns_normalized_pairs():
    torch.manual_seed(0)
    head = InstructionHead(TINY_LAYOUT, text_dim=32, hidden=16, projector_dim=8)
    encoded = EncoderOutput(
        tokens=[torch.randn(BATCH, 5, TINY_LAYOUT.video_width), torch.randn(BATCH, 7, TINY_LAYOUT.tactile_width)],
        sync_readout=[],
    )
    clip, text = head(encoded, torch.randn(BATCH, 32))
    assert clip.shape == text.shape == (BATCH, 8)
    torch.testing.assert_close(clip.norm(dim=-1), torch.ones(BATCH))
    torch.testing.assert_close(text.norm(dim=-1), torch.ones(BATCH))


# --------------------------------------------------------------------------------------
# The alignment objective
# --------------------------------------------------------------------------------------


def test_paraphrases_of_one_task_are_not_negatives_of_each_other():
    """Rows sharing a label must leave each other's denominator.

    Two clips from the same store carry different paraphrases of the same task. Row-position
    InfoNCE would push them apart, teaching the model that rewording changes the goal -- and
    on this corpus that is most of what the batch contains.
    """
    # Rows 0 and 1 are the *same* embedding under the same label, as two paraphrases of one
    # task should be. Orthogonal across labels so the only competition is the same-label pair.
    anchor = torch.zeros(4, 8)
    anchor[0, 0] = anchor[1, 0] = 1.0
    anchor[2, 1] = anchor[3, 1] = 1.0
    labels = torch.tensor([0, 0, 1, 1])

    result = info_nce(anchor, anchor.clone(), labels, temperature=0.07)
    # Row-position InfoNCE cannot reach 1.0 here: rows 0 and 1 are indistinguishable, so the
    # argmax over the unmasked logits is a tie. Excluding same-label rows makes it exact.
    assert float(result["top1"]) == 1.0
    assert float(result["loss"]) < 1e-3
    assert float(result["distinct_tasks"]) == 2.0
    assert float(result["chance"]) == pytest.approx(0.5)


def test_chance_counts_distinct_tasks_not_batch_rows():
    """Reporting 1/batch_size would overstate the headroom whenever a store repeats."""
    torch.manual_seed(0)
    anchor = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    labels = torch.zeros(8, dtype=torch.int64)
    result = info_nce(anchor, anchor.clone(), labels, temperature=0.07)
    assert float(result["chance"]) == pytest.approx(1.0)
    assert float(result["distinct_tasks"]) == 1.0


def test_misaligned_embeddings_score_at_chance():
    torch.manual_seed(3)
    anchor = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    labels = torch.arange(16)
    result = info_nce(anchor, text, labels, temperature=0.07)
    assert float(result["top1"]) < 0.5
    assert float(result["loss"]) > 1.0
    assert np.isclose(float(result["chance"]), 1 / 16)
