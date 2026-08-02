from __future__ import annotations

import dataclasses

import torch

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


@dataclasses.dataclass(frozen=True)
class AttentionLayout:
    att_2d_masks: torch.Tensor
    position_ids: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ExpertAttentionLayout(AttentionLayout):
    prefix_pad_masks: torch.Tensor
    tactile_pad_masks: torch.Tensor | None
    suffix_pad_masks: torch.Tensor

def _build_branch_self_mask(pad_masks: torch.Tensor, att_masks: torch.Tensor | None = None) -> torch.Tensor:
    if att_masks is None:
        return pad_masks[:, :, None] * pad_masks[:, None, :]
    return make_att_2d_masks(pad_masks, att_masks)


def build_prefix_attention_layout(prefix_pad_masks: torch.Tensor, prefix_att_masks: torch.Tensor) -> AttentionLayout:
    return AttentionLayout(
        att_2d_masks=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
        position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
    )


def build_tactile_attention_layout(
    tactile_pad_masks: torch.Tensor,
    *,
    position_offset: torch.Tensor | None = None,
    prefix_pad_masks: torch.Tensor | None = None,
    num_image_tokens: int = 0,
) -> AttentionLayout:
    """Attention layout for the tactile prefill.

    By default the tactile branch is self-attention only, which is FTP-1's behaviour. When
    `prefix_pad_masks` is given and `num_image_tokens > 0` the layout is widened to
    ``(B, tactile_len, prefix_len + tactile_len)`` so the tactile prefill can read the image
    columns of an existing VLM KV cache. Language columns stay closed.
    """
    position_ids = torch.cumsum(tactile_pad_masks, dim=1) - 1
    if position_offset is not None:
        position_ids = position_ids + position_offset

    self_block = tactile_pad_masks[:, :, None] * tactile_pad_masks[:, None, :]
    if prefix_pad_masks is None or num_image_tokens <= 0:
        return AttentionLayout(att_2d_masks=self_block, position_ids=position_ids)

    prefix_cols = torch.zeros(
        tactile_pad_masks.shape[0],
        tactile_pad_masks.shape[1],
        prefix_pad_masks.shape[1],
        dtype=self_block.dtype,
        device=self_block.device,
    )
    img_cols = slice(0, num_image_tokens)
    prefix_cols[:, :, img_cols] = (
        tactile_pad_masks[:, :, None] * prefix_pad_masks[:, None, img_cols]
    )
    return AttentionLayout(
        att_2d_masks=torch.cat([prefix_cols, self_block], dim=2),
        position_ids=position_ids,
    )


def build_expert_attention_layout(
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    *,
    tactile_pad_masks: torch.Tensor | None = None,
    num_image_tokens: int = 0,
) -> ExpertAttentionLayout:
    prefix_block = _build_branch_self_mask(prefix_pad_masks)
    suffix_block = _build_branch_self_mask(suffix_pad_masks, suffix_att_masks)
    if tactile_pad_masks is not None:
        tactile_block = _build_branch_self_mask(tactile_pad_masks)
    else:
        tactile_block = None

    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    tactile_len = 0 if tactile_pad_masks is None else tactile_pad_masks.shape[1]
    suffix_len = suffix_pad_masks.shape[1]
    total_len = prefix_len + tactile_len + suffix_len

    att_2d_masks = torch.zeros(batch_size, total_len, total_len, dtype=torch.bool, device=prefix_pad_masks.device)
    att_2d_masks[:, :prefix_len, :prefix_len] = prefix_block

    tactile_start = prefix_len
    if tactile_block is not None and tactile_pad_masks is not None:
        att_2d_masks[:, tactile_start : tactile_start + tactile_len, tactile_start : tactile_start + tactile_len] = (
            tactile_block
        )

    action_start = prefix_len + tactile_len
    att_2d_masks[:, action_start:, action_start:] = suffix_block

    # [PACT] Vision -> tactile edge. FTP-1 leaves this block at zero, so image and tactile
    # tokens never attend to each other at any layer and meet only as independent K/V for the
    # action expert. Opening the image columns (and only the image columns) lets tactile read
    # vision while vision stays blind to tactile, which keeps the prefix KV cache valid and
    # keeps a fully gated batch bit-identical to a vision-language-only policy.
    if tactile_pad_masks is not None and num_image_tokens > 0:
        tactile_rows = slice(tactile_start, tactile_start + tactile_len)
        img_cols = slice(0, num_image_tokens)
        att_2d_masks[:, tactile_rows, img_cols] = (
            tactile_pad_masks[:, :, None] * prefix_pad_masks[:, None, img_cols]
        )

    action_rows = slice(action_start, action_start + suffix_pad_masks.shape[1])
    vlm_cols = slice(0, prefix_pad_masks.shape[1])
    att_2d_masks[:, action_rows, vlm_cols] = suffix_pad_masks[:, :, None] * prefix_pad_masks[:, None, :]
    if tactile_pad_masks is not None:
        tactile_cols = slice(tactile_start, tactile_start + tactile_pad_masks.shape[1])
        att_2d_masks[:, action_rows, tactile_cols] = suffix_pad_masks[:, :, None] * tactile_pad_masks[:, None, :]

    pad_masks = [prefix_pad_masks]
    if tactile_pad_masks is not None:
        pad_masks.append(tactile_pad_masks)
    pad_masks.append(suffix_pad_masks)
    position_ids = torch.cumsum(torch.cat(pad_masks, dim=1), dim=1) - 1

    return ExpertAttentionLayout(
        att_2d_masks=att_2d_masks,
        position_ids=position_ids,
        prefix_pad_masks=prefix_pad_masks,
        tactile_pad_masks=tactile_pad_masks,
        suffix_pad_masks=suffix_pad_masks,
    )


def build_action_static_prefill_layout(
    prefix_pad_masks: torch.Tensor,
    static_suffix_pad_masks: torch.Tensor,
    static_suffix_att_masks: torch.Tensor,
) -> AttentionLayout:
    static_to_prefix = static_suffix_pad_masks[:, :, None] * prefix_pad_masks[:, None, :]
    static_to_static = make_att_2d_masks(static_suffix_pad_masks, static_suffix_att_masks)
    return AttentionLayout(
        att_2d_masks=torch.cat([static_to_prefix, static_to_static], dim=2),
        position_ids=torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(static_suffix_pad_masks, dim=1) - 1,
    )


def build_action_denoise_layout(
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    *,
    tactile_pad_masks: torch.Tensor | None = None,
    cached_static_suffix_pad_masks: torch.Tensor | None = None,
) -> AttentionLayout:
    action_to_vlm = suffix_pad_masks[:, :, None] * prefix_pad_masks[:, None, :]
    action_to_action = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    if tactile_pad_masks is not None:
        action_to_tactile = suffix_pad_masks[:, :, None] * tactile_pad_masks[:, None, :]
        att_2d_masks = torch.cat([action_to_vlm, action_to_tactile, action_to_action], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.sum(tactile_pad_masks, dim=-1)[:, None]
    elif cached_static_suffix_pad_masks is not None:
        action_to_static = suffix_pad_masks[:, :, None] * cached_static_suffix_pad_masks[:, None, :]
        att_2d_masks = torch.cat([action_to_vlm, action_to_static, action_to_action], dim=2)
        prefix_offsets = (
            torch.sum(prefix_pad_masks, dim=-1)[:, None]
            + torch.sum(
                cached_static_suffix_pad_masks,
                dim=-1,
            )[:, None]
        )
    else:
        att_2d_masks = torch.cat([action_to_vlm, action_to_action], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]

    return AttentionLayout(
        att_2d_masks=att_2d_masks,
        position_ids=prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1,
    )
