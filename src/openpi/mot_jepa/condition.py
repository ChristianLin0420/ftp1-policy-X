"""Conditioning heads for the two post-training stages.

The pretrained backbone is **frozen** in both stages, so everything that trains lives here.
That is the whole reason this is a separate module rather than edits to ``model.py``: Stage 3
and Stage 4 add no capacity to the artifact the probes measure, so a binding result measured
after pretraining is still true after post-training, and re-running P1/P3 is a *check* rather
than a re-measurement.

**Stage 3 -- clip<->language alignment.** ``sub_task_instruction`` turned out to be constant
within an episode and to consist of paraphrases of one task per store, so there is no
within-task instruction variation that could select a different future. Conditioned future
prediction therefore has nothing to learn from, and the objective is alignment instead:
positives are paraphrases of the same task, negatives are other tasks in the same domain.
:class:`InstructionHead` is the entire trainable surface.

**Stage 4 -- action conditioning.** :class:`ActionEmbed` turns a per-step action into a vector
the predictor adds to its mask tokens, at that token's own tubelet step. The predictor gained
one optional ``cond`` argument for this; nothing else in the model changed.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.mot_jepa.action_parse import ACTION_DIM
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.mot_encoder import EncoderOutput


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def pool_experts(encoded: EncoderOutput) -> torch.Tensor:
    """``(B, video_width + tactile_width)`` -- mean over tokens, concatenated per expert.

    Concatenated rather than summed so the alignment head can weight the two modalities
    differently; the tactile-zeroed control then reads directly as "how much of the language
    grounding came from touch".
    """
    return torch.cat([tokens.mean(dim=1) for tokens in encoded.tokens], dim=-1)


class InstructionHead(nn.Module):
    """Projects a pooled clip and a frozen instruction embedding into one space.

    The instruction embeddings are precomputed offline with the repository's own Gemma, so no
    text model runs during training -- this is a frozen table lookup plus two small MLPs.
    """

    def __init__(
        self,
        layout: TokenLayout,
        *,
        text_dim: int = 2048,
        hidden: int = 1024,
        projector_dim: int = 256,
    ) -> None:
        super().__init__()
        clip_dim = layout.video_width + layout.tactile_width
        self.clip_proj = _mlp(clip_dim, hidden, projector_dim)
        self.text_proj = _mlp(text_dim, hidden, projector_dim)

    def forward(self, encoded: EncoderOutput, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns L2-normalized ``(B, projector_dim)`` clip and text embeddings."""
        clip = F.normalize(self.clip_proj(pool_experts(encoded)).to(torch.float32), dim=-1)
        text = F.normalize(self.text_proj(text.to(self.text_proj[0].weight.dtype)).to(torch.float32), dim=-1)
        return clip, text


class ActionEmbed(nn.Module):
    """``(B, num_steps, 120)`` masked actions -> ``(B, num_steps, width)`` predictor conditioning.

    The mask is applied *before* the projection, not after. An absent group -- a domain with
    no arm joints, say -- must contribute exactly zero rather than a learned bias, or the
    predictor could read embodiment identity off the conditioning vector and the action donor
    ratio would improve without the action ever being used.
    """

    def __init__(self, width: int, *, hidden: int = 512, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.proj = _mlp(action_dim, hidden, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, action: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"action must end in {self.action_dim} dims, got {tuple(action.shape)}")
        masked = action * action_mask.to(action.dtype).unsqueeze(1)
        return self.norm(self.proj(masked))


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, labels: torch.Tensor, *, temperature: float) -> dict:
    """Symmetric InfoNCE where *labels* -- not row position -- define the positives.

    Row-position InfoNCE is wrong here. Two clips from the same store carry different
    paraphrases of the same task, and pushing those apart would teach the model that
    rewording changes the goal. Rows sharing a label are therefore excluded from each other's
    negatives rather than treated as hard negatives.
    """
    logits = anchor @ positive.t() / temperature
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    # Mask other same-label rows out of the denominator, keeping each row's own positive.
    logits = logits.masked_fill(same & ~diagonal, float("-inf"))

    target = torch.arange(labels.shape[0], device=labels.device)
    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    with torch.no_grad():
        top1 = (logits.argmax(dim=-1) == target).float().mean()
        # Chance is 1 / (number of distinct tasks in the batch), not 1 / batch_size: rows
        # sharing a label were removed from the denominator above.
        distinct = torch.unique(labels).numel()
        chance = 1.0 / max(distinct, 1)
    return {
        "loss": loss,
        "top1": top1.detach(),
        "chance": torch.tensor(chance, device=labels.device),
        "distinct_tasks": torch.tensor(float(distinct), device=labels.device),
    }
