"""Trainability and DDP contract for supervised MoT-JEPA policy adaptation.

The policy head and backbone must participate in one DDP forward. Updating a backbone outside the
module wrapped by DDP produces one independently drifting encoder per rank while the loss and job
still look healthy. This module keeps that load-bearing wiring small enough to unit-test directly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib

import torch
from torch import nn

from openpi.mot_jepa.action_dit import flow_matching_loss
from openpi.mot_jepa.action_dit import regression_loss
from openpi.mot_jepa.drifting import drifting_loss

BACKBONE_TRAIN_MODES = ("frozen", "last_blocks", "full")

# Neither the policy nor pretraining ``to_inputs`` path currently supplies sensor type ids, so
# these embeddings are outside the executed graph. Leaving them trainable under DDP with
# find_unused_parameters=False fails on the next iteration. They have no effect on UniVTAC output,
# so "full" means every parameter reachable under the current input contract and stamps these two
# exclusions in checkpoint metadata.
UNREACHABLE_SENSOR_PARAMETERS = (
    "embed.gel.sensor_embed.weight",
    "embed.lowdim.sensor_embed.weight",
)


@dataclasses.dataclass(frozen=True)
class BackboneSelection:
    mode: str
    names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]
    excluded_unreachable: tuple[str, ...]

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters)

    @property
    def name_digest(self) -> str:
        payload = "\n".join(self.names).encode()
        return hashlib.sha256(payload).hexdigest()


def configure_backbone_trainability(backbone: nn.Module, *, mode: str, last_n_blocks: int) -> BackboneSelection:
    """Apply one explicit trainability mode and return its stable parameter manifest."""
    if mode not in BACKBONE_TRAIN_MODES:
        raise ValueError(f"unknown backbone train mode {mode!r}; expected one of {BACKBONE_TRAIN_MODES}")

    backbone.requires_grad_(requires_grad=False)
    excluded: tuple[str, ...] = ()
    if mode == "frozen":
        backbone.eval()
    elif mode == "last_blocks":
        blocks = backbone.encoder.blocks
        if not 0 < last_n_blocks <= len(blocks):
            raise ValueError(f"last_n_blocks={last_n_blocks} must be in [1, {len(blocks)}]")
        for block in blocks[-last_n_blocks:]:
            block.requires_grad_(requires_grad=True)
        backbone.encoder.norms.requires_grad_(requires_grad=True)
        backbone.train()
    else:
        backbone.requires_grad_(requires_grad=True)
        named = dict(backbone.named_parameters())
        excluded = tuple(name for name in UNREACHABLE_SENSOR_PARAMETERS if name in named)
        for name in excluded:
            named[name].requires_grad_(requires_grad=False)
        backbone.train()

    selected = tuple((name, parameter) for name, parameter in backbone.named_parameters() if parameter.requires_grad)
    return BackboneSelection(
        mode=mode,
        names=tuple(name for name, _ in selected),
        parameters=tuple(parameter for _, parameter in selected),
        excluded_unreachable=excluded,
    )


def policy_step_seed(base_seed: int, step: int, rank: int) -> int:
    """Stable, non-overlapping seed for one optimizer step and rank."""
    if step < 0 or rank < 0:
        raise ValueError(f"step and rank must be non-negative, got {step}/{rank}")
    modulus = 2**63 - 1
    return (int(base_seed) * 1_000_003 + int(step) * 9_176_593 + int(rank) * 104_729) % modulus


def make_step_generator(device: torch.device, *, base_seed: int, step: int, rank: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(policy_step_seed(base_seed, step, rank))
    return generator


class PolicyTrainModel(nn.Module):
    """Backbone plus head behind one DDP reducer and one forward/loss boundary."""

    def __init__(self, backbone: nn.Module, head: nn.Module, *, objective: str, drifting_config, train_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.objective = objective
        self.drifting_config = drifting_config
        self.train_backbone = bool(train_backbone)

    def forward(
        self,
        inputs,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        domain_id: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        grad_context = contextlib.nullcontext() if self.train_backbone else torch.no_grad()
        device_type = inputs.video.device.type
        with grad_context, torch.autocast(device_type, torch.bfloat16, enabled=device_type == "cuda"):
            encoded = self.backbone.encode_full(inputs)

        # The established head path is FP32. Keep it so the frozen arm remains numerically
        # comparable while BF16 encoder activations still carry gradients in adaptation modes.
        encoded = type(encoded)(
            tokens=[tensor.float() for tensor in encoded.tokens],
            sync_readout=[tensor.float() for tensor in encoded.sync_readout],
            final_readout=[tensor.float() for tensor in encoded.final_readout],
        )
        if self.objective == "drifting":
            return drifting_loss(
                self.head,
                encoded,
                actions,
                action_mask,
                chunk_mask,
                domain_id,
                config=self.drifting_config,
                generator=generator,
            )
        if self.objective == "linear":
            return regression_loss(self.head, encoded, actions, action_mask, chunk_mask, domain_id, generator=generator)
        if self.objective == "flowmatch":
            return flow_matching_loss(
                self.head, encoded, actions, action_mask, chunk_mask, domain_id, generator=generator
            )
        raise ValueError(f"unknown policy objective {self.objective!r}")


def grad_norm(
    parameters: tuple[nn.Parameter, ...] | list[nn.Parameter],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """L2 norm before clipping; an empty or not-yet-reached group reports exact zero."""
    grads = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if grads:
        return torch.stack(grads).norm(2)
    output_device = parameters[0].device if parameters else (device or torch.device("cpu"))
    return torch.zeros((), device=output_device)
