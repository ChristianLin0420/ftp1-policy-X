"""3-D rotary position embeddings with a shared time axis and private spatial axes.

Why this module exists at all, rather than an additive positional embedding:

The synchrony objective asks the model to match tactile at instant *t* to video at instant
*t*. If either stream carried a linearly readable absolute-time vector in its **value**
path, that match would be solvable by reading the code instead of the content -- and a
shuffled donor clip carries exactly the same code, so the shortcut would reproduce the
``shuffle == real`` failure the whole design exists to fix. RoPE rotates only queries and
keys; values are untouched. That is necessary but *not* sufficient, which is why the
positional-shortcut control probe is mandatory alongside it.

The head dimension is partitioned into three contiguous slices ``(t, h, w)``. The ``t``
slice uses one shared frequency table and the clip's tubelet-step index for *every* stream,
so two tokens observing the same instant in different modalities have zero relative time
phase. The ``h``/``w`` slices use a separate, much shorter-period base because spatial
extents are at most 14 patches while time should extrapolate to longer clips at probe time.

Tables are built once at construction into non-persistent buffers: no per-step
trigonometry, and -- critically -- no RNG and no data dependence, so a checkpointed
recompute cannot disagree with the original forward.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from openpi.mot_jepa.layout import TokenLayout


@dataclasses.dataclass(frozen=True)
class Rope3DConfig:
    """Head-dimension split and frequency bases.

    Args:
        head_dim: Attention head dimension. Must equal ``dim_t + dim_h + dim_w``.
        dim_t: Slots devoted to the shared time axis. Must be even.
        dim_h: Slots devoted to the private row axis. Must be even.
        dim_w: Slots devoted to the private column axis. Must be even.
        theta_t: Rotary base for time. Large, so long clips extrapolate.
        theta_hw: Rotary base for space. Small, matching the tiny patch grids.
    """

    head_dim: int = 64
    dim_t: int = 32
    dim_h: int = 16
    dim_w: int = 16
    theta_t: float = 10_000.0
    theta_hw: float = 100.0

    def __post_init__(self) -> None:
        if self.dim_t + self.dim_h + self.dim_w != self.head_dim:
            raise ValueError(
                f"dim_t+dim_h+dim_w={self.dim_t + self.dim_h + self.dim_w} must equal head_dim={self.head_dim}"
            )
        for name in ("dim_t", "dim_h", "dim_w"):
            value = getattr(self, name)
            if value % 2 != 0:
                raise ValueError(f"{name}={value} must be even (rotary acts on coordinate pairs)")
            if value <= 0:
                raise ValueError(f"{name}={value} must be positive")


def _axis_angles(positions: torch.Tensor, num_slots: int, theta: float) -> torch.Tensor:
    """``(n, num_slots // 2)`` rotation angles for one coordinate axis."""
    num_pairs = num_slots // 2
    exponents = torch.arange(num_pairs, dtype=torch.float64) / num_pairs
    inv_freq = theta ** (-exponents)
    return positions.to(torch.float64)[:, None] * inv_freq[None, :]


def build_rope_tables(config: Rope3DConfig, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(L, head_dim // 2)`` cosine and sine tables from ``(L, 3)`` integer coords.

    Pairs are interleaved *within* each axis slice, so the three axes never mix: slots
    ``[0, dim_t)`` are purely temporal, and so on. Computed in float64 and cast down once,
    so the tables do not depend on the autocast dtype in force at call time.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (L, 3), got {tuple(coords.shape)}")
    angles = torch.cat(
        [
            _axis_angles(coords[:, 0], config.dim_t, config.theta_t),
            _axis_angles(coords[:, 1], config.dim_h, config.theta_hw),
            _axis_angles(coords[:, 2], config.dim_w, config.theta_hw),
        ],
        dim=-1,
    )
    return torch.cos(angles).to(torch.float32), torch.sin(angles).to(torch.float32)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, H, L, head_dim)`` by per-token angles.

    Args:
        x: Queries or keys. Never values -- rotating values would put a linearly readable
            positional code into the residual stream, which is precisely the shortcut this
            design forbids.
        cos: ``(B, L, head_dim // 2)`` or ``(L, head_dim // 2)``.
        sin: Same shape as ``cos``.
    """
    if cos.ndim == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    else:
        cos = cos[:, None]
        sin = sin[:, None]

    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    return rotated.reshape(*x.shape)


class Rope3D(nn.Module):
    """Holds the RoPE tables for one :class:`TokenLayout` and gathers per-token rows."""

    cos_table: torch.Tensor
    sin_table: torch.Tensor

    def __init__(self, config: Rope3DConfig, layout: TokenLayout) -> None:
        super().__init__()
        self.config = config
        self.layout = layout
        cos, sin = build_rope_tables(config, layout.coords)
        # Non-persistent: derivable from the layout, so it never bloats a checkpoint and
        # never goes stale when the progressive-resolution schedule changes the grid.
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def gather(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows for the given token indices.

        Args:
            index: ``(L_sel,)`` or ``(B, L_sel)`` int64 token indices.

        Returns:
            ``cos, sin`` broadcast-compatible with :func:`apply_rope`.
        """
        flat = index.reshape(-1)
        cos = self.cos_table.index_select(0, flat).reshape(*index.shape, -1)
        sin = self.sin_table.index_select(0, flat).reshape(*index.shape, -1)
        return cos, sin

    def forward(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gather(index)

    def extra_repr(self) -> str:
        cfg = self.config
        return f"head_dim={cfg.head_dim}, dim_t={cfg.dim_t}, dim_h={cfg.dim_h}, dim_w={cfg.dim_w}"


def relative_phase(cos: torch.Tensor, sin: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Per-pair relative rotation angle between tokens ``i`` and ``j``.

    Exists for the tests: two tokens at the same instant in different streams must have a
    relative *time* phase of exactly zero, which is the property the synchrony objective
    depends on.
    """
    angle_i = torch.atan2(sin[i], cos[i])
    angle_j = torch.atan2(sin[j], cos[j])
    return angle_i - angle_j
