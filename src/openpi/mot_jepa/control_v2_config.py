"""Frozen run and deployable-artifact configuration for MoT-Control V3."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

from openpi.mot_jepa.config import DataConfig
from openpi.mot_jepa.config import _from_dict
from openpi.mot_jepa.control_v2 import ControlV2Config
from openpi.mot_jepa.control_v2_dataset import CONTROL_V2_SPLIT_IDENTITY
from openpi.mot_jepa.layout import LAYOUT_PILOT
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.predictor import MoTPredictorConfig
from openpi.mot_jepa.rope3d import Rope3DConfig

CONTROL_V2_ARTIFACT_SCHEMA = 4
CONTROL_V2_PRODUCTION_EPISODES = 1_000
_ROPE = Rope3DConfig(head_dim=64, dim_t=32, dim_h=16, dim_w=16)


@dataclasses.dataclass(frozen=True)
class ControlV2TrainConfig:
    """One immutable V3 training stage.

    Head-only, final-four adaptation, and recovery fine-tuning are separate completed runs.  That
    keeps optimizer/DDP topology stable within a run and makes every stage independently auditable.
    """

    name: str = "mot_jepa_control_v2"
    exp_name: str = "lift_bottle_head20k"
    run_root: str = ".cache/mot_jepa/runs"
    task: str = "lift_bottle"

    pretrained_run: str = ""
    pretrained_step: int = 100_000
    init_run: str = ""
    init_step: int | None = None
    store_glob: str = ""
    split_identity: str = CONTROL_V2_SPLIT_IDENTITY
    split_seed: int = 42
    validation_fraction: float = 0.15
    require_authoritative_control: bool = True
    minimum_total_episodes: int = 1_000

    observation_stride: int = 2
    action_stride: int = 1
    # Deployment replans at every simulator control step.  Training must cover the same anchors;
    # subsampling starts here silently leaves three of four deployment offsets out of distribution.
    index_step: int = 1
    video_size: int = 224
    gel_size: int = 112
    lowdim_channels: int = 48
    encoder: MoTEncoderConfig = dataclasses.field(
        default_factory=lambda: MoTEncoderConfig(
            depth=12,
            num_local_layers=4,
            num_heads=6,
            head_dim=64,
            rope=_ROPE,
        )
    )
    # ``runtime.load_frozen_backbone`` reconstructs the complete pretraining student before
    # extracting its EMA backbone, so the otherwise-unused predictor/data sections remain part of
    # this run contract.
    predictor: MoTPredictorConfig = dataclasses.field(
        default_factory=lambda: MoTPredictorConfig(
            depth=6,
            width=192,
            num_heads=3,
            head_dim=64,
            rope=_ROPE,
        )
    )
    data: DataConfig = dataclasses.field(
        default_factory=lambda: DataConfig(
            # These are the qualified 100k backbone's embedding/pretraining settings.  V3
            # control sampling is specified by the direct fields above; keeping the two
            # contracts separate lets runtime reconstruct the source architecture exactly.
            strides=(2, 4),
            action_stride=None,
            index_step=1,
            num_workers=12,
            prefetch_factor=4,
            lowdim_channels=48,
            lowdim_log_compress=True,
        )
    )
    head: ControlV2Config = dataclasses.field(default_factory=ControlV2Config)

    backbone_train_mode: str = "frozen"
    backbone_last_n_blocks: int = 4
    gradient_checkpointing: bool = True
    num_train_steps: int = 20_000
    local_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    lr_end_ratio: float = 0.05
    warmup_steps: int = 500
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad_norm: float = 1.0
    overlap_loss_weight: float = 0.5
    anchor_loss_weight: float = 0.1
    order_batch_fraction: float = 0.25
    safety_loss_weight: float = 0.25
    # The normalized SmoothL1 tube objective includes both mean and per-example maximum error.
    # A 0.05 weight keeps it auxiliary to the primary chunk and phase/contact objectives.
    safety_tube_loss_weight: float = 0.05
    # The artifact rate limit is fitted at 110% of the demonstrated maximum.  Penalising above
    # 90% of that limit preserves roughly a 1% buffer beyond the observed expert maximum while
    # still allowing all but the single extremal transition to fit without a safety penalty.
    safety_rate_margin: float = 0.9

    log_interval: int = 25
    validation_interval: int = 1_000
    # 512 nominal examples at local batch two cover every ~150 held-out episode and allocate
    # roughly one early/middle/late anchor per episode while cycling all 32 cold-start scenarios.
    validation_batches: int = 256
    save_interval: int = 1_000
    keep_last: int = 3
    keep_period: int | None = 5_000
    num_workers: int = 6
    prefetch_factor: int = 3
    seed: int = 42
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if self.task != "lift_bottle":
            raise ValueError("Control V3 is intentionally task-specific to lift_bottle")
        if self.pretrained_step != 100_000:
            raise ValueError("Control V3 must start from the qualified 100k MoT backbone")
        if self.backbone_train_mode not in {"frozen", "last_blocks"}:
            raise ValueError("backbone_train_mode must be frozen or last_blocks; full adaptation is disqualified")
        if not 1 <= self.backbone_last_n_blocks <= self.encoder.depth:
            raise ValueError("backbone_last_n_blocks is outside encoder depth")
        if self.observation_stride != 2 or self.action_stride != 1:
            raise ValueError("the qualified V3 cadence is observation/action=2/1")
        if self.data.lowdim_channels != self.lowdim_channels:
            raise ValueError("source-backbone and control lowdim channel widths differ")
        if self.video_size != 224 or self.gel_size != 112:
            raise ValueError("V3 uses official-resolution 224 video and 112 GEL")
        if self.head.horizon != 32 or self.head.qpos_history != 16:
            raise ValueError("V3 requires H32 and a 16-frame state history")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.split_identity != CONTROL_V2_SPLIT_IDENTITY:
            raise ValueError(f"split_identity must be {CONTROL_V2_SPLIT_IDENTITY!r}")
        if self.minimum_total_episodes < 1:
            raise ValueError("minimum_total_episodes must be positive")
        if self.require_authoritative_control and self.minimum_total_episodes != CONTROL_V2_PRODUCTION_EPISODES:
            raise ValueError(
                f"authoritative production V3 must require exactly {CONTROL_V2_PRODUCTION_EPISODES} episodes"
            )
        for value, name in (
            (self.num_train_steps, "num_train_steps"),
            (self.local_batch_size, "local_batch_size"),
            (self.gradient_accumulation_steps, "gradient_accumulation_steps"),
            (self.validation_interval, "validation_interval"),
            (self.validation_batches, "validation_batches"),
            (self.save_interval, "save_interval"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_train_steps < self.save_interval:
            raise ValueError("num_train_steps must reach at least one periodic checkpoint")
        if not 0 < self.lr_end_ratio <= 1:
            raise ValueError("lr_end_ratio must be in (0, 1]")
        if min(self.lr_head, self.lr_backbone, self.clip_grad_norm) <= 0:
            raise ValueError("learning rates and clip_grad_norm must be positive")
        if (
            min(
                self.overlap_loss_weight,
                self.anchor_loss_weight,
                self.safety_loss_weight,
                self.safety_tube_loss_weight,
            )
            < 0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")
        if not 0 <= self.order_batch_fraction <= 1:
            raise ValueError("order_batch_fraction must be in [0, 1]")
        if not 0 < self.safety_rate_margin <= 1:
            raise ValueError("safety_rate_margin must be in (0, 1]")

    @property
    def layout(self) -> TokenLayout:
        # Patch kernels and expert widths remain checkpoint-compatible with the 100k pilot.  RoPE
        # makes the larger spatial grid parameter-free.
        return dataclasses.replace(LAYOUT_PILOT, video_size=self.video_size, gel_size=self.gel_size)

    @property
    def run_dir(self) -> pathlib.Path:
        return pathlib.Path(self.run_root) / self.name / self.exp_name

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self.run_dir / "checkpoints"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ControlV2TrainConfig:
        return _from_dict(cls, json.loads(text))


@dataclasses.dataclass(frozen=True)
class ControlV2Artifact:
    """Small JSON contract required beside every deployable V3 checkpoint."""

    schema_version: int
    data_schema: str
    task: str
    context_mode: str
    state_mode: str
    target_mode: str
    horizon: int
    first_executable_index: int
    chunk_first_n: int
    temporal_ensemble_k: float
    observation_stride: int
    action_stride: int
    control_step_stride: int
    phase_names: tuple[str, ...]
    action_scale: tuple[float, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    max_command_delta: tuple[float, ...]
    authoritative_control: bool
    total_episodes: int
    train_episodes: int
    validation_episodes: int
    phase_counts: tuple[int, ...]
    contact_counts: tuple[int, ...]
    source_stores: tuple[str, ...]
    source_store_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != CONTROL_V2_ARTIFACT_SCHEMA:
            raise ValueError(f"schema_version must be {CONTROL_V2_ARTIFACT_SCHEMA}")
        expected = {
            "data_schema": "mot_jepa_control_v2_data_v2",
            "task": "lift_bottle",
            "context_mode": "dense_video_gel",
            "state_mode": "qpos8_history_command",
            "target_mode": "next_command_chunk_relative_mix8",
            "horizon": 32,
            "first_executable_index": 1,
            "chunk_first_n": 20,
            "observation_stride": 2,
            "action_stride": 1,
            "control_step_stride": 1,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} must be {value!r}, got {getattr(self, name)!r}")
        if self.temporal_ensemble_k != 0.01:
            raise ValueError("temporal_ensemble_k must be 0.01")
        if self.phase_names != ("approach", "close", "lift", "release"):
            raise ValueError("phase_names differ from the V3 four-phase contract")
        for name in ("action_scale", "joint_lower", "joint_upper", "max_command_delta"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 8:
                raise ValueError(f"{name} must contain eight values")
            if name in {"action_scale", "max_command_delta"} and min(values) <= 0:
                raise ValueError(f"{name} values must be positive")
        if any(lo >= hi for lo, hi in zip(self.joint_lower, self.joint_upper, strict=True)):
            raise ValueError("every joint lower limit must be below its upper limit")
        if self.train_episodes < 1 or self.validation_episodes < 1 or not self.source_stores:
            raise ValueError("artifact must identify non-empty train/validation episodes and source stores")
        if self.total_episodes != self.train_episodes + self.validation_episodes:
            raise ValueError("total_episodes must equal train_episodes + validation_episodes")
        if len(self.phase_counts) != 4 or any(count <= 0 for count in self.phase_counts):
            raise ValueError("phase_counts must contain a positive count for each of the four phases")
        if len(self.contact_counts) != 2 or any(count <= 0 for count in self.contact_counts):
            raise ValueError("contact_counts must contain positive absent/contact counts")
        if len(self.source_store_sha256) != 64:
            raise ValueError("source_store_sha256 must be a SHA-256 hex digest")

    @property
    def production_data_qualified(self) -> bool:
        """Whether this artifact closes the preregistered V3 data gate."""

        return self.authoritative_control and self.total_episodes == CONTROL_V2_PRODUCTION_EPISODES

    def require_production_data(self) -> None:
        if not self.production_data_qualified:
            raise ValueError(
                "deployable V3 requires authoritative command/phase/contact labels and exactly 1000 episodes"
            )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> ControlV2Artifact:
        payload: dict[str, Any] = json.loads(text)
        for name in (
            "phase_names",
            "action_scale",
            "joint_lower",
            "joint_upper",
            "max_command_delta",
            "phase_counts",
            "contact_counts",
            "source_stores",
        ):
            if name in payload:
                payload[name] = tuple(payload[name])
        return cls(**payload)


def control_v2_cli() -> ControlV2TrainConfig:
    import tyro  # noqa: PLC0415

    return tyro.cli(ControlV2TrainConfig)
