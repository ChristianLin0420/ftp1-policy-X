"""Training configuration, frozen per run.

The config is written once to ``run_dir/run_config.json`` and re-read on every subsequent
launch. That matters because a 4-hour walltime means a long run is a chain of ~40 requeued
jobs: without pinning, editing the launcher on day three would silently change
hyperparameters mid-run and the W&B history would be a splice of two different experiments.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import typing

import tyro

from openpi.mot_jepa.layout import LAYOUT_BASE
from openpi.mot_jepa.layout import LAYOUT_PILOT
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.losses import LossConfig
from openpi.mot_jepa.masking import DEFAULT_MODE_PROBS
from openpi.mot_jepa.mot_encoder import MoTEncoderConfig
from openpi.mot_jepa.predictor import MoTPredictorConfig
from openpi.mot_jepa.rope3d import Rope3DConfig


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Where clips come from."""

    domain_config: str = ""
    """Path to an FTP-1 domain-config JSON (``dataset_zarr.py:2409-2427`` schema)."""
    store_glob: str = ""
    """Alternative to ``domain_config``: a glob matching ``*.zarr`` stores directly."""
    strides: tuple[int, ...] = (1, 2)
    index_step: int = 1
    """Stride between enumerated clip starts. Raise it to shrink the index on huge corpora."""
    num_workers: int = 12
    prefetch_factor: int = 4
    lowdim_channels: int = 48
    """Widest low-dim unit in the release is uSkin at 4x4x3; a 6-D wrench uses 6."""


@dataclasses.dataclass(frozen=True)
class EmaConfig:
    decay_start: float = 0.998
    decay_end: float = 0.99999
    warmup_steps: int = 30_000
    sync_check_interval: int = 5_000
    """How often ranks compare shadow checksums; a silent divergence is otherwise invisible."""


@dataclasses.dataclass(frozen=True)
class MaskConfig:
    mode_probs: tuple[float, ...] = DEFAULT_MODE_PROBS
    video_mask_frac: float = 0.88
    x_video_mask_frac: float = 0.75
    tactile_window_steps: tuple[int, ...] = (2, 3, 4)
    video_window_steps: tuple[int, ...] = (2, 3, 4)
    min_targets_per_stream: int = 8


@dataclasses.dataclass(frozen=True)
class MotJepaTrainConfig:
    """One pretraining run."""

    name: str = "mot_jepa_pilot"
    exp_name: str = "dev"
    project_name: str = "mot-jepa"
    run_root: str = ".cache/mot_jepa/runs"

    layout_preset: str = "pilot"
    """``pilot`` or ``base``. Selects the token budget; see ``layout.py``."""
    encoder: MoTEncoderConfig = dataclasses.field(default_factory=MoTEncoderConfig)
    predictor: MoTPredictorConfig = dataclasses.field(default_factory=MoTPredictorConfig)
    masking: MaskConfig = dataclasses.field(default_factory=MaskConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    ema: EmaConfig = dataclasses.field(default_factory=EmaConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)

    seed: int = 42
    local_batch_size: int = 8
    num_train_steps: int = 200_000
    lr_peak: float = 1.5e-3
    lr_end: float = 1e-6
    lr_warmup_steps: int = 8_000
    weight_decay: float = 0.04
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad_norm: float = 1.0

    log_interval: int = 50
    probe_interval: int = 2_000
    save_interval: int = 500
    """Deliberately frequent: a 4-hour job that is preempted loses everything since the last
    checkpoint, and the save-on-signal path is a backstop rather than a guarantee."""
    keep_last: int = 3
    keep_period: int | None = 20_000
    """Real retention. ``config.keep_period`` is a no-op in the FTP-1 PyTorch saver, which is
    why a 20k-step run there retains every periodic checkpoint forever."""

    gradient_checkpointing: bool = False
    wandb_enabled: bool = True
    compile_model: bool = False
    find_unused_parameters: bool = True
    """Required: mask mode ``T_HARD`` removes tactile entirely, so the student's tactile
    encoder legitimately receives no gradient on ~10% of steps."""

    @property
    def layout(self) -> TokenLayout:
        presets = {"pilot": LAYOUT_PILOT, "base": LAYOUT_BASE}
        if self.layout_preset not in presets:
            raise ValueError(f"unknown layout_preset={self.layout_preset!r}, expected one of {sorted(presets)}")
        return presets[self.layout_preset]

    @property
    def run_dir(self) -> pathlib.Path:
        return pathlib.Path(self.run_root) / self.name / self.exp_name

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self.run_dir / "checkpoints"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> MotJepaTrainConfig:
        return _from_dict(cls, json.loads(text))

    def diff(self, other: MotJepaTrainConfig) -> dict[str, tuple[object, object]]:
        """Field-by-field differences, used to warn loudly on a config drift after requeue."""
        mine, theirs = dataclasses.asdict(self), dataclasses.asdict(other)
        return {key: (mine[key], theirs[key]) for key in mine if mine[key] != theirs[key]}


def _from_dict(cls: type, payload: dict) -> object:
    """Rebuild a nested frozen dataclass, restoring tuples that JSON turned into lists.

    Annotations are resolved with ``get_type_hints`` rather than read off ``field.type``:
    this module uses ``from __future__ import annotations``, so ``field.type`` is the *string*
    ``"DataConfig"``, and testing it with ``dataclasses.is_dataclass`` silently returns False.
    That would leave every nested section as a raw dict and only surface much later as an
    ``AttributeError`` deep in the trainer.
    """
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in payload:
            continue
        value = payload[field.name]
        field_type = hints.get(field.name, field.type)
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[field.name] = _from_dict(field_type, value)
        elif isinstance(value, list):
            kwargs[field.name] = tuple(value)
        else:
            kwargs[field.name] = value
    return cls(**kwargs)


_PILOT_ROPE = Rope3DConfig(head_dim=64, dim_t=32, dim_h=16, dim_w=16)

#: Named presets. ``debug`` is CPU-runnable and used by the smoke path.
CONFIGS: dict[str, MotJepaTrainConfig] = {
    "mot_jepa_debug": MotJepaTrainConfig(
        name="mot_jepa_debug",
        layout_preset="pilot",
        encoder=MoTEncoderConfig(depth=2, num_local_layers=1, num_heads=6, head_dim=64, rope=_PILOT_ROPE),
        predictor=MoTPredictorConfig(depth=1, width=192, num_heads=3, head_dim=64, rope=_PILOT_ROPE),
        num_train_steps=20,
        local_batch_size=2,
        lr_warmup_steps=2,
        save_interval=10,
        log_interval=1,
        probe_interval=10,
        wandb_enabled=False,
        data=DataConfig(num_workers=0),
    ),
    "mot_jepa_pilot": MotJepaTrainConfig(
        name="mot_jepa_pilot",
        layout_preset="pilot",
        encoder=MoTEncoderConfig(depth=12, num_local_layers=4, num_heads=6, head_dim=64, rope=_PILOT_ROPE),
        predictor=MoTPredictorConfig(depth=6, width=192, num_heads=3, head_dim=64, rope=_PILOT_ROPE),
        num_train_steps=50_000,
        local_batch_size=16,
    ),
    "mot_jepa_base": MotJepaTrainConfig(
        name="mot_jepa_base",
        layout_preset="base",
        num_train_steps=200_000,
        local_batch_size=8,
    ),
}


def cli() -> MotJepaTrainConfig:
    """Mirrors the repository's tyro entrypoint style (``training/config.py:1335``)."""
    return tyro.extras.overridable_config_cli({name: (name, cfg) for name, cfg in CONFIGS.items()})


# ======================================================================================
# Post-training (Stage 3: instruction, Stage 4: action)
# ======================================================================================


@dataclasses.dataclass(frozen=True)
class Stage3Config:
    """Clip<->language alignment."""

    instruction_emb: str = ""
    """Path to ``instruction_emb.npz`` from ``scripts/mot_jepa_embed_instructions.py``."""
    text_dim: int = 768
    """Width of the frozen instruction embedding. 768 for the SigLIP base text tower."""
    hidden: int = 1024
    projector_dim: int = 256
    temperature: float = 0.07
    min_stores_per_domain: int = 4
    """A batch must be able to see several tasks or the retrieval is trivial. This drops RDP
    (2 stores), RDP_Bimanual (1), Unit (1), Unit_Bimanual (2) and QINGLOONG (3), leaving 10
    domains and 518 stores / 248 tasks."""

    holdout_frac: float = 0.2
    """Fraction of TASKS reserved for evaluation, split deterministically by task id.

    This is the stage's falsifier, and it has to be a held-out split rather than a control on
    seen tasks. Within this corpus instruction and task are bijective -- every task is a
    distinct store with distinct objects and lighting -- so "identify the store and emit its
    vector" and "understand the instruction" are behaviourally *identical* on tasks the head
    has seen. No within-batch control can separate them. Retrieval against instructions for
    tasks never seen can: a lookup cannot generalize, language can.
    """
    eval_interval: int = 250

    surrogate_text: bool = False
    """Control arm: replace the instruction table with a fixed random vector per task and
    train the whole run on it. If this reaches the same held-out top-1 as real embeddings,
    language contributed nothing.

    Note this must be a separate *run*, not an inline probe. Feeding surrogate vectors through
    a ``text_proj`` that was fitted to SigLIP's space simply produces noise, so an inline
    version sits at chance regardless of what the head learned -- it tests only that the text
    input is used at all, not that *language* is."""


@dataclasses.dataclass(frozen=True)
class Stage4Config:
    """Action-conditioned latent rollout."""

    split_step: int = 4
    """Observe steps ``[0, split)``, predict ``[split, num_steps)``. Half and half at T=8."""
    weight_latent: float = 1.0
    weight_action_sync: float = 0.1
    """Small InfoNCE matching a clip's transition to its own action sequence. This is the term
    that is *provably* lower for the matched pair, mirroring ``L_sync``'s role in pretraining;
    the latent regression alone has a solution that ignores the action entirely."""
    temperature: float = 0.07
    projector_dim: int = 256
    donor_interval: int = 500
    """How often to measure the action donor ratio -- the stage's falsifier."""


@dataclasses.dataclass(frozen=True)
class MotJepaPosttrainConfig:
    """One post-training run. The pretrained backbone is frozen throughout."""

    name: str = "mot_jepa_stage3"
    exp_name: str = "dev"
    project_name: str = "mot-jepa"
    run_root: str = ".cache/mot_jepa/runs"

    stage: str = "instruction"
    """``instruction`` (Stage 3) or ``action`` (Stage 4)."""
    pretrained_run: str = ""
    """Run directory of the pretraining run whose backbone this stage freezes."""
    pretrained_step: int | None = None
    """Which checkpoint to load. ``None`` takes the latest."""

    layout_preset: str = "pilot"
    encoder: MoTEncoderConfig = dataclasses.field(default_factory=MoTEncoderConfig)
    predictor: MoTPredictorConfig = dataclasses.field(default_factory=MoTPredictorConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    stage3: Stage3Config = dataclasses.field(default_factory=Stage3Config)
    stage4: Stage4Config = dataclasses.field(default_factory=Stage4Config)

    seed: int = 42
    local_batch_size: int = 16
    num_train_steps: int = 20_000
    lr_peak: float = 5e-4
    lr_end: float = 1e-6
    lr_warmup_steps: int = 1_000
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad_norm: float = 1.0

    log_interval: int = 50
    save_interval: int = 500
    keep_last: int = 3
    keep_period: int | None = 5_000

    gradient_checkpointing: bool = False
    wandb_enabled: bool = True
    find_unused_parameters: bool = False
    """False, unlike pretraining: ``T_HARD`` -- the mode that legitimately leaves the tactile
    encoder without gradient -- does not occur in either post-training stage."""

    @property
    def layout(self) -> TokenLayout:
        presets = {"pilot": LAYOUT_PILOT, "base": LAYOUT_BASE}
        if self.layout_preset not in presets:
            raise ValueError(f"unknown layout_preset={self.layout_preset!r}, expected one of {sorted(presets)}")
        return presets[self.layout_preset]

    @property
    def run_dir(self) -> pathlib.Path:
        return pathlib.Path(self.run_root) / self.name / self.exp_name

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self.run_dir / "checkpoints"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> MotJepaPosttrainConfig:
        return _from_dict(cls, json.loads(text))

    def diff(self, other: MotJepaPosttrainConfig) -> dict[str, tuple[object, object]]:
        mine, theirs = dataclasses.asdict(self), dataclasses.asdict(other)
        return {key: (mine[key], theirs[key]) for key in mine if mine[key] != theirs[key]}


POSTTRAIN_CONFIGS: dict[str, MotJepaPosttrainConfig] = {
    "mot_jepa_posttrain_debug": MotJepaPosttrainConfig(
        name="mot_jepa_posttrain_debug",
        layout_preset="pilot",
        encoder=MoTEncoderConfig(depth=2, num_local_layers=1, num_heads=6, head_dim=64, rope=_PILOT_ROPE),
        predictor=MoTPredictorConfig(depth=1, width=192, num_heads=3, head_dim=64, rope=_PILOT_ROPE),
        num_train_steps=20,
        local_batch_size=2,
        lr_warmup_steps=2,
        save_interval=10,
        log_interval=1,
        wandb_enabled=False,
        data=DataConfig(num_workers=0),
        stage3=Stage3Config(min_stores_per_domain=1, eval_interval=10, holdout_frac=0.5),
        stage4=Stage4Config(donor_interval=10),
    ),
    "mot_jepa_stage3": MotJepaPosttrainConfig(
        name="mot_jepa_stage3",
        stage="instruction",
        layout_preset="pilot",
        encoder=MoTEncoderConfig(depth=12, num_local_layers=4, num_heads=6, head_dim=64, rope=_PILOT_ROPE),
        predictor=MoTPredictorConfig(depth=6, width=192, num_heads=3, head_dim=64, rope=_PILOT_ROPE),
        num_train_steps=20_000,
        local_batch_size=32,
    ),
    "mot_jepa_stage4": MotJepaPosttrainConfig(
        name="mot_jepa_stage4",
        stage="action",
        layout_preset="pilot",
        encoder=MoTEncoderConfig(depth=12, num_local_layers=4, num_heads=6, head_dim=64, rope=_PILOT_ROPE),
        predictor=MoTPredictorConfig(depth=6, width=192, num_heads=3, head_dim=64, rope=_PILOT_ROPE),
        num_train_steps=40_000,
        local_batch_size=16,
    ),
}


def posttrain_cli() -> MotJepaPosttrainConfig:
    return tyro.extras.overridable_config_cli({name: (name, cfg) for name, cfg in POSTTRAIN_CONFIGS.items()})
