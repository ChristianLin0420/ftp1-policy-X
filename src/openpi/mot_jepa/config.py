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

from openpi.mot_jepa.action_dit import ActionDiTConfig
from openpi.mot_jepa.drifting import DriftingConfig
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
    probe_batches: int = 16
    """Retrieval candidates = probe_batches * local_batch_size. At 1 the pool is one batch and
    every derived gap is quantised to 1/16, which is coarser than the effect being measured:
    a timeshuffle_gap series of 0.0625, 0.125, 0.0 is one clip, two clips, zero clips. The
    design specified 256 candidates. Each extra batch is one forward at the probe interval."""
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
# Action policy post-training (frozen encoder + DiT action head)
# ======================================================================================


@dataclasses.dataclass(frozen=True)
class PolicyConfig:
    """One action-policy run. The pretrained encoder is frozen throughout.

    ``encoder``/``predictor`` are here only to rebuild the pretrained student so the frozen
    backbone can be loaded into it -- they must match the pretraining run the checkpoint came
    from, or the positional shadow copy raises on the parameter count.
    """

    name: str = "mot_jepa_policy"
    exp_name: str = "dev"
    project_name: str = "mot-jepa"
    run_root: str = ".cache/mot_jepa/runs"

    pretrained_run: str = ""
    """Run directory (or backbone snapshot) whose EMA teacher becomes the frozen encoder."""
    pretrained_step: int | None = None
    holdout_mod: int = 0
    """Hold out every Nth EPISODE for offline evaluation. 0 disables. The evaluator reads the
    same field and takes the complement, so the two cannot drift apart."""
    init_head_from: str = ""
    """Path to a ``student.pt`` whose head weights initialise this run (fine-tuning).

    Applied only on a FRESH run, at step 0, with a fresh optimizer and LR schedule. A requeue
    resumes from its own checkpoint instead. The normalizer is not carried over -- a different
    dataset has different domains, and refitting is the point.
    """

    layout_preset: str = "pilot"
    encoder: MoTEncoderConfig = dataclasses.field(default_factory=MoTEncoderConfig)
    predictor: MoTPredictorConfig = dataclasses.field(default_factory=MoTPredictorConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    head: ActionDiTConfig = dataclasses.field(default_factory=ActionDiTConfig)
    drifting: DriftingConfig = dataclasses.field(default_factory=DriftingConfig)

    seed: int = 42
    local_batch_size: int = 16
    num_train_steps: int = 40_000
    lr_peak: float = 1e-4
    lr_end: float = 1e-6
    lr_warmup_steps: int = 1_000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad_norm: float = 1.0

    log_interval: int = 50
    save_interval: int = 1_000
    keep_last: int = 3
    keep_period: int | None = 10_000
    wandb_enabled: bool = True
    find_unused_parameters: bool = False
    """False, unlike pretraining: every head parameter is used on every step. Pretraining needs
    True only because mask mode T_HARD drops the tactile expert entirely on some steps."""

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
    def from_json(cls, text: str) -> PolicyConfig:
        return _from_dict(cls, json.loads(text))


#: The two arms differ in exactly one field, so the comparison is clean.
POLICY_CONFIGS: dict[str, PolicyConfig] = {
    "mot_jepa_policy_drifting": PolicyConfig(
        # `name` MUST equal the preset key: submit.sh derives the sbatch's RUN_DIR from
        # CONFIG_NAME, while the trainer derives its own from cfg.name. When they disagreed the
        # sbatch wrote PREEMPT_REQUEST into a directory the trainer never watched, so the graceful
        # save never ran and the requeue never armed -- both arms were killed at the 4h walltime
        # at step 18000 on 2026-08-05. Pretraining's presets follow the same rule.
        name="mot_jepa_policy_drifting",
        encoder=MoTEncoderConfig(depth=12, num_local_layers=4, num_heads=6, head_dim=64, rope=_PILOT_ROPE),
        predictor=MoTPredictorConfig(depth=6, width=192, num_heads=3, head_dim=64, rope=_PILOT_ROPE),
        head=ActionDiTConfig(objective="drifting"),
        data=DataConfig(index_step=4, num_workers=6),
    ),
    "mot_jepa_policy_flowmatch": PolicyConfig(
        name="mot_jepa_policy_flowmatch",
        encoder=MoTEncoderConfig(depth=12, num_local_layers=4, num_heads=6, head_dim=64, rope=_PILOT_ROPE),
        predictor=MoTPredictorConfig(depth=6, width=192, num_heads=3, head_dim=64, rope=_PILOT_ROPE),
        head=ActionDiTConfig(objective="flowmatch"),
        data=DataConfig(index_step=4, num_workers=6),
    ),
}


def policy_cli() -> PolicyConfig:
    return tyro.extras.overridable_config_cli({name: (name, cfg) for name, cfg in POLICY_CONFIGS.items()})
