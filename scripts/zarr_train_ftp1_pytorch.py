import contextlib
import dataclasses
import gc
import json
import logging
import os
import platform
import shutil
import sys
import time

# Add project root to Python path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
from openpi.shared.wandb_compat import wandb

from openpi.dataset_zarr import MultiZarrDataset
from openpi.models_pytorch.ftp1_model_config import FTP1_RESERVED_ACTION_DIM
from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_JOINT_DIM
import openpi.models_pytorch.ftp1_pytorch
from openpi.normalization import apply_norm_stats
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.data_loader import _unwrap_dataset
from scripts.train_pytorch import cleanup_ddp
from scripts.train_pytorch import init_logging
from scripts.train_pytorch import init_wandb
from scripts.train_pytorch import log_memory_usage
from scripts.train_pytorch import set_seed
from scripts.train_pytorch import setup_ddp
from scripts.zarr_train_ftp1_utils import _unwrap_compile_model
from scripts.zarr_train_ftp1_utils import check_train_config_consistency
from scripts.zarr_train_ftp1_utils import collect_trainable_parameters
from scripts.zarr_train_ftp1_utils import get_matching_checkpoint_parameter_names
from scripts.zarr_train_ftp1_utils import get_ftp1_latest_checkpoint_step
from scripts.zarr_train_ftp1_utils import load_partial_weights
from scripts.zarr_train_ftp1_utils import load_ftp1_hpt_checkpoint
from scripts.zarr_train_ftp1_utils import load_ftp1_model
from scripts.zarr_train_ftp1_utils import move_to_device
from scripts.zarr_train_ftp1_utils import print_model_statistics
from scripts.zarr_train_ftp1_utils import resolve_model_safetensors_path
from scripts.zarr_train_ftp1_utils import save_ftp1_hpt_checkpoint
from scripts.zarr_train_ftp1_validation_func import _build_first_batch_action_curve_images
from scripts.zarr_train_ftp1_validation_func import aggregate_validation_metrics_across_ranks
from scripts.zarr_train_ftp1_validation_func import compute_validation_metrics


LOADED_PARAM_LR_SCALE = 0.1


def _read_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid integer for %s=%r. Falling back to default=%d.", name, value, default)
        return default




def _read_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    logging.warning("Invalid bool for %s=%r. Falling back to default=%r.", name, value, default)
    return default


def _read_env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default



def _configure_rank_local_compile_cache(rank: int, local_rank: int) -> dict[str, str] | None:
    """Set rank-local torch.compile/triton caches to avoid multi-rank directory collisions."""
    if not _read_env_bool("FTP1_ENABLE_RANK_LOCAL_COMPILE_CACHE", True):
        return None

    base_dir = os.environ.get("FTP1_COMPILE_CACHE_BASE") or os.environ.get("TMPDIR")
    if not base_dir:
        return None

    cache_root = os.path.join(base_dir, "torch_compile_cache")
    suffix = f"rank{rank}_local{local_rank}_pid{os.getpid()}"
    cache_dirs = {
        "TORCH_COMPILE_DIR": os.path.join(cache_root, "compile", suffix),
        "TRITON_CACHE_DIR": os.path.join(cache_root, "triton", suffix),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_root, "inductor", suffix),
    }

    for env_name, path in cache_dirs.items():
        os.makedirs(path, exist_ok=True)
        os.environ[env_name] = path

    return cache_dirs

def _tune_torch_dynamo_recompile_limits() -> dict[str, int]:
    """Increase compile cache/recompile limits to better tolerate heterogeneous batch schemas."""
    target_recompile_limit = _read_env_int("FTP1_TORCHDYNAMO_RECOMPILE_LIMIT", 64)
    target_accumulated_limit = _read_env_int("FTP1_TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT", 1024)
    target_cache_limit = _read_env_int("FTP1_TORCHDYNAMO_CACHE_SIZE_LIMIT", target_recompile_limit)

    dynamo_cfg = torch._dynamo.config

    before = {
        "recompile_limit": int(getattr(dynamo_cfg, "recompile_limit", target_recompile_limit)),
        "accumulated_recompile_limit": int(
            getattr(dynamo_cfg, "accumulated_recompile_limit", target_accumulated_limit)
        ),
        "cache_size_limit": int(getattr(dynamo_cfg, "cache_size_limit", target_cache_limit)),
    }

    dynamo_cfg.recompile_limit = max(before["recompile_limit"], target_recompile_limit)
    dynamo_cfg.accumulated_recompile_limit = max(
        before["accumulated_recompile_limit"],
        target_accumulated_limit,
    )
    dynamo_cfg.cache_size_limit = max(before["cache_size_limit"], target_cache_limit)

    return {
        "recompile_limit": int(dynamo_cfg.recompile_limit),
        "accumulated_recompile_limit": int(dynamo_cfg.accumulated_recompile_limit),
        "cache_size_limit": int(dynamo_cfg.cache_size_limit),
    }


def create_adamw_param_groups(
    model: torch.nn.Module,
    loaded_param_names: set[str],
    config: _config.TrainConfig,
) -> tuple[list[dict], dict[str, int]]:
    """Create AdamW parameter groups, optionally separating checkpoint-loaded params."""
    model_unwrapped = _unwrap_compile_model(model)
    model_to_inspect = (
        model_unwrapped.module
        if isinstance(model_unwrapped, torch.nn.parallel.DistributedDataParallel)
        else model_unwrapped
    )

    loaded_params: list[torch.nn.Parameter] = []
    fresh_params: list[torch.nn.Parameter] = []
    loaded_numel = 0
    fresh_numel = 0

    for name, param in model_to_inspect.named_parameters():
        if not param.requires_grad:
            continue
        if name in loaded_param_names:
            loaded_params.append(param)
            loaded_numel += param.numel()
        else:
            fresh_params.append(param)
            fresh_numel += param.numel()

    param_groups: list[dict] = []
    if loaded_params:
        param_groups.append(
            dict(
                params=loaded_params,
                lr=config.lr_schedule.peak_lr * LOADED_PARAM_LR_SCALE,
                initial_lr=config.lr_schedule.peak_lr * LOADED_PARAM_LR_SCALE,
                betas=(config.optimizer.b1, config.optimizer.b2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
                group_name="loaded",
            )
        )
    if fresh_params:
        param_groups.append(
            dict(
                params=fresh_params,
                lr=config.lr_schedule.peak_lr,
                initial_lr=config.lr_schedule.peak_lr,
                betas=(config.optimizer.b1, config.optimizer.b2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
                group_name="fresh",
            )
        )

    stats = {
        "loaded_tensors": len(loaded_params),
        "fresh_tensors": len(fresh_params),
        "loaded_numel": loaded_numel,
        "fresh_numel": fresh_numel,
    }
    return param_groups, stats


def _extract_batch_domain_name(observation: object) -> str:
    """Best-effort extraction of the batch domain name."""
    domain_names = getattr(observation, "domain_names", None)
    if domain_names is None:
        return "unknown"

    if isinstance(domain_names, (list, tuple)):
        if len(domain_names) == 0:
            return "unknown"
        return str(domain_names[0])

    if isinstance(domain_names, np.ndarray):
        if domain_names.size == 0:
            return "unknown"
        return str(domain_names.reshape(-1)[0])

    if isinstance(domain_names, torch.Tensor):
        if domain_names.numel() == 0:
            return "unknown"
        return str(domain_names.reshape(-1)[0].item())

    return str(domain_names)


def _sanitize_wandb_metric_component(name: str) -> str:
    safe = str(name).strip()
    if not safe:
        return "unknown"
    safe = safe.replace("/", "|").replace("\\", "|").replace(" ", "_")
    return safe


def _extract_loss_and_extras(
    output: object,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(output, dict):
        loss = output["loss"]
        loss_extras = {k: v for k, v in output.items() if k != "loss"}
        return loss, loss_extras

    losses = output
    if isinstance(losses, list | tuple):
        losses = torch.stack(losses)
    elif not isinstance(losses, torch.Tensor):
        losses = torch.tensor(losses, device=device, dtype=torch.float32)
    loss = losses.mean()
    return loss, {}


# Prefixes must match the real parameter paths. Note the `.model.` segment inside paligemma:
# `paligemma_with_expert.paligemma.model.vision_tower.*`. Omitting it silently matches nothing
# and reports a gradient norm of exactly 0.
_MODALITY_PARAM_PREFIXES = {
    "tactile_encoder": ("hpt_tactile_encoder.",),
    "tactile_expert": ("paligemma_with_expert.gemma_tactile_expert.",),
    "vision_tower": ("paligemma_with_expert.paligemma.model.vision_tower.",),
    "vlm_language": ("paligemma_with_expert.paligemma.model.language_model.",),
    "action_expert": ("paligemma_with_expert.gemma_expert.",),
}


def compute_modality_grad_norms(model) -> dict[str, float]:
    """Pre-clipping gradient norm of each modality branch.

    `tactile_over_vision` is the modality-dominance monitor. A tactile branch trained only
    through one cross-attention edge, against a web-pretrained vision-language stream, can be
    starved to the point where the policy ignores touch entirely. If this ratio does not rise
    during training, a headline gain is not coming from the tactile pathway.
    """
    base = model.module if hasattr(model, "module") else model
    sq_sums = dict.fromkeys(_MODALITY_PARAM_PREFIXES, 0.0)
    for name, param in base.named_parameters():
        if param.grad is None:
            continue
        for branch, prefixes in _MODALITY_PARAM_PREFIXES.items():
            if name.startswith(prefixes):
                sq_sums[branch] += float(param.grad.detach().float().pow(2).sum())
                break

    norms = {branch: value**0.5 for branch, value in sq_sums.items()}
    payload = {f"GradNorm/{branch}": value for branch, value in norms.items()}
    tactile = norms["tactile_encoder"] + norms["tactile_expert"]
    payload["GradNorm/tactile_total"] = tactile

    # A branch legitimately reads 0 when it is frozen, so guard every ratio. A silently missing
    # ratio means the numerator or denominator matched no parameters -- check the prefixes.
    for name, denominator in (
        ("tactile_over_vision", norms["vision_tower"]),
        ("tactile_over_vlm", norms["vlm_language"]),
        ("tactile_over_action", norms["action_expert"]),
    ):
        if denominator > 0.0:
            payload[f"GradNorm/{name}"] = tactile / denominator
    return payload


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    rank = dist.get_rank() if use_ddp else 0

    compile_cache_dirs = None
    if config.use_torch_compile:
        compile_cache_dirs = _configure_rank_local_compile_cache(rank=rank, local_rank=local_rank)

    set_seed(config.seed, local_rank)
    logging.info(f"Running on: {platform.node()} | rank={rank}/{world_size} | local_rank={local_rank}")
    if compile_cache_dirs is not None:
        logging.info("Rank-local compile cache dirs: TORCH_COMPILE_DIR=%s, TRITON_CACHE_DIR=%s, TORCHINDUCTOR_CACHE_DIR=%s", compile_cache_dirs["TORCH_COMPILE_DIR"], compile_cache_dirs["TRITON_CACHE_DIR"], compile_cache_dirs["TORCHINDUCTOR_CACHE_DIR"])
    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step, _ = get_ftp1_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

        # Check train config consistency with saved checkpoint
        if is_main:
            try:
                check_train_config_consistency(config, exp_checkpoint_dir, latest_step, strict=False)
                logging.info("[✔] Train config consistency check passed")
            except (ValueError, FileNotFoundError) as e:
                logging.error(f"Train config consistency check failed: {e}")
                raise

    # Initialize wandb (only on main process)
    if is_main:
        logging.info(
            f"Initializing wandb: enabled={config.wandb_enabled}, project={config.project_name}, exp_name={config.exp_name}"
        )
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        if config.wandb_enabled:
            # Verify wandb is initialized
            if wandb.run is not None:
                logging.info(f"Wandb initialized successfully. Run ID: {wandb.run.id}, URL: {wandb.run.url}")
            else:
                logging.warning("Wandb run is None after initialization!")

    # Create data loader
    data_config = config.data.create(config.assets_dirs, config.model)
    if not hasattr(data_config, "dataset_config_path") or not data_config.dataset_config_path:
        raise ValueError("domain_batch_split test requires MultiZarrDataset (dataset_config_path)")

    # Assert norm folder params match current config (must have run compute_norm_stats with same params)
    norm_snapshot_path = config.assets_dirs / data_config.repo_id / "norm_params_snapshot.json"
    if getattr(config, "check_norm_params_snapshot", True):
        if not norm_snapshot_path.exists():
            raise ValueError(
                f"Norm params snapshot not found: {norm_snapshot_path}. "
                "Re-run compute_norm_stats (e.g. ycb_compute_norm_stats.sh) to generate norm stats and norm_params_snapshot.json."
            )
        with open(norm_snapshot_path) as f:
            saved_snapshot = json.load(f)
        current_snapshot = _config.get_norm_params_snapshot(config)
        if current_snapshot is not None:
            diffs = _config.check_norm_params_snapshot_matches(saved_snapshot, current_snapshot)
            if diffs:
                raise ValueError(
                    "Norm params mismatch (norm folder vs current training config). "
                    "Re-run compute_norm_stats with the same params as this script, or align config.\n"
                    + "\n".join(diffs)
                )
        if is_main:
            logging.info("[✔] Norm params snapshot matches current config")
        reuse_norm_repo_id = getattr(config, "reuse_norm_repo_id", "")
        if reuse_norm_repo_id:
            old_snapshot_path = config.assets_dirs / reuse_norm_repo_id / "norm_params_snapshot.json"
            if not old_snapshot_path.exists():
                raise ValueError(
                    f"Old norm params snapshot not found: {old_snapshot_path}. "
                    "The reuse_norm_repo_id assets must contain norm_params_snapshot.json."
                )
            with open(old_snapshot_path) as f:
                old_saved_snapshot = json.load(f)
            old_snapshot_diffs = _config.check_norm_params_snapshot_matches(
                old_saved_snapshot,
                current_snapshot,
                ignored_keys={"repo_id", "dataset_config_path"},
            )
            if old_snapshot_diffs:
                raise ValueError(
                    "Old norm params mismatch (old norm folder vs current training config). "
                    "Align normalization-related config or choose a compatible reuse_norm_repo_id.\n"
                    + "\n".join(old_snapshot_diffs)
                )
            if is_main:
                logging.info("[✔] Old norm params snapshot is compatible with current config: %s", reuse_norm_repo_id)
    elif is_main:
        logging.warning(
            "Skipping norm params snapshot validation because config.check_norm_params_snapshot is disabled"
        )
    train_data_loader, val_data_loader = _data.create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        shuffle=True,
        num_batches=None,
        num_workers=config.num_workers,
        val_num_workers=config.val_num_workers,
        seed=config.seed,
        framework="pytorch",
        domain_batch_split=True,  # Enable domain_batch_split functionality
        load_zarr_norm_stats=True,  # Load zarr normalization statistics
    )
    val_dataset = None
    model_cfg = config.model
    # Update dtype to match pytorch_training_precision
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    # Unwrap the dataset to get the underlying MultiZarrDataset
    from openpi.training.data_loader import _unwrap_dataset

    base_dataset = _unwrap_dataset(train_data_loader._data_loader.torch_loader.dataset)

    # Get dataset length for epoch calculation (based on samples, not batches)
    dataset_len = len(base_dataset)
    steps_per_epoch = (
        dataset_len + config.batch_size - 1
    ) // config.batch_size  # Ceiling division, using global batch_size

    tactile_input_config = base_dataset.default_tactile_input_config_path
    if is_main:
        logging.info(f"Tactile input config file: {tactile_input_config} !!!!")

    # set tactile input config file
    object.__setattr__(model_cfg, "tactile_input_config_file", tactile_input_config)
    model = openpi.models_pytorch.ftp1_pytorch.FTP1Pytorch(model_cfg).to(device)
    if is_main:
        print_model_statistics(model)

    if hasattr(model, "gradient_checkpointing_enable"):
        if getattr(config, "gradient_checkpointing_enable", True):
            enable_gradient_checkpointing = True
            model.gradient_checkpointing_enable()
            logging.info(
                "Enabled gradient checkpointing for memory optimization (via config.gradient_checkpointing_enable)"
            )
        else:
            enable_gradient_checkpointing = False
            logging.info("Gradient checkpointing is disabled by config.gradient_checkpointing_enable")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable CUDA optimizations for all training scenarios
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable for all cases, not just 8+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration for large-scale training
        if world_size >= 8:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
            logging.info("Enabled memory optimizations for 8+ GPU training")
        else:
            logging.info("Enabled CUDA optimizations (cudnn.benchmark, tf32)")

    # Load weights from weight_loader if specified
    # IMPORTANT: Load weights BEFORE torch.compile, otherwise key names won't match (_orig_mod. prefix)
    if resuming and config.pytorch_weight_path is not None:
        raise ValueError("Please set pytorch_weight_path to None when resuming training (resume=True).")
    if config.reduce_loaded_params_lr_warmup_only and config.load_optimizer_from_weight_path:
        raise ValueError("reduce_loaded_params_lr_warmup_only is incompatible with load_optimizer_from_weight_path.")

    loaded_param_names: set[str] = set()
    if config.reduce_loaded_params_lr_warmup_only and config.pytorch_weight_path is None:
        logging.warning(
            "reduce_loaded_params_lr_warmup_only is enabled but pytorch_weight_path is not set; "
            "using standard LR."
        )

    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from {config.pytorch_weight_path} via load_ftp1_model")
        if config.reduce_loaded_params_lr_warmup_only:
            if config.pytorch_weight_path.endswith("pi05") or config.pytorch_weight_path.endswith("pi0"):
                model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            else:
                model_path = str(resolve_model_safetensors_path(config.pytorch_weight_path))
            loaded_param_names = get_matching_checkpoint_parameter_names(model, model_path)
            logging.info(
                "Matched %d trainable parameters for reduced-LR checkpoint group from %s",
                len(loaded_param_names),
                model_path,
            )
            if not loaded_param_names:
                logging.warning("No trainable parameters matched the checkpoint; reduced-LR grouping will be skipped.")

        if config.pytorch_weight_path.endswith("pi05") or config.pytorch_weight_path.endswith("pi0"):
            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            load_partial_weights(model, model_path)
            logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
        else:
            _, _, loaded_ckpt_dir = load_ftp1_model(
                checkpoint_dir=config.pytorch_weight_path,
                tactile_input_config_file=tactile_input_config,
                device=device,
                model=model,
                load_hpt_tokenizers_strict=False,
                allow_partial_load=config.allow_partial_load,
            )
            logging.info(f"Loaded PyTorch weights/tokenizers from checkpoint: {loaded_ckpt_dir}")

    trainable_parameters = collect_trainable_parameters(model)

    load_optimizer_after_ddp = False
    if config.load_optimizer_from_weight_path:
        if config.pytorch_weight_path is None:
            raise ValueError("load_optimizer_from_weight_path requires pytorch_weight_path to be set.")
        optimizer_path = os.path.join(config.pytorch_weight_path, "optimizer.pt")
        if not os.path.exists(optimizer_path):
            raise FileNotFoundError(
                f"load_optimizer_from_weight_path is set but optimizer.pt not found at {optimizer_path}"
            )
        load_optimizer_after_ddp = True

    # Wrap model with DDP first (before torch.compile for multi-GPU compatibility)
    if use_ddp:
        # FTP1 models may have unused parameters due to conditional branches (e.g., tactile expert, state modes),
        # so we always set find_unused_parameters=True to avoid DDP errors.
        # Setting this to False can improve performance but requires all parameters to be used in forward pass.
        find_unused = True
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=False,  # Disable since we need Heterogeneous Batching
        )

    # torch.compile optimization (PyTorch 2.0+)
    # IMPORTANT:
    # 1. torch.compile must happen AFTER loading weights, otherwise checkpoint keys won't match.
    # 2. For multi-GPU: Use "default" mode instead of "reduce-overhead" because reduce-overhead uses CUDA graphs
    #    which are incompatible with DDP. Compile after DDP wrapping to ensure all ranks compile together.
    if config.use_torch_compile:
        if hasattr(torch, "compile"):
            try:
                tuned_limits = _tune_torch_dynamo_recompile_limits()

                # Choose compile mode based on GPU count
                # - Single GPU: use "reduce-overhead" for better performance (uses CUDA graphs)
                # - Multi-GPU: use "default" mode (CUDA graphs incompatible with DDP)
                if use_ddp:
                    compile_mode = "default"
                    compile_reason = "multi-GPU (DDP incompatible with reduce-overhead CUDA graphs)"
                else:
                    compile_mode = "reduce-overhead"
                    compile_reason = "single-GPU"

                model = torch.compile(model, mode=compile_mode)
                logging.info(
                    "Enabled torch.compile with mode='%s' (%s), "
                    "dynamo limits: recompile_limit=%d, accumulated_recompile_limit=%d, cache_size_limit=%d",
                    compile_mode,
                    compile_reason,
                    tuned_limits["recompile_limit"],
                    tuned_limits["accumulated_recompile_limit"],
                    tuned_limits["cache_size_limit"],
                )

                # Warmup: Run a few dummy forward passes to ensure all ranks complete compilation
                # This prevents deadlock where rank 0 finishes compilation while others are still compiling
                if use_ddp:
                    if is_main:
                        logging.info("Warming up torch.compile on all ranks...")
                    model.train()
                    try:
                        # Get a sample batch from data loader for warmup
                        # The data loader returns (observation, actions) tuple
                        warmup_iter = iter(train_data_loader)
                        observation, actions = next(warmup_iter)

                        # Move to device and cast dtypes (same as training loop)
                        observation = move_to_device(observation, device)
                        actions = actions.to(torch.float32)
                        actions = actions.to(device)

                        # Run 2 warmup iterations to ensure compilation completes on all ranks
                        with torch.no_grad():
                            for warmup_step in range(2):
                                _ = model(observation, actions)
                                if is_main and warmup_step == 0:
                                    logging.info(f"Warmup step {warmup_step + 1}/2 completed")

                        # Synchronize all ranks after warmup to ensure all compilation is done
                        torch.distributed.barrier()
                        if is_main:
                            logging.info("torch.compile warmup completed on all ranks")
                    except Exception as warmup_error:
                        logging.warning(f"Warmup failed (non-critical): {warmup_error}. Continuing training...")
                        # Still synchronize even if warmup failed
                        if use_ddp:
                            torch.distributed.barrier()
            except Exception as e:
                logging.warning(f"torch.compile requested but failed; continuing without compile: {e}")
        else:
            logging.warning("torch.compile requested but torch.compile is not available; continuing without compile")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer
    loaded_lr_group_stats: dict[str, int] | None = None
    if config.reduce_loaded_params_lr_warmup_only and loaded_param_names:
        param_groups, loaded_lr_group_stats = create_adamw_param_groups(model, loaded_param_names, config)
        optim = torch.optim.AdamW(param_groups)
    else:
        optim = torch.optim.AdamW(
            trainable_parameters,
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        for pg in optim.param_groups:
            pg["initial_lr"] = peak_lr
            pg.setdefault("group_name", "default")

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_ftp1_hpt_checkpoint(model, optim, config.checkpoint_dir, device, global_step=latest_step)
        logging.info(f"Resumed training from step {global_step}")

    if load_optimizer_after_ddp:
        optimizer_path = os.path.join(config.pytorch_weight_path, "optimizer.pt")
        logging.info(f"Loading optimizer state from weight path: {optimizer_path}")
        optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
        optim.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("Loaded optimizer state (global_step stays at 0 - scheduler is reset)")

    def lr_schedule(step: int, base_lr: float = peak_lr):
        """Cosine schedule with warmup, scaled to each param group's base_lr."""
        lr_multiplier = base_lr / peak_lr if peak_lr > 0 else 1.0
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            scheduled_lr = init_lr + (peak_lr - init_lr) * step / warmup_steps
            return scheduled_lr * lr_multiplier

        if config.reduce_loaded_params_lr_warmup_only:
            progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
            cos = 0.5 * (1 + np.cos(np.pi * progress))
            return end_lr + (peak_lr - end_lr) * cos

        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        scheduled_lr = end_lr + (peak_lr - end_lr) * cos
        return scheduled_lr * lr_multiplier

    model.train()
    start_time = time.time()

    profiler_enabled = _read_env_bool("FTP1_TORCH_PROFILER", False)
    profiler_rank0_only = _read_env_bool("FTP1_TORCH_PROFILER_RANK0_ONLY", True)
    profiler_start_step = _read_env_int("FTP1_PROFILE_START_STEP", -1)
    profiler_end_step = _read_env_int("FTP1_PROFILE_END_STEP", -1)
    profiler_record_shapes = _read_env_bool("FTP1_TORCH_PROFILER_RECORD_SHAPES", False)
    profiler_profile_memory = _read_env_bool("FTP1_TORCH_PROFILER_PROFILE_MEMORY", False)
    profiler_with_stack = _read_env_bool("FTP1_TORCH_PROFILER_WITH_STACK", False)
    profiler_defer_trace_export = _read_env_bool(
        "FTP1_TORCH_PROFILER_DEFER_TRACE_EXPORT",
        use_ddp and profiler_rank0_only,
    )

    profiler = None
    profiler_dir = None
    profiler_trace_handler = None
    should_profile = (
        profiler_enabled
        and profiler_start_step >= 0
        and profiler_end_step > profiler_start_step
        and ((not profiler_rank0_only) or rank == 0)
    )
    if should_profile:
        profiler_dir = _read_env_str(
            "FTP1_PROFILE_DIR",
            str(config.assets_dirs / data_config.repo_id / "profiler" / config.exp_name),
        )
        os.makedirs(profiler_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        active_steps = profiler_end_step - profiler_start_step
        schedule = torch.profiler.schedule(wait=profiler_start_step, warmup=0, active=active_steps, repeat=1)
        profiler_trace_handler = torch.profiler.tensorboard_trace_handler(profiler_dir, worker_name=f"rank{rank}")
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=(None if profiler_defer_trace_export else profiler_trace_handler),
            record_shapes=profiler_record_shapes,
            profile_memory=profiler_profile_memory,
            with_stack=profiler_with_stack,
        )

    infos = []  # Collect stats over log interval

    domain_names_for_logging = [str(name) for name in getattr(base_dataset, "domain_list", [])]
    domain_unknown_name = "__unknown_domain__"
    if domain_unknown_name in domain_names_for_logging:
        domain_unknown_name = "__unknown_domain_fallback__"
    domain_names_for_logging.append(domain_unknown_name)
    domain_name_to_index = {name: idx for idx, name in enumerate(domain_names_for_logging)}
    domain_unknown_idx = domain_name_to_index[domain_unknown_name]

    num_domain_buckets = len(domain_names_for_logging)
    domain_loss_sum_window = np.zeros(num_domain_buckets, dtype=np.float64)
    domain_loss_count_window = np.zeros(num_domain_buckets, dtype=np.float64)
    domain_loss_max_window = np.full(num_domain_buckets, -np.inf, dtype=np.float64)

    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        if loaded_lr_group_stats is not None:
            logging.info(
                "Reduced-LR checkpoint grouping enabled: loaded=%d tensors / %d params at %.2e, "
                "fresh=%d tensors / %d params at %.2e",
                loaded_lr_group_stats["loaded_tensors"],
                loaded_lr_group_stats["loaded_numel"],
                peak_lr * LOADED_PARAM_LR_SCALE,
                loaded_lr_group_stats["fresh_tensors"],
                loaded_lr_group_stats["fresh_numel"],
                peak_lr,
            )
            if config.reduce_loaded_params_lr_warmup_only:
                logging.info("Warmup-only reduced LR is enabled: all parameter groups share the same LR after warmup.")
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")
        if config.wandb_enabled:
            logging.info("Domain train logging enabled: TrainDomainLoss/* aggregated with tensor all-reduce.")
            domain_grad_probe_interval = int(getattr(config, "domain_grad_probe_interval", 0))
            if domain_grad_probe_interval > 0:
                logging.warning(
                    "domain_grad_probe_interval=%d is currently ignored; domain grad probing is disabled.",
                    domain_grad_probe_interval,
                )
        if profiler is not None:
            logging.info(
                "Enabled torch.profiler: steps [%d, %d), rank0_only=%s, defer_trace_export=%s, dir=%s",
                profiler_start_step,
                profiler_end_step,
                profiler_rank0_only,
                profiler_defer_trace_export,
                profiler_dir,
            )

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    with (profiler if profiler is not None else contextlib.nullcontext()) as prof:
        while global_step < config.num_train_steps:
            # Calculate current epoch based on samples processed, not batches.
            # This matches the calculation in compute_norm_stats.py.
            current_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
            train_data_loader._data_loader.set_epoch(current_epoch)

            # TorchDataLoader.__iter__() is an infinite loop when num_batches=None.
            # Keep rank-synchronous progression by iterating directly and stopping on global_step.
            for observation, actions in train_data_loader:
                if global_step >= config.num_train_steps:
                    break

                # The unified data loader returns (observation, actions) tuple
                # Move observation to device (recursively handle nested dicts and tensors)
                with torch.profiler.record_function("h2d"):
                    observation = move_to_device(observation, device)  # noqa: PLW2901
                    actions = actions.to(torch.float32)  # noqa: PLW2901
                    actions = actions.to(device)  # noqa: PLW2901

                # Update LR for each parameter group based on its base lr
                with torch.profiler.record_function("lr_schedule"):
                    for pg in optim.param_groups:
                        base_lr = pg.get("initial_lr", peak_lr)
                        pg["lr"] = lr_schedule(global_step, base_lr)

                domain_name = _extract_batch_domain_name(observation)

                # Forward pass
                with torch.profiler.record_function("forward"):
                    output = model(observation, actions)
                with torch.profiler.record_function("loss_extract"):
                    loss, loss_extras = _extract_loss_and_extras(output, device=device)
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(f"Non-finite training loss at step {global_step}: {loss.item()}")
                    loss_value = float(loss.item())

                domain_idx = domain_name_to_index.get(domain_name, domain_unknown_idx)
                domain_loss_sum_window[domain_idx] += loss_value
                domain_loss_count_window[domain_idx] += 1.0
                domain_loss_max_window[domain_idx] = max(domain_loss_max_window[domain_idx], loss_value)

                # Backward pass for the actual optimizer update
                with torch.profiler.record_function("backward"):
                    loss.backward()

                # Per-branch gradient norms, measured pre-clipping and only on logging steps.
                modality_grad_norms: dict[str, float] = {}
                if is_main and global_step % config.log_interval == 0:
                    modality_grad_norms = compute_modality_grad_norms(model)

                # Gradient clipping
                with torch.profiler.record_function("clip_grad"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=config.optimizer.clip_gradient_norm,
                        error_if_nonfinite=True,
                    )

                # Optimizer step
                with torch.profiler.record_function("optim_step"):
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                # Collect stats
                if is_main:
                    info = {
                        "loss": loss_value,
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                    for k, v in loss_extras.items():
                        info[k] = v.item()
                    infos.append(info)

                if global_step % config.log_interval == 0:
                    if config.wandb_enabled:
                        domain_loss_sum_tensor = torch.tensor(domain_loss_sum_window, device=device, dtype=torch.float64)
                        domain_loss_count_tensor = torch.tensor(domain_loss_count_window, device=device, dtype=torch.float64)
                        domain_loss_max_tensor = torch.tensor(domain_loss_max_window, device=device, dtype=torch.float64)

                        if use_ddp:
                            dist.all_reduce(domain_loss_sum_tensor, op=dist.ReduceOp.SUM)
                            dist.all_reduce(domain_loss_count_tensor, op=dist.ReduceOp.SUM)
                            dist.all_reduce(domain_loss_max_tensor, op=dist.ReduceOp.MAX)

                    if is_main:
                        elapsed = time.time() - start_time

                        # Average stats over log interval
                        avg_loss = sum(info["loss"] for info in infos) / len(infos)
                        avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                        avg_grad_norm = None
                        if any("grad_norm" in info for info in infos):
                            vals = [
                                info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                            ]
                            if len(vals) > 0:
                                avg_grad_norm = sum(vals) / len(vals)

                        logging.info(
                            f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} "
                            f"grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                            if avg_grad_norm is not None
                            else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                        )

                        # Log to wandb
                        if config.wandb_enabled and len(infos) > 0:
                            log_payload = {
                                "Trainning/loss": avg_loss,
                                "Trainning/learning_rate": avg_lr,
                                "Trainning/step": global_step,
                                "Trainning/epoch": current_epoch,
                                "Trainning/time_per_step": elapsed / config.log_interval,
                            }
                            if avg_grad_norm is not None:
                                log_payload["grad_norm"] = avg_grad_norm
                            log_payload.update(modality_grad_norms)

                            for idx, domain in enumerate(domain_names_for_logging):
                                count_value = float(domain_loss_count_tensor[idx].item())
                                if count_value <= 0:
                                    continue
                                safe_domain = _sanitize_wandb_metric_component(domain)
                                sum_value = float(domain_loss_sum_tensor[idx].item())
                                max_value = float(domain_loss_max_tensor[idx].item())
                                log_payload[f"TrainDomainLoss/{safe_domain}/mean"] = sum_value / count_value
                                log_payload[f"TrainDomainLoss/{safe_domain}/max"] = max_value

                            wandb.log(log_payload, step=global_step)

                        start_time = time.time()
                        infos = []  # Reset stats collection

                    domain_loss_sum_window.fill(0.0)
                    domain_loss_count_window.fill(0.0)
                    domain_loss_max_window.fill(-np.inf)

                if (global_step % config.val_interval == 0) or (global_step == config.num_train_steps - 1):
                    if val_data_loader is not None:
                        model.eval()
                        val_infos = []
                        first_batch_action_curve_images = None
                        if val_dataset is None:
                            val_dataset = _unwrap_dataset(val_data_loader._data_loader.torch_loader.dataset)

                        with torch.no_grad():
                            # Get the underlying model if wrapped in DDP and/or torch.compile
                            # IMPORTANT: Model wrapping order is: DDP -> torch.compile
                            # So we need to unwrap compile first, then unwrap DDP
                            # First unwrap torch.compile wrapper if present (outermost wrapper)
                            model_for_inference = _unwrap_compile_model(model)
                            # Then unwrap DDP wrapper if present (inner wrapper)
                            if isinstance(model_for_inference, torch.nn.parallel.DistributedDataParallel):
                                model_for_inference = model_for_inference.module

                            # Get the actual batch count from batch_sampler (for domain_batch_split)
                            # This is the accurate count for this rank, since TorchDataLoader.__iter__()
                            # is an infinite loop when num_batches=None, but we only want to iterate once
                            val_batch_count = None
                            try:
                                val_torch_loader = val_data_loader._data_loader.torch_loader
                                if (
                                    hasattr(val_torch_loader, "batch_sampler")
                                    and val_torch_loader.batch_sampler is not None
                                ):
                                    val_batch_count = len(val_torch_loader.batch_sampler)
                                else:
                                    # Fallback to DataLoaderImpl length
                                    val_batch_count = len(val_data_loader)
                            except (TypeError, AttributeError, NotImplementedError):
                                val_batch_count = None

                            # Create progress bar
                            # Use leave=True to keep the progress bar visible after completion
                            # Use dynamic_ncols=True to prevent text scrolling issues
                            pbar_val = tqdm.tqdm(
                                total=val_batch_count,
                                desc=f"Validating (rank {rank})",
                                disable=not is_main,
                                leave=True,
                                dynamic_ncols=True,
                                mininterval=0.1,  # Update at least every 0.1 seconds
                            )

                            # Process validation batches
                            # TorchDataLoader.__iter__() is an infinite loop when num_batches=None,
                            # so we need to manually break after iterating through all batches once
                            val_batch_iter = iter(val_data_loader)
                            val_idx = 0

                            # Handle case where val_batch_count is 0 (no batches for this rank)
                            if val_batch_count == 0:
                                logging.info(
                                    f"[Rank {rank}] Validation: No batches assigned to this rank, skipping validation loop"
                                )
                            else:
                                while True:
                                    # If we have a known batch count and reached it, break
                                    if val_batch_count is not None and val_idx >= val_batch_count:
                                        break

                                    try:
                                        val_observation, val_actions = next(val_batch_iter)
                                    except StopIteration:
                                        # Dataset exhausted (shouldn't happen if batch_sampler is used, but handle it)
                                        break

                                    val_idx += 1
                                    val_observation = move_to_device(val_observation, device)
                                    sample_actions = model_for_inference.sample_actions(device, val_observation)
                                    gt_actions = val_actions
                                    action_masks = val_observation.action_masks

                                    # Get domain names from observation
                                    # With domain_batch_split=True, all samples in a batch are from the same domain
                                    domain_names = None
                                    if (
                                        hasattr(val_observation, "domain_names")
                                        and val_observation.domain_names is not None
                                    ):
                                        # domain_names should be a list for the entire batch (since domain_batch_split=True)
                                        if (
                                            isinstance(val_observation.domain_names, list)
                                            and len(val_observation.domain_names) > 0
                                        ):
                                            domain_names = val_observation.domain_names[
                                                0
                                            ]  # All samples have same domain, take first

                                    # Compute validation metrics
                                    val_info = compute_validation_metrics(
                                        sample_actions=sample_actions,
                                        gt_actions=gt_actions,
                                        action_masks=action_masks,
                                        domain_names=domain_names,
                                        val_dataset=val_dataset,
                                        single_arm_action_rep_dim=config.single_arm_action_rep_dim,
                                        arm_joints_dim=FTP1_SINGLE_ARM_JOINT_DIM,
                                        reserved_action_dim=FTP1_RESERVED_ACTION_DIM,
                                    )
                                    val_infos.append(val_info)

                                    # Draw action curves from the first validation batch (main rank only)
                                    if (
                                        is_main
                                        and config.wandb_enabled
                                        and val_idx == 1
                                        and first_batch_action_curve_images is None
                                    ):
                                        try:
                                            first_batch_action_curve_images = _build_first_batch_action_curve_images(
                                                sample_actions=sample_actions,
                                                gt_actions=gt_actions,
                                                action_masks=action_masks,
                                                domain_names=domain_names,
                                                val_dataset=val_dataset,
                                                single_arm_action_rep_dim=config.single_arm_action_rep_dim,
                                                arm_joints_dim=FTP1_SINGLE_ARM_JOINT_DIM,
                                                reserved_action_dim=FTP1_RESERVED_ACTION_DIM,
                                                max_samples=10,
                                            )
                                        except Exception as e:
                                            logging.warning(f"Failed to build first-batch action curve plots: {e}")
                                    pbar_val.update(1)

                                    # Explicitly delete tensors to free memory immediately
                                    # This prevents memory accumulation during validation loop
                                    # Note: We don't call empty_cache() here to avoid performance overhead
                                    # Python's garbage collector and explicit del should be sufficient
                                    del sample_actions, gt_actions, action_masks, val_observation, val_actions

                            # Refresh and close progress bar properly to prevent text scrolling
                            pbar_val.refresh()
                            pbar_val.close()

                        # Synchronize all ranks before aggregation (important for dist.all_gather)
                        if use_ddp:
                            dist.barrier()

                        # Aggregate validation metrics across all ranks
                        aggregated_metrics = aggregate_validation_metrics_across_ranks(val_infos, use_ddp, device=device)

                        # Clear val_infos to free memory after aggregation
                        del val_infos
                        # Only clear CUDA cache once at the end of validation to avoid performance overhead
                        # This is safe because validation runs in eval() mode and doesn't affect training state
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Log validation metrics (only on main process)
                        if is_main:
                            if config.wandb_enabled:
                                val_log_payload = {}

                                # Add total metrics first
                                for key, value in aggregated_metrics.get("total", {}).items():
                                    val_log_payload[f"Validation/{key}"] = value

                                # Add domain-specific metrics
                                for domain_names, domain_metrics in aggregated_metrics.get("by_domain", {}).items():
                                    for key, value in domain_metrics.items():
                                        val_log_payload[f"ValidationDomain/{domain_names}/{key}"] = value

                                if first_batch_action_curve_images:
                                    val_log_payload["ValidationFirstBatch/action_curves"] = first_batch_action_curve_images

                                wandb.log(val_log_payload, step=global_step)

                            # Log to console (only on main process)
                            total_metrics = aggregated_metrics.get("total", {})
                            if total_metrics:
                                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in total_metrics.items()])
                                logging.info(
                                    f"Validation step={global_step} (aggregated across {world_size} GPUs): {metrics_str}"
                                )

                        model.train()

                # Save checkpoint using the new mechanism
                save_ftp1_hpt_checkpoint(
                    model, optim, global_step, config, is_main, data_config,
                )
                global_step += 1
                if prof is not None:
                    prof.step()

                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {"loss": f"{loss_value:.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                    )

    if profiler is not None and profiler_dir is not None and ((not profiler_rank0_only) or rank == 0):
        report_path = os.path.join(profiler_dir, f"report_rank{rank}.txt")
        try:
            with open(report_path, "w") as report_f:
                report_f.write("Torch profiler key_averages\n")
                report_f.write(f"profiled_steps=[{profiler_start_step}, {profiler_end_step})\n")
                report_f.write("\n== Self CUDA time (top 50) ==\n")
                report_f.write(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
                report_f.write("\n\n== Self CPU time (top 50) ==\n")
                report_f.write(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
        except Exception as exc:
            logging.warning("Failed to write profiler report: %s", exc)
        else:
            if is_main:
                logging.info("Wrote profiler report: %s", report_path)

    if (
        profiler is not None
        and profiler_trace_handler is not None
        and profiler_defer_trace_export
        and profiler_dir is not None
        and ((not profiler_rank0_only) or rank == 0)
    ):
        try:
            profiler_trace_handler(profiler)
        except Exception as exc:
            logging.warning("Failed to export deferred profiler trace: %s", exc)
        else:
            if is_main:
                logging.info("Exported deferred profiler trace to: %s", profiler_dir)

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    # Finalize config after command line arguments are applied
    # This sets all computed/derived values (e.g., action_dim, assets_dir, cache_t3_pretrained_checkpoint_dir)
    if isinstance(config, _config.FTP1TrainConfig):
        config.finalize_config()
    train_loop(config)


if __name__ == "__main__":
    main()
