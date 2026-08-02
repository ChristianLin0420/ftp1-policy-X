#!/usr/bin/env python3
"""
FTP1 evaluation script for evaluating a trained model on validation set.

This script loads a trained FTP1 model from a checkpoint directory, evaluates it
on the validation set, and saves the results to a JSON file in the checkpoint directory.

Usage:
    python scripts_exp_zarr/zarr_eval_ftp1_pytorch.py \
        --checkpoint_dir /path/to/checkpoint \
        --device cuda
"""

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import sys
import time

# Add project root to Python path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import tqdm
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.data_loader import _unwrap_dataset
from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_JOINT_DIM, FTP1_RESERVED_ACTION_DIM

from scripts.train_pytorch import init_logging, set_seed
from scripts.zarr_train_ftp1_utils import load_ftp1_model, print_model_statistics
from scripts.zarr_train_ftp1_validation_func import (
    compute_validation_metrics,
    aggregate_validation_metrics,
)


from openpi.normalization import apply_norm_stats
from scripts.zarr_train_ftp1_utils import move_to_device


def load_config_from_json(json_path: pathlib.Path) -> _config.TrainConfig:
    """Load config from JSON file."""
    import dataclasses
    
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    # Get base config by name, or use ftp1 as default for FTP1TrainConfig
    config_name = config_dict.get('name', None)
    base_config = None
    
    if config_name:
        try:
            base_config = _config.get_config(config_name)
            logging.info(f"Found base config by name: {config_name}")
        except Exception as e:
            logging.warning(f"Could not load config by name {config_name}: {e}")
    
    # If no base config found, try ftp1 as default for FTP1TrainConfig
    if base_config is None:
        if 'dataset_config_path' in config_dict or 'repo_id' in config_dict:
            try:
                base_config = _config.get_config('ftp1')
                logging.info("Using 'ftp1' as base config")
            except Exception as e:
                logging.warning(f"Could not load 'ftp1' config: {e}")
    
    if base_config is None:
        raise ValueError(
            f"Cannot determine config type from JSON. "
            f"Please ensure 'name' field is present in {json_path}, "
            f"or that the config contains 'dataset_config_path' or 'repo_id' for FTP1TrainConfig."
        )
    
    # Update base config with values from JSON
    # Build updates dict, handling nested configs
    updates = {}
    for key, value in config_dict.items():
        if key in ['name', 'weight_loader', 'freeze_filter', 'data', 'model']:
            # Skip fields that are not directly updatable or are complex objects
            # (data is created in __post_init__, model is loaded separately)
            continue
        
        # Handle nested configs (lr_schedule, optimizer)
        if key == 'lr_schedule' and isinstance(value, dict) and hasattr(base_config, 'lr_schedule'):
            # Update lr_schedule fields
            lr_schedule_updates = {}
            for lr_key, lr_value in value.items():
                if hasattr(base_config.lr_schedule, lr_key):
                    lr_schedule_updates[lr_key] = lr_value
            if lr_schedule_updates:
                updates['lr_schedule'] = dataclasses.replace(base_config.lr_schedule, **lr_schedule_updates)
        elif key == 'optimizer' and isinstance(value, dict) and hasattr(base_config, 'optimizer'):
            # Update optimizer fields
            opt_updates = {}
            for opt_key, opt_value in value.items():
                if hasattr(base_config.optimizer, opt_key):
                    opt_updates[opt_key] = opt_value
            if opt_updates:
                updates['optimizer'] = dataclasses.replace(base_config.optimizer, **opt_updates)
        elif hasattr(base_config, key):
            # Simple field update
            updates[key] = value
    
    # Apply updates
    if updates:
        base_config = dataclasses.replace(base_config, **updates)
    
    return base_config


TACTILE_MODES = ("real", "zero", "noise", "shuffle")


def apply_tactile_ablation(observation, mode: str, previous_tactiles: dict | None):
    """Replace the tactile stream at test time to measure how much the policy actually uses it.

    Tactile has already been normalized by the dataset at this point, so each mode strips a
    different property while staying on the scale the model expects:

      real     unmodified.
      zero     the normalized mean, i.e. an uninformative constant. Distinguishes "uses touch"
               from "ignores touch".
      noise    Gaussian matched to this batch's per-key mean and std. Preserves the first two
               moments and destroys spatial and temporal structure, so it separates "uses the
               structure of touch" from "uses only its energy".
      shuffle  a real tactile reading from a different batch. Keeps real statistics AND real
               structure but breaks correspondence with the current observation. This is the
               strongest control: a policy that scores well here is not binding touch to the
               rest of the scene at all.

    Returns (observation, tactiles_for_next_call).
    """
    tactiles = getattr(observation, "tactiles", None)
    if mode == "real" or not tactiles:
        return observation, tactiles

    current = {key: value.detach().clone() for key, value in tactiles.items()}
    replaced = {}
    for key, value in tactiles.items():
        if mode == "zero":
            replaced[key] = torch.zeros_like(value)
        elif mode == "noise":
            replaced[key] = torch.randn_like(value) * value.std() + value.mean()
        elif mode == "shuffle":
            donor = None if previous_tactiles is None else previous_tactiles.get(key)
            # The first batch has no donor yet; fall back to a within-batch roll, which is a
            # no-op at batch size 1 and is why the first batch is dropped by the caller.
            replaced[key] = value.roll(1, dims=0) if donor is None or donor.shape != value.shape else donor
        else:
            raise ValueError(f"Unknown tactile mode: {mode}")

    return dataclasses.replace(observation, tactiles=replaced), current


def eval_loop(
    checkpoint_dir: pathlib.Path,
    config: _config.TrainConfig | None = None,
    device: str = "cuda",
    tactile_mode: str = "real",
):
    """Evaluate model on validation set."""
    init_logging()
    set_seed(42, 0)  # Fixed seed for reproducibility
    
    logging.info(f"Loading model from checkpoint: {checkpoint_dir}")
    
    # Load model using the same function as inference script
    model, model_config, ckpt_dir = load_ftp1_model(
        checkpoint_dir=checkpoint_dir,
        tactile_input_config_file=None,  # Will be auto-detected from checkpoint
        device=device
    )
    
    logging.info(f"Using checkpoint directory: {ckpt_dir}")
    logging.info(f"Model weights should be loaded from: {ckpt_dir / 'model.safetensors'}")
    
    # Ensure model is on the correct device (in case some parameters weren't moved)
    model = model.to(device)
    
    if torch.cuda.is_available():
        print_model_statistics(model)
    
    model.eval()
    logging.info("Model set to evaluation mode")
    
    # Load base config from checkpoint if not provided
    base_config = None
    if config is None:
        # Try to load config from checkpoint directory
        train_config_json_path = ckpt_dir / "train_config.json"
        if train_config_json_path.exists():
            logging.info(f"Loading config from checkpoint directory: {train_config_json_path}")
            base_config = load_config_from_json(train_config_json_path)
        else:
            # Try parent directory
            train_config_json_path = ckpt_dir.parent / "train_config.json"
            if train_config_json_path.exists():
                logging.info(f"Loading config from parent directory: {train_config_json_path}")
                base_config = load_config_from_json(train_config_json_path)
            else:
                raise ValueError(
                    f"Config not provided and train_config.json not found in checkpoint directory. "
                    f"Please provide config via command line or ensure train_config.json exists at {ckpt_dir / 'train_config.json'}"
                )
        config = base_config
    else:
        # Config provided via command line, but we may want to merge with checkpoint config
        # For now, use the provided config as-is
        logging.info("Using config provided via command line arguments")
    
    # Finalize config
    if isinstance(config, _config.FTP1TrainConfig):
        config.finalize_config()
    
    # Create data config
    data_config = config.data.create(config.assets_dirs, config.model)
    
    if not hasattr(data_config, 'dataset_config_path') or not data_config.dataset_config_path:
        raise ValueError("Evaluation requires MultiZarrDataset (dataset_config_path)")
    
    # Create validation data loader (no training loader needed)
    _, val_data_loader = _data.create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_batches=None,
        num_workers=config.num_workers,
        seed=42,
        framework="pytorch",
        domain_batch_split=True,  # Enable domain_batch_split functionality
        load_zarr_norm_stats=True,  # Load zarr normalization statistics
    )
    
    # Unwrap dataset to get MultiZarrDataset
    val_dataset = _unwrap_dataset(val_data_loader._data_loader.torch_loader.dataset)
    
# tensor([2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795,
#         2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807,
#         2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819,
#         2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831,
#         2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843,
#         2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855,
#         2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867,
#         2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879,
#         2880, 2881, 2882, 2883, 2884, 2885

    # import pdb; pdb.set_trace()
    # data = val_dataset[2824]
    # domain_names = data['domain_name']
    # domain_idx = val_dataset.domain_list.index(domain_names)
    # domain_dataset = val_dataset.domain_dataset_list[domain_idx]
    # gt_data = {'actions': data['actions']}
    # gt_actions_un = apply_norm_stats(gt_data, domain_dataset.norm_stats, unnormalize=True, use_keys=['actions'])['actions']             
    # print(gt_actions_un[0,9:12])
    
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Get single_arm_action_rep_dim from config
    if isinstance(config, _config.FTP1TrainConfig):
        single_arm_action_rep_dim = config.single_arm_action_rep_dim
    else:
        # Try to infer from model config or use default
        single_arm_action_rep_dim = getattr(model_config, 'single_arm_action_rep_dim', 32)
        logging.info(f"Using single_arm_action_rep_dim={single_arm_action_rep_dim} from model config")
    
    # Get batch count for progress bar
    val_batch_count = None
    try:
        val_torch_loader = val_data_loader._data_loader.torch_loader
        if hasattr(val_torch_loader, 'batch_sampler') and val_torch_loader.batch_sampler is not None:
            val_batch_count = len(val_torch_loader.batch_sampler)
        else:
            val_batch_count = len(val_data_loader)
    except (TypeError, AttributeError, NotImplementedError):
        val_batch_count = None
    
    # Evaluation loop
    val_infos = []
    start_time = time.time()
    
    logging.info("Starting evaluation on validation set...")
    
    pbar_val = tqdm.tqdm(
        total=val_batch_count,
        desc="Evaluating",
        dynamic_ncols=True,
        mininterval=0.1
    )
    
    val_batch_iter = iter(val_data_loader)
    val_idx = 0
    previous_tactiles: dict | None = None
    
    if val_batch_count == 0:
        logging.warning("No validation batches found!")
    else:
        with torch.no_grad():
            while True:
                if val_batch_count is not None and val_idx >= val_batch_count:
                    break
                
                try:
                    val_observation, val_actions = next(val_batch_iter)
                except StopIteration:
                    break
                
                val_idx += 1
                val_observation = move_to_device(val_observation, device)
                val_observation, previous_tactiles = apply_tactile_ablation(
                    val_observation, tactile_mode, previous_tactiles
                )
                if tactile_mode == "shuffle" and val_idx == 1:
                    # No donor batch existed yet, so this sample saw its own tactile.
                    pbar_val.update(1)
                    continue
                sample_actions = model.sample_actions(device, val_observation)
                
                gt_actions = val_actions
                action_masks = val_observation.action_masks
                
                # Get domain names from observation
                domain_names = None
                if hasattr(val_observation, 'domain_names') and val_observation.domain_names is not None:
                    if isinstance(val_observation.domain_names, list) and len(val_observation.domain_names) > 0:
                        domain_names = val_observation.domain_names[0]  # All samples have same domain
                
                # import pdb; pdb.set_trace()
                domain_idx = val_dataset.domain_list.index(domain_names)
                domain_dataset = val_dataset.domain_dataset_list[domain_idx]
                sample_data = {'actions': sample_actions.detach().cpu().numpy()}
                gt_data = {'actions': gt_actions.detach().cpu().numpy()}
                gt_actions_un = apply_norm_stats(gt_data, domain_dataset.norm_stats, unnormalize=True, use_keys=['actions'])['actions']       
                sample_actions_un = apply_norm_stats(sample_data, domain_dataset.norm_stats, unnormalize=True, use_keys=['actions'])['actions']       
                
                # if np.abs(gt_actions_un[:,0,9].mean()) > 1e-8:
                #     import pdb; pdb.set_trace()
                
                # Compute validation metrics
                val_info = compute_validation_metrics(
                    sample_actions=sample_actions,
                    gt_actions=gt_actions,
                    action_masks=action_masks,
                    domain_names=domain_names,
                    val_dataset=val_dataset,
                    single_arm_action_rep_dim=single_arm_action_rep_dim,
                    arm_joints_dim=FTP1_SINGLE_ARM_JOINT_DIM,
                    reserved_action_dim=FTP1_RESERVED_ACTION_DIM,
                )
                val_infos.append(val_info)
                pbar_val.update(1)
    
    pbar_val.close()
    
    # Aggregate validation metrics
    aggregated_metrics = aggregate_validation_metrics(val_infos)

    elapsed_time = time.time() - start_time
    
    # Log results
    logging.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    total_metrics = aggregated_metrics.get('total', {})
    if total_metrics:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in total_metrics.items()])
        logging.info(f"Validation metrics: {metrics_str}")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    aggregated_metrics = convert_to_serializable(aggregated_metrics)
    
    # Save to JSON file
    suffix = "" if tactile_mode == "real" else f"_tactile-{tactile_mode}"
    output_json_path = ckpt_dir / f"evaluation_aggregated_metrics{suffix}.json"
    with open(output_json_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    logging.info(f"Evaluation results saved to: {output_json_path}")
    
    return aggregated_metrics


def main():
    # Use tyro to parse config from command line, but also allow checkpoint_dir and device as separate args
    parser = argparse.ArgumentParser(
        description="Evaluate FTP1 model on validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint (can be a step directory or parent directory)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available, else cpu)"
    )
    
    parser.add_argument(
        "--tactile_mode",
        type=str,
        default="real",
        choices=TACTILE_MODES,
        help="Test-time tactile ablation: real | zero | noise | shuffle (default: real)",
    )

    # Parse known args first to get checkpoint_dir
    args, remaining = parser.parse_known_args()
    
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Try to load config from command line using tyro
    # If no config args provided, will be None and we'll load from checkpoint
    config = None
    if remaining:
        try:
            config = tyro.cli(_config.TrainConfig, args=remaining)
            logging.info("Loaded config from command line arguments")
        except Exception as e:
            logging.warning(f"Could not parse config from command line: {e}")
            logging.info("Will try to load config from checkpoint directory")
            config = None
    
    eval_loop(checkpoint_dir, config, args.device, tactile_mode=args.tactile_mode)


if __name__ == "__main__":
    main()
