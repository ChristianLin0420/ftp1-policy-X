"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import json
import os
import pathlib
import shutil

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main(config: _config.TrainConfig):
    # FTP1 derives its MultiZarr data factory from CLI-overridable fields such
    # as dataset_config_path. Rebuild it after tyro has applied those overrides,
    # matching the train and evaluation entrypoints.
    if isinstance(config, _config.FTP1TrainConfig):
        config.finalize_config()
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.model.action_horizon
    batch_size = config.norm_batch_size

    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    output_dir = pathlib.Path(config.assets_dirs) / data_config.repo_id
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_json_path = output_dir / "dataset_stats.json"

    train_dataset, val_dataset = _data_loader.create_torch_dataset(data_config, action_horizon, config.model)

    # Get batch_size from config (for calculating steps per epoch)
    # Try to get from config.batch_size, fallback to norm_batch_size if not available
    batch_size = getattr(config, "batch_size", config.norm_batch_size)

    def get_dataset_stats(dataset, split_name, batch_size):
        """Get statistics for a dataset split and return as structured data."""
        print(f"\n{'=' * 110}")
        print(f"Dataset Statistics - {split_name} Split (before normalization):")
        print("=" * 110)

        # Print table header
        print(f"{'Dataset':<40} {'Tasks':<10} {'Trajectories':<15} {'Frames':<15} {'Steps/Epoch':<15}")
        print("-" * 110)

        total_tasks = 0
        total_trajectories = 0
        total_frames = 0
        domains = []

        if hasattr(dataset, "domain_list") and hasattr(dataset, "domain_dataset_list"):
            # MultiZarrDataset: iterate through each domain
            for domain_name, domain_dataset in zip(dataset.domain_list, dataset.domain_dataset_list, strict=True):
                # Get current split's trajectory count (not total)
                num_frames = len(domain_dataset)
                if hasattr(domain_dataset, "indices") and hasattr(domain_dataset, "episodes_idxs"):
                    # Count unique episodes in current split
                    unique_episodes = set()
                    for idx in domain_dataset.indices:
                        if idx < len(domain_dataset.episodes_idxs):
                            unique_episodes.add(domain_dataset.episodes_idxs[idx])
                    num_trajectories = len(unique_episodes)
                else:
                    # Fallback: use n_episodes (may be inaccurate if split is used)
                    num_trajectories = domain_dataset.n_episodes

                # Get number of zarr files (tasks)
                num_tasks = len(domain_dataset.zarr_file_list) if hasattr(domain_dataset, "zarr_file_list") else 0

                steps_per_epoch = (num_frames + batch_size - 1) // batch_size  # Ceiling division
                total_tasks += num_tasks
                total_trajectories += num_trajectories
                total_frames += num_frames

                domains.append(
                    {
                        "domain_name": domain_name,
                        "tasks": num_tasks,
                        "trajectories": num_trajectories,
                        "frames": num_frames,
                        "steps_per_epoch": steps_per_epoch,
                    }
                )
                print(
                    f"{domain_name:<40} {num_tasks:<10} {num_trajectories:<15} {num_frames:<15} {steps_per_epoch:<15}"
                )
        elif hasattr(dataset, "n_episodes"):
            # Single ZarrDataset
            num_frames = len(dataset)
            if hasattr(dataset, "indices") and hasattr(dataset, "episodes_idxs"):
                # Count unique episodes in current split
                unique_episodes = set()
                for idx in dataset.indices:
                    if idx < len(dataset.episodes_idxs):
                        unique_episodes.add(dataset.episodes_idxs[idx])
                num_trajectories = len(unique_episodes)
            else:
                # Fallback: use n_episodes (may be inaccurate if split is used)
                num_trajectories = dataset.n_episodes

            # Get number of zarr files (tasks)
            num_tasks = len(dataset.zarr_file_list) if hasattr(dataset, "zarr_file_list") else 0

            steps_per_epoch = (num_frames + batch_size - 1) // batch_size
            domain_name = getattr(dataset, "domain_name", "default")

            domains.append(
                {
                    "domain_name": domain_name,
                    "tasks": num_tasks,
                    "trajectories": num_trajectories,
                    "frames": num_frames,
                    "steps_per_epoch": steps_per_epoch,
                }
            )
            print(f"{domain_name:<40} {num_tasks:<10} {num_trajectories:<15} {num_frames:<15} {steps_per_epoch:<15}")
            total_tasks = num_tasks
            total_trajectories = num_trajectories
            total_frames = num_frames

        # Print totals
        total_steps_per_epoch = (total_frames + batch_size - 1) // batch_size
        print("-" * 110)
        print(
            f"{'TOTAL':<40} {total_tasks:<10} {total_trajectories:<15} {total_frames:<15} {total_steps_per_epoch:<15}"
        )
        print("=" * 110)
        print(f"Batch Size: {batch_size}")
        print("=" * 110)

        return {
            "split_name": split_name,
            "batch_size": batch_size,
            "domains": domains,
            "total": {
                "tasks": total_tasks,
                "trajectories": total_trajectories,
                "frames": total_frames,
                "steps_per_epoch": total_steps_per_epoch,
            },
        }

    # Collect train statistics
    train_stats = get_dataset_stats(train_dataset, "TRAIN", batch_size)

    # Collect val statistics if available
    val_stats = None
    if val_dataset is not None:
        val_stats = get_dataset_stats(val_dataset, "VAL", batch_size)
    else:
        print("\n" + "=" * 100)
        print("VAL Split: Not available (use_val_dataset=False or no validation split created)")
        print("=" * 100)

    print()  # Extra newline for spacing

    # Collect normalization configuration
    norm_config = {
        "norm_time_dim": config.norm_time_dim,
        "norm_type": config.norm_type,
        "independent_norm_mode": config.independent_norm_mode,
        "norm_sample_ratio": config.norm_sample_ratio,
        "norm_batch_size": config.norm_batch_size,
        "norm_num_workers": config.norm_num_workers,
        "norm_image_tactile_mode": config.norm_image_tactile_mode,
        "norm_image_channel_pool_max_size": config.norm_image_channel_pool_max_size,
        "norm_image_channel_batch_samples": config.norm_image_channel_batch_samples,
        "contact_detection_threshold_k": config.contact_detection_threshold_k,
    }

    print("Norm-Time-Dim: ", norm_config["norm_time_dim"])
    print("Norm-Type: ", norm_config["norm_type"])
    print("Independent-Norm-Mode: ", norm_config["independent_norm_mode"])
    print("Norm-Sample-Ratio: ", norm_config["norm_sample_ratio"])
    print("Norm-Batch-Size: ", norm_config["norm_batch_size"])
    print("Norm-Num-Workers: ", norm_config["norm_num_workers"])
    print("Norm-Image-Tactile-Mode: ", norm_config["norm_image_tactile_mode"])
    print("Norm-Image-Channel-Pool-Max-Size: ", norm_config["norm_image_channel_pool_max_size"])
    print("Norm-Image-Channel-Batch-Samples: ", norm_config["norm_image_channel_batch_samples"])
    print("Contact-Detection-Threshold-K: ", norm_config["contact_detection_threshold_k"])

    # Prepare complete statistics document
    stats_doc = {
        "repo_id": data_config.repo_id,
        "assets_directory": str(data_config.assets_dirs),
        "normalization_config": norm_config,
        "train_split": train_stats,
        "val_split": val_stats,
    }

    train_dataset.generate_norm_stats(
        batch_size=config.norm_batch_size,
        independent_norm_mode=config.independent_norm_mode,
        norm_time_dim=config.norm_time_dim,
        norm_type=config.norm_type,
        sample_ratio=config.norm_sample_ratio,
        num_workers=config.norm_num_workers,
        image_tactile_norm_mode=config.norm_image_tactile_mode,
        image_channel_pool_max_size=config.norm_image_channel_pool_max_size,
        image_channel_batch_samples=config.norm_image_channel_batch_samples,
    )

    train_dataset.load_norm_stats(
        independent_norm_mode=config.independent_norm_mode,
        norm_time_dim=config.norm_time_dim,
        norm_type=config.norm_type,
    )
    print("Assets directory: ", data_config.assets_dirs)
    print("Norm stats generated and saved successfully, repo_id: ", data_config.repo_id)

    skip_contact = getattr(config, "skip_contact_detection", True)
    if not skip_contact and hasattr(train_dataset, "generate_contact_detection_thresholds"):
        stats_doc["contact_detection_thresholds"] = train_dataset.generate_contact_detection_thresholds()

    action_group_frequency_stats = train_dataset.generate_action_group_frequency_stats(
        batch_size=config.norm_batch_size,
        sample_ratio=config.norm_sample_ratio,
        num_workers=config.norm_num_workers,
        split="train",
    )
    stats_doc["action_group_frequency_stats"] = action_group_frequency_stats
    stats_doc["action_group_frequency_stats_path"] = str(train_dataset.default_action_group_frequency_stats_path)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        if isinstance(obj, pathlib.Path):
            return str(obj)
        return obj

    stats_doc_serializable = convert_to_json_serializable(stats_doc)
    with open(stats_json_path, "w") as f:
        json.dump(stats_doc_serializable, f, indent=2)
    print(f"Dataset statistics saved to: {stats_json_path}")

    # Save norm params snapshot so training can assert they match
    norm_snapshot = _config.get_norm_params_snapshot(config)
    if norm_snapshot is not None:
        snapshot_path = output_dir / "norm_params_snapshot.json"
        snapshot_serializable = convert_to_json_serializable(norm_snapshot)
        with open(snapshot_path, "w") as f:
            json.dump(snapshot_serializable, f, indent=2)
        print(f"Norm params snapshot saved to: {snapshot_path}")

    train_dataset.generate_tactile_input_config()
    print("Tactile input config generated and saved successfully, repo_id: ", data_config.repo_id)

    _, val_dataset = _data_loader.create_torch_dataset(data_config, action_horizon, config.model)
    val_dataset.load_norm_stats(
        independent_norm_mode=config.independent_norm_mode,
        norm_time_dim=config.norm_time_dim,
        norm_type=config.norm_type,
    )
    print("Val dataset norm stats generated and saved successfully, repo_id: ", data_config.repo_id)
    # data = val_dataset[0]
    # import pdb; pdb.set_trace()
    # from openpi.normalization import apply_norm_stats
    # desampled_data = apply_norm_stats(data, val_dataset.domain_dataset_list[0].norm_stats, unnormalize=True)
    # import pdb; pdb.set_trace()
    # for i in range(len(val_dataset)):
    #     data = val_dataset[i]
    #     print(data.keys())
    #     import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(_config.cli())
