"""
UniVTAC HDF5 -> FTP1 Zarr parser.

- Reads UniVTAC HDF5 episodes containing joint state, RGB observations, and
  tactile image streams.
- Writes RGB tensors to zarr; raw JPEG decoding yields BGR, which is converted
  to RGB before export for consistency with eval/inference pipelines.
- Exports only the joint-only control representation:
  7 arm dimensions + 1 gripper dimension.
- Supports both tactile key conventions:
  `left/right_tactile` and `left/right_gsmini`.
- Preserves numeric HDF5 filename stems as per-episode `meta/source_episode_seed` identity.
- Supports `--episodes_per_task` to cap the number of trajectories per task.
- If an earlier version produced BGR zarr outputs, regenerate them with this
  script before training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Add the data_processing root to sys.path so `common` imports work when the
# module is run as a script.
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_PROCESSING_ROOT = _SCRIPT_DIR.parent
if str(_DATA_PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_PROCESSING_ROOT))

from common.replay_buffer import ReplayBuffer  # noqa: E402

# Default FAAS gripper slot, aligned with other FTP1 parsers.
DEFAULT_GRIPPER_IDX = 28
# Tactile sensor name used by T3 pretraining mapping.
TACTILE_SENSOR_NAME = "GelSightMini"
INSTRUCTION_PAD_LEN = 100
IMAGE_SIZE = 224
VIDEO_FPS = 10


def _save_video(
    frames: np.ndarray,
    base_path: Path,
    episode_idx: int,
    fps: int = VIDEO_FPS,
) -> Path | None:
    """Write a `camera_ego_rgb` frame sequence to mp4."""
    if frames is None or len(frames) == 0:
        return None
    out_file = f"{base_path.stem}_ep{episode_idx:04d}.mp4"
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (W, H))
    for t in range(T):
        frame = frames[t]
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    return out_file


def _stream_to_img(data: np.ndarray, out_size: tuple[int, int] | None = (IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """Decode JPEG byte streams from HDF5 into `(N, H, W, 3)` uint8 RGB arrays."""
    flat = data.ravel()
    imgs = []
    for buf in flat:
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            try:
                arr = np.frombuffer(bytes(buf), dtype=np.uint8)
            except Exception as e:
                raise TypeError(f"Unsupported buffer type: {type(buf)}") from e
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from buffer")
        if out_size:
            img = cv2.resize(img, (out_size[1], out_size[0]))  # (width, height)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return np.stack(imgs, axis=0).astype(np.uint8)


def _load_univtac_episode(
    hdf5_path: str | Path,
    downsample: int = 1,
    image_size: int = IMAGE_SIZE,
    use_wrist: bool = True,
) -> dict[str, np.ndarray] | None:
    """Load one UniVTAC HDF5 episode and return an FTP1-compatible episode dict."""
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        return None

    with h5py.File(hdf5_path, "r") as f:
        # import pdb; pdb.set_trace()
        if "embodiment" not in f or "joint" not in f["embodiment"]:
            return None
        joint = np.array(f["embodiment"]["joint"][()], dtype=np.float32)  # (T, 9)
        T_full = joint.shape[0]
        if T_full < 2:
            return None

        # 下采样索引：保留 [0, downsample, 2*downsample, ...]，且保证最后一帧用于 action
        arange = np.arange(0, T_full - 1, downsample)
        if len(arange) == 0:
            return None
        # state = joint[t], action = joint[t+1] -> 我们保留的帧数为 len(arange)，state 与 obs 对齐
        # 与 TactileACT 一致：8D = 7 arm + 1 gripper，直接取 joint[:8]（第 8 维为单指 joint[7]），不做两指平均
        T = len(arange)
        joint_state = joint[arange]   # (T, 9) 或 (T, 8)
        joint_8 = joint_state[:, :8].astype(np.float32)   # (T, 8)：7 arm + 1 gripper

        # V3 control collection records the exact command and task phase.  Historical UniVTAC
        # files do not, so keep them usable with an explicit validity mask instead of silently
        # pretending observed qpos was the commanded target.
        if "command" in f["embodiment"]:
            command_full = np.asarray(f["embodiment"]["command"][()], dtype=np.float32)
            if command_full.shape != (T_full, 8):
                raise ValueError(f"embodiment/command must be {(T_full, 8)}, got {command_full.shape}")
            command_8 = command_full[arange]
            command_valid = np.ones(T, dtype=np.uint8)
        else:
            command_8 = np.concatenate([joint_8[:1], joint_8[:-1]], axis=0)
            command_valid = np.zeros(T, dtype=np.uint8)

        if "step" in f:
            control_step_full = np.asarray(f["step"][()], dtype=np.int64).reshape(-1)
            if control_step_full.shape != (T_full,):
                raise ValueError(f"step must be {(T_full,)}, got {control_step_full.shape}")
            control_step = control_step_full[arange]
            control_step_valid = np.ones(T, dtype=np.uint8)
        else:
            control_step = np.arange(T, dtype=np.int64)
            control_step_valid = np.zeros(T, dtype=np.uint8)

        control = f.get("control")
        if control is not None and "phase" in control:
            phase_id = np.asarray(control["phase"][()], dtype=np.int64)[arange]
            if np.any((phase_id < 0) | (phase_id > 3)):
                raise ValueError(f"control/phase contains values outside [0, 3] in {hdf5_path}")
        else:
            phase_id = _infer_control_phase(joint_8[:, 7])
        if control is not None and "contact" in control:
            contact = np.asarray(control["contact"][()], dtype=np.float32)[arange]
            contact_valid = np.ones(T, dtype=np.uint8)
        else:
            contact = np.zeros(T, dtype=np.float32)
            contact_valid = np.zeros(T, dtype=np.uint8)

        def _read_rgb(*path_parts: str):
            """读取 HDF5 中图像路径并解码，key 不存在或解码失败时 raise，不返回 None。"""
            path_str = "/".join(path_parts)
            try:
                node = f
                for p in path_parts:
                    node = node[p]
                raw = node[()]
                return _stream_to_img(raw, (image_size, image_size))
            except (KeyError, TypeError) as e:
                raise KeyError(f"HDF5 missing or invalid key: {path_str} in {hdf5_path}") from e

        def _read_rgb_with_fallbacks(path_options: list[tuple[str, ...]]) -> np.ndarray:
            last_error = None
            for option in path_options:
                try:
                    return _read_rgb(*option)
                except KeyError as e:
                    last_error = e
            tried = ", ".join("/".join(p) for p in path_options)
            raise KeyError(f"HDF5 missing or invalid key, tried: {tried} in {hdf5_path}") from last_error

        head_rgb = _read_rgb("observation", "head", "rgb")[...,::-1]
        # wrist camera 仅在需要时读取；若 use_wrist=False 则完全忽略
        wrist_rgb = None
        if use_wrist:
            wrist_rgb = _read_rgb("observation", "wrist", "rgb")[...,::-1]
        left_tac = _read_rgb_with_fallbacks(
            [
                ("tactile", "left_gsmini", "rgb_marker"),
                ("tactile", "left_tactile", "rgb_marker"),
            ]
        )[..., ::-1]
        right_tac = _read_rgb_with_fallbacks(
            [
                ("tactile", "right_gsmini", "rgb_marker"),
                ("tactile", "right_tactile", "rgb_marker"),
            ]
        )[..., ::-1]
        # import pdb; pdb.set_trace()
        # from PIL import Image
        # Image.fromarray(head_rgb[0]).save("head_rgb.jpg")
        # Image.fromarray(wrist_rgb[0]).save("wrist_rgb.jpg")
        # Image.fromarray(left_tac[0]).save("left_tac.jpg")
        # Image.fromarray(right_tac[0]).save("right_tac.jpg")

        if head_rgb is None or head_rgb.shape[0] != T_full:
            return None
        head_rgb = head_rgb[arange]   # (T, H, W, 3), already image_size from _read_rgb
        if wrist_rgb is not None:
            wrist_rgb = wrist_rgb[arange]
        right_tac = right_tac[arange][:, np.newaxis, ...]   # (T, 1, H, W, 3)
        left_tac = left_tac[arange][:, np.newaxis, ...]   # (T, 1, H, W, 3)

        # 两只触觉都是右手上的两个 pad，叠成 N=2：(T, 2, H, W, 3)，area [0, 1]
        # 先left后right，因为left对应thumb, right对应index
        right_tac = np.concatenate([left_tac, right_tac], axis=1)   # (T, 2, H, W, 3)

    # 时间戳：无原始时间戳则用步索引
    timestamps = np.arange(T, dtype=np.int64)

    right_arm_joints = joint_8[:, :7].astype(np.float32)
    right_hand_joints = joint_8[:, 7:8].astype(np.float32)
    right_hand_joints_idx = np.full((T, 1), DEFAULT_GRIPPER_IDX, dtype=np.int32)

    # 触觉：右手夹爪两侧两个 pad（HDF5 里为 left_tactile / right_tactile），叠成 N=2，area [0, 1]
    right_tactile_area_gripper = np.tile(np.array([[0, 1]], dtype=np.int64), (T, 1))   # (T, 2)
    right_tactile_sensor_gripper = np.array([TACTILE_SENSOR_NAME] * T)
    right_tactile_type_gripper = np.array(["image"] * T)

    episode = {
        "timestamps": timestamps,
        "camera_ego_rgb": head_rgb,
        "right_arm_joints": right_arm_joints,
        "right_hand_joints": right_hand_joints,
        "right_hand_joints_idx": right_hand_joints_idx,
        "right_tactile_data_gripper": right_tac.astype(np.uint8),
        "right_tactile_area_gripper": right_tactile_area_gripper,
        "right_tactile_sensor_gripper": right_tactile_sensor_gripper,
        "right_tactile_type_gripper": right_tactile_type_gripper,
        # MoT-Control V3-only fields.  FTP-1 ignores unknown arrays; the derived clip builder
        # preserves them for the standalone action decoder.
        "command8": command_8.astype(np.float32),
        "command_valid": command_valid,
        "phase_id": phase_id.astype(np.int64),
        "contact": contact.astype(np.float32),
        "contact_valid": contact_valid,
        "control_step": control_step,
        "control_step_valid": control_step_valid,
    }
    if wrist_rgb is not None:
        episode["right_wrist_camera_rgb"] = wrist_rgb

    return episode


def _infer_control_phase(gripper: np.ndarray, *, threshold: float = 1e-5) -> np.ndarray:
    """Best-effort four-phase labels for legacy demonstrations.

    New V3 episodes carry authoritative simulator phase ids.  The legacy 100-episode release has
    only qpos, but its lift-bottle script is monotonic: settle, close, manipulate, release.  This
    fallback makes the old data usable while remaining visibly weaker through the missing command
    and contact validity masks.
    """
    values = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    delta = np.diff(values, prepend=values[0])
    closing = np.flatnonzero(delta < -threshold)
    opening = np.flatnonzero((delta > threshold) & (np.arange(values.size) > values.size // 2))
    close_start = int(closing[0]) if closing.size else min(1, values.size)
    release_start = int(opening[0]) if opening.size else values.size

    phase = np.full(values.size, 2, dtype=np.int64)
    phase[:close_start] = 0
    if closing.size:
        # Retain one settled post-close frame as LIFT whenever the episode provides one.  This
        # avoids assigning the entire grasp-to-release interval to CLOSE on short legacy demos.
        close_stop = min(int(closing[-1]) + 2, release_start)
        if release_start > close_start + 1:
            close_stop = min(close_stop, release_start - 1)
        phase[close_start:close_stop] = 1
    phase[release_start:] = 3
    return phase


# The upstream repository historically used demo/hdf5/. The published UniVTAC
# dataset uses clean/. Accept both without requiring users to rearrange downloads.
HDF5_SUBDIR_CANDIDATES = ("demo/hdf5", "clean")


def _numeric_episode_seed(path: Path) -> int | None:
    """Return the non-negative integer encoded by an HDF5 filename stem, if any."""
    stem = path.stem
    if not stem.isascii() or not stem.isdecimal():
        return None
    seed = int(stem)
    if seed > np.iinfo(np.int64).max:
        return None
    return seed


def _is_authoritative_v3_episode(episode: dict[str, np.ndarray]) -> bool:
    """Whether an episode carries any authoritative V3 control supervision."""
    validity_keys = ("command_valid", "contact_valid", "control_step_valid")
    return any(bool(np.asarray(episode.get(key, []), dtype=bool).any()) for key in validity_keys)


def _source_episode_seed_array(records: list[tuple[Path, bool]]) -> np.ndarray | None:
    """Build stable episode metadata, requiring it for authoritative V3 collections.

    Legacy UniVTAC releases may use descriptive filenames, so their stores remain readable and
    simply omit this optional metadata.  Once any accepted episode is V3-authoritative, however,
    every accepted episode must have a unique numeric source filename: a partial identity vector
    would be worse than a loud conversion failure.
    """
    if not records:
        return None
    seeds = [_numeric_episode_seed(path) for path, _authoritative in records]
    authoritative = any(is_authoritative for _path, is_authoritative in records)
    invalid = [path.name for (path, _is_authoritative), seed in zip(records, seeds, strict=True) if seed is None]
    numeric = [seed for seed in seeds if seed is not None]
    seen: set[int] = set()
    duplicate: set[int] = set()
    for seed in numeric:
        if seed in seen:
            duplicate.add(seed)
        seen.add(seed)
    if authoritative and invalid:
        raise ValueError(f"authoritative V3 HDF5 filenames must have numeric int64 stems; invalid: {invalid}")
    if authoritative and duplicate:
        raise ValueError(f"authoritative V3 HDF5 filenames encode duplicate episode seeds: {sorted(duplicate)}")
    if invalid or duplicate:
        return None
    return np.asarray(numeric, dtype=np.int64)


def _discover_task_dirs(base_dir: Path, task_list: list[str] | None) -> list[tuple[str, Path, Path]]:
    """
    发现所有包含 demo/hdf5/*.hdf5 的 task 目录。
    Support both base_dir/<task>/demo/hdf5/<id>.hdf5 and the released
    base_dir/<task>/clean/<id>.hdf5 layout.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []

    results: list[tuple[str, Path, Path]] = []
    try:
        for first in sorted(base_dir.iterdir()):
            if not first.is_dir():
                continue
            hdf5_dir = next(
                (
                    first / subdir
                    for subdir in HDF5_SUBDIR_CANDIDATES
                    if (first / subdir).is_dir() and any((first / subdir).glob("*.hdf5"))
                ),
                None,
            )
            if hdf5_dir is None:
                continue
            task_id = first.name
            if task_list is not None and task_id not in task_list:
                continue
            results.append((task_id, first, hdf5_dir))
    except OSError:
        pass
    return results


def _run(
    base_dir: str,
    save_dir: str,
    episodes_per_task: int | None = None,
    task_list: list[str] | None = None,
    downsample: int = 1,
    image_size: int = IMAGE_SIZE,
) -> None:
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 读取本地 task_settings.json，用于判断是否需要腕部相机（camera_type == 'all' 才保留 wrist）
    settings_path = Path(__file__).parent / "task_settings.json"
    if settings_path.exists():
        with open(settings_path, "r") as f:
            task_settings = json.load(f)
    else:
        task_settings = {}
    task_dirs = _discover_task_dirs(base_dir, task_list)
    if not task_dirs:
        layouts = " or ".join(f"base_dir/<task>/{subdir}/<id>.hdf5" for subdir in HDF5_SUBDIR_CANDIDATES)
        print(f"No UniVTAC task HDF5 directories found under {base_dir}")
        print(f"  Expected layout: {layouts}")
        return

    instruction_map = {
        "grasp_classify": "use grasped tool for tactile sensing to move to target surface.",
        "insert_HDMI": "insert the HDMI to the fixed slot.",
        "insert_hole": "insert the stick to the hole.",
        "insert_tube": "insert the tube to the fixed slot.",
        "lift_can": "grasp the can and lifts it vertically without slippage.",
        "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base within 5 cm of the wall.",
        "pull_out_key": "pull out the key.",
        "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
    }

    for task_id, _task_dir, hdf5_dir in task_dirs:
        def _episode_sort_key(p: Path) -> tuple[bool, int, str]:
            seed = _numeric_episode_seed(p)
            return (seed is None, seed if seed is not None else 0, p.name)

        hdf5_files = sorted(hdf5_dir.glob("*.hdf5"), key=_episode_sort_key)
        if not hdf5_files:
            continue

        count = 0
        # 根据 task_settings 判定是否使用 wrist camera
        cam_cfg = task_settings.get(task_id, {})
        use_wrist = cam_cfg.get("camera_type", "head") == "all"

        # 每任务一个 zarr 文件，文件名中 / 替换为 _
        safe_id = task_id.replace("/", "_")
        zarr_path = save_dir / f"{safe_id}"
        if use_wrist:
            zarr_path = str(zarr_path) + "_all.zarr"
        else:
            zarr_path = str(zarr_path) + "_head.zarr"
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        replay_buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="a")

        source_episode_records: list[tuple[Path, bool]] = []
        for h5_path in tqdm(hdf5_files, desc=f"UniVTAC {task_id}"):
            if episodes_per_task is not None and count >= episodes_per_task:
                break
            episode = _load_univtac_episode(
                h5_path,
                downsample=downsample,
                image_size=image_size,
                use_wrist=use_wrist,
            )
            if episode is None:
                continue
            instruction_org = instruction_map.get(task_id, f"solve the task: {task_id}.")
            instruction = instruction_org.ljust(INSTRUCTION_PAD_LEN)
            episode["sub_task_instruction"] = np.array([instruction] * len(episode["timestamps"]))

            # print(f"====================================== Task: {task_id} ======================================")
            # _save_video(episode["camera_ego_rgb"], zarr_path, count, fps=VIDEO_FPS)
            # for key in episode.keys():
            #     print(f"{key}: {episode[key].shape}")
            # print(f"instruction: {instruction}")
            # timestamps: (56,)
            # camera_ego_rgb: (56, 224, 224, 3)
            # right_arm_joints: (56, 7)
            # right_hand_joints: (56, 1)
            # right_hand_joints_idx: (56, 1)
            # right_tactile_data_gripper: (56, 2, 224, 224, 3)
            # right_tactile_area_gripper: (56, 2)
            # right_tactile_sensor_gripper: (56,)
            # right_tactile_type_gripper: (56,)
            # right_wrist_camera_rgb: (56, 224, 224, 3)
            # sub_task_instruction: (56,)
            # import pdb; pdb.set_trace()
            replay_buffer.add_episode(episode, compressors="disk")
            source_episode_records.append((h5_path, _is_authoritative_v3_episode(episode)))
            count += 1
            # break

        source_episode_seeds = _source_episode_seed_array(source_episode_records)
        if source_episode_seeds is not None:
            chunk_size = max(1, min(1024, int(source_episode_seeds.size)))
            replay_buffer.root["meta"].create_array(
                name="source_episode_seed",
                data=source_episode_seeds,
                chunks=(chunk_size,),
                compressor=None,
            )

        print(f"  {task_id}: wrote {count} episodes -> {zarr_path}")


def main():
    parser = argparse.ArgumentParser(description="UniVTAC HDF5 -> FTP1 Zarr (joint control only).")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="UniVTAC data root containing per-task HDF5 directories.",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for exported zarr files.")
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=None,
        help="Optional cap on the number of trajectories retained per task.",
    )
    parser.add_argument(
        "--task_list",
        type=str,
        default=None,
        help="Optional comma-separated subset of task names to process.",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Temporal downsample factor.")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE, help="Output size for RGB and tactile images.")
    args = parser.parse_args()

    task_list = None
    if args.task_list:
        task_list = [s.strip() for s in args.task_list.split(",") if s.strip()]

    _run(
        base_dir=args.base_dir,
        save_dir=args.save_dir,
        episodes_per_task=args.episodes_per_task,
        task_list=task_list,
        downsample=args.downsample,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
