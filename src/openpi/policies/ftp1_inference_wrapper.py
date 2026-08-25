"""
FTP1 Inference Wrapper

This module provides a wrapper class for FTP1 model inference, handling:
1. Model loading from checkpoint
2. Input normalization (images, state, tactile)
3. Output denormalization (actions)

================================================================================
INPUT FORMAT SPECIFICATION
================================================================================

1. images: dict[str, np.ndarray]
   -------------------------------------------------------------------------
   Camera RGB images. Keys should match the training data format.

   Format: {camera_name: image_array}

   - image_array shape: (H, W, 3) or (B, H, W, 3)
   - dtype: uint8 [0-255] or float32 [-1, 1] (auto-detected)
   - Resolution: Images are forwarded as-is; the SigLIP vision tower expects
     224x224, so resize before calling if your raw images are at a different
     resolution.

   Common camera names (must match what the model was trained on):
   - "camera_ego_rgb_0" / "base_0_rgb": Third-person/ego camera
   - "left_wrist_camera_rgb_0" / "left_wrist_0_rgb": Left wrist camera
   - "right_wrist_camera_rgb_0" / "right_wrist_0_rgb": Right wrist camera

   Example:
       images = {
           "camera_ego_rgb_0": np.zeros((224, 224, 3), dtype=np.uint8),
           "right_wrist_camera_rgb_0": np.zeros((224, 224, 3), dtype=np.uint8),
       }

2. state: np.ndarray
   -------------------------------------------------------------------------
   Robot proprioceptive state (wrist poses, arm joints, hand joints, head pose,
   supplementary joints).

   Shape: (T, state_dim) where:
   - T: Number of history time steps (1 is standard; FTP1 defaults to
     `disable_history=True`, so only the current state is consumed).
   - state_dim: Equal to the model's `action_dim` (see below).

   dtype: float32, RAW SCALE (not normalized).

   state_dim is derived from the training config and is NOT hard-coded to any
   single number. Use `wrapper.get_state_dim()` to read it from the loaded
   model. A common value is:

   FTP1 default (dual-arm + head + reserved):
       state_dim = 2 * single_arm_action_rep_dim + 9 + reserved_action_dim
                 = 2 * (9 + 7 + 32) + 9 + 15
                 = 2 * 48 + 9 + 15
                 = 120

       Canonical layout (matches `ftp1_action_groups.get_ftp1_action_group_slices`):
       - [  0:  3]  right wrist position (xyz)
       - [  3:  9]  right wrist rotation (6D)          → [0:9]    right wrist pose9d
       - [  9: 16]  right arm joints                   (7 DoF, padded to 32)
       - [ 16: 48]  right hand joints                  (32-slot canonical vector)
       - [ 48: 51]  left wrist position
       - [ 51: 57]  left wrist rotation (6D)           → [48:57]  left wrist pose9d
       - [ 57: 64]  left arm joints                    (7 DoF, padded to 32)
       - [ 64: 96]  left hand joints                   (32-slot canonical vector)
       - [ 96: 99]  head/ego position
       - [ 99:105]  head/ego rotation (6D)             → [96:105] head pose9d
       - [105:120]  supplementary joints               (reserved, up to 15 DoF)

   Important — hand DoF < 32 handling:
   - FTP1 reserves 32 canonical hand slots (`FTP1_SINGLE_HAND_JOINT_DIM = 32`).
     Robots with fewer hand DoFs (e.g., Sharpa with 22) still emit 32-slot
     vectors: `dataset_zarr.py` builds them via
         state[:, 9 + FTP1_SINGLE_ARM_JOINT_DIM + hand_idx] = joints
     i.e., slots start at offset 16 (after the 9-dim wrist and 7-dim arm block)
     and are written at the per-robot `hand_joints_idx` positions. Unused slots
     stay zero.
   - When constructing `state` manually for inference, you must provide the
     **already-expanded** 32-slot hand vectors at [16:48] (right) and [64:96]
     (left), matching the same canonical indexing used at training time.

   Example:
       state_dim = wrapper.get_state_dim()                 # e.g., 120
       state = np.zeros((1, state_dim), dtype=np.float32)  # current state only

3. prompt: str
   -------------------------------------------------------------------------
   Natural language task instruction.

   Example:
       prompt = "pick up the red cube and place it in the box"

4. tactiles: dict[str, np.ndarray] (OPTIONAL)
   -------------------------------------------------------------------------
   Tactile sensor readings. Only required if model was trained with tactile.

   Format: {tactile_sensor_key: tactile_array}

   Shape depends on tactile sensor type (defined in tactile_input_config_file.json):

   - type="image" (e.g., FreeTacMan, GelSight, ViTaMIn):
     Shape: (T, num_areas, H, W, 3) where T=1, H=W=224
     Example: (1, 2, 224, 224, 3) for 2 tactile areas

   - type="matrix" (e.g., 3DViTac, pressure arrays):
     Shape: (T, num_areas, rows, cols)
     Example: (1, 2, 12, 32) for 2 areas with 12x32 taxels

   - type="binary" (e.g., contact sensors):
     Shape: (T, num_areas, num_sensors)
     Binary values (0 or 1)

   Example:
       tactiles = {
           "right_tactile_gripper": np.zeros((1, 2, 224, 224, 3), dtype=np.float32),
       }

5. tactile_function_areas: dict[str, list[int]] (OPTIONAL, required if tactiles provided)
   -------------------------------------------------------------------------
   Maps each tactile sensor to its "function areas" - indices indicating which
   physical contact regions the sensor covers.

   Function areas are used for:
   - Positional encoding: Different areas get different position embeddings
   - Grouping: Model knows which tactile readings belong to which finger/region

   Common function area assignments:
   - [0]: Single area (e.g., palm sensor)
   - [0, 1]: Two areas (e.g., fingertip + finger pad)
   - [0, 1, 2, 3, 4]: Five areas (e.g., 5-finger tactile glove)

   The area indices should match the second dimension of the tactile array.
   If tactile shape is (1, 2, 224, 224, 3), function_areas should have 2 elements.

   Example:
       tactile_function_areas = {
           "right_tactile_gripper": [0, 1],  # Two tactile areas
       }

6. tactile_sensors: dict[str, str] (OPTIONAL, required if tactiles provided)
   -------------------------------------------------------------------------
   Maps each tactile key to the sensor type/name. This is used to select
   the correct encoder (CNN for matrix, ViT for image, MLP for binary).

   Sensor names should match those in tactile_input_config_file.json.

   Common sensor types:
   - "FreeTacMan": Vision-based tactile sensor (image type)
   - "GelSight": Vision-based tactile sensor (image type)
   - "ViTaMIn": Vision-based tactile sensor (image type)
   - "3DViTac": Pressure matrix sensor (matrix type)
   - "BinaryContact": Simple contact sensor (binary type)

   Example:
       tactile_sensors = {
           "right_tactile_gripper": "FreeTacMan",
       }

================================================================================
OUTPUT FORMAT
================================================================================

action: np.ndarray
   Shape: (action_horizon, action_dim)
   - action_horizon: Number of future action steps. FTP1 default is 32
     (`FTP1_ACTION_HORIZON`). Use `wrapper.get_action_horizon()` to read it.
   - action_dim: Same as state_dim; use `wrapper.get_action_dim()`.

   dtype: float32, RAW SCALE (denormalized)

   Action structure (FTP1 default, action_dim=120; same layout as state):
   - [  0:  9]  right wrist action pose9d
   - [  9: 16]  right arm joint action                (7 DoF, padded to 32)
   - [ 16: 48]  right hand joint action               (32-slot canonical vector)
   - [ 48: 57]  left wrist action pose9d
   - [ 57: 64]  left arm joint action                 (7 DoF, padded to 32)
   - [ 64: 96]  left hand joint action                (32-slot canonical vector)
   - [ 96:105]  head/ego action pose9d
   - [105:120]  supplementary joint action            (reserved)

   Relative vs absolute semantics depend on the training config:
   - `action_pose_rep` controls wrist/head pose9d (relative or absolute).
   - `action_joint_rep` controls arm/hand/supplementary joint deltas:
       - "absolute": all joint segments absolute.
       - "relative": arm, hand, and supplementary all stay relative to the
         current state.
       - "mix":       arm joints stay relative; hand joints stay relative
         EXCEPT the gripper slot (slot index 28, `FTP1_GRIPPER_HAND_SLOT_INDEX`)
         which stays absolute; supplementary joints stay absolute.
   - For a programmatic layout lookup, see
     `openpi.ftp1_action_groups.get_ftp1_action_group_slices`.

USAGE EXAMPLE
================================================================================

    from openpi.policies import FTP1InferenceWrapper
    import numpy as np

    # Initialize wrapper
    wrapper = FTP1InferenceWrapper(
        checkpoint_dir="/path/to/checkpoint/7999",
        domain_name="FreeTacMan_Train",  # Must match a folder in checkpoint/normalization/
        device="cuda",
        num_inference_steps=10,
    )

    # Prepare inputs
    images = {
        "camera_ego_rgb_0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "right_wrist_camera_rgb_0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    }
    state_dim = wrapper.get_state_dim()  # e.g., 120 for FTP1
    state = np.random.randn(1, state_dim).astype(np.float32)
    prompt = "pick up the red cube"

    # With tactile input (optional, only if the checkpoint was trained with tactile)
    tactiles = {
        "right_tactile_gripper": np.random.randn(1, 2, 224, 224, 3).astype(np.float32),
    }
    tactile_function_areas = {"right_tactile_gripper": [0, 1]}
    tactile_sensors = {"right_tactile_gripper": "FreeTacMan"}

    # Run inference
    action = wrapper.infer(
        images=images,
        state=state,
        prompt=prompt,
        tactiles=tactiles,
        tactile_function_areas=tactile_function_areas,
        tactile_sensors=tactile_sensors,
    )

    # Shape: (action_horizon, action_dim), e.g., (32, 120) for FTP1 default.
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

================================================================================
NORMALIZATION FLOW
================================================================================

Training time (data pipeline):
    raw data → domain-independent norm → shared norm → model

Inference time (this wrapper):
    Input:  raw data → domain-independent norm → shared norm → model
    Output: model prediction → shared denorm → domain-independent denorm → raw scale

================================================================================
"""

import json
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import torch

# Add project root to path for imports
_project_root = pathlib.Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.openpi.models.model import Observation
from src.openpi.models.tokenizer import PaligemmaTokenizer
from src.openpi.normalization import apply_norm_stats

logger = logging.getLogger(__name__)


def norm_stats_from_jsonable(data: dict) -> dict:
    """Convert JSON-loaded norm stats to the format expected by apply_norm_stats.

    This function parses the norm_stats JSON format and converts arrays to numpy.
    """
    if not data or not data.get('params'):
        return None

    result = {
        'norm_time_dim': data.get('norm_time_dim', False),
        'norm_type': data.get('norm_type', 'quantile'),
        'params': {},
        'norm_dim': data.get('norm_dim', {}),
        'data_dim': data.get('data_dim', {}),
    }

    # Convert params to proper format
    for key, value in data.get('params', {}).items():
        if isinstance(value, dict):
            # Check if it's a tactile dict (nested) or a quantile params dict
            if 'q01' in value or 'q99' in value or 'mean' in value or 'std' in value or 'median' in value or 'mad' in value:
                # This is a quantile/z-score params dict
                result['params'][key] = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in value.items()
                }
            else:
                # This is a nested dict (e.g., tactile sensors)
                result['params'][key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        result['params'][key][sub_key] = {
                            k: np.array(v) if isinstance(v, list) else v
                            for k, v in sub_value.items()
                        }
                    else:
                        # Special values like 'image_norm', 'ignore'
                        result['params'][key][sub_key] = sub_value
        elif isinstance(value, str):
            # Special values like 'image_norm', 'ignore'
            result['params'][key] = value

    return result


class FTP1InferenceWrapper:
    """Wrapper for FTP1 model inference with normalization handling.

    This class loads a trained FTP1 model from a checkpoint directory and
    handles input normalization and output denormalization automatically.

    Attributes:
        model: The loaded FTP1Pytorch model instance.
        model_config: The FTP1ModelConfig used to create the model.
        domain_name: The domain name used for normalization statistics.
        device: The device (cuda/cpu) the model is loaded on.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        domain_name: str,
        tactile_input_config_file: str | None = None,
        device: str = "cuda",
        num_inference_steps: int = 10,
        skip_normalization: bool = False,
    ):
        """Initialize the FTP1 inference wrapper.

        Args:
            checkpoint_dir: Path to the checkpoint directory (step directory).
            domain_name: Domain name for normalization statistics
                        (e.g., "FreeTacMan_Train", "TouchInTheWild_Train").
            tactile_input_config_file: Optional path to tactile input config file.
                                      If None, will try to find it in checkpoint_dir.
            device: Device to load model on ("cuda" or "cpu").
            num_inference_steps: Number of denoising steps for inference (default: 10).
            skip_normalization: If True, skip input/output normalization (default: False).
        """
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.domain_name = domain_name
        self.tactile_input_config_file = tactile_input_config_file
        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self.skip_normalization = skip_normalization

        # These will be set during initialization
        self.model = None
        self.model_config = None
        self.ckpt_dir = None
        self.tokenizer = None

        # Normalization statistics
        self.domain_norm_stats = None
        self.shared_norm_stats = None
        self.tactile_key_type_map: dict[str, str] = {}

        # Initialize
        self._load_model()
        self._load_tactile_type_map()
        if not self.skip_normalization:
            self._load_norm_stats()
        self._load_tokenizer()

        logger.info("FTP1InferenceWrapper initialized successfully")
        logger.info(f"  Checkpoint: {self.ckpt_dir}")
        logger.info(f"  Domain: {self.domain_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Inference steps: {self.num_inference_steps}")
        logger.info(f"  Skip normalization: {self.skip_normalization}")

    def _load_model(self) -> None:
        """Load the FTP1 model from checkpoint (via openpi load_utils to avoid tyro/Isaac typing_extensions conflict)."""
        from openpi.policies.ftp1_load_utils import load_ftp1_model

        logger.info(f"Loading FTP1 model from {self.checkpoint_dir}")

        self.model, self.model_config, self.ckpt_dir = load_ftp1_model(
            checkpoint_dir=self.checkpoint_dir,
            tactile_input_config_file=self.tactile_input_config_file,
            device=str(self.device),
        )
        self.model.eval()

        logger.info(f"Model loaded from {self.ckpt_dir}")
        logger.info(f"  Action dim: {self.model_config.action_dim}")
        logger.info(f"  Action horizon: {self.model_config.action_horizon}")
        logger.info(f"  Use tactile: {self.model_config.use_tactile_input}")

    def _load_tactile_type_map(self) -> None:
        """Load tactile key -> type mapping from tactile_input_config_file.json."""
        tactile_config_path = self.tactile_input_config_file or self.model_config.tactile_input_config_file
        if tactile_config_path is None:
            return

        tactile_config_path = pathlib.Path(tactile_config_path)
        if not tactile_config_path.exists():
            logger.warning(f"Tactile input config file not found: {tactile_config_path}")
            return

        with open(tactile_config_path, "r") as f:
            tactile_config = json.load(f)

        domain_config = tactile_config.get(self.domain_name)
        if domain_config is None and len(tactile_config) == 1:
            only_domain_name, domain_config = next(iter(tactile_config.items()))
            logger.warning(
                "Domain %s not found in %s; using the only tactile config domain %s",
                self.domain_name,
                tactile_config_path,
                only_domain_name,
            )
        if domain_config is None:
            logger.warning(
                "Domain %s not found in tactile config %s; runtime tactile_type lookup will be unavailable",
                self.domain_name,
                tactile_config_path,
            )
            return

        self.tactile_key_type_map = {
            tactile_key: str(tactile_entry["type"])
            for tactile_key, tactile_entry in domain_config.items()
            if isinstance(tactile_entry, dict) and "type" in tactile_entry
        }

    def _resolve_runtime_tactile_types(self, tactile_keys: list[str]) -> dict[str, str] | None:
        if not tactile_keys:
            return None
        resolved = {
            tactile_key: self.tactile_key_type_map[tactile_key]
            for tactile_key in tactile_keys
            if tactile_key in self.tactile_key_type_map
        }
        return resolved or None

    def _should_ignore_tactiles(self) -> tuple[bool, str | None]:
        """Return whether runtime tactile inputs should be ignored for this checkpoint."""
        if self.model_config is not None and not getattr(self.model_config, "use_tactile_input", True):
            return True, "checkpoint has use_tactile_input=False"
        return False, None

    def _load_norm_stats(self) -> None:
        """Load normalization statistics from checkpoint."""
        norm_dir = self.ckpt_dir / "normalization"

        if not norm_dir.exists():
            logger.warning(f"Normalization directory not found: {norm_dir}")
            logger.warning("Proceeding without normalization (not recommended)")
            return

        def _parse_bool(v: Any, default: bool = False) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"1", "true", "t", "yes", "y"}:
                    return True
                if s in {"0", "false", "f", "no", "n"}:
                    return False
            return default

        def _load_json_norm_stats(path: pathlib.Path) -> dict | None:
            try:
                with open(path) as f:
                    data = json.load(f)
                # norm_stats_from_jsonable returns None if params is empty/missing
                return norm_stats_from_jsonable(data)
            except FileNotFoundError:
                return None
            except Exception as e:
                logger.warning(f"Failed to load norm stats from {path}: {e}")
                return None

        def _rank_candidate(p: pathlib.Path) -> tuple[int, int, int, str]:
            """Heuristic ordering: prefer joint > all > none, prefer t0 > t1, prefer quantile > zscore."""
            name = p.name
            mode_rank = 3
            if "_joint_" in name:
                mode_rank = 0
            elif "_all_" in name:
                mode_rank = 1
            elif "_none_" in name:
                mode_rank = 2
            t_rank = 0 if "_t0_" in name else (1 if "_t1_" in name else 2)
            n_rank = 0 if name.endswith("_quantile.json") else (1 if name.endswith("_zscore.json") else 2)
            return (mode_rank, t_rank, n_rank, name)

        def _find_first_existing(paths: list[pathlib.Path]) -> pathlib.Path | None:
            for p in paths:
                if p.exists():
                    return p
            return None

        # Prefer using the exact normalization config from the checkpoint if available.
        independent_norm_mode = None
        norm_time_dim = None
        norm_type = None
        train_cfg_path = self.ckpt_dir / "train_config.json"
        if train_cfg_path.exists():
            try:
                with open(train_cfg_path) as f:
                    train_cfg = json.load(f)
                independent_norm_mode = train_cfg.get("independent_norm_mode")
                norm_time_dim = _parse_bool(train_cfg.get("norm_time_dim"), default=False)
                norm_type = train_cfg.get("norm_type")
                if isinstance(norm_type, str):
                    norm_type = norm_type.strip().lower()
                if isinstance(independent_norm_mode, str):
                    independent_norm_mode = independent_norm_mode.strip().lower()
                logger.info(
                    f"Loaded train_config.json norm settings: independent_norm_mode={independent_norm_mode}, "
                    f"norm_time_dim={norm_time_dim}, norm_type={norm_type}"
                )
            except Exception as e:
                logger.warning(f"Failed to parse train_config.json at {train_cfg_path}: {e}")

        # Shared stats live in normalization/ (root).
        self.shared_norm_stats = None
        shared_candidates: list[pathlib.Path] = []
        if isinstance(independent_norm_mode, str) and isinstance(norm_type, str) and norm_type in {"quantile", "zscore"}:
            t_int = int(bool(norm_time_dim))
            shared_candidates.append(norm_dir / f"share_norm_stats_{independent_norm_mode}_t{t_int}_{norm_type}.json")
        # Fallback: glob for the supported naming scheme: *_quantile.json / *_zscore.json
        shared_candidates.extend(sorted(norm_dir.glob("share_norm_stats_*_t*_quantile.json"), key=_rank_candidate))
        shared_candidates.extend(sorted(norm_dir.glob("share_norm_stats_*_t*_zscore.json"), key=_rank_candidate))
        # De-duplicate while keeping order
        seen = set()
        shared_candidates = [p for p in shared_candidates if not (p in seen or seen.add(p))]

        shared_path = _find_first_existing(shared_candidates)
        if shared_path is not None:
            self.shared_norm_stats = _load_json_norm_stats(shared_path)
            if self.shared_norm_stats is not None:
                logger.info(f"Loaded shared normalization stats from {shared_path}")
            else:
                # It's valid for shared stats to be empty (e.g., independent_norm_mode='all')
                logger.info(f"Shared normalization stats file found but empty/invalid: {shared_path}")
        else:
            logger.info("No shared normalization stats found (this may be expected for independent_norm_mode='all').")

        # Domain-specific (independent) stats usually live in normalization/<domain_name>/.
        # For robustness, also check normalization/ root.
        self.domain_norm_stats = None
        domain_subdir = norm_dir / self.domain_name
        if domain_subdir.exists() and domain_subdir.is_dir():
            domain_dir = domain_subdir
        else:
            # If the checkpoint contains domain subdirectories, treat missing domain as a hard error.
            available_domains = [d.name for d in norm_dir.iterdir() if d.is_dir()]
            root_indep = list(norm_dir.glob("independent_norm_stats_*_t*_quantile.json")) + list(
                norm_dir.glob("independent_norm_stats_*_t*_zscore.json")
            )
            if available_domains and not root_indep:
                raise ValueError(
                    f"Domain '{self.domain_name}' not found in {norm_dir}. Available domains: {available_domains}"
                )
            # Otherwise, fall back to root-level independent stats (older/flattened layout).
            domain_dir = norm_dir
            if available_domains:
                logger.warning(
                    f"Domain '{self.domain_name}' directory not found, but root-level independent norm stats exist. "
                    f"Falling back to {norm_dir} (available domain dirs: {available_domains})"
                )
            else:
                logger.warning(
                    f"Domain '{self.domain_name}' directory not found and no domain subdirs present. "
                    f"Falling back to root normalization dir: {norm_dir}"
                )

        domain_candidates: list[pathlib.Path] = []
        if isinstance(independent_norm_mode, str) and isinstance(norm_type, str) and norm_type in {"quantile", "zscore"}:
            t_int = int(bool(norm_time_dim))
            domain_candidates.append(domain_dir / f"independent_norm_stats_{independent_norm_mode}_t{t_int}_{norm_type}.json")
        domain_candidates.extend(sorted(domain_dir.glob("independent_norm_stats_*_t*_quantile.json"), key=_rank_candidate))
        domain_candidates.extend(sorted(domain_dir.glob("independent_norm_stats_*_t*_zscore.json"), key=_rank_candidate))
        seen = set()
        domain_candidates = [p for p in domain_candidates if not (p in seen or seen.add(p))]

        domain_path = _find_first_existing(domain_candidates)
        if domain_path is not None:
            self.domain_norm_stats = _load_json_norm_stats(domain_path)
            if self.domain_norm_stats is not None:
                logger.info(f"Loaded domain normalization stats from {domain_path}")
            else:
                logger.warning(f"Domain normalization stats file found but empty/invalid: {domain_path}")
        else:
            logger.warning(
                f"No domain normalization stats found in {domain_dir}. "
                f"(Expected a file like independent_norm_stats_<mode>_t{{0|1}}_{{quantile|zscore}}.json)"
            )

    def _load_tokenizer(self) -> None:
        """Load the PaliGemma tokenizer."""
        max_len = self.model_config.max_token_len
        self.tokenizer = PaligemmaTokenizer(max_len=max_len)
        logger.info(f"Loaded PaliGemma tokenizer with max_len={max_len}")

    def normalize_images(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Normalize images to [-1, 1] range.

        Automatically detects input format:
        - uint8 [0-255]: converts to float [-1, 1]
        - float [0-255]: converts to [-1, 1]
        - float [-1, 1]: returns as-is

        Args:
            images: Dictionary of image arrays {key: (H, W, 3) or (B, H, W, 3)}

        Returns:
            Dictionary of normalized images in float32 [-1, 1] range.
        """
        out = {}
        for key, img in images.items():
            img = np.asarray(img)

            if img.dtype == np.uint8:
                # uint8 [0-255] -> float [-1, 1]
                img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif img.dtype in [np.float32, np.float64]:
                if img.max() > 1.0:
                    # float [0-255] -> float [-1, 1]
                    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                # else: already in [-1, 1] range, keep as-is
                img = img.astype(np.float32)
            else:
                # Convert other types and normalize
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0 * 2.0 - 1.0

            out[key] = img

        return out

    def normalize_inputs(
        self,
        state: np.ndarray,
        tactiles: dict[str, np.ndarray] | None = None,
        tactile_function_areas: dict[str, list[int]] | None = None,
        tactile_sensors: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Normalize state and tactile inputs.

        Applies normalization in order:
        1. Domain-independent layer normalization
        2. Shared layer normalization

        Args:
            state: State array of shape (T, state_dim), raw scale.
            tactiles: Optional dictionary of tactile arrays.

        Returns:
            Tuple of (normalized_state, normalized_tactiles).
        """
        data = {"state": np.asarray(state, dtype=np.float32).copy()}
        if tactiles is not None:
            # Force float dtype to avoid uint8 in-place writeback during normalization.
            data["tactile"] = {k: np.asarray(v, dtype=np.float32).copy() for k, v in tactiles.items()}
            runtime_tactile_types = self._resolve_runtime_tactile_types(list(data["tactile"].keys()))
            if runtime_tactile_types is not None:
                data["tactile_type"] = runtime_tactile_types
            if tactile_function_areas is not None:
                data["tactile_function_area"] = {
                    k: np.asarray(v, dtype=np.int64).copy() for k, v in tactile_function_areas.items()
                }
            if tactile_sensors is not None:
                data["tactile_sensor"] = dict(tactile_sensors)

        # Determine which keys to normalize (only keys present in data)
        use_keys_list = ["state"]
        if tactiles is not None:
            use_keys_list.append("tactile")

        # Apply domain-independent normalization first
        if self.domain_norm_stats is not None:
            data = apply_norm_stats(data, self.domain_norm_stats, unnormalize=False, use_keys=use_keys_list)

        # Apply shared normalization second
        if self.shared_norm_stats is not None:
            data = apply_norm_stats(data, self.shared_norm_stats, unnormalize=False, use_keys=use_keys_list)

        return data["state"], data.get("tactile")

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action output to raw scale.

        Applies denormalization in reverse order:
        1. Shared layer denormalization
        2. Domain-independent layer denormalization

        Args:
            action: Action array of shape (B, action_horizon, action_dim), normalized.

        Returns:
            Action array in raw scale.
        """
        data = {'actions': action.copy()}

        # Apply shared denormalization first (reverse of normalization order)
        if self.shared_norm_stats is not None:
            data = apply_norm_stats(
                data, self.shared_norm_stats,
                unnormalize=True, use_keys=['actions']
            )

        # Apply domain-independent denormalization second
        if self.domain_norm_stats is not None:
            data = apply_norm_stats(
                data, self.domain_norm_stats,
                unnormalize=True, use_keys=['actions']
            )

        return data['actions']

    def _tokenize_prompt(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the language prompt.

        Args:
            prompt: The language instruction string.

        Returns:
            Tuple of (tokens, mask) arrays.
        """
        tokens, mask = self.tokenizer.tokenize(prompt, state=None)
        return tokens, mask

    def build_observation(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str,
        tactiles: dict[str, np.ndarray] | None = None,
        tactile_function_areas: dict[str, list[int]] | None = None,
        tactile_sensors: dict[str, str] | None = None,
        action_mask: np.ndarray | None = None,
    ) -> Observation:
        """Build an Observation object for model inference.

        Args:
            images: Dictionary of image arrays {key: (H, W, 3)}, raw (uint8) or normalized.
            state: State array of shape ``(T, model_config.action_dim)``.
            prompt: Language instruction string.
            tactiles: Optional dictionary of tactile arrays.
            tactile_function_areas: Optional dictionary mapping tactile keys to function area indices.
            tactile_sensors: Optional dictionary mapping tactile keys to sensor names.
            action_mask: Optional action validity mask with shape ``(B, T, D)``
                matching ``(batch_size, action_horizon, action_dim)``.

        Returns:
            Observation object ready for model inference.
        """
        ignore_tactiles, tactile_ignore_reason = self._should_ignore_tactiles()
        if ignore_tactiles and tactiles is not None:
            logger.warning("%s; ignoring passed tactiles.", tactile_ignore_reason)
            tactiles = None
            tactile_function_areas = None
            tactile_sensors = None

        if self.skip_normalization:
            # Skip normalization - use inputs directly
            # Images still need to be converted to float [-1, 1] for model compatibility
            images_norm = {}
            for key, img in images.items():
                img = np.asarray(img)
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                else:
                    img = img.astype(np.float32)
                images_norm[key] = img
            state_norm = np.asarray(state, dtype=np.float32).copy()
            tactiles_norm = {k: v.copy() for k, v in tactiles.items()} if tactiles is not None else None
        else:
            # Normalize images
            images_norm = self.normalize_images(images)

            # Normalize state and tactiles.
            state_norm, tactiles_norm = self.normalize_inputs(
                state,
                tactiles,
                tactile_function_areas=tactile_function_areas,
                tactile_sensors=tactile_sensors,
            )
        tokens, mask = self._tokenize_prompt(prompt)

        # Add batch dimension if not present
        batch_size = 1

        # Prepare images with batch dimension
        images_batch = {}
        image_masks_batch = {}
        for key, img in images_norm.items():
            if img.ndim == 3:
                img = img[np.newaxis, ...]  # Add batch dim
            images_batch[key] = torch.from_numpy(img).to(self.device)
            image_masks_batch[key] = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # Prepare state with batch dimension
        if state_norm.ndim == 2:
            state_norm = state_norm[np.newaxis, ...]  # Add batch dim
        state_tensor = torch.from_numpy(state_norm).to(dtype=torch.float32, device=self.device)

        # Prepare tokens with batch dimension
        tokens_tensor = torch.from_numpy(tokens[np.newaxis, ...]).to(dtype=torch.long, device=self.device)
        mask_tensor = torch.from_numpy(mask[np.newaxis, ...]).to(dtype=torch.bool, device=self.device)

        # Prepare tactiles if provided
        tactiles_tensor = None
        tactile_function_areas_tensor = None
        tactile_sensors_list = None

        if tactiles_norm is not None:
            tactiles_tensor = {}
            for key, tac in tactiles_norm.items():
                if isinstance(tac, np.ndarray):
                    if tac.ndim == len(tac.shape):  # Add batch dim if needed
                        tac = tac[np.newaxis, ...]
                    tactiles_tensor[key] = torch.from_numpy(tac).to(dtype=torch.float32, device=self.device)

        if tactile_function_areas is not None:
            tactile_function_areas_tensor = {}
            for key, areas in tactile_function_areas.items():
                areas_arr = np.array(areas)
                if areas_arr.ndim == 1:
                    areas_arr = areas_arr[np.newaxis, ...]  # Add batch dim
                tactile_function_areas_tensor[key] = torch.from_numpy(areas_arr).to(dtype=torch.long, device=self.device)

        if tactile_sensors is not None:
            # Tactile sensors need to be a list of lists for batch processing
            tactile_sensors_list = {key: [sensor] for key, sensor in tactile_sensors.items()}

        action_masks_tensor = None
        if action_mask is not None:
            if action_mask.ndim == 1:
                action_mask = action_mask[None,None,:]
                action_mask = np.repeat(action_mask, batch_size, axis=0)
                action_mask = np.repeat(action_mask, self.model.config.action_horizon, axis=1)
            action_mask_arr = np.asarray(action_mask, dtype=np.float32)
            if action_mask_arr.ndim != 3:
                raise ValueError(
                    f"action_mask must have shape (B, T, D), got {action_mask_arr.shape}"
                )
            expected_shape = (batch_size, self.model.config.action_horizon, self.model.action_dim)
            if tuple(action_mask_arr.shape) != expected_shape:
                raise ValueError(
                    f"action_mask shape mismatch: got {action_mask_arr.shape}, expected {expected_shape}"
                )
            action_masks_tensor = torch.from_numpy(action_mask_arr).to(dtype=torch.float32, device=self.device)

        # Create Observation object
        observation = Observation(
            images=images_batch,
            image_masks=image_masks_batch,
            state=state_tensor,
            tokenized_prompt=tokens_tensor,
            tokenized_prompt_mask=mask_tensor,
            token_ar_mask=None,
            token_loss_mask=None,
            tactiles=tactiles_tensor,
            tactile_function_areas=tactile_function_areas_tensor,
            tactile_sensors=tactile_sensors_list,
            image_function_areas=None,
            domain_names=[self.domain_name],
            action_masks=action_masks_tensor,
        )

        return observation

    @torch.no_grad()
    def infer(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str,
        tactiles: dict[str, np.ndarray] | None = None,
        tactile_function_areas: dict[str, list[int]] | None = None,
        tactile_sensors: dict[str, str] | None = None,
        action_mask: np.ndarray | None = None,
        noise_generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """Run inference and return denormalized actions.

        This is the main entry point for inference. It handles:
        1. Image normalization (auto-detects uint8/float format)
        2. State and tactile normalization using loaded norm_stats
        3. Prompt tokenization
        4. Model forward pass (diffusion sampling)
        5. Action denormalization to raw scale

        Args:
            images: dict[str, np.ndarray]
                Camera RGB images.
                - Keys: Camera names (e.g., "camera_ego_rgb_0", "right_wrist_camera_rgb_0")
                - Values: Image arrays of shape (H, W, 3) or (B, H, W, 3)
                - dtype: uint8 [0-255] or float32 (auto-detected and normalized to [-1, 1])

            state: np.ndarray
                Robot proprioceptive state.
                - Shape: (T, get_state_dim()), typically T=1.
                - dtype: float32, RAW SCALE (will be normalized internally)
                - Example: joint positions, end-effector poses

            prompt: str
                Natural language task instruction.
                - Example: "pick up the red cube and place it on the table"

            tactiles: dict[str, np.ndarray] | None
                Tactile sensor readings (optional, only if model uses tactile).
                - Keys: Tactile sensor names (e.g., "right_tactile_gripper")
                - Values: Tactile arrays, shape depends on sensor type:
                    - Image type (FreeTacMan, GelSight): (T, num_areas, H, W, 3)
                    - Matrix type (3DViTac): (T, num_areas, rows, cols)
                    - Binary type: (T, num_areas, num_sensors)
                - dtype: float32, RAW SCALE

            tactile_function_areas: dict[str, list[int]] | None
                Maps tactile keys to function area indices.
                - Function areas identify which physical contact region each tactile
                  reading corresponds to (e.g., fingertip=0, finger_pad=1).
                - The list length should match num_areas in the tactile array.
                - Example: {"right_tactile_gripper": [0, 1]} for 2 tactile areas

            tactile_sensors: dict[str, str] | None
                Maps tactile keys to sensor type names.
                - Used to select the correct encoder for each tactile input.
                - Must match sensor names in tactile_input_config_file.json.
                - Example: {"right_tactile_gripper": "FreeTacMan"}

            action_mask: np.ndarray | None
                Optional mask forwarded into ``Observation.action_masks`` for compatibility.
                Shape ``(B, T, D) = (1, action_horizon, action_dim)``.
                - Current FTP1 master sampling keeps diffusion ``x_t`` unmasked.

            noise_generator: torch.Generator | None
                Optional episode-local generator for the initial flow noise. Supplying this makes
                rollouts repeatable without changing the process-global RNG used by the simulator.

        Returns:
            np.ndarray: Predicted actions in RAW SCALE (denormalized).
                - Shape: (action_horizon, action_dim), where ``action_dim`` matches
                  ``config.action_dim`` (for example 120).
                - action_horizon: ``config.action_horizon`` (FTP1 default 32).
                - dtype: float32

        Example:
            >>> wrapper = FTP1InferenceWrapper(
            ...     checkpoint_dir="/path/to/ckpt/7999",
            ...     domain_name="FreeTacMan_Train",
            ... )
            >>> state_dim = wrapper.get_state_dim()
            >>> action = wrapper.infer(
            ...     images={"camera_ego_rgb_0": np.zeros((224, 224, 3), dtype=np.uint8)},
            ...     state=np.zeros((1, state_dim), dtype=np.float32),
            ...     prompt="pick up the object",
            ... )
            >>> print(action.shape)  # (action_horizon, action_dim), e.g., (32, 120)
        """
        import time as time_module
        timings = {}

        # Build observation (includes normalization)
        t0 = time_module.perf_counter()
        observation = self.build_observation(
            images=images,
            state=state,
            prompt=prompt,
            tactiles=tactiles,
            tactile_function_areas=tactile_function_areas,
            tactile_sensors=tactile_sensors,
            action_mask=action_mask,
        )
        timings['build_observation'] = time_module.perf_counter() - t0

        # Run model inference
        t0 = time_module.perf_counter()
        noise = None
        if noise_generator is not None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(
                    observation.state.shape[0],
                    self.model_config.action_horizon,
                    self.model_config.action_dim,
                ),
                generator=noise_generator,
                dtype=torch.float32,
                device=self.device,
            )
        action = self.model.sample_actions(
            device=self.device,
            observation=observation,
            noise=noise,
            num_steps=self.num_inference_steps,
        )
        timings['sample_actions'] = time_module.perf_counter() - t0

        # Convert to numpy and remove batch dimension
        t0 = time_module.perf_counter()
        action_np = action.cpu().numpy()
        if action_np.ndim == 3:
            action_np = action_np[0]  # Remove batch dim: (1, T, D) -> (T, D)
        timings['to_numpy'] = time_module.perf_counter() - t0

        # Denormalize action (skip if skip_normalization is True)
        t0 = time_module.perf_counter()
        if self.skip_normalization:
            action_denorm = action_np
            timings['denormalize'] = 0.0
        else:
            action_denorm = self.denormalize_action(action_np)
            timings['denormalize'] = time_module.perf_counter() - t0

        # Log timing breakdown (print each on its own line, not overwritten by progress bar)
        print(
            f"[FTP1Wrapper] build_obs={timings['build_observation']*1000:.1f}ms, "
            f"sample_actions={timings['sample_actions']*1000:.1f}ms, "
            f"to_numpy={timings['to_numpy']*1000:.1f}ms, "
            f"denorm={timings['denormalize']*1000:.1f}ms (skip={self.skip_normalization})",
            flush=True,
        )

        return action_denorm

    def get_action_dim(self) -> int:
        """Return the model action dim (``config.action_dim``)."""
        return self.model_config.action_dim

    def get_action_horizon(self) -> int:
        """Get the action horizon."""
        return self.model_config.action_horizon

    def get_state_dim(self) -> int:
        """Return the state dim (equal to ``config.action_dim``)."""
        return self.model_config.action_dim
