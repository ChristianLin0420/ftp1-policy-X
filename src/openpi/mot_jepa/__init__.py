"""MoT-JEPA: a mixture-of-transformers video-tactile world model.

Self-supervised pretraining package, kept entirely separate from the FTP-1 flow-matching
policy path so the baseline arm stays bit-identical.

Nothing heavy is imported at module scope: no CUDA initialization and no ``jax``, so
importing this package inside a dataloader worker is cheap and side-effect free.
"""

from openpi.mot_jepa.layout import LAYOUT_BASE
from openpi.mot_jepa.layout import LAYOUT_BASE_STAGES
from openpi.mot_jepa.layout import LAYOUT_PILOT
from openpi.mot_jepa.layout import STREAM_ORDER
from openpi.mot_jepa.layout import ExpertId
from openpi.mot_jepa.layout import StreamId
from openpi.mot_jepa.layout import TokenLayout
from openpi.mot_jepa.masking import DEFAULT_MODE_PROBS
from openpi.mot_jepa.masking import ClipMasks
from openpi.mot_jepa.masking import MaskMode
from openpi.mot_jepa.masking import MaskSpec
from openpi.mot_jepa.masking import assert_mask_invariants
from openpi.mot_jepa.masking import build_batch_masks
from openpi.mot_jepa.masking import derive_mask_seed
from openpi.mot_jepa.masking import draw_mode
from openpi.mot_jepa.masking import expected_target_counts

__all__ = [
    "DEFAULT_MODE_PROBS",
    "LAYOUT_BASE",
    "LAYOUT_BASE_STAGES",
    "LAYOUT_PILOT",
    "STREAM_ORDER",
    "ClipMasks",
    "ExpertId",
    "MaskMode",
    "MaskSpec",
    "StreamId",
    "TokenLayout",
    "assert_mask_invariants",
    "build_batch_masks",
    "derive_mask_seed",
    "draw_mode",
    "expected_target_counts",
]
