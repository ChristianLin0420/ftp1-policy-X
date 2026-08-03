"""Canonical parsing of every FTP-1 tactile type.

The release carries four declared types and a genuinely heterogeneous set of per-frame
layouts. Getting this wrong is silent: the arrays still load, training still runs, the loss
still falls, and most of the touch signal is simply gone. A survey of the extracted corpus
found all of the following, and each needed handling that a naive reader does not give it.

===========  ==========================  ================================================
type         observed per-frame shapes   sensors
===========  ==========================  ================================================
``image``    ``(N, H, W, 3)``            FreeTacMan, GelSightMini, MCTac, ViTaMIn,
                                         OpenLoongVTouch, exUMI
``image``    ``(3, H, W)``               SharpaWave -- channels FIRST, no pad axis
``matrix``   ``(N, h, w, C)``            uSkin, 2 pads of 4x4 taxels x 3 force axes
``state``    ``(N, D)``                  ATIAxia/Franka 6-D wrench, InspireHand, gripper
===========  ==========================  ================================================

Four things this module exists to get right:

1. **Multi-pad keys.** Most image sensors pack two pads into one key as ``(T, 2, H, W, 3)``.
   Taking ``raw[:, 0]`` discards the second pad -- half the touch data on the largest
   image-tactile domain in the corpus.
2. **Channels-first images.** SharpaWave is ``(T, 3, 224, 224)``. Read as ``(T, N, H, W)``
   it becomes "3 pads of a 224x224 single-channel image", and a subsequent ``cv2.resize``
   reinterprets the axes again. The result is not a gel image at all.
3. **``matrix`` is not ``state``.** Routing it through a scalar path and keeping channel 0
   throws away 47 of 48 uSkin values.
4. **Wide ``state``.** A 6-D force/torque wrench truncated to its first channel keeps the
   x-force and discards y, z and all three torques.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import zarr

TACTILE_DATA_RE = re.compile(r"^(?P<side>left|right)_tactile_data_(?P<detail>.+)$")

#: A ``matrix`` stream with at least this many taxels is spatially rich enough to be worth
#: treating as an image (patch-tokenized). Below it, upsampling a 4x4 grid to 112x112 would
#: manufacture 49 patches out of 16 real measurements; the flattened vector carries exactly
#: the same information for a fraction of the cost.
MATRIX_AS_IMAGE_MIN_TAXELS = 64

#: Route labels.
GEL = "gel"
LOWDIM = "lowdim"


@dataclasses.dataclass(frozen=True)
class TactileSpec:
    """How one tactile key should be read."""

    key: str
    tactile_type: str
    sensor: str
    route: str
    num_units: int
    """Pads/slots this key contributes -- each becomes its own gel pad or lowdim slot."""
    unit_shape: tuple[int, ...]
    """``(H, W, 3)`` for gel units, ``(D,)`` for lowdim units."""
    channels_first: bool = False

    @property
    def width(self) -> int:
        return int(np.prod(self.unit_shape))


def _looks_channels_first(shape: tuple[int, ...]) -> bool:
    """``(3, H, W)`` rather than ``(N, H, W)``.

    A leading axis of exactly 3 with two large trailing axes is a CHW image; a genuine pad
    axis of 3 alongside 2-D pads would be indistinguishable, but no such sensor exists in the
    release and the CHW reading is the only one that yields a valid image.
    """
    return len(shape) == 3 and shape[0] == 3 and shape[1] > 8 and shape[2] > 8


def classify(data: zarr.Group, key: str, *, tactile_type: str, sensor: str) -> TactileSpec:
    """Decide how to read one tactile key from its declared type and per-frame shape."""
    per_frame = tuple(int(dim) for dim in data[key].shape[1:])

    if tactile_type == "image":
        if _looks_channels_first(per_frame):
            channels, height, width = per_frame
            return TactileSpec(key, tactile_type, sensor, GEL, 1, (height, width, channels), channels_first=True)
        if len(per_frame) == 4:  # (N, H, W, C)
            pads, height, width, channels = per_frame
            return TactileSpec(key, tactile_type, sensor, GEL, pads, (height, width, channels))
        if len(per_frame) == 3:  # (H, W, C)
            height, width, channels = per_frame
            return TactileSpec(key, tactile_type, sensor, GEL, 1, (height, width, channels))
        if len(per_frame) == 2:  # (H, W) greyscale
            height, width = per_frame
            return TactileSpec(key, tactile_type, sensor, GEL, 1, (height, width, 1))

    elif tactile_type == "matrix":
        # (N, h, w, C) taxel grid, or (h, w) / (h, w, C) without a pad axis.
        if len(per_frame) == 4:
            pads, rows, cols, channels = per_frame
        elif len(per_frame) == 3:
            pads, (rows, cols, channels) = 1, per_frame
        elif len(per_frame) == 2:
            pads, rows, cols, channels = 1, per_frame[0], per_frame[1], 1
        else:
            pads, rows, cols, channels = 1, 1, int(np.prod(per_frame)), 1
        if rows * cols >= MATRIX_AS_IMAGE_MIN_TAXELS:
            return TactileSpec(key, tactile_type, sensor, GEL, pads, (rows, cols, max(channels, 1)))
        return TactileSpec(key, tactile_type, sensor, LOWDIM, pads, (rows * cols * max(channels, 1),))

    # state, binary, and anything unrecognized: a per-pad vector.
    if len(per_frame) >= 2:
        pads = per_frame[0]
        width = int(np.prod(per_frame[1:]))
    elif len(per_frame) == 1:
        pads, width = 1, per_frame[0]
    else:
        pads, width = 1, 1
    return TactileSpec(key, tactile_type, sensor, LOWDIM, pads, (width,))


def read_gel(raw: np.ndarray, spec: TactileSpec) -> np.ndarray:
    """``(T, num_units, H, W, 3)`` uint8, every pad preserved."""
    if spec.channels_first:  # (T, 3, H, W) -> (T, 1, H, W, 3)
        return np.ascontiguousarray(raw.transpose(0, 2, 3, 1))[:, None]
    frames = raw.shape[0]
    array = raw.reshape(frames, spec.num_units, *spec.unit_shape)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        array = array[..., :3]
    return array


def read_lowdim(raw: np.ndarray, spec: TactileSpec) -> np.ndarray:
    """``(T, num_units, width)`` float32, every channel preserved."""
    return raw.reshape(raw.shape[0], spec.num_units, spec.width).astype(np.float32)


def is_degenerate(array: zarr.Array, *, num_samples: int = 32) -> bool:
    """True when a stream is constant, i.e. the sensor recorded nothing.

    The released ``RDP_Bimanual`` store labels two identically-zero gel streams as ``image``,
    so the declared type is not evidence that a sensor was on.
    """
    length = array.shape[0]
    if length == 0:
        return True
    index = np.unique(np.linspace(0, length - 1, min(num_samples, length)).astype(np.int64))
    return bool(np.asarray(array[index]).astype(np.float32).std() == 0.0)


def specs_for_store(data: zarr.Group, *, drop_degenerate: bool = True) -> list[TactileSpec]:
    """Every usable tactile stream in a store, classified and in stable key order."""
    keys = set(data.array_keys())
    specs: list[TactileSpec] = []
    for key in sorted(keys):
        match = TACTILE_DATA_RE.match(key)
        if match is None:
            continue
        if drop_degenerate and is_degenerate(data[key]):
            continue
        type_key = f"{match['side']}_tactile_type_{match['detail']}"
        sensor_key = f"{match['side']}_tactile_sensor_{match['detail']}"
        tactile_type = str(data[type_key][-1]) if type_key in keys else "state"
        sensor = str(data[sensor_key][-1]) if sensor_key in keys else "unknown"
        specs.append(classify(data, key, tactile_type=tactile_type, sensor=sensor))
    return specs
