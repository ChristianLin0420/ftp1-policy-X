"""Every layout observed in the released corpus, plus the failure each one used to cause."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from openpi.mot_jepa import tactile_parse as tp


def store_with(key: str, shape: tuple[int, ...], tactile_type: str, sensor: str, *, frames: int = 8):
    root = zarr.open(zarr.storage.MemoryStore(), mode="w")
    data = root.create_group("data")
    rng = np.random.default_rng(0)
    dtype = "uint8" if tactile_type in {"image", "matrix"} else "float32"
    arr = data.create_array(key, shape=(frames, *shape), dtype=dtype)
    arr[:] = rng.integers(1, 255, (frames, *shape)).astype(dtype)
    side = key.split("_", 1)[0]
    detail = key.split("_tactile_data_")[1]
    for name, value in ((f"{side}_tactile_type_{detail}", tactile_type), (f"{side}_tactile_sensor_{detail}", sensor)):
        a = data.create_array(name, shape=(frames,), dtype=f"<U{max(len(value), 4)}")
        a[:] = np.array([value] * frames)
    return data


# -- real layouts from the corpus survey ------------------------------------------------

REAL_LAYOUTS = [
    # (shape, type, sensor, expected route, expected units, expected unit_shape)
    ((2, 224, 224, 3), "image", "FreeTacMan", tp.GEL, 2, (224, 224, 3)),
    ((2, 240, 240, 3), "image", "OpenLoongVTouch", tp.GEL, 2, (240, 240, 3)),
    ((1, 240, 320, 3), "image", "GelSightMini", tp.GEL, 1, (240, 320, 3)),
    ((1, 460, 680, 3), "image", "exUMI", tp.GEL, 1, (460, 680, 3)),
    ((3, 224, 224), "image", "SharpaWave", tp.GEL, 1, (224, 224, 3)),
    ((2, 4, 4, 3), "matrix", "uSkin", tp.LOWDIM, 2, (48,)),
    ((1, 6), "state", "ATIAxia80M20", tp.LOWDIM, 1, (6,)),
    ((1, 6), "state", "FrankaTorque", tp.LOWDIM, 1, (6,)),
    ((4, 1), "state", "InspireHand", tp.LOWDIM, 4, (1,)),
    ((1, 2), "state", "InspireHand", tp.LOWDIM, 1, (2,)),
    ((1, 1), "state", "FlexivGripperForce", tp.LOWDIM, 1, (1,)),
]


@pytest.mark.parametrize(
    ("shape", "ttype", "sensor", "route", "units", "unit_shape"),
    REAL_LAYOUTS,
    ids=[f"{s}-{t}-{n}" for s, t, n, *_ in REAL_LAYOUTS],
)
def test_every_observed_layout_classifies_correctly(shape, ttype, sensor, route, units, unit_shape):
    data = store_with("right_tactile_data_x", shape, ttype, sensor)
    spec = tp.classify(data, "right_tactile_data_x", tactile_type=ttype, sensor=sensor)
    assert spec.route == route
    assert spec.num_units == units
    assert spec.unit_shape == unit_shape


def test_both_pads_of_a_two_pad_image_key_are_kept():
    """The worst of the four defects: raw[:, 0] silently discarded half the touch data.

    Most image sensors pack two pads into one key, including FreeTacMan -- the largest
    image-tactile domain in the corpus.
    """
    data = store_with("right_tactile_data_g", (2, 8, 8, 3), "image", "FreeTacMan")
    spec = tp.classify(data, "right_tactile_data_g", tactile_type="image", sensor="FreeTacMan")
    raw = np.asarray(data["right_tactile_data_g"][:])
    out = tp.read_gel(raw, spec)
    assert out.shape == (8, 2, 8, 8, 3)
    np.testing.assert_array_equal(out[:, 0], raw[:, 0])
    np.testing.assert_array_equal(out[:, 1], raw[:, 1])
    assert not np.array_equal(out[:, 0], out[:, 1]), "the two pads must stay distinct"


def test_channels_first_image_is_transposed_not_reinterpreted():
    """SharpaWave is (T, 3, H, W). Read as (T, N, H, W) it becomes three fake grey pads."""
    data = store_with("right_tactile_data_s", (3, 16, 20), "image", "SharpaWave")
    spec = tp.classify(data, "right_tactile_data_s", tactile_type="image", sensor="SharpaWave")
    assert spec.channels_first
    raw = np.asarray(data["right_tactile_data_s"][:])
    out = tp.read_gel(raw, spec)
    assert out.shape == (8, 1, 16, 20, 3)
    # Channel c of the source must land in channel c of the output, not become a pad.
    for channel in range(3):
        np.testing.assert_array_equal(out[:, 0, :, :, channel], raw[:, channel])


def test_matrix_keeps_every_taxel_channel():
    """uSkin is 2 pads x 4x4 taxels x 3 axes. The old path kept 1 of 48 values per pad."""
    data = store_with("right_tactile_data_u", (2, 4, 4, 3), "matrix", "uSkin")
    spec = tp.classify(data, "right_tactile_data_u", tactile_type="matrix", sensor="uSkin")
    assert spec.route == tp.LOWDIM
    assert spec.width == 48
    raw = np.asarray(data["right_tactile_data_u"][:])
    out = tp.read_lowdim(raw, spec)
    assert out.shape == (8, 2, 48)
    np.testing.assert_allclose(out[:, 0], raw[:, 0].reshape(8, 48))
    np.testing.assert_allclose(out[:, 1], raw[:, 1].reshape(8, 48))


def test_wide_state_keeps_all_six_wrench_channels():
    """A 6-D force/torque wrench truncated to channel 0 keeps x-force and drops the rest."""
    data = store_with("right_tactile_data_f", (1, 6), "state", "ATIAxia80M20")
    spec = tp.classify(data, "right_tactile_data_f", tactile_type="state", sensor="ATIAxia80M20")
    assert spec.width == 6
    raw = np.asarray(data["right_tactile_data_f"][:])
    out = tp.read_lowdim(raw, spec)
    assert out.shape == (8, 1, 6)
    np.testing.assert_allclose(out[:, 0, :], raw[:, 0, :])


def test_large_taxel_grids_route_to_the_image_path():
    """3DViTac-style (12, 32) is a taxel image; 4x4 uSkin is not worth upsampling."""
    big = store_with("right_tactile_data_b", (1, 12, 32, 3), "matrix", "3DViTac")
    spec_big = tp.classify(big, "right_tactile_data_b", tactile_type="matrix", sensor="3DViTac")
    assert spec_big.route == tp.GEL
    assert spec_big.unit_shape == (12, 32, 3)

    small = store_with("right_tactile_data_s", (2, 4, 4, 3), "matrix", "uSkin")
    spec_small = tp.classify(small, "right_tactile_data_s", tactile_type="matrix", sensor="uSkin")
    assert spec_small.route == tp.LOWDIM


def test_greyscale_image_is_promoted_to_three_channels():
    data = store_with("right_tactile_data_g", (1, 16, 16, 1), "image", "Grey")
    spec = tp.classify(data, "right_tactile_data_g", tactile_type="image", sensor="Grey")
    out = tp.read_gel(np.asarray(data["right_tactile_data_g"][:]), spec)
    assert out.shape[-1] == 3
    np.testing.assert_array_equal(out[..., 0], out[..., 2])


def test_specs_for_store_covers_a_mixed_store_and_drops_dead_streams():
    """A store with all three types at once, which RH20TCfg7Tactile actually is."""
    root = zarr.open(zarr.storage.MemoryStore(), mode="w")
    data = root.create_group("data")
    rng = np.random.default_rng(0)

    def add(key, shape, ttype, sensor, *, constant=False):
        arr = data.create_array(key, shape=(8, *shape), dtype="float32")
        arr[:] = np.zeros((8, *shape)) if constant else rng.random((8, *shape))
        detail = key.split("_tactile_data_")[1]
        side = key.split("_", 1)[0]
        for name, value in ((f"{side}_tactile_type_{detail}", ttype), (f"{side}_tactile_sensor_{detail}", sensor)):
            a = data.create_array(name, shape=(8,), dtype=f"<U{max(len(value), 4)}")
            a[:] = np.array([value] * 8)

    add("right_tactile_data_gel", (2, 8, 8, 3), "image", "FreeTacMan")
    add("right_tactile_data_uskin", (2, 4, 4, 3), "matrix", "uSkin")
    add("right_tactile_data_ft", (1, 6), "state", "ATIAxia80M20")
    add("left_tactile_data_dead", (1, 8, 8, 3), "image", "GelSightMini", constant=True)

    specs = tp.specs_for_store(data)
    by_route = {spec.key: spec.route for spec in specs}
    assert by_route == {
        "right_tactile_data_ft": tp.LOWDIM,
        "right_tactile_data_gel": tp.GEL,
        "right_tactile_data_uskin": tp.LOWDIM,
    }, "the dead stream must be dropped and each live one routed by type"
    assert sum(s.num_units for s in specs if s.route == tp.LOWDIM) == 3  # 2 uSkin pads + 1 wrench
    assert sum(s.num_units for s in specs if s.route == tp.GEL) == 2  # 2 gel pads


def test_max_lowdim_width_across_the_corpus_fits_the_configured_channels():
    """uSkin at 48 is the widest low-dim unit observed; the layout must not truncate it."""
    widths = []
    for shape, ttype, sensor, route, _, _ in REAL_LAYOUTS:
        if route != tp.LOWDIM:
            continue
        data = store_with("right_tactile_data_x", shape, ttype, sensor)
        widths.append(tp.classify(data, "right_tactile_data_x", tactile_type=ttype, sensor=sensor).width)
    assert max(widths) == 48
