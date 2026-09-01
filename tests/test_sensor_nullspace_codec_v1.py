import numpy as np

from anchor.corrected_sgta.sensor_nullspace_codec_v1 import (
    LUMA_BT709,
    decode_legacy_gray,
    encode_residual_rgb,
    project_to_null,
)


def test_projected_carrier_is_unit_luma_null():
    carrier = project_to_null(np.asarray([0.6, -0.2, 0.8]))
    assert np.isclose(np.linalg.norm(carrier), 1.0)
    assert np.isclose(LUMA_BT709 @ carrier, 0.0, atol=1e-12)


def test_codec_exactly_preserves_all_legacy_gray_levels():
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rng = np.random.default_rng(42)
    residual = rng.uniform(-0.5 / 255.0, 0.5 / 255.0, size=gray.shape)
    rgb, audit = encode_residual_rgb(
        gray,
        residual,
        carrier=np.asarray([0.6, -0.2, 0.8]),
        gain=96.0,
    )
    assert np.array_equal(decode_legacy_gray(rgb), gray)
    assert audit.legacy_mismatch_pixels == 0
    assert rgb.dtype == np.uint8


def test_zero_residual_returns_grayscale_replication():
    gray = np.asarray([[0, 1, 127, 254, 255]], dtype=np.uint8)
    rgb, audit = encode_residual_rgb(
        gray,
        np.zeros_like(gray, dtype=np.float64),
        carrier=np.asarray([1.0, 0.0, 0.0]),
        gain=96.0,
    )
    assert np.array_equal(rgb, np.repeat(gray[..., None], 3, axis=-1))
    assert audit.rgb_change_fraction == 0.0


def test_capacity_prevents_boundary_clipping_without_breaking_luma():
    gray = np.asarray([[0, 1, 2, 253, 254, 255]], dtype=np.uint8)
    residual = np.asarray([[0.49, -0.49, 0.49, -0.49, 0.49, -0.49]]) / 255.0
    rgb, audit = encode_residual_rgb(
        gray,
        residual,
        carrier=np.asarray([0.8, -0.3, 0.4]),
        gain=256.0,
    )
    assert np.array_equal(decode_legacy_gray(rgb), gray)
    assert audit.capacity_limited_fraction > 0
