from __future__ import annotations

import json

import numpy as np

from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    BASELINE_VIEW,
    DicomPixels,
    build_render_views,
    linear_voi,
    valid_completed_shard,
)


def synthetic_pixels(photometric: str = "MONOCHROME2") -> DicomPixels:
    values = np.tile(np.linspace(0.0, 4095.0, 128, dtype=np.float32), (96, 1))
    values[10:86, 12:116] += 120.0 * np.sin(np.linspace(0, np.pi, 104))[None]
    return DicomPixels(
        modality=values,
        valid=np.ones_like(values, dtype=bool),
        photometric=photometric,
        window_center=2048.0,
        window_width=4096.0,
        window_source="dicom_window_center_width",
        metadata={},
    )


def test_linear_voi_matches_dicom_boundary_definition() -> None:
    center, width = 40.0, 400.0
    lower = center - 0.5 - (width - 1.0) / 2.0
    upper = center - 0.5 + (width - 1.0) / 2.0
    values = linear_voi(np.asarray([lower - 1, lower, center - 0.5, upper, upper + 1]), center, width)
    assert np.allclose(values, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_render_contract_and_secondary_exclusion() -> None:
    claim_boxes = [{"x_min": 35, "y_min": 30, "x_max": 60, "y_max": 55}]
    other_boxes = claim_boxes + [{"x_min": 70, "y_min": 20, "x_max": 90, "y_max": 42}]
    views = build_render_views(synthetic_pixels(), claim_boxes, other_boxes)
    by_name = {view["name"]: view for view in views}
    assert set(by_name) == {
        BASELINE_VIEW,
        "identity_lossless_duplicate",
        "native_linear",
        "center_minus_0p05w",
        "center_plus_0p05w",
        "width_x0p8",
        "width_x1p25",
        "native_sigmoid",
        "blank_border_zoom",
        "polarity_toggle",
        "content_loss_downsample",
    }
    assert not by_name[BASELINE_VIEW]["is_primary"]
    assert by_name["native_linear"]["is_primary"]
    assert not by_name["native_sigmoid"]["is_primary"]
    assert not by_name["polarity_toggle"]["is_primary"]
    assert by_name["polarity_toggle"]["parameters"]["cannot_drive_primary_gate"]
    assert by_name["blank_border_zoom"]["audit"]["bbox_retention"]
    assert by_name["identity_lossless_duplicate"]["audit"]["identity_hash_matches_baseline"]
    assert all(view["audit"]["roi_source"] == "fixed_central_thorax_proxy" for view in views)
    assert all(len(view["audit"]["pixel_sha256"]) == 64 for view in views)


def test_resume_requires_exact_complete_view_contract(tmp_path) -> None:
    path = tmp_path / "claim.json"
    names = [
        BASELINE_VIEW,
        "identity_lossless_duplicate",
        "native_linear",
        "center_minus_0p05w",
        "center_plus_0p05w",
        "width_x0p8",
        "width_x1p25",
        "native_sigmoid",
        "blank_border_zoom",
        "polarity_toggle",
        "content_loss_downsample",
    ]
    payload = {
        "status": "ok",
        "record_key": "finding__image__hash",
        "config_fingerprint": "fingerprint",
        "views": [
            {"name": name, "audit": {}, "scores": {"polarity": 0.0, "commitment": 0.0}}
            for name in names
        ],
    }
    path.write_text(json.dumps(payload))
    assert valid_completed_shard(path, "finding__image__hash", "fingerprint")
    payload["views"].pop()
    path.write_text(json.dumps(payload))
    assert not valid_completed_shard(path, "finding__image__hash", "fingerprint")
