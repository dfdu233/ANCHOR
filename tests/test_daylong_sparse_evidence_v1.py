from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from anchor.corrected_sgta.analyze_sparse_lesion_boundary_v1 import union_area
from anchor.corrected_sgta.analyze_sparse_patch_scan_v1 import (
    FINDINGS,
    design,
    higher_criticism,
    multiscale_scan,
)
from anchor.corrected_sgta.collect_sparse_patch_scores_v1 import infer_grid
from anchor.corrected_sgta.search_reuse_crop_probe_v1 import analyze as analyze_search_reuse
from anchor.corrected_sgta.search_reuse_crop_probe_v1 import crop_for_grid
from anchor.corrected_sgta.search_reuse_crop_probe_v1 import window_scores


def test_infer_model_native_patch_grids() -> None:
    assert infer_grid(576) == (1, 24)
    assert infer_grid(980) == (5, 14)


def test_rectangle_union_does_not_double_count_overlap() -> None:
    boxes = [
        {"x_min": 0.0, "y_min": 0.0, "x_max": 2.0, "y_max": 2.0},
        {"x_min": 1.0, "y_min": 1.0, "x_max": 3.0, "y_max": 3.0},
    ]
    assert union_area(boxes) == 7.0


def test_spatial_scan_rewards_connected_evidence_not_scattered_peaks() -> None:
    connected = np.zeros(64)
    connected.reshape(8, 8)[2:4, 2:4] = 2.0
    scattered = np.zeros(64)
    scattered[[0, 7, 56, 63]] = 2.0
    assert multiscale_scan(connected, 1, 8) > multiscale_scan(scattered, 1, 8)


def test_higher_criticism_is_finite_for_null_and_sparse_shift() -> None:
    null = np.zeros(100)
    sparse = null.copy()
    sparse[:5] = 4.0
    assert np.isfinite(higher_criticism(null))
    assert higher_criticism(sparse) > higher_criticism(null)


def test_sparse_scan_primary_null_contains_all_standard_poolers() -> None:
    n = len(FINDINGS) * 2
    data = {
        "finding": np.asarray(list(FINDINGS) * 2),
        "final_margin": np.arange(n, dtype=float),
        "patch_mean": np.arange(n, dtype=float),
        "patch_max_z": np.arange(n, dtype=float),
        "patch_top5_z": np.arange(n, dtype=float),
        "multiscale_scan": np.arange(n, dtype=float),
    }
    base = design(data, include_scan=False)
    enhanced = design(data, include_scan=True)
    assert base.shape == (n, len(FINDINGS) - 1 + 4)
    assert enhanced.shape == (n, base.shape[1] + 1)


def test_search_reuse_window_score_finds_planted_region() -> None:
    grid = np.zeros((8, 8), dtype=float)
    grid[3:5, 4:6] = 3.0
    surface = window_scores(grid.ravel(), side=8, window=2)
    assert np.unravel_index(int(np.argmax(surface)), surface.shape) == (3, 4)


def test_search_reuse_crops_have_equal_size_at_edges() -> None:
    image = Image.new("RGB", (240, 240))
    center = crop_for_grid(image, row=8, col=8, side=24, window=6)
    corner = crop_for_grid(image, row=0, col=0, side=24, window=6)
    assert center.size == corner.size


def test_search_reuse_analyzer_tracks_region_growth_and_fp(tmp_path) -> None:
    selections, raw = [], []
    qid = 1
    for image_index in range(40):
        for claim_count, region_count, selected_margin in (
            (1, 16, -1.0),
            (1, 64, -0.5),
            (1, 361, 0.5),
            (7, 361, 1.0),
        ):
            for variant, margin in (
                ("selected", selected_margin),
                ("random", -1.0),
                ("full", -1.0),
            ):
                selections.append({
                    "qid": qid,
                    "image_id": f"image-{image_index}",
                    "claim_count": claim_count,
                    "region_count": region_count,
                    "finding": "nodule_mass",
                    "variant": variant,
                    "selected_internal_score": float(region_count),
                })
                raw.append({
                    "question_id": qid,
                    "status": "ok",
                    "scores": {"original_margin": margin},
                })
                qid += 1
    selections_path = tmp_path / "selections.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    output = tmp_path / "analysis.json"
    selections_path.write_text("".join(json.dumps(row) + "\n" for row in selections))
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw))
    analyze_search_reuse(SimpleNamespace(
        selections=selections_path,
        raw=raw_path,
        output=output,
        bootstrap_draws=100,
        seed=42,
    ))
    result = json.loads(output.read_text())
    assert result["primary"]["gate"] is True
    assert result["primary"]["region_only_selected_random_gap_growth"] > 0
