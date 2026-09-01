from corrected_sgta.prepare_vindr_spatial_reader_manifest_v1 import (
    build_records,
    classify_horizontal_extent,
    classify_vertical_extent,
    data_progression_gate,
    split_image,
    summarize,
)


def box(x0, y0, x1, y1):
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}


def test_horizontal_extent_is_flip_invariant_and_central_is_ambiguous():
    left = [box(10, 20, 35, 60)]
    right = [box(65, 20, 90, 60)]
    both = left + right
    assert classify_horizontal_extent(left, 100, 100) == "single_image_hemifield"
    assert classify_horizontal_extent(right, 100, 100) == "single_image_hemifield"
    assert classify_horizontal_extent(both, 100, 100) == "both_image_hemifields"
    assert classify_horizontal_extent([box(46, 20, 54, 60)], 100, 100).startswith("ambiguous")


def test_wide_box_can_cover_both_hemifields_without_box_count_heuristic():
    assert classify_horizontal_extent([box(20, 10, 80, 90)], 100, 100) == "both_image_hemifields"


def test_vertical_labels_are_only_image_regions():
    assert classify_vertical_extent([box(20, 5, 40, 30)], 100, 100) == "upper_image_region"
    assert classify_vertical_extent([box(20, 70, 40, 95)], 100, 100) == "lower_image_region"
    assert classify_vertical_extent([box(20, 20, 40, 80)], 100, 100) == "multiple_image_height_zones"
    assert classify_vertical_extent([box(20, 47, 40, 53)], 100, 100).startswith("ambiguous")


def test_invalid_coordinates_fail_closed():
    try:
        classify_horizontal_extent([box(-1, 2, 5, 8)], 100, 100)
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-image box was accepted")


def test_split_is_image_global_and_deterministic():
    assert split_image("same-image", 42) == split_image("same-image", 42)
    assert split_image("same-image", 42) in {"pilot", "dev", "test"}


def test_records_require_all_positive_readers_and_all_reader_boxes():
    key = ("image-a", "pleural_effusion")
    labels = {key: {"R8", "R9", "R10"}}
    boxes = {
        key: {
            "R8": [box(10, 10, 30, 30)],
            "R9": [box(12, 10, 32, 30)],
            "R10": [box(11, 10, 31, 30)],
        }
    }
    rows, audit = build_records(labels, boxes, {"image-a": {"columns": 100, "rows": 100}}, 42)
    assert len(rows) == 1
    assert rows[0]["parent_finding_support"]["positive_votes"] == 3
    assert rows[0]["horizontal_extent"]["unanimous_value"] == "single_image_hemifield"
    assert audit["bbox_label_mismatches"] == 0

    boxes[key].pop("R10")
    rows, audit = build_records(labels, boxes, {"image-a": {"columns": 100, "rows": 100}}, 42)
    assert rows == []
    assert audit["bbox_label_mismatches"] == 1


def test_gate_never_authorizes_clinical_semantics():
    # No data can bypass the separate clinical-admission requirement.
    rows = []
    summary = summarize(rows)
    gate = data_progression_gate(summary)
    assert not gate["clinical_semantics_authorized"]
    assert not gate["screen_pass"]
