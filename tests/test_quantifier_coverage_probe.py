from corrected_sgta.run_slake_quantifier_coverage_probe import (
    half_boxes,
    intersection_area,
    matched_control_box,
    normalize_box,
)


def test_matched_control_box_preserves_shape_and_avoids_target() -> None:
    target = normalize_box([70, 55, 24, 18], width=160, height=120, padding=0.1)
    control = matched_control_box(target, width=160, height=120)
    assert control[2] - control[0] == target[2] - target[0]
    assert control[3] - control[1] == target[3] - target[1]
    assert intersection_area(target, control) == 0


def test_half_boxes_partition_an_odd_width_image() -> None:
    boxes = half_boxes(width=161, height=120)
    left = boxes["left_half_occlusion"]
    right = boxes["right_half_occlusion"]
    assert intersection_area(left, right) == 0
    assert (left[2] - left[0]) * 120 + (right[2] - right[0]) * 120 == 161 * 120
