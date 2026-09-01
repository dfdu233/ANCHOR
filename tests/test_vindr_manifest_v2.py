import csv
from pathlib import Path

from corrected_sgta.prepare_vindr_reader_manifest_v2 import (
    fixed_panel_records,
    read_bbox_annotations,
    select_split_balanced,
    three_way_split,
)


def _row(image: str, finding: str, vote: int, panel=("R8", "R9", "R10")):
    return {
        "image_id": image,
        "finding": finding,
        "positive_votes": vote,
        "reader_ids": list(panel),
    }


def test_three_way_split_is_global_across_claims():
    image = "same-radiograph"
    assert three_way_split(image, 42) == three_way_split(image, 42)
    assert three_way_split(image, 42) in {"pilot", "dev", "confirmation"}


def test_fixed_panel_rejects_panel_composition_confound():
    rows = [
        _row("a", "effusion", 2),
        _row("b", "effusion", 2, ("R1", "R2", "R3")),
    ]
    selected = fixed_panel_records(rows, ("R8", "R9", "R10"))
    assert [row["image_id"] for row in selected] == ["a"]
    assert selected[0]["panel_policy"] == "exact_fixed_panel"


def test_balanced_selection_has_exact_quotas_and_no_image_leakage():
    rows = []
    findings = {"effusion", "nodule"}
    # Populate each hash split directly; sharing IDs across findings exercises
    # the global image-level leakage constraint.
    split_ids = {name: [] for name in ("pilot", "dev", "confirmation")}
    index = 0
    while min(map(len, split_ids.values())) < 8:
        image = f"image-{index}"
        split_ids[three_way_split(image, 42)].append(image)
        index += 1
    for finding in findings:
        for vote in range(4):
            for split, images in split_ids.items():
                for image in images[:4]:
                    rows.append(_row(image, finding, vote))
    selected, _ = select_split_balanced(
        rows, findings, {"pilot": 2, "dev": 2, "confirmation": 2}, seed=42
    )
    counts = {}
    image_splits = {}
    for row in selected:
        key = (row["finding"], row["positive_votes"], row["experiment_split"])
        counts[key] = counts.get(key, 0) + 1
        image_splits.setdefault(row["image_id"], set()).add(row["experiment_split"])
    assert set(counts.values()) == {2}
    assert all(len(splits) == 1 for splits in image_splits.values())


def test_bbox_reader_filter_and_coordinate_validation(tmp_path: Path):
    path = tmp_path / "bbox.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id", "class_name", "rad_id", "x_min", "y_min", "x_max", "y_max"
            ],
        )
        writer.writeheader()
        writer.writerow(
            dict(image_id="a", class_name="Pleural effusion", rad_id="R8", x_min=1, y_min=2, x_max=8, y_max=9)
        )
        writer.writerow(
            dict(image_id="a", class_name="Pleural effusion", rad_id="R2", x_min=1, y_min=2, x_max=8, y_max=9)
        )
        writer.writerow(
            dict(image_id="a", class_name="Pleural effusion", rad_id="R9", x_min=8, y_min=2, x_max=1, y_max=9)
        )
    rows, audit = read_bbox_annotations(
        path, {("a", "pleural_effusion")}, {"R8", "R9", "R10"}
    )
    assert len(rows) == 1
    assert rows[0]["boxes"][0]["rad_id"] == "R8"
    assert audit["valid_boxes"] == 1
