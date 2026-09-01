import json

from corrected_sgta.run_oe_sanity_audit import (
    load_manifest,
    mismatched_donor_indices,
    summarize_rows,
)


def test_report_manifest_accepts_common_protocol_img_name(tmp_path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "img_name": "study/image.jpg",
                    "answer": "Small left pleural effusion.",
                    "dataset": "mimic",
                }
            ]
        )
    )
    rows = load_manifest(path)
    assert rows[0]["image"] == "study/image.jpg"
    assert rows[0]["reference_abnormal"]


def test_manifest_recovers_mimic_patient_from_path(tmp_path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {
            "id": "image-a",
            "img_name": "p15/p15378103/s50000001/a.jpg",
            "answer": "Left pleural effusion.",
            "dataset": "mimic",
        },
        {
            "id": "image-b",
            "img_name": "p15/p15378103/s50000002/b.jpg",
            "answer": "No focal opacity.",
            "dataset": "mimic",
        },
    ]))
    rows = load_manifest(path)
    assert rows[0]["patient_id"] == "p15378103"
    assert rows[1]["patient_id"] == "p15378103"


def test_dependency_gate_rejects_identical_real_and_null_outputs() -> None:
    rows = []
    for index in range(8):
        for view in ("real", "null"):
            rows.append(
                {
                    "id": str(index),
                    "method": "greedy",
                    "conv_mode": "native",
                    "prompt_mode": "mmedrag",
                    "view": view,
                    "text": f"finding {index}",
                    "reference": "finding",
                }
            )
    result = summarize_rows(rows)
    assert not result["admissible_for_report_generation_claim"]
    assert result["pairwise_image_dependency"]["real_vs_null"]["exact_same_rate"] == 1.0


def test_existing_only_audit_is_noncollapse_not_grounding() -> None:
    rows = [
        {
            "id": str(index),
            "method": "greedy",
            "conv_mode": "native",
            "prompt_mode": "official_zero_shot",
            "view": "real",
            "text": f"Finding number {index} is present.",
            "reference": "Finding is present.",
        }
        for index in range(8)
    ]
    result = summarize_rows(rows)
    assert result["output_noncollapse_pass"]
    assert not result["image_dependency_tested"]
    assert not result["image_dependency_pass"]
    assert not result["admissible_for_report_generation_claim"]


def test_official_prompt_requires_both_controls_and_material_change() -> None:
    rows = []
    for index in range(8):
        for view, text in (
            ("real", f"left pleural effusion {index}"),
            ("null", f"normal cardiac silhouette {index}"),
            ("shuffled", f"no focal opacity {index}"),
        ):
            rows.append({
                "id": str(index),
                "method": "greedy",
                "conv_mode": "native",
                "prompt_mode": "mmedrag",
                "view": view,
                "text": text,
                "reference": "left pleural effusion",
            })
    result = summarize_rows(rows)
    assert result["image_dependency_tested"]
    assert result["image_dependency_pass"]
    assert result["admissible_for_report_generation_claim"]


def test_mismatched_control_uses_different_patient_and_image(tmp_path) -> None:
    records = [
        {"id": "a", "patient_id": "p1"},
        {"id": "b", "patient_id": "p2"},
        {"id": "c", "patient_id": "p3"},
    ]
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(bytes([index]))
        paths.append(path)
    donors = mismatched_donor_indices(records, paths)
    assert len(set(donors)) == 3
    for source, donor in enumerate(donors):
        assert source != donor
        assert records[source]["patient_id"] != records[donor]["patient_id"]


def test_mismatched_control_is_one_to_one_with_repeated_patient(tmp_path) -> None:
    records = [
        {"id": "a", "patient_id": "p1"},
        {"id": "b", "patient_id": "p1"},
        {"id": "c", "patient_id": "p2"},
        {"id": "d", "patient_id": "p3"},
    ]
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(bytes([index]))
        paths.append(path)
    donors = mismatched_donor_indices(records, paths)
    assert len(set(donors)) == len(records)
    for source, donor in enumerate(donors):
        assert records[source]["patient_id"] != records[donor]["patient_id"]
