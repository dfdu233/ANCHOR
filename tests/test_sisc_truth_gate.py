import json
from pathlib import Path

from anchor.corrected_sgta.build_sisc_truth_gate import (
    assess_gate,
    build_study_view_manifest,
    load_tam2020_visible_truth,
)


def _mimic_row(subject: int, study: int, image: str, report: str = "same report") -> dict:
    return {
        "id": image,
        "study_id": study,
        "subject_id": subject,
        "report": report,
        "image_path": [f"p{subject}/s{study}/{image}.jpg"],
        "split": "test",
    }


def test_manifest_is_patient_disjoint_and_does_not_emit_report_text(tmp_path: Path) -> None:
    source = tmp_path / "mimic.json"
    source.write_text(
        json.dumps(
            [
                _mimic_row(1, 10, "a"),
                _mimic_row(1, 10, "b"),
                _mimic_row(1, 11, "c", "other report"),
            ]
        )
    )
    manifest, audit = build_study_view_manifest(source)
    assert audit["paired_view_studies"] == 1
    assert len({row["gate_split"] for row in manifest}) == 1
    assert all("report" not in row for row in manifest)
    assert all(row["view_position"] == "unknown" for row in manifest)


def test_missing_expert_box_is_not_negative_or_unassessable(tmp_path: Path) -> None:
    source = tmp_path / "mimic.json"
    source.write_text(json.dumps([_mimic_row(1, 10, "a"), _mimic_row(1, 10, "b")]))
    manifest, _ = build_study_view_manifest(source)
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "a.jpg"}],
                "categories": [{"id": 1, "name": "pneumonia"}],
                "annotations": [
                    {"id": 9, "image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4], "creator": "R1"}
                ],
            }
        )
    )
    truth, audit = load_tam2020_visible_truth(annotations, manifest)
    assert len(truth) == 1
    assert truth[0]["image_id"] == "a"
    assert truth[0]["truth_state"] == "visible"
    assert audit["missing_box_interpreted_as_refuted"] is False
    result = assess_gate(manifest, truth, min_paired_studies=1, min_findings=1, min_view_exclusive_per_finding=1)
    assert result["decision"] == "NO-GO"
    assert result["counts"]["view_exclusive_studies_by_finding"] == {}


def test_gate_passes_only_with_explicit_visible_and_nonvisible_truth() -> None:
    manifest = []
    truth = []
    for index in range(100):
        for side in ("frontal", "lateral"):
            image_id = f"{index}-{side}"
            manifest.append(
                {
                    "image_id": image_id,
                    "study_id": str(index),
                    "subject_id": str(index),
                    "gate_split": "train" if index < 70 else "dev" if index < 85 else "test",
                }
            )
        for finding in ("a", "b", "c"):
            truth.append(
                {
                    "image_id": f"{index}-frontal",
                    "finding": finding,
                    "truth_state": "visible",
                    "independent_image_local_evidence": True,
                }
            )
            truth.append(
                {
                    "image_id": f"{index}-lateral",
                    "finding": finding,
                    "truth_state": "unassessable" if index % 2 else "refuted",
                    "independent_image_local_evidence": True,
                }
            )
    result = assess_gate(manifest, truth)
    assert result["decision"] == "GO"
    assert result["counts"]["eligible_findings"] == ["a", "b", "c"]


def test_outcome_keys_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    row = _mimic_row(1, 10, "a")
    row["prediction"] = "forbidden"
    source.write_text(json.dumps([row]))
    try:
        build_study_view_manifest(source)
    except ValueError as exc:
        assert "model-outcome" in str(exc)
    else:
        raise AssertionError("model outcome was not rejected")
