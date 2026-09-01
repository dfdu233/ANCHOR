import csv
import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import (
    ReviewMergeError,
    merge_reviews,
)


REVIEW_FIELDS = [
    "edge_entailment_admitted",
    "parent_visual_support",
    "child_visual_support",
    "increment_observability",
    "logical_scope_preserved",
    "reviewer_confidence",
    "clinical_usefulness_if_backed_off",
    "clinically_harmful_if_wrong",
]


def _write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path):
    candidates = [
        {
            "case_id": "case-a",
            "edge_id": "edge-a",
            "question": "What finding?",
            "image_relpath": "test_images/a.jpg",
            "parent_proposal": "opacity",
            "child_proposal": "left opacity",
            "added_constraint_proposal": "left",
            "edge_type": "laterality",
            "proposal_only": True,
        },
        {
            "case_id": "case-b",
            "edge_id": "edge-b",
            "question": "What finding?",
            "image_relpath": "test_images/b.jpg",
            "parent_proposal": "lesion",
            "child_proposal": "large lesion",
            "added_constraint_proposal": "large",
            "edge_type": "size_morph",
            "proposal_only": True,
        },
    ]
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    allowed = {
        "edge_entailment_admitted": ["yes", "no", "uncertain"],
        "parent_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
        "child_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
        "increment_observability": ["observable_on_supplied_image", "uncertain"],
        "logical_scope_preserved": ["yes", "no", "not_applicable"],
        "reviewer_confidence": ["low", "medium", "high"],
        "clinical_usefulness_if_backed_off": ["improves", "unchanged", "minor_loss", "major_loss", "uncertain"],
        "clinically_harmful_if_wrong": ["no", "minor", "major", "uncertain"],
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"protocol_id": "specificity-ratchet-physician-pack-v2", "fields": allowed}))
    immutable = list(candidates[0])
    reviewer_fields = immutable + ["reviewer_id", *REVIEW_FIELDS, "rationale"]
    reviewer_paths = []
    for number in (1, 2):
        rows = []
        for candidate in candidates:
            row = {key: str(value) if not isinstance(value, bool) else str(value) for key, value in candidate.items()}
            row.update(
                reviewer_id=f"doctor-{number}",
                edge_entailment_admitted="yes",
                parent_visual_support="supported",
                child_visual_support="undetermined" if number == 1 else "supported",
                increment_observability="observable_on_supplied_image",
                logical_scope_preserved="yes",
                reviewer_confidence="high",
                clinical_usefulness_if_backed_off="unchanged",
                clinically_harmful_if_wrong="minor",
                rationale=f"Independent review {number}.",
            )
            rows.append(row)
        path = tmp_path / f"reviewer-{number}.csv"
        _write_csv(path, reviewer_fields, rows)
        reviewer_paths.append(path)
    copied = [f"r{number}_{field}" for number in (1, 2) for field in [*REVIEW_FIELDS, "rationale"]]
    final = [
        "final_edge_entailment_admitted",
        "final_parent_visual_support",
        "final_child_visual_support",
        "final_increment_observability",
        "final_logical_scope_preserved",
        "final_clinical_usefulness_if_backed_off",
        "final_clinically_harmful_if_wrong",
        "adjudicator_id",
        "disagreement_reason",
        "adjudication_rationale",
    ]
    template_path = tmp_path / "adjudication.csv"
    _write_csv(
        template_path,
        ["case_id", "edge_id", *copied, *final],
        [{"case_id": row["case_id"], "edge_id": row["edge_id"]} for row in candidates],
    )
    return candidates_path, schema_path, template_path, reviewer_paths


def test_merge_copies_only_reviewer_fields_and_keeps_final_blank(tmp_path):
    candidates, schema, template, reviewers = _fixture(tmp_path)
    header, rows, metadata = merge_reviews(
        candidates_path=candidates,
        schema_path=schema,
        template_path=template,
        reviewer_1_path=reviewers[0],
        reviewer_2_path=reviewers[1],
    )
    assert metadata["reviewer_ids"] == ["doctor-1", "doctor-2"]
    assert metadata["clinical_truth_created"] is False
    assert len(rows) == 2
    assert rows[0]["r1_child_visual_support"] == "undetermined"
    assert rows[0]["r2_child_visual_support"] == "supported"
    assert all(not value for key, value in rows[0].items() if key.startswith("final_"))
    assert not rows[0]["adjudicator_id"]
    assert set(rows[0]) == set(header)


def test_merge_rejects_changed_immutable(tmp_path):
    candidates, schema, template, reviewers = _fixture(tmp_path)
    fields, rows = _read(reviewers[1])
    rows[0]["parent_proposal"] = "changed"
    _write_csv(reviewers[1], fields, rows)
    with pytest.raises(ReviewMergeError, match="immutable field changed"):
        merge_reviews(
            candidates_path=candidates,
            schema_path=schema,
            template_path=template,
            reviewer_1_path=reviewers[0],
            reviewer_2_path=reviewers[1],
        )


def test_merge_rejects_same_reviewer_identity(tmp_path):
    candidates, schema, template, reviewers = _fixture(tmp_path)
    fields, rows = _read(reviewers[1])
    for row in rows:
        row["reviewer_id"] = "doctor-1"
    _write_csv(reviewers[1], fields, rows)
    with pytest.raises(ReviewMergeError, match="distinct reviewer IDs"):
        merge_reviews(
            candidates_path=candidates,
            schema_path=schema,
            template_path=template,
            reviewer_1_path=reviewers[0],
            reviewer_2_path=reviewers[1],
        )


def test_merge_rejects_blank_or_formula_rationale(tmp_path):
    candidates, schema, template, reviewers = _fixture(tmp_path)
    fields, rows = _read(reviewers[0])
    rows[0]["rationale"] = "=HYPERLINK(\"bad\")"
    _write_csv(reviewers[0], fields, rows)
    with pytest.raises(ReviewMergeError, match="formula prefix"):
        merge_reviews(
            candidates_path=candidates,
            schema_path=schema,
            template_path=template,
            reviewer_1_path=reviewers[0],
            reviewer_2_path=reviewers[1],
        )


def test_merge_rejects_nonblank_adjudication_template(tmp_path):
    candidates, schema, template, reviewers = _fixture(tmp_path)
    fields, rows = _read(template)
    rows[0]["final_parent_visual_support"] = "supported"
    _write_csv(template, fields, rows)
    with pytest.raises(ReviewMergeError, match="final values"):
        merge_reviews(
            candidates_path=candidates,
            schema_path=schema,
            template_path=template,
            reviewer_1_path=reviewers[0],
            reviewer_2_path=reviewers[1],
        )


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)
