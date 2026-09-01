import csv
import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.compile_specificity_ratchet_mechanism_manifest_v1 import (
    compile_manifest,
    exact_constraint_spans,
    exact_observed_child_span,
)
from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
    AdjudicationValidationError,
    validate_adjudication,
)


REVIEW_ANNOTATION_FIELDS = [
    "reviewer_id",
    "edge_entailment_admitted",
    "parent_visual_support",
    "child_visual_support",
    "increment_observability",
    "logical_scope_preserved",
    "reviewer_confidence",
    "clinical_usefulness_if_backed_off",
    "clinically_harmful_if_wrong",
    "rationale",
]
ADJUDICATION_FIELDS = [
    "case_id",
    "edge_id",
    "r1_edge_entailment_admitted",
    "r2_edge_entailment_admitted",
    "r1_parent_visual_support",
    "r2_parent_visual_support",
    "r1_child_visual_support",
    "r2_child_visual_support",
    "r1_increment_observability",
    "r2_increment_observability",
    "r1_logical_scope_preserved",
    "r2_logical_scope_preserved",
    "r1_clinical_usefulness_if_backed_off",
    "r2_clinical_usefulness_if_backed_off",
    "r1_clinically_harmful_if_wrong",
    "r2_clinically_harmful_if_wrong",
    "r1_reviewer_confidence",
    "r2_reviewer_confidence",
    "r1_rationale",
    "r2_rationale",
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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate(index: int, constraint: str, edge_type: str = "laterality") -> dict:
    return {
        "added_constraint_proposal": constraint,
        "anatomy_stratum": "thorax" if index % 2 == 0 else "neuro",
        "answer_length_stratum": "short_le_50",
        "answer_span": f"A {constraint} lesion is present.",
        "case_id": f"CASE-{index}",
        "child_proposal": f"A {constraint} lesion is present.",
        "edge_id": f"EDGE-{index}",
        "edge_type": edge_type,
        "image_relpath": f"test_images/{index}.jpg",
        "modality_stratum": "XR" if index % 2 == 0 else "MRI",
        "observability_screen": "potentially_single_image_decidable",
        "parent_proposal": "A lesion is present.",
        "prompt_requested_increment": False,
        "proposal_only": True,
        "question": "What is present?",
    }


def _build_complete_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    candidates = [
        _candidate(0, "left"),
        _candidate(1, "right"),
        _candidate(2, "small", "size_morph"),
        _candidate(3, "large", "size_morph"),
    ]
    (pack / "candidates.blinded.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidates)
    )
    schema = {
        "fields": {
            "edge_entailment_admitted": ["yes", "no", "uncertain"],
            "parent_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
            "child_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
            "increment_observability": [
                "observable_on_supplied_image",
                "requires_other_view_or_sequence",
                "requires_history_lab_pathology_or_prior",
                "fundamentally_nonvisual_knowledge",
                "uncertain",
            ],
            "logical_scope_preserved": ["yes", "no", "not_applicable"],
            "reviewer_confidence": ["low", "medium", "high"],
            "clinical_usefulness_if_backed_off": [
                "improves", "unchanged", "minor_loss", "major_loss", "uncertain"
            ],
            "clinically_harmful_if_wrong": ["no", "minor", "major", "uncertain"],
        }
    }
    (pack / "annotation_schema.json").write_text(json.dumps(schema))
    child_states = ["supported", "refuted", "undetermined", "unobservable"]
    sources = [
        "observable_on_supplied_image",
        "observable_on_supplied_image",
        "observable_on_supplied_image",
        "requires_history_lab_pathology_or_prior",
    ]
    reviewer_rows = []
    for reviewer_id in ("PHYS-A", "PHYS-B"):
        rows = []
        for candidate, child, source in zip(candidates, child_states, sources):
            row = dict(candidate)
            row.update(
                {
                    "reviewer_id": reviewer_id,
                    "edge_entailment_admitted": "yes",
                    "parent_visual_support": "supported",
                    "child_visual_support": child,
                    "increment_observability": source,
                    "logical_scope_preserved": "yes",
                    "reviewer_confidence": "high",
                    "clinical_usefulness_if_backed_off": "minor_loss",
                    "clinically_harmful_if_wrong": "minor",
                    "rationale": "Independent image review completed.",
                }
            )
            rows.append(row)
        reviewer_rows.append(rows)
        _write_csv(
            pack / f"annotations.reviewer_{1 if reviewer_id == 'PHYS-A' else 2}.csv",
            [*candidates[0], *REVIEW_ANNOTATION_FIELDS],
            rows,
        )
    adjudication = []
    for index, (candidate, child, source) in enumerate(zip(candidates, child_states, sources)):
        row = {"case_id": candidate["case_id"], "edge_id": candidate["edge_id"]}
        for reviewer_number, reviewer in enumerate(reviewer_rows, start=1):
            source_row = reviewer[index]
            for field in (
                "edge_entailment_admitted",
                "parent_visual_support",
                "child_visual_support",
                "increment_observability",
                "logical_scope_preserved",
                "clinical_usefulness_if_backed_off",
                "clinically_harmful_if_wrong",
                "reviewer_confidence",
                "rationale",
            ):
                row[f"r{reviewer_number}_{field}"] = source_row[field]
        row.update(
            {
                "final_edge_entailment_admitted": "yes",
                "final_parent_visual_support": "supported",
                "final_child_visual_support": child,
                "final_increment_observability": source,
                "final_logical_scope_preserved": "yes",
                "final_clinical_usefulness_if_backed_off": "minor_loss",
                "final_clinically_harmful_if_wrong": "minor",
                "adjudicator_id": "PHYS-C",
                "disagreement_reason": "",
                "adjudication_rationale": "Final state confirmed while blinded.",
            }
        )
        adjudication.append(row)
    _write_csv(pack / "adjudication.csv", ADJUDICATION_FIELDS, adjudication)
    (pack / "physician_attestations.json").write_text(
        json.dumps(
            {
                "protocol_id": "specificity-ratchet-physician-pack-v2",
                "reviewers": [
                    {
                        "reviewer_id": reviewer_id,
                        "role": "physician",
                        "independent_review": True,
                        "blinded_to_private_provenance": True,
                        "completed_at_utc": "2026-08-02T00:00:00Z",
                    }
                    for reviewer_id in ("PHYS-A", "PHYS-B")
                ],
                "adjudicator": {
                    "adjudicator_id": "PHYS-C",
                    "role": "physician",
                    "blinded_to_private_provenance": True,
                    "completed_at_utc": "2026-08-02T01:00:00Z",
                },
            }
        )
    )
    return pack


def test_blank_real_templates_fail_closed_without_outputs(tmp_path):
    pack = Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2")
    output = tmp_path / "samples.jsonl"
    metadata = tmp_path / "metadata.json"
    with pytest.raises(AdjudicationValidationError):
        compile_manifest(pack, output, metadata)
    assert not output.exists()
    assert not metadata.exists()


def test_complete_two_physician_adjudication_compiles_grouped_manifest(tmp_path):
    pack = _build_complete_pack(tmp_path)
    output = tmp_path / "samples.jsonl"
    metadata = tmp_path / "metadata.json"
    result = compile_manifest(pack, output, metadata)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 4
    assert {row["scientific_role"] for row in rows} == {
        "supported_specificity_control",
        "causal_escalation_error",
        "evidence_source_boundary",
    }
    by_case = {}
    for row in rows:
        assert by_case.setdefault(row["case_id"], row["split"]) == row["split"]
        assert row["mitigation_claim_count_delta"] == 0
        assert row["constraint_char_spans_in_child"]
        assert row["child_target_exact_observed_substring"] is True
        assert row["child_target_span_in_observed_generation"]["utf8_sha256"]
    assert set(by_case.values()) == {"dev", "test"}
    assert result["image_disjoint"] is True
    assert result["manifest_sha256"]


def test_adjudication_must_exactly_copy_frozen_reviewer_sheet(tmp_path):
    pack = _build_complete_pack(tmp_path)
    with (pack / "adjudication.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["r1_child_visual_support"] = "refuted"
    _write_csv(pack / "adjudication.csv", ADJUDICATION_FIELDS, rows)
    with pytest.raises(AdjudicationValidationError, match="exactly copy"):
        validate_adjudication(pack)


def test_physician_attestations_are_mandatory(tmp_path):
    pack = _build_complete_pack(tmp_path)
    (pack / "physician_attestations.json").unlink()
    with pytest.raises(AdjudicationValidationError, match="missing physician attestation"):
        validate_adjudication(pack)


def test_adjudicator_must_be_identity_distinct_from_both_reviewers(tmp_path):
    pack = _build_complete_pack(tmp_path)
    with (pack / "adjudication.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["adjudicator_id"] = "PHYS-A"
    _write_csv(pack / "adjudication.csv", ADJUDICATION_FIELDS, rows)
    attestation = json.loads((pack / "physician_attestations.json").read_text())
    attestation["adjudicator"]["adjudicator_id"] = "PHYS-A"
    (pack / "physician_attestations.json").write_text(json.dumps(attestation))
    with pytest.raises(AdjudicationValidationError, match="must differ"):
        validate_adjudication(pack)


def test_unavailable_evidence_source_requires_unobservable_child(tmp_path):
    pack = _build_complete_pack(tmp_path)
    with (pack / "adjudication.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["final_child_visual_support"] = "undetermined"
    _write_csv(pack / "adjudication.csv", ADJUDICATION_FIELDS, rows)
    with pytest.raises(AdjudicationValidationError, match="requires child=unobservable"):
        validate_adjudication(pack)


def test_exact_constraint_span_keeps_all_repeated_added_occurrences():
    candidate = _candidate(0, "right")
    candidate["child_proposal"] = "A right lesion abuts the right pleura."
    spans = exact_constraint_spans(candidate)
    assert [span["text"] for span in spans] == ["right", "right"]
    assert spans[0]["char_end_exclusive"] <= spans[1]["char_start"]


def test_observed_child_anchor_refuses_missing_or_ambiguous_rewrite():
    candidate = _candidate(0, "left")
    candidate["answer_span"] = "A left lesion is present. A left lesion is present."
    with pytest.raises(ValueError, match="exactly once"):
        exact_observed_child_span(candidate)
    candidate["answer_span"] = "A right lesion is present."
    with pytest.raises(ValueError, match="exactly once"):
        exact_observed_child_span(candidate)
