from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.audit_claim_boundary_regrounding_substrate_v1 import (
    AuditError,
    atomic_write_new,
    audit,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def claim_artifact(path: Path, *, records: list[dict[str, object]]) -> None:
    entities = {
        str(index + 1): {
            "tokens": str(record["claim"]["finding"]),
            "start_ix": index * 4 + 1,
            "end_ix": index * 4 + 1,
        }
        for index, record in enumerate(records)
    }
    normalized = []
    for index, record in enumerate(records):
        row = dict(record)
        row["root_entity_id"] = str(index + 1)
        row["component_entity_ids"] = [str(index + 1)]
        normalized.append(row)
    write_json(
        path,
        {
            "config": {
                "version": "missing-third-state-radgraph-claims-v2",
                "fingerprint": path.stem,
            },
            "reports": [
                {
                    "id": "case-1",
                    "report": "Finding one. Finding two.",
                    "claims": [row["claim"] for row in normalized],
                    "audit": {
                        "records": normalized,
                        "radgraph_entities": entities,
                    },
                }
            ],
        },
    )


def two_records() -> list[dict[str, object]]:
    return [
        {
            "claim": {
                "finding": "finding_one",
                "polarity": "present",
                "uncertainty": "definite",
                "provenance": "image_grounded",
            }
        },
        {
            "claim": {
                "finding": "finding_two",
                "polarity": "absent",
                "uncertainty": "definite",
                "provenance": "image_grounded",
            }
        },
    ]


def test_audit_is_fail_closed_without_independent_truth_or_claimwise_cf(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    prediction = tmp_path / "prediction.json"
    summary = tmp_path / "summary.json"
    counterfactual = tmp_path / "counterfactual.jsonl"
    claim_artifact(reference, records=two_records())
    claim_artifact(prediction, records=two_records())
    write_json(
        summary,
        {
            "config": {"evidence_grade": "C"},
            "interpretation_contract": {
                "claim_ceiling": "single reference report; not clinical truth"
            },
        },
    )
    counterfactual.write_text(
        json.dumps(
            {
                "id": "case-1",
                "text": "Finding one. Finding two.",
                "zero_visual_generated_evidence": {"mean_nll": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = audit(
        reference,
        summary,
        [("greedy", prediction)],
        [("zero", counterfactual)],
    )

    assert result["status"] == "no_go_current_substrate"
    assert result["reference_truth_audit"]["independent_per_claim_visual_truth"] is False
    assert result["counterfactual_audit"]["zero"][
        "claim_boundary_counterfactual_eligible"
    ] is False
    assert result["gates"]["formal_mechanism_analysis_authorized"] is False
    assert result["gates"]["gpu_authorized"] is False
    assert result["outcome_blind_contract"][
        "generated_vs_reference_claim_matching_performed"
    ] is False
    ordinal = result["prediction_ordinal_audit"]["greedy"]
    assert ordinal["explicit_native_ordinal_records"] == 0
    assert ordinal["text_position_recoverable_records"] == 2


def test_missing_radgraph_position_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    prediction = tmp_path / "prediction.json"
    summary = tmp_path / "summary.json"
    claim_artifact(reference, records=two_records())
    claim_artifact(prediction, records=two_records())
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    del payload["reports"][0]["audit"]["radgraph_entities"]["2"]["start_ix"]
    write_json(prediction, payload)
    write_json(summary, {"config": {}, "interpretation_contract": {}})

    with pytest.raises(AuditError, match="lacks RadGraph start_ix"):
        audit(reference, summary, [("greedy", prediction)], [])


def test_atomic_write_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    atomic_write_new(output, b"first\n")
    with pytest.raises(FileExistsError):
        atomic_write_new(output, b"second\n")

