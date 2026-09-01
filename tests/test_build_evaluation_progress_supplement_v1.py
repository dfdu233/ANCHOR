import json
from pathlib import Path

from anchor.medeval.build_evaluation_progress_supplement_v1 import build


def test_supplement_updates_only_r6_and_never_promotes_paper(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    internal = tmp_path / "internal.json"
    baseline = tmp_path / "baseline.json"
    base.write_text(
        json.dumps(
            {
                "paper_ready": False,
                "submission_claim_authorized": False,
                "requirements": [
                    {"id": "R1", "status": "pending_external", "evidence": "x"},
                    {"id": "R6", "status": "source_complete_execution_partial", "evidence": "old"},
                ],
            }
        )
    )
    controls = ["temperature_length_controls", "self_consistency", "calibrated_abstention"]
    internal.write_text(
        json.dumps(
            {
                "summary": {
                    "t2_pass": controls,
                    "t2_missing": [],
                    "t2_failed": [],
                    "t3_pass": [],
                    "full_pass": [],
                }
            }
        )
    )
    baseline.write_text(json.dumps({"version": "baseline-coverage-audit-v3"}))
    result = build(base_audit=base, internal_audit=internal, baseline_audit=baseline)
    assert result["paper_ready"] is False
    assert result["submission_claim_authorized"] is False
    rows = {row["id"]: row for row in result["requirements"]}
    assert rows["R1"]["status"] == "pending_external"
    assert rows["R6"]["status"] == "t2_engineering_complete_clinical_metrics_pending"
