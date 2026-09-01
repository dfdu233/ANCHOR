from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / (
    "corrected_runs/unified_eval/physician_review/"
    "vqa_rad_t2_multiarm_v1/clinical_analysis_prereg_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_physician_analysis_prereg_is_label_blind_and_source_bound() -> None:
    row = json.loads(PREREG.read_text())
    assert row["frozen_before_physician_labels"] is True
    assert row["clinical_labels_inspected"] is False
    assert row["all_gates_required"] is True
    provenance = row["provenance"]
    expected = {
        "prepare_adjudication_source_sha256": ROOT
        / "anchor/medeval/prepare_physician_oe_adjudication.py",
        "finalize_consensus_source_sha256": ROOT
        / "anchor/medeval/finalize_physician_oe_consensus.py",
        "analysis_source_sha256": ROOT
        / "anchor/medeval/analyze_physician_oe_multiarm.py",
        "review_template_sha256": ROOT
        / (
            "corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/"
            "review.template.jsonl"
        ),
        "private_mapping_sha256": ROOT
        / (
            "corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/"
            "review.private_mapping.jsonl"
        ),
        "delivery_manifest_sha256": ROOT
        / (
            "corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/"
            "deliveries_v1/delivery_manifest.json"
        ),
    }
    for field, path in expected.items():
        assert provenance[field] == _sha256(path)

