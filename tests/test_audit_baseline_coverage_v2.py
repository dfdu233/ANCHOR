import json
from pathlib import Path

import pytest

from anchor.medeval.audit_baseline_coverage_v3 import audit


ROOT = Path(__file__).resolve().parents[1]


def _kwargs(internal: Path) -> dict:
    return {
        "config_path": ROOT / "configs/unified_eval/method_ladder_v1.json",
        "t0_path": ROOT / "corrected_runs/unified_eval/provenance/method_ladder_t0_v3.json",
        "evidence_path": ROOT / "corrected_runs/unified_eval/provenance/method_evidence_ladder_v9.json",
        "registry_path": ROOT / "corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl",
        "native_acceptance_path": ROOT / "corrected_runs/unified_eval/full/native_oe_greedy256_acceptance_v1.json",
        "rag_causal_path": ROOT / "corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_causal_controls_v2.json",
        "internal_control_path": internal,
        "report_audits": [
            ("hulu", ROOT / "corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/sanity_audit/summary.json"),
            ("llava", ROOT / "corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/sanity_audit/summary.json"),
        ],
    }


def test_current_coverage_binds_fail_closed_internal_controls() -> None:
    internal = ROOT / "corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v4.json"
    result = audit(**_kwargs(internal))
    assert result["version"] == "baseline-coverage-audit-v3"
    assert result["gates"]["internal_control_contract_enforced"] is True
    assert result["summary"]["internal_control_qualification"]["t2_pass"] == [
        "temperature_length_controls",
        "self_consistency",
    ]
    assert result["summary"]["internal_control_qualification"]["t2_missing"] == []
    assert result["summary"]["t2_failed_after_t0_pass"] == ["calibrated_abstention"]
    assert result["paper_baseline_claim_authorized"] is False


def test_v2_rejects_permissive_internal_control_claim(tmp_path: Path) -> None:
    source = ROOT / "corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v4.json"
    payload = json.loads(source.read_text())
    payload["paper_control_claim_authorized"] = True
    tampered = tmp_path / "internal.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stale or permissive"):
        audit(**_kwargs(tampered))
