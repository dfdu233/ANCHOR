import json

from anchor.medeval.audit_rag_dual_track_v1 import audit_dual_track


def test_common_generation_pass_does_not_authorize_native_or_efficacy(tmp_path) -> None:
    methods = []
    for name in ("RULE", "MMed-RAG", "FactMM-RAG", "MR-RAG"):
        methods.append(
            {
                "name": name,
                "stages": {"T0": {"status": "not_admissible", "reason": "missing"}},
            }
        )
    methods.append(
        {
            "name": "shared_medical_rag",
            "stages": {
                "T0": {"status": "pass"},
                "T1": {"status": "pass"},
                "T2": {"status": "pass"},
                "T3": {"status": "pass"},
                "full": {"status": "failed_cutoff", "reason": "grounding failed"},
            },
        }
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"methods": methods}))
    factmm = tmp_path / "factmm.json"
    factmm.write_text(
        json.dumps(
            {
                "paper_native_t0_status": "not_admissible",
                "missing_requirements": ["generator"],
            }
        )
    )
    result = audit_dual_track(evidence, factmm)
    assert result["common_protocol"]["T3_generation_qualification"] == "pass"
    assert result["common_protocol"]["clinical_grounding_or_utility_authorized"] is False
    assert result["any_rag_efficacy_authorized"] is False
    assert all(row["efficacy_authorized"] is False for row in result["paper_native"])
