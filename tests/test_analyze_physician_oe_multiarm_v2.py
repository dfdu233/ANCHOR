from anchor.medeval.analyze_physician_oe_multiarm_v2 import (
    EXPECTED_GATE_SPEC,
    bind_t3,
)


def test_t3_binding_rejects_gate_drift_and_fixes_scope() -> None:
    gates = {
        "primary_error_reduction": True,
        "holm_adjusted_primary_p_below_0p05": True,
        "matched_coverage_error_reduction": True,
        "required_recall_noninferior_0p05": True,
        "direct_correctness_noninferior_0p05": True,
        "harm_not_increased_0p05": True,
        "refusal_not_increased_0p01": True,
        "length_at_least_90pct": True,
        "visual_claims_at_least_90pct": True,
    }
    result = {"contrasts": {"method": {"promotion_gates": gates, "t3_promotion_authorized": True}}}
    bound = bind_t3(result, {"machine_gate_spec": EXPECTED_GATE_SPEC})
    assert bound["evidence_stage"] == "T3"
    assert "Full efficacy is not established" in bound["claim_boundary"]

    bad = dict(EXPECTED_GATE_SPEC)
    bad["length_ratio_at_least"] = 0.5
    try:
        bind_t3(result, {"machine_gate_spec": bad})
    except ValueError as error:
        assert "gate spec" in str(error)
    else:
        raise AssertionError("gate drift was accepted")
