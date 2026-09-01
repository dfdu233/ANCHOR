from anchor.medeval.fit_calibrated_abstention_t2_v1 import (
    _uncertain_claims,
    fit_isotonic,
    select_nll_threshold,
    t2_gate_passed,
)


def test_isotonic_pooling_is_monotone() -> None:
    blocks = fit_isotonic([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0])
    probabilities = [block["estimated_correctness"] for block in blocks]
    assert probabilities == sorted(probabilities)
    assert sum(block["weight"] for block in blocks) == 4


def test_threshold_respects_coverage_and_uses_higher_coverage_tie_break() -> None:
    rows = [
        {"qid": "a", "nll": 0.1, "correct_proxy": True},
        {"qid": "b", "nll": 0.2, "correct_proxy": True},
        {"qid": "c", "nll": 0.3, "correct_proxy": False},
        {"qid": "d", "nll": 0.4, "correct_proxy": False},
    ]
    result = select_nll_threshold(rows, 0.5)
    assert result["nll_threshold"] == 0.2
    assert result["calibration_coverage"] == 0.5
    assert result["calibration_error_risk"] == 0.0


def test_rejected_claims_become_uncertain_without_deletion() -> None:
    claims = [
        {"finding": "effusion", "provenance": "image_grounded", "uncertainty": "definite"},
        {"finding": "cause", "provenance": "knowledge", "uncertainty": "definite"},
    ]
    output, changed = _uncertain_claims(claims)
    assert len(output) == len(claims)
    assert changed == 1
    assert output[0]["uncertainty"] == "uncertain"
    assert output[1]["uncertainty"] == "definite"


def _qualification(per_model):
    return {
        "calibration": {"deterministic_replay_passed": True},
        "accounting": {"claim_selective_oe": True},
        "diagnostics": {"per_model": per_model},
    }


def test_t2_gate_is_per_model_and_rejects_pooled_non_degeneracy() -> None:
    valid = {
        "calibration_positive_proxy_rows": 1,
        "calibration_negative_proxy_rows": 15,
        "validation_coverage_fraction": 0.875,
        "validation_extracted_claims": 10,
        "validation_claims_marked_uncertain": 1,
    }
    degenerate = {
        **valid,
        "calibration_positive_proxy_rows": 0,
        "calibration_negative_proxy_rows": 16,
        "validation_coverage_fraction": 1.0,
        "validation_claims_marked_uncertain": 0,
    }
    assert t2_gate_passed(_qualification({"hulu": valid}))
    assert not t2_gate_passed(_qualification({"hulu": valid, "huatuo": degenerate}))
