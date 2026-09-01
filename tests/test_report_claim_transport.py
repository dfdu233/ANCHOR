from corrected_sgta.analyze_report_claim_transport import _counts, analyze
from corrected_sgta.prepare_report_claim_transport import _reference_state


def test_reference_state_never_treats_missing_as_negative() -> None:
    assert _reference_state([]) == "unverified"
    assert _reference_state([{
        "polarity": "present", "uncertainty": "definite", "provenance": "knowledge"
    }]) == "unverified"
    assert _reference_state([{
        "polarity": "absent", "uncertainty": "definite", "provenance": "image_grounded"
    }]) == "refuted"


def test_unknown_aware_fixed_k_counts() -> None:
    image = {
        "reference_states": {
            "a": "supported", "b": "refuted", "c": "unverified", "d": "undetermined"
        }
    }
    result = _counts(image, {"a", "c"})
    assert result["k"] == 2
    assert result["supported"] == 1
    assert result["unverified"] == 1
    assert result["verified_coverage"] == 0.5


def test_clean_refuted_to_supported_transport_passes() -> None:
    images = []
    for index in range(12):
        images.append({
            "image_id": str(index),
            "baseline": {
                "k": 1, "supported": 0, "refuted": 1, "undetermined": 0,
                "unverified": 0, "verified": 1, "verified_precision": 0.0,
                "supported_recall": 0.0, "verified_coverage": 1.0, "supported_total": 1,
            },
            "candidate": {
                "k": 1, "supported": 1, "refuted": 0, "undetermined": 0,
                "unverified": 0, "verified": 1, "verified_precision": 1.0,
                "supported_recall": 1.0, "verified_coverage": 1.0, "supported_total": 1,
            },
        })
    result = analyze(images, draws=200, seed=5)
    assert result["screening_gate"]["passed"] is True


def test_bootstrap_draw_without_verified_claims_is_not_an_error() -> None:
    image = {
        "image_id": "unknown-only",
        "baseline": {
            "k": 1, "supported": 0, "refuted": 0, "undetermined": 0,
            "unverified": 1, "verified": 0, "verified_precision": None,
            "supported_recall": None, "verified_coverage": 0.0, "supported_total": 0,
        },
        "candidate": {
            "k": 1, "supported": 0, "refuted": 0, "undetermined": 1,
            "unverified": 0, "verified": 0, "verified_precision": None,
            "supported_recall": None, "verified_coverage": 0.0, "supported_total": 0,
        },
    }
    result = analyze([image], draws=20, seed=1)
    assert result["screening_gate"]["passed"] is False
    assert result["candidate_minus_baseline_image_bootstrap"]["verified_precision"]["ci_low"] is None
