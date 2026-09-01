from corrected_sgta.analyze_claim_action_audit import aggregate, finding_axes, report_counts


def report(claims, text="brief report"):
    return {
        "report": text,
        "claims": claims,
        "audit": {"n_observation_roots": len(claims), "unmatched_observations": []},
    }


def claim(finding, polarity="present", uncertainty="definite"):
    return {
        "finding": finding,
        "polarity": polarity,
        "uncertainty": uncertainty,
        "provenance": "image_grounded",
    }


def test_hedged_positive_keeps_content_polarity():
    reference = report([claim("effusion", polarity="absent")])
    prediction = report([claim("effusion", uncertainty="uncertain")])
    counts = report_counts(prediction, reference)
    metrics = aggregate([counts])
    assert metrics["axis_aware_explicit_negative_false_positive_rate"] == 1.0
    assert metrics["legacy_collapsed_third_state_false_positive_rate"] is None
    assert metrics["masked_hedged_false_positive_count"] == 1


def test_unmentioned_reference_finding_is_not_assumed_negative():
    reference = report([claim("effusion")])
    prediction = report([claim("nodule")])
    counts = report_counts(prediction, reference)
    assert counts["axis_resolved_positive"] == 0.0
    assert counts["axis_false_positive"] == 0.0


def test_positive_duplicate_dominates_negative_but_marks_contradiction():
    axes = finding_axes(
        report([claim("edema", polarity="absent"), claim("edema")])
    )
    assert axes["edema"]["polarity"] == "present"
    assert axes["edema"]["contradictory"] is True
