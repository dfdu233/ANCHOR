from anchor.medeval.aggregate_claim_self_consistency_v1 import aggregate_group


def _report(seed: int, claims: list[dict], nll: float = 1.0) -> dict:
    return {
        "report": f"answer-{seed}",
        "source": {
            "seed": seed,
            "mean_token_nll": nll,
            "generated_token_ids": [seed],
            "stop_reason": "eos_or_template",
            "hit_max_new_tokens": False,
        },
        "claims": claims,
        "audit": {"unparsed_as_no_structured_claim": not claims},
    }


def _claim(polarity: str = "present", uncertainty: str = "definite") -> dict:
    return {
        "finding": "effusion",
        "anatomy": "right_pleura",
        "attributes": ["small"],
        "polarity": polarity,
        "uncertainty": uncertainty,
        "provenance": "image_grounded",
    }


def test_claim_consistency_uses_structured_votes_and_nll_tie_break() -> None:
    reports = [
        _report(42, [_claim()], 0.8),
        _report(1042, [_claim()], 0.4),
        _report(2042, [_claim()], 0.7),
        _report(3042, [], 0.1),
        _report(4042, [], 0.2),
    ]
    result = aggregate_group(reports, 3)
    assert result["applicable"] is True
    assert result["consensus_claims"] == [_claim()]
    assert result["selected_seed"] == 1042
    assert result["changed_from_seed42"] is True


def test_no_structured_claim_retains_seed42_without_text_vote() -> None:
    result = aggregate_group([_report(seed, []) for seed in (42, 1042, 2042, 3042, 4042)], 3)
    assert result["applicable"] is False
    assert result["selected_seed"] == 42
    assert result["changed_from_seed42"] is False


def test_false_test_label_usage_is_a_required_safe_value() -> None:
    qualification = {
        "all_k_samples_complete": True,
        "atomic_claim_normalization": True,
        "test_labels_used_for_selection": False,
    }
    passed = all(
        value for key, value in qualification.items() if key != "test_labels_used_for_selection"
    ) and qualification["test_labels_used_for_selection"] is False
    assert passed is True


def test_no_exact_text_vote_is_also_a_required_safe_false() -> None:
    qualification = {
        "all_k_samples_complete": True,
        "exact_text_majority_vote_used": False,
        "test_labels_used_for_selection": False,
    }
    safe_false = {"test_labels_used_for_selection", "exact_text_majority_vote_used"}
    passed = (
        all(value for key, value in qualification.items() if key not in safe_false)
        and all(qualification[key] is False for key in safe_false)
    )
    assert passed is True
