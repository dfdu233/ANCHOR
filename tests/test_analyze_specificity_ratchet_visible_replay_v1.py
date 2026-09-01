from __future__ import annotations

from corrected_sgta.analyze_specificity_ratchet_visible_replay_v1 import analyze_rows


def _row(index: int, role: str, *, swap_transition: float, own_transition: float, early: float):
    return {
        "sample_id": f"{role}-{index}",
        "case_id": f"case-{role}-{index}",
        "split": "test",
        "status": "ok",
        "scientific_role": role,
        "edge_type": ("laterality", "size_morph", "subtype")[index % 3],
        "modality_stratum": "XR",
        "anatomy_stratum": "thorax",
        "prompt_requested_increment": bool(index % 2),
        "constraint_lexical_key_sha256": f"constraint-{index}",
        "signals": {
            "layer_ids": ["early", "late"],
            "token_counts": {
                "full_visible_answer": 12 + index % 2,
                "constraint": 1,
                "child_surface": 4,
                "matched_child_nonconstraint": 1,
            },
            "raw_commitment": {
                "constraint_minus_matched": [0.0, own_transition],
                "mean_swap_constraint_logp": [0.0, swap_transition],
                "mean_swap_matched_nonconstraint_logp": [0.0, 0.0],
            },
            "primary_own_minus_matched_swaps": {
                "constraint_minus_matched_difference_in_differences": [early, early + own_transition - swap_transition],
                "per_swap_difference_in_differences": [
                    [early, early + own_transition - swap_transition],
                    [early, early + own_transition - swap_transition],
                ],
            },
            "text_only_secondary": {"difference_in_differences": [0.0, 0.0]},
        },
    }


def _balanced_rows(*, error_swap: float = 1.4):
    rows = []
    for index in range(12):
        jitter = (index % 4) * 0.01
        rows.append(
            _row(
                index,
                "supported_specificity_control",
                swap_transition=jitter,
                own_transition=jitter,
                early=1.0 + jitter,
            )
        )
        rows.append(
            _row(
                index,
                "causal_escalation_error",
                swap_transition=error_swap + jitter,
                own_transition=1.2 + jitter,
                early=-1.0 + jitter,
            )
        )
    return rows


def test_joint_ratchet_signature_passes_only_with_swap_survival_and_early_deficit():
    result = analyze_rows(_balanced_rows(), bootstrap_replicates=300, seed=9)
    assert result["status"] == "passed"
    assert all(result["gate_checks"].values())
    fraction = result["estimates"]["swap_survival_fraction_of_own_transition"]
    assert 1.1 < fraction["estimate"] < 1.2


def test_own_only_late_gain_fails_language_side_ratchet_gate():
    result = analyze_rows(
        _balanced_rows(error_swap=0.0), bootstrap_replicates=300, seed=11
    )
    assert result["status"] == "failed"
    assert result["gate_checks"]["own_commitment_ratchet_positive"] is True
    assert result["gate_checks"]["swap_language_ratchet_positive"] is False
    assert result["gate_checks"]["swap_exceeds_half_own_by_linear_contrast"] is False


def test_sparse_roles_are_reported_underpowered_even_with_large_effect():
    rows = _balanced_rows()[:6]
    result = analyze_rows(rows, bootstrap_replicates=200, seed=17)
    assert result["status"] == "underpowered"


def test_exact_lexical_singletons_cap_positive_mechanism_at_pilot_only():
    rows = _balanced_rows()
    for row in rows:
        row["constraint_lexical_key_sha256"] = row["sample_id"]
    result = analyze_rows(rows, bootstrap_replicates=300, seed=23)
    assert all(result["gate_checks"].values())
    assert result["status"] == "pilot_only"
    assert result["exact_lexical_overlap"]["cross_role_keys"] == 0


def test_mean_swap_cannot_hide_one_failed_swap_or_positive_visual_catchup():
    rows = _balanced_rows(error_swap=0.8)
    result = analyze_rows(rows, bootstrap_replicates=300, seed=29)
    assert result["gate_checks"]["swap_exceeds_half_own_by_linear_contrast"] is True
    assert result["gate_checks"]["no_positive_image_specific_late_catchup"] is False
    assert result["status"] == "failed"

    rows = _balanced_rows(error_swap=1.4)
    for row in rows:
        if row["scientific_role"] != "causal_escalation_error":
            continue
        own = row["signals"]["raw_commitment"]["constraint_minus_matched"][-1]
        early = row["signals"]["primary_own_minus_matched_swaps"][
            "constraint_minus_matched_difference_in_differences"
        ][0]
        row["signals"]["primary_own_minus_matched_swaps"][
            "per_swap_difference_in_differences"
        ] = [[early, early + own - 2.8], [early, early + own - 0.0]]
    result = analyze_rows(rows, bootstrap_replicates=300, seed=31)
    assert result["gate_checks"]["swap_language_ratchet_positive"] is True
    assert result["gate_checks"]["swap2_ratchet_positive"] is False
    assert result["status"] == "failed"
