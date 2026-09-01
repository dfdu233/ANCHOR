from anchor.corrected_sgta.audit_ppi_source_two_plane_assignment_v1 import (
    BITS,
    SourceUnit,
    assignment_metrics,
    solve_plus,
    solve_zero,
    source_labels,
)


def _units():
    states = [
        ("positive", "positive"),
        ("positive", "uncertain"),
        ("negative", "negative"),
        ("negative", "uncertain"),
        ("uncertain", "positive"),
        ("uncertain", "negative"),
        ("unmentioned", "positive"),
        ("unmentioned", "negative"),
    ] * 2
    return [
        SourceUnit(
            response_unit_id=str(index),
            source_group=f"pmc_{index}",
            response="two source words",
            word_count=3,
            modality="CT",
            length_bin="000-024",
            archive="images_1.zip",
            labels={"claim_a": first, "claim_b": second},
        )
        for index, (first, second) in enumerate(states)
    ]


def test_source_label_semantics_do_not_turn_missing_or_uncertain_into_polarity():
    assert source_labels({"x": "positive"}, "x") == (1, 1)
    assert source_labels({"x": "negative"}, "x") == (-1, 1)
    assert source_labels({"x": "uncertain"}, "x") == (0, -1)
    assert source_labels({}, "x") == (0, 0)


def test_plus_solver_balances_combos_and_respects_registered_pairing():
    units = _units()
    assignment, solver = solve_plus(units, ["claim_a", "claim_b"], 1, 17)
    metrics = assignment_metrics(units, ["claim_a", "claim_b"], assignment)
    assert metrics["combo_counts"] == {0: 4, 1: 4, 2: 4, 3: 4}
    assert solver["optimal_min_target_contrast"] > 0
    assert metrics["claims"]["claim_a"]["u_mu"] > 0
    assert metrics["claims"]["claim_b"]["u_mu"] > 0
    assert metrics["claims"]["claim_a"]["v_kappa"] > 0
    assert metrics["claims"]["claim_b"]["v_kappa"] < 0
    minus = [3 - combo for combo in assignment]
    assert all(
        BITS[minus_combo] == (-BITS[plus_combo][0], -BITS[plus_combo][1])
        for plus_combo, minus_combo in zip(assignment, minus)
    )


def test_zero_solver_minimizes_all_source_associations():
    units = _units()
    assignment, solver = solve_zero(units, ["claim_a", "claim_b"], 23)
    metrics = assignment_metrics(units, ["claim_a", "claim_b"], assignment)
    assert metrics["combo_counts"] == {0: 4, 1: 4, 2: 4, 3: 4}
    realized = [abs(value) for claim in metrics["claims"].values() for value in claim.values()]
    assert max(realized) <= solver["optimal_max_absolute_contrast"] + 0.011
