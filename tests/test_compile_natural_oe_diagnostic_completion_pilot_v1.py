from anchor.corrected_sgta.audit_diagnostic_completion_substrate_v1 import EDGE_SPECS
from anchor.corrected_sgta.compile_natural_oe_diagnostic_completion_pilot_v1 import (
    PROMPT,
    compile_design,
)


def _votes():
    rows = {}
    for spec in EDGE_SPECS[:2]:
        for child_votes in (0, 3):
            for index in range(5):
                image_id = f"{spec.edge_id}-{child_votes}-{index}"
                row = {edge.parent_label: 0 for edge in EDGE_SPECS}
                row.update({edge.child_label: 0 for edge in EDGE_SPECS})
                row[spec.parent_label] = 3
                row[spec.child_label] = child_votes
                rows[image_id] = row
    return rows


def test_compiler_balances_two_extremes_without_model_outputs():
    result = compile_design(
        _votes(),
        edge_types=2,
        per_extreme=4,
        maximum_images=16,
        minimum_parent_votes=3,
    )
    rows = result["assignments"]
    assert len(rows) == 16
    assert len({row["image_id"] for row in rows}) == 16
    cells = {
        (row["edge_id"], row["child_votes"]): 0 for row in rows
    }
    for row in rows:
        cells[(row["edge_id"], row["child_votes"])] += 1
        assert row["parent_votes"] == 3
    assert set(cells.values()) == {4}


def test_prompt_does_not_name_diagnosis_uncertainty_or_absence_obligation():
    lowered = PROMPT.lower()
    for forbidden in (
        "pneumonia",
        "atelectasis",
        "tumor",
        "uncertain",
        "absent",
        "differential",
    ):
        assert forbidden not in lowered
