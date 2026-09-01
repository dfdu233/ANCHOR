from anchor.corrected_sgta.build_reader_boundary_paper_packet_v1 import (
    build_rows,
    render_markdown,
)


def comparison(value=.01):
    metric = {"estimate": value, "ci_low": value - .01, "ci_high": value + .01}
    return {"delta_auc": metric, "relative_brier_improvement": metric}


def test_packet_rows_and_markdown_preserve_fail_closed_decision():
    comparisons = {
        name: comparison()
        for name in (
            "early_vs_evidence", "final_vs_evidence", "early_vs_final",
            "early_vs_random", "early_vs_direct_maybe", "early_vs_confidence",
            "early_vs_entropy",
        )
    }
    confirmation = {
        "model_id": "m",
        "results": {
            "negative_0v1": {
                "n": 120,
                "boundary": "Layer-stable",
                "representation_controls_pass": False,
                "comparisons": comparisons,
                "finding_wise": {
                    "f": {"n": 120, "boundary": "Layer-stable", "comparisons": comparisons}
                },
            }
        },
    }
    pooled, findings = build_rows([confirmation])
    assert len(pooled) == len(findings) == 1
    assert pooled[0]["early_vs_final_delta_auc_estimate"] == .01
    summary = {
        "paper_gate": {"qualified_cells": 1, "consistent_cells": 1, "strict_majority_consistent": False},
        "observational_early_erasure_all_models": False,
        "next_action": "do_not_build_decoder",
    }
    markdown = render_markdown(summary, pooled)
    assert "Decoder/method authorized: False" in markdown
    assert "Layer-stable" in markdown
