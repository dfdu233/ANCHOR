from anchor.corrected_sgta.freeze_reader_residual_specs_v1 import select_direction_cells


def cell(layer, family, negative_brier, negative_auc, positive_brier, positive_auc):
    return {
        "layer": layer,
        "family": family,
        "crossfit": {
            "negative_0v1": {
                "relative_brier_improvement": negative_brier,
                "delta_auc": negative_auc,
            },
            "positive_2v3": {
                "relative_brier_improvement": positive_brier,
                "delta_auc": positive_auc,
            },
        },
    }


def test_direction_specs_select_on_dev_and_exclude_final_layer():
    results = {
        "early_a": cell(7, "claim", .10, .03, .01, .02),
        "early_b": cell(14, "visual_std", .04, .08, .12, .04),
        "final": cell(28, "claim", .90, .90, .90, .90),
    }
    selected = select_direction_cells(results, final_layer=28)
    assert selected["negative_0v1"]["layer"] == 7
    assert selected["positive_2v3"]["layer"] == 14
    assert all(value["layer"] != 28 for value in selected.values())
