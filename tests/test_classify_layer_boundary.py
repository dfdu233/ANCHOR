from anchor.medeval.classify_layer_boundary import classify, summarize


def row(estimate, low, high, early_control, final_control, *, patch=False, powered=True):
    return {
        "model_id": "huatuo",
        "finding": "effusion",
        "direction": "negative",
        "direction_bins": ["0/3", "1/3"],
        "test_vote_bin_counts": {"0/3": 20, "1/3": 20},
        "early_minus_final_auroc": {"estimate": estimate, "ci_low": low, "ci_high": high},
        "increment_over_strongest_control": {
            "early": {"ci_high": early_control},
            "final": {"ci_high": final_control},
        },
        "causal_patch_passed": patch,
        "powered_for_margin": powered,
        "all_preregistered_controls_present": True,
    }


def test_four_boundary_states_and_indeterminate():
    assert classify(row(.08, .02, .13, .04, .02, patch=True)) == "early_erasure"
    assert classify(row(-.08, -.13, -.02, .04, .04)) == "late_emergence"
    assert classify(row(.01, -.03, .04, .02, .02)) == "layer_stable"
    assert classify(row(.01, -.08, .09, 0, -.01)) == "not_decodable"
    assert classify(row(.08, .02, .13, .04, .02, patch=False)) == "indeterminate"


def test_method_trigger_requires_two_models_majority_and_both_directions():
    rows = []
    for model in ("huatuo", "hulu"):
        for finding in ("effusion", "edema", "nodule"):
            for direction, bins in (("negative", ["0/3", "1/3"]), ("positive", ["3/3", "2/3"])):
                value = row(.08, .02, .13, .04, .02, patch=True)
                value.update(model_id=model, finding=finding, direction=direction, direction_bins=bins,
                             test_vote_bin_counts={name: 20 for name in bins})
                rows.append(value)
    assert summarize(rows)["method_branch_authorized"] is True
