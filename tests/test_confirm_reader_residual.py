import numpy as np

from anchor.corrected_sgta.confirm_reader_residual_v1 import (
    bootstrap_pair,
    classify_boundary,
    passes_positive_control,
)


def comp(auc, low, high, brier=.08, blow=.01, bhigh=.12):
    return {"delta_auc": {"estimate": auc, "ci_low": low, "ci_high": high}, "relative_brier_improvement": {"estimate": brier, "ci_low": blow, "ci_high": bhigh}}


def test_boundary_classifier_is_fail_closed():
    assert classify_boundary(comp(.08,.02,.13), comp(.01,-.02,.04), comp(.07,.01,.12)) == "Early erasure"
    assert classify_boundary(comp(.01,-.02,.04), comp(.08,.02,.13), comp(-.07,-.12,-.01)) == "Late emergence"
    assert classify_boundary(comp(.02,-.01,.04), comp(.02,-.01,.04), comp(0,-.03,.03)) == "Layer-stable"
    assert classify_boundary(comp(0,-.03,.03,brier=0,blow=-.03,bhigh=.03), comp(0,-.03,.03,brier=0,blow=-.03,bhigh=.03), comp(.08,-.08,.12)) == "Not decodable"
    assert classify_boundary(
        comp(0,-.03,.03,brier=0,blow=-.03,bhigh=.03),
        comp(0,-.03,.03,brier=0,blow=-.03,bhigh=.03),
        comp(.08,-.08,.12),
        decodability_controls=(comp(.08,.01,.12),),
    ) == "Indeterminate"
    assert classify_boundary(comp(.04,-.01,.08), comp(.02,-.02,.06), comp(.03,-.07,.09)) == "Indeterminate"


def test_cluster_bootstrap_reports_direct_point_estimate_and_cluster_count():
    y = np.asarray([0, 0, 1, 1])
    groups = np.asarray(["a", "b", "c", "d"])
    left = np.asarray([.4, .3, .6, .7])
    right = np.asarray([.2, .1, .8, .9])
    result = bootstrap_pair(y, groups, left, right, draws=200, seed=7)
    assert result["delta_auc"]["estimate"] == 0.0
    assert result["relative_brier_improvement"]["estimate"] > 0
    assert result["n_clusters"] == 4


def test_representation_control_requires_both_metrics_and_ci():
    assert passes_positive_control(comp(.08, .01, .12, brier=.06, blow=.01))
    assert not passes_positive_control(comp(.08, -.01, .12, brier=.06, blow=.01))
    assert not passes_positive_control(comp(.08, .01, .12, brier=.06, blow=-.01))
