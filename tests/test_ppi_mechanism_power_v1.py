import numpy as np

from anchor.corrected_sgta.simulate_ppi_mechanism_power_v1 import one_trial


def test_registered_logit_assay_separates_gated_prior_from_artifacts():
    gated = one_trial("evidence_gated", 100, np.random.default_rng(1))
    trigger = one_trial("unconditional_trigger", 100, np.random.default_rng(2))
    margin = one_trial("margin_artifact", 100, np.random.default_rng(3))
    assert gated["mechanism_admitted"]
    assert gated["clarity_interaction_slope"] < -0.10
    assert not trigger["interaction_pass"]
    assert not trigger["mechanism_admitted"]
    assert margin["surface_weak_margin_pattern"]
    assert not margin["diagonal_pass"]
    assert not margin["mechanism_admitted"]
