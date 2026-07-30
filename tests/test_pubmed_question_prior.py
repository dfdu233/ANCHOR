import numpy as np

from anchor.corrected_sgta.analyze_pubmed_question_prior import (
    group_bootstrap_difference,
)


def test_group_bootstrap_question_prior_is_positive():
    cue = np.asarray([1, 1, 0, 0])
    positive = np.asarray([1, 1, 0, 0])
    groups = np.asarray(["a", "b", "c", "d"])
    lower, upper = group_bootstrap_difference(
        cue, positive, groups, draws=100
    )
    assert lower >= 0
    assert upper <= 1
