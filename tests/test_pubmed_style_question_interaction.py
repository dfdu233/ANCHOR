import numpy as np

from anchor.corrected_sgta.analyze_pubmed_style_question_interaction import (
    interaction_features,
    shuffled_within_family,
)


def test_interaction_features_are_samplewise_kronecker_products():
    question = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    style = np.asarray([[5.0, 6.0], [7.0, 8.0]])
    result = interaction_features(question, style)
    assert result.shape == (2, 4)
    np.testing.assert_allclose(result[0], [5.0, 6.0, 10.0, 12.0])


def test_shuffle_preserves_each_family_multiset():
    style = np.arange(12).reshape(6, 2)
    families = np.asarray(["a", "a", "a", "b", "b", "b"])
    shuffled = shuffled_within_family(
        style, families, np.random.default_rng(7)
    )
    for family in ["a", "b"]:
        original = style[families == family]
        result = shuffled[families == family]
        assert sorted(map(tuple, original)) == sorted(map(tuple, result))
