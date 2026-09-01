from corrected_sgta.run_slake_claim_involution_probe import (
    binary_auc,
    involution_coordinates,
)


def _score(supported: float, refuted: float, undetermined: float):
    return {
        "probabilities": {
            "supported": supported,
            "refuted": refuted,
            "undetermined": undetermined,
        }
    }


def test_involution_cancels_a_shared_yes_response_bias() -> None:
    affirmative = _score(0.6, 0.3, 0.1)
    complement = _score(0.6, 0.3, 0.1)
    coordinates = involution_coordinates(affirmative, complement)
    assert abs(coordinates["semantic_presence_margin"]) < 1e-12
    assert coordinates["framing_disagreement"] > 0.0


def test_involution_recovers_semantic_presence_across_complement_frames() -> None:
    affirmative = _score(0.8, 0.1, 0.1)
    complement = _score(0.1, 0.8, 0.1)
    coordinates = involution_coordinates(affirmative, complement)
    assert coordinates["semantic_presence_margin"] > 0.0
    assert abs(coordinates["framing_disagreement"]) < 1e-12


def test_binary_auc_counts_ties_as_half() -> None:
    assert binary_auc([0, 0, 1, 1], [0.0, 1.0, 1.0, 2.0]) == 0.875
