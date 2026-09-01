from __future__ import annotations

import pytest

from anchor.medeval.evaluate_oe_brevity_controls import evaluate, first_sentence


def test_first_sentence_preserves_unsplit_short_answer() -> None:
    assert first_sentence("right") == "right"
    assert first_sentence("Right. Added unsupported claim.") == "Right."


def test_brevity_control_exposes_length_coverage_tradeoff() -> None:
    manifest = [
        {"qid": "q1", "answer": "right", "image_sha256": "a"},
        {"qid": "q2", "answer": "left lower lobe", "image_sha256": "b"},
    ]
    answers = [
        {"question_id": "q1", "text": "Right. Added unsupported claim."},
        {"question_id": "q2", "text": "Long preamble before left lower lobe appears."},
    ]
    result = evaluate(manifest, answers, replicates=100, seed=3)
    first = result["first_sentence"]
    assert first["changed_rate"] == pytest.approx(1 / 2)
    assert first["absolute"]["output_diagnostics"]["reference_phrase_coverage_rate"] == 1
    assert first["lexically_coverage_matched_point_estimate"] is True
    short = result["first_8_words"]
    assert "paired_lexical_vs_original" in short
    assert "paired_diagnostics_vs_original" in short
