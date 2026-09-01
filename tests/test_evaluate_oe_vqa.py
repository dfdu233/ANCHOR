from __future__ import annotations

import pytest

from anchor.medeval.evaluate_oe_vqa import (
    align_and_score,
    answer_token_recall,
    answer_tokens,
    normalize_answer,
    paired_summary,
    summarize,
)


def manifest():
    return [
        {"qid": "q1", "answer": "the right side", "image_sha256": "image-a"},
        {"qid": "q2", "answer": "two", "image_sha256": "image-a"},
        {"qid": "q3", "answer": "left lower lobe", "image_sha256": "image-b"},
    ]


def test_normalization_is_transparent_and_vqa_style() -> None:
    assert answer_tokens("The TWO right-sided lesions.") == ["2", "right", "sided", "lesions"]
    assert normalize_answer("Left-lower lobe") == "left lower lobe"
    assert answer_token_recall("It is in the right side of the chest", "right side") == 1


def test_alignment_rejects_reference_drift_and_missing_rows() -> None:
    with pytest.raises(ValueError, match="alignment failure"):
        align_and_score(manifest(), [{"question_id": "q1", "text": "right"}])
    answers = [
        {"question_id": "q1", "text": "right side", "gt_ans": "WRONG"},
        {"question_id": "q2", "text": "2"},
        {"question_id": "q3", "text": "left lower lobe"},
    ]
    with pytest.raises(ValueError, match="reference mismatch"):
        align_and_score(manifest(), answers)


def test_summary_bootstraps_images_not_questions() -> None:
    answers = [
        {"question_id": "q1", "text": "right side"},
        {"question_id": "q2", "text": "2"},
        {"question_id": "q3", "text": "wrong"},
    ]
    rows = align_and_score(manifest(), answers)
    result = summarize(rows, replicates=100, seed=7)
    assert result["n_questions"] == 3
    assert result["n_images"] == 2
    assert result["metrics"]["normalized_exact"]["clusters"] == 2
    assert result["metrics"]["normalized_exact"]["estimate"] == pytest.approx(2 / 3)
    assert result["metrics"]["bleu_1"]["estimate"] > 0
    assert result["metrics"]["rouge_1_recall"]["estimate"] > 0
    assert result["metrics"]["rouge_l_f1"]["estimate"] > 0
    assert result["output_diagnostics"]["median_reference_tokens"] == 2
    assert result["output_diagnostics"]["generated_token_count_coverage"] == 0
    assert result["output_diagnostics"]["token_budget_hit_rate"] is None


def test_summary_records_length_and_budget_as_nonclinical_diagnostics() -> None:
    answers = [
        {
            "question_id": "q1",
            "text": "The right side is visible.",
            "metadata": {"generated_token_count": 8},
        },
        {
            "question_id": "q2",
            "text": "There are two findings",
            "metadata": {"generated_token_count": 7},
        },
        {
            "question_id": "q3",
            "text": "left lower lobe",
            "metadata": {"generated_token_count": 3},
        },
    ]
    result = summarize(
        align_and_score(manifest(), answers),
        replicates=100,
        seed=7,
        max_new_tokens=8,
    )
    diagnostics = result["output_diagnostics"]
    assert diagnostics["generated_token_count_coverage"] == 1
    assert diagnostics["token_budget_hit_rate"] == pytest.approx(1 / 3)
    assert diagnostics["reference_phrase_coverage_rate"] == 1
    assert diagnostics["terminal_punctuation_rate"] == pytest.approx(1 / 3)
    assert "clinical hallucination score" in diagnostics["interpretation"]


def test_paired_summary_has_explicit_direction() -> None:
    baseline = align_and_score(manifest(), [
        {"question_id": "q1", "text": "wrong"},
        {"question_id": "q2", "text": "wrong"},
        {"question_id": "q3", "text": "left lower lobe"},
    ])
    candidate = align_and_score(manifest(), [
        {"question_id": "q1", "text": "right side"},
        {"question_id": "q2", "text": "2"},
        {"question_id": "q3", "text": "left lower lobe"},
    ])
    result = paired_summary(candidate, baseline, replicates=100, seed=11)
    assert result["direction"] == "candidate_minus_baseline"
    assert result["metrics"]["normalized_exact"]["estimate"] == pytest.approx(2 / 3)
