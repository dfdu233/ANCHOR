import pytest

from corrected_sgta.analyze_no_free_grounding import (
    align_inputs,
    analyze,
    exact_mcnemar_pvalue,
    metrics,
)


def _report(predictions, truths=None):
    truths = truths or ["yes"] * len(predictions)
    return {
        "details": [
            {
                "question_id": index,
                "prediction": None if pred == "invalid" else [pred],
                "ground_truth": [str(truth).lower()],
            }
            for index, (pred, truth) in enumerate(zip(predictions, truths))
        ]
    }


def _questions(truths):
    return [
        {"qid": index, "answer": truth, "img_name": f"image-{index // 2}.jpg"}
        for index, truth in enumerate(truths)
    ]


def test_metrics_count_invalid_as_class_specific_errors() -> None:
    rows = [
        {"truth": "yes", "prediction": "yes"},
        {"truth": "yes", "prediction": "invalid"},
        {"truth": "no", "prediction": "no"},
        {"truth": "no", "prediction": "yes"},
    ]
    result = metrics(rows)
    assert result["accuracy_invalid_as_error"] == 0.5
    assert result["balanced_accuracy_invalid_as_error"] == 0.5
    assert result["positive_recall"] == 0.5
    assert result["hallucination_risk_among_positive_claims"] == 0.5
    assert result["parse_rate"] == 0.75


def test_alignment_uses_qid_and_image_clusters() -> None:
    questions = _questions(["Yes", "No"])
    baseline, method, clusters = align_inputs(
        _report(["yes", "no"], ["yes", "no"]),
        _report(["no", "invalid"], ["yes", "no"]),
        questions,
    )
    assert [row["truth"] for row in baseline] == ["yes", "no"]
    assert [row["prediction"] for row in method] == ["no", "invalid"]
    assert clusters == ["image-0.jpg", "image-0.jpg"]


def test_analyze_rejects_answer_redistribution_with_recall_loss() -> None:
    truths = ["Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No"]
    baseline = _report(
        ["yes", "yes", "yes", "yes", "yes", "yes", "no", "no"], truths
    )
    method = _report(
        ["no", "no", "yes", "yes", "no", "no", "no", "no"], truths
    )
    result = analyze(baseline, method, _questions(truths), draws=300, seed=9)
    assert result["method"]["hallucination_risk_among_positive_claims"] == 0.0
    assert result["method"]["positive_recall"] == 0.5
    assert result["admission_gate"]["passed"] is False
    assert result["paired_correctness_transitions"]["baseline_only_correct"] == 2
    assert result["paired_correctness_transitions"]["method_only_correct"] == 2


def test_exact_mcnemar_pvalue_is_symmetric() -> None:
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(1, 4) == exact_mcnemar_pvalue(4, 1)
    assert exact_mcnemar_pvalue(0, 10) == pytest.approx(2 / 1024)
