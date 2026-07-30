from anchor.corrected_sgta.analyze_mimic_question_frame_probe import (
    metrics,
)
from anchor.corrected_sgta.run_mimic_question_frame_probe import (
    framed_questions,
)


def test_neutral_frame_preserves_original_question():
    question = "Is there pleural effusion?"
    frames = framed_questions(question)
    assert question in frames["original"]
    assert question in frames["neutral"]


def test_question_frame_metrics_count_confusion():
    rows = [
        {
            "ground_truth": "Yes.",
            "rule_prediction": "yes",
        },
        {
            "ground_truth": "No.",
            "rule_prediction": "yes",
        },
    ]
    result = metrics(rows)
    assert result["accuracy"] == 0.5
    assert result["tp"] == 1
    assert result["fp"] == 1
