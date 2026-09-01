import unittest

from corrected_sgta.evaluate_rule_vqa import (
    evaluate_rule_rows,
    rule_normalized_prediction,
)


class RuleNormalizedPredictionTest(unittest.TestCase):
    def test_normalizes_bracketed_labels(self) -> None:
        self.assertEqual(rule_normalized_prediction("[no]."), "no")
        self.assertEqual(rule_normalized_prediction("[yes]."), "yes")

    def test_preserves_first_sentence_negative_word_convention(self) -> None:
        self.assertEqual(
            rule_normalized_prediction("The image does not show edema."),
            "no",
        )
        self.assertEqual(
            rule_normalized_prediction("No edema is present. Yes later."),
            "no",
        )

    def test_uses_word_boundaries(self) -> None:
        self.assertEqual(rule_normalized_prediction("Nocturnal symptoms."), "yes")

    def test_primary_metric_requires_a_leading_explicit_decision(self) -> None:
        questions = [
            {"question_id": "1", "answer": "no"},
            {"question_id": "2", "answer": "yes"},
        ]
        answers = [
            {"question_id": "1", "text": "[no]."},
            {"question_id": "2", "text": "Yes, the finding is present."},
        ]
        metrics, records = evaluate_rule_rows(questions, answers)
        self.assertEqual(
            metrics["primary_metric"],
            "leading_explicit.accuracy_invalid_as_error",
        )
        self.assertEqual(metrics["leading_explicit"]["correct"], 2)
        self.assertEqual(metrics["leading_explicit"]["accuracy_invalid_as_error"], 1.0)
        self.assertEqual(records[0]["rule_normalized_prediction"], "no")

    def test_later_negation_cannot_flip_leading_yes(self) -> None:
        questions = [{"question_id": "1", "answer": "yes"}]
        answers = [{"question_id": "1", "text": "Yes, edema was not present previously."}]
        metrics, records = evaluate_rule_rows(questions, answers)
        self.assertEqual(records[0]["leading_explicit_prediction"], "yes")
        self.assertEqual(metrics["leading_explicit"]["correct"], 1)
        self.assertEqual(records[0]["rule_normalized_prediction"], "no")

    def test_natural_sentence_without_leading_decision_is_invalid(self) -> None:
        questions = [{"question_id": "1", "answer": "no"}]
        answers = [{"question_id": "1", "text": "There is no pleural effusion."}]
        metrics, _ = evaluate_rule_rows(questions, answers)
        self.assertEqual(metrics["leading_explicit"]["parseable"], 0)
        self.assertEqual(metrics["leading_explicit"]["correct"], 0)


if __name__ == "__main__":
    unittest.main()
