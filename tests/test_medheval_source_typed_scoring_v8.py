import json
import sys

import anchor.corrected_sgta.evaluate_medheval_answers as evaluator
from anchor.corrected_sgta.evaluate_medheval_answers import (
    _multiclass_metrics,
    evaluate_rows,
    official_binary_label,
    official_choice_label,
)
from anchor.medeval.hashing import sha256_file
from anchor.medeval.prepare_baseline_matrix_inputs import normalize_question_type


def test_manifest_type_trusts_medheval_source_type() -> None:
    assert normalize_question_type("binary", "Normal", "") == "binary"
    assert normalize_question_type("binary", "Unclear", "") == "ternary"
    assert normalize_question_type(
        "multi-choice",
        "The lungs are clear.",
        "A. Congested, B. Clear, C. Inflamed",
    ) == "choice"


def test_official_proxy_matches_released_binary_and_choice_rules() -> None:
    assert official_binary_label("There is no edema.") == "no"
    assert official_binary_label("Uncertain") == "yes"
    choices = "A. Congested, B. Clear, C. Inflamed"
    assert official_choice_label("The lungs are clear.", choices) == "B"
    assert official_choice_label("B", choices) == "B"


def test_mixed_ce_reports_overall_accuracy_but_not_global_macro_metrics() -> None:
    rows = [
        {
            "qid": "binary",
            "source_question_type": "binary",
            "question_type": "short_answer",  # stale manifest field must not win
            "question": "Condition?",
            "answer": "Normal",
            "text": "Yes.",
        },
        {
            "qid": "choice",
            "source_question_type": "multi-choice",
            "question_type": "short_answer",  # stale manifest field must not win
            "question": "Condition?",
            "choices": "A. Congested, B. Clear, C. Inflamed",
            "answer": "The lungs are clear.",
            "text": "B",
        },
    ]
    report = evaluate_rows(rows)
    assert report["primary_metric"] == "primary_multiclass.accuracy_invalid_as_error"
    metrics = _multiclass_metrics(report["details"])
    assert report["official_benchmark_proxy"]["accuracy"] == 1.0
    assert [row["answer_type"] for row in report["details"]] == ["binary", "choice"]
    assert metrics["accuracy_invalid_as_error"] == 1.0
    assert metrics["balanced_accuracy"] is None
    assert metrics["macro_f1"] is None
    assert metrics["by_answer_type"]["binary"]["balanced_accuracy"] == 1.0
    assert metrics["by_answer_type"]["choice"]["macro_f1"] == 1.0


def test_free_text_singletons_cannot_enter_global_macro_f1() -> None:
    details = [
        {"answer_type": "short_answer", "ground_truth": ["unique one"], "prediction": ["unique one"], "correct": True},
        {"answer_type": "short_answer", "ground_truth": ["unique two"], "prediction": None, "correct": False},
    ]
    metrics = _multiclass_metrics(details)
    assert metrics["accuracy_invalid_as_error"] == 0.5
    assert metrics["balanced_accuracy"] is None
    assert metrics["macro_f1"] is None
    assert metrics["by_answer_type"] == {}


def test_cli_output_hash_binds_official_medheval_sources(tmp_path, monkeypatch) -> None:
    questions = tmp_path / "questions.json"
    answers = tmp_path / "answers.jsonl"
    output = tmp_path / "score.json"
    questions.write_text(json.dumps([{
        "qid": "one", "source_question_type": "binary", "question_type": "binary",
        "question": "Visible?", "answer": "No", "patient_id": "p1",
    }]))
    answers.write_text(json.dumps({"question_id": "one", "text": "No"}) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--answers", str(answers), "--questions", str(questions),
        "--output", str(output), "--bootstrap-replicates", "10",
    ])
    evaluator.main()
    payload = json.loads(output.read_text())
    provenance = payload["official_medheval_source_provenance"]
    assert len(provenance) == 3
    assert all(row["sha256"] == sha256_file(evaluator.Path(row["path"])) for row in provenance)
    assert payload["primary_multiclass"]["balanced_accuracy"] is None
