from pathlib import Path

from corrected_sgta.evaluate_oe_reports import normalize_rows, score_text_pair
from corrected_sgta.report_protocol import (
    has_unnegated_abnormal_finding,
    infer_report_task,
    is_normal_template,
    report_prompt,
)


def test_task_and_modality_are_not_conflated():
    mimic = infer_report_task({"dataset": "mimic_report_oe", "task": "report"})
    harvard = infer_report_task({"dataset": "harvard_test", "task": "report"})
    knowledge = infer_report_task({"dataset": "MedHEval", "task": "knowledge_oe", "question": "What is shown?"})
    assert mimic.modality == "chest_radiograph"
    assert mimic.clinical_metric_family == "chest_radiograph"
    assert harvard.modality == "ophthalmology"
    assert harvard.clinical_metric_family is None
    assert knowledge.task == "open_vqa"


def test_official_prompt_uses_no_target_reference():
    sample = {"dataset": "mimic_report_oe", "task": "report", "question": "<image> Generate a report.", "answer": "SECRET TARGET REPORT"}
    prompt = report_prompt(sample, "official_zero_shot")
    assert "SECRET TARGET REPORT" not in prompt
    assert "professional radiologist" in prompt


def test_rag_prompt_requires_retrieval():
    sample = {"dataset": "iuxray", "task": "report"}
    try:
        report_prompt(sample, "official_rag")
    except ValueError:
        pass
    else:
        raise AssertionError("official_rag accepted an empty retrieval set")
    assert "retrieved source report" in report_prompt(sample, "official_rag", ["retrieved source report"])


def test_negation_aware_sanity_flags():
    assert not has_unnegated_abnormal_finding("No pneumothorax, pleural effusion, focal opacity, or fractures.")
    assert has_unnegated_abnormal_finding("No pneumothorax. There are fractures of the sternotomy wires.")
    assert is_normal_template("The lungs are clear. No acute cardiopulmonary process.")
    assert not is_normal_template("No acute cardiopulmonary process. There are fractures of the sternotomy wires.")


def test_text_metrics_are_identical_on_exact_match():
    values = score_text_pair("No acute disease.", "No acute disease.")
    assert values["rouge_l"] == 1.0
    assert values["meteor"] > 0.9
    assert values["bleu"] > 0.5  # sentence BLEU-4 penalizes references shorter than four tokens


def test_normalization_restricts_counterfactual_views(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"id": "1", "dataset": "mimic_report_oe", "task": "report", "view": "real", "reference": "No acute disease.", "text": "No acute disease."},
        {"id": "1", "dataset": "mimic_report_oe", "task": "report", "view": "null", "reference": "No acute disease.", "text": "Normal."},
    ]
    path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows))
    normalized = normalize_rows([path], "auto", real_view_only=True)
    assert len(normalized) == 1
    assert normalized[0]["clinical_metric_family"] == "chest_radiograph"
