import json
from pathlib import Path

from anchor.medeval.audit_baseline_matrix_execution_v1 import ROOT, score_binding_failure
from anchor.medeval.hashing import sha256_file


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_ce_score_requires_current_evaluator_hash(tmp_path: Path) -> None:
    answers = _write(tmp_path / "answers.jsonl", {"answer": "yes"})
    questions = _write(tmp_path / "questions.jsonl", {"answer": "yes"})
    evaluator = ROOT / "anchor" / "corrected_sgta" / "evaluate_medheval_answers.py"
    payload = {
        "evaluator_source": str(evaluator),
        "evaluator_source_sha256": sha256_file(evaluator),
        "answers": str(answers),
        "answers_sha256": sha256_file(answers),
        "questions": str(questions),
        "questions_sha256": sha256_file(questions),
    }
    score = _write(tmp_path / "ce.json", payload)
    assert score_binding_failure(score, "mixed_ce") is None
    payload["evaluator_source_sha256"] = "0" * 64
    _write(score, payload)
    assert "current evaluator" in score_binding_failure(score, "mixed_ce")


def test_oe_score_requires_current_evaluator_path(tmp_path: Path) -> None:
    answers = _write(tmp_path / "answers.jsonl", {"answer": "yes"})
    manifest = _write(tmp_path / "manifest.jsonl", {"answer": "yes"})
    evaluator = ROOT / "anchor" / "medeval" / "evaluate_oe_vqa.py"
    payload = {
        "evaluator_source": str(evaluator),
        "evaluator_source_sha256": sha256_file(evaluator),
        "answers": [str(answers)],
        "answer_sha256": [sha256_file(answers)],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    score = _write(tmp_path / "oe.json", payload)
    assert score_binding_failure(score, "open_vqa") is None
    payload["evaluator_source"] = str(tmp_path / "other.py")
    _write(score, payload)
    assert "current evaluator" in score_binding_failure(score, "open_vqa")


def test_report_score_requires_current_evaluator_hash(tmp_path: Path) -> None:
    pairs = _write(tmp_path / "pairs.jsonl", {"model_answer": "normal"})
    evaluator = ROOT / "corrected_sgta" / "evaluate_oe_reports.py"
    payload = {
        "config": {
            "code_sha256": sha256_file(evaluator),
            "inputs": [{"path": str(pairs), "sha256": sha256_file(pairs)}],
        }
    }
    score = _write(tmp_path / "report.json", payload)
    assert score_binding_failure(score, "report_generation") is None
    payload["config"]["code_sha256"] = "f" * 64
    _write(score, payload)
    assert "current evaluator" in score_binding_failure(score, "report_generation")
