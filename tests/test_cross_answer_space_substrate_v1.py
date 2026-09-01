from __future__ import annotations

from anchor.corrected_sgta.audit_cross_answer_space_substrate_v1 import (
    audit_provenance,
    gate_summary,
    literal_answer_in_closed_question_candidates,
    summarize_original_qa,
)


def test_provenance_binds_every_input_and_program(tmp_path, monkeypatch):
    source_names = (
        "train.json",
        "validation.json",
        "test.json",
        "slake_qa_pairs.json",
        "rad_vqa_pairs.json",
        "xray_closed_pairs.json",
        "vqa_train.parquet",
        "vqa_test.parquet",
        "annotation.json",
    )
    for index, name in enumerate(source_names):
        (tmp_path / name).write_text(f"source-{index}")
    args = type(
        "Args",
        (),
        {
            "slake_root": tmp_path,
            "vqa_rad_train": tmp_path / "vqa_train.parquet",
            "vqa_rad_test": tmp_path / "vqa_test.parquet",
            "medheval_fine_root": tmp_path,
            "iuxray_annotation": tmp_path / "annotation.json",
        },
    )()
    monkeypatch.setattr("sys.argv", ["audit.py", "--output", "result.json"])
    provenance = audit_provenance(args)
    assert provenance["method"] == "cross-answer-space-substrate-audit-v1"
    assert provenance["seed"] == "not_applicable_deterministic_audit"
    assert len(provenance["sources"]) == 9
    assert all(len(row["sha256"]) == 64 for row in provenance["sources"].values())
    assert len(provenance["evaluator"]["sha256"]) == 64
    assert len(provenance["fingerprint"]) == 64


def test_same_image_cross_product_is_only_an_upper_bound():
    rows = [
        {"image_id": "a", "question": "What pathology?", "answer": "edema"},
        {"image_id": "a", "question": "Is edema present?", "answer": "yes"},
        {"image_id": "a", "question": "Is fracture present?", "answer": "no"},
        {"image_id": "b", "question": "What pathology?", "answer": "mass"},
    ]
    summary = summarize_original_qa(rows)
    assert summary["open_rows"] == 2
    assert summary["closed_rows"] == 2
    assert summary["images_with_both_answer_spaces"] == 1
    assert summary["same_image_cross_product_upper_bound"] == 2
    assert summary["same_image_exact_question_cross_space_pairs"] == 0


def test_literal_candidate_requires_entire_answer_phrase():
    rows = [
        {"image_id": "a", "question": "What pathology?", "answer": "pleural effusion"},
        {"image_id": "a", "question": "Is pleural effusion present?", "answer": "yes"},
        {"image_id": "a", "question": "Is effusion present?", "answer": "yes"},
        {"image_id": "b", "question": "What organ?", "answer": "lung"},
        {"image_id": "b", "question": "Are the lungs clear?", "answer": "no"},
    ]
    candidates = literal_answer_in_closed_question_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0]["literal_answer"] == "pleural effusion"
    assert candidates[0]["status"] == "automatic_candidate_only"


def test_declared_slake_answer_space_overrides_answer_string():
    rows = [
        {
            "img_name": "x/source.jpg",
            "question": "Which is abnormal, lung or heart?",
            "answer": "heart",
            "answer_type": "CLOSED",
        },
        {
            "img_name": "x/source.jpg",
            "question": "What disease?",
            "answer": "cardiomegaly",
            "answer_type": "OPEN",
        },
    ]
    summary = summarize_original_qa(rows)
    assert summary["closed_rows"] == 1
    assert summary["open_rows"] == 1
    assert summary["same_image_cross_product_upper_bound"] == 1


def test_no_dual_review_manifest_fails_closed():
    gates = gate_summary()
    assert gates["observed_formal_dual_reviewed_pairs"] == 0
    assert gates["f6"]["verdict"] == "KILL"
    assert gates["f7"]["verdict"] == "KILL"
    assert gates["overall"] == "NO_GO"
