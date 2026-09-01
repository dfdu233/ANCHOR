import json

import pytest

from anchor.medeval.run_native_oe_vqa import load_resume, stable_seed


def test_native_resume_requires_exact_manifest_prefix(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"question_id": "a", "text": "left"}) + "\n")
    assert len(load_resume(path, ["a", "b"])) == 1
    path.write_text(json.dumps({"question_id": "b", "text": "right"}) + "\n")
    with pytest.raises(ValueError):
        load_resume(path, ["a", "b"])


def test_native_seed_is_stable_and_qid_specific():
    assert stable_seed(42, "a") == stable_seed(42, "a")
    assert stable_seed(42, "a") != stable_seed(42, "b")
