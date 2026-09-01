from __future__ import annotations

import json
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
from PIL import Image

from anchor.corrected_sgta.run_target_blind_canary_v1 import (
    _forbidden_paths,
    build_output_record,
    load_input_image,
    load_strict_resume,
    load_target_blind_manifest,
    preflight_inputs,
)


def _write_manifest(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_manifest_is_target_blind_and_qids_are_unique(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [{"qid": "a", "question": "What is present?", "img_name": "a.jpg"}],
    )
    assert load_target_blind_manifest(manifest)[0]["qid"] == "a"

    _write_manifest(
        manifest,
        [{"qid": "a", "question": "Q", "img_name": "a.jpg", "metadata": {"answer": "x"}}],
    )
    with pytest.raises(ValueError, match="target-bearing"):
        load_target_blind_manifest(manifest)

    _write_manifest(
        manifest,
        [
            {"qid": "a", "question": "Q1", "img_name": "a.jpg"},
            {"qid": "a", "question": "Q2", "img_name": "b.jpg"},
        ],
    )
    with pytest.raises(ValueError, match="not unique"):
        load_target_blind_manifest(manifest)


def test_cpu_jpeg_preflight_decodes_inputs(tmp_path) -> None:
    Image.new("RGB", (8, 6), color=(100, 120, 140)).save(tmp_path / "a.jpg")
    rows = [{"qid": "a", "question": "What is present?", "img_name": "a.jpg"}]
    assert preflight_inputs(rows, tmp_path) == {
        "n": 1,
        "image_types": {"raster": 1, "dicom": 0},
        "target_fields_present": 0,
    }


def test_dicom_path_uses_existing_standard_renderer(tmp_path, monkeypatch) -> None:
    path = tmp_path / "study.dicom"
    path.write_bytes(b"renderer stub input")
    renderer = ModuleType("anchor.corrected_sgta.run_huatuo_vindr_commitment_probe")
    calls = []

    def render(received):
        calls.append(received)
        return Image.new("RGB", (7, 5))

    renderer.dicom_to_pil = render
    monkeypatch.setitem(
        sys.modules,
        "anchor.corrected_sgta.run_huatuo_vindr_commitment_probe",
        renderer,
    )
    image = load_input_image(path)
    assert image.size == (7, 5)
    assert calls == [path]


def test_output_record_never_contains_target_fields() -> None:
    result = SimpleNamespace(text="No acute finding.", token_count=3, token_ids=(1, 2, 3), uncertainty=0.2)
    record = build_output_record(
        qid="q1",
        model="huatuo",
        result=result,
        max_new_tokens=32,
        sample_seed=123,
        fingerprint="abc",
    )
    assert not _forbidden_paths(record)
    assert set(record) == {"question_id", "text", "model_id", "metadata"}


def test_resume_requires_exact_prefix_fingerprint_and_no_target(tmp_path) -> None:
    answers = tmp_path / "answers.jsonl"
    good = {
        "question_id": "a",
        "text": "text",
        "model_id": "hulu",
        "metadata": {"fingerprint": "fp"},
    }
    answers.write_text(json.dumps(good) + "\n", encoding="utf-8")
    assert len(load_strict_resume(answers, ["a", "b"], "fp", "hulu")) == 1

    answers.write_text(json.dumps({**good, "gt_ans": "yes"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leaked"):
        load_strict_resume(answers, ["a", "b"], "fp", "hulu")

    answers.write_text(json.dumps(good) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint drift"):
        load_strict_resume(answers, ["a", "b"], "different", "hulu")
