import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.smoke_physician_oe_review_form import (
    extract_regular_files,
    immutable_projection,
    synthetic_completed_rows,
)


def _seed():
    return [
        {
            "bundle_id": "bundle",
            "group_id": "group",
            "review_order": 0,
            "review_phase": "calibration",
            "reviewer_slot": "A",
            "image": {"relative_path": "hash.jpg", "sha256": "hash"},
            "question": "What is shown?",
            "benchmark_reference": "A finding",
            "reference_annotation": {
                "visual_observability": None,
                "benchmark_reference_correctness": None,
                "required_answer_claims": [],
                "notes": "",
            },
            "candidate_answers": [
                {
                    "answer_id": "answer",
                    "answer_text": "A finding",
                    "annotation": {
                        "direct_answer_correctness": None,
                        "direct_answer_state": None,
                        "atomic_claims": [],
                        "no_clinical_claims": None,
                        "omitted_required_claim_ids": [],
                        "overall_clinically_harmful": None,
                        "reviewer_confidence": None,
                        "rationale": "",
                    },
                }
            ],
        }
    ]


def test_synthetic_completion_preserves_immutable_content_without_mutating_seed():
    seed = _seed()
    before = json.loads(json.dumps(seed))
    completed = synthetic_completed_rows(seed)
    assert seed == before
    assert immutable_projection(completed) == immutable_projection(seed)
    annotation = completed[0]["candidate_answers"][0]["annotation"]
    assert annotation["no_clinical_claims"] is True
    assert annotation["atomic_claims"] == []
    assert annotation["reviewer_confidence"] == 3


def test_safe_extraction_accepts_regular_single_root_and_rejects_traversal(tmp_path: Path):
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as archive:
        payload = b"ok"
        info = tarfile.TarInfo("root/file.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    root = extract_regular_files(good, tmp_path / "good-out")
    assert (root / "file.txt").read_bytes() == b"ok"

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extract_regular_files(bad, tmp_path / "bad-out")
