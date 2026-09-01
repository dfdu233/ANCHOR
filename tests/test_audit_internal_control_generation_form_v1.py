from __future__ import annotations

import json
from pathlib import Path

from anchor.medeval.audit_internal_control_generation_form_v1 import audit
from anchor.medeval.hashing import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _fixture(tmp_path: Path, text: str = "A complete sentence.") -> dict[str, Path]:
    manifest = tmp_path / "manifest.json"
    contract = tmp_path / "contract.json"
    upstream = tmp_path / "generation_audit.json"
    run_root = tmp_path / "run"
    _write_json(manifest, [{"qid": "q1", "question": "Describe the pathology"}])
    _write_json(
        contract,
        {
            "models": ["model"],
            "generation": {"max_new_tokens": 8},
            "oe_qualification": {
                "clinical_raw_arms": ["greedy"],
                "max_cap_rate": 0.05,
                "min_nonempty_rate": 0.95,
                "max_function_only_rate": 0.01,
            },
        },
    )
    _write_json(
        run_root / "model" / "greedy" / "answers.jsonl",
        {
            "question_id": "q1",
            "text": text,
            "gt_ans": "private reference",
            "metadata": {"generated_token_count": 3},
        },
    )
    _write_json(
        upstream,
        {
            "passed": True,
            "manifest_sha256": sha256_file(manifest),
            "execution_contract_sha256": sha256_file(contract),
        },
    )
    return {
        "manifest": manifest,
        "contract": contract,
        "upstream": upstream,
        "run_root": run_root,
    }


def test_hash_bound_form_audit_passes_without_using_co_resident_reference(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = audit(
        run_root=paths["run_root"],
        manifest_path=paths["manifest"],
        execution_contract_path=paths["contract"],
        generation_audit_path=paths["upstream"],
    )
    assert result["passed"]
    assert result["physician_pack_operationally_authorized"]
    assert not result["clinical_efficacy_authorized"]
    assert result["records"][0]["reference_fields_co_resident"] == ["gt_ans"]
    assert not result["records"][0]["reference_fields_accessed_by_auditor"]


def test_sentence_required_generation_without_terminal_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, text="Incomplete answer")
    result = audit(
        run_root=paths["run_root"],
        manifest_path=paths["manifest"],
        execution_contract_path=paths["contract"],
        generation_audit_path=paths["upstream"],
    )
    assert not result["passed"]
    qualification = result["records"][0]["qualification"]
    assert qualification["terminal_required_count"] == 1
    assert qualification["terminal_completeness_rate"] == 0.0


def test_upstream_hash_drift_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["manifest"].write_text(
        json.dumps([{"qid": "q1", "question": "Describe the pathology changed"}]) + "\n"
    )
    result = audit(
        run_root=paths["run_root"],
        manifest_path=paths["manifest"],
        execution_contract_path=paths["contract"],
        generation_audit_path=paths["upstream"],
    )
    assert not result["passed"]
    assert "upstream generation audit manifest hash mismatch" in result["errors"]
