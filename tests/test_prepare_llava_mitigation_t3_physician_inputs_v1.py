from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_llava_mitigation_t3_physician_inputs_v1 import prepare


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _fixture(tmp_path: Path, *, leak: bool = False, authorized: bool = True):
    run = tmp_path / "run"
    source = run / "method" / "chunk.answers.jsonl"
    source.parent.mkdir(parents=True)
    rows = [
        {"question_id": qid, "text": f"Finding {qid}."}
        for qid in ("q1", "q2")
    ]
    if leak:
        rows[0]["metadata"] = {"reference": "truth"}
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract = tmp_path / "contract.json"
    _json(contract, {"methods": ["method"]})
    selected = tmp_path / "selected.json"
    _json(selected, [{
        "qid": "q2", "image_sha256": "image-2", "answer": "truth",
    }])
    prereg = tmp_path / "prereg.json"
    prereg_payload = {
        "model_outputs_read": False,
        "execution_contract_sha256": sha256_file(contract),
        "methods": ["method"],
        "selected_manifest_sha256": sha256_file(selected),
        "selected_qids": ["q2"],
        "selected_groups": 1,
    }
    prereg_payload["fingerprint"] = sha256_json(prereg_payload)
    _json(prereg, prereg_payload)
    audit = tmp_path / "audit.json"
    _json(audit, {
        "protocol_version": "llava-mitigation-t3-generation-audit-v1",
        "all_operational_gates_passed": authorized,
        "physician_pack_authorized": authorized,
        "clinical_efficacy_authorized": False,
        "execution_contract_sha256": sha256_file(contract),
        "run_root": str(run.resolve()),
        "method_records": [{
            "method": "method", "eligible": True,
            "answer_paths": [str(source.resolve())],
            "answer_sha256": [sha256_file(source)],
        }],
    })
    return run, audit, contract, prereg, selected


def test_prepare_is_ordered_reference_free_and_hash_bound(tmp_path: Path) -> None:
    run, audit, contract, prereg, selected = _fixture(tmp_path)
    result = prepare(
        run_root=run,
        audit_path=audit,
        execution_contract_path=contract,
        selection_prereg_path=prereg,
        selected_manifest_path=selected,
        output_dir=tmp_path / "out",
    )
    output = Path(result["outputs"]["method"]["selected_output"])
    row = json.loads(output.read_text().strip())
    assert row["question_id"] == "q2"
    assert result["first_authorized_reference_join"] is True
    assert result["clinical_efficacy_authorized"] is False


@pytest.mark.parametrize("leak,authorized", [(True, True), (False, False)])
def test_prepare_fails_closed_on_leak_or_failed_gate(
    tmp_path: Path, leak: bool, authorized: bool
) -> None:
    run, audit, contract, prereg, selected = _fixture(
        tmp_path, leak=leak, authorized=authorized
    )
    with pytest.raises(RuntimeError):
        prepare(
            run_root=run,
            audit_path=audit,
            execution_contract_path=contract,
            selection_prereg_path=prereg,
            selected_manifest_path=selected,
            output_dir=tmp_path / "out",
        )
