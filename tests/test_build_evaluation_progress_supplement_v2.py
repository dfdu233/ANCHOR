import json

from anchor.medeval.build_evaluation_progress_supplement_v2 import build
from anchor.medeval.hashing import sha256_file


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_supplement_never_promotes_generation_or_rag_availability(tmp_path) -> None:
    base = _write(tmp_path / "base", {"paper_ready": False})
    rag = _write(
        tmp_path / "rag",
        {
            "tracks_kept_separate": True,
            "any_rag_efficacy_authorized": False,
            "common_protocol": {"T3_generation_qualification": "pass", "full_efficacy": "failed_cutoff"},
            "paper_native": [{"method": "RULE", "T0": "not_admissible"}],
        },
    )
    failure = _write(tmp_path / "failure", {"all_eligible": False})
    execution = _write(tmp_path / "execution", {"repair_trigger": {"artifact_sha256": sha256_file(failure)}})
    provenance = _write(tmp_path / "provenance", {"execution_contract_sha256": sha256_file(execution)})
    v1 = _write(tmp_path / "v1", {"status": "running"})
    v2 = _write(tmp_path / "v2", {"status": "running"})
    result = build(
        base_audit=base,
        rag_audit=rag,
        v1_failure=failure,
        v2_execution=execution,
        v2_provenance=provenance,
        v1_job_state=v1,
        v2_job_state=v2,
    )
    assert result["paper_ready"] is False
    assert result["evaluation_state"]["t3_v2_generation_qualified"] is False
    assert result["evaluation_state"]["any_rag_efficacy_authorized"] is False
