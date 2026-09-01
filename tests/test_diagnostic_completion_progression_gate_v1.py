import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = (
    ROOT
    / "corrected_runs"
    / "specificity_ratchet"
    / "diagnostic_completion_progression_gate_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_progression_gate_binds_failed_source_and_refuses_full_union():
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    assert gate["decision"] == "no_go_full_union_generation"
    assert gate["paper_evidence_status"] == "not_admissible_for_progression"
    assert gate["full_parent_union_generation_authorized"] is False
    assert gate["confirmatory_hidden_state_replay_authorized"] is False
    assert gate["gpu_job_authorized_now"] is False

    qualification_ref = gate["source_generation"]
    qualification_path = ROOT / qualification_ref["qualification_path"]
    assert _sha256(qualification_path) == qualification_ref["qualification_sha256"]
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    assert qualification["passed"] is False
    assert qualification["human_claim_audit_authorized"] is False
    assert qualification["second_model_generation_authorized_from_this_model"] is False


def test_every_substrate_is_hash_bound_and_locally_refuses_replay():
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    assert {row["condition"] for row in gate["substrate_audits"]} == {
        "neutral",
        "existential",
        "negative_obligation",
    }
    for reference in gate["substrate_audits"]:
        path = ROOT / reference["path"]
        assert _sha256(path) == reference["sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["gates"]["confirmatory_hidden_state_replay_authorized"] is False
        assert payload["strict_transition_events"] == reference["strict_transition_events"]


def test_only_bounded_natural_oe_pilot_survives():
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    next_stage = gate["allowed_next_stage"]
    assert next_stage["name"] == "natural_oe_bounded_construct_pilot"
    assert next_stage["maximum_images"] <= 128
    assert next_stage["full_union_forbidden"] is True
    assert len(next_stage["requirements_before_launch"]) >= 5
