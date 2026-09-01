import json
from pathlib import Path

from anchor.medeval.artifact_registry import append_qualification, qualification_for
from anchor.medeval.audit_method_evidence_ladder import audit


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_fail_closed_ladder_only_promotes_registered_exact_scopes(tmp_path: Path):
    t0 = {
        "methods": [
            {"name": "greedy", "family": "control", "tracks": ["common_protocol"], "tasks": ["oe_vqa"], "cutoff": "retain", "t0_status": "pass", "t0_reasons": []},
            {"name": "beam", "family": "control", "tracks": ["common_protocol"], "tasks": ["oe_vqa"], "cutoff": "matched", "t0_status": "pass", "t0_reasons": []},
            {"name": "temperature_length_controls", "family": "control", "tracks": ["common_protocol"], "tasks": ["oe_vqa"], "cutoff": "matched", "t0_status": "pass", "t0_reasons": []},
            {"name": "calibrated_abstention", "family": "control", "tracks": ["common_protocol"], "tasks": ["oe_vqa"], "cutoff": "selective", "t0_status": "pass", "t0_reasons": []},
            {"name": "shared_medical_rag", "family": "rag", "tracks": ["common_protocol"], "tasks": ["ce_generation"], "cutoff": "causal", "t0_status": "pass", "t0_reasons": []},
            {"name": "VCD", "family": "decode", "tracks": ["common_protocol"], "tasks": ["oe_vqa"], "cutoff": "T2", "t0_status": "pass", "t0_reasons": []},
            {"name": "VISTA", "family": "decode", "tracks": ["common_protocol"], "tasks": ["oe_vqa"], "cutoff": "T2", "t0_status": "pass", "t0_reasons": []},
            {"name": "DoLa", "family": "decode", "tracks": ["common_protocol"], "tasks": ["oe_vqa"], "cutoff": "license", "t0_status": "not_admissible", "t0_reasons": ["license_missing"]},
        ]
    }
    t0_path = _write(tmp_path / "t0.json", json.dumps(t0))
    registry = tmp_path / "registry.jsonl"
    for arm in ("no_context", "rag"):
        for stage in ("T2_n32", "T3_n200"):
            artifact = _write(tmp_path / arm / stage / "answers.jsonl")
            event = qualification_for(
                artifact,
                status="admissible",
                evaluator_version="test",
                evidence_scope=f"common_protocol visual CE-G; mimic; hulu; {arm}; {stage}",
                reason="test",
            )
            append_qualification(registry, event)
    # A suggestive filename with a foreign scope must not promote DoLa.
    fake = _write(tmp_path / "DoLa" / "answers.jsonl")
    append_qualification(
        registry,
        qualification_for(
            fake,
            status="admissible",
            evaluator_version="test",
            evidence_scope="historical diagnostic",
            reason="test",
        ),
    )
    vista = _write(
        tmp_path / "vista" / "t2_ablation_audit.json",
        json.dumps(
            {
                "version": "vista-llava-med-t2-256-ablation-audit-v1",
                "method_off": {
                    "passed": True,
                    "generated_token_exact_rate": 1.0,
                    "answer_files_byte_identical": True,
                },
                "t1": {
                    "status": "passed",
                    "generated_token_exact_rate": 1.0,
                },
                "t2": {
                    "status": "passed_functional_activation_only",
                    "changed_generated_sequences": 3,
                },
                "clinical_efficacy_claim": False,
                "t3_authorized": False,
            }
        ),
    )
    append_qualification(
        registry,
        qualification_for(
            vista,
            status="admissible",
            evaluator_version="vista-t2-v1",
            evidence_scope=(
                "canonical OE-VQA mitigation smoke; vqa-rad; llava; "
                "VISTA; T2_n32"
            ),
            reason="test",
        ),
    )
    beam = _write(tmp_path / "beam" / "answers.jsonl")
    append_qualification(
        registry,
        qualification_for(
            beam,
            status="admissible",
            evaluator_version="test",
            evidence_scope="canonical OE-VQA functional smoke; vqa-rad; hulu; beam4_256; T2_n32",
            reason="test",
        ),
    )
    vcd = _write(tmp_path / "vcd" / "answers.jsonl")
    append_qualification(
        registry,
        qualification_for(
            vcd,
            status="admissible",
            evaluator_version="test",
            evidence_scope="canonical OE-VQA mitigation smoke; vqa-rad; llava; VCD; T2_n32",
            reason="test",
        ),
    )
    internal_temperature = _write(tmp_path / "internal" / "temperature.json")
    append_qualification(
        registry,
        qualification_for(
            internal_temperature,
            status="admissible",
            evaluator_version="internal-test",
            evidence_scope=(
                "internal control qualification; temperature_length_controls; T2"
            ),
            reason="test",
        ),
    )
    internal_abstention = _write(tmp_path / "internal" / "abstention.json")
    append_qualification(
        registry,
        qualification_for(
            internal_abstention,
            status="failed_cutoff",
            evaluator_version="internal-test",
            evidence_scope=(
                "internal control qualification; calibrated_abstention; T2"
            ),
            reason="non-degenerate per-model action gate failed",
        ),
    )
    identity = _write(tmp_path / "identity.json", json.dumps({"protocol": "id", "passed": True, "backends": {"hulu": {}}}))
    mitigation_identity = _write(
        tmp_path / "mitigation_identity.json",
        json.dumps({"protocol": "port-id", "passed": True}),
    )
    causal = _write(tmp_path / "causal.json", json.dumps({"protocol_version": "causal", "supported": []}))

    result = audit(
        t0_audit=t0_path,
        registry=registry,
        identity_gate=identity,
        mitigation_identity_gate=mitigation_identity,
        rag_causal_summary=causal,
    )
    rows = {row["name"]: row for row in result["methods"]}
    assert rows["greedy"]["stages"]["T3"]["status"] == "pass"
    assert rows["beam"]["stages"]["T2"]["status"] == "pass"
    assert rows["beam"]["stages"]["T3"]["status"] == "missing"
    assert rows["temperature_length_controls"]["stages"]["T1"]["status"] == "pass"
    assert rows["temperature_length_controls"]["stages"]["T2"]["status"] == "pass"
    assert rows["calibrated_abstention"]["stages"]["T2"]["status"] == "failed_cutoff"
    assert rows["shared_medical_rag"]["stages"]["T3"]["status"] == "pass"
    assert rows["shared_medical_rag"]["stages"]["full"]["status"] == "failed_cutoff"
    assert rows["VCD"]["stages"]["T1"]["status"] == "pass"
    assert rows["VCD"]["stages"]["T2"]["status"] == "pass"
    assert rows["VCD"]["stages"]["T3"]["status"] == "missing"
    assert rows["VISTA"]["stages"]["T1"]["status"] == "pass"
    assert rows["VISTA"]["stages"]["T2"]["status"] == "pass"
    assert rows["VISTA"]["stages"]["T3"]["status"] == "missing"
    assert rows["DoLa"]["stages"]["T2"]["status"] == "not_admissible"
    assert result["summary"]["full_pass"] == []


def test_hash_drift_makes_registry_event_stale(tmp_path: Path):
    t0 = {"methods": [{"name": "shared_medical_rag", "family": "rag", "tracks": ["common_protocol"], "tasks": ["ce_generation"], "cutoff": "causal", "t0_status": "pass", "t0_reasons": []}]}
    t0_path = _write(tmp_path / "t0.json", json.dumps(t0))
    artifact = _write(tmp_path / "answers.jsonl")
    registry = tmp_path / "registry.jsonl"
    append_qualification(
        registry,
        qualification_for(
            artifact,
            status="admissible",
            evaluator_version="test",
            evidence_scope="common_protocol visual CE-G; mimic; hulu; rag; T2_n32",
            reason="test",
        ),
    )
    artifact.write_text("changed\n")
    result = audit(t0_audit=t0_path, registry=registry)
    assert result["methods"][0]["stages"]["T2"]["status"] == "missing"
    assert result["summary"]["stale_registry_events"] == 1
