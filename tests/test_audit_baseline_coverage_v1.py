import json
from pathlib import Path

import pytest

from anchor.medeval.audit_baseline_coverage_v1 import audit
from anchor.medeval.hashing import sha256_file


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    config = _write(
        tmp_path / "config.json",
        {
            "methods": [
                {
                    "name": "greedy",
                    "family": "control",
                    "tracks": ["common_protocol"],
                    "tasks": ["oe_vqa"],
                    "cutoff": "retain",
                },
                {
                    "name": "blocked",
                    "family": "decode",
                    "tracks": ["paper_native"],
                    "tasks": ["oe_vqa"],
                    "cutoff": "missing license",
                },
            ]
        },
    )
    registry = tmp_path / "registry.jsonl"
    registry.write_text("{}\n")
    t0 = _write(
        tmp_path / "t0.json",
        {
            "config_sha256": sha256_file(config),
            "methods": [
                {"name": "greedy"},
                {"name": "blocked"},
            ],
        },
    )
    evidence = _write(
        tmp_path / "evidence.json",
        {
            "t0_audit_sha256": sha256_file(t0),
            "artifact_registry_sha256": sha256_file(registry),
            "summary": {"stale_registry_events": 0},
            "methods": [
                {
                    "name": "greedy",
                    "stages": {
                        "T0": {"status": "pass", "evidence": []},
                        "T1": {"status": "pass", "evidence": []},
                        "T2": {"status": "pass", "evidence": []},
                        "T3": {
                            "status": "pass",
                            "evidence": [
                                {
                                    "evidence_scope": "qualified raw OE generation; vqa-rad; hulu; greedy256; clinical claim evaluation pending",
                                    "artifact_sha256": "a",
                                },
                                {
                                    "evidence_scope": "qualified raw OE generation; vqa-rad; llava; greedy256; clinical claim evaluation pending",
                                    "artifact_sha256": "b",
                                },
                            ],
                        },
                        "full": {"status": "reference_only", "evidence": []},
                    },
                },
                {
                    "name": "blocked",
                    "stages": {
                        stage: {"status": "not_admissible", "evidence": []}
                        for stage in ("T0", "T1", "T2", "T3", "full")
                    },
                },
            ],
        },
    )
    native = _write(
        tmp_path / "native.json",
        {"passed": True, "models": [{"model": "hulu"}, {"model": "llava"}]},
    )
    rag = _write(tmp_path / "rag.json", {"supported": []})
    report = _write(
        tmp_path / "report.json",
        {
            "n_rows": 10,
            "admissible_for_report_generation_claim": False,
            "invalid_reasons": ["missing controls"],
        },
    )
    return {
        "config": config,
        "registry": registry,
        "t0": t0,
        "evidence": evidence,
        "native": native,
        "rag": rag,
        "report": report,
    }


def test_audit_distinguishes_source_coverage_from_efficacy(tmp_path: Path) -> None:
    p = _fixture(tmp_path)
    result = audit(
        config_path=p["config"],
        t0_path=p["t0"],
        evidence_path=p["evidence"],
        registry_path=p["registry"],
        native_acceptance_path=p["native"],
        rag_causal_path=p["rag"],
        report_audits=[("model", p["report"])],
    )
    assert result["gates"]["configuration_closure"] is True
    assert result["gates"]["source_qualification_complete"] is True
    assert result["gates"]["clinical_claim_evaluation_complete"] is False
    assert result["gates"]["report_generation_controls_passed"] is False
    assert result["paper_baseline_claim_authorized"] is False
    assert result["summary"]["t0_not_admissible"] == ["blocked"]


def test_audit_rejects_stale_configuration_binding(tmp_path: Path) -> None:
    p = _fixture(tmp_path)
    payload = json.loads(p["config"].read_text())
    payload["methods"][0]["cutoff"] = "changed"
    p["config"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="current method configuration"):
        audit(
            config_path=p["config"],
            t0_path=p["t0"],
            evidence_path=p["evidence"],
            registry_path=p["registry"],
            native_acceptance_path=p["native"],
            rag_causal_path=p["rag"],
            report_audits=[],
        )


def test_audit_rejects_stale_registry_event_count(tmp_path: Path) -> None:
    p = _fixture(tmp_path)
    payload = json.loads(p["evidence"].read_text())
    payload["summary"]["stale_registry_events"] = 1
    p["evidence"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bind the current T0 audit|stale registry"):
        audit(
            config_path=p["config"],
            t0_path=p["t0"],
            evidence_path=p["evidence"],
            registry_path=p["registry"],
            native_acceptance_path=p["native"],
            rag_causal_path=p["rag"],
            report_audits=[],
        )
