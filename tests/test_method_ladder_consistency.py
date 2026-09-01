from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "configs/unified_eval/method_ladder_v1.json"
RUNTIME = ROOT / "configs/unified_eval/methods/common_baselines.yaml"
T0_AUDIT = ROOT / "results_reference/baseline_t0_source_audit_20260802.json"


def test_runtime_registry_delegates_all_qualification_to_one_authority():
    ladder = json.loads(LADDER.read_text())
    runtime = yaml.safe_load(RUNTIME.read_text())
    assert runtime["qualification_authority"] == str(LADDER.relative_to(ROOT))
    assert ladder["qualification_authority"] is True

    methods = ladder["methods"]
    names = [row["name"] for row in methods]
    assert len(names) == len(set(names))
    known = set(names)
    for row in runtime["methods"]:
        assert "admissibility" not in row
        assert row["qualification_key"] in known


def test_t2_claims_are_bound_to_the_existing_canonical_audit():
    ladder = json.loads(LADDER.read_text())
    by_name = {row["name"]: row for row in ladder["methods"]}
    evidence = (
        ROOT
        / "corrected_runs/unified_eval/smoke/"
        "vqa_rad_oe_mitigation_v5_t2_256/t2_audit.json"
    )
    assert evidence.is_file()
    audit = json.loads(evidence.read_text())
    assert audit["passed"] is True
    assert audit["T3_authorized_methods"] == []
    audited = {row["method"].lower() for row in audit["methods"]}
    for name in ("VCD", "OPERA", "PAI", "AvisC"):
        row = by_name[name]
        assert (ROOT / row["qualification_evidence"]).resolve() == evidence.resolve()
        assert name.lower() in audited
        assert "T3 remains unauthorized" in row["cutoff"]


def test_t0_license_decisions_are_fail_closed():
    ladder = json.loads(LADDER.read_text())
    audit = json.loads(T0_AUDIT.read_text())
    by_name = {row["name"]: row for row in ladder["methods"]}
    fingerprint_payload = {
        "protocol": audit["protocol"],
        "repositories": {
            name: {
                "head": row["official_head"],
                "root_license_sha256": row["root_license_sha256"],
                "entry_sha256": row["official_entry_sha256"],
            }
            for name, row in audit["repositories"].items()
        },
    }
    observed_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert audit["fingerprint"] == observed_fingerprint
    assert (ROOT / by_name["VISTA"]["license"]).is_file()
    assert by_name["VHR"]["license"] == "Apache-2.0 upstream root"
    assert audit["repositories"]["VISTA"]["t0"] == "pass"
    assert audit["repositories"]["VHR"]["t0"] == "pass"
    vista_evidence = ROOT / by_name["VISTA"]["qualification_evidence"]
    assert vista_evidence.is_file()
    vista_audit = json.loads(vista_evidence.read_text())
    assert vista_audit["t1"]["generated_token_exact_rate"] == 1.0
    assert vista_audit["t2"]["changed_generated_sequences"] > 0
    assert vista_audit["clinical_efficacy_claim"] is False
    assert vista_audit["t3_authorized"] is False
    assert "T3 remains unauthorized" in by_name["VISTA"]["cutoff"]
    for name in ("AGLA", "ClearSight"):
        assert by_name[name]["license"] is None
        assert by_name[name]["cutoff"].startswith("not_admissible")
        assert audit["repositories"][name]["t0"] == "not_admissible"
    medvr = audit["repositories"]["MedVR"]
    assert by_name["MedVR"]["tracks"] == ["paper_native"]
    assert by_name["MedVR"]["checkpoint"] is None
    assert by_name["MedVR"]["implementation"] is None
    assert by_name["MedVR"]["cutoff"].startswith("not_admissible")
    assert medvr["root_license"] == "Apache-2.0"
    assert medvr["released_checkpoint"] is None
    assert medvr["evaluation_entry_point"] is None
    assert medvr["t0"] == "not_admissible"
