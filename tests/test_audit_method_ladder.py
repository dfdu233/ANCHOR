import json

from anchor.medeval.audit_method_ladder import audit


def test_t0_fails_closed_on_missing_official_code(tmp_path):
    implementation = tmp_path / "runner.py"
    implementation.write_text("pass\n")
    config = tmp_path / "methods.json"
    config.write_text(json.dumps({"methods": [{
        "name": "missing",
        "tracks": ["paper_native"],
        "tasks": ["oe_vqa"],
        "official_code_required": True,
        "source": None,
        "license": None,
        "implementation": "runner.py",
    }]}))
    row = audit(config, tmp_path)["methods"][0]
    assert row["t0_status"] == "not_admissible"
    assert "official_source_missing" in row["t0_reasons"]
    assert "license_missing" in row["t0_reasons"]


def test_internal_control_can_pass_without_external_license(tmp_path):
    implementation = tmp_path / "runner.py"
    implementation.write_text("pass\n")
    config = tmp_path / "methods.json"
    config.write_text(json.dumps({"methods": [{
        "name": "greedy",
        "tracks": ["common_protocol"],
        "tasks": ["oe_vqa"],
        "official_code_required": False,
        "implementation": "runner.py",
    }]}))
    assert audit(config, tmp_path)["methods"][0]["t0_status"] == "pass"
