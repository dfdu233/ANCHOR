import json
from pathlib import Path

from anchor.medeval.build_physician_oe_prereg_v1 import build


def test_prereg_binds_blinded_inputs_and_dynamic_baseline(tmp_path: Path) -> None:
    for relative in (
        "anchor/medeval/prepare_physician_oe_adjudication.py",
        "anchor/medeval/finalize_physician_oe_consensus.py",
        "anchor/medeval/analyze_physician_oe_multiarm.py",
        "anchor/medeval/analyze_physician_oe_multiarm_v2.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    template, mapping, delivery = (tmp_path / name for name in ("t", "m", "d"))
    for path in (template, mapping, delivery):
        path.write_text(path.name)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "baseline": "greedy256",
        "candidate_arms": ["sc"],
        "statistics": {"cluster_unit": "image", "bootstrap_iterations": 100, "bootstrap_seed": 7, "multiplicity": "Holm"},
        "primary_endpoint": "error",
        "no_exchange_gates": {"omissions_counted": True},
        "machine_gate_spec": {"primary_ci_high_below": 0.0},
    }))
    result = build(template=template, mapping=mapping, delivery=delivery, contract=contract, baseline="greedy256", candidates=["sc"], root=tmp_path)
    assert result["baseline"] == "greedy256"
    assert result["candidate_methods"] == ["sc"]
    assert result["clinical_labels_inspected"] is False
    assert result["analysis_module"].endswith("_v2")


def test_prereg_rejects_cli_method_drift_from_contract(tmp_path: Path) -> None:
    for relative in (
        "anchor/medeval/prepare_physician_oe_adjudication.py",
        "anchor/medeval/finalize_physician_oe_consensus.py",
        "anchor/medeval/analyze_physician_oe_multiarm.py",
        "anchor/medeval/analyze_physician_oe_multiarm_v2.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    template, mapping, delivery = (tmp_path / name for name in ("t", "m", "d"))
    for path in (template, mapping, delivery):
        path.write_text(path.name)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "baseline": "greedy512",
        "candidate_arms": ["sc"],
        "statistics": {"cluster_unit": "image", "bootstrap_iterations": 100, "bootstrap_seed": 7, "multiplicity": "Holm"},
        "primary_endpoint": "error",
        "no_exchange_gates": {"omissions_counted": True},
        "machine_gate_spec": {"primary_ci_high_below": 0.0},
    }))
    import pytest
    with pytest.raises(ValueError, match="baseline"):
        build(template=template, mapping=mapping, delivery=delivery, contract=contract, baseline="greedy256", candidates=["sc"], root=tmp_path)
