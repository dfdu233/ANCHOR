import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.summarize_reader_residual_confirmation_v1 import synthesize


def result(boundary, findings, controls=False):
    return {
        "boundary": boundary,
        "n": 960,
        "representation_controls_pass": controls,
        "finding_wise": {
            finding: {"n": 120, "boundary": value}
            for finding, value in findings.items()
        },
    }


def model(model_id, negative, positive, gate=False):
    return {
        "status": "complete",
        "model_id": model_id,
        "observational_gate_passed": gate,
        "results": {
            "negative_0v1": result(*negative),
            "positive_2v3": result(*positive),
        },
    }


def stored(tmp_path, name, row):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(row))
    return path, row


def test_two_model_strict_majority_gate_and_no_method_authorization(tmp_path):
    findings = {f"f{i}": "Layer-stable" for i in range(4)}
    rows = [
        stored(tmp_path, "a", model("a", ("Layer-stable", findings), ("Layer-stable", findings))),
        stored(tmp_path, "b", model("b", ("Layer-stable", findings), ("Layer-stable", findings))),
    ]
    output = synthesize(rows)
    assert output["paper_gate"]["strict_majority_consistent"] is True
    assert output["paper_gate"]["consistent_cells"] == 8
    assert output["method_authorized"] is False
    assert output["next_action"].startswith("do_not_build_decoder")


def test_observational_early_erasure_only_requests_causal_patch(tmp_path):
    findings = {f"f{i}": "Early erasure" for i in range(4)}
    rows = [
        stored(tmp_path, "a", model("a", ("Early erasure", findings, True), ("Early erasure", findings, True), True)),
        stored(tmp_path, "b", model("b", ("Early erasure", findings, True), ("Early erasure", findings, True), True)),
    ]
    output = synthesize(rows)
    assert output["observational_early_erasure_all_models"] is True
    assert output["method_authorized"] is False
    assert output["next_action"] == "run_preregistered_causal_activation_patch_only"


def test_duplicate_models_are_rejected():
    findings = {"f": "Layer-stable"}
    row = model("same", ("Layer-stable", findings), ("Layer-stable", findings))
    with pytest.raises(ValueError, match="unique"):
        synthesize([(Path("a"), row), (Path("b"), row)])


def test_indeterminate_cells_remain_in_majority_denominator(tmp_path):
    a = {f"f{i}": ("Layer-stable" if i < 2 else "Indeterminate") for i in range(4)}
    rows = [
        stored(tmp_path, "a", model("a", ("Layer-stable", a), ("Layer-stable", a))),
        stored(tmp_path, "b", model("b", ("Layer-stable", a), ("Layer-stable", a))),
    ]
    output = synthesize(rows)
    assert output["paper_gate"]["qualified_cells"] == 8
    assert output["paper_gate"]["consistent_cells"] == 4
    assert output["paper_gate"]["strict_majority_consistent"] is False


def test_pooled_cross_model_disagreement_blocks_boundary_claim(tmp_path):
    findings = {f"f{i}": "Layer-stable" for i in range(4)}
    rows = [
        stored(tmp_path, "a", model("a", ("Early erasure", findings), ("Layer-stable", findings))),
        stored(tmp_path, "b", model("b", ("Late emergence", findings), ("Layer-stable", findings))),
    ]
    output = synthesize(rows)
    assert output["paper_gate"]["consistent_cells"] == 8
    assert output["paper_gate"]["all_pooled_directions_cross_model_consistent"] is False
    assert output["paper_gate"]["strict_majority_consistent"] is False
