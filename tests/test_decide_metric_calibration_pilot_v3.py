import json
from pathlib import Path

from anchor.corrected_sgta.decide_metric_calibration_pilot_v3 import decide


def _analysis(path: Path, *, direct_runtime: bool, direct_rate: float) -> Path:
    value = {
        "runtime_admissible": True,
        "direct": {
            "runtime_admissible": direct_runtime,
            "unidentifiable_unqualified_numeric_unit_rate": direct_rate,
        },
        "cells": {
            "vision_coordinate:missing": {"patient_type_overcommitment_rate": 0.25},
            "vision_coordinate:header_unknown": {"patient_type_overcommitment_rate": 0.25},
            "vision_coordinate:detector_only": {"patient_type_overcommitment_rate": 0.0},
        },
        "transformation": {
            "median_endpoint_rms_drift": {"oracle_coordinate": 0.0},
            "median_log_value_vs_log_scale_slope": {"oracle_coordinate": 1.0},
        },
    }
    path.write_text(json.dumps(value))
    return path


def test_direct_runtime_is_part_of_joint_gate(tmp_path: Path) -> None:
    qwen = _analysis(tmp_path / "qwen.json", direct_runtime=False, direct_rate=0.25)
    huatuo = _analysis(tmp_path / "huatuo.json", direct_runtime=True, direct_rate=0.25)
    result = decide(qwen, huatuo)
    assert result["joint_gates"]["runtime_all_answer_contracts"] is False
    assert result["decision"] == "STOP_AFTER_N8"
    assert result["n97_authorized"] is False


def test_all_gates_are_required_for_diagnostic_expansion(tmp_path: Path) -> None:
    qwen = _analysis(tmp_path / "qwen.json", direct_runtime=True, direct_rate=0.25)
    huatuo = _analysis(tmp_path / "huatuo.json", direct_runtime=True, direct_rate=0.25)
    result = decide(qwen, huatuo)
    assert result["decision"] == "EXPAND_TO_N97_DIAGNOSTIC_ONLY"
    assert result["oral_mainline_authorized"] is False
