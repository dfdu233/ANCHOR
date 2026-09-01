import math
import json

from anchor.corrected_sgta.build_metric_calibration_probe_v1 import (
    box_endpoints,
    box_iou,
    metadata_line,
    nominal_distance_mm,
)
from anchor.corrected_sgta.analyze_metric_calibration_v1 import extract_json
from anchor.corrected_sgta.analyze_metric_calibration_v2 import analyze
from anchor.corrected_sgta.build_metric_calibration_probe_v2 import (
    expected_value,
    structured_prompt_v2,
)


def test_consensus_geometry_uses_normalized_dimensionless_endpoints():
    endpoints = box_endpoints((10, 20, 50, 40), rows=100, columns=100)
    assert endpoints == ((0.1, 0.3), (0.5, 0.3))
    assert math.isclose(nominal_distance_mm(endpoints, 100, 100, (0.2, 0.2)), 8.0)


def test_calibration_states_do_not_promote_detector_or_unknown_spacing():
    certified = metadata_line("certified_x2", (0.2, 0.3))
    detector = metadata_line("detector_only", (0.2, 0.3))
    unknown = metadata_line("header_unknown", (0.2, 0.3))
    missing = metadata_line("missing", (0.2, 0.3))
    assert certified[1:] == ("patient-mm", 2.0, "mm")
    assert detector[1:] == ("detector-mm", None, None)
    assert unknown[1:] == ("pixel-only/unknown", None, None)
    assert missing[1:] == ("pixel-only/unknown", None, None)


def test_unit_reexpression_is_not_a_scale_change():
    millimetres = metadata_line("certified_x1", (0.2, 0.3))
    centimetres = metadata_line("certified_cm", (0.2, 0.3))
    assert millimetres[2] == 1.0 and millimetres[3] == "mm"
    assert centimetres[2] == 0.1 and centimetres[3] == "cm"
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_strict_output_parser_rejects_schema_drift():
    valid, error = extract_json(
        '{"visible":"yes","endpoints_normalized":null,"measurement_type":"pixel-only/unknown","physical_value":null,"unit":null}'
    )
    assert error is None and valid["physical_value"] is None
    invalid, error = extract_json('{"measurement_type":"patient-mm","physical_value":12,"unit":"mm"}')
    assert invalid is None and error == "json_schema_keys"


def test_v2_oracle_exposes_raster_dimensions_and_fixes_cm_reexpression():
    endpoints = ((0.1, 0.2), (0.5, 0.2))
    prompt = structured_prompt_v2("certified", endpoints, rows=100, columns=200)
    assert "200 columns and 100 rows" in prompt
    millimetres, mm_unit = expected_value("certified_x1", endpoints, 100, 200, (0.2, 0.2))
    centimetres, cm_unit = expected_value("certified_cm", endpoints, 100, 200, (0.2, 0.2))
    assert math.isclose(millimetres, 16.0) and mm_unit == "mm"
    assert math.isclose(centimetres, 1.6) and cm_unit == "cm"


def test_v2_runtime_gate_is_structured_and_direct_has_independent_audit(tmp_path):
    structured = {
        "item_id": "image:oracle_coordinate:missing:structured-v2",
        "image_id": "image",
        "arm": "oracle_coordinate",
        "condition": "missing",
        "question_contract": "structured_neutral_v2",
        "raw_text": '{"visible":"yes","endpoints_normalized":null,"measurement_type":"pixel-only/unknown","physical_value":null,"unit":null}',
        "stop_reason": "eos",
        "expected_measurement_type": "pixel-only/unknown",
        "expected_physical_value": None,
        "expected_unit": None,
        "patient_value_identifiable": False,
    }
    direct = {
        **structured,
        "item_id": "image:vision_coordinate:missing:direct-v2",
        "arm": "vision_coordinate",
        "question_contract": "clinical_direct_v2",
        "raw_text": "Calibration is unavailable, so a patient-space value cannot be determined.",
        "stop_reason": "max_new_tokens",
    }
    answers = tmp_path / "answers.jsonl"
    answers.write_text("\n".join(json.dumps(row) for row in (structured, direct)) + "\n")
    result = analyze(answers)
    assert result["runtime_admissible"] is True
    assert result["runtime"]["structured_json_valid_rate"] == 1.0
    assert result["direct"]["runtime_admissible"] is False
    assert result["direct"]["manual_audit_candidates"][0]["disposition"] == "explicit_abstention_without_numeric_unit"
