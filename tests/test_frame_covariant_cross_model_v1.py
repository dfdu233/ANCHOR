from pathlib import Path

from anchor.corrected_sgta.run_frame_covariant_cross_model_v1 import (
    analyze,
    compile_screen_to_patient,
    erase_frame_words,
    load_orientation_certificates,
    parse_finding_side,
    screen_coordinate_contract,
    swap_laterality,
)


def test_swap_is_involution_and_preserves_non_frame_content():
    text = "Left pleural effusion; RIGHT nodule. No bilateral opacity."
    swapped = swap_laterality(text)
    assert swapped == "Right pleural effusion; LEFT nodule. No bilateral opacity."
    assert swap_laterality(swapped) == text
    assert erase_frame_words(text) == erase_frame_words(swapped)


def test_parse_finding_side():
    text = "There is a nodule or mass in the left upper lung and a right pleural effusion."
    assert parse_finding_side(text, "Nodule/Mass") == "left"
    assert parse_finding_side(text, "Pleural effusion") == "right"
    assert parse_finding_side(text, "Pneumothorax") == "unparsed"


def test_typed_screen_compiler_changes_frame_not_content():
    text = "Screen-right nodule or mass and screen-left pleural effusion."
    compiled = compile_screen_to_patient(text)
    assert compiled == "patient's left nodule or mass and patient's right pleural effusion."
    assert erase_frame_words(text) == erase_frame_words(compiled)
    assert screen_coordinate_contract(text) is True
    assert screen_coordinate_contract("The patient's left lung has a mass.") is False
    assert screen_coordinate_contract("A left lung mass is visible.") is False


def test_analyze_fails_closed_without_enough_rows():
    rows = [
        {
            "status": "ok",
            "case_key": "a",
            "left_finding": "Nodule/Mass",
            "right_finding": "Pleural effusion",
            "named_direct_parse": "swapped",
            "named_screen_contract": True,
            "named_screen_parse": "correct",
            "named_compiled_parse": "correct",
            "named_content_preserved": True,
            "natural_answer": "Right nodule or mass and left pleural effusion.",
            "natural_compiled_answer": "Left nodule or mass and right pleural effusion.",
            "natural_content_preserved": True,
        }
    ]
    result = analyze(rows, draws=100, seed=7)
    assert result["status"] == "NO_GO_FRAME_COMPILATION"
    assert result["named_gate_passed"] is False
    assert result["natural_gate_passed"] is False


def test_natural_analysis_uses_explicit_screen_arm_and_audits_recall():
    rows = [
        {
            "status": "ok",
            "case_key": "a",
            "left_finding": "Nodule/Mass",
            "right_finding": "Pleural effusion",
            "named_direct_parse": "swapped",
            "named_screen_parse": "correct",
            "named_compiled_parse": "correct",
            "named_content_preserved": True,
            "natural_answer": "Left nodule or mass and right pleural effusion.",
            # This diagnostic field is deliberately wrong.  The v2 method
            # must use the explicit screen-coordinate arm below instead.
            "natural_compiled_answer": "Right nodule or mass and left pleural effusion.",
            "natural_content_preserved": True,
            "natural_screen_answer": "Screen-right nodule or mass and screen-left pleural effusion.",
            "natural_screen_contract": True,
            "natural_screen_compiled_answer": "Patient's left nodule or mass and patient's right pleural effusion.",
            "natural_screen_content_preserved": True,
        }
    ]
    result = analyze(rows, draws=100, seed=7)
    natural = result["natural"]
    assert natural["compiled_errors"] == 0
    assert natural["method_target_finding_recall"] == 1.0
    assert natural["native_target_finding_recall"] == 1.0
    assert natural["exact_target_mention_set_rate"] == 1.0


def test_orientation_certificate_is_strict_and_nonempty():
    certificate = load_orientation_certificates(
        Path("configs/frame_covariant_orientation_cert_v1.json")
    )
    assert len(certificate) == 39
    assert certificate["7d0e636b3ef2ccbb0c67b3243a1478ce"] == {
        "marker": "R",
        "marker_screen_side": "left",
    }
