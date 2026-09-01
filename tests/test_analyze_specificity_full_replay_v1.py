from __future__ import annotations

import copy

import pytest

from anchor.corrected_sgta.analyze_specificity_full_replay_v1 import (
    AnalysisError,
    CONTROL,
    ERROR,
    analyze_test,
    freeze_dev_spec,
)


def _row(case: str, role: str, own, swap, visual, text):
    return {
        "status": "ok",
        "sample_id": case,
        "case_id": case,
        "scientific_role": role,
        "edge_type": "laterality",
        "modality_stratum": "XR",
        "anatomy_stratum": "thorax",
        "prompt_requested_increment": False,
        "signals": {
            "layer_ids": ["decoder.8", "decoder.32"],
            "token_counts": {"full_visible_answer": 20, "constraint": 1},
            "own_image": {"constraint_minus_matched": own},
            "swap_images": {"mean_constraint_minus_matched": swap},
            "primary_own_minus_swap_difference_in_differences": visual,
            "text_only_secondary": {"constraint_minus_matched": text},
        },
    }


def _rows(prefix: str):
    rows = []
    for index in range(12):
        jitter = 0.01 * (index - 5.5)
        rows.append(_row(f"{prefix}-C-{index}", CONTROL, [0.5 + jitter, 0.55 + jitter], [0.1, 0.15], [0.4 + jitter, 0.4 - jitter], [0.0, 0.05]))
        rows.append(_row(f"{prefix}-E-{index}", ERROR, [0.0 + jitter, 0.8 + jitter], [0.0, 0.8], [0.0 + jitter, 0.0 + jitter], [0.0, 0.8]))
    return rows


def _config():
    return {
        "manifest_sha256": "manifest",
        "metadata_sha256": "metadata",
        "identity_canary_sha256": "identity",
        "adapter_fingerprint": {"adapter": "fake"},
    }


def test_conjunctive_mechanism_gate_passes_only_separating_signature():
    spec = freeze_dev_spec(_rows("dev"), _config())
    result = analyze_test(_rows("test"), _config(), spec)
    assert result["all_primary_gates_pass"] is True
    assert result["status"] == "mechanism_gate_passed"


def test_visual_late_shift_kills_image_independent_ratchet_claim():
    spec = freeze_dev_spec(_rows("dev"), _config())
    rows = _rows("test")
    for row in rows:
        if row["scientific_role"] == ERROR:
            row["signals"]["primary_own_minus_swap_difference_in_differences"] = [0.0, 0.5]
    result = analyze_test(rows, _config(), spec)
    assert result["gates"]["no_late_visual_residual_equivalent"] is False
    assert result["all_primary_gates_pass"] is False


def test_analysis_refuses_dev_test_case_leakage():
    spec = freeze_dev_spec(_rows("same"), _config())
    with pytest.raises(AnalysisError, match="leakage"):
        analyze_test(copy.deepcopy(_rows("same")), _config(), spec)
