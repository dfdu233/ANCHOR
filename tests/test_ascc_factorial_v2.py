from __future__ import annotations

import numpy as np
import pytest

from anchor.corrected_sgta.analyze_ascc_factorial_v2 import (
    acquire_formal_identity_lock,
    apply_crossfold_affine,
    analyze_layer,
    fixed_overlap_weights,
    gauge_contrasts,
    stratified_level,
    stratified_effect,
    write_once_json,
)
from anchor.corrected_sgta.prepare_ascc_factorial_v2 import (
    MARKERS,
    PROMPTS,
    validate_factorial,
)
from anchor.corrected_sgta.run_huatuo_ascc_factorial_v2 import marker_coordinates


def test_symmetric_marker_semantics_and_factorial_identity() -> None:
    assert MARKERS == (" absent", " uncertain", " present")
    validate_factorial()
    assert {
        (row["speech_act"], row["clinical_noun"]) for row in PROMPTS
    } == {
        (speech_act, noun)
        for speech_act in ("describe", "list")
        for noun in ("findings", "abnormalities")
    }


def test_commitment_is_log_definite_mass_not_geometric_mean() -> None:
    result = marker_coordinates(
        {" absent": 0.0, " uncertain": 4.0, " present": 8.0}
    )
    assert result["commitment"] == pytest.approx(np.logaddexp(8.0, 0.0) - 4.0)
    assert result["commitment"] > 3.9
    assert result["polarity"] == pytest.approx(8.0)
    assert result["uncertainty_preference"] == pytest.approx(-result["commitment"])


def test_overlap_weighted_effect_uses_independent_strata_not_arbitrary_pairs() -> None:
    rows = []
    items = []
    for parent, aspect, offset in ((2, "wide", 0.0), (3, "square", 10.0)):
        for vote, effect in ((0, 0.0), (1, 2.0)):
            for index in range(3 + parent):
                row = {
                    "parent_votes": parent,
                    "aspect_bucket": aspect,
                    "child_votes": vote,
                }
                rows.append(row)
                items.append({"row": row, "metric": offset + effect})
    weights = fixed_overlap_weights(rows, 1, 0)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert stratified_effect(items, "metric", 1, 0, weights) == pytest.approx(2.0)


def test_analyzer_registers_crossfit_residual_as_a_local_gate_input() -> None:
    # Prevent a late KeyError after the expensive bootstrap: noun_did explicitly
    # consumes this local summary key.
    assert '"affine_residual_interaction": "affine_residual_noun_delta"' in (
        __import__("inspect").getsource(analyze_layer)
    )


def test_gauge_contrasts_ignore_common_logit_offsets() -> None:
    baseline = gauge_contrasts([-2.0, 0.5, 3.0])
    shifted = gauge_contrasts([98.0, 100.5, 103.0])
    assert shifted == pytest.approx(baseline)


def test_crossfold_affine_removes_only_common_scale_and_endpoint_bias() -> None:
    items = []
    for index in range(160):
        vote = (0, 3, 1, 2)[index % 4]
        x_negative = -2.0 + 0.025 * index
        x_positive = 1.5 - 0.017 * index
        commitment = float(np.logaddexp(x_negative, x_positive))
        raw_logits = {}
        commitments = {}
        for speech_index, speech_act in enumerate(("describe", "list")):
            source = [x_negative + 0.03 * speech_index, 0.0, x_positive]
            offset = 50.0 + index
            target = [
                1.25 * source[0] - 0.30 + offset,
                offset,
                1.25 * source[2] + 0.20 + offset,
            ]
            raw_logits[(speech_act, "findings")] = source
            raw_logits[(speech_act, "abnormalities")] = target
            commitments[(speech_act, "findings")] = float(
                np.logaddexp(source[0], source[2])
            )
        items.append(
            {
                "row": {
                    "image_id": f"image-{index}",
                    "child_votes": vote,
                    "parent_votes": 2 + index % 2,
                    "aspect_bucket": ("portrait", "wide", "square")[index % 3],
                },
                "raw_logits": raw_logits,
                "commitment": commitments,
            }
        )
    fitted, diagnostics = apply_crossfold_affine(items)
    assert diagnostics["valid"] is True
    assert max(abs(item["affine_residual_noun_delta"]) for item in fitted) < 1e-10
    assert all(
        0.5 <= fit["slope"] <= 2.0 for fit in diagnostics["fits"].values()
    )


def test_overlap_standardized_level_excludes_nonoverlap_stratum() -> None:
    rows = []
    items = []
    for vote, value in ((0, 0.0), (1, 2.0)):
        for index in range(5):
            row = {"parent_votes": 2, "aspect_bucket": "wide", "child_votes": vote}
            rows.append(row)
            items.append({"row": row, "metric": value})
    for index in range(100):
        row = {"parent_votes": 3, "aspect_bucket": "square", "child_votes": 1}
        rows.append(row)
        items.append({"row": row, "metric": 100.0})
    weights = fixed_overlap_weights(rows, 1, 0)
    assert stratified_level(items, "metric", 1, weights) == pytest.approx(2.0)


def test_positive_boundary_polarity_admission_uses_three_minus_two() -> None:
    rows = []
    items = []
    for vote, polarity in ((2, 0.5), (3, 2.0)):
        for index in range(10):
            row = {"parent_votes": 3, "aspect_bucket": "wide", "child_votes": vote}
            rows.append(row)
            items.append({"row": row, "polarity": polarity})
    weights = fixed_overlap_weights(rows, 2, 3)
    assert stratified_effect(items, "polarity", 3, 2, weights) == pytest.approx(1.5)


def test_write_once_json_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "analysis.json"
    write_once_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_once_json(path, {"value": 2})
    assert path.read_text().strip().endswith("}")


def test_formal_identity_lock_binds_canonical_output_and_fingerprint(tmp_path) -> None:
    output = tmp_path / "primary_analysis_v2_1_blind_locked.json"
    lock_path, first = acquire_formal_identity_lock(
        tmp_path,
        output,
        "lung_opacity_to_pneumonia",
        "primary",
        "score-a",
    )
    repeated_path, repeated = acquire_formal_identity_lock(
        tmp_path,
        output,
        "lung_opacity_to_pneumonia",
        "primary",
        "score-a",
    )
    assert repeated_path == lock_path
    assert repeated == first
    with pytest.raises(ValueError):
        acquire_formal_identity_lock(
            tmp_path,
            output,
            "lung_opacity_to_pneumonia",
            "primary",
            "score-b",
        )
    with pytest.raises(ValueError):
        acquire_formal_identity_lock(
            tmp_path,
            tmp_path / "alternate.json",
            "lung_opacity_to_pneumonia",
            "primary",
            "score-a",
        )


def test_affine_quality_cannot_cancel_opposite_clear_bin_biases() -> None:
    items = []
    for index in range(160):
        vote = (0, 3, 1, 2)[index % 4]
        x_negative = -1.5 + 0.02 * index
        x_positive = 1.0 - 0.01 * index
        raw_logits = {}
        commitments = {}
        clear_shift = 0.35 if vote == 0 else -0.35 if vote == 3 else 0.0
        for speech_act in ("describe", "list"):
            source = [x_negative, 0.0, x_positive]
            target = [
                source[0] + clear_shift,
                0.0,
                source[2] + clear_shift,
            ]
            raw_logits[(speech_act, "findings")] = source
            raw_logits[(speech_act, "abnormalities")] = target
            commitments[(speech_act, "findings")] = float(
                np.logaddexp(source[0], source[2])
            )
        items.append(
            {
                "row": {
                    "image_id": f"biased-{index}",
                    "child_votes": vote,
                    "parent_votes": 2 + index % 2,
                    "aspect_bucket": ("portrait", "wide", "square")[index % 3],
                },
                "raw_logits": raw_logits,
                "commitment": commitments,
            }
        )
    _, diagnostics = apply_crossfold_affine(items)
    assert diagnostics["quality_valid"] is False
    assert any(
        abs(value) > 0.1
        for value in diagnostics["heldout_clear_commitment_bias"].values()
    )
