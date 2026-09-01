import json
from pathlib import Path

import numpy as np
import pytest

from anchor.corrected_sgta.analyze_reader_threshold_aliasing_v1 import (
    AliasingError,
    MODELS,
    PRIMARY_FINDINGS,
    confirm,
    design_matrix,
    feature_names,
    fit_dev,
    fit_standardization,
    group_folds,
    load_rows,
)
from anchor.corrected_sgta.validate_reader_threshold_aliasing_preflight_v1 import (
    PreflightError,
    sha256_file,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/reader_threshold_aliasing_preflight_v1.json"


def _synthetic(stage: str, seed: int, replicates: int = 4):
    rng = np.random.default_rng(seed)
    rows = []
    patterns = [f"{value:03b}" for value in range(8)]
    for model_index, model in enumerate(MODELS):
        for finding_index, finding in enumerate(PRIMARY_FINDINGS):
            for pattern_index, pattern in enumerate(patterns):
                votes = tuple(int(value) for value in pattern)
                for replicate in range(replicates):
                    margin = float(rng.normal())
                    logit = (
                        -0.4
                        + 0.30 * margin
                        + 1.5 * votes[2]
                        + 0.5 * votes[1]
                        - 0.7 * votes[0]
                        + 0.05 * model_index
                        + 0.01 * finding_index
                    )
                    probability = 1 / (1 + np.exp(-logit))
                    target = int(rng.random() < probability)
                    image = f"{stage}-image-{model}-{finding}-{pattern}-{replicate}"
                    rows.append(
                        {
                            "record_key": f"{image}:{finding}:{model}",
                            "image_id": image,
                            "finding": finding,
                            "model": model,
                            "stage": stage,
                            "endpoint": "positive_commitment",
                            "votes": votes,
                            "pattern": pattern,
                            "positive_votes": sum(votes),
                            "clean_margin": margin,
                            "target": target,
                        }
                    )
    return rows


def _json_row(stage="dev_fit"):
    return {
        "record_key": "record-1",
        "image_id": "image-1",
        "finding": PRIMARY_FINDINGS[0],
        "model": "huatuo",
        "stage": stage,
        "task": "ce",
        "condition": "clean",
        "reader_votes": {"R10": 1, "R8": 0, "R9": 1},
        "positive_votes": 2,
        "clean_margin": 0.3,
        "endpoint": "positive_commitment",
        "target": 1,
    }


def test_loader_uses_named_fixed_panel_and_fails_on_vote_mismatch(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_json_row()) + "\n", encoding="utf-8")
    rows = load_rows(
        path, "dev_fit", findings=[PRIMARY_FINDINGS[0]], models=["huatuo"],
        require_complete_cells=False,
    )
    assert rows[0]["votes"] == (0, 1, 1)
    assert rows[0]["pattern"] == "011"
    bad = _json_row()
    bad["positive_votes"] = 3
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(AliasingError, match="vote count"):
        load_rows(
            path, "dev_fit", findings=[PRIMARY_FINDINGS[0]], models=["huatuo"],
            require_complete_cells=False,
        )


def test_loader_rejects_ambiguous_positional_reader_votes(tmp_path):
    row = _json_row()
    # CECD factorial rows historically preserved this sorted manifest order as
    # values only: R10, R8, R9.  Treating it as R8, R9, R10 silently swaps
    # reader identity, so outcome-blind controls must join the named manifest.
    row["reader_votes"] = [1, 0, 1]
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(AliasingError, match="positional reader vote lists are ambiguous"):
        load_rows(
            path, "dev_fit", findings=[PRIMARY_FINDINGS[0]], models=["huatuo"],
            require_complete_cells=False,
        )


def test_exact_pattern_increment_is_zero_on_unanimous_clear_cases():
    rows = _synthetic("dev_fit", 1, replicates=2)
    standardization = fit_standardization(rows)
    augmented = design_matrix(rows, standardization, PRIMARY_FINDINGS, MODELS, True)
    baseline_width = len(feature_names(PRIMARY_FINDINGS, MODELS, False))
    clear = np.asarray([row["positive_votes"] in (0, 3) for row in rows])
    assert clear.any()
    assert np.all(augmented[clear, baseline_width:] == 0)


def test_group_folds_never_split_an_image():
    rows = _synthetic("dev_fit", 2, replicates=2)
    # Add a second finding record for one image to exercise whole-image grouping.
    duplicate = dict(rows[0])
    duplicate["record_key"] += ":second"
    duplicate["finding"] = PRIMARY_FINDINGS[1]
    rows.append(duplicate)
    folds = group_folds(rows, 3, 17)
    by_image = {}
    for row, fold in zip(rows, folds):
        by_image.setdefault(row["image_id"], set()).add(int(fold))
    assert all(len(values) == 1 for values in by_image.values())


def test_dev_fit_then_confirmation_is_serialized_without_refit():
    dev = _synthetic("dev_fit", 3, replicates=4)
    locked = _synthetic("confirmation_locked", 4, replicates=4)
    fit = fit_dev(
        dev, input_sha256="1" * 64, folds=3, bootstrap_draws=120, seed=11,
    )
    assert fit["confirmation_consumed"] is False
    assert fit["status"] == "dev_fit_complete_confirmation_not_read"
    result = confirm(
        fit, locked, input_sha256="2" * 64, bootstrap_draws=120, seed=12,
    )
    assert result["confirmation_refit"] is False
    assert result["fit_fingerprint"] == fit["fingerprint"]
    assert result["gates"]["clear_case_identity_increment_defined"] is False
    assert result["clear_case_increment"]["status"] == "structurally_nonidentifiable_from_exact_reader_pattern"
    assert result["clear_case_increment"]["maximum_absolute_prediction_difference"] == 0
    assert "both_models_identity_increment" in result["gates"]
    assert result["paper_claim_authorized"] is False
    assert result["mitigation_authorized"] is False


def test_saturated_cell_sensitivity_does_not_create_reader_alias_gain():
    rng = np.random.default_rng(903)
    rows = []
    for model_index, model in enumerate(MODELS):
        for finding_index, finding in enumerate(PRIMARY_FINDINGS):
            slope = -2.0 + 0.27 * (2 * finding_index + model_index)
            for pattern_index in range(8):
                pattern = f"{pattern_index:03b}"
                votes = tuple(int(value) for value in pattern)
                for replicate in range(16):
                    margin = float(rng.normal())
                    probability = 1 / (1 + np.exp(-(-0.2 + .35 * sum(votes) + slope * margin)))
                    target = int(rng.random() < probability)
                    image = f"confound-{model}-{finding}-{pattern}-{replicate}"
                    rows.append({
                        "record_key": image, "image_id": image, "finding": finding,
                        "model": model, "stage": "dev_fit", "endpoint": "positive_commitment",
                        "votes": votes, "pattern": pattern, "positive_votes": sum(votes),
                        "clean_margin": margin, "target": target,
                    })
    fit = fit_dev(rows, input_sha256="3" * 64, folds=4, bootstrap_draws=100, seed=19)
    # Pattern is independent of the deliberately heterogeneous cell slopes.
    assert fit["dev_oof"]["delta_auroc"]["estimate"] < .03
    assert fit["dev_oof"]["relative_nll_improvement"]["estimate"] < .05


def test_clinical_error_endpoint_cannot_yield_reader_operating_order_claim():
    dev = _synthetic("dev_fit", 31, replicates=2)
    locked = _synthetic("confirmation_locked", 32, replicates=2)
    for row in [*dev, *locked]:
        row["endpoint"] = "clinical_error"
    fit = fit_dev(dev, input_sha256="4" * 64, folds=2, bootstrap_draws=100, seed=22)
    assert fit["dev_reader_order"]["status"] == "not_applicable_for_clinical_error"
    result = confirm(fit, locked, input_sha256="5" * 64, bootstrap_draws=100, seed=23)
    assert result["gates"]["ordering_6_of_8_each_model"] is False
    assert result["classification"] != "reader_disagreement_semantics_only"


def test_fit_fingerprint_drift_is_rejected():
    dev = _synthetic("dev_fit", 5, replicates=2)
    locked = _synthetic("confirmation_locked", 6, replicates=2)
    fit = fit_dev(dev, input_sha256="1" * 64, folds=2, bootstrap_draws=100, seed=7)
    fit["models_fit"]["baseline"]["intercept"] += 1
    with pytest.raises(AliasingError, match="invalid or drifted"):
        confirm(fit, locked, input_sha256="2" * 64, bootstrap_draws=100, seed=8)


def test_current_preflight_is_truthfully_blocked_without_outcome_bindings():
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    result = validate(payload, root=ROOT)
    assert result["passed"] is False
    assert result["outcomes_read"] is False
    assert result["reader_alias_execution_ready"] is False
    assert result["cecd_primary_gate_modification_authorized"] is False
    assert "binding_missing:dev_fit_input" in result["blockers"]
    assert "binding_missing:confirmation_locked_input" in result["blockers"]


def test_preflight_rejects_claim_promotion_and_late_output(tmp_path):
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["claim_boundaries"]["causal_mechanism_claimed"] = True
    with pytest.raises(PreflightError, match="claim_boundaries"):
        validate(payload, root=ROOT)

    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    for name in ("dev_fit_input", "confirmation_locked_input", "listing_matched_count_length_input"):
        bound = tmp_path / f"{name}.jsonl"
        bound.write_text("{}\n", encoding="utf-8")
        payload["bindings"][name] = {
            "path": str(bound), "sha256": sha256_file(bound), "bytes": bound.stat().st_size,
        }
    for name in payload["output_roots"]:
        output = tmp_path / name
        output.mkdir()
        payload["output_roots"][name] = str(output)
    assert validate(payload, root=ROOT)["passed"] is True
    (tmp_path / "dev_fit" / "late.json").write_text("{}\n", encoding="utf-8")
    result = validate(payload, root=ROOT)
    assert result["passed"] is False
    assert "output_root_not_empty_preflight_too_late:dev_fit" in result["blockers"]
