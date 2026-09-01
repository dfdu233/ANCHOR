#!/usr/bin/env python3
"""Outcome-blind preflight for the reader-threshold aliasing control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


VERSION = "reader-threshold-aliasing-preflight-v1"
ROOT = Path("/home/dbw/ANCHOR")
READERS = ["R8", "R9", "R10"]
FINDINGS = [
    "aortic_enlargement", "cardiomegaly", "lung_opacity", "nodule_mass",
    "other_lesion", "pleural_effusion", "pleural_thickening", "pulmonary_fibrosis",
]
MODELS = ["huatuo", "hulu"]


class PreflightError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _file_record(record: Any, root: Path, label: str) -> tuple[Path | None, str | None]:
    if record is None:
        return None, f"binding_missing:{label}"
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
        raise PreflightError(f"{label}: invalid file-record schema")
    path = _resolve(record["path"], root)
    if not path.is_file() or path.is_symlink():
        return path, f"binding_missing_or_symlink:{label}"
    if (
        not isinstance(record["sha256"], str)
        or len(record["sha256"]) != 64
        or sha256_file(path) != record["sha256"]
        or isinstance(record["bytes"], bool)
        or record["bytes"] != path.stat().st_size
    ):
        return path, f"binding_drift:{label}"
    return path, None


def validate(payload: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    required = {
        "schema_version", "frozen_before_reader_alias_outcomes",
        "reader_alias_outcomes_consumed", "scientific_role", "reader_panel",
        "findings", "models", "task", "stages", "endpoint_contract",
        "model_contract", "statistics", "claim_boundaries", "bindings",
        "output_roots",
    }
    if set(payload) != required:
        raise PreflightError(
            f"schema drift missing={sorted(required-set(payload))} "
            f"extra={sorted(set(payload)-required)}"
        )
    if payload["schema_version"] != VERSION:
        raise PreflightError("schema version drift")
    if payload["frozen_before_reader_alias_outcomes"] is not True:
        raise PreflightError("preflight must precede outcomes")
    if payload["reader_alias_outcomes_consumed"] is not False:
        raise PreflightError("outcome-blind preflight cannot consume outcomes")
    expected_exact = {
        "scientific_role": "alternative_explanation_control_not_mainline",
        "reader_panel": READERS,
        "findings": FINDINGS,
        "models": MODELS,
        "task": "atomic_ce_clean_condition_only",
        "stages": {
            "fit": "dev_fit_group_crossfit_then_full_dev_serialization",
            "confirmation": "confirmation_locked_apply_once_no_refit_no_tuning",
        },
        "endpoint_contract": {
            "allowed": ["positive_commitment", "clinical_error"],
            "one_endpoint_per_frozen_run": True,
            "binary_target": True,
            "clean_margin_is_predictor_not_truth": True,
        },
        "model_contract": {
            "baseline": "vote_count+saturated_model_by_finding_intercepts+clean_z_by_model_by_finding_slopes",
            "increment": "frozen_baseline_logit_plus_exact_R8_R9_R10_pattern_residual_by_model_by_finding",
            "regularization_C": 1.0,
            "hyperparameter_tuning": False,
            "finding_specific_clean_sensitivity_control": True,
            "baseline_coefficients_frozen_during_pattern_fit": True,
        },
        "statistics": {
            "primary_delta_auroc_min": 0.05,
            "relative_nll_improvement_min": 0.05,
            "both_cluster_bootstrap_ci_low_above_zero": True,
            "bootstrap_unit": "whole_image_id",
            "bootstrap_draws": 5000,
            "group_crossfit_folds": 5,
            "ordering": "same_dev_reader_order_in_at_least_6_of_8_findings_for_each_model",
            "both_models_required": True,
            "clear_case_identity_increment_defined": False,
            "clear_case_reason": "exact_reader_identity_has_no_variation_on_000_or_111",
        },
        "claim_boundaries": {
            "can_modify_cecd_primary_gate": False,
            "causal_mechanism_claimed": False,
            "mitigation_authorized": False,
            "paper_claim_authorized": False,
            "disagreement_only_if_clear_case_fails": True,
            "listing_transfer_required_before_reopen": True,
            "future_clear_case_reopen_requires_separate_preregistration": True,
        },
    }
    for key, expected in expected_exact.items():
        if payload[key] != expected:
            raise PreflightError(f"frozen contract drift: {key}")
    bindings = payload["bindings"]
    expected_bindings = {
        "candidate_b_protocol", "analyzer_source", "dev_fit_input",
        "confirmation_locked_input", "listing_matched_count_length_input",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise PreflightError("binding closure drift")
    blockers: list[str] = []
    resolved: dict[str, str | None] = {}
    for label, record in bindings.items():
        path, blocker = _file_record(record, root, label)
        resolved[label] = str(path) if path is not None else None
        if blocker:
            blockers.append(blocker)
    output_roots = payload["output_roots"]
    if not isinstance(output_roots, Mapping) or set(output_roots) != {"dev_fit", "confirmation"}:
        raise PreflightError("output root closure drift")
    for label, value in output_roots.items():
        path = _resolve(value, root)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            blockers.append(f"output_root_not_empty_preflight_too_late:{label}")
    # Fit and confirmation must be distinct immutable files.  Content and
    # outcomes are deliberately not opened by this preflight.
    dev_path = resolved["dev_fit_input"]
    confirmation_path = resolved["confirmation_locked_input"]
    if dev_path is not None and dev_path == confirmation_path:
        blockers.append("dev_confirmation_binding_not_distinct")
    ready = not blockers
    return {
        "schema_version": VERSION,
        "status": "ready_outcome_blind" if ready else "blocked_outcome_blind",
        "passed": ready,
        "blockers": blockers,
        "resolved_bindings": resolved,
        "reader_alias_execution_ready": ready,
        "confirmation_refit_authorized": False,
        "cecd_primary_gate_modification_authorized": False,
        "mitigation_authorized": False,
        "paper_claim_authorized": False,
        "outcomes_read": False,
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    result = validate(payload, root=ROOT)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    _main()
