#!/usr/bin/env python3
"""Validate the outcome-blind CECD full-method collision preflight.

This gate is deliberately downstream of the behavioral three-stage gate and
the narrow Treble dual-semantics envelope.  It does not inspect model outputs,
authorize a paper claim, or alter any behavioral threshold.  Its sole purpose
is to prevent a mitigation experiment from starting before the 2026 closest
mechanism and evaluation collisions have executable, hash-bound controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


VERSION = "cecd-adversarial-method-preflight-v1"
ROOT = Path("/home/dbw/ANCHOR")
HEX64 = set("0123456789abcdef")
MODEL_FAMILIES = ("huatuo", "hulu")

COLLISION_SOURCES = {
    "hallucxr_length_omission": "arxiv:2605.20469",
    "system_mediated_yes_bias": "acl:2026.findings-acl.1940",
    "prompt_induced_heads": "acl:2026.acl-long.1941",
    "halp_architecture_specific_probe": "acl:2026.eacl-long.287",
    "cebc_evidence_bounded_editing": "acl:2026.acl-long.2142",
    "hallutrace_source_targeted_decoding": "acl:2026.alvr-main.29",
    "vli_instance_specific_steering": "acl:2026.acl-long.1784",
    "conrad_proper_score_confidence": "arxiv:2603.29492",
}

REQUIRED_CLAIM_BOUNDARIES = {
    "prompt_robustness_novelty_claimed": False,
    "generic_yes_bias_mitigation_novelty_claimed": False,
    "pre_generation_hallucination_detection_novelty_claimed": False,
    "minimal_evidence_bounded_editing_novelty_claimed": False,
    "generic_source_decomposition_novelty_claimed": False,
    "generic_adaptive_steering_novelty_claimed": False,
    "confidence_calibration_novelty_claimed": False,
    "universal_early_visual_late_language_layer_claimed": False,
}

REQUIRED_EVALUATION_CONTRACT = {
    "tasks": ["ce", "vindr_fixed_ontology_listing"],
    "truth": "independent_reader_votes_plus_admitted_atomic_claims",
    "automatic_judge_defines_truth": False,
    "positive_claim_count": "fixed_K_per_record_for_cecd_and_coverage_matched_controls",
    "cecd_claim_exchange": "one_for_one_only_no_deletion",
    "candidate_ontology": "frozen_14_finding_vindr_ontology",
    "recordwise_claim_identity_audit": True,
    "matched_claim_coverage_absolute_tolerance": 0.01,
    "matched_mean_length_relative_tolerance": 0.05,
    "length_stratified_analysis": True,
    "length_only_risk_baseline": True,
    "omission_point_increase_max": 0.0,
    "omission_one_sided_ci_upper_max": 0.01,
    "refusal_point_increase_max": 0.0,
    "negative_rate_reported": True,
    "hedge_rate_reported": True,
    "deleted_and_inserted_claims_reported": True,
    "reader_distribution_metrics": ["brier", "nll"],
    "bootstrap": {
        "replicates": 10_000,
        "unit": "patient_or_whole_image_cluster",
        "paired": True,
    },
}

REQUIRED_ANALYSIS_CONTROLS = {
    "behavioral_incremental_ladder": (
        "clean_score+entropy+length+marginals+full_grid_stability+behavioral_PID"
        "+system_attention+prompt_copy_head_score+HALP_risk_then_CECD"
    ),
    "halp_layer_selection": "dev_only_per_model_no_cross_architecture_layer_assumption",
    "prompt_head_selection": "dev_only_per_model_locked_test_never_scanned",
    "system_attention_test": "causal_redistribution_plus_no_image_and_balanced_yes_rate",
    "source_attribution_test": "vision_encoder_projector_decoder_component_ablation",
    "cebc_comparison": "paper_native_Pareto_plus_fixed_K_common_protocol",
    "adaptive_steering_comparison": "instance_specific_conflict_conditioned_control",
    "confidence_control": "proper_log_score_confidence_only_no_content_edit",
    "primary_contrasts": [
        "cecd_vs_unmitigated",
        "cecd_vs_full_orbit",
        "cecd_vs_stronger_treble_variant",
        "cecd_vs_system_attention_redistribution",
        "cecd_vs_prompt_copy_head_ablation",
        "cecd_vs_cebc_fixed_K",
        "cecd_vs_hallutrace_HAD",
        "cecd_vs_VLI",
    ],
}

ALLOWED_STATUS = {"ready", "not_implemented", "incompatible"}
ALLOWED_FIDELITY = {
    "evaluation_guard",
    "official_or_author_faithful_port",
    "independent_common_protocol",
    "proper_score_common_protocol_control",
    "not_available",
}
EXPECTED_READY_FIDELITY = {
    "hallucxr_length_omission": {"evaluation_guard"},
    "system_mediated_yes_bias": {
        "official_or_author_faithful_port", "independent_common_protocol",
    },
    "prompt_induced_heads": {
        "official_or_author_faithful_port", "independent_common_protocol",
    },
    "halp_architecture_specific_probe": {
        "official_or_author_faithful_port", "independent_common_protocol",
    },
    "cebc_evidence_bounded_editing": {
        "official_or_author_faithful_port", "independent_common_protocol",
    },
    "hallutrace_source_targeted_decoding": {
        "official_or_author_faithful_port", "independent_common_protocol",
    },
    "vli_instance_specific_steering": {
        "official_or_author_faithful_port", "independent_common_protocol",
    },
    "conrad_proper_score_confidence": {
        "official_or_author_faithful_port", "proper_score_common_protocol_control",
    },
}


class PreflightError(ValueError):
    """Raised when a purported outcome-blind preflight changes the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and set(text) <= HEX64


def _resolve(path: Any, root: Path) -> Path:
    value = Path(str(path))
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _validate_source_files(
    files: Any, *, root: Path, collision: str, status: str
) -> list[str]:
    if not isinstance(files, list):
        raise PreflightError(f"{collision}: source_files must be a list")
    errors: list[str] = []
    if status == "ready" and not files:
        errors.append(f"{collision}:ready_without_hash_bound_source")
    for index, record in enumerate(files):
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
            raise PreflightError(f"{collision}: source file {index} schema drift")
        path = _resolve(record["path"], root)
        repository = root.resolve()
        if path == repository or repository not in path.parents:
            errors.append(f"{collision}:source_outside_repository:{path}")
            continue
        if not path.is_file():
            errors.append(f"{collision}:missing_source:{path}")
            continue
        if not _hex64(record["sha256"]) or sha256_file(path) != record["sha256"]:
            errors.append(f"{collision}:source_hash_mismatch:{path}")
        if record["bytes"] != path.stat().st_size:
            errors.append(f"{collision}:source_size_mismatch:{path}")
    return errors


def validate_plan(payload: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    required = {
        "schema_version",
        "frozen_before_method_outputs",
        "method_outputs_consumed",
        "collision_sources",
        "claim_boundaries",
        "evaluation_contract",
        "analysis_controls",
        "controls",
        "bindings",
        "method_output_root",
    }
    if set(payload) != required:
        raise PreflightError(
            f"preflight fields missing={sorted(required-set(payload))} "
            f"extra={sorted(set(payload)-required)}"
        )
    if payload["schema_version"] != VERSION:
        raise PreflightError("preflight schema version mismatch")
    if payload["frozen_before_method_outputs"] is not True:
        raise PreflightError("preflight must be frozen before method outputs")
    if payload["method_outputs_consumed"] is not False:
        raise PreflightError("outcome-blind preflight cannot consume method outputs")
    if payload["collision_sources"] != COLLISION_SOURCES:
        raise PreflightError("collision source closure drifted")
    if payload["claim_boundaries"] != REQUIRED_CLAIM_BOUNDARIES:
        raise PreflightError("occupied novelty claim boundary drifted")
    if payload["evaluation_contract"] != REQUIRED_EVALUATION_CONTRACT:
        raise PreflightError("length/omission/coverage evaluation contract drifted")
    if payload["analysis_controls"] != REQUIRED_ANALYSIS_CONTROLS:
        raise PreflightError("mechanism control ladder drifted")

    bindings = payload["bindings"]
    expected_bindings = {
        "three_stage_input_gate",
        "locked_confirmation_analysis",
        "dual_semantics_preflight",
        "oe_manifest",
        "claim_contract",
        "evaluator_source",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise PreflightError("preflight binding closure drifted")
    blockers: list[str] = []
    binding_paths: list[Path] = []
    for label, record in bindings.items():
        if record is None:
            blockers.append(f"binding_not_frozen:{label}")
            continue
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
            raise PreflightError(f"binding {label} schema drift")
        path = _resolve(record["path"], root)
        binding_paths.append(path)
        if not path.is_file():
            blockers.append(f"binding_missing:{label}")
        elif (
            not _hex64(record["sha256"])
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            blockers.append(f"binding_drift:{label}")
    if len(binding_paths) != len(set(binding_paths)):
        blockers.append("binding_paths_alias_distinct_contract_objects")

    controls = payload["controls"]
    if not isinstance(controls, Mapping) or set(controls) != set(COLLISION_SOURCES):
        raise PreflightError("collision-control closure drifted")
    control_status: dict[str, Any] = {}
    for collision in COLLISION_SOURCES:
        record = controls[collision]
        expected = {
            "status", "implementation_fidelity", "models", "dev_only_selection",
            "locked_test_untouched", "source_files", "notes",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise PreflightError(f"{collision}: control schema drift")
        status = record["status"]
        fidelity = record["implementation_fidelity"]
        if status not in ALLOWED_STATUS or fidelity not in ALLOWED_FIDELITY:
            raise PreflightError(f"{collision}: unknown status or fidelity")
        if not isinstance(record["notes"], str):
            raise PreflightError(f"{collision}: notes must be text")
        models = record["models"]
        if not isinstance(models, Mapping) or set(models) != set(MODEL_FAMILIES):
            raise PreflightError(f"{collision}: both model families must be explicit")
        if any(value not in ALLOWED_STATUS for value in models.values()):
            raise PreflightError(f"{collision}: invalid model readiness")
        if status == "ready":
            if set(models.values()) != {"ready"}:
                blockers.append(f"{collision}:not_ready_on_both_models")
            if record["dev_only_selection"] is not True:
                blockers.append(f"{collision}:selection_not_dev_only")
            if record["locked_test_untouched"] is not True:
                blockers.append(f"{collision}:locked_test_touched")
            if fidelity == "not_available":
                blockers.append(f"{collision}:ready_but_fidelity_unavailable")
            if fidelity not in EXPECTED_READY_FIDELITY[collision]:
                blockers.append(f"{collision}:inadmissible_ready_fidelity")
        else:
            blockers.append(f"{collision}:{status}")
            if fidelity != "not_available":
                blockers.append(f"{collision}:unready_fidelity_not_not_available")
            if "ready" in set(models.values()):
                blockers.append(f"{collision}:unready_control_has_ready_model")
        blockers.extend(
            _validate_source_files(
                record["source_files"], root=root, collision=collision, status=status
            )
        )
        control_status[collision] = {
            "status": status,
            "fidelity": fidelity,
            "models": dict(models),
        }

    output_root = _resolve(payload["method_output_root"], root)
    repository = root.resolve()
    if output_root == repository or repository not in output_root.parents:
        raise PreflightError("method_output_root must be a narrow repository child")
    if output_root.exists() and any(output_root.iterdir()):
        blockers.append("method_output_root_not_empty_preflight_too_late")

    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "version": VERSION,
        "status": (
            "ready_for_outcome_blind_full_method_execution"
            if ready else "blocked_mechanism_paper_scope_only"
        ),
        "passed": ready,
        "blockers": blockers,
        "controls": control_status,
        "full_method_execution_ready": ready,
        "mitigation_novelty_authorized": False,
        "paper_claim_authorized": False,
        "mechanism_paper_scope_only": not ready,
        "three_stage_thresholds_modified": False,
        "method_outputs_consumed": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PreflightError("preflight plan must be a JSON object")
    result = validate_plan(payload, root=args.root)
    result["plan"] = {
        "path": str(args.plan.resolve()),
        "sha256": sha256_file(args.plan),
        "bytes": args.plan.stat().st_size,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
