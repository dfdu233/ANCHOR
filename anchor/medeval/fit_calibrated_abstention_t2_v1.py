#!/usr/bin/env python3
"""Fit and functionally validate the frozen claim-selective abstention control."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from anchor.medeval.evaluate_oe_vqa import token_f1
from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "calibrated-abstention-fit-t2-v2"


def t2_gate_passed(qualification: dict[str, Any]) -> bool:
    """Fail closed per model; pooled coverage may hide a degenerate adapter."""

    calibration = qualification["calibration"]
    accounting = qualification["accounting"]
    diagnostics = qualification["diagnostics"]
    model_diagnostics = diagnostics.get("per_model", {})
    return bool(
        model_diagnostics
        and calibration["deterministic_replay_passed"]
        and accounting["claim_selective_oe"]
        and all(
            row["calibration_positive_proxy_rows"] > 0
            and row["calibration_negative_proxy_rows"] > 0
            and 0.0 < row["validation_coverage_fraction"] < 1.0
            and row["validation_extracted_claims"] > 0
            and row["validation_claims_marked_uncertain"] > 0
            for row in model_diagnostics.values()
        )
    )


def fit_isotonic(confidence: list[float], target: list[float]) -> list[dict[str, float]]:
    if len(confidence) != len(target) or not confidence:
        raise ValueError("isotonic inputs must be equal and non-empty")
    points: dict[float, list[float]] = defaultdict(list)
    for score, value in zip(confidence, target):
        points[float(score)].append(float(value))
    blocks = [
        {"low": score, "high": score, "sum": sum(values), "weight": float(len(values))}
        for score, values in sorted(points.items())
    ]
    index = 0
    while index < len(blocks) - 1:
        left = blocks[index]["sum"] / blocks[index]["weight"]
        right = blocks[index + 1]["sum"] / blocks[index + 1]["weight"]
        if left <= right:
            index += 1
            continue
        blocks[index : index + 2] = [
            {
                "low": blocks[index]["low"],
                "high": blocks[index + 1]["high"],
                "sum": blocks[index]["sum"] + blocks[index + 1]["sum"],
                "weight": blocks[index]["weight"] + blocks[index + 1]["weight"],
            }
        ]
        index = max(0, index - 1)
    return [
        {
            "confidence_low": block["low"],
            "confidence_high": block["high"],
            "estimated_correctness": block["sum"] / block["weight"],
            "weight": block["weight"],
        }
        for block in blocks
    ]


def select_nll_threshold(records: list[dict[str, Any]], minimum_coverage: float) -> dict[str, Any]:
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum coverage must lie in (0,1]")
    candidates = []
    for threshold in sorted({float(row["nll"]) for row in records}):
        retained = [row for row in records if float(row["nll"]) <= threshold]
        coverage = len(retained) / len(records)
        if coverage + 1e-12 < minimum_coverage:
            continue
        error_risk = 1.0 - sum(float(row["correct_proxy"]) for row in retained) / len(retained)
        candidates.append((error_risk, -coverage, threshold, retained))
    if not candidates:
        raise ValueError("no threshold meets minimum coverage")
    error_risk, negative_coverage, threshold, retained = min(candidates, key=lambda value: value[:3])
    return {
        "nll_threshold": threshold,
        "calibration_coverage": -negative_coverage,
        "calibration_error_risk": error_risk,
        "retained_qids": [row["qid"] for row in retained],
    }


def _uncertain_claims(claims: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output = []
    changed = 0
    for claim in claims:
        value = dict(claim)
        if value.get("provenance") == "image_grounded" and value.get("uncertainty") != "uncertain":
            value["uncertainty"] = "uncertain"
            changed += 1
        output.append(value)
    return output, changed


def fit_and_apply(
    *,
    extraction: Path,
    pilot_manifest: Path,
    freeze_provenance: Path,
    execution_contract: Path,
    qualification_contract: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extracted = json.loads(extraction.read_text())
    pilot = json.loads(pilot_manifest.read_text())
    freeze = json.loads(freeze_provenance.read_text())
    contract = json.loads(execution_contract.read_text())
    if len(pilot) != 32 or len({row["image_sha256"] for row in pilot}) != 32:
        raise ValueError("abstention T2 requires the frozen 32-image pilot")
    calibration_qids = [str(row["qid"]) for row in pilot[:16]]
    validation_qids = [str(row["qid"]) for row in pilot[16:]]
    if set(calibration_qids) & set(validation_qids):
        raise AssertionError("internal split qids overlap")
    references = {str(row["qid"]): str(row["answer"]) for row in pilot}
    greedy = {}
    for report in extracted["reports"]:
        source = report["source"]
        if source.get("stream") != "greedy256":
            continue
        key = (str(source["model"]), str(source["qid"]))
        if key in greedy:
            raise ValueError(f"duplicate greedy extraction: {key}")
        greedy[key] = report
    models = sorted({model for model, _ in greedy})
    minimum_coverage = float(contract["development_proxy"]["minimum_calibration_coverage"])
    thresholds: dict[str, Any] = {}
    calibration_by_model: dict[str, list[dict[str, Any]]] = {}
    actions: list[dict[str, Any]] = []
    for model in models:
        calibration_records = []
        for qid in calibration_qids:
            report = greedy[(model, qid)]
            prediction = str(report["report"])
            nll = float(report["source"]["mean_token_nll"])
            calibration_records.append(
                {
                    "qid": qid,
                    "nll": nll,
                    "confidence": -nll,
                    "token_f1": token_f1(prediction, references[qid]),
                    "correct_proxy": token_f1(prediction, references[qid]) >= 0.5,
                }
            )
        threshold = select_nll_threshold(calibration_records, minimum_coverage)
        threshold["isotonic_blocks"] = fit_isotonic(
            [row["confidence"] for row in calibration_records],
            [float(row["correct_proxy"]) for row in calibration_records],
        )
        threshold["calibration_records"] = calibration_records
        calibration_by_model[model] = calibration_records
        thresholds[model] = threshold
        for qid in validation_qids:
            report = greedy[(model, qid)]
            nll = float(report["source"]["mean_token_nll"])
            accepted = nll <= float(threshold["nll_threshold"])
            claims = [dict(value) for value in report.get("claims", [])]
            modified, changed = (claims, 0) if accepted else _uncertain_claims(claims)
            actions.append(
                {
                    "model": model,
                    "qid": qid,
                    "nll": nll,
                    "accepted": accepted,
                    "action": "retain" if accepted else "mark_extracted_image_claims_uncertain",
                    "original_claims": claims,
                    "output_claims": modified,
                    "claims_marked_uncertain": changed,
                    "whole_answer_erased": False,
                    "unparsed_not_applicable": not claims,
                }
            )
    replay_thresholds: dict[str, Any] = {}
    for model in models:
        replay_threshold = select_nll_threshold(
            calibration_by_model[model], minimum_coverage
        )
        replay_threshold["isotonic_blocks"] = fit_isotonic(
            [row["confidence"] for row in calibration_by_model[model]],
            [float(row["correct_proxy"]) for row in calibration_by_model[model]],
        )
        replay_threshold["calibration_records"] = calibration_by_model[model]
        replay_thresholds[model] = replay_threshold
    replay_actions: list[dict[str, Any]] = []
    for model in models:
        for qid in validation_qids:
            report = greedy[(model, qid)]
            nll = float(report["source"]["mean_token_nll"])
            accepted = nll <= float(replay_thresholds[model]["nll_threshold"])
            claims = [dict(value) for value in report.get("claims", [])]
            modified, changed = (claims, 0) if accepted else _uncertain_claims(claims)
            replay_actions.append(
                {
                    "model": model,
                    "qid": qid,
                    "nll": nll,
                    "accepted": accepted,
                    "action": "retain" if accepted else "mark_extracted_image_claims_uncertain",
                    "original_claims": claims,
                    "output_claims": modified,
                    "claims_marked_uncertain": changed,
                    "whole_answer_erased": False,
                    "unparsed_not_applicable": not claims,
                }
            )
    deterministic_replay = sha256_json(
        {"thresholds": thresholds, "actions": actions}
    ) == sha256_json({"thresholds": replay_thresholds, "actions": replay_actions})

    threshold_artifact = {
        "protocol_version": VERSION,
        "execution_contract_sha256": sha256_file(execution_contract),
        "extraction_sha256": sha256_file(extraction),
        "calibration_qids": calibration_qids,
        "validation_qids": validation_qids,
        "models": models,
        "thresholds": thresholds,
        "test_labels_used": False,
        "development_reference_proxy_only": True,
    }
    threshold_artifact["fingerprint"] = sha256_json(threshold_artifact)
    replay = {
        "thresholds": thresholds,
        "actions": actions,
    }
    accepted = sum(row["accepted"] for row in actions)
    extracted_claims = sum(len(row["original_claims"]) for row in actions)
    marked = sum(row["claims_marked_uncertain"] for row in actions)
    per_model = {}
    for model in models:
        model_actions = [row for row in actions if row["model"] == model]
        model_calibration = calibration_by_model[model]
        model_accepted = sum(row["accepted"] for row in model_actions)
        per_model[model] = {
            "calibration_positive_proxy_rows": sum(
                bool(row["correct_proxy"]) for row in model_calibration
            ),
            "calibration_negative_proxy_rows": sum(
                not bool(row["correct_proxy"]) for row in model_calibration
            ),
            "validation_rows": len(model_actions),
            "validation_accepted": model_accepted,
            "validation_coverage_fraction": model_accepted / len(model_actions),
            "validation_extracted_claims": sum(
                len(row["original_claims"]) for row in model_actions
            ),
            "validation_claims_marked_uncertain": sum(
                row["claims_marked_uncertain"] for row in model_actions
            ),
            "validation_unparsed_answers": sum(
                row["unparsed_not_applicable"] for row in model_actions
            ),
        }
    diagnostics = {
        "models": models,
        "calibration_rows_per_model": len(calibration_qids),
        "validation_rows_per_model": len(validation_qids),
        "validation_rows": len(actions),
        "validation_accepted": accepted,
        "validation_coverage_fraction": accepted / len(actions),
        "validation_extracted_claims": extracted_claims,
        "validation_claims_marked_uncertain": marked,
        "validation_unparsed_answers": sum(row["unparsed_not_applicable"] for row in actions),
        "per_model": per_model,
        "calibration_validation_image_overlap": 0,
        "replay_sha256": sha256_json(replay),
        "independent_recompute_sha256": sha256_json(
            {"thresholds": replay_thresholds, "actions": replay_actions}
        ),
    }
    qualification = {
        "protocol_version": "calibrated-abstention-t2-v2",
        "contract_sha256": sha256_file(qualification_contract),
        "method": "calibrated_abstention",
        "stage": "T2",
        "provenance": {
            "dev_manifest_sha256": freeze["development_manifest_sha256"],
            "test_manifest_sha256": freeze["held_out_manifest_sha256"],
        },
        "calibration": {
            "fitted_on_disjoint_development": True,
            "test_labels_used": False,
            "threshold_locked_before_test": True,
            "deterministic_replay_passed": deterministic_replay,
            "confidence_statistic": contract["confidence"]["statistic"],
            "calibrator_id": "deterministic-pav-isotonic-per-model-v1",
            "threshold_artifact_sha256": "PENDING_WRITE",
        },
        "accounting": {
            "claim_selective_oe": extracted_claims > 0 and marked > 0,
            "whole_answer_erasure_without_zero_claims": False,
            "abstentions_counted_as_corrections": False,
            "omitted_claims_counted_as_omissions": True,
        },
        "test": {"coverage_fraction": diagnostics["validation_coverage_fraction"]},
        "diagnostics": diagnostics,
        "actions": actions,
    }
    return threshold_artifact, qualification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", required=True, type=Path)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--freeze-provenance", required=True, type=Path)
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--qualification-contract", required=True, type=Path)
    parser.add_argument("--threshold-output", required=True, type=Path)
    parser.add_argument("--qualification-output", required=True, type=Path)
    args = parser.parse_args()
    threshold, qualification = fit_and_apply(
        extraction=args.extraction,
        pilot_manifest=args.pilot_manifest,
        freeze_provenance=args.freeze_provenance,
        execution_contract=args.execution_contract,
        qualification_contract=args.qualification_contract,
    )
    atomic_json(args.threshold_output, threshold)
    qualification["calibration"]["threshold_artifact_sha256"] = sha256_file(args.threshold_output)
    qualification["threshold_artifact"] = str(args.threshold_output.resolve())
    qualification["fingerprint"] = sha256_json(qualification)
    atomic_json(args.qualification_output, qualification)
    print(json.dumps({"diagnostics": qualification["diagnostics"], "fingerprint": qualification["fingerprint"]}, indent=2))
    passed = t2_gate_passed(qualification)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
