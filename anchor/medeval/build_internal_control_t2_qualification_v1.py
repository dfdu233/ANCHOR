#!/usr/bin/env python3
"""Build machine-checkable T2 artifacts for temperature and self-consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "internal-control-t2-qualification-builder-v1"


def build(
    *,
    generation_audit: Path,
    aggregation: Path,
    freeze_provenance: Path,
    execution_contract: Path,
    aggregation_contract: Path,
    qualification_contract: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation = json.loads(generation_audit.read_text())
    aggregate = json.loads(aggregation.read_text())
    freeze = json.loads(freeze_provenance.read_text())
    execution = json.loads(execution_contract.read_text())
    aggregation_spec = json.loads(aggregation_contract.read_text())
    if not generation.get("passed"):
        raise ValueError("generation audit did not pass")
    if not aggregate.get("passed_t2_functional"):
        raise ValueError("claim self-consistency functional gate did not pass")
    contract_sha = sha256_file(qualification_contract)
    provenance = {
        "dev_manifest_sha256": freeze["development_manifest_sha256"],
        "test_manifest_sha256": freeze["held_out_manifest_sha256"],
    }
    model_audits = list(generation["models"].values())
    arms = execution["temperature_length_controls"]["arms"]
    temperature = {
        "protocol_version": "temperature-length-control-t2-v1",
        "contract_sha256": contract_sha,
        "method": "temperature_length_controls",
        "stage": "T2",
        "provenance": provenance,
        "design": {
            "development_grid_frozen_before_test": True,
            "test_labels_used_for_tuning": False,
            "temperature_top_p_grid": [
                [float(arm.get("temperature", 1.0)), float(arm.get("top_p", 1.0))]
                for arm in arms if arm["decode_mode"] == "sample"
            ],
        },
        "generation": {
            "generated_token_ids_recorded": all(
                arm["token_trace_rows"] == generation["limit"]
                for model in model_audits for arm in model["arms"].values()
            ),
            "seed_ledger_complete": all(model["sampling_k"] == 5 for model in model_audits),
            "sampling_activation_non_degenerate": all(
                model["sampling_non_degenerate"] for model in model_audits
            ),
            "exact_qid_coverage": all(
                arm["rows"] == generation["limit"]
                for model in model_audits for arm in model["arms"].values()
            ),
            "stop_and_cap_provenance_complete": all(
                arm["stop_cap_trace_rows"] == generation["limit"]
                for model in model_audits for arm in model["arms"].values()
            ),
            "posthoc_truncation_used": False,
        },
        "analysis": {"matched_length_plan_frozen": True},
        "diagnostics": {
            "generation_audit_sha256": sha256_file(generation_audit),
            "models": sorted(generation["models"]),
            "development_rows": generation["limit"],
            "interpretation": "functional generation control qualification only; no efficacy claim",
        },
    }
    temperature["fingerprint"] = sha256_json(temperature)

    qualification = aggregate["qualification"]
    self_consistency = {
        "protocol_version": "claim-self-consistency-t2-v1",
        "contract_sha256": contract_sha,
        "method": "self_consistency",
        "stage": "T2",
        "provenance": provenance,
        "design": {
            "frozen_before_test": True,
            "test_labels_used_for_selection": False,
        },
        "sampling": {
            "all_k_samples_complete": qualification["all_k_samples_complete"],
            "seed_ledger_complete": all(model["sampling_k"] == 5 for model in model_audits),
            "deterministic_replay_passed": all(
                model["deterministic_replay_exact"] for model in model_audits
            ),
            "non_degenerate_diversity": all(
                model["sampling_non_degenerate"] for model in model_audits
            ),
            "k": int(aggregation_spec["aggregation"]["k"]),
        },
        "aggregation": {
            "atomic_claim_normalization": qualification["atomic_claim_normalization"],
            "preserves_polarity_anatomy_attributes_uncertainty": qualification[
                "preserves_polarity_anatomy_attributes_uncertainty"
            ],
            "exact_text_majority_vote_used": qualification["exact_text_majority_vote_used"],
            "trace_hash_recorded": qualification["trace_hash_recorded"],
            "rule_id": "radgraph-structured-claim-frequency-v1",
        },
        "diagnostics": {
            "generation_audit_sha256": sha256_file(generation_audit),
            "aggregation_sha256": sha256_file(aggregation),
            "applicable_groups": aggregate["summary"]["applicable_groups"],
            "changed_groups": aggregate["summary"]["changed_from_seed42_groups"],
            "clinical_efficacy": "not established at T2",
        },
    }
    self_consistency["fingerprint"] = sha256_json(self_consistency)
    return temperature, self_consistency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-audit", required=True, type=Path)
    parser.add_argument("--aggregation", required=True, type=Path)
    parser.add_argument("--freeze-provenance", required=True, type=Path)
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--aggregation-contract", required=True, type=Path)
    parser.add_argument("--qualification-contract", required=True, type=Path)
    parser.add_argument("--temperature-output", required=True, type=Path)
    parser.add_argument("--self-consistency-output", required=True, type=Path)
    args = parser.parse_args()
    temperature, self_consistency = build(
        generation_audit=args.generation_audit,
        aggregation=args.aggregation,
        freeze_provenance=args.freeze_provenance,
        execution_contract=args.execution_contract,
        aggregation_contract=args.aggregation_contract,
        qualification_contract=args.qualification_contract,
    )
    atomic_json(args.temperature_output, temperature)
    atomic_json(args.self_consistency_output, self_consistency)
    print(
        json.dumps(
            {
                "temperature": temperature["fingerprint"],
                "self_consistency": self_consistency["fingerprint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
