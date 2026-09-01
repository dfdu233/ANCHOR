"""Contract-driven T3 trace and OE-eligibility audit for repaired control matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.audit_internal_control_generation_t2_v1 import _audit_arm, _load_jsonl
from anchor.medeval.audit_oe_generation_qualification_v1 import audit_qualification
from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.run_native_oe_control_matrix_v1 import frozen_arms
from anchor.medeval.store import atomic_write_json


def expected_arm_names(contract: dict[str, Any]) -> list[str]:
    names = [arm.name for arm in frozen_arms(contract)]
    if len(names) != len(set(names)):
        raise ValueError("execution contract expands to duplicate arm names")
    return names


def audit(
    run_root: Path,
    manifest_path: Path,
    provenance_path: Path,
    contract_path: Path,
    *,
    limit: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())[:limit]
    provenance = json.loads(provenance_path.read_text())
    contract = json.loads(contract_path.read_text())
    expected_qids = [str(row["qid"]) for row in manifest]
    arms = expected_arm_names(contract)
    errors: list[str] = []
    if provenance.get("pilot_manifest_sha256") != sha256_file(manifest_path):
        errors.append("held-out manifest hash differs from frozen provenance")
    if provenance.get("execution_contract_sha256") != sha256_file(contract_path):
        errors.append("execution contract hash differs from frozen provenance")
    if provenance.get("source_test_image_overlap") != 0:
        errors.append("development/test image leakage")
    if len(expected_qids) != len(set(expected_qids)):
        errors.append("held-out qids are not unique")

    model_results: dict[str, Any] = {}
    for model in contract["models"]:
        arm_results = {
            arm: _audit_arm(run_root / model / arm, expected_qids, model) for arm in arms
        }
        model_errors = [
            f"{arm}: {message}"
            for arm, result in arm_results.items()
            for message in result["errors"]
        ]
        seed_arms = [
            f"sample_t07_p09_seed{seed}"
            for seed in contract["generation"]["sampling_seed_ledger"]
        ]
        seed_rows = {
            arm: _load_jsonl(run_root / model / arm / "answers.jsonl")
            if arm_results[arm]["passed"]
            else []
            for arm in seed_arms
        }
        diverse_qids = 0
        if all(len(rows) == len(expected_qids) for rows in seed_rows.values()):
            for index in range(len(expected_qids)):
                traces = {
                    tuple(seed_rows[arm][index]["metadata"]["generated_token_ids"])
                    for arm in seed_arms
                }
                diverse_qids += int(len(traces) > 1)
        original = seed_rows.get("sample_t07_p09_seed42", [])
        replay = (
            _load_jsonl(run_root / model / "replay_t07_p09_seed42" / "answers.jsonl")
            if arm_results["replay_t07_p09_seed42"]["passed"]
            else []
        )
        replay_exact = bool(original) and len(original) == len(replay) and all(
            left["text"] == right["text"]
            and left["metadata"]["generated_token_ids"]
            == right["metadata"]["generated_token_ids"]
            for left, right in zip(original, replay)
        )
        if diverse_qids == 0:
            model_errors.append("all five sampling streams are degenerate")
        if not replay_exact:
            model_errors.append("same-seed deterministic replay differs")
        model_results[model] = {
            "arms": arm_results,
            "diverse_qids": diverse_qids,
            "sampling_non_degenerate": diverse_qids > 0,
            "deterministic_replay_exact": replay_exact,
            "trace_passed": not model_errors,
            "passed": not model_errors,
            "errors": model_errors,
        }
        errors.extend(f"{model}: {message}" for message in model_errors)

    oe = contract["oe_qualification"]
    qualification = audit_qualification(
        run_root,
        models=list(contract["models"]),
        arms=list(oe["clinical_raw_arms"]),
        expected_rows=len(expected_qids),
        max_cap_rate=float(oe["max_cap_rate"]),
        min_nonempty_rate=float(oe["min_nonempty_rate"]),
        max_function_only_rate=float(oe["max_function_only_rate"]),
    )
    if not qualification["all_eligible"]:
        errors.append("one or more clinical OE arms failed the frozen generation qualification")
    result = {
        "protocol_version": "internal-control-generation-t3-audit-v2",
        "stage": "T3",
        "evidence_scope": "held-out trace and operational OE qualification; no clinical efficacy claim",
        "run_root": str(run_root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_sha256": sha256_file(provenance_path),
        "execution_contract_sha256": sha256_file(contract_path),
        "reference_answers_used_for_qualification": False,
        "clinical_labels_used_for_qualification": False,
        "expected_qids": expected_qids,
        "expected_arms": arms,
        "baseline_arm": contract["clinical_analysis"]["baseline"],
        "models": model_results,
        "oe_qualification": qualification,
        "passed": not errors,
        "errors": errors,
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.run_root,
        args.manifest,
        args.provenance,
        args.execution_contract,
        limit=args.limit,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
