"""Hash-bound output-form audit for the repaired internal-control T3 matrix.

This audit is deliberately separate from clinical scoring.  It adds the
question-conditioned sentence-completion gate required by the unified OE
contract without reading reference answers or changing completed generations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.qualify_oe_generation import qualify
from anchor.medeval.store import atomic_write_json


REFERENCE_KEYS = {"answer", "gt_ans", "reference", "references"}
PROTOCOL_VERSION = "internal-control-generation-form-audit-v1"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(
    *,
    run_root: Path,
    manifest_path: Path,
    execution_contract_path: Path,
    generation_audit_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(execution_contract_path.read_text())
    generation_audit = json.loads(generation_audit_path.read_text())
    manifest_sha256 = sha256_file(manifest_path)
    contract_sha256 = sha256_file(execution_contract_path)
    errors: list[str] = []

    if not generation_audit.get("passed"):
        errors.append("upstream generation audit did not pass")
    if generation_audit.get("manifest_sha256") != manifest_sha256:
        errors.append("upstream generation audit manifest hash mismatch")
    if generation_audit.get("execution_contract_sha256") != contract_sha256:
        errors.append("upstream generation audit execution-contract hash mismatch")

    expected_rows = len(manifest)
    max_new_tokens = int(contract["generation"]["max_new_tokens"])
    oe = contract["oe_qualification"]
    records: list[dict[str, Any]] = []
    for model in contract["models"]:
        for arm in oe["clinical_raw_arms"]:
            answers_path = run_root / model / arm / "answers.jsonl"
            if not answers_path.is_file():
                records.append(
                    {
                        "model": model,
                        "arm": arm,
                        "answers_path": str(answers_path.resolve()),
                        "eligible": False,
                        "errors": ["answers missing"],
                    }
                )
                errors.append(f"{model}/{arm}: answers missing")
                continue
            rows = _load_jsonl(answers_path)
            result = qualify(
                manifest,
                rows,
                limit=expected_rows,
                min_nonempty_rate=float(oe["min_nonempty_rate"]),
                min_unique_rate=0.10,
                max_function_word_only_rate=float(oe["max_function_only_rate"]),
                max_new_tokens=max_new_tokens,
                max_cap_hit_rate=float(oe["max_cap_rate"]),
                require_terminal_completeness=True,
                min_terminal_completeness_rate=0.95,
                terminal_question_policy="explicit_sentence_instruction",
            )
            reference_fields = sorted(
                {key for row in rows for key in REFERENCE_KEYS if key in row}
            )
            record = {
                "model": model,
                "arm": arm,
                "answers_path": str(answers_path.resolve()),
                "answers_sha256": sha256_file(answers_path),
                "rows": len(rows),
                "qualification": result,
                "reference_fields_co_resident": reference_fields,
                "reference_fields_accessed_by_auditor": False,
                "eligible": bool(result["passed"]),
                "errors": [] if result["passed"] else ["output-form qualification failed"],
            }
            records.append(record)
            if not result["passed"]:
                errors.append(f"{model}/{arm}: output-form qualification failed")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_scope": (
            "outcome-blind generation-form qualification only; no clinical efficacy claim"
        ),
        "run_root": str(run_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "execution_contract_path": str(execution_contract_path.resolve()),
        "execution_contract_sha256": contract_sha256,
        "generation_audit_path": str(generation_audit_path.resolve()),
        "generation_audit_sha256": sha256_file(generation_audit_path),
        "reference_answers_used_for_qualification": False,
        "clinical_labels_used_for_qualification": False,
        "terminal_question_policy": "explicit_sentence_instruction",
        "minimum_terminal_completeness_rate": 0.95,
        "records": records,
        "passed": bool(records) and not errors,
        "physician_pack_operationally_authorized": bool(records) and not errors,
        "clinical_efficacy_authorized": False,
        "errors": errors,
    }
    payload["fingerprint"] = sha256_json(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--generation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        run_root=args.run_root,
        manifest_path=args.manifest,
        execution_contract_path=args.execution_contract,
        generation_audit_path=args.generation_audit,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
