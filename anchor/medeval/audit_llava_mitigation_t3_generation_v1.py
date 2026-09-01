"""Outcome-blind operational audit for the LLaVA mitigation T3 matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.qualify_oe_generation import terminal_required
from anchor.medeval.store import atomic_write_json


REFERENCE_KEYS = {"answer", "gt_ans", "reference", "references"}
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its",
    "no", "not", "of", "on", "or", "she", "that", "the", "their", "there",
    "they", "this", "to", "was", "we", "were", "with", "yes", "you",
}


def _load_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def _function_only(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and all(word in FUNCTION_WORDS for word in words)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def audit(
    *,
    run_root: Path,
    execution_contract_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    contract = json.loads(execution_contract_path.read_text())
    errors: list[str] = []
    bindings = contract["source_bindings"]
    binding_records = {}
    for name, binding in bindings.items():
        path = _resolve(repository_root, binding["path"])
        actual = sha256_file(path) if path.is_file() else None
        binding_records[name] = {
            "path": str(path),
            "expected_sha256": binding["sha256"],
            "actual_sha256": actual,
            "matched": actual == binding["sha256"],
        }
        if actual != binding["sha256"]:
            errors.append(f"source binding mismatch: {name}")

    manifest_path = Path(binding_records["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    expected_limit = int(contract["generation"]["limit"])
    expected = [str(row["qid"]) for row in manifest]
    if len(manifest) != expected_limit or len(set(expected)) != expected_limit:
        errors.append("manifest count or qid uniqueness mismatch")
    reference_fields = sorted(
        {key for row in manifest for key in REFERENCE_KEYS if key in row}
    )
    if reference_fields:
        errors.append(f"reference fields present in generation manifest: {reference_fields}")

    generation_contract_path = run_root / "generation_contract.json"
    generation_contract = (
        json.loads(generation_contract_path.read_text())
        if generation_contract_path.is_file() else {}
    )
    frozen_generation = contract["generation"]
    if generation_contract.get("question_file_sha256") != bindings["manifest"]["sha256"]:
        errors.append("runtime generation contract is not bound to the redacted manifest")
    if generation_contract.get("max_new_tokens") != frozen_generation["max_new_tokens"]:
        errors.append("runtime max_new_tokens differs from the frozen contract")
    if generation_contract.get("keyword_stopping_enabled") is not False:
        errors.append("keyword stopping was not disabled")
    if generation_contract.get("seed") != frozen_generation["seed"]:
        errors.append("runtime seed differs from the frozen contract")

    gates = contract["operational_gates"]
    methods = list(contract["methods"])
    records: list[dict[str, Any]] = []
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        method_dir = run_root / "vqa_rad" / "official_test_oe_image_disjoint_n120" / "open_vqa" / method
        answer_paths = sorted(method_dir.glob("chunk_*.answers.jsonl"))
        method_errors: list[str] = []
        if not answer_paths:
            records.append({
                "method": method,
                "eligible": False,
                "errors": ["answer chunks missing"],
                "answer_paths": [],
            })
            continue
        rows = _load_jsonl_many(answer_paths)
        rows_by_method[method] = rows
        received = [str(row.get("question_id", "")) for row in rows]
        if received != expected:
            method_errors.append("qid order/coverage differs from the frozen manifest")
        if any(any(key in row for key in REFERENCE_KEYS) for row in rows):
            method_errors.append("reference field leaked into raw generation output")
        texts = [str(row.get("text", "")).strip() for row in rows]
        metadata = [row.get("metadata") if isinstance(row.get("metadata"), dict) else {} for row in rows]
        trace_complete = all(
            isinstance(meta.get("generated_token_ids"), list)
            and isinstance(meta.get("raw_generated_token_ids"), list)
            and isinstance(meta.get("terminal_token_ids"), list)
            and meta.get("stop_reason") in {"eos", "pad", "max_new_tokens"}
            and meta.get("keyword_stopping_enabled") is False
            for meta in metadata
        )
        cap_hits = sum(meta.get("stop_reason") == "max_new_tokens" for meta in metadata)
        denominator = len(rows) or 1
        cap_rate = cap_hits / denominator
        nonempty_rate = sum(bool(text) for text in texts) / denominator
        function_only_rate = sum(_function_only(text) for text in texts if text) / denominator
        terminal_required_mask = [
            terminal_required(str(row.get("question", "")), "explicit_sentence_instruction")
            for row in manifest
        ]
        terminal_outcomes = [
            bool(re.search(r"[.!?][\]\)}'\"]*\s*$", text))
            for text, required in zip(texts, terminal_required_mask) if required
        ]
        terminal_rate = (
            sum(terminal_outcomes) / len(terminal_outcomes) if terminal_outcomes else 1.0
        )
        if len(rows) != expected_limit:
            method_errors.append(f"expected {expected_limit} rows, observed {len(rows)}")
        if not trace_complete:
            method_errors.append("generated-token or stop-reason trace is incomplete")
        if cap_rate > gates["maximum_cap_hit_rate"]:
            method_errors.append("cap-hit rate exceeds the frozen ceiling")
        if nonempty_rate < gates["minimum_nonempty_rate"]:
            method_errors.append("nonempty rate is below the frozen floor")
        if function_only_rate > gates["maximum_function_word_only_rate"]:
            method_errors.append("function-word-only rate exceeds the frozen ceiling")
        if terminal_rate < gates["minimum_explicit_sentence_terminal_rate"]:
            method_errors.append("explicit-sentence terminal rate is below the frozen floor")
        records.append({
            "method": method,
            "answer_paths": [str(path.resolve()) for path in answer_paths],
            "answer_sha256": [sha256_file(path) for path in answer_paths],
            "rows": len(rows),
            "exact_qid_alignment": received == expected,
            "reference_fields_absent": not any(any(key in row for key in REFERENCE_KEYS) for row in rows),
            "trace_complete": trace_complete,
            "cap_hits": cap_hits,
            "cap_hit_rate": cap_rate,
            "nonempty_rate": nonempty_rate,
            "function_word_only_rate": function_only_rate,
            "explicit_sentence_required_count": len(terminal_outcomes),
            "explicit_sentence_terminal_rate": terminal_rate,
            "eligible": not method_errors,
            "errors": method_errors,
        })

    identity_pair = contract["method_off_identity_pair"]
    left = rows_by_method.get(identity_pair[0], [])
    right = rows_by_method.get(identity_pair[1], [])
    identity_rate = 0.0
    if left and len(left) == len(right):
        identity_rate = sum(
            a.get("metadata", {}).get("generated_token_ids")
            == b.get("metadata", {}).get("generated_token_ids")
            for a, b in zip(left, right)
        ) / len(left)
    if identity_rate != gates["method_off_token_exact_rate"]:
        errors.append("method-off token identity gate failed")

    reference_rows = rows_by_method.get(contract["reference_method"], [])
    activation = {}
    for method, rows in rows_by_method.items():
        if method == contract["reference_method"] or len(rows) != len(reference_rows):
            continue
        changed = sum(
            row.get("metadata", {}).get("generated_token_ids")
            != ref.get("metadata", {}).get("generated_token_ids")
            for row, ref in zip(rows, reference_rows)
        )
        activation[method] = {
            "changed_sequences": changed,
            "changed_rate": changed / len(rows) if rows else 0.0,
            "clinical_direction_inferred": False,
        }

    all_operational = bool(records) and not errors and all(row["eligible"] for row in records)
    result = {
        "protocol_version": "llava-mitigation-t3-generation-audit-v1",
        "execution_contract": str(execution_contract_path.resolve()),
        "execution_contract_sha256": sha256_file(execution_contract_path),
        "run_root": str(run_root.resolve()),
        "generation_contract": str(generation_contract_path.resolve()),
        "generation_contract_sha256": sha256_file(generation_contract_path) if generation_contract_path.is_file() else None,
        "source_bindings": binding_records,
        "manifest_rows": len(manifest),
        "manifest_unique_images": len({str(row.get("image_sha256")) for row in manifest}),
        "reference_fields_in_manifest": reference_fields,
        "reference_answers_used": False,
        "clinical_labels_used": False,
        "method_records": records,
        "method_off_identity": {
            "pair": identity_pair,
            "generated_token_exact_rate": identity_rate,
            "passed": identity_rate == gates["method_off_token_exact_rate"],
        },
        "activation_vs_greedy": activation,
        "all_operational_gates_passed": all_operational,
        "clinical_efficacy_authorized": False,
        "physician_pack_authorized": all_operational,
        "errors": errors,
        "interpretation": "Operational eligibility and activation do not establish clinical mitigation; blinded claim evaluation remains mandatory.",
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("/home/dbw/ANCHOR"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        run_root=args.run_root,
        execution_contract_path=args.execution_contract,
        repository_root=args.repository_root,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_operational_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
