"""Audit OE generation eligibility without using reference answers or clinical labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its",
    "no", "not", "of", "on", "or", "she", "that", "the", "their", "there",
    "they", "this", "to", "was", "we", "were", "with", "yes", "you",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _function_only(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and all(word in FUNCTION_WORDS for word in words)


def audit_qualification(
    run_root: Path,
    *,
    models: list[str],
    arms: list[str],
    expected_rows: int,
    max_cap_rate: float = 0.05,
    min_nonempty_rate: float = 0.95,
    max_function_only_rate: float = 0.01,
) -> dict[str, Any]:
    records = []
    for model in models:
        for arm in arms:
            path = run_root / model / arm / "answers.jsonl"
            errors: list[str] = []
            if not path.is_file():
                records.append(
                    {
                        "model": model,
                        "arm": arm,
                        "path": str(path.resolve()),
                        "complete": False,
                        "eligible": False,
                        "errors": ["answers missing"],
                    }
                )
                continue
            rows = _load_jsonl(path)
            qids = [str(row.get("question_id", "")) for row in rows]
            if len(rows) != expected_rows:
                errors.append(f"expected {expected_rows} rows, observed {len(rows)}")
            if len(qids) != len(set(qids)) or any(not qid for qid in qids):
                errors.append("qid sequence is empty or non-unique")
            texts = [str(row.get("text", "")).strip() for row in rows]
            metadata = [row.get("metadata") or {} for row in rows]
            caps = sum(value.get("hit_max_new_tokens") is True for value in metadata)
            nonempty = sum(bool(text) for text in texts)
            function_only = sum(_function_only(text) for text in texts if text)
            denominator = len(rows) or 1
            cap_rate = caps / denominator
            nonempty_rate = nonempty / denominator
            function_only_rate = function_only / denominator
            if cap_rate > max_cap_rate:
                errors.append(f"cap-hit rate {cap_rate:.6f} exceeds {max_cap_rate:.6f}")
            if nonempty_rate < min_nonempty_rate:
                errors.append(
                    f"nonempty rate {nonempty_rate:.6f} below {min_nonempty_rate:.6f}"
                )
            if function_only_rate > max_function_only_rate:
                errors.append(
                    f"function-only rate {function_only_rate:.6f} exceeds "
                    f"{max_function_only_rate:.6f}"
                )
            records.append(
                {
                    "model": model,
                    "arm": arm,
                    "path": str(path.resolve()),
                    "answers_sha256": sha256_file(path),
                    "rows": len(rows),
                    "complete": len(rows) == expected_rows,
                    "cap_hits": caps,
                    "cap_rate": cap_rate,
                    "nonempty_rate": nonempty_rate,
                    "function_only_rate": function_only_rate,
                    "eligible": not errors,
                    "errors": errors,
                }
            )
    result = {
        "protocol_version": "oe-generation-qualification-v1",
        "reference_answers_used": False,
        "clinical_labels_used": False,
        "selection_signal": "termination, nonempty, and function-word-only traces",
        "run_root": str(run_root.resolve()),
        "models": models,
        "arms": arms,
        "thresholds": {
            "expected_rows": expected_rows,
            "max_cap_rate": max_cap_rate,
            "min_nonempty_rate": min_nonempty_rate,
            "max_function_only_rate": max_function_only_rate,
        },
        "records": records,
        "all_eligible": bool(records) and all(row["eligible"] for row in records),
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--max-cap-rate", type=float, default=0.05)
    parser.add_argument("--min-nonempty-rate", type=float, default=0.95)
    parser.add_argument("--max-function-only-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_qualification(
        args.run_root,
        models=args.model,
        arms=args.arm,
        expected_rows=args.expected_rows,
        max_cap_rate=args.max_cap_rate,
        min_nonempty_rate=args.min_nonempty_rate,
        max_function_only_rate=args.max_function_only_rate,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
