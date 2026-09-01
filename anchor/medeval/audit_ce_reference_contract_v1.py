#!/usr/bin/env python3
"""Prove that every frozen CE reference is parseable under the paper rules."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from anchor.corrected_sgta.evaluate_medheval_answers import evaluate_rows
from anchor.medeval.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = ("cxr_vishal", "knowledge_mimic_ce", "slake_fine_grained")
    records = []
    for dataset in datasets:
        path = args.input_root / f"{dataset}.json"
        rows = json.loads(path.read_text())
        probes = [
            {
                "qid": row["qid"],
                "question": row["question"],
                "question_type": row["question_type"],
                "gt_ans": row["answer"],
                "text": row["answer"],
                "img_name": row["img_name"],
            }
            for row in rows
        ]
        result = evaluate_rows(probes)
        passed = (
            result["invalid_ground_truth"] == 0
            and result["parse_rate"] == 1.0
            and result["accuracy_invalid_as_error"] == 1.0
        )
        records.append(
            {
                "dataset": dataset,
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "rows": len(rows),
                "question_types": dict(sorted(Counter(row["question_type"] for row in rows).items())),
                "invalid_ground_truth": result["invalid_ground_truth"],
                "reference_self_parse_rate": result["parse_rate"],
                "reference_self_accuracy": result["accuracy_invalid_as_error"],
                "passed": passed,
            }
        )
    output = {
        "protocol": "ce-reference-contract-audit-v1",
        "rule": "binary/ternary explicit state; choice label-or-option normalization; short answer normalized exact",
        "datasets": records,
        "passed": all(row["passed"] for row in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
