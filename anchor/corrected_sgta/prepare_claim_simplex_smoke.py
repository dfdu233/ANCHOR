#!/usr/bin/env python3
"""Convert an existing balanced RULE probe into a claim-simplex smoke manifest.

The resulting labels are report-derived and therefore grade-C diagnostics, not
formal clinical truth.  They are useful only for testing whether polarity and
commitment are empirically distinct coordinates before VinDr reader votes are
available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "claim-simplex-smoke-manifest-v2"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(row: dict[str, Any], seed: int) -> str:
    payload = f"{seed}:{row.get('id')}:{row.get('image')}:{row.get('prompt')}"
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, Any]]] = {"yes": [], "no": []}
    for row in load_jsonl(args.input):
        evaluation = row.get("ce_evaluation") or {}
        label = evaluation.get("ground_truth_explicit")
        image = Path(str(row.get("image", "")))
        if row.get("status") == "ok" and label in groups and image.is_file():
            groups[str(label)].append(row)
    selected = []
    for label in ("yes", "no"):
        ordered = sorted(groups[label], key=lambda row: stable_key(row, args.seed))
        if len(ordered) < args.per_label:
            raise ValueError(f"only {len(ordered)} usable {label} rows")
        chosen = ordered[: args.per_label]
        for index, row in enumerate(chosen):
            row["_claim_simplex_split"] = (
                "dev" if index < args.per_label // 2 else "test"
            )
        selected.extend(chosen)

    records = []
    for row in sorted(selected, key=lambda value: stable_key(value, args.seed)):
        label = str(row["ce_evaluation"]["ground_truth_explicit"])
        question_id = str(row["id"])
        question = str(row["prompt"]).strip()
        records.append(
            {
                "version": VERSION,
                "image_id": f"{row.get('domain', 'rule')}:{question_id}",
                "finding": f"rule_claim_{question_id}",
                "question": question
                + " Answer with exactly one word: Yes, No, or Maybe.",
                "image_path": str(Path(str(row["image"])).resolve()),
                "positive_votes": 3 if label == "yes" else 0,
                "reader_count": 3,
                "reader_support": 1.0 if label == "yes" else 0.0,
                "reader_state": "supported" if label == "yes" else "refuted",
                "reference_source": "single_report_derived_rule_label",
                "evidence_grade": "C",
                "formal_reference": False,
                "experiment_split": str(row["_claim_simplex_split"]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "version": VERSION,
                "rows": len(records),
                "yes": args.per_label,
                "no": args.per_label,
                "formal_reference": False,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
