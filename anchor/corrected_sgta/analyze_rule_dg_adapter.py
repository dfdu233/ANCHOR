"""Paired, fail-closed analysis for base and DG-adapted RULE outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from corrected_sgta.train_rule_dg_adapter import rule_label


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty paired result")
    base_correct = adapted_correct = rescues = harms = changed = 0
    parse_empty = {"base": 0, "adapted": 0}
    repetition = {"base": 0, "adapted": 0}
    details = []
    seen = set()
    for row in rows:
        qid = str(row.get("question_id"))
        if qid in seen:
            raise ValueError(f"duplicate qid: {qid}")
        seen.add(qid)
        if row.get("status") != "ok":
            raise ValueError(f"non-success row: {qid}")
        gt_raw = row.get("gt_answer")
        if gt_raw is None or not str(gt_raw).strip():
            raise ValueError(f"missing ground truth: {qid}")
        gt = rule_label(gt_raw)
        base_text = str(row.get("base_text", "")).strip()
        adapted_text = str(row.get("adapted_text", "")).strip()
        parse_empty["base"] += int(not base_text)
        parse_empty["adapted"] += int(not adapted_text)
        base_tokens, adapted_tokens = base_text.lower().split(), adapted_text.lower().split()
        repetition["base"] += int(len(base_tokens) >= 8 and len(set(base_tokens)) <= 2)
        repetition["adapted"] += int(len(adapted_tokens) >= 8 and len(set(adapted_tokens)) <= 2)
        base, adapted = rule_label(base_text), rule_label(adapted_text)
        left, right = base == gt, adapted == gt
        base_correct += int(left)
        adapted_correct += int(right)
        changed += int(base != adapted)
        rescues += int((not left) and right)
        harms += int(left and (not right))
        if base != adapted:
            details.append({
                "question_id": row.get("question_id"), "gt": gt, "base": base, "adapted": adapted,
                "base_text": base_text, "adapted_text": adapted_text,
            })
    n = len(rows)
    return {
        "n": n, "base": {"correct": base_correct, "accuracy": base_correct / n},
        "adapted": {"correct": adapted_correct, "accuracy": adapted_correct / n},
        "delta_pp": 100.0 * (adapted_correct - base_correct) / n,
        "paired_flips": {"changed": changed, "rescues": rescues, "harms": harms, "net": rescues - harms},
        "safety": {"empty": parse_empty, "degenerate_repetition": repetition},
        "changed_details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
