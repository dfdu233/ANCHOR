"""Fail-closed paired analysis for RULE visual DC-PMI outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from corrected_sgta.train_rule_dg_adapter import rule_label


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty visual DC-PMI result")
    names = tuple(rows[0].get("variants", {}))
    if not names:
        raise ValueError("no variants")
    seen = set()
    correct = {name: 0 for name in names}
    predictions = {name: {} for name in names}
    for row in rows:
        qid = str(row.get("question_id"))
        if qid in seen:
            raise ValueError(f"duplicate qid: {qid}")
        seen.add(qid)
        if row.get("status") != "ok":
            raise ValueError(f"non-success row: {qid}")
        if tuple(row.get("variants", {})) != names:
            raise ValueError(f"variant mismatch: {qid}")
        gt = rule_label(row.get("gt_answer"))
        for name in names:
            prediction = row["variants"][name].get("prediction")
            if prediction not in {"Yes", "No"}:
                raise ValueError(f"invalid prediction for {qid}/{name}: {prediction!r}")
            predictions[name][qid] = prediction
            correct[name] += int(prediction == gt)
    n = len(rows)
    base_name = "base" if "base" in names else names[0]
    comparisons = {}
    for name in names:
        if name == base_name:
            continue
        rescues = harms = changed = 0
        for row in rows:
            qid = str(row["question_id"])
            gt = rule_label(row["gt_answer"])
            left, right = predictions[base_name][qid], predictions[name][qid]
            changed += int(left != right)
            rescues += int(left != gt and right == gt)
            harms += int(left == gt and right != gt)
        comparisons[name] = {
            "changed": changed,
            "rescues": rescues,
            "harms": harms,
            "net": rescues - harms,
            "delta_pp": 100.0 * (correct[name] - correct[base_name]) / n,
        }
    return {
        "n": n,
        "primary_metric": "RULE binary accuracy from fixed full-sequence visual DC-PMI argmax",
        "variants": {
            name: {"correct": correct[name], "accuracy": correct[name] / n}
            for name in names
        },
        "base_variant": base_name,
        "paired_vs_base": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    result = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
