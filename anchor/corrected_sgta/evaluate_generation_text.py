#!/usr/bin/env python3
"""Evaluate open-ended/report generations with lightweight text metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corrected_sgta.oe_metrics_v2 import lexical_metrics


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for row in load_jsonl(args.answers):
        prediction = str(row.get("text") or row.get("model_answer") or "").strip()
        reference = str(row.get("gt_ans") or row.get("answer") or row.get("ground_truth") or "").strip()
        if not prediction or not reference:
            continue
        rows.append({"question_id": row.get("question_id"), "metrics": lexical_metrics(prediction, reference)})
    if not rows:
        raise SystemExit("no valid prediction/reference pairs")
    keys = sorted(rows[0]["metrics"])
    summary = {key: sum(float(row["metrics"][key]) for row in rows) / len(rows) for key in keys}
    payload = {"n": len(rows), "metrics": summary, "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), "n": len(rows), "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
