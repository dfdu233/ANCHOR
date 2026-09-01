#!/usr/bin/env python3
"""Summarize reference-grounded pairwise medical judge records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


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
    valid = [row for row in rows if row.get("judge")]
    first = valid[0] if valid else {}
    score_keys = (
        "a_factuality",
        "b_factuality",
        "a_hallucination",
        "b_hallucination",
    )
    payload = {
        "version": "reference-grounded-pairwise-judge-summary-v1",
        "model": (
            first.get("provenance", {}).get("response_model")
            or "deepseek-v4-flash"
        ),
        "name_a": first.get("name_a"),
        "name_b": first.get("name_b"),
        "n": len(rows),
        "n_success": len(valid),
        "preferences": dict(
            Counter(row["judge"]["preference"] for row in valid)
        ),
        "mean_scores": {
            key: (
                float(np.mean([row["judge"][key] for row in valid]))
                if valid
                else None
            )
            for key in score_keys
        },
        "limitations": [
            "judge cannot see the image",
            "reference reports may omit visible findings",
            "a prior three-pair order swap in this protocol was only 2/3 consistent",
        ],
    }
    if valid:
        scores = payload["mean_scores"]
        payload["delta_b_minus_a"] = {
            "factuality": scores["b_factuality"] - scores["a_factuality"],
            "hallucination": (
                scores["b_hallucination"] - scores["a_hallucination"]
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
