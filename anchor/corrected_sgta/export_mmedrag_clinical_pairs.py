#!/usr/bin/env python3
"""Export radiology report variants for the clinical metric runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    rows = []
    for record in payload["records"]:
        if record["dataset"] not in ("iuxray", "mimic"):
            continue
        if args.variant in record["candidates"]:
            answer = record["candidates"][args.variant]
        elif args.variant == "sequence_anchor":
            answer = record["sequence_anchor"]
        else:
            answer = record["source_neighbor"]["report"]
        rows.append(
            {
                "item_id": f"{record['dataset']}:{record['id']}:{args.variant}",
                "dataset": record["dataset"],
                "ground_truth": record["ground_truth"],
                "model_answer": answer,
            }
        )
    if not rows:
        raise ValueError("no IU-Xray or MIMIC records found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(json.dumps({"variant": args.variant, "n": len(rows)}))


if __name__ == "__main__":
    main()
