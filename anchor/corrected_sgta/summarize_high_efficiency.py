#!/usr/bin/env python3
"""Summarize source-separated high-efficiency experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("corrected_runs/high_efficiency"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for metric_file in sorted(args.root.rglob("*.metrics.json")):
        data = read_json(metric_file)
        parts = metric_file.relative_to(args.root).parts
        if len(parts) < 5:
            continue
        source, dataset, task, method = parts[-5:-1]
        item = {
            "source": source,
            "dataset": dataset,
            "task": task,
            "method": method,
            "metric_file": str(metric_file),
            "n": int(data.get("n", 0)),
        }
        item.update(data.get("metrics", {}))
        rows.append(item)
    state_rows = []
    for state_file in sorted(args.root.rglob("queue_state.jsonl")):
        for line in state_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["state_file"] = str(state_file)
                state_rows.append(row)
    output = {"metric_rows": rows, "state_rows": state_rows}
    out = args.output or args.root / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(out), "metric_rows": len(rows), "state_rows": len(state_rows)}, indent=2))


if __name__ == "__main__":
    main()
