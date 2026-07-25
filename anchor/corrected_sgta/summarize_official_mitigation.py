#!/usr/bin/env python3
"""Summarize corrected_sgta official mitigation chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL_VERSION = "corrected-sgta-official-mitigation-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("corrected_runs/aaai_medheval_mitigation_full_v1"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root / "llava_med"
    chunks = []
    for eval_file in sorted(root.glob("*/*/*/chunk_*.eval.json")):
        dataset, method, source = eval_file.relative_to(root).parts[:3]
        data = json.loads(eval_file.read_text())
        meta_file = eval_file.with_name(eval_file.name.replace(".eval.json", ".meta.json"))
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        chunks.append({
            "dataset": dataset,
            "method": method,
            "source": source,
            "chunk": eval_file.stem.replace(".eval", ""),
            "n": int(data.get("n", 0)),
            "parseable": int(data.get("parseable", 0)),
            "correct": int(data.get("correct", 0)),
            "fingerprint": meta.get("fingerprint"),
            "status": meta.get("status", "legacy_no_meta"),
            "path": str(eval_file),
        })
    grouped = {}
    for row in chunks:
        key = (row["dataset"], row["method"])
        item = grouped.setdefault(key, {"dataset": row["dataset"], "method": row["method"], "n": 0, "parseable": 0, "correct": 0, "chunks": 0, "sources": set(), "fingerprints": []})
        item["n"] += row["n"]
        item["parseable"] += row["parseable"]
        item["correct"] += row["correct"]
        item["chunks"] += 1
        item["sources"].add(row["source"])
        if row.get("fingerprint"):
            item["fingerprints"].append(row["fingerprint"])
    summary = []
    for item in grouped.values():
        item["sources"] = sorted(item["sources"])
        item["fingerprints"] = sorted(set(item["fingerprints"]))
        item["accuracy_invalid_as_error"] = item["correct"] / item["n"] if item["n"] else None
        item["accuracy_parseable_only"] = item["correct"] / item["parseable"] if item["parseable"] else None
        item["parse_rate"] = item["parseable"] / item["n"] if item["n"] else None
        summary.append(item)
    summary.sort(key=lambda x: (x["dataset"], x["method"]))
    output = {"protocol_version": PROTOCOL_VERSION, "root": str(args.root), "chunks": chunks, "summary": summary}
    out = args.output or (args.root / "llava_med_official_mitigation_summary.json")
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(out), "summary_rows": len(summary), "chunks": len(chunks)}, indent=2))


if __name__ == "__main__":
    main()
