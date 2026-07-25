#!/usr/bin/env python3
"""Protocol-aware summary for current baseline and optimization methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corrected_sgta.aaai_baseline_registry import CE_METHODS, CE_TASKS, MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-dir", required=True, type=Path)
    parser.add_argument("--generative-summary", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    args = parse_args()
    rows = []
    for model in MODELS:
        for task in CE_TASKS:
            summary = load_json(args.ce_dir / f"{model}_{task}.summary.json")
            scat = load_json(args.ce_dir / f"{model}_{task}.scat.json")
            for label, spec in CE_METHODS.items():
                report = summary if spec["source"] == "summary" else scat
                point = report.get("point_accuracy", {}).get(spec["key"], {})
                if not point:
                    rows.append({
                        "protocol": "ce_logits",
                        "model": model,
                        "task": task,
                        "method": label,
                        "method_key": spec["key"],
                        "n": None,
                        "accuracy": None,
                        "scope": spec["scope"],
                        "status": "missing",
                    })
                    continue
                rows.append({
                    "protocol": "ce_logits",
                    "model": model,
                    "task": task,
                    "method": label,
                    "method_key": spec["key"],
                    "n": point.get("n"),
                    "accuracy": point.get("accuracy"),
                    "scope": spec["scope"],
                    "status": "ok",
                })

    if args.generative_summary and args.generative_summary.exists():
        data = load_json(args.generative_summary)
        for item in data.get("summary", []):
            rows.append({
                "protocol": "official_generative",
                "model": "llava",
                "task": item.get("dataset"),
                "method": item.get("method"),
                "n": item.get("n"),
                "accuracy": item.get("accuracy_invalid_as_error"),
                "parseable_accuracy": item.get("accuracy_parseable_only"),
                "parseable": item.get("parseable"),
                "scope": "official LLaVA-Med generative runner; parse-rate sensitive",
                "status": "ok",
            })

    output = {
        "version": "aaai-aligned-baseline-summary-v1",
        "ce_dir": str(args.ce_dir),
        "generative_summary": str(args.generative_summary) if args.generative_summary else None,
        "rules": [
            "CE-logit rows are the main cross-model comparison.",
            "Official generative mitigation rows are supplementary and must not be mixed with CE-logit rows as direct peers.",
            "Rows with different n/scope are valid only within their stated scope.",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    tsv = args.output.with_suffix(".tsv")
    fields = ["protocol", "model", "task", "method", "n", "accuracy", "parseable_accuracy", "scope", "status"]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join("" if row.get(f) is None else str(row.get(f)) for f in fields))
    tsv.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "tsv": str(tsv), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
