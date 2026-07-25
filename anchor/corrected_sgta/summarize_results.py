#!/usr/bin/env python3
"""Aggregate corrected CE, SCA-T, and ConfGen JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-dir", required=True, type=Path)
    parser.add_argument("--oe-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def identity(path: Path, suffix: str) -> tuple[str, str]:
    stem = path.name.removesuffix(suffix)
    model, dataset = stem.split("_", 1)
    return model, dataset


def main() -> None:
    args = parse_args()
    records = []
    for path in sorted(args.ce_dir.glob("*.summary.json")):
        model, dataset = identity(path, ".summary.json")
        report = json.loads(path.read_text())
        for method, point in report.get("point_accuracy", {}).items():
            conformal = report.get("conformal", {}).get(method, {})
            lac = conformal.get("lac", {}).get("0.1", {})
            aps = conformal.get("aps", {}).get("0.1", {})
            records.append(
                {
                    "family": "ce",
                    "model": model,
                    "dataset": dataset,
                    "method": method,
                    "n": point.get("n"),
                    "accuracy": point.get("accuracy"),
                    "coverage_lac_0.1": lac.get("coverage"),
                    "set_size_lac_0.1": lac.get("average_set_size"),
                    "coverage_aps_0.1": aps.get("coverage"),
                    "set_size_aps_0.1": aps.get("average_set_size"),
                }
            )
    for path in sorted(args.ce_dir.glob("*.scat.json")):
        model, dataset = identity(path, ".scat.json")
        report = json.loads(path.read_text())
        for method, point in report.get("point_accuracy", {}).items():
            conformal = report.get("conformal", {}).get(method, {})
            lac = conformal.get("lac", {}).get("0.1", {})
            aps = conformal.get("aps", {}).get("0.1", {})
            records.append(
                {
                    "family": "scat_yes_no",
                    "model": model,
                    "dataset": dataset,
                    "method": method,
                    "n": point.get("n"),
                    "accuracy": point.get("accuracy"),
                    "coverage_lac_0.1": lac.get("coverage"),
                    "set_size_lac_0.1": lac.get("average_set_size"),
                    "coverage_aps_0.1": aps.get("coverage"),
                    "set_size_aps_0.1": aps.get("average_set_size"),
                }
            )
    if args.oe_dir and args.oe_dir.is_dir():
        for path in sorted(args.oe_dir.glob("*.confgen.json")):
            model, dataset = identity(path, ".confgen.json")
            report = json.loads(path.read_text())
            baseline = report.get("greedy_baseline_test", {})
            records.append(
                {
                    "family": "oe",
                    "model": model,
                    "dataset": dataset,
                    "method": "greedy_baseline",
                    "n": baseline.get("n"),
                    "rouge_l": baseline.get("metrics", {}).get("rouge_l"),
                    "token_f1": baseline.get("metrics", {}).get("token_f1"),
                    "proxy_admissibility": baseline.get("lexical_admissibility_rate"),
                }
            )
            for method, values in report.get("methods", {}).items():
                gamma = values.get("gamma", {}).get("0.9", {})
                records.append(
                    {
                        "family": "oe",
                        "model": model,
                        "dataset": dataset,
                        "method": method,
                        "n": values.get("n_test"),
                        "coverage_gamma_0.9": gamma.get("empirical_coverage"),
                        "set_size_gamma_0.9": gamma.get("average_set_size"),
                        "rouge_l": gamma.get(
                            "confidence_reduced_output_metrics", {}
                        ).get("rouge_l"),
                        "token_f1": gamma.get(
                            "confidence_reduced_output_metrics", {}
                        ).get("token_f1"),
                        "proxy_admissibility": gamma.get(
                            "confidence_reduced_admissibility_rate"
                        ),
                    }
                )
    output = {
        "ce_dir": str(args.ce_dir),
        "oe_dir": str(args.oe_dir) if args.oe_dir else None,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    markdown = args.output.with_suffix(".md")
    lines = [
        "# Corrected MedHEval experiment summary",
        "",
        "| family | model | dataset | method | n | accuracy/coverage |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in records:
        primary = row.get("accuracy", row.get("coverage_gamma_0.9"))
        display = "-" if primary is None else f"{100 * primary:.2f}%"
        lines.append(
            f"| {row['family']} | {row['model']} | {row['dataset']} | "
            f"{row['method']} | {row.get('n', '-')} | {display} |"
        )
    markdown.write_text("\n".join(lines) + "\n")
    print(f"Saved {args.output} and {markdown}")


if __name__ == "__main__":
    main()
