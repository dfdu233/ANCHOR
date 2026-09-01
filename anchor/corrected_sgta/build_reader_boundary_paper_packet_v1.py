#!/usr/bin/env python3
"""Build provenance-bound paper tables from formal reader-boundary results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "vindr-reader-boundary-paper-packet-v1"
COMPARISONS = (
    "early_vs_evidence",
    "final_vs_evidence",
    "early_vs_final",
    "early_vs_random",
    "early_vs_direct_maybe",
    "early_vs_confidence",
    "early_vs_entropy",
)


def flatten_comparison(prefix: str, value: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for metric in ("delta_auc", "relative_brier_improvement"):
        for field in ("estimate", "ci_low", "ci_high"):
            output[f"{prefix}_{metric}_{field}"] = value[metric][field]
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty paper table: {path.name}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("paper table rows have inconsistent columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def build_rows(confirmations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pooled, finding_wise = [], []
    for confirmation in confirmations:
        model = confirmation["model_id"]
        for direction, result in confirmation["results"].items():
            row = {
                "model_id": model,
                "direction": direction,
                "n": result["n"],
                "boundary": result["boundary"],
                "representation_controls_pass": result["representation_controls_pass"],
            }
            for comparison in COMPARISONS:
                row.update(flatten_comparison(comparison, result["comparisons"][comparison]))
            pooled.append(row)
            for finding, value in result["finding_wise"].items():
                frow = {
                    "model_id": model,
                    "direction": direction,
                    "finding": finding,
                    "n": value["n"],
                    "boundary": value["boundary"],
                }
                for comparison in COMPARISONS:
                    frow.update(flatten_comparison(comparison, value["comparisons"][comparison]))
                finding_wise.append(frow)
    return pooled, finding_wise


def render_markdown(summary: dict[str, Any], pooled: list[dict[str, Any]]) -> str:
    gate = summary["paper_gate"]
    lines = [
        "# VinDr reader-unanimity layer boundary",
        "",
        "This document is generated from frozen development specifications and held-out confirmation predictions.",
        "It is a mechanism-boundary artifact, not evidence that a mitigation method works.",
        "",
        "| Model | Direction | n | Boundary | Early-final ΔAUROC (95% CI) |",
        "|---|---:|---:|---|---:|",
    ]
    for row in pooled:
        estimate = row["early_vs_final_delta_auc_estimate"]
        low = row["early_vs_final_delta_auc_ci_low"]
        high = row["early_vs_final_delta_auc_ci_high"]
        lines.append(
            f"| {row['model_id']} | {row['direction']} | {row['n']} | "
            f"{row['boundary']} | {estimate:.3f} [{low:.3f}, {high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Fail-closed decision",
            "",
            f"- Qualified finding/direction cells: {gate['qualified_cells']}.",
            f"- Cross-model consistent cells: {gate['consistent_cells']}.",
            f"- Mechanism-boundary paper gate: {gate['strict_majority_consistent']}.",
            f"- Observational Early erasure across all models: {summary['observational_early_erasure_all_models']}.",
            "- Decoder/method authorized: False.",
            f"- Next action: `{summary['next_action']}`.",
            "",
            "Hedging, shorter output, abstention, or a positive observational probe cannot substitute for the preregistered causal test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--confirmation", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    confirmations = [json.loads(path.read_text(encoding="utf-8")) for path in args.confirmation]
    if summary.get("status") != "complete" or summary.get("method_authorized") is not False:
        raise ValueError("invalid fail-closed two-model summary")
    expected = {row["sha256"] for row in summary["provenance"]["inputs"]}
    observed = {sha256_file(path) for path in args.confirmation}
    if expected != observed:
        raise ValueError("summary/confirmation provenance mismatch")
    pooled, finding_wise = build_rows(confirmations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = args.output_dir / "pooled_direction_results.csv"
    finding_path = args.output_dir / "finding_direction_results.csv"
    markdown_path = args.output_dir / "RESULTS.md"
    write_csv(pooled_path, pooled); write_csv(finding_path, finding_wise)
    markdown = render_markdown(summary, pooled)
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text(markdown, encoding="utf-8"); temporary.replace(markdown_path)
    manifest = {
        "version": VERSION,
        "status": "complete",
        "decision": {
            "paper_gate": summary["paper_gate"],
            "method_authorized": False,
            "next_action": summary["next_action"],
        },
        "inputs": {
            "summary": {"path": str(args.summary.resolve()), "sha256": sha256_file(args.summary)},
            "confirmations": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.confirmation
            ],
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (pooled_path, finding_path, markdown_path)
        },
        "code_sha256": sha256_file(Path(__file__)),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
