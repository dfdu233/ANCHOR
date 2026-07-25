#!/usr/bin/env python3
"""Aggregate strict phase gates without averaging away unsafe task failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corrected_sgta.protocol_v2 import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION

CE_METHOD_VERSION = "matched-center-sgta-v4"
CONFGEN_METHOD_VERSION = "matched-center-sgta-confgen-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=("auto", "ce", "confgen"), default="auto")
    return parser.parse_args()


def formal_provenance(report: dict, expected_method: str) -> tuple[bool, str]:
    checks = {
        "protocol": report.get("protocol_version") == PROTOCOL_VERSION,
        "schema": report.get("cache_schema_version") == CACHE_SCHEMA_VERSION,
        "evidence_status": report.get("evidence_status") == "formal_v5.4",
        "method": report.get("method_version") == expected_method,
    }
    source = Path(str(report.get("source_cache", "")))
    metadata_path = source.with_suffix(source.suffix + ".meta.json")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        checks["source_metadata"] = False
    else:
        checks.update(
            source_metadata=True,
            source_fingerprint=metadata.get("fingerprint") == report.get("fingerprint"),
            source_schema=metadata.get("cache_schema_version") == CACHE_SCHEMA_VERSION,
            source_protocol=metadata.get("protocol_version") == PROTOCOL_VERSION,
            matched_center=metadata.get("config", {}).get("center_policy") == "matched",
        )
    failed = [key for key, value in checks.items() if not value]
    return not failed, ",".join(failed) if failed else "formal_v5.4_verified"


def ce_report(directory: Path) -> dict:
    files = sorted(directory.glob("*.sgta_v4.json"))
    rows = []
    for path in files:
        report = json.loads(path.read_text())
        provenance_ok, provenance_reason = formal_provenance(
            report, CE_METHOD_VERSION
        )
        model = path.name.split("_")[0]
        structure = report.get("structure", {})
        for name, channel in report.get("channels", {}).items():
            test = channel["test"]
            rows.append(
                {
                    "file": str(path),
                    "model": model,
                    "channel": name,
                    "delta_accuracy": test["delta_accuracy"],
                    "oracle_headroom": test["oracle_headroom"],
                    "aurc_relative_reduction": channel.get("aurc_relative_reduction"),
                    "channel_gate": channel["gate"],
                    "formal_provenance": provenance_ok,
                    "provenance_reason": provenance_reason,
                    "structure_available": structure.get("available", False),
                    "structure_pass_rate": structure.get("pass_rate"),
                    "passed": (
                        provenance_ok
                        and channel["gate"].get("passed", False)
                        and structure.get("available", False)
                        and structure.get("pass_rate") == 1.0
                    ),
                }
            )
    models = sorted({row["model"] for row in rows})
    model_gate = {
        model: any(
            row["passed"] and row["delta_accuracy"] > 0
            for row in rows if row["model"] == model
        )
        for model in models
    }
    no_large_regression = all(row["delta_accuracy"] >= -0.01 for row in rows)
    passed = bool(rows) and set(models) >= {"hulu", "llava"} and all(model_gate.values()) and no_large_regression
    return {
        "kind": "ce_wave2",
        "rows": rows,
        "model_has_positive_passing_task": model_gate,
        "no_channel_regression_over_1pp": no_large_regression,
        "passed": passed,
    }


def confgen_report(directory: Path) -> dict:
    files = sorted(directory.glob("*.confgen_v2.json"))
    rows = []
    for path in files:
        report = json.loads(path.read_text())
        provenance_ok, provenance_reason = formal_provenance(
            report, CONFGEN_METHOD_VERSION
        )
        comparisons = report.get("comparison", {})
        for gamma, value in comparisons.items():
            rows.append(
                {
                    "file": str(path),
                    "model": path.name.split("_")[0],
                    "task": report.get("task"),
                    "gamma": gamma,
                    "paper_status": report.get("admissibility", {}).get("paper_status"),
                    "formal_provenance": provenance_ok,
                    "provenance_reason": provenance_reason,
                    "gate": value["gate"],
                    "passed": provenance_ok
                    and value["gate"].get("passed", False)
                    and report.get("admissibility", {}).get("paper_status") == "eligible",
                }
            )
    required = {(model, task) for model in ("hulu", "llava") for task in ("knowledge", "report")}
    observed = {(row["model"], row["task"]) for row in rows if row["passed"]}
    return {
        "kind": "confgen_wave3",
        "rows": rows,
        "required_model_tasks": sorted([list(value) for value in required]),
        "passing_model_tasks": sorted([list(value) for value in observed]),
        "passed": bool(rows) and required <= observed,
    }


def main() -> None:
    args = parse_args()
    kind = args.kind
    if kind == "auto":
        kind = "confgen" if list(args.input.glob("*.confgen_v2.json")) else "ce"
    report = confgen_report(args.input) if kind == "confgen" else ce_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
