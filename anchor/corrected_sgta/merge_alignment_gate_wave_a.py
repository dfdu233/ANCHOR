"""Merge the independent structure audit into the frozen Wave-A report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-report", required=True, type=Path)
    parser.add_argument("--structure-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.frozen_report.read_text())
    structure = json.loads(args.structure_audit.read_text())
    if report["fingerprint"] != structure["fingerprint"]:
        raise RuntimeError("structure audit/cache fingerprint mismatch")
    report["version"] = "sgta-alignment-wave-a-final-report-v1"
    report["structure_audit"] = structure
    checks = report["gate"]["checks"]
    checks["ssim_and_clinical_structure_audit_present"] = bool(
        structure["formal_matched_structure_pass"]
    )
    report["gate"]["pass"] = report["n"] == 256 and all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)
    print(json.dumps({"gate": report["gate"], "structure": structure["matched"]}, indent=2))


if __name__ == "__main__":
    main()
