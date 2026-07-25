"""Combine frozen prediction and complete structure gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text()); structure = json.loads(args.structure.read_text())
    if analysis["fingerprint"] != structure["fingerprint"]:
        raise RuntimeError("analysis/structure fingerprint mismatch")
    checks = {
        **analysis["preregistered_prediction_gate"]["checks"],
        "one_paired_view_per_row": structure["matched"]["n"] == analysis["n"] and structure["wrong_control"]["n"] == analysis["n"],
        "all_matched_views_pass_complete_structure_audit": bool(structure["formal_matched_structure_pass"]),
    }
    passed = all(checks.values())
    report = {
        "version": "sgta-source-spectrum-final-adjudication-release2-v1",
        "fingerprint": analysis["fingerprint"], "n": analysis["n"],
        "analysis": str(args.analysis.resolve()), "structure": str(args.structure.resolve()),
        "checks": checks, "pass": passed,
        "decision": "allow_256" if passed else "stop_pixel_frequency_route",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2)); temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

