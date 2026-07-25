#!/usr/bin/env python3
"""Validate provenance/readiness for locally reproducible Report-OE metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_SCORERS = ("chexbert", "radgraph", "ratescore")
REQUIRED_DIRECTIONS = (
    "normal_no_finding_direction",
    "abnormal_finding_direction",
    "critical_contradiction_direction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-validation", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    scorers = manifest.get("scorers") or {}
    directions = manifest.get("direction_checks") or {}
    weights_hash = str(manifest.get("weights_sha256", ""))
    n_validation = int(manifest.get("n_validation", 0))
    checks = {
        "all_scorer_versions_present": all(str(scorers.get(key, "")).strip() for key in REQUIRED_SCORERS),
        "weights_hash_valid": len(weights_hash) == 64 and all(c in "0123456789abcdef" for c in weights_hash.lower()),
        "direction_checks_pass": all(directions.get(key) is True for key in REQUIRED_DIRECTIONS),
        "validation_size_sufficient": n_validation >= args.min_validation,
    }
    report = {
        "validation_type": "report_metric_validation",
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "n_validation": n_validation,
        "scorers": scorers,
        "direction_checks": directions,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
