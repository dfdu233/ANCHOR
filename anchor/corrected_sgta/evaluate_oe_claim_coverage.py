#!/usr/bin/env python3
"""Evaluate structured OE/report claims with frozen matched-coverage semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from corrected_sgta.clinical_claims import (
    VERSION as CLAIM_VERSION,
    evaluate_oe_methods_matched_coverage,
)


VERSION = "missing-third-state-oe-coverage-v4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--maximum-natural-coverage-drop", type=float, default=0.01)
    parser.add_argument(
        "--plumbing-only",
        action="store_true",
        help="Allow missing reader/physician provenance; result is inadmissible as paper evidence",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = evaluate_oe_methods_matched_coverage(
        rows,
        args.baseline_method,
        target_count=args.target_count,
        maximum_natural_coverage_drop=args.maximum_natural_coverage_drop,
        require_reference_provenance=not args.plumbing_only,
    )
    config = {
        "version": VERSION,
        "claim_contract_version": CLAIM_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "baseline_method": args.baseline_method,
        "target_count": args.target_count,
        "maximum_natural_coverage_drop": args.maximum_natural_coverage_drop,
        "plumbing_only": args.plumbing_only,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    payload = {"config": config, "result": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
