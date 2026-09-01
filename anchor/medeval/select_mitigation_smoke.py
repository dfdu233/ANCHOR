#!/usr/bin/env python3
"""Select only mitigation methods whose smoke run passed plumbing checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.smoke_root / "queue_state.jsonl"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    latest = {}
    for line in state_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["method"]] = row
    methods = []
    rejected = {}
    for method, row in sorted(latest.items()):
        audit = row.get("output_audit") or {}
        reasons = []
        if row.get("status") not in {"done", "skipped"}:
            reasons.append(f"status={row.get('status')}")
        if row.get("evidence_grade") != "A":
            reasons.append(f"evidence_grade={row.get('evidence_grade')}")
        if not audit.get("aligned"):
            reasons.append("output_not_aligned")
        reasons.extend(audit.get("degenerate_reasons") or [])
        if reasons:
            rejected[method] = reasons
        else:
            methods.append(method)
    payload = {
        "smoke_root": str(args.smoke_root.resolve()),
        "methods": methods,
        "rejected": rejected,
        "rule": (
            "status done/skipped + exact run fingerprint grade A + sample alignment "
            "+ no output degeneration"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(" ".join(methods))


if __name__ == "__main__":
    main()
