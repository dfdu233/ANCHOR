#!/usr/bin/env python3
"""Non-destructively supplement the completion audit with newer control evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "iclr-evaluation-progress-supplement-v1"


def build(*, base_audit: Path, internal_audit: Path, baseline_audit: Path) -> dict[str, Any]:
    base = json.loads(base_audit.read_text())
    internal = json.loads(internal_audit.read_text())
    baseline = json.loads(baseline_audit.read_text())
    if base.get("paper_ready") is not False or base.get("submission_claim_authorized") is not False:
        raise ValueError("supplement cannot consume a promoted base audit")
    if baseline.get("version") != "baseline-coverage-audit-v3":
        raise ValueError("baseline v3 audit required")
    internal_summary = internal.get("summary", {})
    all_controls = [
        "temperature_length_controls",
        "self_consistency",
        "calibrated_abstention",
    ]
    t2_pass = list(internal_summary.get("t2_pass", []))
    t2_complete = t2_pass == all_controls
    if internal_summary.get("t3_pass") or internal_summary.get("full_pass"):
        raise ValueError("T3/full controls require a new primary completion audit, not a supplement")
    requirements = [dict(row) for row in base["requirements"]]
    r6 = next(row for row in requirements if row["id"] == "R6")
    r6["status"] = (
        "t2_engineering_complete_clinical_metrics_pending"
        if t2_complete
        else "source_complete_execution_partial"
    )
    r6["evidence"] = (
        f"Frozen internal controls passing T2: {t2_pass}; missing: "
        f"{internal_summary.get('t2_missing', [])}; failed: {internal_summary.get('t2_failed', [])}. "
        "T2 is functional qualification only; no T3 clinical efficacy or no-exchange claim is authorized."
    )
    result = {
        "version": VERSION,
        "paper_ready": False,
        "submission_claim_authorized": False,
        "base_completion_audit": {"path": str(base_audit.resolve()), "sha256": sha256_file(base_audit)},
        "internal_control_audit": {"path": str(internal_audit.resolve()), "sha256": sha256_file(internal_audit)},
        "baseline_coverage_audit": {"path": str(baseline_audit.resolve()), "sha256": sha256_file(baseline_audit)},
        "requirements": requirements,
        "evaluation_progress": {
            "t2_pass": t2_pass,
            "t2_missing": internal_summary.get("t2_missing", []),
            "t2_failed": internal_summary.get("t2_failed", []),
            "t3_pass": [],
            "full_pass": [],
            "paper_efficacy_table_authorized": False,
        },
        "interpretation": (
            "This supplement updates evaluation engineering progress only. The authoritative "
            "negative mechanism gates and external clinical-review requirements remain unchanged."
        ),
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", required=True, type=Path)
    parser.add_argument("--internal-audit", required=True, type=Path)
    parser.add_argument("--baseline-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(
        base_audit=args.base_audit,
        internal_audit=args.internal_audit,
        baseline_audit=args.baseline_audit,
    )
    atomic_json(args.output, result)
    print(json.dumps({"fingerprint": result["fingerprint"], **result["evaluation_progress"]}, indent=2))


if __name__ == "__main__":
    main()
