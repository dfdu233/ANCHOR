#!/usr/bin/env python3
"""Exit successfully only for a current, qualified paper-score artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.medeval.audit_baseline_matrix_execution_v1 import (
    qualification_state,
    score_binding_failure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("mixed_ce", "open_vqa", "report_generation"), required=True
    )
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    qualification, qualification_reason = qualification_state(
        args.qualification, args.expected, args.task
    )
    score_reason = score_binding_failure(args.score, args.task)
    result = {
        "version": "score-artifact-binding-validation-v1",
        "score": str(args.score.resolve()),
        "qualification": str(args.qualification.resolve()),
        "qualification_status": qualification,
        "qualification_reason": qualification_reason,
        "score_reason": score_reason,
        "passed": qualification == "passed" and score_reason is None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
