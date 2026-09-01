#!/usr/bin/env python3
"""Cache-only L0 for frame-covariant laterality transport.

The intervention applies the known radiological display involution only to
left/right attribute words in an existing answer.  All finding words and the
number of claims remain untouched.  This is a narrow mechanism test for
reference-frame hallucination, not a general medical hallucination result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from anchor.corrected_sgta.run_huatuo_binding_conservation_probe_v1 import (
    parse_binding,
)


VERSION = "frame-covariant-decoding-l0-v1"


def swap_laterality(text: str) -> str:
    """Apply the Z2 action left <-> right without touching other text."""

    placeholder = "__FRAME_SIDE_PLACEHOLDER__"
    output = re.sub(r"\bleft\b", placeholder, text, flags=re.IGNORECASE)
    output = re.sub(r"\bright\b", "left", output, flags=re.IGNORECASE)
    return output.replace(placeholder, "right")


def erase_frame_words(text: str) -> str:
    """Canonical text used to verify that only the frame attribute changed."""

    output = re.sub(r"\b(left|right)\b", "<side>", text.lower())
    return re.sub(r"\s+", " ", output).strip()


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    source = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for row in source:
        if row.get("status") != "ok":
            continue
        answer = str(row.get("oe_answer", ""))
        compiled = swap_laterality(answer)
        direct = parse_binding(answer, row["left_finding"], row["right_finding"])
        transported = parse_binding(
            compiled, row["left_finding"], row["right_finding"]
        )
        if direct == "unparsed" or transported == "unparsed":
            continue
        rows.append(
            {
                "case_key": row["case_key"],
                "image_id": row["image_id"],
                "atomic_correct": bool(row.get("atomic_correct", False)),
                "answer": answer,
                "compiled_answer": compiled,
                "direct_parse": direct,
                "compiled_parse": transported,
                "content_preserved": erase_frame_words(answer)
                == erase_frame_words(compiled),
            }
        )

    if not rows:
        raise RuntimeError("no jointly parseable rows")

    direct_correct = np.asarray(
        [row["direct_parse"] == "correct" for row in rows], dtype=float
    )
    compiled_correct = np.asarray(
        [row["compiled_parse"] == "correct" for row in rows], dtype=float
    )
    paired_delta = compiled_correct - direct_correct
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(rows), size=(args.bootstrap_draws, len(rows)))
    bootstrap = paired_delta[indices].mean(axis=1)
    ci = percentile_interval(bootstrap)
    content_preservation = float(np.mean([row["content_preserved"] for row in rows]))
    direct_accuracy = float(direct_correct.mean())
    compiled_accuracy = float(compiled_correct.mean())
    delta = compiled_accuracy - direct_accuracy
    passed = bool(
        len(rows) >= 8
        and delta >= 0.20
        and ci[0] > 0
        and content_preservation == 1.0
    )

    result = {
        "version": VERSION,
        "status": "GO_L0_FRAME_SUBPROBLEM" if passed else "NO_GO_L0_FRAME_SUBPROBLEM",
        "gate_passed": passed,
        "gate": (
            "at least 8 jointly parseable cases; paired laterality accuracy +20pp; "
            "image bootstrap CI lower bound >0; exact non-frame content preservation"
        ),
        "n_source": len(source),
        "n_jointly_parseable": len(rows),
        "direct_accuracy": direct_accuracy,
        "compiled_accuracy": compiled_accuracy,
        "compiled_minus_direct": delta,
        "compiled_minus_direct_ci95": ci,
        "content_preservation_rate": content_preservation,
        "mathematical_operation": (
            "Z2 acts trivially on finding identity and swaps only the left/right attribute"
        ),
        "boundary": (
            "This tests one reference-frame attribute on a Huatuo canary whose prompt names "
            "the findings. It does not establish open-generation or cross-model mitigation."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
