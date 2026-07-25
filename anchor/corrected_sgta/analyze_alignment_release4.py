"""Correctly described diagnostic analyzer for same-modality shuffle control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import analyze_alignment_v2 as implementation


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    implementation.main()
    output = Path(argument("--output"))
    report = json.loads(output.read_text())
    report["version"] = "sgta-alignment-analysis-release4-v1"
    report["method_note"] = (
        "Matched and shuffled-control graphs are analyzed separately. The control is an "
        "actual same-modality transformation from another source, evaluated against the "
        "unchanged target feature center. Wave A fixes one matched and one paired shuffled "
        "view, so the output budgets are equal and cannot collapse to a two-source set "
        "permutation. Laplacian variants are diagnostic only."
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
