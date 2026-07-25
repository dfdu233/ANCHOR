"""Diagnostic analysis with feature-transport-specific method semantics."""

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
    report["version"] = "sgta-feature-transport-analysis-v1"
    report["method_note"] = (
        "The matched view shifts frozen visual tokens toward the nearest same-modality "
        "source-center direction. The paired control shifts the same tokens toward the "
        "other X-ray source at identical beta. Pixels and model weights are unchanged. "
        "The fixed primary rule is original-plus-one-matched uniform mean; Laplacian "
        "and style oracle remain diagnostic only."
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
