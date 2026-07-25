"""Analyze token alignment while retaining the frozen alignment metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import analyze_feature_transport as implementation


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    implementation.main()
    output = Path(argument("--output"))
    report = json.loads(output.read_text())
    report["version"] = "sgta-source-guided-token-alignment-analysis-v1"
    report["method_note"] = (
        "Projected visual tokens receive the same pooled source-center residual as "
        "uniform transport, allocated by fixed token-center cosine attention. The "
        "target source, beta, temperature, and cap are label-free and preregistered."
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
