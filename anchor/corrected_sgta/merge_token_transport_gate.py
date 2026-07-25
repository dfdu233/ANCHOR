"""Attach source-guided token-alignment provenance to the final smoke report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import merge_alignment_gate_wave_a as implementation
from corrected_sgta.token_transport_provenance import token_transport_identity


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    implementation.main()
    output = Path(argument("--output"))
    report = json.loads(output.read_text())
    report["version"] = "sgta-source-guided-token-alignment-complete-report-v1"
    report["analysis_code_identity"] = token_transport_identity(
        Path(__file__).resolve().parents[1]
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
