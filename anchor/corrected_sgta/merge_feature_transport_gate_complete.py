"""Freeze a feature-transport report with a transitive analysis fingerprint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import merge_alignment_gate_wave_a as implementation
from corrected_sgta.transport_provenance_complete import complete_transport_identity


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    implementation.main()
    output = Path(argument("--output"))
    report = json.loads(output.read_text())
    report["version"] = "sgta-feature-transport-complete-report-v1"
    report["analysis_code_identity"] = complete_transport_identity(
        Path(__file__).resolve().parents[1]
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
