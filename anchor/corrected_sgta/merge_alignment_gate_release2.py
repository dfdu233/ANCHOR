"""Merge release-2 gates and attach immutable analysis-code identity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import merge_alignment_gate_wave_a as implementation
from corrected_sgta.analysis_provenance_release2 import analysis_code_identity


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    implementation.main()
    output = Path(argument("--output"))
    report = json.loads(output.read_text())
    report["version"] = "sgta-alignment-wave-a-release2-final-v1"
    report["analysis_code_identity"] = analysis_code_identity(Path(__file__).resolve().parents[1])
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
