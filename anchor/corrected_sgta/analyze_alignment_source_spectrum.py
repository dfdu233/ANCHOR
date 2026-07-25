"""Analysis wrapper with accurate source-spectrum method semantics."""

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
    report["version"] = "sgta-source-spectrum-analysis-v1"
    report["preregistered_stop_pass"] = bool(
        report["domain_diagnostics"]["matched_cross_view_prediction_disagreement_rate"] > 0
        and report["matched_style_oracle_headroom_diagnostic_only"] >= 0.02
        and report["flips_vs_original"][report["best_matched_method_diagnostic_only"]]["rescues"]
        >= report["flips_vs_original"][report["best_matched_method_diagnostic_only"]]["harmful"]
    )
    report["method_note"] = (
        "Original and one label-free-selected PubMed source-aligned view are evaluated. "
        "The transform blends every non-DC Fourier-amplitude bin and preserves target phase "
        "and DC. The paired control uses an IU-Xray or MIMIC-CXR amplitude at identical alpha. "
        "Laplacian variants are diagnostic until the preregistered stopping gate passes."
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()

