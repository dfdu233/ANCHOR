"""Frozen no-label-selection analysis for source-spectrum release 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from corrected_sgta import analyze_alignment_v2 as implementation
from corrected_sgta.cache import iter_successes


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    implementation.main()
    output = Path(argument("--output"))
    cache = Path(argument("--cache"))
    report = json.loads(output.read_text())
    metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    rows = list(iter_successes(cache, metadata["fingerprint"]))
    disagreements, original_correct, oracle_correct = [], [], []
    for row in rows:
        decoded = row.get("style_decoded_prediction")
        matched = [i for i, role in enumerate(row.get("style_roles", [])) if role == "matched"]
        gt = int(row["gt_index"])
        if decoded is None or not matched:
            disagreements.append(False); original_correct.append(False); oracle_correct.append(False)
            continue
        original = decoded[0]
        views = [decoded[i] for i in matched]
        disagreements.append(original is not None and any(v is not None and v != original for v in views))
        original_correct.append(original is not None and int(original) == gt)
        oracle_correct.append(
            (original is not None and int(original) == gt)
            or any(v is not None and int(v) == gt for v in views)
        )
    decoded_base = float(np.mean(original_correct))
    decoded_oracle = float(np.mean(oracle_correct))
    decoded_disagreement = float(np.mean(disagreements))
    fixed_method = "matched_laplacian_anchor_l1"
    fixed_flips = report["flips_vs_original"][fixed_method]
    checks = {
        "surface_or_decoded_disagreement_nonzero": bool(
            report["domain_diagnostics"]["matched_cross_view_prediction_disagreement_rate"] > 0
            or decoded_disagreement > 0
        ),
        "surface_or_decoded_oracle_headroom_ge_2pp": bool(
            report["matched_style_oracle_headroom_diagnostic_only"] >= 0.02
            or decoded_oracle - decoded_base >= 0.02
        ),
        "fixed_laplacian_l1_rescues_ge_harmful": fixed_flips["rescues"] >= fixed_flips["harmful"],
    }
    report["version"] = "sgta-source-spectrum-analysis-release2-v1"
    report["decoded_diagnostics"] = {
        "original_accuracy_invalid_as_error": decoded_base,
        "source_style_oracle_accuracy_invalid_as_error": decoded_oracle,
        "source_style_oracle_headroom": decoded_oracle - decoded_base,
        "original_vs_matched_disagreement_rate": decoded_disagreement,
    }
    report["legacy_accuracy_selected_gate_ignored"] = report.pop("gate")
    report["preregistered_prediction_gate"] = {
        "fixed_method": fixed_method, "checks": checks, "pass": all(checks.values())
    }
    report["method_note"] = (
        "The matched view is selected without labels. Progression is frozen to anchored "
        "Laplacian lambda=1; all accuracy-selected summaries are diagnostic only."
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()

