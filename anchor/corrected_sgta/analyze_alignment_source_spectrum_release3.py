"""Same-channel stopping gate for source-spectrum release 3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from corrected_sgta import analyze_alignment_v2 as implementation
from corrected_sgta.cache import iter_successes


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def decoded_state(value):
    return ("invalid",) if value is None else ("label", int(value))


def main() -> None:
    implementation.main()
    output = Path(argument("--output")); cache = Path(argument("--cache"))
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
        original = decoded[0]; views = [decoded[i] for i in matched]
        disagreements.append(any(decoded_state(v) != decoded_state(original) for v in views))
        original_correct.append(original is not None and int(original) == gt)
        oracle_correct.append(
            (original is not None and int(original) == gt)
            or any(v is not None and int(v) == gt for v in views)
        )
    decoded_base = float(np.mean(original_correct)); decoded_oracle = float(np.mean(oracle_correct))
    decoded_disagreement = float(np.mean(disagreements))
    surface_disagreement = report["domain_diagnostics"]["matched_cross_view_prediction_disagreement_rate"]
    surface_headroom = report["matched_style_oracle_headroom_diagnostic_only"]
    decoded_headroom = decoded_oracle - decoded_base
    surface_channel_pass = surface_disagreement > 0 and surface_headroom >= 0.02
    decoded_channel_pass = decoded_disagreement > 0 and decoded_headroom >= 0.02
    fixed_method = "matched_laplacian_anchor_l1"
    fixed_flips = report["flips_vs_original"][fixed_method]
    checks = {
        "same_channel_disagreement_and_oracle_headroom_ge_2pp": bool(
            surface_channel_pass or decoded_channel_pass
        ),
        "fixed_laplacian_l1_rescues_ge_harmful": fixed_flips["rescues"] >= fixed_flips["harmful"],
    }
    report["version"] = "sgta-source-spectrum-analysis-release3-v1"
    report["channel_progression_diagnostics"] = {
        "surface": {
            "disagreement_rate": surface_disagreement,
            "oracle_headroom": surface_headroom,
            "pass": surface_channel_pass,
        },
        "decoded_invalid_as_error": {
            "original_accuracy": decoded_base,
            "source_style_oracle_accuracy": decoded_oracle,
            "oracle_headroom": decoded_headroom,
            "disagreement_rate": decoded_disagreement,
            "pass": decoded_channel_pass,
        },
    }
    report["legacy_accuracy_selected_gate_ignored"] = report.pop("gate")
    report["preregistered_prediction_gate"] = {
        "fixed_method": fixed_method, "checks": checks, "pass": all(checks.values())
    }
    report["method_note"] = (
        "A prediction channel must itself show both disagreement and >=2pp oracle headroom. "
        "Progression output is frozen to anchored Laplacian lambda=1; label-selected summaries "
        "are diagnostic only. Invalid decoded labels are one explicit error state."
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2)); temporary.replace(output)


if __name__ == "__main__":
    main()

