"""Freeze Wave A gates to a prespecified uniform matched-view consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes
from corrected_sgta.methods import softmax_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--diagnostic-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def accuracy(predictions: list[int | None], labels: list[int]) -> dict:
    valid = [(pred, gt) for pred, gt in zip(predictions, labels) if pred is not None]
    return {
        "n_total": len(labels),
        "n_valid": len(valid),
        "accuracy": None
        if not valid
        else float(np.mean([int(pred == gt) for pred, gt in valid])),
    }


def majority(values: list[int | None]) -> int | None:
    valid = [int(value) for value in values if value is not None]
    if not valid:
        return None
    counts = np.bincount(valid)
    return int(np.flatnonzero(counts == counts.max())[0])


def main() -> None:
    args = parse_args()
    diagnostic = json.loads(args.diagnostic_analysis.read_text())
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    rows = list(iter_successes(args.cache, metadata["fingerprint"]))
    labels = [int(row["gt_index"]) for row in rows]
    nll_original: list[int | None] = []
    nll_matched_uniform: list[int | None] = []
    decode_original: list[int | None] = []
    decode_matched_majority: list[int | None] = []
    decode_disagreement = []
    for row in rows:
        roles = row["style_roles"]
        matched = [0] + [index for index, role in enumerate(roles) if role == "matched"]
        nll = row.get("style_sequence_nll")
        if nll and not any(value is None for value in nll):
            values = np.asarray(nll, dtype=np.float64)
            nll_original.append(int(np.argmin(values[0])))
            nll_matched_uniform.append(
                int(np.argmax(softmax_np(-values[matched]).mean(axis=0)))
            )
        else:
            nll_original.append(None)
            nll_matched_uniform.append(None)
        decoded = row.get("style_decoded_prediction")
        if decoded is None:
            decode_original.append(None)
            decode_matched_majority.append(None)
        else:
            decode_original.append(decoded[0])
            decode_matched_majority.append(majority([decoded[index] for index in matched]))
            valid = [decoded[index] for index in matched if decoded[index] is not None]
            if valid:
                decode_disagreement.append(len(set(valid)) > 1)

    point = diagnostic["point_accuracy"]
    flips = diagnostic["flips_vs_original"]
    domain = diagnostic["domain_diagnostics"]
    primary = "matched_uniform_mean"
    wrong = "wrong_control_uniform_mean"
    primary_flip = flips[primary]
    checks = {
        "n_is_256": len(rows) == 256,
        "matched_median_relative_closure_ge_20pct": (
            domain["matched_relative_closure_median"] is not None
            and domain["matched_relative_closure_median"] >= 0.20
        ),
        "matched_closure_gt_wrong_control": (
            domain["matched_relative_closure_median"] is not None
            and domain["wrong_control_relative_closure_median"] is not None
            and domain["matched_relative_closure_median"]
            > domain["wrong_control_relative_closure_median"]
        ),
        "matched_style_oracle_headroom_ge_2pp": (
            diagnostic["matched_style_oracle_headroom_diagnostic_only"] >= 0.02
        ),
        "fixed_uniform_rescues_ge_harmful": primary_flip["rescues"] >= primary_flip["harmful"],
        "fixed_uniform_matched_accuracy_gt_wrong_control": (
            point[primary]["accuracy"] > point[wrong]["accuracy"]
        ),
        "actual_decode_present": all(value is not None for value in decode_original),
        "ssim_and_clinical_structure_audit_present": False,
    }
    output = {
        "version": "sgta-alignment-frozen-report-v3",
        "preregistration": "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V2.md",
        "source_cache": str(args.cache),
        "fingerprint": metadata["fingerprint"],
        "n": len(rows),
        "primary_method": primary,
        "primary_surface_accuracy": point[primary],
        "original_surface_accuracy": point["original"],
        "wrong_control_surface_accuracy": point[wrong],
        "primary_flips_vs_original": primary_flip,
        "matched_minus_wrong_control_accuracy": (
            point[primary]["accuracy"] - point[wrong]["accuracy"]
        ),
        "evidence_channels": {
            "accepted_label_nll_original": accuracy(nll_original, labels),
            "accepted_label_nll_matched_uniform": accuracy(nll_matched_uniform, labels),
            "actual_decode_original": accuracy(decode_original, labels),
            "actual_decode_matched_majority": accuracy(decode_matched_majority, labels),
            "actual_decode_cross_matched_disagreement_rate": None
            if not decode_disagreement
            else float(np.mean(decode_disagreement)),
        },
        "domain_diagnostics": domain,
        "style_oracle_diagnostic_only": {
            "accuracy": diagnostic["matched_style_oracle_accuracy_diagnostic_only"],
            "headroom": diagnostic["matched_style_oracle_headroom_diagnostic_only"],
        },
        "laplacian_diagnostic_only": {
            name: value for name, value in point.items() if "laplacian" in name
        },
        "gate": {
            "stage": "formal" if len(rows) == 256 else "smoke_diagnostic_only",
            "checks": checks,
            "pass": len(rows) == 256 and all(checks.values()),
            "note": (
                "The fixed primary method is never selected by evaluation accuracy. "
                "The structure-audit check remains false until the separate SSIM/clinical audit is merged."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
