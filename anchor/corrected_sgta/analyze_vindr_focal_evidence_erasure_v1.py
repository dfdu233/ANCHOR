#!/usr/bin/env python3
"""Cluster-bootstrap analysis for the VinDr focal evidence-erasure pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


VERSION = "vindr-focal-evidence-erasure-analysis-v1"


def summarize(rows: list[dict], draws: int, seed: int) -> dict:
    original = np.asarray([r["scores"]["original"]["yes_minus_no"] for r in rows])
    erased = np.asarray([r["scores"]["lesion_erased"]["yes_minus_no"] for r in rows])
    mirror = np.asarray([r["scores"]["mirror_erased"]["yes_minus_no"] for r in rows])
    image_ids = np.asarray([r["image_id"] for r in rows])
    clusters = np.unique(image_ids)
    lesion_drop = original - erased
    mirror_drop = original - mirror
    premium = mirror - erased
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, 3), dtype=float)
    for draw in range(draws):
        selected = rng.choice(clusters, len(clusters), replace=True)
        indices = np.concatenate([np.where(image_ids == image_id)[0] for image_id in selected])
        boot[draw] = [
            lesion_drop[indices].mean(),
            mirror_drop[indices].mean(),
            premium[indices].mean(),
        ]

    def estimate(values: np.ndarray, column: int) -> dict:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "cluster_bootstrap_95_ci": [
                float(np.quantile(boot[:, column], 0.025)),
                float(np.quantile(boot[:, column], 0.975)),
            ],
            "fraction_positive": float(np.mean(values > 0)),
            "fraction_zero": float(np.mean(values == 0)),
        }

    areas = np.asarray([r["mask_area_fraction"] for r in rows])
    area_r, area_p = spearmanr(areas, lesion_drop)
    admitted = original > 0
    return {
        "n_claims": len(rows),
        "n_images": int(len(clusters)),
        "directional_admission": {
            "criterion": "original Yes-minus-No > 0 on a reader-supported positive claim",
            "n": int(admitted.sum()),
            "rate": float(admitted.mean()),
            "mean_original_margin": float(original.mean()),
        },
        "lesion_box_erasure_drop": estimate(lesion_drop, 0),
        "mirror_control_erasure_drop": estimate(mirror_drop, 1),
        "region_specificity_premium": estimate(premium, 2),
        "mask_area_vs_lesion_drop_spearman": {"rho": float(area_r), "p": float(area_p)},
        "admitted_subset_descriptive_only": (
            {
                "n": int(admitted.sum()),
                "lesion_drop_mean": float(lesion_drop[admitted].mean()),
                "mirror_drop_mean": float(mirror_drop[admitted].mean()),
                "premium_mean": float(premium[admitted].mean()),
            }
            if admitted.any()
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError("raw run is empty or incomplete")
    overall = summarize(rows, args.bootstrap_draws, 20260806)
    findings = {
        finding: summarize(
            [row for row in rows if row["finding"] == finding],
            args.bootstrap_draws,
            20260806 + index + 1,
        )
        for index, finding in enumerate(sorted({row["finding"] for row in rows}))
    }
    lesion_ci = overall["lesion_box_erasure_drop"]["cluster_bootstrap_95_ci"]
    finding_ci = {
        finding: values["lesion_box_erasure_drop"]["cluster_bootstrap_95_ci"]
        for finding, values in findings.items()
    }
    result = {
        "version": VERSION,
        "status": "heterogeneous_no_general_causal_admission",
        "estimand": {
            "lesion_drop": "score(original)-score(reader-box-erased)",
            "mirror_drop": "score(original)-score(horizontal-mirror-box-erased)",
            "specificity_premium": "lesion_drop-mirror_drop = score(mirror-erased)-score(reader-box-erased)",
        },
        "overall": overall,
        "by_finding": findings,
        "decision": {
            "general_lesion_sensitivity_ci_lower_above_zero": lesion_ci[0] > 0,
            "every_finding_lesion_sensitivity_ci_lower_above_zero": all(ci[0] > 0 for ci in finding_ci.values()),
            "advance_generic_E_feature": False,
            "reason": (
                "The aggregate lesion-erasure drop CI crosses zero and Nodule/Mass moves in the wrong mean direction. "
                "A positive reader-box-vs-mirror premium is insufficient because mirror erasure itself raises the claim score."
            ),
        },
        "interpretation_boundary": (
            "The experiment has independent positive localization truth but no clinically normal counterfactual, "
            "negative-claim lesion target, patient identifiers, or hallucination labels. It tests directional evidence sensitivity only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
