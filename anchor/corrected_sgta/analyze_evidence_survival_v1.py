#!/usr/bin/env python3
"""Analyze lesion-vs-background evidence survival curves without threshold repair."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "reader-evidence-survival-analysis-v1"
_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:  # NumPy < 2.0
    _TRAPEZOID = np.trapz


def summarize_record(row: dict) -> dict:
    baseline = row["baseline_coordinates"]
    baseline_logits = row["baseline_logits"]
    baseline_positive_commitment = float(baseline_logits["supported"]) - float(
        baseline_logits["undetermined"]
    )
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for value in row["interventions"]:
        grouped[(str(value["region"]), float(value["dose"]))].append(value)
    doses = sorted({dose for _, dose in grouped})
    curves = {}
    for region in ("roi", "background"):
        commitment = [baseline_positive_commitment]
        polarity = [float(baseline["polarity"])]
        for dose in doses:
            values = grouped[(region, dose)]
            commitment.append(
                float(
                    np.mean(
                        [
                            float(value["logits"]["supported"])
                            - float(value["logits"]["undetermined"])
                            for value in values
                        ]
                    )
                )
            )
            polarity.append(
                float(np.mean([value["coordinates"]["polarity"] for value in values]))
            )
        x = np.asarray([0.0, *doses], dtype=float)
        curves[region] = {
            "positive_commitment": commitment,
            "polarity": polarity,
            "commitment_curve_auc": float(_TRAPEZOID(commitment, x)),
            "polarity_curve_auc": float(_TRAPEZOID(polarity, x)),
            "critical_commitment_dose": next(
                (dose for dose, value in zip(doses, commitment[1:]) if value <= 0), None
            ),
            "critical_polarity_dose": next(
                (dose for dose, value in zip(doses, polarity[1:]) if value <= 0), None
            ),
        }
    return {
        "record_key": row["record_key"],
        "image_id": row["image_id"],
        "finding": row["finding"],
        "positive_votes": int(row["positive_votes"]),
        "roi_tokens": int(row["roi_tokens"]),
        "baseline_positive_commitment": baseline_positive_commitment,
        "baseline_polarity": float(baseline["polarity"]),
        # Survival of a definite positive claim is defined only when the
        # baseline prefers both positive polarity and definite commitment.
        "directionally_admitted": (
            float(baseline["polarity"]) > 0 and baseline_positive_commitment > 0
        ),
        "doses": doses,
        "curves": curves,
        "roi_minus_background_survival_auc": (
            curves["roi"]["commitment_curve_auc"]
            - curves["background"]["commitment_curve_auc"]
        ),
    }


def adjusted_unanimity_coefficient(rows: list[dict]) -> float:
    target = np.asarray([row["curves"]["roi"]["commitment_curve_auc"] for row in rows])
    findings = sorted({row["finding"] for row in rows})
    design = np.column_stack(
        [
            np.ones(len(rows)),
            [float(row["positive_votes"] == 3) for row in rows],
            [row["baseline_positive_commitment"] for row in rows],
            np.log1p([row["roi_tokens"] for row in rows]),
            *[[float(row["finding"] == finding) for row in rows] for finding in findings[1:]],
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coefficients[1])


def cluster_bootstrap(rows: list[dict], draws: int, seed: int) -> dict:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    images = sorted(by_image)
    estimate = adjusted_unanimity_coefficient(rows)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(images, size=len(images), replace=True)
        batch = [row for image in sampled for row in by_image[str(image)]]
        if {row["positive_votes"] for row in batch} != {2, 3}:
            continue
        values.append(adjusted_unanimity_coefficient(batch))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "valid_draws": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    skipped = [row for row in raw if row.get("status") != "ok"]
    rows = [summarize_record(row) for row in raw if row.get("status") == "ok"]
    admitted = [row for row in rows if row["directionally_admitted"]]
    if len(admitted) < 20:
        result = {
            "version": VERSION,
            "status": "perception_limited",
            "n": len(rows),
            "directionally_admitted_n": len(admitted),
            "skipped_n": len(skipped),
            "method_branch_authorized": False,
        }
    else:
        pooled = cluster_bootstrap(admitted, args.bootstrap_draws, args.seed)
        by_finding = {}
        for finding in sorted({row["finding"] for row in admitted}):
            subset = [row for row in admitted if row["finding"] == finding]
            by_finding[finding] = {
                "n": len(subset),
                "vote_bins": sorted({int(row["positive_votes"]) for row in subset}),
                "unanimity_survival_coefficient": adjusted_unanimity_coefficient(subset),
                "mean_roi_minus_background_survival_auc": float(
                    np.mean([row["roi_minus_background_survival_auc"] for row in subset])
                ),
            }
        # Higher ROI survival in 3/3 is the unique redundancy prediction.
        # ROI minus background is expected to be negative because lesion
        # ablation should reduce commitment more than background ablation.
        two_finding_contrast_ready = len(by_finding) >= 2 and all(
            value["vote_bins"] == [2, 3] for value in by_finding.values()
        )
        gate = {
            "pooled_unanimity_survival_ci_above_zero": float(pooled["ci_low"]) > 0,
            "two_findings_with_both_vote_bins": two_finding_contrast_ready,
            "both_findings_unanimity_survival_positive": two_finding_contrast_ready and all(
                value["unanimity_survival_coefficient"] > 0 for value in by_finding.values()
            ),
            "roi_ablation_stronger_than_background_both_findings": two_finding_contrast_ready and all(
                value["mean_roi_minus_background_survival_auc"] < 0 for value in by_finding.values()
            ),
        }
        gate["evidence_survival_pilot_passed"] = all(gate.values())
        result = {
            "version": VERSION,
            "status": "complete",
            "n": len(rows),
            "directionally_admitted_n": len(admitted),
            "skipped_n": len(skipped),
            "pooled_adjusted_unanimity_survival": pooled,
            "by_finding": by_finding,
            "gate": gate,
            "method_branch_authorized": False,
            "authorization_note": "pilot cannot authorize mitigation; dev replication and causal controls remain required",
            "records": rows,
        }
    result["provenance"] = {
        "raw": str(args.raw.resolve()),
        "raw_sha256": sha256_file(args.raw),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "bootstrap_draws": int(args.bootstrap_draws),
        "seed": int(args.seed),
        "command": " ".join(sys.argv),
    }
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
