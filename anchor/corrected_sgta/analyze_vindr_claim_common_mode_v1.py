#!/usr/bin/env python3
"""Paired reader-grounded analysis of cross-claim common-mode centering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


VERSION = "vindr-claim-common-mode-analysis-v1"


def metrics(rows: list[dict]) -> dict[str, float]:
    support = np.asarray([row["reader_support"] for row in rows])
    raw = np.asarray([row["raw_score"] for row in rows])
    centered = np.asarray([row["centered_score"] for row in rows])
    clear = np.isin(support, (0.0, 1.0))
    labels = support[clear].astype(int)
    return {
        "raw_auroc": float(roc_auc_score(labels, raw[clear])),
        "centered_auroc": float(roc_auc_score(labels, centered[clear])),
        "auroc_delta": float(
            roc_auc_score(labels, centered[clear]) - roc_auc_score(labels, raw[clear])
        ),
        "raw_spearman": float(spearmanr(support, raw).statistic),
        "centered_spearman": float(spearmanr(support, centered).statistic),
        "spearman_delta": float(
            spearmanr(support, centered).statistic - spearmanr(support, raw).statistic
        ),
    }


def paired_bootstrap(rows: list[dict], draws: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    auc, rho = [], []
    n = len(rows)
    for _ in range(draws):
        sample = [rows[index] for index in rng.integers(0, n, n)]
        try:
            value = metrics(sample)
        except ValueError:
            continue
        auc.append(value["auroc_delta"])
        rho.append(value["spearman_delta"])
    return {
        "auroc_delta_95_ci": [float(x) for x in np.quantile(auc, [0.025, 0.975])],
        "spearman_delta_95_ci": [float(x) for x in np.quantile(rho, [0.025, 0.975])],
        "draws_used": min(len(auc), len(rho)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--reference-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    source_rows = [
        row
        for row in map(json.loads, args.raw.read_text().splitlines())
        if row.get("status") == "ok"
    ]
    gate = json.loads(args.gate.read_text())
    rows = []
    for row in source_rows:
        target = float(row["ontology_scores"][row["finding"]]["yes_minus_no"])
        controls = [
            float(score["yes_minus_no"])
            for finding, score in row["ontology_scores"].items()
            if finding != row["finding"]
        ]
        common = float(np.median(controls))
        rows.append(
            {
                "record_key": row["record_key"],
                "image_id": row["image_id"],
                "finding": row["finding"],
                "reader_support": float(row["reader_support"]),
                "raw_score": target,
                "common_mode": common,
                "centered_score": target - common,
            }
        )
    if len(rows) != len({row["image_id"] for row in rows}):
        raise RuntimeError("canary rows are not image unique")
    overall = metrics(rows)
    bootstrap = paired_bootstrap(rows, args.draws, args.seed)
    by_finding = {}
    noninferior = 0
    for finding in sorted({row["finding"] for row in rows}):
        subset = [row for row in rows if row["finding"] == finding]
        value = metrics(subset)
        by_finding[finding] = value
        noninferior += value["auroc_delta"] >= -0.05

    references = {
        row["record_key"]: row
        for row in map(json.loads, args.reference_metadata.read_text().splitlines())
    }
    differences = []
    for row in rows:
        reference = references[row["record_key"]]["diagnostic_plain_logit_lens"]["28"]
        reference_margin = float(reference["supported"] - reference["refuted"])
        differences.append(row["raw_score"] - reference_margin)
    threshold = gate["primary_gates"]
    checks = {
        "clear_0_vs_3_auroc_delta_at_least": overall["auroc_delta"]
        >= float(threshold["clear_0_vs_3_auroc_delta_at_least"]),
        "clear_0_vs_3_auroc_delta_image_bootstrap_95_ci_lower_above_zero": bootstrap[
            "auroc_delta_95_ci"
        ][0]
        > 0,
        "reader_support_spearman_delta_at_least": overall["spearman_delta"]
        >= float(threshold["reader_support_spearman_delta_at_least"]),
        "reader_support_spearman_delta_image_bootstrap_95_ci_lower_above_zero": bootstrap[
            "spearman_delta_95_ci"
        ][0]
        > 0,
    }
    noninferiority_passed = noninferior >= 6
    passed = all(checks.values()) and noninferiority_passed
    common = np.asarray([row["common_mode"] for row in rows])
    support = np.asarray([row["reader_support"] for row in rows])
    result = {
        "version": VERSION,
        "status": "go" if passed else "no_go",
        "n": len(rows),
        "n_clear": sum(row["reader_support"] in (0.0, 1.0) for row in rows),
        "overall": overall,
        "paired_image_bootstrap": bootstrap,
        "by_finding": by_finding,
        "common_mode_audit": {
            "mean": float(common.mean()),
            "std": float(common.std()),
            "spearman_with_target_reader_support": float(spearmanr(common, support).statistic),
        },
        "native_replay_conformance": {
            "mean_signed_margin_difference": float(np.mean(differences)),
            "max_absolute_margin_difference": float(np.max(np.abs(differences))),
        },
        "primary_gate_checks": checks,
        "finding_noninferiority_count": noninferior,
        "finding_noninferiority_required": 6,
        "finding_noninferiority_passed": noninferiority_passed,
        "decision": {
            "advance_common_mode_feature": passed,
            "authorize_confirmation_run": passed,
            "reason": (
                "Cross-claim centering must improve both clear-case discrimination and the full reader-support ordering."
            ),
        },
        "collision_boundary": (
            "Even a positive result is an image-conditioned contextual-calibration observation, not yet a novel hallucination method."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
