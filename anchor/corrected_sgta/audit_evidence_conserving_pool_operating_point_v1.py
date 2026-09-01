#!/usr/bin/env python3
"""Independent operating-point audit of evidence-conserving patch pooling.

This script deliberately imports the frozen v1 feature construction instead
of reimplementing it.  It asks whether e-mixture improves ranking or merely
moves the positive operating point relative to final-margin and raw-max scores.
Thresholds are finding-specific and frozen on development vote-0 examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from anchor.corrected_sgta.analyze_evidence_conserving_pool_v1 import (
    PARTITIONS,
    build_rows,
)
from anchor.corrected_sgta.analyze_sparse_patch_scan_v1 import (
    BOOTSTRAP_DRAWS,
    FINDINGS,
    SEED,
    hidden_rows,
    patch_artifact,
)


VERSION = "evidence-conserving-pool-operating-point-audit-v1"
FPR_TARGET = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def score_names() -> list[str]:
    names = ["final_margin"]
    for count in PARTITIONS:
        names.extend((f"raw_max_{count}", f"e_mix_{count}"))
    return names


def arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "finding": np.asarray([row["finding"] for row in rows]),
        "label": np.asarray([row["label"] for row in rows], dtype=int),
    }
    for name in score_names():
        result[name] = np.asarray([row[name] for row in rows], dtype=float)
    return result


def freeze_thresholds(dev: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for score in score_names():
        thresholds[score] = {}
        for finding in FINDINGS:
            mask = (dev["finding"] == finding) & (dev["label"] == 0)
            # Linear interpolation is frozen explicitly. With 60 distinct
            # null values and a strict '>' decision, three exceed the 95th
            # percentile; ties can only make the achieved development FPR
            # more conservative.
            thresholds[score][finding] = float(
                np.quantile(dev[score][mask], 1.0 - FPR_TARGET, method="linear")
            )
    return thresholds


def metric(
    data: dict[str, np.ndarray], score: str,
    thresholds: dict[str, dict[str, float]], indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if indices is None:
        indices = np.arange(len(data["label"]))
    finding = data["finding"][indices]
    label = data["label"][indices]
    values = data[score][indices]
    by_finding: dict[str, dict[str, float]] = {}
    aucs, fprs, recalls = [], [], []
    for name in FINDINGS:
        mask = finding == name
        y, value = label[mask], values[mask]
        prediction = value > thresholds[score][name]
        auc = float(roc_auc_score(y, value))
        fpr = float(prediction[y == 0].mean())
        recall = float(prediction[y == 1].mean())
        aucs.append(auc)
        fprs.append(fpr)
        recalls.append(recall)
        by_finding[name] = {"auroc": auc, "fpr": fpr, "recall": recall}
    return {
        "macro_auroc": float(np.mean(aucs)),
        "macro_fpr": float(np.mean(fprs)),
        "macro_recall": float(np.mean(recalls)),
        "by_finding": by_finding,
    }


def bootstrap(
    data: dict[str, np.ndarray], thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    cells = [
        np.flatnonzero((data["finding"] == finding) & (data["label"] == label))
        for finding in FINDINGS for label in (0, 1)
    ]
    rng = np.random.default_rng(SEED)
    names = score_names()
    draws: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in ("macro_auroc", "macro_fpr", "macro_recall")}
        for name in names
    }
    deltas: dict[str, dict[str, list[float]]] = {}
    comparison_pairs: list[tuple[str, str]] = []
    for count in PARTITIONS:
        comparison_pairs.extend(
            (
                (f"e_mix_{count}", "final_margin"),
                (f"e_mix_{count}", f"raw_max_{count}"),
            )
        )
    for left, right in comparison_pairs:
        deltas[f"{left}_minus_{right}"] = {
            key: [] for key in ("macro_auroc", "macro_fpr", "macro_recall")
        }

    for _ in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        current = {name: metric(data, name, thresholds, indices) for name in names}
        for name in names:
            for key in draws[name]:
                draws[name][key].append(current[name][key])
        for left, right in comparison_pairs:
            destination = deltas[f"{left}_minus_{right}"]
            for key in destination:
                destination[key].append(current[left][key] - current[right][key])

    def summarize(values: list[float]) -> list[float]:
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]

    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "unit": "paired resampling within finding x clear label cells; identical draw for every score",
        "score_ci95": {
            name: {f"{key}_ci95": summarize(values) for key, values in metrics.items()}
            for name, metrics in draws.items()
        },
        "paired_delta_ci95": {
            name: {f"{key}_delta_ci95": summarize(values) for key, values in metrics.items()}
            for name, metrics in deltas.items()
        },
    }


def classify(metrics: dict[str, dict[str, Any]], boot: dict[str, Any]) -> dict[str, Any]:
    decisions = {}
    any_ranking_gain = False
    for count in PARTITIONS:
        e_name, raw_name = f"e_mix_{count}", f"raw_max_{count}"
        e = metrics[e_name]
        final = metrics["final_margin"]
        e_final_ci = boot["paired_delta_ci95"][f"{e_name}_minus_final_margin"]
        ranking_gain = (
            e["macro_auroc"] > final["macro_auroc"]
            and e_final_ci["macro_auroc_delta_ci95"][0] > 0
        )
        any_ranking_gain = any_ranking_gain or ranking_gain
        conservative_vs_final = (
            e["macro_fpr"] < final["macro_fpr"]
            and e["macro_recall"] < final["macro_recall"]
        )
        decisions[str(count)] = {
            "ranking_gain_over_final_margin_ci_excludes_zero": bool(ranking_gain),
            "lower_fpr_and_lower_recall_than_final_margin": bool(conservative_vs_final),
            "e_mix_minus_final_macro_auroc": e["macro_auroc"] - final["macro_auroc"],
            "e_mix_minus_raw_max_macro_auroc": e["macro_auroc"] - metrics[raw_name]["macro_auroc"],
        }
    return {
        "by_partition": decisions,
        "any_partition_has_confirmed_ranking_gain_over_final_margin": bool(any_ranking_gain),
        "verdict": (
            "incremental_discrimination_signal"
            if any_ranking_gain
            else "no_incremental_discrimination; differences are compatible with score calibration/operating-point behavior"
        ),
        "claim_boundary": (
            "Without a paired AUROC gain over final margin, lower FPR alone is not evidence recovery; "
            "it can be obtained by a more conservative positive decision rule."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-hidden", type=Path, required=True)
    parser.add_argument("--confirmation-hidden", type=Path, required=True)
    parser.add_argument("--patch-scores", type=Path, required=True)
    parser.add_argument("--v1-source", type=Path, required=True)
    parser.add_argument("--v1-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    dev_rows = hidden_rows(args.development_hidden)
    test_rows = hidden_rows(args.confirmation_hidden)
    scores, image_index, geometry = patch_artifact(args.patch_scores)
    dev_rows, test_rows = build_rows(dev_rows, test_rows, scores, image_index, geometry["side"])
    dev, test = arrays(dev_rows), arrays(test_rows)
    thresholds = freeze_thresholds(dev)
    development_metrics = {name: metric(dev, name, thresholds) for name in score_names()}
    confirmation_metrics = {name: metric(test, name, thresholds) for name in score_names()}
    boot = bootstrap(test, thresholds)

    result = {
        "version": VERSION,
        "status": "complete",
        "question": "Does evidence-conserving pooling improve discrimination, or only change the operating point?",
        "sample": {
            "development_claims": len(dev_rows),
            "development_images": len({row["image_id"] for row in dev_rows}),
            "confirmation_claims": len(test_rows),
            "confirmation_images": len({row["image_id"] for row in test_rows}),
            "findings": list(FINDINGS),
        },
        "protocol": {
            "target_development_fpr": FPR_TARGET,
            "threshold": "per-finding 95th percentile of development vote-0 score; linear quantile; strict >",
            "partitions": list(PARTITIONS),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
        },
        "development": development_metrics,
        "confirmation": confirmation_metrics,
        "bootstrap": boot,
        "assessment": classify(confirmation_metrics, boot),
        "provenance": {
            "source_sha256": sha256(Path(__file__)),
            "v1_source": str(args.v1_source),
            "v1_source_sha256": sha256(args.v1_source),
            "v1_artifact": str(args.v1_artifact),
            "v1_artifact_sha256": sha256(args.v1_artifact),
        },
        "command": " ".join(sys.argv),
    }
    atomic_json(args.output, result)
    print(json.dumps({"assessment": result["assessment"], "confirmation": confirmation_metrics}, indent=2))


if __name__ == "__main__":
    main()
