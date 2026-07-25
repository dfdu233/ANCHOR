#!/usr/bin/env python3
"""Analyze cache-risk features for source-center trade-off diagnostics.

The input is produced by ``corrected_sgta.cache_risk_gate``. This script keeps
the analysis cache-only: it ranks individual risk scores for rescue/harmful
prediction and evaluates a leave-one-slice-out logistic gate using calibration
rows from the remaining slices only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - optional dependency fallback.
    LogisticRegression = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank-tsv", required=True, type=Path)
    parser.add_argument("--loso-jsonl", required=True, type=Path)
    parser.add_argument("--min-train-events", type=int, default=20)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty risk feature file: {path}")
    return rows


def mean_bool(rows: list[dict], key: str) -> float | None:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else None


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(values)
    values = values[ok]
    truth = truth[ok]
    pos = values[truth == 1]
    neg = values[truth == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for value in pos:
        wins += float((value > neg).sum()) + 0.5 * float((value == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def safe_float(value) -> float:
    if value is None:
        return math.nan
    return float(value)


def feature_keys(rows: list[dict]) -> list[str]:
    keys = set()
    for row in rows:
        keys.update(row.get("risk_scores", {}))
    return sorted(keys)


def feature_matrix(rows: list[dict], keys: list[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(
        [[safe_float(row.get("risk_scores", {}).get(key)) for key in keys] for row in rows],
        dtype=np.float64,
    )
    missing = ~np.isfinite(raw)
    if medians is None:
        medians = np.nanmedian(raw, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = raw.copy()
    for col in range(filled.shape[1]):
        filled[missing[:, col], col] = medians[col]
    return filled.astype(np.float64), medians


def standardize_fit(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def utility(row: dict) -> int:
    return int(bool(row["rescue"])) - int(bool(row["harmful"]))


def accuracy(rows: list[dict], actions: list[bool] | None = None) -> float | None:
    if not rows:
        return None
    correct = []
    for index, row in enumerate(rows):
        use_candidate = actions[index] if actions is not None else True
        correct.append(bool(row["candidate_correct"]) if use_candidate else bool(row["baseline_correct"]))
    return float(np.mean(correct))


def evaluate_threshold(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    actions = [float(score) >= threshold for score in scores]
    routed = [row for row, action in zip(rows, actions) if action]
    return {
        "threshold": float(threshold),
        "accuracy": accuracy(rows, actions),
        "delta_vs_baseline": None if not rows else accuracy(rows, actions) - accuracy(rows, [False] * len(rows)),
        "routed": len(routed),
        "coverage": float(len(routed) / len(rows)) if rows else None,
        "rescues": int(sum(row["rescue"] for row in routed)),
        "harmful": int(sum(row["harmful"] for row in routed)),
    }


def best_threshold(rows: list[dict], scores: np.ndarray) -> dict | None:
    if len(rows) == 0:
        return None
    finite = sorted({float(score) for score in scores if np.isfinite(score)})
    if not finite:
        return None
    thresholds = [finite[0] - 1e-9] + [(a + b) / 2.0 for a, b in zip(finite, finite[1:])] + [finite[-1] + 1e-9]
    # Include abstain-to-baseline. It is the conservative tie breaker.
    candidates = [evaluate_threshold(rows, scores, threshold) for threshold in thresholds]
    candidates.append(
        {
            "threshold": float("inf"),
            "accuracy": accuracy(rows, [False] * len(rows)),
            "delta_vs_baseline": 0.0,
            "routed": 0,
            "coverage": 0.0,
            "rescues": 0,
            "harmful": 0,
        }
    )
    return max(candidates, key=lambda item: (item["accuracy"], -item["routed"], item["rescues"] - item["harmful"]))


def single_feature_stats(rows: list[dict], keys: list[str]) -> list[dict]:
    out = []
    selectors = sorted({row["selector"] for row in rows})
    for selector in selectors:
        subset = [row for row in rows if row["selector"] == selector]
        for key in keys:
            values = [safe_float(row.get("risk_scores", {}).get(key)) for row in subset]
            rescue = [bool(row["rescue"]) for row in subset]
            harmful = [bool(row["harmful"]) for row in subset]
            changed = [bool(row["rescue"]) or bool(row["harmful"]) for row in subset]
            resc_values = [value for value, row in zip(values, subset) if np.isfinite(value) and row["rescue"]]
            harm_values = [value for value, row in zip(values, subset) if np.isfinite(value) and row["harmful"]]
            neutral_values = [
                value
                for value, row in zip(values, subset)
                if np.isfinite(value) and not row["rescue"] and not row["harmful"]
            ]
            out.append(
                {
                    "selector": selector,
                    "feature": key,
                    "n_finite": int(sum(np.isfinite(values))),
                    "auroc_rescue_vs_rest": auroc(values, rescue),
                    "auroc_harmful_vs_rest": auroc(values, harmful),
                    "auroc_changed_vs_neutral": auroc(values, changed),
                    "mean_rescue": float(np.mean(resc_values)) if resc_values else None,
                    "mean_harmful": float(np.mean(harm_values)) if harm_values else None,
                    "mean_neutral": float(np.mean(neutral_values)) if neutral_values else None,
                }
            )
    return out


def loso_gate(rows: list[dict], keys: list[str], min_train_events: int) -> list[dict]:
    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is required for LOSO logistic gate")
    outputs = []
    selectors = sorted({row["selector"] for row in rows})
    slices = sorted({row["slice"] for row in rows})
    for selector in selectors:
        selector_rows = [row for row in rows if row["selector"] == selector]
        for heldout in slices:
            train = [
                row
                for row in selector_rows
                if row["slice"] != heldout and row["split"] == "calibration" and (row["rescue"] or row["harmful"])
            ]
            threshold_train = [
                row
                for row in selector_rows
                if row["slice"] != heldout and row["split"] == "calibration"
            ]
            test = [row for row in selector_rows if row["slice"] == heldout and row["split"] == "test"]
            y = np.asarray([1 if row["rescue"] else 0 for row in train], dtype=np.int64)
            base_test_acc = accuracy(test, [False] * len(test))
            cand_test_acc = accuracy(test, [True] * len(test))
            result = {
                "selector": selector,
                "heldout_slice": heldout,
                "n_train_events": len(train),
                "n_threshold_train": len(threshold_train),
                "n_test": len(test),
                "baseline_test_accuracy": base_test_acc,
                "candidate_test_accuracy": cand_test_acc,
                "candidate_test_delta": None if base_test_acc is None or cand_test_acc is None else cand_test_acc - base_test_acc,
            }
            if len(train) < min_train_events or len(set(y.tolist())) < 2:
                result["status"] = "insufficient_train_events"
                result["gated_test_accuracy"] = base_test_acc
                result["gated_test_delta"] = 0.0
                result["routed"] = 0
                result["coverage"] = 0.0
                outputs.append(result)
                continue
            train_x, medians = feature_matrix(train, keys)
            threshold_x, _ = feature_matrix(threshold_train, keys, medians)
            test_x, _ = feature_matrix(test, keys, medians)
            mean, std = standardize_fit(train_x)
            train_x = standardize_apply(train_x, mean, std)
            threshold_x = standardize_apply(threshold_x, mean, std)
            test_x = standardize_apply(test_x, mean, std)
            model = LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=42)
            model.fit(train_x, y)
            threshold_scores = model.predict_proba(threshold_x)[:, 1]
            test_scores = model.predict_proba(test_x)[:, 1]
            gate = best_threshold(threshold_train, threshold_scores)
            if gate is None:
                result["status"] = "no_threshold"
                result["gated_test_accuracy"] = base_test_acc
                result["gated_test_delta"] = 0.0
                result["routed"] = 0
                result["coverage"] = 0.0
            else:
                test_gate = evaluate_threshold(test, test_scores, gate["threshold"])
                result.update(
                    {
                        "status": "ok",
                        "train_threshold": gate,
                        "gated_test_accuracy": test_gate["accuracy"],
                        "gated_test_delta": test_gate["delta_vs_baseline"],
                        "routed": test_gate["routed"],
                        "coverage": test_gate["coverage"],
                        "rescues": test_gate["rescues"],
                        "harmful": test_gate["harmful"],
                    }
                )
            outputs.append(result)
    return outputs


def summarize_loso(rows: list[dict]) -> dict:
    out = {}
    for selector in sorted({row["selector"] for row in rows}):
        subset = [row for row in rows if row["selector"] == selector]
        out[selector] = {
            "mean_candidate_delta": float(np.mean([row["candidate_test_delta"] for row in subset])),
            "mean_gated_delta": float(np.mean([row["gated_test_delta"] for row in subset])),
            "min_candidate_delta": float(np.min([row["candidate_test_delta"] for row in subset])),
            "min_gated_delta": float(np.min([row["gated_test_delta"] for row in subset])),
            "num_candidate_negative": int(sum(row["candidate_test_delta"] < 0 for row in subset)),
            "num_gated_negative": int(sum(row["gated_test_delta"] < 0 for row in subset)),
            "num_status_ok": int(sum(row.get("status") == "ok" for row in subset)),
        }
    return out


def main() -> None:
    args = parse_args()
    rows = load_rows(args.features)
    keys = feature_keys(rows)
    stats = single_feature_stats(rows, keys)
    loso = loso_gate(rows, keys, args.min_train_events)
    selectors = sorted({row["selector"] for row in rows})
    slices = sorted({row["slice"] for row in rows})
    payload = {
        "version": "risk-feature-analysis-v1",
        "source_features": str(args.features),
        "n_rows": len(rows),
        "n_features": len(keys),
        "feature_keys": keys,
        "selectors": selectors,
        "slices": slices,
        "label_summary": {
            selector: {
                "n": len([row for row in rows if row["selector"] == selector]),
                "rescue": int(sum(row["rescue"] for row in rows if row["selector"] == selector)),
                "harmful": int(sum(row["harmful"] for row in rows if row["selector"] == selector)),
                "baseline_accuracy": mean_bool([row for row in rows if row["selector"] == selector], "baseline_correct"),
                "candidate_accuracy": mean_bool([row for row in rows if row["selector"] == selector], "candidate_correct"),
            }
            for selector in selectors
        },
        "top_features": {},
        "loso_summary": summarize_loso(loso),
        "loso_gate_rows": loso,
        "interpretation": {
            "claim_use": "diagnostic/selector development only; LOSO gates use calibration labels from other slices and no held-out slice labels for threshold fitting.",
            "recommended_plot_inputs": [
                "risk feature rank TSV",
                "risk_features_v1.jsonl per-sample rows",
                "LOSO gate rows for per-slice heatmaps",
            ],
        },
    }
    for selector in selectors:
        subset = [row for row in stats if row["selector"] == selector]
        payload["top_features"][selector] = {
            "rescue": sorted(
                subset,
                key=lambda row: -1.0 if row["auroc_rescue_vs_rest"] is None else -abs(row["auroc_rescue_vs_rest"] - 0.5),
            )[:10],
            "harmful": sorted(
                subset,
                key=lambda row: -1.0 if row["auroc_harmful_vs_rest"] is None else -abs(row["auroc_harmful_vs_rest"] - 0.5),
            )[:10],
            "changed": sorted(
                subset,
                key=lambda row: -1.0 if row["auroc_changed_vs_neutral"] is None else -abs(row["auroc_changed_vs_neutral"] - 0.5),
            )[:10],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    args.rank_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.rank_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in stats:
            writer.writerow(row)
    args.loso_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.loso_jsonl.open("w") as handle:
        for row in loso:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload["loso_summary"], indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.rank_tsv}")
    print(f"wrote {args.loso_jsonl}")


if __name__ == "__main__":
    main()
