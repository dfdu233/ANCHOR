"""Cross-fitted screen of a DICOM-render response fingerprint.

This deliberately does not search for a globally beneficial render.  It asks
whether the patient-aligned vector of responses across clinically admitted
display windows contains label information beyond the canonical render.
Only unanimous VinDr reader labels are used.  Every learned prediction is
out-of-fold, and a delta-shuffle placebo tests patient alignment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN = Path("corrected_runs/vindr_v2/dicom_render_huatuo_pilot_v1")
OUT = Path("corrected_runs/render_response_fingerprint_v1/result.json")
VIEWS = [
    "baseline_percentile",
    "center_minus_0p05w",
    "center_plus_0p05w",
    "native_linear",
    "width_x1p25",
]


def _fold(image_id: str, finding: str) -> int:
    digest = hashlib.sha256(f"{image_id}|{finding}|render-fingerprint-v1".encode()).hexdigest()
    return int(digest[:8], 16) % 5


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "n": int(len(y)),
        "balanced_accuracy": float((tpr + tnr) / 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = []
    for path in sorted((RUN / "shards").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok" or record.get("positive_votes") not in (0, 3):
            continue
        by_view = {view["name"]: view for view in record["views"]}
        if not all(name in by_view for name in VIEWS):
            continue
        polarity = [float(by_view[name]["scores"]["polarity"]) for name in VIEWS]
        rows.append({
            "image_id": record["image_id"],
            "finding": record["finding"],
            "y": int(record["positive_votes"] == 3),
            "polarity": polarity,
        })
    x = np.asarray([row["polarity"] for row in rows], dtype=float)
    y = np.asarray([row["y"] for row in rows], dtype=int)
    folds = np.asarray([_fold(row["image_id"], row["finding"]) for row in rows], dtype=int)
    clusters = [row["image_id"] for row in rows]
    return x, y, folds, clusters


def _crossfit(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    score = np.zeros(len(y), dtype=float)
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42),
        )
        model.fit(x[train], y[train])
        score[test] = model.predict_proba(x[test])[:, 1]
    return score


def _cluster_bootstrap(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, clusters: list[str], repetitions: int = 5000
) -> dict[str, float]:
    unique = np.asarray(sorted(set(clusters)))
    index = {cluster: np.flatnonzero(np.asarray(clusters) == cluster) for cluster in unique}
    rng = np.random.default_rng(20260810)
    deltas = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([index[cluster] for cluster in sampled])
        deltas.append(
            _metrics(y[take], a[take])["balanced_accuracy"]
            - _metrics(y[take], b[take])["balanced_accuracy"]
        )
    return {
        "point": float(_metrics(y, a)["balanced_accuracy"] - _metrics(y, b)["balanced_accuracy"]),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


def main() -> None:
    x, y, folds, clusters = _load()
    baseline_score = _crossfit(x[:, :1], y, folds)
    fingerprint_score = _crossfit(x, y, folds)

    # Preserve each canonical response and each intervention's marginal delta,
    # but break the link between the intervention response and the patient.
    rng = np.random.default_rng(20260810)
    placebo = x.copy()
    delta = x[:, 1:] - x[:, :1]
    for fold in range(5):
        local = np.flatnonzero(folds == fold)
        for column in range(delta.shape[1]):
            placebo[local, column + 1] = x[local, 0] + delta[rng.permutation(local), column]
    placebo_score = _crossfit(placebo, y, folds)

    baseline = (baseline_score >= 0.5).astype(int)
    fingerprint = (fingerprint_score >= 0.5).astype(int)
    placebo_pred = (placebo_score >= 0.5).astype(int)
    per_view = {
        name: _metrics(y, (x[:, index] >= 0).astype(int))
        for index, name in enumerate(VIEWS)
    }
    result = {
        "status": "exploratory_crossfit_not_paper_authorized",
        "model": "HuatuoGPT-Vision-7B",
        "dataset": "VinDr unanimous reader claims",
        "n": int(len(y)),
        "views": VIEWS,
        "fold_counts": {str(fold): int(np.sum(folds == fold)) for fold in range(5)},
        "raw_per_view": per_view,
        "crossfit": {
            "canonical_only": _metrics(y, baseline),
            "full_response_fingerprint": _metrics(y, fingerprint),
            "patient_misaligned_delta_placebo": _metrics(y, placebo_pred),
            "fingerprint_vs_canonical": _cluster_bootstrap(y, fingerprint, baseline, clusters),
            "fingerprint_vs_placebo": _cluster_bootstrap(y, fingerprint, placebo_pred, clusters),
        },
        "interpretation": (
            "A positive result would show that heterogeneous render responses can act as a "
            "diagnostic fingerprint; it would not establish clinical render equivalence or a "
            "training-domain center."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
