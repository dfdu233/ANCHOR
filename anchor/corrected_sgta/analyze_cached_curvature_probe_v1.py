"""Retrospective curvature screen over cached Huatuo VinDr render probes.

The cache contains independent center and width perturbation pairs but not
their full Cartesian compositions.  Consequently this script measures
within-axis second differences and their alignment; it is a cheap fatal
screen, not the formal mixed-intervention curvature experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("corrected_runs/domain_orbit_head_v1")
OUT = Path("corrected_runs/intervention_curvature_screen_v1/result.json")
TRAIN_FILES = [ROOT / "pilot_n16/result.json", ROOT / "dev_n32/result.json"]
TEST_FILES = [ROOT / "confirmation_n32/result.json"]
LAYERS = (0, 7, 14, 21, 27)


def _entropy(logits: dict[str, float]) -> float:
    values = np.asarray([logits[key] for key in ("supported", "refuted", "undetermined")])
    probability = np.exp(values - values.max())
    probability /= probability.sum()
    return float(-(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum())


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.clip(denominator, 1e-12, None)


def extract(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    views = record["views"]
    base = views["baseline_percentile"]
    scalar = [
        abs(float(base["polarity"])),
        _entropy(base["logits"]),
        float(record["summary"]["polarity_range"]),
        float(record["summary"]["any_prediction_flip"]),
    ]
    first_order, curvature = [], []
    audit = {}
    for layer in LAYERS:
        key = str(layer)
        b = np.asarray(base["layers"][key]["head_output"], dtype=np.float64)
        minus = np.asarray(views["center_minus_0p05w"]["layers"][key]["head_output"], dtype=np.float64)
        plus = np.asarray(views["center_plus_0p05w"]["layers"][key]["head_output"], dtype=np.float64)
        narrow = np.asarray(views["width_x0p8"]["layers"][key]["head_output"], dtype=np.float64)
        wide = np.asarray(views["width_x1p25"]["layers"][key]["head_output"], dtype=np.float64)
        d_center = 0.5 * (plus - minus)
        d_width = 0.5 * (wide - narrow)
        c_center = plus + minus - 2.0 * b
        c_width = wide + narrow - 2.0 * b
        d_center_norm = np.linalg.norm(d_center, axis=1)
        d_width_norm = np.linalg.norm(d_width, axis=1)
        c_center_norm = np.linalg.norm(c_center, axis=1)
        c_width_norm = np.linalg.norm(c_width, axis=1)
        first_order.extend(
            [d_center_norm.mean(), d_width_norm.mean(), d_center_norm.max(), d_width_norm.max()]
        )
        curvature.extend(
            [
                c_center_norm.mean(),
                c_width_norm.mean(),
                c_center_norm.max(),
                c_width_norm.max(),
                _cosine_rows(c_center, c_width).mean(),
                c_center_norm.mean() / (d_center_norm.mean() + 1e-8),
                c_width_norm.mean() / (d_width_norm.mean() + 1e-8),
            ]
        )
        audit[key] = {
            "center_first_mean": float(d_center_norm.mean()),
            "width_first_mean": float(d_width_norm.mean()),
            "center_curvature_mean": float(c_center_norm.mean()),
            "width_curvature_mean": float(c_width_norm.mean()),
        }
    error = int(base["prediction"] != "supported")
    return np.asarray(scalar), np.asarray(first_order), np.asarray(curvature), error, audit


def load(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        rows.extend(extract(record) for record in data["records"])
    scalar = np.stack([row[0] for row in rows])
    first = np.stack([row[1] for row in rows])
    curve = np.stack([row[2] for row in rows])
    target = np.asarray([row[3] for row in rows])
    return scalar, first, curve, target, [row[4] for row in rows]


def _score(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(5000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values.append(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index]))
    return {
        "delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "valid_bootstraps": len(values),
    }


def main() -> None:
    train_scalar, train_first, train_curve, train_y, _ = load(TRAIN_FILES)
    test_scalar, test_first, test_curve, test_y, test_audit = load(TEST_FILES)
    matrices = {
        "scalar_only": (train_scalar, test_scalar),
        "scalar_plus_first_order": (
            np.column_stack([train_scalar, train_first]),
            np.column_stack([test_scalar, test_first]),
        ),
        "scalar_first_plus_curvature": (
            np.column_stack([train_scalar, train_first, train_curve]),
            np.column_stack([test_scalar, test_first, test_curve]),
        ),
    }
    result = {
        "status": "retrospective_fatal_screen_not_formal_confirmation",
        "limitation": (
            "Only within-axis second differences are cached; formal IRC requires composed "
            "center-by-width interventions and held-out intervention families."
        ),
        "train_n": int(len(train_y)),
        "train_errors": int(train_y.sum()),
        "test_n": int(len(test_y)),
        "test_errors": int(test_y.sum()),
        "models": {},
        "univariate_test_curvature_aurocs": {},
    }
    scores = {}
    for name, (x_train, x_test) in matrices.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000, random_state=42),
        )
        model.fit(x_train, train_y)
        score = model.predict_proba(x_test)[:, 1]
        scores[name] = score
        result["models"][name] = _score(test_y, score)
    result["curvature_delta_vs_scalar"] = _bootstrap_delta(
        test_y, scores["scalar_first_plus_curvature"], scores["scalar_only"]
    )
    for index in range(test_curve.shape[1]):
        value = test_curve[:, index]
        auc = roc_auc_score(test_y, value)
        result["univariate_test_curvature_aurocs"][str(index)] = float(max(auc, 1.0 - auc))
    result["confirmation_record_audit"] = test_audit
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "confirmation_record_audit"}, indent=2))


if __name__ == "__main__":
    main()
