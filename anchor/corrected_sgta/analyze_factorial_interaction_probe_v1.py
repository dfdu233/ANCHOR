"""Summarize a 2x2 render-by-prompt interaction probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    rows = data["records"]
    error = np.asarray([not row["summary"]["base_correct"] for row in rows], dtype=int)
    features = {
        "absolute_polarity_mixed": np.asarray(
            [abs(row["summary"]["polarity_mixed_interaction"]) for row in rows]
        ),
        "polarity_mixed_to_main_ratio": np.asarray(
            [row["summary"]["polarity_mixed_to_main_ratio"] for row in rows]
        ),
        "mean_head_mixed_norm": np.asarray(
            [
                np.mean(
                    [layer["mean_mixed_norm"] for layer in row["summary"]["layers"].values()]
                )
                for row in rows
            ]
        ),
        "mean_head_mixed_to_main_ratio": np.asarray(
            [
                np.mean(
                    [layer["mixed_to_main_ratio"] for layer in row["summary"]["layers"].values()]
                )
                for row in rows
            ]
        ),
        "mean_head_first_order_norm": np.asarray(
            [
                np.mean(
                    [
                        layer["mean_image_main_norm"] + layer["mean_prompt_main_norm"]
                        for layer in row["summary"]["layers"].values()
                    ]
                )
                for row in rows
            ]
        ),
    }
    summaries = {}
    for name, value in features.items():
        auc = float(roc_auc_score(error, value)) if len(np.unique(error)) == 2 else None
        summaries[name] = {
            "auroc_error_high": auc,
            "correct_median": float(np.median(value[error == 0])) if np.any(error == 0) else None,
            "error_median": float(np.median(value[error == 1])) if np.any(error == 1) else None,
        }
    result = {
        "status": "pilot_feature_screen_only",
        "n": len(rows),
        "errors": int(error.sum()),
        "prediction_flips": int(sum(row["summary"]["any_cell_prediction_flip"] for row in rows)),
        "features": summaries,
        "gate": (
            "Pilot only selects a predeclared feature and direction. A disjoint dev split must "
            "show AUROC >= 0.60 and >=0.05 over first-order norm before training is justified."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
