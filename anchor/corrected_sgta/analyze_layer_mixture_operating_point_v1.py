#!/usr/bin/env python3
"""Separate ranking gains from operating-point shifts in layer mixtures.

This is an exploratory, post-hoc audit over already-opened VinDr artifacts.  It
uses the old fixed-panel artifact for development-only selection and the fresh
532-image artifact for a single confirmation read.  The audit does not claim a
new prospective endpoint; it asks whether convex early/final margin mixtures
improve threshold-free ranking, or merely move the zero-threshold decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
ALPHAS = np.linspace(0.0, 1.0, 21)
BOOTSTRAP_DRAWS = 5000
SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_rows(directory: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (directory / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["finding"] in FINDINGS]
    if not rows:
        raise ValueError(f"no frozen finding rows in {directory}")
    keys = [row["record_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate record_key in {directory}")
    return rows


def layer_margins(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer_sets = [
        tuple(sorted(int(key) for key in row["diagnostic_plain_logit_lens"]))
        for row in rows
    ]
    if len(set(layer_sets)) != 1:
        raise ValueError("layer sets differ across rows")
    layers = np.asarray(layer_sets[0], dtype=int)
    margins = np.empty((len(rows), len(layers)), dtype=float)
    for i, row in enumerate(rows):
        logits = row["diagnostic_plain_logit_lens"]
        for j, layer in enumerate(layers):
            values = logits[str(int(layer))]
            margins[i, j] = float(values["supported"] - values["refuted"])
    labels = np.asarray([int(row["positive_votes"] == 3) for row in rows])
    votes = np.asarray([int(row["positive_votes"]) for row in rows])
    return layers, margins, labels, votes


def clear_mask(votes: np.ndarray) -> np.ndarray:
    return (votes == 0) | (votes == 3)


def macro_auc(rows: list[dict[str, Any]], votes: np.ndarray, score: np.ndarray) -> float:
    values = []
    finding = np.asarray([row["finding"] for row in rows])
    for name in FINDINGS:
        mask = (finding == name) & clear_mask(votes)
        y = (votes[mask] == 3).astype(int)
        values.append(roc_auc_score(y, score[mask]))
    return float(np.mean(values))


def per_finding_auc(
    rows: list[dict[str, Any]], votes: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    finding = np.asarray([row["finding"] for row in rows])
    answer = {}
    for name in FINDINGS:
        mask = (finding == name) & clear_mask(votes)
        answer[name] = float(
            roc_auc_score((votes[mask] == 3).astype(int), score[mask])
        )
    return answer


def choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    unique = np.unique(score)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[:-1] + unique[1:]) / 2.0,
        np.nextafter(unique[-1], np.inf),
    ]
    quality = np.asarray(
        [balanced_accuracy_score(y, score >= threshold) for threshold in candidates]
    )
    best = np.flatnonzero(quality == quality.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def fit_thresholds(
    rows: list[dict[str, Any]], votes: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    finding = np.asarray([row["finding"] for row in rows])
    result = {}
    for name in FINDINGS:
        mask = (finding == name) & clear_mask(votes)
        result[name] = choose_threshold(
            (votes[mask] == 3).astype(int), score[mask]
        )
    return result


def decision_metrics(
    rows: list[dict[str, Any]],
    votes: np.ndarray,
    score: np.ndarray,
    thresholds: dict[str, float] | None,
) -> dict[str, float]:
    mask = clear_mask(votes)
    y = (votes[mask] == 3).astype(int)
    if thresholds is None:
        prediction = score[mask] >= 0.0
    else:
        names = np.asarray([row["finding"] for row in rows])[mask]
        prediction = np.asarray(
            [value >= thresholds[name] for value, name in zip(score[mask], names)]
        )
    return {
        "n": int(mask.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(prediction == y)),
        "positive_rate": float(np.mean(prediction)),
    }


def mixture(final: np.ndarray, early: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * final + alpha * early


def select_auc_configuration(
    rows: list[dict[str, Any]], layers: np.ndarray, margins: np.ndarray, votes: np.ndarray
) -> dict[str, Any]:
    final = margins[:, -1]
    candidates = []
    for column, layer in enumerate(layers[:-1]):
        for alpha in ALPHAS:
            score = mixture(final, margins[:, column], float(alpha))
            candidates.append(
                {
                    "layer": int(layer),
                    "alpha": float(alpha),
                    "macro_auroc": macro_auc(rows, votes, score),
                }
            )
    candidates.sort(key=lambda item: (-item["macro_auroc"], item["alpha"], item["layer"]))
    return {"selected": candidates[0], "grid": candidates}


def select_fixed_zero_bacc_configuration(
    rows: list[dict[str, Any]], layers: np.ndarray, margins: np.ndarray, votes: np.ndarray
) -> dict[str, Any]:
    final = margins[:, -1]
    candidates = []
    for column, layer in enumerate(layers[:-1]):
        for alpha in ALPHAS:
            score = mixture(final, margins[:, column], float(alpha))
            candidates.append(
                {
                    "layer": int(layer),
                    "alpha": float(alpha),
                    **decision_metrics(rows, votes, score, None),
                }
            )
    candidates.sort(
        key=lambda item: (-item["balanced_accuracy"], item["alpha"], item["layer"])
    )
    return {"selected": candidates[0], "grid": candidates}


def paired_bootstrap_auc(
    rows: list[dict[str, Any]],
    votes: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    finding = np.asarray([row["finding"] for row in rows])
    cells = [
        np.flatnonzero((finding == name) & (votes == value))
        for name in FINDINGS
        for value in (0, 3)
    ]
    if any(len(cell) == 0 for cell in cells):
        raise ValueError("empty finding x clear-vote bootstrap cell")
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = np.concatenate(
            [rng.choice(cell, len(cell), replace=True) for cell in cells]
        )
        sampled_rows = [rows[int(index)] for index in sampled]
        sampled_votes = votes[sampled]
        deltas.append(
            macro_auc(sampled_rows, sampled_votes, candidate[sampled])
            - macro_auc(sampled_rows, sampled_votes, baseline[sampled])
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "unit": "paired resampling within finding x {0/3 reader-vote} cells",
        "delta_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "probability_delta_gt_zero": float(np.mean(np.asarray(deltas) > 0.0)),
    }


def audit_model(model: str, dev_dir: Path, test_dir: Path) -> dict[str, Any]:
    dev_rows, test_rows = read_rows(dev_dir), read_rows(test_dir)
    dev_layers, dev_margins, _, dev_votes = layer_margins(dev_rows)
    test_layers, test_margins, _, test_votes = layer_margins(test_rows)
    if not np.array_equal(dev_layers, test_layers):
        raise ValueError(f"{model}: development/confirmation layers differ")

    auc_selection = select_auc_configuration(
        dev_rows, dev_layers, dev_margins, dev_votes
    )
    zero_selection = select_fixed_zero_bacc_configuration(
        dev_rows, dev_layers, dev_margins, dev_votes
    )
    layer_to_column = {int(layer): i for i, layer in enumerate(test_layers)}
    final_dev, final_test = dev_margins[:, -1], test_margins[:, -1]

    chosen = auc_selection["selected"]
    candidate_dev = mixture(
        final_dev,
        dev_margins[:, layer_to_column[int(chosen["layer"])]],
        float(chosen["alpha"]),
    )
    candidate_test = mixture(
        final_test,
        test_margins[:, layer_to_column[int(chosen["layer"])]],
        float(chosen["alpha"]),
    )
    final_thresholds = fit_thresholds(dev_rows, dev_votes, final_dev)
    candidate_thresholds = fit_thresholds(dev_rows, dev_votes, candidate_dev)

    zero = zero_selection["selected"]
    zero_score = mixture(
        final_test,
        test_margins[:, layer_to_column[int(zero["layer"])]],
        float(zero["alpha"]),
    )
    baseline_auc = macro_auc(test_rows, test_votes, final_test)
    candidate_auc = macro_auc(test_rows, test_votes, candidate_test)
    by_finding_base = per_finding_auc(test_rows, test_votes, final_test)
    by_finding_candidate = per_finding_auc(test_rows, test_votes, candidate_test)
    finding_deltas = {
        name: by_finding_candidate[name] - by_finding_base[name] for name in FINDINGS
    }
    bootstrap = paired_bootstrap_auc(
        test_rows, test_votes, final_test, candidate_test
    )
    ci = bootstrap["delta_ci95"]
    passes = (
        candidate_auc - baseline_auc >= 0.02
        and ci[0] > 0.0
        and sum(value > 0 for value in finding_deltas.values()) >= 4
    )
    return {
        "model": model,
        "layers": test_layers.tolist(),
        "development_n": len(dev_rows),
        "confirmation_n": len(test_rows),
        "selection": {
            "objective": "development macro AUROC over clear 0/3 cases",
            "auc_selected": chosen,
            "fixed_zero_bacc_selected": zero,
        },
        "confirmation": {
            "final_macro_auroc": baseline_auc,
            "mixture_macro_auroc": candidate_auc,
            "macro_auroc_delta": candidate_auc - baseline_auc,
            "by_finding_final_auroc": by_finding_base,
            "by_finding_mixture_auroc": by_finding_candidate,
            "by_finding_delta": finding_deltas,
            "final_fixed_zero": decision_metrics(
                test_rows, test_votes, final_test, None
            ),
            "mixture_selected_for_auc_fixed_zero": decision_metrics(
                test_rows, test_votes, candidate_test, None
            ),
            "mixture_selected_for_zero_bacc_fixed_zero": decision_metrics(
                test_rows, test_votes, zero_score, None
            ),
            "final_dev_thresholds": decision_metrics(
                test_rows, test_votes, final_test, final_thresholds
            ),
            "mixture_dev_thresholds": decision_metrics(
                test_rows, test_votes, candidate_test, candidate_thresholds
            ),
            "paired_bootstrap": bootstrap,
        },
        "gate": {
            "rule": "confirmation AUROC delta >= 0.02, bootstrap lower CI > 0, and >=4/7 finding deltas > 0",
            "pass": bool(passes),
        },
        "interpretation_boundary": (
            "A pass would establish threshold-free complementarity for this convex "
            "margin family only. A fail closes this family, not arbitrary visual-token methods."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    inputs = {
        name: {
            "path": str(path.resolve()),
            "metadata_sha256": sha256(path / "metadata.jsonl"),
        }
        for name, path in {
            "huatuo_development": args.huatuo_dev,
            "huatuo_confirmation": args.huatuo_confirmation,
            "hulu_development": args.hulu_dev,
            "hulu_confirmation": args.hulu_confirmation,
        }.items()
    }
    result = {
        "protocol": "layer-mixture-operating-point-audit-v1",
        "status": "complete",
        "scope": "exploratory post-hoc audit over already-opened artifacts",
        "configuration": {
            "alpha_grid": ALPHAS.tolist(),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "findings": list(FINDINGS),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "source_sha256": sha256(Path(__file__)),
        },
        "inputs": inputs,
        "models": {
            "huatuo": audit_model(
                "huatuo", args.huatuo_dev, args.huatuo_confirmation
            ),
            "hulu": audit_model("hulu", args.hulu_dev, args.hulu_confirmation),
        },
    }
    result["joint_gate"] = {
        "both_models_pass": all(
            value["gate"]["pass"] for value in result["models"].values()
        ),
        "decision": (
            "retain_convex_layer_mixture_as_evidence_candidate"
            if all(value["gate"]["pass"] for value in result["models"].values())
            else "close_convex_layer_mixture_as_incremental_evidence_route"
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result["joint_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
