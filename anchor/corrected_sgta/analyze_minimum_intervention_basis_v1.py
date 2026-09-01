"""Source-only selection of a minimum diagnostic intervention basis.

All subset choices and thresholds use Knowledge-MIMIC only.  The frozen subset
and linear decoder are then evaluated on CXR-VisHal.  This is a fatal screen for
the hypothesis that class-conditional response geometry, rather than merely
the number of ensemble arms, identifies a transferable low-cost basis.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _bootstrap_delta,
    _load_expert,
    _metrics,
    _split,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
OUT = Path("corrected_runs/minimum_intervention_basis_v1/result.json")
PATHS = {
    "huatuo_plain": {
        "source": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/no_context",
        "target": ROOT / "shared_rag_generation/huatuo/cxr_vishal/no_context",
    },
    "huatuo_rag": {
        "source": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/rag",
        "target": ROOT / "shared_rag_generation/huatuo/cxr_vishal/rag",
    },
    "hulu_plain": {
        "source": ROOT / "shared_rag_generation/hulu/knowledge_mimic_ce/no_context",
        "target": ROOT / "shared_rag_generation/hulu/cxr_vishal/no_context",
    },
    "hulu_rag": {
        "source": ROOT / "shared_rag_generation/hulu/knowledge_mimic_ce/rag",
        "target": ROOT / "shared_rag_generation/hulu/cxr_vishal/rag",
    },
    "qwen_dola": {
        "source": ROOT / "derived_scores/qwen/DoLa/knowledge_mimic_ce",
        "target": ROOT / "derived_scores/qwen/DoLa/cxr_vishal",
    },
}


def load(domain: str) -> dict[str, np.ndarray]:
    loaded = {name: _load_expert(paths[domain]) for name, paths in PATHS.items()}
    names = list(PATHS)
    qids = sorted(set.intersection(*(set(loaded[name]) for name in names)))
    pred = np.asarray([[loaded[name][qid]["pred"] for name in names] for qid in qids])
    nll = np.asarray([[loaded[name][qid]["nll"] for name in names] for qid in qids])
    return {
        "names": names,
        "qid": np.asarray(qids),
        "signed": (2 * pred - 1) * np.exp(-np.clip(nll, 0.0, 20.0)),
        "pred": pred,
        "target": np.asarray([loaded[names[0]][qid]["target"] for qid in qids]),
        "cluster": np.asarray([loaded[names[0]][qid]["cluster"] for qid in qids]),
    }


def fisher_distance(x: np.ndarray, y: np.ndarray) -> float:
    means = [x[y == value].mean(axis=0) for value in (0, 1)]
    centered = np.concatenate([x[y == value] - means[value] for value in (0, 1)], axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 2)
    covariance = np.atleast_2d(covariance) + 0.05 * np.eye(x.shape[1])
    delta = np.atleast_1d(means[1] - means[0])
    return float(delta @ np.linalg.solve(covariance, delta))


def fit_subset(
    subset: tuple[int, ...], source: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict:
    split = np.asarray([_split(cluster) for cluster in source["cluster"]])
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    sx = source["signed"][:, subset]
    tx = target["signed"][:, subset]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(sx[train], source["target"][train])
    source_score = model.predict_proba(sx)[:, 1]
    target_score = model.predict_proba(tx)[:, 1]
    grid = np.linspace(0.1, 0.9, 81)
    threshold = max(
        grid,
        key=lambda value: _metrics(
            source["target"][validation],
            (source_score[validation] >= value).astype(int),
        )["balanced_accuracy"],
    )
    target_pred = (target_score >= threshold).astype(int)
    return {
        "arms": [source["names"][index] for index in subset],
        "fisher_train": fisher_distance(sx[train], source["target"][train]),
        "source_validation": _metrics(
            source["target"][validation],
            (source_score[validation] >= threshold).astype(int),
        ),
        "target": _metrics(target["target"], target_pred),
        "target_prediction": target_pred,
    }


def main() -> None:
    source, target = load("source"), load("target")
    rows = []
    for size in range(1, len(source["names"]) + 1):
        for subset in itertools.combinations(range(len(source["names"])), size):
            rows.append(fit_subset(subset, source, target))
    baseline = max(
        (row for row in rows if len(row["arms"]) == 1),
        key=lambda row: row["source_validation"]["balanced_accuracy"],
    )
    selected = {}
    for size in range(1, len(source["names"]) + 1):
        local = [row for row in rows if len(row["arms"]) == size]
        for rule, key in (
            ("source_validation", lambda row: row["source_validation"]["balanced_accuracy"]),
            ("fisher_train", lambda row: row["fisher_train"]),
        ):
            row = max(local, key=key)
            selected[f"k{size}_{rule}"] = {
                key: value for key, value in row.items() if key != "target_prediction"
            }
            selected[f"k{size}_{rule}"]["target_delta_vs_source_selected_single"] = _bootstrap_delta(
                target["target"], row["target_prediction"], baseline["target_prediction"], target["cluster"]
            )
    result = {
        "status": "source_only_basis_selection_target_evaluation",
        "source": "Knowledge-MIMIC CE",
        "target": "CXR-VisHal",
        "source_n": int(len(source["target"])),
        "target_n": int(len(target["target"])),
        "candidate_arms": source["names"],
        "source_selected_single": {
            key: value for key, value in baseline.items() if key != "target_prediction"
        },
        "selected": selected,
        "all_subsets": [
            {
                "arms": row["arms"],
                "fisher_train": row["fisher_train"],
                "source_validation_bacc": row["source_validation"]["balanced_accuracy"],
                "target_bacc": row["target"]["balanced_accuracy"],
            }
            for row in rows
        ],
        "rank_prediction": {
            "fisher_vs_target_bacc_spearman": float(spearmanr(
                [row["fisher_train"] for row in rows],
                [row["target"]["balanced_accuracy"] for row in rows],
            ).statistic),
            "source_validation_vs_target_bacc_spearman": float(spearmanr(
                [row["source_validation"]["balanced_accuracy"] for row in rows],
                [row["target"]["balanced_accuracy"] for row in rows],
            ).statistic),
            "within_size": {
                str(size): {
                    "n": len(local),
                    "fisher_vs_target_bacc_spearman": (
                        float(spearmanr(
                            [row["fisher_train"] for row in local],
                            [row["target"]["balanced_accuracy"] for row in local],
                        ).statistic) if len(local) >= 3 else None
                    ),
                }
                for size in range(1, len(source["names"]) + 1)
                for local in [[row for row in rows if len(row["arms"]) == size]]
            },
        },
        "falsification_rule": (
            "Reject a transferable minimum-basis claim if train Fisher selection does not "
            "track target benefit or requires the full arm set."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
