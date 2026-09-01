"""Retrospective source-to-target transfer check for two-view Probe-and-Pool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _bootstrap_delta, _load_expert, _metrics, _split,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
OUT = Path("corrected_runs/source_calibrated_probe_pool_transfer_v1/result.json")


def load(dataset: str):
    paths = [
        ROOT / f"shared_rag_generation/huatuo/{dataset}/no_context",
        ROOT / f"shared_rag_generation/huatuo/{dataset}/rag",
    ]
    experts = [_load_expert(path) for path in paths]
    qids = sorted(set(experts[0]) & set(experts[1]))
    rows = [[expert[qid] for expert in experts] for qid in qids]
    pred = np.asarray([[item["pred"] for item in row] for row in rows])
    nll = np.asarray([[item["nll"] for item in row] for row in rows])
    target = np.asarray([row[0]["target"] for row in rows])
    clusters = np.asarray([row[0]["cluster"] for row in rows])
    probability = 1 / (1 + np.exp(-((2 * pred - 1) * np.exp(-nll)).sum(axis=1)))
    return pred, target, clusters, probability


def main() -> None:
    source_pred, source_target, source_clusters, source_probability = load("knowledge_mimic_ce")
    source_split = np.asarray([_split(cluster) for cluster in source_clusters])
    source_validation = np.flatnonzero(source_split == "validation")
    grid = np.linspace(0.1, 0.9, 81)
    threshold = max(
        grid,
        key=lambda value: _metrics(
            source_target[source_validation],
            (source_probability[source_validation] >= value).astype(int),
        )["balanced_accuracy"],
    )

    target_pred, target, target_clusters, target_probability = load("cxr_vishal")
    pooled = (target_probability >= threshold).astype(int)
    baseline = target_pred[:, 1]
    result = {
        "status": "retrospective_external_dataset_transfer_development_evidence",
        "source": "Knowledge-MIMIC CE",
        "target": "CXR-VisHal binary intersection",
        "interventions": ["Huatuo no-context", "Huatuo BM25-RAG"],
        "source_validation_selected_threshold": float(threshold),
        "target_threshold_tuning": False,
        "target_n": int(len(target)),
        "target_rag": _metrics(target, baseline),
        "target_probe_and_pool": _metrics(target, pooled),
        "target_delta_cluster_bootstrap": _bootstrap_delta(
            target, pooled, baseline, target_clusters
        ),
        "warning": (
            "The target labels had been inspected in earlier development analyses. A future unfinished "
            "model/dataset cell remains necessary for a truly blind confirmation."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
