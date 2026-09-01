"""Paired, cluster-aware uncertainty estimates."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Hashable, Iterable


def cluster_bootstrap_metric(
    rows: Iterable[dict],
    metric: Callable[[list[dict]], float],
    *,
    cluster_key: str = "cluster_id",
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate an arbitrary row metric with cluster-resampled intervals.

    All rows belonging to a sampled cluster are retained.  This is important
    for VQA datasets where several questions share one image.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped: dict[Hashable, list[dict]] = defaultdict(list)
    materialized = list(rows)
    for row in materialized:
        grouped[row[cluster_key]].append(row)
    clusters = sorted(grouped, key=str)
    if not clusters:
        raise ValueError("cluster bootstrap requires at least one row")
    generator = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled: list[dict] = []
        for cluster in generator.choices(clusters, k=len(clusters)):
            sampled.extend(grouped[cluster])
        values.append(metric(sampled))
    values.sort()
    lower = values[int(0.025 * replicates)]
    upper = values[min(int(0.975 * replicates), replicates - 1)]
    return {
        "estimate": metric(materialized),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "clusters": len(clusters),
        "replicates": replicates,
        "seed": seed,
    }


def cluster_bootstrap_difference(
    rows: Iterable[dict],
    metric: Callable[[list[dict]], float],
    *,
    cluster_key: str = "cluster_id",
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    return cluster_bootstrap_metric(
        rows,
        metric,
        cluster_key=cluster_key,
        replicates=replicates,
        seed=seed,
    )
