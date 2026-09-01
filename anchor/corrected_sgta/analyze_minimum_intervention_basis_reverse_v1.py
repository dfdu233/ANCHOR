"""Reverse-domain replication of the minimum intervention basis screen."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from scipy.stats import spearmanr

from anchor.corrected_sgta.analyze_minimum_intervention_basis_v1 import (
    fit_subset,
    load,
)


OUT = Path("corrected_runs/minimum_intervention_basis_v1/reverse_result.json")


def main() -> None:
    source, target = load("target"), load("source")
    rows = []
    for size in range(1, len(source["names"]) + 1):
        for subset in itertools.combinations(range(len(source["names"])), size):
            row = fit_subset(subset, source, target)
            rows.append({
                "arms": row["arms"],
                "fisher_train": row["fisher_train"],
                "source_validation_bacc": row["source_validation"]["balanced_accuracy"],
                "target_bacc": row["target"]["balanced_accuracy"],
            })
    result = {
        "status": "reverse_domain_replication",
        "source": "CXR-VisHal",
        "target": "Knowledge-MIMIC CE",
        "source_n": int(len(source["target"])),
        "target_n": int(len(target["target"])),
        "fisher_vs_target_bacc_spearman": float(spearmanr(
            [row["fisher_train"] for row in rows],
            [row["target_bacc"] for row in rows],
        ).statistic),
        "source_validation_vs_target_bacc_spearman": float(spearmanr(
            [row["source_validation_bacc"] for row in rows],
            [row["target_bacc"] for row in rows],
        ).statistic),
        "within_size": {
            str(size): {
                "n": len(local),
                "fisher_vs_target_bacc_spearman": (
                    float(spearmanr(
                        [row["fisher_train"] for row in local],
                        [row["target_bacc"] for row in local],
                    ).statistic) if len(local) >= 3 else None
                ),
            }
            for size in range(1, len(source["names"]) + 1)
            for local in [[row for row in rows if len(row["arms"]) == size]]
        },
        "all_subsets": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
