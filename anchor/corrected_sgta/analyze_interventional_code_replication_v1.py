"""Outcome-honest replication of interventional response coding on two CE sets."""

from __future__ import annotations

import json
from pathlib import Path

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import _load_expert
from anchor.corrected_sgta.analyze_interventional_code_pilot_v1 import run_cohort


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
OUT = Path("corrected_runs/interventional_code_replication_v1/result.json")


def paths(dataset: str) -> dict[str, Path]:
    return {
        "huatuo_no_context": ROOT / f"shared_rag_generation/huatuo/{dataset}/no_context",
        "huatuo_rag": ROOT / f"shared_rag_generation/huatuo/{dataset}/rag",
        "qwen_vcd": ROOT / f"cross_model_methods/qwen/vcd/{dataset}",
        "qwen_dola": ROOT / f"derived_scores/qwen/DoLa/{dataset}",
    }


def main() -> None:
    datasets = {}
    for dataset in ("cxr_vishal", "slake_fine_grained"):
        source = paths(dataset)
        loaded = {name: _load_expert(path) for name, path in source.items()}
        datasets[dataset] = run_cohort(list(source), loaded)
    result = {
        "status": "completed_cached_cpu_replication",
        "datasets": datasets,
        "claim_boundary": (
            "Binary/ternary parseable CE intersection only; short-answer and multiple-choice rows "
            "are not silently converted to binary claims."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
