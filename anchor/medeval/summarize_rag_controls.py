#!/usr/bin/env python3
"""Summarize whether an apparent RAG gain survives causal controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import atomic_write_json


VERSION = "common-rag-causal-control-summary-v2-provenance"


def summarize(root: Path, datasets: list[str]) -> dict:
    records = []
    for dataset in datasets:
        directory = root / dataset / "visual_ce_v2" / "ladder_v3" / "causal_controls_v1"
        relevance_v2 = directory / "rag_vs_shuffled_context_v2.json"
        image_v2 = directory / "rag_vs_image_swap_v2.json"
        relevance_path = (
            relevance_v2 if relevance_v2.is_file()
            else directory / "rag_vs_shuffled_context.json"
        )
        image_path = image_v2 if image_v2.is_file() else directory / "rag_vs_image_swap.json"
        relevance = json.loads(relevance_path.read_text()) if relevance_path.is_file() else None
        image = json.loads(image_path.read_text()) if image_path.is_file() else None
        relevance_passed = bool(relevance and relevance.get("full_run_authorized") is True)
        image_passed = bool(image and image.get("full_run_authorized") is True)
        records.append(
            {
                "dataset": dataset,
                "model": "llava",
                "relevance_control": relevance,
                "image_identity_control": image,
                "relevance_artifact": str(relevance_path.resolve()) if relevance else None,
                "image_identity_artifact": str(image_path.resolve()) if image else None,
                "relevance_passed": relevance_passed,
                "image_identity_passed": image_passed,
                "rag_grounding_supported": relevance_passed and image_passed,
            }
        )
    return {
        "protocol_version": VERSION,
        "claim_boundary": (
            "a raw no-context improvement is retained as RAG grounding only when "
            "relevant retrieval beats a disjoint context permutation and the real "
            "image beats a different-patient image under the identical RAG prompt"
        ),
        "records": records,
        "supported": [
            {"dataset": row["dataset"], "model": row["model"]}
            for row in records
            if row["rag_grounding_supported"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["iuxray", "mimic"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.datasets)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
