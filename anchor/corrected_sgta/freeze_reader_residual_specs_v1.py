#!/usr/bin/env python3
"""Freeze direction-specific early-layer specs before confirmation is visible."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file
from corrected_sgta.screen_reader_residual_v1 import (
    DEFAULT_K,
    STRATUM_NAMES,
    load_inputs,
    select_k_inner,
)


VERSION = "vindr-reader-residual-direction-lock-v1"


def select_direction_cells(results: dict[str, Any], final_layer: int) -> dict[str, dict[str, Any]]:
    output = {}
    for stratum_name in STRATUM_NAMES.values():
        eligible = [
            value for value in results.values()
            if int(value["layer"]) != final_layer
        ]
        if not eligible:
            raise ValueError("no non-final candidate layer")
        selected = min(
            eligible,
            key=lambda value: (
                -float(value["crossfit"][stratum_name]["relative_brier_improvement"]),
                -float(value["crossfit"][stratum_name]["delta_auc"]),
                int(value["layer"]),
                str(value["family"]),
            ),
        )
        output[stratum_name] = {
            "layer": int(selected["layer"]),
            "family": str(selected["family"]),
            "dev_crossfit": selected["crossfit"][stratum_name],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--residual-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any((args.confirmation_dir / name).exists() for name in ("metadata.jsonl", "hidden_states.npz")):
        raise RuntimeError("confirmation features already exist; refusing post-hoc specification")
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    if screen.get("status") != "complete" or screen.get("scope") != "development_screen_only_not_paper_evidence":
        raise ValueError("input is not a complete dev-only residual screen")
    if screen.get("provenance", {}).get("hidden_states_sha256") != sha256_file(args.features_dir / "hidden_states.npz"):
        raise ValueError("screen and dev feature hashes disagree")
    data = load_inputs(args.features_dir)
    layers = [int(value) for value in data["layers"]]
    final_layer = max(layers)
    selected = select_direction_cells(screen["results"], final_layer)
    for stratum_value, stratum_name in STRATUM_NAMES.items():
        spec = selected[stratum_name]
        mask = data["stratum"] == stratum_value
        layer_index = layers.index(spec["layer"])
        features = data["families"][spec["family"]][mask, layer_index]
        k, scores = select_k_inner(
            features=features,
            target=data["target"][mask],
            evidence=data["evidence"][mask],
            stratum=data["stratum"][mask],
            findings=data["findings"][mask],
            groups=data["groups"][mask],
            requested_k=DEFAULT_K,
            folds=args.inner_folds,
            seed=args.seed + 10000 * (stratum_value + 1),
            residual_alpha=args.residual_alpha,
        )
        spec.update(
            {
                "pca_k": k,
                "inner_mean_brier": scores,
                "final_comparator_layer": final_layer,
                "final_comparator_family": spec["family"],
                "final_comparator_pca_k": k,
            }
        )
    output = {
        "version": VERSION,
        "model_id": args.model_id,
        "status": "frozen_before_confirmation_features",
        "target": "reader unanimity within adjacent polarity bins",
        "selection": "per direction: non-final dev crossfit relative Brier, then delta AUROC; matched-capacity final comparator",
        "directions": selected,
        "provenance": {
            "screen": str(args.screen.resolve()),
            "screen_sha256": sha256_file(args.screen),
            "dev_hidden_states_sha256": sha256_file(args.features_dir / "hidden_states.npz"),
            "dev_metadata_sha256": sha256_file(args.features_dir / "metadata.jsonl"),
            "confirmation_dir_absent_at_freeze": True,
            "code_sha256": sha256_file(Path(__file__)),
            "seed": args.seed,
        },
    }
    output["fingerprint"] = hashlib.sha256(
        json.dumps(output, sort_keys=True).encode()
    ).hexdigest()
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
