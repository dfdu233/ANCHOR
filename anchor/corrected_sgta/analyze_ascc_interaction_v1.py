#!/usr/bin/env python3
"""Fail-closed analysis of the frozen ASCC framing-by-reader interaction."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .prepare_ascc_interaction_v1 import PROMPTS
from .run_huatuo_ascc_interaction_v1 import (
    PRIMARY_EDGE,
    VERSION as SCORE_VERSION,
    load_jsonl,
    record_key,
    validate_substrate,
)


VERSION = "ascc-reader-interaction-analysis-v1"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def percentile_interval(values: Sequence[float], level: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - level) / 2.0
    return {
        "level": level,
        "low": float(np.quantile(array, alpha)),
        "high": float(np.quantile(array, 1.0 - alpha)),
    }


def bootstrap_two_families(
    families: Mapping[str, Sequence[float]],
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    seed: int,
    iterations: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    arrays = {key: np.asarray(values, dtype=np.float64) for key, values in families.items()}
    if set(arrays) != {"negative_boundary", "positive_boundary"}:
        raise ValueError("both ASCC local boundary families are required")
    if any(array.size == 0 for array in arrays.values()):
        raise ValueError("ASCC boundary family is empty")
    output = []
    for _ in range(iterations):
        sampled = {
            key: array[rng.integers(0, array.size, size=array.size)]
            for key, array in arrays.items()
        }
        output.append(float(statistic(sampled)))
    return output


def _did(values: Mapping[str, np.ndarray]) -> float:
    return 0.5 * (
        float(values["negative_boundary"].mean())
        + float(values["positive_boundary"].mean())
    )


def load_complete_scores(
    score_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    config_path = score_dir / "score_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    if config.get("version") != SCORE_VERSION:
        raise ValueError("wrong ASCC score version")
    fingerprint = str(config.get("fingerprint", ""))
    shards: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for row in rows:
        for prompt in PROMPTS:
            path = score_dir / "shards" / f"{record_key(row['item_id'], prompt['name'])}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            shard = json.loads(path.read_text())
            if shard.get("fingerprint") != fingerprint:
                raise ValueError(f"incompatible ASCC shard: {path}")
            if (
                shard.get("item_id") != row["item_id"]
                or shard.get("prompt_name") != prompt["name"]
                or shard.get("child_votes") != row["child_votes"]
            ):
                raise ValueError(f"ASCC shard identity mismatch: {path}")
            shards[(str(row["item_id"]), str(prompt["name"]))] = shard
    if missing:
        raise FileNotFoundError(
            f"ASCC analysis requires complete registered scores; missing={len(missing)}, first={missing[0]}"
        )
    return config, shards


def analyze_edge(
    rows: Sequence[Mapping[str, Any]],
    shards: Mapping[tuple[str, str], Mapping[str, Any]],
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    prompt_pairs = sorted({str(prompt["prompt_pair_id"]) for prompt in PROMPTS})
    prompt_lookup = {
        (str(prompt["prompt_pair_id"]), str(prompt["framing"])): str(prompt["name"])
        for prompt in PROMPTS
    }
    first = next(iter(shards.values()))
    layer_ids = list(first["layer_scores"])
    output: dict[str, Any] = {"layers": {}}
    for layer_index, layer_id in enumerate(layer_ids):
        per_item: dict[str, dict[str, Any]] = {}
        for row in rows:
            item_id = str(row["item_id"])
            pair_values = {}
            neutral_polarity = []
            for prompt_pair in prompt_pairs:
                neutral = shards[(item_id, prompt_lookup[(prompt_pair, "neutral")])]
                existential = shards[(item_id, prompt_lookup[(prompt_pair, "existential")])]
                neutral_coordinates = neutral["layer_scores"][layer_id]["coordinates"]
                existential_coordinates = existential["layer_scores"][layer_id]["coordinates"]
                pair_values[prompt_pair] = {
                    "delta_commitment": float(
                        existential_coordinates["commitment"]
                        - neutral_coordinates["commitment"]
                    ),
                    "delta_polarity": float(
                        existential_coordinates["polarity"]
                        - neutral_coordinates["polarity"]
                    ),
                }
                neutral_polarity.append(float(neutral_coordinates["polarity"]))
            per_item[item_id] = {
                "row": row,
                "prompt_pairs": pair_values,
                "delta_commitment": float(
                    np.mean([value["delta_commitment"] for value in pair_values.values()])
                ),
                "delta_polarity": float(
                    np.mean([value["delta_polarity"] for value in pair_values.values()])
                ),
                "neutral_polarity": float(np.mean(neutral_polarity)),
            }

        pair_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in per_item.values():
            pair_rows[str(item["row"]["matched_pair_id"])].append(item)
        family_commitment: dict[str, list[float]] = defaultdict(list)
        family_polarity: dict[str, list[float]] = defaultdict(list)
        family_admission: dict[str, list[float]] = defaultdict(list)
        prompt_family_commitment: dict[str, dict[str, list[float]]] = {
            prompt_pair: defaultdict(list) for prompt_pair in prompt_pairs
        }
        for pair_id, pair in pair_rows.items():
            if len(pair) != 2:
                raise ValueError(f"analysis pair does not have two rows: {pair_id}")
            by_votes = {int(item["row"]["child_votes"]): item for item in pair}
            family = str(pair[0]["row"]["comparison_family"])
            clear_vote, ambiguous_vote = (
                (0, 1) if family == "negative_boundary" else (3, 2)
            )
            clear = by_votes[clear_vote]
            ambiguous = by_votes[ambiguous_vote]
            family_commitment[family].append(
                ambiguous["delta_commitment"] - clear["delta_commitment"]
            )
            family_polarity[family].append(
                ambiguous["delta_polarity"] - clear["delta_polarity"]
            )
            # Directional visual admission always compares higher minus lower support.
            lower_vote, higher_vote = (
                (0, 1) if family == "negative_boundary" else (2, 3)
            )
            family_admission[family].append(
                by_votes[higher_vote]["neutral_polarity"]
                - by_votes[lower_vote]["neutral_polarity"]
            )
            for prompt_pair in prompt_pairs:
                prompt_family_commitment[prompt_pair][family].append(
                    ambiguous["prompt_pairs"][prompt_pair]["delta_commitment"]
                    - clear["prompt_pairs"][prompt_pair]["delta_commitment"]
                )

        commitment_boot = bootstrap_two_families(
            family_commitment, _did, seed + layer_index * 101, iterations
        )
        polarity_boot = bootstrap_two_families(
            family_polarity, _did, seed + layer_index * 101 + 1, iterations
        )
        admission_boot = bootstrap_two_families(
            family_admission, _did, seed + layer_index * 101 + 2, iterations
        )
        commitment_did = _did(
            {key: np.asarray(value) for key, value in family_commitment.items()}
        )
        polarity_did = _did(
            {key: np.asarray(value) for key, value in family_polarity.items()}
        )
        admission = _did(
            {key: np.asarray(value) for key, value in family_admission.items()}
        )
        prompt_dids = {
            prompt_pair: _did(
                {key: np.asarray(value) for key, value in families.items()}
            )
            for prompt_pair, families in prompt_family_commitment.items()
        }
        layer_summary = {
            "layer_id": layer_id,
            "local_commitment_interactions": {
                key: {
                    "n_pairs": len(values),
                    "mean": float(np.mean(values)),
                }
                for key, values in sorted(family_commitment.items())
            },
            "commitment_did": commitment_did,
            "commitment_did_ci95": percentile_interval(commitment_boot, 0.95),
            "polarity_did": polarity_did,
            "polarity_did_ci90": percentile_interval(polarity_boot, 0.90),
            "neutral_directional_admission": admission,
            "neutral_directional_admission_ci95": percentile_interval(admission_boot, 0.95),
            "prompt_pair_commitment_dids": prompt_dids,
        }
        output["layers"][layer_id] = layer_summary

    final = output["layers"][layer_ids[-1]]
    polarity_ci = final["polarity_did_ci90"]
    output["final_layer_id"] = layer_ids[-1]
    output["gate"] = {
        "directional_admission_passed": final["neutral_directional_admission_ci95"]["low"] > 0,
        "commitment_did_passed": final["commitment_did_ci95"]["low"] > 0,
        "both_local_interactions_positive": all(
            value["mean"] > 0
            for value in final["local_commitment_interactions"].values()
        ),
        "both_prompt_pairs_positive": all(
            value > 0 for value in final["prompt_pair_commitment_dids"].values()
        ),
        "polarity_equivalence_passed": polarity_ci["low"] >= -0.2
        and polarity_ci["high"] <= 0.2,
    }
    output["gate"]["ascc_behavioral_gate_passed"] = all(output["gate"].values())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--edge", default=PRIMARY_EDGE)
    parser.add_argument("--seed", type=int, default=82421)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("formal ASCC bootstrap requires at least 1000 iterations")

    substrate, all_rows = validate_substrate(args.substrate_dir, args.image_root)
    rows = [row for row in all_rows if row["edge_id"] == args.edge]
    if not rows:
        raise ValueError(f"edge absent from ASCC substrate: {args.edge}")
    score_config, shards = load_complete_scores(args.score_dir, rows)
    result = {
        "version": VERSION,
        "substrate_fingerprint": substrate["fingerprint"],
        "score_fingerprint": score_config["fingerprint"],
        "edge_id": args.edge,
        "registered_rows": len(rows),
        "registered_jobs": len(rows) * len(PROMPTS),
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "analysis": analyze_edge(rows, shards, args.seed, args.bootstrap_iterations),
        "claim_ceiling": (
            "controlled next-token instrument for radiograph-attributable reader "
            "disagreement; not patient-level diagnostic truth or OE clinical quality"
        ),
    }
    if args.output.exists():
        raise FileExistsError("ASCC formal analysis is write-once")
    atomic_json(args.output, result)
    print(json.dumps(result["analysis"]["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
