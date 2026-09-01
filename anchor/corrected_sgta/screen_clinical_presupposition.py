#!/usr/bin/env python3
"""Matched-length, bidirectional screen for clinical presupposition errors.

The input is answer-level claim auditing produced by the shared evaluator.  No
text regex or LLM judgment is performed here.  The screen survives only when
an existential prompt increases false-positive claims and a negative-obligation
prompt increases false-negative claims, relative to the same neutral prompt,
in at least two models after strict within-image length matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Mapping


VERSION = "clinical-presupposition-bidirectional-screen-v1"
CONDITIONS = {"neutral", "existential", "negative_obligation"}
ADMISSIBLE_ADJUDICATION = {"physician", "multi_reader_consensus"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_row(row: Mapping[str, object]) -> None:
    required = {
        "item_id",
        "model_id",
        "prompt_condition",
        "token_count",
        "claim_universe_sha256",
        "supported_claim_count",
        "refuted_claim_count",
        "positive_claim_error_count",
        "negative_claim_error_count",
        "omitted_supported_claim_count",
        "formal_reference",
        "adjudication_source",
        "automatic_labeler_only",
        "ground_truth_used_for_generation_or_selection",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"presupposition row missing fields: {missing}")
    if row["prompt_condition"] not in CONDITIONS:
        raise ValueError(f"unknown prompt condition: {row['prompt_condition']}")
    if (
        row["formal_reference"] is not True
        or row["adjudication_source"] not in ADMISSIBLE_ADJUDICATION
        or row["automatic_labeler_only"] is not False
        or row["ground_truth_used_for_generation_or_selection"] is not False
    ):
        raise ValueError("screen requires independent human/multi-reader reference truth")
    if not valid_sha256(row["claim_universe_sha256"]):
        raise ValueError("claim_universe_sha256 must bind the fixed evaluator universe")
    integer_fields = (
        "token_count",
        "supported_claim_count",
        "refuted_claim_count",
        "positive_claim_error_count",
        "negative_claim_error_count",
        "omitted_supported_claim_count",
    )
    if any(int(row[field]) < 0 for field in integer_fields):
        raise ValueError("claim counts and token_count must be non-negative")
    if int(row["positive_claim_error_count"]) > int(row["refuted_claim_count"]):
        raise ValueError("positive errors exceed refuted reference claims")
    if int(row["negative_claim_error_count"]) > int(row["supported_claim_count"]):
        raise ValueError("negative errors exceed supported reference claims")


def error_rate(row: Mapping[str, object], kind: str) -> float:
    if kind == "positive":
        return int(row["positive_claim_error_count"]) / max(
            1, int(row["refuted_claim_count"])
        )
    if kind == "negative":
        return int(row["negative_claim_error_count"]) / max(
            1, int(row["supported_claim_count"])
        )
    if kind == "omission":
        return int(row["omitted_supported_claim_count"]) / max(
            1, int(row["supported_claim_count"])
        )
    raise ValueError(kind)


def length_matched(
    first: Mapping[str, object],
    second: Mapping[str, object],
    maximum_absolute_gap: int,
    maximum_relative_gap: float,
) -> bool:
    first_length = int(first["token_count"])
    second_length = int(second["token_count"])
    gap = abs(first_length - second_length)
    return bool(
        gap <= maximum_absolute_gap
        and gap / max(1, first_length, second_length) <= maximum_relative_gap
    )


def bootstrap_delta(
    pairs: list[tuple[Mapping[str, object], Mapping[str, object]]],
    metric: Callable[[Mapping[str, object]], float],
    draws: int,
    seed: int,
) -> dict[str, float | int | None]:
    differences = [metric(first) - metric(second) for first, second in pairs]
    if not differences:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "pairs": 0,
        }
    rng = random.Random(seed)
    samples = [mean(rng.choice(differences) for _ in differences) for _ in range(draws)]
    ordered = sorted(samples)
    low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return {
        "estimate": mean(differences),
        "ci_low": low,
        "ci_high": high,
        "pairs": len(differences),
    }


def analyze(
    rows: list[dict[str, object]],
    bootstrap_draws: int,
    seed: int,
    minimum_pairs: int,
    minimum_models: int,
    maximum_absolute_length_gap: int,
    maximum_relative_length_gap: float,
) -> dict[str, object]:
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        validate_row(row)
        key = (str(row["model_id"]), str(row["item_id"]))
        condition = str(row["prompt_condition"])
        if condition in grouped[key]:
            raise ValueError(f"duplicate model/item/condition: {key + (condition,)}")
        grouped[key][condition] = row

    complete = {key: value for key, value in grouped.items() if set(value) == CONDITIONS}
    by_model: dict[str, list[dict[str, dict[str, object]]]] = defaultdict(list)
    for (model_id, item_id), conditions in complete.items():
        universes = {str(row["claim_universe_sha256"]) for row in conditions.values()}
        denominators = {
            (int(row["supported_claim_count"]), int(row["refuted_claim_count"]))
            for row in conditions.values()
        }
        if len(universes) != 1 or len(denominators) != 1:
            raise ValueError(f"claim universe/reference changed across prompts: {item_id}")
        by_model[model_id].append(conditions)

    model_results = {}
    passed_models = []
    for model_index, (model_id, items) in enumerate(sorted(by_model.items())):
        existential_pairs = [
            (item["existential"], item["neutral"])
            for item in items
            if length_matched(
                item["existential"],
                item["neutral"],
                maximum_absolute_length_gap,
                maximum_relative_length_gap,
            )
        ]
        negative_pairs = [
            (item["negative_obligation"], item["neutral"])
            for item in items
            if length_matched(
                item["negative_obligation"],
                item["neutral"],
                maximum_absolute_length_gap,
                maximum_relative_length_gap,
            )
        ]
        positive_delta = bootstrap_delta(
            existential_pairs,
            lambda row: error_rate(row, "positive"),
            bootstrap_draws,
            seed + 100 * model_index,
        )
        negative_delta = bootstrap_delta(
            negative_pairs,
            lambda row: error_rate(row, "negative"),
            bootstrap_draws,
            seed + 100 * model_index + 1,
        )
        omission_delta = bootstrap_delta(
            existential_pairs,
            lambda row: error_rate(row, "omission"),
            bootstrap_draws,
            seed + 100 * model_index + 2,
        )
        gates = {
            "existential_matched_pairs_ge_minimum": len(existential_pairs)
            >= minimum_pairs,
            "negative_obligation_matched_pairs_ge_minimum": len(negative_pairs)
            >= minimum_pairs,
            "existential_false_positive_increase_ci_above_zero": bool(
                positive_delta["ci_low"] is not None
                and float(positive_delta["ci_low"]) > 0.0
            ),
            "negative_obligation_false_negative_increase_ci_above_zero": bool(
                negative_delta["ci_low"] is not None
                and float(negative_delta["ci_low"]) > 0.0
            ),
        }
        gates["bidirectional_presupposition_passed"] = all(gates.values())
        if gates["bidirectional_presupposition_passed"]:
            passed_models.append(model_id)
        model_results[model_id] = {
            "complete_prompt_triplets": len(items),
            "existential_vs_neutral_positive_error_delta": positive_delta,
            "negative_obligation_vs_neutral_negative_error_delta": negative_delta,
            "existential_vs_neutral_omission_delta": omission_delta,
            "gates": gates,
        }
    overall = len(passed_models) >= minimum_models
    return {
        "version": VERSION,
        "complete_model_item_prompt_triplets": len(complete),
        "length_match": {
            "maximum_absolute_token_gap": maximum_absolute_length_gap,
            "maximum_relative_token_gap": maximum_relative_length_gap,
            "within_same_model_and_image": True,
        },
        "model_results": model_results,
        "passed_models": passed_models,
        "minimum_passing_models": minimum_models,
        "clinical_presupposition_amplification_survives": overall,
        "claim_ceiling": (
            "This screen identifies a bidirectional prompt-conditioned clinical "
            "error signature. It does not establish a latent-layer mechanism."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-pairs", type=int, default=50)
    parser.add_argument("--minimum-models", type=int, default=2)
    parser.add_argument("--maximum-absolute-length-gap", type=int, default=12)
    parser.add_argument("--maximum-relative-length-gap", type=float, default=0.1)
    args = parser.parse_args()
    if args.bootstrap_draws <= 0 or args.minimum_pairs <= 0 or args.minimum_models <= 0:
        raise ValueError("draws, minimum pairs, and minimum models must be positive")
    if args.maximum_absolute_length_gap < 0 or not 0 <= args.maximum_relative_length_gap <= 1:
        raise ValueError("invalid length-match tolerance")
    result = analyze(
        load_jsonl(args.input),
        args.bootstrap_draws,
        args.seed,
        args.minimum_pairs,
        args.minimum_models,
        args.maximum_absolute_length_gap,
        args.maximum_relative_length_gap,
    )
    result["input"] = str(args.input.resolve())
    result["input_sha256"] = sha256_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
