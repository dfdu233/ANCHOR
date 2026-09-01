#!/usr/bin/env python3
"""Paired discovery audit for prompt-conditioned diagnostic commitment.

The audit compares two already-generated answer spaces on the *same* image.
Pairs are admitted only when both outputs independently contain the same
reader-labeled parent observation, the same diagnosis, and the same normalized
observation prefix.  It therefore does not compare arbitrary answer lengths or
different claims.  The surface classes (uncertain/definite) are lexical
discovery labels and do not replace physician certainty scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit_diagnostic_completion_substrate_v1 import (
    DEFAULT_GENERATIONS,
    DEFAULT_LABELS,
    extract_events,
    load_exact_panel_votes,
    load_jsonl,
    sha256_file,
)


VERSION = "pragmatic-diagnostic-commitment-pairs-audit-v1"


def _event_map(
    generations: Iterable[Mapping[str, Any]], condition: str
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in generations:
        if str(row.get("prompt_condition")) != condition:
            continue
        image_id = str(row["image_id"])
        for event in extract_events(str(row["text"])):
            key = (
                image_id,
                str(event["parent_label"]),
                str(event["child_label"]),
                str(event["observation_key"]),
            )
            if key in result:
                raise ValueError(f"duplicate within-condition paired event: {key}")
            result[key] = {
                **event,
                "image_id": image_id,
                "full_answer_token_count": int(row["generated_token_count"]),
                "sentence_word_count": len(str(event["sentence"]).split()),
            }
    return result


def exact_two_sided_sign_pvalue(up: int, down: int) -> float | None:
    n = up + down
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(up, down) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def audit_pairs(
    generations: Iterable[Mapping[str, Any]],
    votes: Mapping[str, Mapping[str, int]],
    *,
    reference_condition: str,
    focused_condition: str,
    maximum_sentence_word_gap: int,
    minimum_pairs: int,
) -> dict[str, Any]:
    rows = list(generations)
    reference = _event_map(rows, reference_condition)
    focused = _event_map(rows, focused_condition)
    shared = sorted(set(reference) & set(focused))
    pairs = []
    directions = Counter()
    for key in shared:
        first = reference[key]
        second = focused[key]
        gap = abs(int(first["sentence_word_count"]) - int(second["sentence_word_count"]))
        if gap > maximum_sentence_word_gap:
            continue
        image_id, parent_label, child_label, observation_key = key
        image_votes = votes[image_id]
        rank = {"uncertain": 0, "definite": 1}
        delta = rank[str(second["commitment"])] - rank[str(first["commitment"])]
        direction = "up" if delta > 0 else "down" if delta < 0 else "equal"
        directions[direction] += 1
        pairs.append(
            {
                "image_id": image_id,
                "parent_label": parent_label,
                "child_label": child_label,
                "observation_key": observation_key,
                "parent_votes": int(image_votes[parent_label]),
                "child_votes": int(image_votes[child_label]),
                "reference_surface_commitment": first["commitment"],
                "focused_surface_commitment": second["commitment"],
                "surface_commitment_delta": delta,
                "direction": direction,
                "reference_sentence": first["sentence"],
                "focused_sentence": second["sentence"],
                "reference_sentence_word_count": first["sentence_word_count"],
                "focused_sentence_word_count": second["sentence_word_count"],
                "sentence_word_gap": gap,
                "reference_full_answer_token_count": first[
                    "full_answer_token_count"
                ],
                "focused_full_answer_token_count": second["full_answer_token_count"],
            }
        )

    up = int(directions["up"])
    down = int(directions["down"])
    gates = {
        "paired_same_image_parent_child_observation_ge_minimum": len(pairs)
        >= minimum_pairs,
        "more_upward_than_downward_surface_shifts": up > down,
        "exact_sign_test_two_sided_p_below_0p05": bool(
            exact_two_sided_sign_pvalue(up, down) is not None
            and float(exact_two_sided_sign_pvalue(up, down)) < 0.05
        ),
    }
    gates["confirmatory_mechanism_authorized"] = all(gates.values())
    return {
        "version": VERSION,
        "scope": (
            "post-hoc discovery audit of lexical commitment surfaces; physician-"
            "scored certainty and a new frozen confirmatory split are required"
        ),
        "reference_condition": reference_condition,
        "focused_condition": focused_condition,
        "identity_contract": (
            "same image, VinDr parent label, VinDr child diagnosis label, and "
            "normalized observation prefix"
        ),
        "maximum_sentence_word_gap": maximum_sentence_word_gap,
        "shared_identity_pairs_before_length_gate": len(shared),
        "admitted_pairs": len(pairs),
        "directions": dict(directions),
        "exact_two_sided_sign_pvalue_excluding_ties": exact_two_sided_sign_pvalue(
            up, down
        ),
        "pairs": pairs,
        "gates": gates,
        "next_action": (
            "Freeze a new image-disjoint confirmatory substrate before any more outputs; "
            "do not interpret this four-case discovery as evidence."
            if not gates["confirmatory_mechanism_authorized"]
            else "Open the pre-registered hidden-state localization stage."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--reference-condition", default="neutral")
    parser.add_argument("--focused-condition", default="existential")
    parser.add_argument("--maximum-sentence-word-gap", type=int, default=4)
    parser.add_argument("--minimum-pairs", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_sentence_word_gap < 0 or args.minimum_pairs <= 0:
        raise ValueError("invalid pair gate")
    result = audit_pairs(
        load_jsonl(args.generations),
        load_exact_panel_votes(args.labels),
        reference_condition=args.reference_condition,
        focused_condition=args.focused_condition,
        maximum_sentence_word_gap=args.maximum_sentence_word_gap,
        minimum_pairs=args.minimum_pairs,
    )
    result["generations"] = str(args.generations.resolve())
    result["generations_sha256"] = sha256_file(args.generations)
    result["labels"] = str(args.labels.resolve())
    result["labels_sha256"] = sha256_file(args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
