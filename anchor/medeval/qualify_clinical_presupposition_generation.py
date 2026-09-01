"""Fail-closed qualification for generation-only presupposition probes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "clinical-presupposition-generation-qualification-v1"
CONDITIONS = ("neutral", "existential", "negative_obligation")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _length_matched(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_length = int(first["generated_token_count"])
    second_length = int(second["generated_token_count"])
    gap = abs(first_length - second_length)
    return gap <= 12 and gap / max(1, first_length, second_length) <= 0.1


def qualify(rows: list[dict[str, Any]], minimum_pairs: int = 50) -> dict[str, Any]:
    _require(rows, "generation file is empty")
    _require(minimum_pairs > 0, "minimum_pairs must be positive")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    fingerprints = set()
    for row in rows:
        item_id = str(row.get("item_id", ""))
        condition = str(row.get("prompt_condition", ""))
        _require(item_id, "generation has empty item_id")
        _require(condition in CONDITIONS, f"{item_id}: unknown prompt condition")
        _require(condition not in grouped[item_id], f"{item_id}: duplicate {condition}")
        text = row.get("text")
        token_ids = row.get("generated_token_ids")
        token_count = row.get("generated_token_count")
        _require(isinstance(text, str) and text.strip(), f"{item_id}/{condition}: empty text")
        _require(isinstance(token_ids, list), f"{item_id}/{condition}: missing token IDs")
        _require(isinstance(token_count, int) and token_count == len(token_ids), f"{item_id}/{condition}: token accounting mismatch")
        _require(row.get("clinical_claim_evaluation_status") == "pending_shared_audit", f"{item_id}/{condition}: generation assigned clinical truth")
        _require(row.get("ground_truth_used_for_generation_or_selection") is False, f"{item_id}/{condition}: ground truth leakage")
        _require(row.get("automatic_labeler_used") is False, f"{item_id}/{condition}: automatic clinical labeler used")
        universe = row.get("claim_universe_sha256")
        _require(isinstance(universe, str) and len(universe) == 64, f"{item_id}/{condition}: invalid claim-universe hash")
        fingerprint = row.get("fingerprint")
        _require(isinstance(fingerprint, str) and fingerprint, f"{item_id}/{condition}: missing fingerprint")
        fingerprints.add(fingerprint)
        grouped[item_id][condition] = row
    _require(len(fingerprints) == 1, "generation fingerprints differ")
    _require(all(set(values) == set(CONDITIONS) for values in grouped.values()), "incomplete prompt triplet")
    for item_id, values in grouped.items():
        _require(
            len({str(row["claim_universe_sha256"]) for row in values.values()}) == 1,
            f"{item_id}: claim universe differs across prompts",
        )

    by_condition = {}
    for condition in CONDITIONS:
        subset = [values[condition] for values in grouped.values()]
        by_condition[condition] = {
            "n": len(subset),
            "mean_generated_tokens": sum(int(row["generated_token_count"]) for row in subset) / len(subset),
            "cap_hits": sum(bool(row.get("hit_max_new_tokens")) for row in subset),
            "surface_refusal_matches": sum(bool(row.get("surface_refusal_match")) for row in subset),
            "response_form_at_most_30_whitespace_words": sum(
                len(str(row["text"]).split()) <= 30 for row in subset
            ),
        }
    pairs = {}
    for condition in ("existential", "negative_obligation"):
        matched_ids = sorted(
            item_id
            for item_id, values in grouped.items()
            if _length_matched(values[condition], values["neutral"])
        )
        pairs[f"{condition}_vs_neutral"] = {
            "matched_pairs": len(matched_ids),
            "total_triplets": len(grouped),
            "matched_item_ids": matched_ids,
            "minimum_required": minimum_pairs,
            "passed": len(matched_ids) >= minimum_pairs,
        }
    length_gate = all(value["passed"] for value in pairs.values())
    return {
        "version": VERSION,
        "passed": length_gate,
        "items": len(grouped),
        "generations": len(rows),
        "fingerprint": next(iter(fingerprints)),
        "condition_diagnostics": by_condition,
        "strict_within_image_length_match": {
            "maximum_absolute_generated_token_gap": 12,
            "maximum_relative_generated_token_gap": 0.1,
            "contrasts": pairs,
        },
        "human_claim_audit_authorized": length_gate,
        "second_model_generation_authorized_from_this_model": length_gate,
        "interpretation": (
            "Generation integrity and pre-registered length-overlap qualification only. "
            "No clinical claim correctness or mechanism is inferred."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=50)
    args = parser.parse_args()
    result = qualify(load_jsonl(args.generations), args.minimum_pairs)
    result.update({
        "generations_path": str(args.generations.resolve()),
        "generations_sha256": sha256_file(args.generations),
    })
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
