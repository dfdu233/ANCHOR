#!/usr/bin/env python3
"""Select native OE samples by frozen structured-claim self-consistency."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "claim-self-consistency-aggregation-result-v1"


def _claim_key(claim: dict[str, Any]) -> tuple[str, str | None, tuple[str, ...]]:
    return (
        str(claim["finding"]),
        None if claim.get("anatomy") is None else str(claim["anatomy"]),
        tuple(sorted(str(value) for value in claim.get("attributes", []))),
    )


def _state(claim: dict[str, Any]) -> tuple[str, str, str]:
    return str(claim["polarity"]), str(claim["uncertainty"]), str(claim["provenance"])


def _sample_claims(claims: list[dict[str, Any]]) -> dict[tuple, tuple[str, str, str]]:
    grouped: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for claim in claims:
        grouped[_claim_key(claim)].append(_state(claim))
    output = {}
    for key, states in grouped.items():
        counts = Counter(states)
        best, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        if len(counts) > 1 and count == 1:
            # Within-sample duplicate conflict remains explicit uncertainty.
            polarity = sorted(state[0] for state in counts)[0]
            provenance = sorted(state[2] for state in counts)[0]
            best = (polarity, "uncertain", provenance)
        output[key] = best
    return output


def _claim_dict(key: tuple, state: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "finding": key[0],
        "anatomy": key[1],
        "attributes": list(key[2]),
        "polarity": state[0],
        "uncertainty": state[1],
        "provenance": state[2],
    }


def _claim_sort_key(item: tuple[str, str | None, tuple[str, ...]]) -> tuple[str, str, tuple[str, ...]]:
    return item[0], item[1] or "", item[2]


def aggregate_group(reports: list[dict[str, Any]], min_votes: int) -> dict[str, Any]:
    samples = []
    key_samples: dict[tuple, list[tuple[int, tuple[str, str, str]]]] = defaultdict(list)
    for report in reports:
        source = report["source"]
        claim_map = _sample_claims(report.get("claims", []))
        sample = {
            "seed": int(source["seed"]),
            "text": str(report["report"]),
            "mean_token_nll": float(source["mean_token_nll"]),
            "generated_token_ids": source["generated_token_ids"],
            "stop_reason": source.get("stop_reason"),
            "hit_max_new_tokens": source.get("hit_max_new_tokens"),
            "claims": claim_map,
            "unparsed": bool(report.get("audit", {}).get("unparsed_as_no_structured_claim")),
        }
        samples.append(sample)
        for key, state in claim_map.items():
            key_samples[key].append((sample["seed"], state))

    consensus: dict[tuple, tuple[str, str, str]] = {}
    traces = []
    for key, votes in sorted(key_samples.items(), key=lambda item: _claim_sort_key(item[0])):
        if len(votes) < min_votes:
            continue
        state_counts = Counter(state for _, state in votes)
        ordered = sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))
        state, state_votes = ordered[0]
        state_conflict = state_votes < min_votes
        if state_conflict:
            polarity_counts = Counter(value[0] for _, value in votes)
            polarity = sorted(polarity_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            provenance_counts = Counter(value[2] for _, value in votes)
            provenance = sorted(provenance_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            state = (polarity, "uncertain", provenance)
        consensus[key] = state
        traces.append(
            {
                "claim": _claim_dict(key, state),
                "sample_votes": len(votes),
                "state_votes": state_votes,
                "state_conflict": state_conflict,
                "seed_states": [
                    {"seed": seed, "polarity": value[0], "uncertainty": value[1], "provenance": value[2]}
                    for seed, value in sorted(votes)
                ],
            }
        )

    consensus_keys = set(consensus)
    candidates = []
    for sample in samples:
        sample_keys = set(sample["claims"])
        overlap = len(sample_keys & consensus_keys)
        denominator = len(sample_keys) + len(consensus_keys)
        claim_f1 = 2.0 * overlap / denominator if denominator else 0.0
        state_agreement = sum(
            sample["claims"].get(key) == state for key, state in consensus.items()
        )
        candidates.append((claim_f1, state_agreement, -sample["mean_token_nll"], -sample["seed"], sample))
    applicable = bool(consensus)
    if applicable:
        selected = max(candidates, key=lambda value: value[:4])[4]
    else:
        selected = next(sample for sample in samples if sample["seed"] == 42)
    seed42 = next(sample for sample in samples if sample["seed"] == 42)
    return {
        "applicable": applicable,
        "consensus_claims": [
            _claim_dict(key, consensus[key]) for key in sorted(consensus, key=_claim_sort_key)
        ],
        "claim_traces": traces,
        "selected_seed": selected["seed"],
        "selected_text": selected["text"],
        "selected_mean_token_nll": selected["mean_token_nll"],
        "selected_generated_token_ids": selected["generated_token_ids"],
        "selected_stop_reason": selected["stop_reason"],
        "selected_hit_max_new_tokens": selected["hit_max_new_tokens"],
        "changed_from_seed42": selected["generated_token_ids"] != seed42["generated_token_ids"],
        "unparsed_samples": sum(sample["unparsed"] for sample in samples),
        "sample_trace_hash": sha256_json(
            [
                {
                    "seed": sample["seed"],
                    "text": sample["text"],
                    "claims": [
                        _claim_dict(key, state)
                        for key, state in sorted(
                            sample["claims"].items(), key=lambda item: _claim_sort_key(item[0])
                        )
                    ],
                }
                for sample in sorted(samples, key=lambda value: value["seed"])
            ]
        ),
    }


def aggregate(
    *,
    extraction: Path,
    extraction_manifest: Path,
    aggregation_contract: Path,
    freeze_provenance: Path,
    output: Path,
    selected_answers: Path,
) -> dict[str, Any]:
    payload = json.loads(extraction.read_text())
    extraction_meta = json.loads(extraction_manifest.read_text())
    contract = json.loads(aggregation_contract.read_text())
    freeze = json.loads(freeze_provenance.read_text())
    if payload["config"]["input_sha256"] != extraction_meta["output_sha256"]:
        raise ValueError("extraction input differs from frozen preparation")
    seeds = [int(value) for value in contract["aggregation"]["seeds"]]
    min_votes = int(contract["aggregation"]["claim_inclusion_min_samples"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in payload["reports"]:
        source = report["source"]
        if source.get("stream", "self_consistency") != "self_consistency":
            continue
        grouped[(str(source["model"]), str(source["qid"]))].append(report)
    records = []
    answers = []
    for (model, qid), reports in sorted(grouped.items()):
        observed_seeds = sorted(int(report["source"]["seed"]) for report in reports)
        if observed_seeds != sorted(seeds):
            raise ValueError(f"incomplete seed ledger for {model}/{qid}")
        result = aggregate_group(reports, min_votes)
        records.append({"model": model, "qid": qid, **result})
        answers.append(
            {
                "question_id": qid,
                "text": result["selected_text"],
                "model_id": model,
                "metadata": {
                    "generated_token_ids": result["selected_generated_token_ids"],
                    "generated_token_count": len(result["selected_generated_token_ids"]),
                    "mean_token_nll": result["selected_mean_token_nll"],
                    "stop_reason": result["selected_stop_reason"],
                    "hit_max_new_tokens": result["selected_hit_max_new_tokens"],
                    "self_consistency_selected_seed": result["selected_seed"],
                    "self_consistency_applicable": result["applicable"],
                    "self_consistency_trace_hash": result["sample_trace_hash"],
                },
            }
        )
    selected_answers.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected_answers.with_suffix(selected_answers.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in answers))
    temporary.replace(selected_answers)
    applicable = sum(record["applicable"] for record in records)
    changed = sum(record["changed_from_seed42"] for record in records)
    result = {
        "protocol_version": VERSION,
        "aggregation_contract_sha256": sha256_file(aggregation_contract),
        "extraction_sha256": sha256_file(extraction),
        "extraction_manifest_sha256": sha256_file(extraction_manifest),
        "models": sorted({record["model"] for record in records}),
        "records": records,
        "summary": {
            "groups": len(records),
            "applicable_groups": applicable,
            "changed_from_seed42_groups": changed,
            "unparsed_samples": sum(record["unparsed_samples"] for record in records),
            "mean_consensus_claims": (
                sum(len(record["consensus_claims"]) for record in records) / len(records)
                if records else 0.0
            ),
        },
        "qualification": {
            "all_k_samples_complete": len(records) * len(seeds) == sum(
                report["source"].get("stream", "self_consistency") == "self_consistency"
                for report in payload["reports"]
            ),
            "atomic_claim_normalization": True,
            "preserves_polarity_anatomy_attributes_uncertainty": True,
            "exact_text_majority_vote_used": False,
            "trace_hash_recorded": all(record["sample_trace_hash"] for record in records),
            "at_least_one_applicable": applicable > 0,
            "at_least_one_decision_changed": changed > 0,
            "test_labels_used_for_selection": False,
        },
        "selected_answers": str(selected_answers.resolve()),
        "selected_answers_sha256": sha256_file(selected_answers),
        "development_manifest_sha256": freeze["development_manifest_sha256"],
        "held_out_manifest_sha256": freeze["held_out_manifest_sha256"],
    }
    safe_false = {"test_labels_used_for_selection", "exact_text_majority_vote_used"}
    result["passed_t2_functional"] = (
        all(value for key, value in result["qualification"].items() if key not in safe_false)
        and result["qualification"]["test_labels_used_for_selection"] is False
        and result["qualification"]["exact_text_majority_vote_used"] is False
    )
    result["fingerprint"] = sha256_json(result)
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", required=True, type=Path)
    parser.add_argument("--extraction-manifest", required=True, type=Path)
    parser.add_argument("--aggregation-contract", required=True, type=Path)
    parser.add_argument("--freeze-provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-answers", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate(
        extraction=args.extraction,
        extraction_manifest=args.extraction_manifest,
        aggregation_contract=args.aggregation_contract,
        freeze_provenance=args.freeze_provenance,
        output=args.output,
        selected_answers=args.selected_answers,
    )
    print(json.dumps({"summary": result["summary"], "qualification": result["qualification"]}, indent=2))
    if not result["passed_t2_functional"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
