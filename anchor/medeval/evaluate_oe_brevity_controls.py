#!/usr/bin/env python3
"""Post-hoc brevity controls for open medical VQA.

These controls deliberately remove answer suffixes without using the image or
reference. They are not mitigation methods. Their purpose is to expose gains
that can be explained by saying less, under a transparent lexical-coverage
check and image-clustered paired uncertainty.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from anchor.medeval.evaluate_oe_vqa import (
    _load_json,
    _load_jsonl,
    _row_id,
    _prediction,
    align_and_score,
    paired_summary,
    summarize,
)
from anchor.medeval.hashing import sha256_file
from anchor.medeval.stats import cluster_bootstrap_metric


PROTOCOL_ID = "anchor-oe-brevity-controls-v1"


def first_sentence(text: str) -> str:
    match = re.match(r"^.*?[.!?](?:\s|$)", text.strip(), flags=re.DOTALL)
    return match.group(0).strip() if match else text.strip()


def first_words(text: str, count: int) -> str:
    return " ".join(text.strip().split()[:count])


POLICIES: dict[str, Callable[[str], str]] = {
    "original": lambda text: text,
    "first_sentence": first_sentence,
}
for _word_budget in range(8, 65, 8):
    POLICIES[f"first_{_word_budget}_words"] = (
        lambda text, count=_word_budget: first_words(text, count)
    )


def transformed_answers(
    answers: list[dict[str, Any]], transform: Callable[[str], str]
) -> list[dict[str, Any]]:
    return [
        {"question_id": _row_id(row), "text": transform(_prediction(row))}
        for row in answers
    ]


def paired_diagnostics(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_id = {row["question_id"]: row for row in baseline}
    paired = []
    for row in candidate:
        other = baseline_by_id[row["question_id"]]
        if row["cluster_id"] != other["cluster_id"]:
            raise ValueError(f"cluster mismatch for {row['question_id']!r}")
        paired.append({
            "cluster_id": row["cluster_id"],
            "delta_reference_phrase_covered": (
                float(row["reference_phrase_covered"])
                - float(other["reference_phrase_covered"])
            ),
            "delta_prediction_tokens": (
                float(row["prediction_tokens"]) - float(other["prediction_tokens"])
            ),
        })
    return {
        "direction": "control_minus_original",
        "reference_phrase_coverage": cluster_bootstrap_metric(
            paired,
            lambda sample: sum(x["delta_reference_phrase_covered"] for x in sample)
            / len(sample),
            replicates=replicates,
            seed=seed,
        ),
        "prediction_tokens": cluster_bootstrap_metric(
            paired,
            lambda sample: sum(x["delta_prediction_tokens"] for x in sample) / len(sample),
            replicates=replicates,
            seed=seed + 1,
        ),
    }


def evaluate(
    manifest: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    original_text = {_row_id(row): _prediction(row) for row in answers}
    scored = {
        name: align_and_score(manifest, transformed_answers(answers, transform))
        for name, transform in POLICIES.items()
    }
    baseline = scored["original"]
    output: dict[str, Any] = {}
    for index, (name, rows) in enumerate(scored.items()):
        absolute = summarize(rows, replicates=replicates, seed=seed + index * 10)
        record: dict[str, Any] = {
            "absolute": absolute,
            "changed_rate": sum(
                row["prediction"] != original_text[row["question_id"]] for row in rows
            ) / len(rows),
        }
        if name != "original":
            record["paired_lexical_vs_original"] = paired_summary(
                rows,
                baseline,
                replicates=replicates,
                seed=seed + index * 10,
            )
            record["paired_diagnostics_vs_original"] = paired_diagnostics(
                rows,
                baseline,
                replicates=replicates,
                seed=seed + index * 10 + 3,
            )
            coverage_delta = record["paired_diagnostics_vs_original"][
                "reference_phrase_coverage"
            ]["estimate"]
            record["lexically_coverage_matched_point_estimate"] = coverage_delta >= -0.01
            record["lexical_coverage_tolerance"] = -0.01
        output[name] = record
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--answers", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = {
        "protocol_id": PROTOCOL_ID,
        "command": [sys.executable, *sys.argv],
        "evaluator_source": str(Path(__file__).resolve()),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "interpretation": (
            "post-hoc output-shortening controls only; lexical reference-phrase coverage "
            "is not clinical claim coverage and these controls are not hallucination methods"
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "answers": [str(path.resolve()) for path in args.answers],
        "answer_sha256": [sha256_file(path) for path in args.answers],
        "policies": evaluate(
            _load_json(args.manifest),
            _load_jsonl(args.answers),
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        name: {
            "token_f1": row["absolute"]["metrics"]["token_f1"]["estimate"],
            "median_tokens": row["absolute"]["output_diagnostics"][
                "median_prediction_tokens"
            ],
            "reference_phrase_coverage": row["absolute"]["output_diagnostics"][
                "reference_phrase_coverage_rate"
            ],
            "coverage_matched_point_estimate": row.get(
                "lexically_coverage_matched_point_estimate"
            ),
        }
        for name, row in payload["policies"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
