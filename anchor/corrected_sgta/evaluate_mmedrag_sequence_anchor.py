#!/usr/bin/env python3
"""Evaluate unrestricted report candidates with the repository's report metrics.

The implementation intentionally matches ``utils/Metrics_Compute/
cal_report_metrics.py``: lowercase, insert a space before periods, NLTK
sentence BLEU without smoothing, ``rouge`` F1, and NLTK METEOR.  In addition
to corpus means, it reports paired bootstrap intervals and per-metric candidate
oracles.  The latter are diagnostics only and must not be reported as a method.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge


METRICS = ("bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL", "meteor")
ROUGE = Rouge()


def prep(text: str) -> list[str]:
    return [token for token in str(text).lower().replace(".", " .").split(" ") if token]


def clean(text: str) -> str:
    return str(text).replace("Findings:", "").replace("Impression:", "").strip()


def score_one(prediction: str, reference: str) -> dict[str, float]:
    prediction = clean(prediction)
    hyp = prep(prediction)
    ref = prep(reference)
    if not hyp or not ref:
        return {name: 0.0 for name in METRICS}
    bleu_weights = {
        "bleu1": (1.0,),
        "bleu2": (0.5, 0.5),
        "bleu3": (1 / 3, 1 / 3, 1 / 3),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }
    out = {
        name: float(sentence_bleu([ref], hyp, weights=weights))
        for name, weights in bleu_weights.items()
    }
    rouge = ROUGE.get_scores(prediction.lower()[:2048], reference.lower())[0]
    out.update(
        rouge1=float(rouge["rouge-1"]["f"]),
        rouge2=float(rouge["rouge-2"]["f"]),
        rougeL=float(rouge["rouge-l"]["f"]),
        meteor=float(single_meteor_score(reference=ref, hypothesis=hyp)),
    )
    return out


def get_text(record: dict[str, Any], variant: str) -> str:
    if variant in record["candidates"]:
        return record["candidates"][variant]
    if variant == "sequence_anchor":
        return record["sequence_anchor"]
    if variant == "source_neighbor":
        return record["source_neighbor"]["report"]
    raise KeyError(variant)


def summarize(rows: list[dict[str, Any]], seed: int, bootstrap: int) -> dict[str, Any]:
    common_candidates = set(rows[0]["candidates"])
    for row in rows[1:]:
        common_candidates.intersection_update(row["candidates"])
    if "baseline" not in common_candidates:
        raise ValueError("every record must contain a baseline candidate")
    variants = ["baseline", *sorted(common_candidates.difference(("baseline",)))]
    if all("sequence_anchor" in row for row in rows):
        variants.append("sequence_anchor")
    if all("source_neighbor" in row for row in rows):
        variants.append("source_neighbor")
    by_variant = {
        variant: {metric: [] for metric in METRICS}
        for variant in variants
    }
    lengths = {variant: [] for variant in variants}
    for row in rows:
        for variant in variants:
            text = get_text(row, variant)
            scored = score_one(text, row["ground_truth"])
            for metric, value in scored.items():
                by_variant[variant][metric].append(value)
            lengths[variant].append(len(prep(text)))

    means = {
        variant: {
            **{metric: float(np.mean(values)) for metric, values in scores.items()},
            "avg_bleu1_4": float(
                np.mean([np.mean(scores[f"bleu{i}"]) for i in range(1, 5)])
            ),
            "mean_tokens": float(np.mean(lengths[variant])),
        }
        for variant, scores in by_variant.items()
    }

    rng = np.random.default_rng(seed)
    n = len(rows)
    paired = {}
    for variant in variants:
        if variant == "baseline":
            continue
        paired[variant] = {}
        for metric in METRICS:
            delta = (
                np.asarray(by_variant[variant][metric])
                - np.asarray(by_variant["baseline"][metric])
            )
            samples = np.empty(bootstrap, dtype=np.float64)
            for i in range(bootstrap):
                samples[i] = delta[rng.integers(0, n, n)].mean()
            paired[variant][metric] = {
                "delta": float(delta.mean()),
                "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
                "win_tie_loss": [
                    int((delta > 0).sum()),
                    int((delta == 0).sum()),
                    int((delta < 0).sum()),
                ],
            }

    oracle = None
    if "guided" in by_variant:
        oracle = {}
        for metric in METRICS:
            baseline = np.asarray(by_variant["baseline"][metric])
            guided = np.asarray(by_variant["guided"][metric])
            oracle_values = np.maximum(baseline, guided)
            oracle[metric] = {
                "mean": float(oracle_values.mean()),
                "headroom_over_baseline": float((oracle_values - baseline).mean()),
                "guided_better": int((guided > baseline).sum()),
            }
    return {
        "n": n,
        "selection": dict(
            Counter(row.get("selected", "not_applicable") for row in rows)
        ),
        "means": means,
        "paired_bootstrap_vs_baseline": paired,
        "two_candidate_metric_oracle_diagnostic": oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270725)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    rows = payload["records"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)

    result = {
        "version": "mmedrag-sequence-anchor-eval-v1",
        "source_fingerprint": payload["fingerprint"],
        "metric_implementation": "utils/Metrics_Compute/cal_report_metrics.py compatible",
        "ground_truth_used_for_generation_or_selection": False,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "overall": summarize(rows, args.seed, args.bootstrap),
        "by_dataset": {
            name: summarize(group, args.seed, args.bootstrap)
            for name, group in sorted(groups.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
