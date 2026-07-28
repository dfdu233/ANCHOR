#!/usr/bin/env python3
"""Paired clinical and lexical analysis for LET report evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rouge import Rouge


CLINICAL_METRICS = (
    "radgraph_simple",
    "radgraph_partial",
    "radgraph_complete",
    "ratescore",
    "chexbert_example_f1_14",
    "chexbert_exact_match_5",
)
ROUGE = Rouge()


def record_key(item_id: str) -> str:
    fields = str(item_id).rsplit(":", 1)
    if len(fields) != 2:
        raise ValueError(f"invalid item_id: {item_id}")
    return fields[0]


def load_clinical(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = record_key(row["item_id"])
        if key in output:
            raise ValueError(f"duplicate clinical key: {key}")
        output[key] = row
    return output


def rouge_l(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    return float(
        ROUGE.get_scores(prediction.lower()[:2048], reference.lower()[:2048])[0][
            "rouge-l"
        ]["f"]
    )


def paired_summary(
    rows: list[dict[str, Any]], bootstrap: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {"n": len(rows), "metrics": {}}
    for metric in (*CLINICAL_METRICS, "rouge_l", "word_count"):
        baseline = np.asarray(
            [float(row["baseline_metrics"][metric]) for row in rows],
            dtype=np.float64,
        )
        let = np.asarray(
            [float(row["let_metrics"][metric]) for row in rows],
            dtype=np.float64,
        )
        delta = let - baseline
        samples = np.empty(bootstrap, dtype=np.float64)
        for index in range(bootstrap):
            selected = rng.integers(0, len(rows), len(rows))
            samples[index] = delta[selected].mean()
        output["metrics"][metric] = {
            "baseline": float(baseline.mean()),
            "let": float(let.mean()),
            "delta": float(delta.mean()),
            "paired_bootstrap_95ci": [
                float(value) for value in np.quantile(samples, (0.025, 0.975))
            ],
            "win_tie_loss": [
                int(np.sum(delta > 0)),
                int(np.sum(delta == 0)),
                int(np.sum(delta < 0)),
            ],
        }
    output["normal_template_rate"] = {
        "baseline": float(
            np.mean([row["baseline_normal_template"] for row in rows])
        ),
        "let": float(np.mean([row["let_normal_template"] for row in rows])),
    }
    output["unique_output_count"] = {
        "baseline": len({row["baseline_text"] for row in rows}),
        "let": len({row["let_text"] for row in rows}),
    }
    output["text_change_rate"] = float(
        np.mean([row["baseline_text"] != row["let_text"] for row in rows])
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--let-records", type=Path, required=True)
    parser.add_argument("--baseline-aggregate", type=Path, required=True)
    parser.add_argument("--let-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    baseline = load_clinical(args.baseline_records)
    let = load_clinical(args.let_records)
    if set(baseline) != set(let):
        raise ValueError("baseline and LET clinical keys differ")
    paired = []
    for row in predictions["records"]:
        key = f"{row['dataset']}:{row['id']}"
        if key not in baseline:
            raise ValueError(f"clinical score missing for {key}")
        reference = row["ground_truth"]
        baseline_text = row["candidates"]["baseline"]
        let_text = row["candidates"]["let"]
        baseline_metrics = dict(baseline[key]["metrics"])
        let_metrics = dict(let[key]["metrics"])
        baseline_metrics.update(
            rouge_l=rouge_l(baseline_text, reference),
            word_count=len(baseline_text.split()),
        )
        let_metrics.update(
            rouge_l=rouge_l(let_text, reference),
            word_count=len(let_text.split()),
        )
        paired.append(
            {
                "dataset": row["dataset"],
                "id": row["id"],
                "baseline_text": baseline_text,
                "let_text": let_text,
                "baseline_normal_template": row["normal_template"]["baseline"],
                "let_normal_template": row["normal_template"]["let"],
                "baseline_metrics": baseline_metrics,
                "let_metrics": let_metrics,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[row["dataset"]].append(row)
    baseline_aggregate = json.loads(args.baseline_aggregate.read_text())
    let_aggregate = json.loads(args.let_aggregate.read_text())
    result = {
        "version": "anchor-let-report-analysis-v1",
        "source_fingerprint": predictions["fingerprint"],
        "primary_metric": "radgraph_simple",
        "primary_endpoint_pre_registered_before_scoring": True,
        "ground_truth_used_for_generation_or_selection": False,
        "ce_parser_used": False,
        "label_logits_used": False,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "metric_direction_validation": baseline_aggregate[
            "direction_validation"
        ],
        "clinical_scorer_fingerprints": {
            "baseline": baseline_aggregate["fingerprint"],
            "let": let_aggregate["fingerprint"],
        },
        "overall": paired_summary(paired, args.bootstrap, args.seed),
        "by_dataset": {
            name: paired_summary(rows, args.bootstrap, args.seed)
            for name, rows in sorted(groups.items())
        },
        "interpretation_guard": (
            "Pilot evidence only. A longer or less templated report is not "
            "necessarily more factual; clinical metric deltas and confidence "
            "intervals govern the claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
