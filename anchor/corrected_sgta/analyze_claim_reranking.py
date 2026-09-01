#!/usr/bin/env python3
"""Evaluate fixed-K ontology reranking with original and null-centered claim scores."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_claim_transport import claim_set_metrics
from corrected_sgta.analyze_no_free_grounding import _binary, sha256_file


VERSION = "fixed-k-ontology-reranking-screen-v1"
SCORE_NAMES = ("original_margin", "null_margin", "null_centered_margin")


def align(
    baseline_report: dict[str, Any], score_rows: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline = {int(row["question_id"]): row for row in baseline_report["details"]}
    scores = {int(row["question_id"]): row for row in score_rows if row.get("status") == "ok"}
    question_by_id = {int(row["qid"]): row for row in questions}
    if set(baseline) != set(scores) or set(baseline) != set(question_by_id):
        raise ValueError("baseline, scores, and questions are not exactly aligned")
    rows = []
    for qid in sorted(baseline):
        truth = _binary(baseline[qid].get("ground_truth"))
        if truth == "invalid":
            raise ValueError(f"invalid truth for qid={qid}")
        rows.append({
            "question_id": qid,
            "image": str(question_by_id[qid].get("img_name") or question_by_id[qid].get("img_id")),
            "truth": truth,
            "baseline": _binary(baseline[qid].get("prediction")),
            "scores": {name: float(scores[qid]["scores"][name]) for name in SCORE_NAMES},
        })
    return rows


def evaluate_image(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {index for index, row in enumerate(rows) if row["baseline"] == "yes"}
    output = {"baseline": claim_set_metrics(rows, baseline)}
    for name in SCORE_NAMES:
        order = sorted(
            range(len(rows)),
            key=lambda index: (-rows[index]["scores"][name], rows[index]["question_id"]),
        )
        selected = set(order[: len(baseline)])
        output[name] = claim_set_metrics(rows, selected)
        if output[name]["n_predicted_claims"] != output["baseline"]["n_predicted_claims"]:
            raise AssertionError("fixed-K reranking changed claim count")
    score_point = {
        index for index, row in enumerate(rows)
        if row["scores"]["original_margin"] >= 0.0
    }
    centered_order = sorted(
        range(len(rows)),
        key=lambda index: (
            -rows[index]["scores"]["null_centered_margin"],
            rows[index]["question_id"],
        ),
    )
    centered_at_score_k = set(centered_order[: len(score_point)])
    output["original_sign_point"] = claim_set_metrics(rows, score_point)
    output["null_centered_at_original_sign_k"] = claim_set_metrics(
        rows, centered_at_score_k
    )
    output["image"] = rows[0]["image"]
    return output


def aggregate(images: list[dict[str, Any]], name: str) -> dict[str, Any]:
    keys = ("n_claims", "n_true_claims", "n_predicted_claims", "tp", "fp", "fn")
    total = {key: sum(image[name][key] for image in images) for key in keys}
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    total.update({
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    })
    return total


def bootstrap_deltas(
    images: list[dict[str, Any]], candidate: str, reference: str, draws: int, seed: int
) -> dict[str, dict[str, float]]:
    def one(sample: list[dict[str, Any]]) -> tuple[float, float, float]:
        left, right = aggregate(sample, candidate), aggregate(sample, reference)
        return (
            left["precision"] - right["precision"],
            left["recall"] - right["recall"],
            left["f1"] - right["f1"],
        )

    observed = one(images)
    rng = np.random.default_rng(seed)
    sampled = [[], [], []]
    for _ in range(draws):
        indices = rng.integers(0, len(images), len(images))
        values = one([images[index] for index in indices])
        for target, value in zip(sampled, values):
            if np.isfinite(value):
                target.append(value)
    return {
        name: {
            "estimate": float(observed[index]),
            "ci_low": float(np.quantile(sampled[index], 0.025)),
            "ci_high": float(np.quantile(sampled[index], 0.975)),
        }
        for index, name in enumerate(("precision", "recall", "f1"))
    }


def analyze(
    baseline_report: dict[str, Any],
    score_rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rows = align(baseline_report, score_rows, questions)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append(row)
    images = [evaluate_image(grouped[name]) for name in sorted(grouped)]
    metric_names = (
        "baseline", *SCORE_NAMES,
        "original_sign_point", "null_centered_at_original_sign_k",
    )
    metrics = {name: aggregate(images, name) for name in metric_names}
    versus_baseline = {
        name: bootstrap_deltas(images, name, "baseline", draws, seed + index * 10)
        for index, name in enumerate(SCORE_NAMES)
    }
    original_vs_null = bootstrap_deltas(
        images, "original_margin", "null_margin", draws, seed + 100
    )
    centered_vs_original_same_decoded_k = bootstrap_deltas(
        images, "null_centered_margin", "original_margin", draws, seed + 110
    )
    score_only_centering = bootstrap_deltas(
        images,
        "null_centered_at_original_sign_k",
        "original_sign_point",
        draws,
        seed + 120,
    )
    centered = versus_baseline["null_centered_margin"]
    passed = centered["precision"]["ci_low"] > 0 and centered["recall"]["ci_low"] > 0
    return {
        "n_claims": len(rows),
        "n_images": len(images),
        "metrics": metrics,
        "candidate_minus_decoded_baseline_cluster_bootstrap": versus_baseline,
        "original_minus_null_same_K_cluster_bootstrap": original_vs_null,
        "null_centered_minus_original_same_decoded_K_cluster_bootstrap": centered_vs_original_same_decoded_k,
        "score_only_null_centered_minus_original_sign_cluster_bootstrap": score_only_centering,
        "screening_gate": {
            "passed": passed,
            "rule": (
                "null-centered fixed-K reranking must improve both precision and recall "
                "with image-bootstrap 95% confidence"
            ),
        },
        "identity": "All rerankers preserve each image's K, so delta(FP)=delta(FN)=-delta(TP).",
        "claim_ceiling": (
            "SLAKE exact Yes/No questions are used as a grouped claim universe. "
            "This is not yet natural OE draft extraction or a radiology report result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=719)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score_rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": analyze(
            json.loads(args.baseline.read_text()),
            score_rows,
            json.loads(args.questions.read_text()),
            draws=args.bootstrap_draws,
            seed=args.seed,
        ),
        "provenance": {
            "code_sha256": sha256_file(Path(__file__)),
            "baseline_sha256": sha256_file(args.baseline),
            "scores_sha256": sha256_file(args.scores),
            "questions_sha256": sha256_file(args.questions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
