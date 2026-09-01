#!/usr/bin/env python3
"""Screen coverage-preserving claim transport on grouped open-ended claim sets.

For each image, the baseline positive-claim count K is conserved.  A mitigation
change is admitted only as a paired removal and addition.  With K fixed, every
increase in true positives decreases both false positives and false negatives
by the same amount.  Binary mitigation outputs contain no within-flip ranking,
so this script reports a frozen tie-break result, random-tie expectation, and
oracle/worst ceilings separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_no_free_grounding import _binary, sha256_file


VERSION = "coverage-preserving-claim-transport-screen-v1"


def claim_set_metrics(rows: list[dict[str, Any]], selected: set[int]) -> dict[str, Any]:
    truth = {index for index, row in enumerate(rows) if row["truth"] == "yes"}
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    return {
        "n_claims": len(rows),
        "n_true_claims": len(truth),
        "n_predicted_claims": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / len(selected) if selected else None,
        "recall": tp / len(truth) if truth else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def _rank(image: str, question_id: int, role: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{image}|{question_id}|{role}".encode()).hexdigest()


def image_transport(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    baseline = {index for index, row in enumerate(rows) if row["baseline"] == "yes"}
    method = {index for index, row in enumerate(rows) if row["method"] == "yes"}
    removed = sorted(baseline - method)
    added = sorted(method - baseline)
    swaps = min(len(removed), len(added))
    image = rows[0]["image"]
    chosen_removed = sorted(
        removed, key=lambda index: _rank(image, rows[index]["question_id"], "remove", seed)
    )[:swaps]
    chosen_added = sorted(
        added, key=lambda index: _rank(image, rows[index]["question_id"], "add", seed)
    )[:swaps]
    transported = (baseline - set(chosen_removed)) | set(chosen_added)
    truth = {index for index, row in enumerate(rows) if row["truth"] == "yes"}
    removed_true = len(set(removed) & truth)
    added_true = len(set(added) & truth)
    removed_false = len(removed) - removed_true
    added_false = len(added) - added_true
    expected_delta = (
        swaps * (added_true / len(added) - removed_true / len(removed))
        if swaps else 0.0
    )
    best_delta = (
        min(swaps, added_true) - max(0, swaps - removed_false)
    )
    worst_delta = (
        max(0, swaps - added_false) - min(swaps, removed_true)
    )
    result = {
        "image": image,
        "n_claims": len(rows),
        "baseline": claim_set_metrics(rows, baseline),
        "method": claim_set_metrics(rows, method),
        "transport": claim_set_metrics(rows, transported),
        "changes": {
            "removed": len(removed),
            "added": len(added),
            "admitted_swaps": swaps,
            "unmatched_removals": len(removed) - swaps,
            "unmatched_additions": len(added) - swaps,
            "removed_true": removed_true,
            "added_true": added_true,
        },
        "transport_tp_delta": len(transported & truth) - len(baseline & truth),
        "random_tie_expected_tp_delta": expected_delta,
        "oracle_best_tp_delta": best_delta,
        "worst_tp_delta": worst_delta,
    }
    if len(transported) != len(baseline):
        raise AssertionError("transport failed to conserve per-image claim count")
    delta_tp = result["transport"]["tp"] - result["baseline"]["tp"]
    if (
        result["transport"]["fp"] - result["baseline"]["fp"] != -delta_tp
        or result["transport"]["fn"] - result["baseline"]["fn"] != -delta_tp
    ):
        raise AssertionError("fixed-cardinality FP/FN conservation identity failed")
    return result


def align(
    baseline_report: dict[str, Any],
    method_report: dict[str, Any],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    question_by_id = {int(row["qid"]): row for row in questions}
    baseline = {int(row["question_id"]): row for row in baseline_report["details"]}
    method = {int(row["question_id"]): row for row in method_report["details"]}
    if set(baseline) != set(method) or not set(baseline).issubset(question_by_id):
        raise ValueError("reports and question file are not exactly aligned")
    rows = []
    for question_id in sorted(baseline):
        truth = _binary(baseline[question_id].get("ground_truth"))
        if truth == "invalid" or truth != _binary(method[question_id].get("ground_truth")):
            raise ValueError(f"invalid or inconsistent truth for question {question_id}")
        question = question_by_id[question_id]
        rows.append({
            "question_id": question_id,
            "image": str(question.get("img_name") or question.get("img_id")),
            "truth": truth,
            "baseline": _binary(baseline[question_id].get("prediction")),
            "method": _binary(method[question_id].get("prediction")),
        })
    return rows


def _aggregate(image_results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    totals = {name: 0 for name in ("n_claims", "n_true_claims", "n_predicted_claims", "tp", "fp", "fn")}
    for result in image_results:
        for name in totals:
            totals[name] += result[key][name]
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    totals.update({
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    })
    return totals


def expected_delta_bootstrap(
    image_results: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    def deltas(sample: list[dict[str, Any]]) -> tuple[float, float]:
        delta_tp = sum(row["random_tie_expected_tp_delta"] for row in sample)
        predicted = sum(row["baseline"]["n_predicted_claims"] for row in sample)
        truth = sum(row["baseline"]["n_true_claims"] for row in sample)
        return (
            delta_tp / predicted if predicted else float("nan"),
            delta_tp / truth if truth else float("nan"),
        )

    observed_precision, observed_recall = deltas(image_results)
    precision, recall = [], []
    for _ in range(draws):
        indices = rng.integers(0, len(image_results), len(image_results))
        p_delta, r_delta = deltas([image_results[index] for index in indices])
        if np.isfinite(p_delta):
            precision.append(p_delta)
        if np.isfinite(r_delta):
            recall.append(r_delta)
    return {
        "precision_delta": {
            "estimate": observed_precision,
            "ci_low": float(np.quantile(precision, 0.025)),
            "ci_high": float(np.quantile(precision, 0.975)),
        },
        "recall_delta": {
            "estimate": observed_recall,
            "ci_low": float(np.quantile(recall, 0.025)),
            "ci_high": float(np.quantile(recall, 0.975)),
        },
    }


def analyze(
    baseline_report: dict[str, Any],
    method_report: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    rows = align(baseline_report, method_report, questions)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append(row)
    image_results = [
        image_transport(grouped[image], seed=seed) for image in sorted(grouped)
    ]
    changes = {
        key: sum(row["changes"][key] for row in image_results)
        for key in image_results[0]["changes"]
    }
    opportunities = {
        "images": len(image_results),
        "images_with_admitted_swap": sum(row["changes"]["admitted_swaps"] > 0 for row in image_results),
        **changes,
        "random_tie_expected_tp_delta": sum(row["random_tie_expected_tp_delta"] for row in image_results),
        "oracle_best_tp_delta": sum(row["oracle_best_tp_delta"] for row in image_results),
        "worst_tp_delta": sum(row["worst_tp_delta"] for row in image_results),
    }
    bootstrap = expected_delta_bootstrap(image_results, draws=draws, seed=seed + 1)
    return {
        "n_claims": len(rows),
        "n_images": len(image_results),
        "baseline": _aggregate(image_results, "baseline"),
        "raw_method": _aggregate(image_results, "method"),
        "frozen_tie_transport": _aggregate(image_results, "transport"),
        "opportunities": opportunities,
        "random_tie_expected_cluster_bootstrap": bootstrap,
        "screening_gate": {
            "passed": bootstrap["recall_delta"]["ci_low"] > 0,
            "rule": "random-tie expected recall gain has image-bootstrap 95% CI entirely above zero",
        },
        "identity": (
            "At fixed per-image K, delta(FP) = delta(FN) = -delta(TP). "
            "This prevents hallucination gains obtained by deleting claims."
        ),
        "image_results": image_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-name", default="greedy")
    parser.add_argument("--method-name", default="VCD")
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison": {"baseline": args.baseline_name, "method": args.method_name},
        "result": analyze(
            json.loads(args.baseline.read_text()),
            json.loads(args.method.read_text()),
            json.loads(args.questions.read_text()),
            seed=args.seed,
            draws=args.bootstrap_draws,
        ),
        "provenance": {
            "code_sha256": sha256_file(Path(__file__)),
            "baseline_sha256": sha256_file(args.baseline),
            "method_sha256": sha256_file(args.method),
            "questions_sha256": sha256_file(args.questions),
        },
        "claim_ceiling": (
            "This is a binary grouped-claim feasibility screen. The frozen tie-break "
            "is not a continuous scoring method, and oracle headroom is not performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    compact = dict(payload)
    compact["result"] = {key: value for key, value in payload["result"].items() if key != "image_results"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
