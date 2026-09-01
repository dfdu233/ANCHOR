#!/usr/bin/env python3
"""Paired audit of whether a mitigation creates correction or redistributes answers.

The primary comparison treats invalid generations as errors and bootstraps image
clusters.  A common-parse analysis prevents a method from looking better merely
because it changes answerability.  This is an admission screen, not evidence that
the evaluated CE task represents open-ended clinical hallucination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "no-free-grounding-paired-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary(value: Any) -> str:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    value = str(value).strip().lower()
    if value in {"yes", "no"}:
        return value
    return "invalid"


def metrics(rows: list[dict[str, Any]], indices: Iterable[int] | None = None) -> dict[str, Any]:
    chosen = list(range(len(rows))) if indices is None else list(indices)
    counts = {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        "invalid_positive": 0, "invalid_negative": 0,
    }
    for index in chosen:
        row = rows[index]
        truth, prediction = row["truth"], row["prediction"]
        if prediction == "invalid":
            counts[f"invalid_{'positive' if truth == 'yes' else 'negative'}"] += 1
        elif truth == "yes":
            counts["tp" if prediction == "yes" else "fn"] += 1
        else:
            counts["tn" if prediction == "no" else "fp"] += 1
    n = len(chosen)
    n_positive = counts["tp"] + counts["fn"] + counts["invalid_positive"]
    n_negative = counts["tn"] + counts["fp"] + counts["invalid_negative"]
    definite_positive = counts["tp"] + counts["fp"]
    parseable = n - counts["invalid_positive"] - counts["invalid_negative"]
    sensitivity = counts["tp"] / n_positive if n_positive else float("nan")
    specificity = counts["tn"] / n_negative if n_negative else float("nan")
    return {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        **counts,
        "accuracy_invalid_as_error": (counts["tp"] + counts["tn"]) / n if n else float("nan"),
        "balanced_accuracy_invalid_as_error": (sensitivity + specificity) / 2.0,
        "positive_recall": sensitivity,
        "negative_recall": specificity,
        "positive_claim_precision": counts["tp"] / definite_positive if definite_positive else float("nan"),
        "hallucination_risk_among_positive_claims": counts["fp"] / definite_positive if definite_positive else float("nan"),
        "parse_rate": parseable / n if n else float("nan"),
        "positive_answer_rate": definite_positive / n if n else float("nan"),
        "negative_answer_rate": (counts["tn"] + counts["fn"]) / n if n else float("nan"),
        "invalid_rate": (n - parseable) / n if n else float("nan"),
    }


def exact_mcnemar_pvalue(baseline_only_correct: int, method_only_correct: int) -> float:
    """Two-sided exact McNemar/binomial p-value for paired correctness."""
    discordant = baseline_only_correct + method_only_correct
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only_correct, method_only_correct) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _finite_delta(method_value: float, baseline_value: float) -> float | None:
    delta = method_value - baseline_value
    return float(delta) if np.isfinite(delta) else None


def cluster_bootstrap(
    baseline_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    clusters: list[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    if not (len(baseline_rows) == len(method_rows) == len(clusters)) or not clusters:
        raise ValueError("rows and non-empty cluster labels must be aligned")
    by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        by_cluster.setdefault(str(cluster), []).append(index)
    cluster_ids = sorted(by_cluster)
    names = (
        "accuracy_invalid_as_error",
        "balanced_accuracy_invalid_as_error",
        "positive_recall",
        "hallucination_risk_among_positive_claims",
        "parse_rate",
        "positive_answer_rate",
    )
    observed_base = metrics(baseline_rows)
    observed_method = metrics(method_rows)
    values: dict[str, list[float]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled = [index for cluster in sampled_clusters for index in by_cluster[str(cluster)]]
        base = metrics(baseline_rows, sampled)
        method = metrics(method_rows, sampled)
        for name in names:
            delta = _finite_delta(method[name], base[name])
            if delta is not None:
                values[name].append(delta)
    output = {}
    for name in names:
        observed = _finite_delta(observed_method[name], observed_base[name])
        valid = values[name]
        output[name] = {
            "estimate": observed,
            "ci_low": float(np.quantile(valid, 0.025)) if valid else None,
            "ci_high": float(np.quantile(valid, 0.975)) if valid else None,
            "valid_draws": len(valid),
        }
    return output


def align_inputs(
    baseline_report: dict[str, Any],
    method_report: dict[str, Any],
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    question_by_id = {int(row.get("qid", index)): row for index, row in enumerate(questions)}
    baseline = {int(row["question_id"]): row for row in baseline_report["details"]}
    method = {int(row["question_id"]): row for row in method_report["details"]}
    if set(baseline) != set(method):
        raise ValueError("baseline and method question IDs differ")
    if not set(baseline).issubset(question_by_id):
        raise ValueError("evaluation contains IDs absent from the question file")
    baseline_rows, method_rows, clusters = [], [], []
    for question_id in sorted(baseline):
        question = question_by_id[question_id]
        # Use the evaluator's frozen binary normalization (e.g. Abnormal -> yes)
        # rather than silently maintaining a second label vocabulary here.
        truth = _binary(baseline[question_id].get("ground_truth"))
        method_truth = _binary(method[question_id].get("ground_truth"))
        if truth == "invalid" or method_truth != truth:
            raise ValueError(f"question {question_id} has invalid or inconsistent binary truth")
        common = {
            "question_id": question_id,
            "truth": truth,
            "image": str(question.get("img_name") or question.get("img_id")),
        }
        baseline_rows.append({**common, "prediction": _binary(baseline[question_id].get("prediction"))})
        method_rows.append({**common, "prediction": _binary(method[question_id].get("prediction"))})
        clusters.append(common["image"])
    return baseline_rows, method_rows, clusters


def analyze(
    baseline_report: dict[str, Any],
    method_report: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    baseline_rows, method_rows, clusters = align_inputs(baseline_report, method_report, questions)
    baseline_metrics = metrics(baseline_rows)
    method_metrics = metrics(method_rows)
    common_parse = [
        index for index, (baseline, method) in enumerate(zip(baseline_rows, method_rows))
        if baseline["prediction"] != "invalid" and method["prediction"] != "invalid"
    ]
    base_common = metrics(baseline_rows, common_parse)
    method_common = metrics(method_rows, common_parse)
    transitions = {
        "both_correct": 0,
        "baseline_only_correct": 0,
        "method_only_correct": 0,
        "both_wrong": 0,
    }
    by_truth = {"yes": {key: 0 for key in transitions}, "no": {key: 0 for key in transitions}}
    for baseline, method in zip(baseline_rows, method_rows):
        baseline_correct = baseline["prediction"] == baseline["truth"]
        method_correct = method["prediction"] == method["truth"]
        key = (
            "both_correct" if baseline_correct and method_correct else
            "baseline_only_correct" if baseline_correct else
            "method_only_correct" if method_correct else "both_wrong"
        )
        transitions[key] += 1
        by_truth[baseline["truth"]][key] += 1
    bootstrap = cluster_bootstrap(
        baseline_rows, method_rows, clusters, draws=draws, seed=seed
    )
    gate = (
        bootstrap["balanced_accuracy_invalid_as_error"]["ci_low"] is not None
        and bootstrap["balanced_accuracy_invalid_as_error"]["ci_low"] > 0
        and bootstrap["hallucination_risk_among_positive_claims"]["ci_high"] is not None
        and bootstrap["hallucination_risk_among_positive_claims"]["ci_high"] < 0
        and bootstrap["positive_recall"]["ci_low"] is not None
        and bootstrap["positive_recall"]["ci_low"] >= -0.01
        and bootstrap["parse_rate"]["ci_low"] is not None
        and bootstrap["parse_rate"]["ci_low"] >= -0.01
    )
    return {
        "n": len(baseline_rows),
        "n_image_clusters": len(set(clusters)),
        "class_balance": {
            "positive": baseline_metrics["n_positive"],
            "negative": baseline_metrics["n_negative"],
            "always_negative_accuracy": baseline_metrics["n_negative"] / len(baseline_rows),
        },
        "baseline": baseline_metrics,
        "method": method_metrics,
        "method_minus_baseline_cluster_bootstrap": bootstrap,
        "common_parse_matched_subset": {
            "n": len(common_parse),
            "baseline": base_common,
            "method": method_common,
        },
        "paired_correctness_transitions": transitions,
        "paired_correctness_transitions_by_truth": by_truth,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(
            transitions["baseline_only_correct"], transitions["method_only_correct"]
        ),
        "admission_gate": {
            "passed": bool(gate),
            "rule": (
                "cluster-bootstrap 95% CI requires balanced-accuracy gain and "
                "positive-claim hallucination-risk reduction, with positive-recall "
                "and parse-rate losses bounded to at most 1 percentage point"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-name", default="greedy")
    parser.add_argument("--method-name", default="VCD")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=317)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    method_report = json.loads(args.method.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    result = analyze(
        baseline_report, method_report, questions,
        draws=args.bootstrap_draws, seed=args.seed,
    )
    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison": {"baseline": args.baseline_name, "method": args.method_name},
        "result": result,
        "provenance": {
            "code_sha256": sha256_file(Path(__file__)),
            "baseline_sha256": sha256_file(args.baseline),
            "method_sha256": sha256_file(args.method),
            "questions_sha256": sha256_file(args.questions),
        },
        "claim_ceiling": (
            "This paired CE screen can reject a mitigation-as-correction account. "
            "It cannot establish a universal impossibility theorem, reader uncertainty, "
            "or open-ended/report hallucination performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
