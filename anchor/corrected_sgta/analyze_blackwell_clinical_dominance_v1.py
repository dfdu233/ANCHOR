#!/usr/bin/env python3
"""Audit criterion-independent information in completed CE generations.

This is a CPU-only, post-hoc audit.  It treats the strict semantic parser output
(`yes`, `no`, or `invalid`) as a *finite signal* about a binary ground-truth
state and computes its likelihood-ratio ROC envelope.  No confidence score is
reconstructed from text length or token ids.

The resulting finite-signal ROC is only a lower-resolution audit of the final
answer channel.  In particular, it cannot identify whether a mitigation method
created additional continuous evidence inside the model because the completed
artifacts contain no logits, log-probabilities, or calibrated confidences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable


VERSION = "blackwell-clinical-dominance-v1"
DEFAULT_ROOT = Path(
    "corrected_runs/paper_baselines_v1/full_matrix_v1/derived_scores/llava"
)
DEFAULT_METHODS = ("AvisC", "DoLa", "OPERA", "PAI", "VCD", "VISTA")
SIGNALS = ("yes", "no", "invalid")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of empty values")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ci(values: list[float]) -> list[float]:
    return [quantile(values, 0.025), quantile(values, 0.975)]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=EVALUATION_JSON")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("method and path must be nonempty")
    return name.strip(), Path(raw_path)


def strict_signal(row: dict[str, Any]) -> str:
    prediction = row.get("prediction")
    if prediction == ["yes"]:
        return "yes"
    if prediction == ["no"]:
        return "no"
    return "invalid"


def truth(row: dict[str, Any]) -> int:
    value = row.get("ground_truth")
    if value == ["yes"]:
        return 1
    if value == ["no"]:
        return 0
    raise ValueError(f"not a binary row: {value!r}")


def finite_channel(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    counts = {0: {signal: 0 for signal in SIGNALS}, 1: {signal: 0 for signal in SIGNALS}}
    totals = {0: 0, 1: 0}
    for row in rows:
        y = truth(row)
        counts[y][strict_signal(row)] += 1
        totals[y] += 1
    if not totals[0] or not totals[1]:
        raise ValueError("both binary states must be present")
    return {
        "negative": {signal: counts[0][signal] / totals[0] for signal in SIGNALS},
        "positive": {signal: counts[1][signal] / totals[1] for signal in SIGNALS},
        "counts_negative": counts[0],
        "counts_positive": counts[1],
        "n_negative": totals[0],
        "n_positive": totals[1],
    }


def roc_from_channel(channel: dict[str, dict[str, float]]) -> list[list[float]]:
    """Return the Neyman-Pearson ROC polygon for a finite signal channel."""
    ranked = []
    for signal in SIGNALS:
        fpr_mass = channel["negative"][signal]
        tpr_mass = channel["positive"][signal]
        if fpr_mass == 0.0:
            ratio = math.inf if tpr_mass > 0.0 else 0.0
        else:
            ratio = tpr_mass / fpr_mass
        ranked.append((ratio, signal, fpr_mass, tpr_mass))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    points = [[0.0, 0.0]]
    fpr = tpr = 0.0
    for _, _, fpr_mass, tpr_mass in ranked:
        fpr += fpr_mass
        tpr += tpr_mass
        points.append([min(1.0, fpr), min(1.0, tpr)])
    points[-1] = [1.0, 1.0]
    return points


def roc_value(points: list[list[float]], x: float) -> float:
    if x <= 0.0:
        return points[0][1]
    if x >= 1.0:
        return points[-1][1]
    for left, right in zip(points, points[1:]):
        if left[0] <= x <= right[0]:
            width = right[0] - left[0]
            if width == 0.0:
                return max(left[1], right[1])
            fraction = (x - left[0]) / width
            return left[1] + fraction * (right[1] - left[1])
    raise AssertionError("ROC interpolation failed")


def auc(points: list[list[float]]) -> float:
    total = 0.0
    for left, right in zip(points, points[1:]):
        total += (right[0] - left[0]) * (right[1] + left[1]) / 2.0
    return total


def optimal_risk(points: list[list[float]], false_negative_weight: float) -> float:
    """Minimum conditional risk over randomized decisions on the ROC polygon.

    Risk is w * FNR + (1-w) * FPR, so w=0.5 corresponds to balanced error.
    A linear objective reaches its optimum at a ROC vertex.
    """
    return min(
        false_negative_weight * (1.0 - tpr)
        + (1.0 - false_negative_weight) * fpr
        for fpr, tpr in points
    )


def empirical_deficiency(
    candidate: list[list[float]], reference: list[list[float]], grid: list[float]
) -> float:
    """Maximum vertical ROC shortfall; descriptive, not Le Cam deficiency."""
    return max(0.0, max(roc_value(reference, x) - roc_value(candidate, x) for x in grid))


def spearman(values_x: dict[str, float], values_y: dict[str, float]) -> float:
    names = sorted(set(values_x) & set(values_y))

    def ranks(values: dict[str, float]) -> dict[str, float]:
        ordered = sorted((values[name], name) for name in names)
        result: dict[str, float] = {}
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][0] == ordered[index][0]:
                end += 1
            rank = (index + end - 1) / 2.0
            for _, name in ordered[index:end]:
                result[name] = rank
            index = end
        return result

    rx, ry = ranks(values_x), ranks(values_y)
    mean_x = sum(rx.values()) / len(names)
    mean_y = sum(ry.values()) / len(names)
    numerator = sum((rx[n] - mean_x) * (ry[n] - mean_y) for n in names)
    denom_x = sum((rx[n] - mean_x) ** 2 for n in names)
    denom_y = sum((ry[n] - mean_y) ** 2 for n in names)
    if denom_x == 0.0 or denom_y == 0.0:
        return float("nan")
    return numerator / math.sqrt(denom_x * denom_y)


def categorical_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    channel = finite_channel(rows)
    roc = roc_from_channel(channel)
    cost_grid = [index / 20.0 for index in range(1, 20)]
    roc_grid = [index / 20.0 for index in range(21)]
    signals = [strict_signal(row) for row in rows]
    observed_fpr = channel["negative"]["yes"]
    observed_tpr = channel["positive"]["yes"]
    return {
        "channel": channel,
        "likelihood_ratio_ordered_roc": roc,
        "auc": auc(roc),
        "roc_grid": {f"{x:.2f}": roc_value(roc, x) for x in roc_grid},
        "optimal_conditional_risk": {
            f"{weight:.2f}": optimal_risk(roc, weight) for weight in cost_grid
        },
        "observed_output_rates": {
            signal: sum(value == signal for value in signals) / len(signals)
            for signal in SIGNALS
        },
        "observed_yes_as_positive_operating_point": {
            "fpr": observed_fpr,
            "tpr": observed_tpr,
            "balanced_error": ((1.0 - observed_tpr) + observed_fpr) / 2.0,
            "envelope_tpr_at_same_fpr": roc_value(roc, observed_fpr),
        },
    }


def remap_accounting(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["details"]
    strict = [float(bool(row["correct"])) for row in rows]
    official = [float(bool(row["official_benchmark_correct"])) for row in rows]
    invalid = [strict_signal(row) == "invalid" for row in rows]
    recovered_invalid = sum(
        is_invalid and bool(row["official_benchmark_correct"])
        for is_invalid, row in zip(invalid, rows)
    )
    harmed_strict_correct = sum(
        bool(row["correct"]) and not bool(row["official_benchmark_correct"])
        for row in rows
    )
    return {
        "n_all_questions": len(rows),
        "strict_accuracy": float(report["decoded_strict"]["accuracy_invalid_as_error"]),
        "official_accuracy": float(report["official_benchmark_proxy"]["accuracy"]),
        "official_minus_strict": float(report["official_benchmark_proxy"]["accuracy"])
        - float(report["decoded_strict"]["accuracy_invalid_as_error"]),
        "official_detail_accuracy_all_rows_diagnostic": sum(official) / len(rows),
        "strict_invalid_rate": sum(invalid) / len(rows),
        "invalid_rows_recovered_by_official": recovered_invalid,
        "strict_correct_rows_harmed_by_official": harmed_strict_correct,
        "note": (
            "This is exact accounting on the same outputs, not out-of-sample prediction."
        ),
    }


def ranking(values: dict[str, float], higher_is_better: bool = True) -> list[str]:
    return sorted(values, key=lambda name: ((-1.0 if higher_is_better else 1.0) * values[name], name))


def rank_flips(left: dict[str, float], right: dict[str, float]) -> list[list[str]]:
    names = sorted(left)
    flips = []
    for index, a in enumerate(names):
        for b in names[index + 1 :]:
            if (left[a] - left[b]) * (right[a] - right[b]) < 0.0:
                flips.append([a, b])
    return flips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_named_path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", default="cxr_vishal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = args.input or [
        (method, args.root / method / args.dataset / "evaluation_ce_v7.json")
        for method in DEFAULT_METHODS
    ]
    if len({name for name, _ in inputs}) != len(inputs):
        raise ValueError("duplicate method names")

    reports: dict[str, dict[str, Any]] = {}
    rows_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: dict[str, dict[str, str]] = {}
    score_audit: dict[str, Any] = {}
    for method, path in inputs:
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[method] = report
        indexed = {str(row["question_id"]): row for row in report["details"]}
        if len(indexed) != len(report["details"]):
            raise ValueError(f"{method}: duplicate question ids")
        rows_by_method[method] = indexed
        answers_path = Path(report["answers"])
        answer_meta_fields: set[str] = set()
        answer_fields: set[str] = set()
        with answers_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                answer = json.loads(line)
                answer_fields.update(answer)
                answer_meta_fields.update(answer.get("metadata", {}))
        continuous_candidates = sorted(
            field
            for field in answer_fields | answer_meta_fields
            if any(token in field.lower() for token in ("logit", "prob", "confidence", "score"))
        )
        score_audit[method] = {
            "continuous_clinical_score_available": bool(continuous_candidates),
            "continuous_candidate_fields": continuous_candidates,
            "available_semantic_signal": "strict parser category: yes/no/invalid",
            "rejected_as_clinical_scores": [
                "generated_token_ids",
                "token counts",
                "answer length",
                "stop reason",
            ],
        }
        provenance[method] = {
            "evaluation_path": str(path.resolve()),
            "evaluation_sha256": sha256_file(path),
            "answers_path": str(answers_path.resolve()),
            "answers_sha256": sha256_file(answers_path),
        }

    methods = sorted(reports)
    reference_qids = set(rows_by_method[methods[0]])
    for method in methods[1:]:
        if set(rows_by_method[method]) != reference_qids:
            raise ValueError(f"{method}: qid set differs")

    binary_qids = sorted(
        qid
        for qid in reference_qids
        if rows_by_method[methods[0]][qid].get("answer_type") == "binary"
        and rows_by_method[methods[0]][qid].get("ground_truth") in (["yes"], ["no"])
    )
    for qid in binary_qids:
        reference = rows_by_method[methods[0]][qid]
        for method in methods[1:]:
            row = rows_by_method[method][qid]
            if row["cluster_id"] != reference["cluster_id"] or row["ground_truth"] != reference["ground_truth"]:
                raise ValueError(f"unaligned row at {method}/{qid}")

    point_metrics = {
        method: categorical_metrics([rows_by_method[method][qid] for qid in binary_qids])
        for method in methods
    }
    accounting = {method: remap_accounting(reports[method]) for method in methods}

    roc_grid = [index / 100.0 for index in range(101)]
    pairwise: dict[str, Any] = {}
    for index, left in enumerate(methods):
        for right in methods[index + 1 :]:
            left_roc = point_metrics[left]["likelihood_ratio_ordered_roc"]
            right_roc = point_metrics[right]["likelihood_ratio_ordered_roc"]
            left_shortfall = empirical_deficiency(left_roc, right_roc, roc_grid)
            right_shortfall = empirical_deficiency(right_roc, left_roc, roc_grid)
            tolerance = 1e-12
            if left_shortfall <= tolerance and right_shortfall > tolerance:
                relation = f"{left}_coarsened_roc_dominates_{right}"
            elif right_shortfall <= tolerance and left_shortfall > tolerance:
                relation = f"{right}_coarsened_roc_dominates_{left}"
            elif left_shortfall <= tolerance and right_shortfall <= tolerance:
                relation = "coarsened_roc_equivalent"
            else:
                relation = "roc_curves_cross"
            pairwise[f"{left}__vs__{right}"] = {
                "relation": relation,
                f"{left}_vertical_shortfall_vs_{right}": left_shortfall,
                f"{right}_vertical_shortfall_vs_{left}": right_shortfall,
                "auc_difference_left_minus_right": point_metrics[left]["auc"] - point_metrics[right]["auc"],
                "matched_fpr_envelope_comparison": {
                    "at_left_observed_fpr": {
                        "fpr": point_metrics[left]["observed_yes_as_positive_operating_point"]["fpr"],
                        f"{left}_tpr": roc_value(
                            left_roc,
                            point_metrics[left]["observed_yes_as_positive_operating_point"]["fpr"],
                        ),
                        f"{right}_tpr": roc_value(
                            right_roc,
                            point_metrics[left]["observed_yes_as_positive_operating_point"]["fpr"],
                        ),
                    },
                    "at_right_observed_fpr": {
                        "fpr": point_metrics[right]["observed_yes_as_positive_operating_point"]["fpr"],
                        f"{left}_tpr": roc_value(
                            left_roc,
                            point_metrics[right]["observed_yes_as_positive_operating_point"]["fpr"],
                        ),
                        f"{right}_tpr": roc_value(
                            right_roc,
                            point_metrics[right]["observed_yes_as_positive_operating_point"]["fpr"],
                        ),
                    },
                },
            }

    # Paired image-cluster bootstrap.  The same sampled clusters are used for
    # every method in a draw, preserving method pairing.
    by_cluster: dict[str, list[str]] = {}
    for qid in binary_qids:
        cluster = str(rows_by_method[methods[0]][qid]["cluster_id"])
        by_cluster.setdefault(cluster, []).append(qid)
    clusters = sorted(by_cluster)
    rng = random.Random(args.seed)
    bootstrap_auc = {method: [] for method in methods}
    bootstrap_risk = {method: [] for method in methods}
    bootstrap_pair_auc = {key: [] for key in pairwise}
    bootstrap_pair_deficiency = {key: {"left_vs_right": [], "right_vs_left": []} for key in pairwise}
    for _ in range(args.bootstrap_draws):
        sampled_clusters = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sampled_qids = [qid for cluster in sampled_clusters for qid in by_cluster[cluster]]
        replicate_rocs: dict[str, list[list[float]]] = {}
        for method in methods:
            channel = finite_channel(rows_by_method[method][qid] for qid in sampled_qids)
            replicate_rocs[method] = roc_from_channel(channel)
            bootstrap_auc[method].append(auc(replicate_rocs[method]))
            bootstrap_risk[method].append(optimal_risk(replicate_rocs[method], 0.5))
        for index, left in enumerate(methods):
            for right in methods[index + 1 :]:
                key = f"{left}__vs__{right}"
                bootstrap_pair_auc[key].append(auc(replicate_rocs[left]) - auc(replicate_rocs[right]))
                bootstrap_pair_deficiency[key]["left_vs_right"].append(
                    empirical_deficiency(replicate_rocs[left], replicate_rocs[right], roc_grid)
                )
                bootstrap_pair_deficiency[key]["right_vs_left"].append(
                    empirical_deficiency(replicate_rocs[right], replicate_rocs[left], roc_grid)
                )

    for method in methods:
        point_metrics[method]["paired_image_cluster_bootstrap"] = {
            "auc_ci95": ci(bootstrap_auc[method]),
            "balanced_optimal_risk_ci95": ci(bootstrap_risk[method]),
        }
    for key in pairwise:
        pairwise[key]["auc_difference_ci95"] = ci(bootstrap_pair_auc[key])
        pairwise[key]["vertical_shortfall_ci95"] = {
            "left_vs_right": ci(bootstrap_pair_deficiency[key]["left_vs_right"]),
            "right_vs_left": ci(bootstrap_pair_deficiency[key]["right_vs_left"]),
        }

    strict_values = {method: accounting[method]["strict_accuracy"] for method in methods}
    official_values = {method: accounting[method]["official_accuracy"] for method in methods}
    auc_values = {method: point_metrics[method]["auc"] for method in methods}
    bonus_values = {method: accounting[method]["official_minus_strict"] for method in methods}
    invalid_values = {method: accounting[method]["strict_invalid_rate"] for method in methods}
    flips = rank_flips(strict_values, official_values)
    reversal_accounting = []
    for left, right in flips:
        reversal_accounting.append(
            {
                "methods": [left, right],
                "strict_gap_left_minus_right": strict_values[left] - strict_values[right],
                "official_bonus_gap_left_minus_right": bonus_values[left] - bonus_values[right],
                "official_gap_left_minus_right": official_values[left] - official_values[right],
                "bonus_overcomes_strict_gap": (
                    (strict_values[left] - strict_values[right])
                    * (official_values[left] - official_values[right])
                    < 0.0
                ),
            }
        )

    any_continuous = any(item["continuous_clinical_score_available"] for item in score_audit.values())
    result = {
        "version": VERSION,
        "status": "complete",
        "dataset": args.dataset,
        "model": "LLaVA-Med-v1.5-Mistral-7B",
        "methods": methods,
        "seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "command": " ".join(sys.argv),
        "input_provenance": provenance,
        "artifact_fingerprint": hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "cohort": {
            "binary_questions": len(binary_qids),
            "image_clusters": len(clusters),
            "all_questions_for_rank_accounting": len(reference_qids),
        },
        "continuous_score_source_audit": score_audit,
        "continuous_roc_identifiability": {
            "identified": any_continuous,
            "decision": "GO" if any_continuous else "NO_GO",
            "reason": (
                "At least one artifact exposes a continuous clinical score."
                if any_continuous
                else "No method artifact stores logits, probabilities, confidence, or another predeclared continuous clinical score. Token ids and text length are not valid confidence scores."
            ),
        },
        "finite_semantic_channel_audit": point_metrics,
        "paired_coarsened_roc_comparisons": pairwise,
        "strict_official_rank_accounting": {
            "per_method": accounting,
            "strict_ranking": ranking(strict_values),
            "official_ranking": ranking(official_values),
            "finite_channel_auc_ranking": ranking(auc_values),
            "strict_to_official_pair_reversals": flips,
            "n_reversals": len(flips),
            "reversal_accounting": reversal_accounting,
            "spearman_invalid_rate_vs_official_bonus": spearman(invalid_values, bonus_values),
        },
        "scope_boundary": {
            "identified": (
                "The empirical three-category final-answer channel P(parsed output | binary truth), its likelihood-ratio ROC envelope, and decision risks under post-hoc randomized remapping."
            ),
            "not_identified": (
                "Continuous/internal clinical evidence, token-level confidence, and Blackwell dominance of the underlying model representations. The finite output channel is a coarse lower-resolution observation and mixes heterogeneous questions."
            ),
            "theory_positioning": (
                "Blackwell comparison, Neyman-Pearson ROC ordering, and Le Cam deficiency are classical tools and are not claimed as methodological novelty. Reported vertical ROC shortfall is only an empirical approximation, not formal Le Cam deficiency."
            ),
        },
        "decision": {
            "method_information_gain_claim": "NO_GO",
            "reason": (
                "The saved generations permit only a coarsened final-output channel audit. Without continuous scores, an apparent accuracy or ranking gain cannot establish that a method added discriminative clinical information rather than moved/remapped its operating point."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
