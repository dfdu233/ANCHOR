#!/usr/bin/env python3
"""ConfGen and domain-calibrated SGTA-ConfGen on disjoint OE splits."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from conformal_generation import ConformalGeneration

from corrected_sgta.cache import iter_successes
from corrected_sgta.oe_metrics_v2 import lexical_admissible, lexical_metrics
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split

METHOD_VERSION = "domain-center-sgta-confgen-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument(
        "--style-learning-fraction",
        type=float,
        default=0.5,
        help="fraction of outer calibration reserved for style learning; the rest is proper conformal calibration",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, nargs="*", default=(0.8, 0.9, 0.95))
    parser.add_argument("--rouge-threshold", type=float, default=0.30)
    parser.add_argument(
        "--style-beta", type=float, nargs="*", default=(0.0, 0.25, 0.5, 1.0, 2.0)
    )
    return parser.parse_args()


def uncertainty(output: dict) -> float:
    value = output.get("uncertainty")
    return 1e6 if value is None or not math.isfinite(float(value)) else float(value)


def sequence_confidence(output: dict) -> float:
    """Geometric mean processed-token probability for RunningMax stopping."""

    return math.exp(-min(50.0, uncertainty(output)))


def adjusted_confidence(
    output: dict, style_reliability: dict[str, float] | None, style_beta: float
) -> float:
    """Reference-free generation confidence with a learned domain-style prior."""

    base = sequence_confidence(output)
    if not style_reliability or style_beta == 0:
        return base
    prior = style_reliability.get(
        str(output.get("style", "original")), style_reliability.get("__global__", 0.0)
    )
    return base * math.exp(max(-20.0, min(20.0, style_beta * prior)))


def attach_metrics(
    outputs: list[dict], reference: str, threshold: float
) -> tuple[list[dict], list[bool]]:
    copied, admissibilities = [], []
    for output in outputs:
        value = dict(output)
        value["metrics"] = lexical_metrics(value.get("text", ""), reference)
        copied.append(value)
        admissibilities.append(
            lexical_admissible(value.get("text", ""), reference, threshold)
        )
    return copied, admissibilities


def average_metrics(outputs: list[dict]) -> dict[str, float | None]:
    return {
        key: mean(output["metrics"][key] for output in outputs) if outputs else None
        for key in ("bleu4", "rouge_1", "rouge_2", "rouge_l", "token_f1")
    }


def baseline_report(records: list[dict], threshold: float) -> dict:
    outputs = []
    admissible = []
    for row in records:
        candidate, values = attach_metrics([row["greedy"]], row["answer"], threshold)
        outputs.extend(candidate)
        admissible.extend(values)
    return {
        "n": len(outputs),
        "metrics": average_metrics(outputs),
        "lexical_admissibility_rate": mean(admissible) if admissible else None,
    }


def learn_style_reliability(
    field: str,
    by_qid: dict[str, dict],
    qids: list[str],
    rouge_threshold: float,
) -> dict[str, float]:
    """Beta-smoothed log-odds that each style yields an admissible candidate."""

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total_success = 0
    total_count = 0
    for qid in qids:
        row = by_qid[qid]
        for output in row[field]:
            style = str(output.get("style", "original"))
            success = int(
                lexical_admissible(output.get("text", ""), row["answer"], rouge_threshold)
            )
            counts[style][0] += success
            counts[style][1] += 1
            total_success += success
            total_count += 1
    global_rate = (total_success + 1.0) / (total_count + 2.0)
    result = {"__global__": math.log(global_rate / (1.0 - global_rate))}
    for style, (successes, count) in counts.items():
        # Shrink each sparse style estimate toward the global rate.
        rate = (successes + 2.0 * global_rate) / (count + 2.0)
        rate = min(max(rate, 1e-6), 1.0 - 1e-6)
        result[style] = math.log(rate / (1.0 - rate))
    return result


def choose_style_beta(
    field: str,
    by_qid: dict[str, dict],
    validation_qids: list[str],
    rouge_threshold: float,
    style_reliability: dict[str, float],
    candidates: list[float],
) -> tuple[float, list[dict]]:
    """Choose prior strength by top-one admissibility on an inner holdout."""

    search = []
    for beta in candidates:
        successes = []
        nll = []
        for qid in validation_qids:
            row = by_qid[qid]
            chosen = max(
                row[field],
                key=lambda output: adjusted_confidence(output, style_reliability, beta),
            )
            successes.append(
                lexical_admissible(
                    chosen.get("text", ""), row["answer"], rouge_threshold
                )
            )
            nll.append(uncertainty(chosen))
        search.append(
            {
                "beta": float(beta),
                "n": len(successes),
                "top1_admissibility": mean(successes) if successes else None,
                "mean_uncertainty": mean(nll) if nll else None,
            }
        )
    valid = [row for row in search if row["top1_admissibility"] is not None]
    if not valid:
        return 0.0, search
    best = sorted(
        valid,
        key=lambda row: (
            -row["top1_admissibility"],
            row["mean_uncertainty"],
            abs(row["beta"]),
        ),
    )[0]
    return float(best["beta"]), search


def analyze_stream(
    name: str,
    field: str,
    by_qid: dict[str, dict],
    calibration_qids: list[str],
    test_qids: list[str],
    gammas: list[float],
    rouge_threshold: float,
    style_reliability: dict[str, float] | None = None,
    style_beta: float = 0.0,
) -> dict:
    if not calibration_qids:
        raise ValueError("proper calibration split is empty")
    score_fn = lambda _instance, output: adjusted_confidence(
        output, style_reliability, style_beta
    )
    cal_inputs, cal_outputs, cal_admissibility = [], [], []
    for qid in calibration_qids:
        row = by_qid[qid]
        outputs, admissibilities = attach_metrics(
            row[field], row["answer"], rouge_threshold
        )
        cal_inputs.append({"qid": qid})
        cal_outputs.append(outputs)
        cal_admissibility.append(admissibilities)
    generator = ConformalGeneration.from_score_function(
        input_dataset=cal_inputs,
        raw_generated_dataset=cal_outputs,
        score_fn=score_fn,
        score_method="running_max",
        admissibility_dataset=cal_admissibility,
        admissibility_aggregation=max,
        admissibility_function_lower_bound=0.0,
        use_cache=True,
    )
    result = {
        "name": name,
        "selector": "conf-gen RunningMaxSequenceSelector",
        "score": (
            "whole-sequence confidence times a calibration-learned domain-style prior"
            if style_reliability and style_beta != 0
            else "exp(-mean processed-sampling token NLL), i.e. whole-sequence confidence"
        ),
        "style_beta": float(style_beta),
        "style_reliability": style_reliability,
        "n_calibration": len(calibration_qids),
        "n_test": len(test_qids),
        "gamma": {},
    }
    raw_oracle = []
    for qid in test_qids:
        row = by_qid[qid]
        _, admissibilities = attach_metrics(row[field], row["answer"], rouge_threshold)
        raw_oracle.append(any(admissibilities))
    result["raw_stream_oracle_coverage_evaluation_only"] = (
        mean(raw_oracle) if raw_oracle else None
    )

    for gamma in gammas:
        generator.calibrate(gamma=gamma, recalibrate=True)
        selected_all: list[dict] = []
        chosen_all: list[dict] = []
        covered, sizes, unique_sizes, empty = [], [], [], []
        for qid in test_qids:
            row = by_qid[qid]
            candidates, _ = attach_metrics(row[field], row["answer"], rouge_threshold)
            selected = generator.select({"qid": qid}, candidates)
            selected_all.extend(selected)
            sizes.append(len(selected))
            unique_sizes.append(
                len({candidate.get("text", "").strip() for candidate in selected})
            )
            empty.append(not selected)
            covered.append(
                any(
                    lexical_admissible(
                        candidate.get("text", ""), row["answer"], rouge_threshold
                    )
                    for candidate in selected
                )
            )
            # Reference-free point reduction. An empty conformal set remains an
            # explicit failure instead of being silently filled by an oracle.
            if selected:
                chosen_all.append(max(selected, key=lambda output: score_fn({}, output)))
        threshold = generator.conformal_threshold
        result["gamma"][str(gamma)] = {
            "lambda": None if not math.isfinite(threshold) else float(threshold),
            "lambda_is_infinite": not math.isfinite(threshold),
            "empirical_coverage": mean(covered) if covered else None,
            "coverage_gap": (mean(covered) - gamma) if covered else None,
            "average_set_size": mean(sizes) if sizes else None,
            "average_unique_set_size": mean(unique_sizes) if unique_sizes else None,
            "empty_set_rate": mean(empty) if empty else None,
            "selected_set_candidate_metrics": average_metrics(selected_all),
            "confidence_reduced_output_metrics": average_metrics(chosen_all),
            "confidence_reduced_admissibility_rate": mean(
                lexical_admissible(
                    chosen.get("text", ""), by_qid[qid]["answer"], rouge_threshold
                )
                for qid, chosen in zip(
                    [qid for qid, is_empty in zip(test_qids, empty) if not is_empty],
                    chosen_all,
                )
            )
            if chosen_all
            else None,
        }
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.style_learning_fraction < 1.0:
        raise ValueError("style-learning-fraction must be in (0, 1)")
    metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    fingerprint = metadata["fingerprint"]
    records = list(iter_successes(args.cache, fingerprint))
    qids = [str(row["qid"]) for row in records]
    outer_calibration_qids, test_qids = deterministic_split(
        qids, args.calibration_fraction, args.seed
    )
    style_learning_qids, proper_calibration_qids = deterministic_split(
        outer_calibration_qids, args.style_learning_fraction, args.seed + 17
    )
    reliability_qids, beta_validation_qids = deterministic_split(
        style_learning_qids, 0.5, args.seed + 31
    )
    if not proper_calibration_qids:
        raise RuntimeError("not enough records for a non-empty proper calibration split")
    by_qid = {str(row["qid"]): row for row in records}
    methods = {
        "confgen_original": analyze_stream(
            "confgen_original",
            "sampled",
            by_qid,
            proper_calibration_qids,
            test_qids,
            list(args.gamma),
            args.rouge_threshold,
        )
    }
    optimization = None
    if records and all(row.get("style_sampled") for row in records):
        methods["sgta_confgen_style_augmented"] = analyze_stream(
            "sgta_confgen_style_augmented",
            "style_sampled",
            by_qid,
            proper_calibration_qids,
            test_qids,
            list(args.gamma),
            args.rouge_threshold,
        )
        inner_reliability = learn_style_reliability(
            "style_sampled", by_qid, reliability_qids, args.rouge_threshold
        )
        selected_beta, beta_search = choose_style_beta(
            "style_sampled",
            by_qid,
            beta_validation_qids,
            args.rouge_threshold,
            inner_reliability,
            list(args.style_beta),
        )
        fitted_reliability = learn_style_reliability(
            "style_sampled", by_qid, style_learning_qids, args.rouge_threshold
        )
        methods["sgta_confgen_domain_calibrated"] = analyze_stream(
            "sgta_confgen_domain_calibrated",
            "style_sampled",
            by_qid,
            proper_calibration_qids,
            test_qids,
            list(args.gamma),
            args.rouge_threshold,
            fitted_reliability,
            selected_beta,
        )
        optimization = {
            "selected_style_beta": selected_beta,
            "beta_search_inner_validation": beta_search,
            "fitted_style_reliability_log_odds": fitted_reliability,
        }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION,
        "source_cache": str(args.cache),
        "fingerprint": fingerprint,
        "n": len(records),
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "style_learning_fraction_of_outer_calibration": args.style_learning_fraction,
            "n_style_reliability_train": len(reliability_qids),
            "n_style_beta_validation": len(beta_validation_qids),
            "n_proper_calibration": len(proper_calibration_qids),
            "n_test": len(test_qids),
            "style_reliability_qids": reliability_qids,
            "style_beta_validation_qids": beta_validation_qids,
            "proper_calibration_qids": proper_calibration_qids,
            "test_qids": test_qids,
        },
        "admissibility": {
            "definition": f"ROUGE-L >= {args.rouge_threshold:g}",
            "status": "reproducible lexical proxy; not clinical correctness or proprietary LLM judge",
        },
        "metric_availability": {
            "computed_offline": ["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "token-F1"],
            "not_computed_missing_dependencies": [
                "METEOR (WordNet)",
                "BERTScore",
                "RaTEScore",
                "proprietary MedHEval Bedrock hallucination judge",
            ],
        },
        "greedy_baseline_test": baseline_report(
            [by_qid[qid] for qid in test_qids], args.rouge_threshold
        ),
        "optimization": optimization,
        "methods": methods,
        "method_scope": {
            "confgen_original": "M identically sampled original-image candidates; greedy excluded",
            "sgta_confgen_style_augmented": (
                "same M-candidate budget, round-robin original/FedDG/gamma styles"
            ),
            "sgta_confgen_domain_calibrated": (
                "same style candidate stream with domain reliability learned on style-learning rows; "
                "proper conformal calibration and test remain disjoint"
            ),
            "candidate_reduction": "maximum adjusted whole-sequence confidence; reference-free at test time",
            "oracle_fields": "evaluation-only upper bounds and never used for test selection",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: value["gamma"] for name, value in methods.items()}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
