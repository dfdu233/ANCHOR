#!/usr/bin/env python3
"""SGTA-ConfGen-v2 with clinical admissibility and leakage-safe calibration."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from conformal_generation import ConformalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer

from corrected_sgta.cache import iter_successes
from corrected_sgta.clinical_judgments import (
    candidate_key,
    clinical_admissible,
    load_judgments,
    validate_judgment,
)
from corrected_sgta.oe_metrics_v2 import lexical_admissible, lexical_metrics
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    deterministic_split,
)

METHOD_VERSION = "matched-center-sgta-confgen-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("knowledge", "report"))
    parser.add_argument("--admissibility", choices=("clinical", "lexical"), default="clinical")
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--judge-agreement", type=Path)
    parser.add_argument("--router-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--min-proper-calibration", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, nargs="*", default=(0.9,))
    parser.add_argument("--rouge-threshold", type=float, default=0.30)
    parser.add_argument("--candidate-budget", type=int, default=8)
    parser.add_argument("--allow-legacy-diagnostic", action="store_true")
    return parser.parse_args()


def uncertainty(output: dict) -> float:
    value = output.get("uncertainty")
    return 50.0 if value is None or not math.isfinite(float(value)) else min(50.0, float(value))


def center_distance(output: dict) -> float:
    value = output.get("center_distance")
    if isinstance(value, dict):
        value = value.get("log_amplitude_cosine_distance", 0.0)
    return 0.0 if value is None else max(0.0, float(value))


def semantic_agreement(outputs: list[dict]) -> list[float]:
    texts = [str(output.get("text", "")).strip() for output in outputs]
    if len(texts) <= 1 or not any(texts):
        return [1.0] * len(texts)
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform(texts)
        similarity = (matrix @ matrix.T).toarray()
        return [
            float((similarity[index].sum() - similarity[index, index]) / (len(texts) - 1))
            for index in range(len(texts))
        ]
    except ValueError:
        return [0.0] * len(texts)


def ordered_outputs(
    row: dict, field: str, budget: int, fingerprint: str = ""
) -> list[dict]:
    fingerprint = row.get("_cache_fingerprint", fingerprint)
    values = [dict(output) for output in (row.get(field) or [])[:budget]]
    for index, output in enumerate(values):
        output["_item_id"] = candidate_key(row["qid"], field, index, output, fingerprint)
        output["_source_index"] = index
    # Recompute acquisition order for legacy caches. It is reference-free and
    # prevents fixed style position from being a hidden RunningMax feature.
    values.sort(key=lambda output: (uncertainty(output), output["_item_id"]))
    agreements = semantic_agreement(values)
    for step, (output, agreement) in enumerate(zip(values, agreements)):
        output["acquisition_step"] = step
        output["acquisition_policy"] = "ascending_sequence_nll_then_candidate_id"
        output["_semantic_agreement"] = agreement
        output["_clinical_self_consistency"] = float(
            output.get("clinical_self_consistency", 0.0) or 0.0
        )
    return values


def three_way_split(qids: list[str], router_fraction: float, calibration_fraction: float, seed: int):
    development_fraction = router_fraction + calibration_fraction
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("router + calibration fractions must be in (0, 1)")
    development, test = deterministic_split(qids, development_fraction, seed)
    router, calibration = deterministic_split(
        development, router_fraction / development_fraction, seed + 1
    )
    return router, calibration, test


class Admissibility:
    def __init__(
        self, mode: str, task: str, judgments: dict[str, dict],
        rouge_threshold: float, fingerprint: str, evidence_validation: dict | None
    ):
        self.mode = mode
        self.task = task
        self.judgments = judgments
        self.rouge_threshold = rouge_threshold
        self.fingerprint = fingerprint
        self.evidence_validation = evidence_validation or {}

    def value(self, output: dict, reference: str) -> bool:
        if self.mode == "lexical":
            return lexical_admissible(output.get("text", ""), reference, self.rouge_threshold)
        item_id = output["_item_id"]
        if item_id not in self.judgments:
            raise KeyError(f"missing clinical judgment for candidate {item_id}")
        judgment = self.judgments[item_id]
        if judgment.get("cache_fingerprint") != self.fingerprint:
            raise ValueError(f"judgment {item_id} is bound to a different cache fingerprint")
        if (
            self.task == "report"
            and judgment.get("metric_manifest_sha256")
            != self.evidence_validation.get("manifest_sha256")
        ):
            raise ValueError(
                f"report judgment {item_id} is bound to a different metric manifest"
            )
        validate_judgment(judgment, self.task)
        return clinical_admissible(judgment, self.task)

    def fact_recall(self, output: dict) -> float | None:
        judgment = self.judgments.get(output.get("_item_id", ""), {})
        value = judgment.get("clinical_fact_recall")
        return None if value is None else float(value)


WEIGHT_GRID = [
    {"agreement": agreement, "domain": domain, "style": style, "clinical": clinical}
    for agreement in (0.0, 0.5, 1.0)
    for domain in (0.0, 0.25, 0.5)
    for style in (0.0, 0.5)
    for clinical in (0.0, 0.5)
]


def learn_style_reliability(field, by_qid, qids, admissibility, budget):
    counts = defaultdict(lambda: [0, 0])
    total_success = total_count = 0
    for qid in qids:
        row = by_qid[qid]
        for output in ordered_outputs(row, field, budget):
            style = str(output.get("style", "original"))
            success = int(admissibility.value(output, row["answer"]))
            counts[style][0] += success
            counts[style][1] += 1
            total_success += success
            total_count += 1
    global_rate = (total_success + 1.0) / (total_count + 2.0)
    result = {"__global__": math.log(global_rate / (1.0 - global_rate))}
    for style, (successes, count) in counts.items():
        rate = (successes + 4.0 * global_rate) / (count + 4.0)
        rate = min(max(rate, 1e-6), 1.0 - 1e-6)
        result[style] = math.log(rate / (1.0 - rate))
    return result


def candidate_score(output: dict, weights: dict, reliability: dict | None) -> float:
    prior = 0.0
    if reliability:
        prior = reliability.get(str(output.get("style", "original")), reliability["__global__"])
    return float(
        -uncertainty(output)
        + weights["agreement"] * output.get("_semantic_agreement", 0.0)
        + weights["clinical"] * output.get("_clinical_self_consistency", 0.0)
        + weights["style"] * prior
        - weights["domain"] * center_distance(output)
    )


def choose_weights(field, by_qid, qids, admissibility, reliability, budget):
    search = []
    for weights in WEIGHT_GRID:
        chosen = []
        for qid in qids:
            row = by_qid[qid]
            outputs = ordered_outputs(row, field, budget)
            if outputs:
                chosen.append(max(outputs, key=lambda output: candidate_score(output, weights, reliability)))
        successes = [
            admissibility.value(output, by_qid[qid]["answer"])
            for qid, output in zip([qid for qid in qids if ordered_outputs(by_qid[qid], field, budget)], chosen)
        ]
        search.append(
            {
                "weights": weights,
                "n": len(chosen),
                "top1_admissibility": mean(successes) if successes else None,
                "mean_uncertainty": mean(uncertainty(output) for output in chosen) if chosen else None,
            }
        )
    valid = [row for row in search if row["top1_admissibility"] is not None]
    if not valid:
        return WEIGHT_GRID[0], search
    best = sorted(
        valid,
        key=lambda row: (
            -row["top1_admissibility"],
            row["mean_uncertainty"],
            sum(abs(value) for value in row["weights"].values()),
        ),
    )[0]
    return best["weights"], search


def average_lexical(outputs: list[dict], references: list[str]) -> dict:
    if not outputs:
        return {key: None for key in ("bleu4", "rouge_1", "rouge_2", "rouge_l", "token_f1")}
    metrics = [lexical_metrics(output.get("text", ""), reference) for output, reference in zip(outputs, references)]
    return {key: mean(row[key] for row in metrics) for key in metrics[0]}


def analyze_stream(name, field, by_qid, train_qids, validation_qids, calibration_qids, test_qids, args, admissibility):
    reliability = learn_style_reliability(field, by_qid, train_qids, admissibility, args.candidate_budget)
    weights, weight_search = choose_weights(
        field, by_qid, validation_qids, admissibility, reliability, args.candidate_budget
    )
    score_fn = lambda _instance, output: candidate_score(output, weights, reliability)
    cal_outputs, cal_admissibilities = [], []
    for qid in calibration_qids:
        row = by_qid[qid]
        outputs = ordered_outputs(row, field, args.candidate_budget)
        cal_outputs.append(outputs)
        cal_admissibilities.append([admissibility.value(output, row["answer"]) for output in outputs])
    generator = ConformalGeneration.from_score_function(
        input_dataset=[{"qid": qid} for qid in calibration_qids],
        raw_generated_dataset=cal_outputs,
        score_fn=score_fn,
        score_method="running_max",
        admissibility_dataset=cal_admissibilities,
        admissibility_aggregation=max,
        admissibility_function_lower_bound=0.0,
        use_cache=True,
    )
    result = {
        "name": name,
        "field": field,
        "n_router_fit": len(train_qids),
        "n_router_validation": len(validation_qids),
        "n_proper_calibration": len(calibration_qids),
        "n_locked_test": len(test_qids),
        "selected_weights": weights,
        "style_reliability_log_odds": reliability,
        "weight_search": weight_search,
        "gamma": {},
    }
    for gamma in args.gamma:
        generator.calibrate(gamma=gamma, recalibrate=True)
        threshold = generator.conformal_threshold
        selected_sets, reduced, references = [], [], []
        covered, sizes, unique_sizes, audit = [], [], [], []
        for qid in test_qids:
            row = by_qid[qid]
            outputs = ordered_outputs(row, field, args.candidate_budget)
            selected = generator.select({"qid": qid}, outputs)
            selected_sets.extend(selected)
            references.extend([row["answer"]] * len(selected))
            coverage = any(admissibility.value(output, row["answer"]) for output in selected)
            covered.append(coverage)
            sizes.append(len(selected))
            unique_sizes.append(len({output.get("text", "").strip() for output in selected}))
            chosen = max(selected, key=lambda output: score_fn({}, output)) if selected else None
            if chosen is not None:
                reduced.append((qid, chosen))
            audit.append(
                {
                    "qid": qid,
                    "set_candidate_ids": [output["_item_id"] for output in selected],
                    "set_size": len(selected),
                    "covered_evaluation_only": coverage,
                    "selected_candidate_id": None if chosen is None else chosen["_item_id"],
                    "selected_style": None if chosen is None else chosen.get("style", "unknown"),
                    "fallback_to_original": chosen is not None and chosen.get("style", "original") == "original",
                    "selection_reason": "max calibrated reference-free risk score",
                }
            )
        reduced_success = [
            admissibility.value(output, by_qid[qid]["answer"]) for qid, output in reduced
        ]
        recalls = [admissibility.fact_recall(output) for _, output in reduced]
        recalls = [value for value in recalls if value is not None]
        empirical = mean(covered) if covered else None
        vacuous = (not math.isfinite(threshold)) or (
            sizes and mean(sizes) >= args.candidate_budget - 1e-12
        )
        result["gamma"][str(gamma)] = {
            "lambda": None if not math.isfinite(threshold) else float(threshold),
            "lambda_is_infinite": not math.isfinite(threshold),
            "vacuous_guarantee": bool(vacuous),
            "empirical_coverage": empirical,
            "coverage_gap": None if empirical is None else empirical - gamma,
            "average_set_size": mean(sizes) if sizes else None,
            "average_unique_set_size": mean(unique_sizes) if unique_sizes else None,
            "confidence_reduced_admissibility_rate": mean(reduced_success) if reduced_success else None,
            "confidence_reduced_hallucination_error": 1.0 - mean(reduced_success) if reduced_success else None,
            "confidence_reduced_clinical_fact_recall": mean(recalls) if recalls else None,
            "selected_set_lexical_metrics": average_lexical(selected_sets, references),
            "audit": audit,
        }
    return result


def main() -> None:
    args = parse_args()
    if args.admissibility == "clinical" and args.judgments is None:
        raise ValueError("--judgments is required for clinical admissibility")
    agreement = None
    if args.judge_agreement:
        agreement = json.loads(args.judge_agreement.read_text())
        expected_type = (
            "knowledge_judge_agreement" if args.task == "knowledge"
            else "report_metric_validation"
        )
        if agreement.get("validation_type") != expected_type:
            raise RuntimeError(
                f"expected evidence gate type {expected_type}, got "
                f"{agreement.get('validation_type')!r}"
            )
        if not agreement.get("passed"):
            raise RuntimeError("clinical evidence validation gate did not pass")
    elif args.admissibility == "clinical":
        raise ValueError("--judge-agreement/evidence validation JSON is required in clinical mode")
    judgments = load_judgments(args.judgments) if args.judgments else {}
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    if (
        agreement
        and args.task == "knowledge"
        and agreement.get("cache_fingerprint") != metadata.get("fingerprint")
    ):
        raise RuntimeError("knowledge agreement gate is bound to a different cache fingerprint")
    legacy = metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION
    if legacy and not args.allow_legacy_diagnostic:
        raise RuntimeError(
            "SGTA-ConfGen-v2 requires a v5.4 evidence cache; legacy caches are "
            "diagnostic-only and require --allow-legacy-diagnostic"
        )
    if not legacy and metadata.get("config", {}).get("center_policy") != "matched":
        raise RuntimeError("formal SGTA-ConfGen-v2 requires center_policy=matched")
    records = list(iter_successes(args.cache, metadata["fingerprint"]))
    if not legacy:
        for row in records:
            if row.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                raise RuntimeError(f"qid {row.get('qid')} has an incompatible row schema")
    qids = [str(row["qid"]) for row in records]
    router_qids, calibration_qids, test_qids = three_way_split(
        qids, args.router_fraction, args.calibration_fraction, args.seed
    )
    router_train, router_validation = deterministic_split(router_qids, 0.5, args.seed + 19)
    for row in records:
        row["_cache_fingerprint"] = metadata["fingerprint"]
    by_qid = {str(row["qid"]): row for row in records}
    if args.admissibility == "clinical":
        for row in records:
            if len(row.get("sampled") or []) != args.candidate_budget:
                raise RuntimeError(
                    f"qid {row['qid']} vanilla stream does not have exactly "
                    f"{args.candidate_budget} candidates"
                )
            if row.get("style_sampled") is not None and len(row["style_sampled"]) != args.candidate_budget:
                raise RuntimeError(
                    f"qid {row['qid']} SGTA stream does not have exactly "
                    f"{args.candidate_budget} candidates"
                )
    admissibility = Admissibility(
        args.admissibility,
        args.task,
        judgments,
        args.rouge_threshold,
        metadata["fingerprint"],
        agreement,
    )
    methods = {
        "vanilla_confgen": analyze_stream(
            "vanilla_confgen", "sampled", by_qid, router_train, router_validation,
            calibration_qids, test_qids, args, admissibility
        )
    }
    if records and all(row.get("style_sampled") for row in records):
        methods["sgta_confgen_v2"] = analyze_stream(
            "sgta_confgen_v2", "style_sampled", by_qid, router_train, router_validation,
            calibration_qids, test_qids, args, admissibility
        )
    comparison = {}
    if "sgta_confgen_v2" in methods:
        for gamma in args.gamma:
            key = str(gamma)
            vanilla = methods["vanilla_confgen"]["gamma"][key]
            sgta = methods["sgta_confgen_v2"]["gamma"][key]
            base_error = vanilla["confidence_reduced_hallucination_error"]
            new_error = sgta["confidence_reduced_hallucination_error"]
            relative_error_reduction = (
                (base_error - new_error) / base_error if base_error and new_error is not None else None
            )
            base_recall = vanilla["confidence_reduced_clinical_fact_recall"]
            new_recall = sgta["confidence_reduced_clinical_fact_recall"]
            recall_delta = (
                new_recall - base_recall if new_recall is not None and base_recall is not None else None
            )
            gates = {
                "finite_lambda": not sgta["lambda_is_infinite"],
                "coverage_within_3pp": sgta["coverage_gap"] is not None and sgta["coverage_gap"] >= -0.03,
                "mean_set_size_at_most_half_budget": (
                    sgta["average_set_size"] is not None
                    and sgta["average_set_size"] <= args.candidate_budget / 2
                ),
                "hallucination_error_relative_reduction_10pct": (
                    relative_error_reduction is not None and relative_error_reduction >= 0.10
                ),
                "report_fact_recall_drop_at_most_2pp": (
                    args.task != "report" or (recall_delta is not None and recall_delta >= -0.02)
                ),
                "proper_calibration_at_least_required": len(calibration_qids) >= args.min_proper_calibration,
                "nonvacuous": not sgta["vacuous_guarantee"],
            }
            comparison[key] = {
                "relative_hallucination_error_reduction": relative_error_reduction,
                "clinical_fact_recall_delta": recall_delta,
                "gate": {**gates, "passed": all(gates.values())},
            }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION,
        "source_cache": str(args.cache),
        "fingerprint": metadata["fingerprint"],
        "cache_schema_version": metadata.get("cache_schema_version"),
        "evidence_status": "legacy_diagnostic_only" if legacy else "formal_v5.4",
        "task": args.task,
        "admissibility": {
            "mode": args.admissibility,
            "knowledge_rule": "MedHEval hallucination score <= 2" if args.task == "knowledge" else None,
            "report_rule": (
                "clinical entity precision >= 0.80, clinical fact recall >= 0.50, "
                "and no critical contradiction" if args.task == "report" else None
            ),
            "judge_agreement": agreement,
            "paper_status": (
                "eligible"
                if (
                    not legacy
                    and args.admissibility == "clinical"
                    and agreement
                    and agreement.get("passed")
                )
                else "exploratory_only"
            ),
        },
        "candidate_budget": args.candidate_budget,
        "split": {
            "seed": args.seed,
            "n_router_train": len(router_train),
            "n_router_validation": len(router_validation),
            "n_proper_calibration": len(calibration_qids),
            "n_locked_test": len(test_qids),
            "router_train_qids": router_train,
            "router_validation_qids": router_validation,
            "proper_calibration_qids": calibration_qids,
            "locked_test_qids": test_qids,
        },
        "methods": methods,
        "comparison": comparison,
        "scope": (
            "Weights and style priors use router rows only; ConfGen lambda uses proper calibration only; "
            "locked-test judgments are evaluation-only. Candidate order is reference-independent."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
