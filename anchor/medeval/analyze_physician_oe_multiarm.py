"""Pre-registered, fail-closed analysis for blinded multi-arm physician OE review.

The private method mapping is joined only after a model-blinded consensus has
been frozen.  Every contrast is paired by image group; answer units are never
treated as independent observations.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import sha256_file
from .store import atomic_write_json
from .validate_physician_oe_review import load_jsonl, validate_completed


VERSION = "anchor-physician-oe-multiarm-analysis-v1"
CONSENSUS_VERSION = "anchor-physician-oe-consensus-v1"
ERRORS = {
    "fabricated",
    "false_negation",
    "location",
    "attribute",
    "inappropriate_certainty",
}
REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable to)\s+(?:determine|answer|interpret)|"
    r"\b(?:consult|ask)\s+(?:a|your)\s+(?:doctor|physician|radiologist)\b",
    re.IGNORECASE,
)
DIRECT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
HARM_SCORE = {"no": 0.0, "possibly": 0.5, "yes": 1.0}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    candidate: Mapping[str, float | None],
    baseline: Mapping[str, float | None],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    keys = sorted(
        key
        for key in set(candidate) & set(baseline)
        if candidate[key] is not None and baseline[key] is not None
    )
    if not keys:
        return {"n_groups": 0, "delta": None, "ci95": None, "p_lower_better": None}
    deltas = [float(candidate[key]) - float(baseline[key]) for key in keys]
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    return {
        "n_groups": len(keys),
        "delta": _mean(deltas),
        "ci95": {"low": _quantile(draws, 0.025), "high": _quantile(draws, 0.975)},
        "p_lower_better": (1 + sum(value >= 0 for value in draws)) / (iterations + 1),
    }


def holm_adjust(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    present = sorted((value, key) for key, value in pvalues.items() if value is not None)
    adjusted: dict[str, float | None] = {key: None for key in pvalues}
    running = 0.0
    total = len(present)
    for rank, (value, key) in enumerate(present):
        running = max(running, min(1.0, float(value) * (total - rank)))
        adjusted[key] = running
    return adjusted


def _answer_metrics(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    annotation = candidate["annotation"]
    claims = [
        claim
        for claim in annotation["atomic_claims"]
        if claim["claim_type"] == "visual" and claim["relevance"] != "out_of_scope"
    ]
    evaluated = [claim for claim in claims if claim["error_type"] != "indeterminate"]
    correct = [
        claim
        for claim in evaluated
        if claim["visual_support"] == "supported" and claim["error_type"] == "none"
    ]
    errors = [claim for claim in evaluated if claim["error_type"] in ERRORS]
    required = row["reference_annotation"]["required_answer_claims"]
    omissions = annotation["omitted_required_claim_ids"]
    correctness = DIRECT_SCORE.get(annotation["direct_answer_correctness"])
    harm = HARM_SCORE.get(annotation["overall_clinically_harmful"])
    text = str(candidate["answer_text"])
    return {
        "any_visual_error": float(bool(errors)),
        "visual_error_count": float(len(errors)),
        "evaluated_visual_claim_count": float(len(evaluated)),
        "correct_visual_claim_count": float(len(correct)),
        "fabricated_or_refuted_count": float(
            sum(
                claim["visual_support"] == "refuted"
                or claim["error_type"] in {"fabricated", "false_negation"}
                for claim in evaluated
            )
        ),
        "inappropriate_certainty_count": float(
            sum(claim["error_type"] == "inappropriate_certainty" for claim in evaluated)
        ),
        "required_claim_count": float(len(required)),
        "omitted_required_count": float(len(omissions)),
        "required_recall": (
            (len(required) - len(omissions)) / len(required) if required else None
        ),
        "direct_correctness": correctness,
        "harm_score": harm,
        "word_count": float(len(text.split())),
        "refusal": float(bool(REFUSAL_RE.search(text))),
        "no_clinical_claims": float(bool(annotation["no_clinical_claims"])),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = sorted(rows[0]["metrics"]) if rows else []
    means = {
        metric: _mean(
            [float(row["metrics"][metric]) for row in rows if row["metrics"][metric] is not None]
        )
        for metric in metrics
    }
    correct = sum(row["metrics"]["correct_visual_claim_count"] for row in rows)
    evaluated = sum(row["metrics"]["evaluated_visual_claim_count"] for row in rows)
    required = sum(row["metrics"]["required_claim_count"] for row in rows)
    omitted = sum(row["metrics"]["omitted_required_count"] for row in rows)
    return {
        "groups": len(rows),
        "means": means,
        "visual_claim_precision_micro": correct / evaluated if evaluated else None,
        "required_claim_recall_micro": (required - omitted) / required if required else None,
        "totals": {
            "evaluated_visual_claims": evaluated,
            "correct_visual_claims": correct,
            "required_claims": required,
            "omitted_required_claims": omitted,
        },
    }


def _validate_consensus_provenance(
    provenance: Mapping[str, Any], consensus_path: Path, bundle_id: str
) -> None:
    if provenance.get("protocol_version") != CONSENSUS_VERSION:
        raise ValueError("wrong consensus provenance version")
    if provenance.get("bundle_id") != bundle_id:
        raise ValueError("consensus provenance bundle mismatch")
    if provenance.get("consensus_sha256") != sha256_file(consensus_path):
        raise ValueError("consensus hash mismatch")
    if provenance.get("model_identity_visible_during_adjudication") is not False:
        raise ValueError("adjudication was not model-blinded")
    if provenance.get("private_mapping_joined_before_consensus") is not False:
        raise ValueError("private mapping was joined before consensus freeze")
    reviewers = provenance.get("reviewers")
    if not isinstance(reviewers, list) or len(set(reviewers)) < 2:
        raise ValueError("two distinct reviewers are required")
    if provenance.get("unresolved_disagreements") != 0:
        raise ValueError("unresolved physician disagreements remain")
    if not str(provenance.get("adjudicator_id", "")).strip():
        raise ValueError("adjudicator identity is missing")
    for field in (
        "reviewer_a_completed_sha256",
        "reviewer_b_completed_sha256",
        "reviewer_a_validation_sha256",
        "reviewer_b_validation_sha256",
        "clarification_log_sha256",
    ):
        value = provenance.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"missing hash-bound consensus provenance: {field}")


def analyze_multiarm(
    consensus: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "greedy",
    seed: int = 20260802,
    iterations: int = 10000,
) -> dict[str, Any]:
    if iterations < 1000:
        raise ValueError("formal physician analysis requires at least 1000 bootstraps")
    answer_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    group_lookup = {}
    for row in consensus:
        group_id = str(row["group_id"])
        if group_id in group_lookup:
            raise ValueError("duplicate consensus group")
        group_lookup[group_id] = row
        for answer in row["candidate_answers"]:
            key = (group_id, str(answer["answer_id"]))
            if key in answer_lookup:
                raise ValueError("duplicate consensus answer unit")
            answer_lookup[key] = answer

    assignments: dict[tuple[str, str], str] = {}
    methods_by_group: dict[str, set[str]] = defaultdict(set)
    for item in mapping:
        group_id = str(item["group_id"])
        method = str(item["source_model"])
        answer_id = str(item["answer_id"])
        key = (group_id, method)
        if key in assignments:
            raise ValueError(f"duplicate method assignment: {key}")
        answer = answer_lookup.get((group_id, answer_id))
        if answer is None:
            raise ValueError(f"mapping references absent answer: {(group_id, answer_id)}")
        import hashlib

        digest = hashlib.sha256(str(answer["answer_text"]).encode()).hexdigest()
        if digest != item.get("answer_text_sha256"):
            raise ValueError("private mapping answer hash mismatch")
        assignments[key] = answer_id
        methods_by_group[group_id].add(method)
    if set(methods_by_group) != set(group_lookup):
        raise ValueError("private mapping group set differs from consensus")
    method_sets = {tuple(sorted(value)) for value in methods_by_group.values()}
    if len(method_sets) != 1:
        raise ValueError("method set differs across image groups")
    methods = list(next(iter(method_sets)))
    if baseline not in methods:
        raise ValueError("baseline method absent")

    per_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for group_id, row in group_lookup.items():
        for method in methods:
            answer = answer_lookup[(group_id, assignments[(group_id, method)])]
            per_method[method].append(
                {
                    "group_id": group_id,
                    "answer_id": answer["answer_id"],
                    "metrics": _answer_metrics(row, answer),
                }
            )

    aggregates = {method: _aggregate(rows) for method, rows in per_method.items()}
    by_method_group = {
        method: {row["group_id"]: row["metrics"] for row in rows}
        for method, rows in per_method.items()
    }
    contrasts = {}
    raw_primary_p = {}
    for method_index, method in enumerate(method for method in methods if method != baseline):
        candidate = by_method_group[method]
        control = by_method_group[baseline]
        metric_results = {}
        for metric_index, metric in enumerate(
            (
                "any_visual_error",
                "visual_error_count",
                "required_recall",
                "direct_correctness",
                "harm_score",
                "word_count",
                "evaluated_visual_claim_count",
                "refusal",
            )
        ):
            metric_results[metric] = paired_bootstrap(
                {key: value[metric] for key, value in candidate.items()},
                {key: value[metric] for key, value in control.items()},
                seed=seed + method_index * 101 + metric_index,
                iterations=iterations,
            )
        matched = [
            key
            for key in candidate
            if candidate[key]["omitted_required_count"]
            == control[key]["omitted_required_count"]
            and candidate[key]["evaluated_visual_claim_count"]
            == control[key]["evaluated_visual_claim_count"]
            and candidate[key]["refusal"] == control[key]["refusal"]
        ]
        matched_primary = paired_bootstrap(
            {key: candidate[key]["any_visual_error"] for key in matched},
            {key: control[key]["any_visual_error"] for key in matched},
            seed=seed + method_index * 101 + 97,
            iterations=iterations,
        )
        length_base = aggregates[baseline]["means"]["word_count"]
        claims_base = aggregates[baseline]["means"]["evaluated_visual_claim_count"]
        length_ratio = (
            aggregates[method]["means"]["word_count"] / length_base if length_base else None
        )
        claim_ratio = (
            aggregates[method]["means"]["evaluated_visual_claim_count"] / claims_base
            if claims_base
            else None
        )
        raw_primary_p[method] = metric_results["any_visual_error"]["p_lower_better"]
        contrasts[method] = {
            "versus": baseline,
            "paired_metrics": metric_results,
            "matched_coverage": {
                "definition": "same omitted-required count, evaluated visual-claim count, and refusal state",
                "minimum_groups": 12,
                "result": matched_primary,
            },
            "length_ratio": length_ratio,
            "evaluated_visual_claim_ratio": claim_ratio,
        }
    adjusted = holm_adjust(raw_primary_p)
    for method, contrast in contrasts.items():
        primary = contrast["paired_metrics"]["any_visual_error"]
        matched = contrast["matched_coverage"]["result"]
        recall = contrast["paired_metrics"]["required_recall"]
        direct = contrast["paired_metrics"]["direct_correctness"]
        harm = contrast["paired_metrics"]["harm_score"]
        refusal = contrast["paired_metrics"]["refusal"]
        gates = {
            "primary_error_reduction": bool(
                primary["ci95"] is not None and primary["ci95"]["high"] < 0
            ),
            "holm_adjusted_primary_p_below_0p05": bool(
                adjusted[method] is not None and adjusted[method] < 0.05
            ),
            "matched_coverage_error_reduction": bool(
                matched["n_groups"] >= 12
                and matched["ci95"] is not None
                and matched["ci95"]["high"] < 0
            ),
            "required_recall_noninferior_0p05": bool(
                recall["ci95"] is not None and recall["ci95"]["low"] >= -0.05
            ),
            "direct_correctness_noninferior_0p05": bool(
                direct["ci95"] is not None and direct["ci95"]["low"] >= -0.05
            ),
            "harm_not_increased_0p05": bool(
                harm["ci95"] is not None and harm["ci95"]["high"] <= 0.05
            ),
            "refusal_not_increased_0p01": bool(
                refusal["ci95"] is not None and refusal["ci95"]["high"] <= 0.01
            ),
            "length_at_least_90pct": bool(
                contrast["length_ratio"] is not None and contrast["length_ratio"] >= 0.90
            ),
            "visual_claims_at_least_90pct": bool(
                contrast["evaluated_visual_claim_ratio"] is not None
                and contrast["evaluated_visual_claim_ratio"] >= 0.90
            ),
        }
        contrast["primary_p_lower_better_holm"] = adjusted[method]
        contrast["promotion_gates"] = gates
        contrast["t3_promotion_authorized"] = all(gates.values())

    return {
        "protocol_version": VERSION,
        "baseline": baseline,
        "bootstrap_unit": "image group",
        "bootstrap_iterations": iterations,
        "seed": seed,
        "multiplicity": "Holm adjustment across non-baseline methods on paired any-visual-error reduction",
        "methods": methods,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "promoted_methods": sorted(
            method for method, row in contrasts.items() if row["t3_promotion_authorized"]
        ),
        "claim_boundary": (
            "T2 clinical screen only; promotion requires error reduction at matched coverage "
            "without omission, direct-correctness, harm, refusal, length, or claim-count exchange"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--consensus-provenance", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--baseline", default="greedy")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = load_jsonl(args.template)
    consensus = load_jsonl(args.consensus)
    validation = validate_completed(template, consensus)
    bundle_id = str(template[0]["bundle_id"])
    provenance = json.loads(args.consensus_provenance.read_text())
    _validate_consensus_provenance(provenance, args.consensus, bundle_id)
    result = analyze_multiarm(
        consensus,
        load_jsonl(args.mapping),
        baseline=args.baseline,
        seed=args.seed,
        iterations=args.bootstrap_iterations,
    )
    result["validation"] = validation
    result["provenance"] = {
        "template": str(args.template.resolve()),
        "template_sha256": sha256_file(args.template),
        "consensus": str(args.consensus.resolve()),
        "consensus_sha256": sha256_file(args.consensus),
        "consensus_provenance": str(args.consensus_provenance.resolve()),
        "consensus_provenance_sha256": sha256_file(args.consensus_provenance),
        "mapping": str(args.mapping.resolve()),
        "mapping_sha256": sha256_file(args.mapping),
    }
    if args.output.exists():
        raise FileExistsError("physician multi-arm analysis is write-once")
    atomic_write_json(args.output, result)
    print(json.dumps({"promoted_methods": result["promoted_methods"]}, indent=2))


if __name__ == "__main__":
    main()
