#!/usr/bin/env python3
"""Compare nuisance image swaps with claim-state-changing image swaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

from corrected_sgta.clinical_claims import (
    epistemic_coordinates,
    paired_clinical_selectivity,
)


VERSION = "clinical-selectivity-analysis-v7"
ROLES = {"anchor", "same_state_swap", "opposite_state_swap"}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(labels: list[int], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return None
    wins = sum(
        float(pos > neg) + 0.5 * float(pos == neg)
        for pos in positive
        for neg in negative
    )
    return wins / (len(positive) * len(negative))


def interval(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"estimate": None, "ci_low": None, "ci_high": None}
    ordered = sorted(values)
    low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return {"estimate": mean(values), "ci_low": low, "ci_high": high}


def bootstrap_mean(
    values: list[float], draws: int, seed: int
) -> dict[str, float | None]:
    if not values:
        return interval([])
    rng = random.Random(seed)
    samples = [
        mean(rng.choice(values) for _ in values)
        for _ in range(draws)
    ]
    result = interval(samples)
    result["estimate"] = mean(values)
    return result


def bootstrap_spearman(
    pairs: list[tuple[float, float]], draws: int, seed: int
) -> dict[str, float | int | None]:
    """Image-level bootstrap for ordinal reader support versus claim polarity."""

    def statistic(sample: list[tuple[float, float]]) -> float | None:
        relation = spearmanr(
            [value[0] for value in sample], [value[1] for value in sample]
        )
        estimate = float(relation.statistic)
        return estimate if math.isfinite(estimate) else None

    observed = statistic(pairs)
    if observed is None or len(pairs) < 2:
        return {
            "estimate": observed,
            "ci_low": None,
            "ci_high": None,
            "valid_draws": 0,
        }
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        value = statistic([rng.choice(pairs) for _ in pairs])
        if value is not None:
            samples.append(value)
    result = interval(samples)
    result["estimate"] = observed
    result["valid_draws"] = len(samples)
    return result


def select_dev_layer(per_layer: dict[str, object]) -> tuple[int, bool]:
    """Freeze a layer from dev without consulting locked-test outcomes."""

    candidates = []
    for layer_text, record_object in per_layer.items():
        record = dict(record_object)
        support = record["reader_support_spearman_bootstrap"]
        selectivity = record["clinical_selectivity_gap"]
        eligible = bool(
            support["estimate"] is not None
            and float(support["estimate"]) > 0.0
            and selectivity["estimate"] is not None
            and float(selectivity["estimate"]) > 0.0
        )
        candidates.append(
            (
                eligible,
                float(selectivity["estimate"] or -math.inf),
                float(support["estimate"] or -math.inf),
                -int(layer_text),
                int(layer_text),
            )
        )
    if not candidates:
        raise ValueError("cannot select a layer from an empty analysis")
    selected = max(candidates)
    return selected[-1], bool(selected[0])


def directional_admission_gates(
    layer_record: dict[str, object],
    formal_reference: bool,
    dev_selection_eligible: bool,
    min_test_triplets_per_bin: int,
) -> tuple[dict[str, bool], dict[str, object]]:
    """Apply the preregistered locked-test DCR admission rule."""

    qualified = []
    passed = []
    finding_records = layer_record["by_finding"]
    for finding, finding_record_object in finding_records.items():
        finding_record = dict(finding_record_object)
        counts = finding_record["anchor_vote_bin_counts"]
        is_qualified = all(
            int(counts.get(str(votes), 0)) >= min_test_triplets_per_bin
            for votes in range(4)
        )
        if not is_qualified:
            continue
        qualified.append(finding)
        support = finding_record["reader_support_spearman_bootstrap"]
        selectivity = finding_record["clinical_selectivity_gap_bootstrap"]
        if (
            support["ci_low"] is not None
            and float(support["ci_low"]) > 0.0
            and selectivity["ci_low"] is not None
            and float(selectivity["ci_low"]) > 0.0
        ):
            passed.append(finding)
    support = layer_record["reader_support_spearman_bootstrap"]
    selectivity = layer_record["clinical_selectivity_gap"]
    majority = bool(qualified) and len(passed) > len(qualified) / 2
    gates = {
        "formal_reader_reference": formal_reference,
        "dev_layer_selection_eligible": dev_selection_eligible,
        "global_reader_support_ordering_ci_above_zero": bool(
            support["ci_low"] is not None and float(support["ci_low"]) > 0.0
        ),
        "global_clinical_change_exceeds_same_support_drift_ci_above_zero": bool(
            selectivity["ci_low"] is not None
            and float(selectivity["ci_low"]) > 0.0
        ),
        "majority_of_qualified_findings_pass": majority,
    }
    gates["directional_admission_authorized"] = all(gates.values())
    summary = {
        "minimum_test_triplets_per_vote_bin": min_test_triplets_per_bin,
        "qualified_findings": qualified,
        "passed_findings": passed,
        "pass_fraction": len(passed) / len(qualified) if qualified else None,
    }
    return gates, summary


def bootstrap_auc_delta(
    labels: list[int],
    candidate_scores: list[float],
    baseline_scores: list[float],
    draws: int,
    seed: int,
) -> dict[str, float | None]:
    """Cluster-level bootstrap of candidate minus baseline error AUROC."""

    candidate = auc(labels, candidate_scores)
    baseline = auc(labels, baseline_scores)
    if candidate is None or baseline is None:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "candidate_auroc": candidate,
            "baseline_auroc": baseline,
        }
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    samples = []
    for _ in range(draws):
        chosen = [rng.choice(indices) for _ in indices]
        sampled_labels = [labels[index] for index in chosen]
        sampled_candidate = auc(
            sampled_labels, [candidate_scores[index] for index in chosen]
        )
        sampled_baseline = auc(
            sampled_labels, [baseline_scores[index] for index in chosen]
        )
        if sampled_candidate is not None and sampled_baseline is not None:
            samples.append(sampled_candidate - sampled_baseline)
    result = interval(samples)
    result.update(
        {
            "estimate": candidate - baseline,
            "candidate_auroc": candidate,
            "baseline_auroc": baseline,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-id",
        help="stable model identifier; required for a formal locked-test gate",
    )
    parser.add_argument(
        "--dev-analysis",
        type=Path,
        help=(
            "dev analysis that freezes the layer; required when "
            "--experiment-split=test"
        ),
    )
    parser.add_argument(
        "--min-test-triplets-per-bin",
        type=int,
        default=10,
        help="minimum anchor triplets in every 0/3,1/3,2/3,3/3 bin per finding",
    )
    parser.add_argument(
        "--experiment-split", choices=("all", "dev", "test"), default="all"
    )
    args = parser.parse_args()
    if args.min_test_triplets_per_bin <= 0:
        raise ValueError("min-test-triplets-per-bin must be positive")
    if args.experiment_split == "test" and (
        args.dev_analysis is None or not args.model_id
    ):
        raise ValueError(
            "formal locked-test admission requires --dev-analysis and --model-id"
        )
    if args.dev_analysis is not None and args.experiment_split != "test":
        raise ValueError("--dev-analysis is only valid with --experiment-split=test")

    manifest = {}
    for row in load_jsonl(args.manifest):
        key = (str(row["finding"]), str(row["image_id"]))
        if key in manifest:
            raise ValueError(f"duplicate finding/image contract: {key}")
        manifest[key] = row
    raw = [
        row
        for path in args.raw
        for row in load_jsonl(path)
        if row.get("status") == "ok"
    ]
    joined = []
    for row in raw:
        contract = manifest.get((str(row["finding"]), str(row["image_id"])))
        if contract is None:
            raise ValueError(f"raw image_id missing from manifest: {row['image_id']}")
        if (
            args.experiment_split == "all"
            or contract.get("experiment_split") == args.experiment_split
        ):
            joined.append((row, contract))
    if not joined:
        raise ValueError("no successful rows")

    layers = sorted(
        int(layer)
        for layer in joined[0][0]["measurement"]["trajectory"].keys()
    )
    triplets: dict[str, dict[str, tuple[dict, dict]]] = defaultdict(dict)
    for row, contract in joined:
        role = str(contract["swap_role"])
        if role not in ROLES:
            raise ValueError(f"unexpected swap role: {role}")
        triplets[str(contract["triplet_id"])][role] = (row, contract)
    complete = {
        key: members for key, members in triplets.items() if set(members) == ROLES
    }

    per_layer: dict[str, object] = {}
    for layer in layers:
        labels: list[int] = []
        polarities: list[float] = []
        supports: list[float] = []
        support_polarities: list[float] = []
        nuisance_changes: list[float] = []
        signed_clinical_changes: list[float] = []
        absolute_clinical_changes: list[float] = []
        magnitude_selectivity: list[float] = []
        pairwise_correct: list[float] = []
        unsigned_responsive: list[float] = []
        misdirected_responsive: list[float] = []
        same_state_flips: list[float] = []
        opposite_state_flips: list[float] = []
        anchor_error_labels: list[int] = []
        margin_error_scores: list[float] = []
        signed_response_error_scores: list[float] = []
        unsigned_selectivity_error_scores: list[float] = []
        csg_error_scores: list[float] = []
        by_finding: dict[str, list[tuple[float, float]]] = defaultdict(list)
        support_pairs_by_finding: dict[str, list[tuple[float, float]]] = defaultdict(list)
        anchor_vote_counts: dict[str, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        def state_polarity_margin(
            member: tuple[dict, dict]
        ) -> tuple[str, float, float]:
            row, _ = member
            trajectory = row["measurement"]["trajectory"][str(layer)]
            logits = trajectory["real_logits"]
            coordinates = epistemic_coordinates(logits)
            ordered = sorted((float(value) for value in logits.values()), reverse=True)
            return (
                str(trajectory["baseline_state"]),
                float(coordinates["polarity"]),
                ordered[0] - ordered[1],
            )

        for row, contract in joined:
            _, polarity, _ = state_polarity_margin((row, contract))
            positive_votes = int(contract["positive_votes"])
            reader_count = int(contract["reader_count"])
            if positive_votes in {0, reader_count}:
                labels.append(int(positive_votes == reader_count))
                polarities.append(polarity)
            supports.append(float(contract["reader_support"]))
            support_polarities.append(polarity)
            support_pairs_by_finding[str(contract["finding"])].append(
                (float(contract["reader_support"]), polarity)
            )

        for members in complete.values():
            anchor_state, anchor_p, anchor_margin = state_polarity_margin(
                members["anchor"]
            )
            same_state, same_p, _ = state_polarity_margin(members["same_state_swap"])
            opposite_state, opposite_p, _ = state_polarity_margin(
                members["opposite_state_swap"]
            )
            anchor_contract = members["anchor"][1]
            selectivity = paired_clinical_selectivity(
                anchor_p,
                same_p,
                opposite_p,
                float(anchor_contract["reader_support"]),
                float(members["opposite_state_swap"][1]["reader_support"]),
            )
            nuisance = selectivity["absolute_nuisance_change"]
            clinical = selectivity["signed_clinical_change"]
            absolute_clinical = selectivity["absolute_clinical_change"]
            nuisance_changes.append(nuisance)
            signed_clinical_changes.append(clinical)
            absolute_clinical_changes.append(absolute_clinical)
            magnitude_selectivity.append(selectivity["unsigned_selectivity_gap"])
            pairwise_correct.append(float(clinical > 0.0))
            unsigned_responsive.append(selectivity["unsigned_responsive"])
            misdirected_responsive.append(selectivity["misdirected_responsive"])
            same_state_flips.append(float(anchor_state != same_state))
            opposite_state_flips.append(float(anchor_state != opposite_state))
            anchor_support = float(anchor_contract["reader_support"])
            anchor_reference = (
                "supported" if anchor_support == 1.0 else
                "refuted" if anchor_support == 0.0 else
                "undetermined"
            )
            anchor_error_labels.append(int(anchor_state != anchor_reference))
            margin_error_scores.append(-anchor_margin)
            signed_response_error_scores.append(-clinical)
            unsigned_selectivity_error_scores.append(
                -(absolute_clinical - nuisance)
            )
            csg_error_scores.append(-(clinical - nuisance))
            by_finding[str(anchor_contract["finding"])].append((clinical, nuisance))
            anchor_vote_counts[str(anchor_contract["finding"])][
                int(anchor_contract["positive_votes"])
            ] += 1

        clinical_selectivity = [
            clinical - nuisance
            for clinical, nuisance in zip(signed_clinical_changes, nuisance_changes)
        ]
        support_pairs = list(zip(supports, support_polarities))
        support_spearman_result = bootstrap_spearman(
            support_pairs, args.bootstrap_draws, args.seed + 500 + layer
        )
        support_spearman = support_spearman_result["estimate"]
        clear_state_auroc = auc(labels, polarities)
        responsive_count = sum(unsigned_responsive)
        mean_absolute_clinical = mean(absolute_clinical_changes)
        per_layer[str(layer)] = {
            "claim_state_auroc": clear_state_auroc,
            "unanimous_claim_state_auroc": clear_state_auroc,
            "reader_support_spearman": support_spearman,
            "reader_support_spearman_bootstrap": support_spearman_result,
            "vote_bin_mean_polarity": {
                str(votes): (
                    mean(
                        polarity
                        for support, polarity in support_pairs
                        if support == votes / 3
                    )
                    if any(support == votes / 3 for support, _ in support_pairs)
                    else None
                )
                for votes in range(4)
            },
            "mean_absolute_nuisance_change": mean(nuisance_changes),
            "mean_signed_clinical_change": mean(signed_clinical_changes),
            "mean_absolute_clinical_change": mean_absolute_clinical,
            "directional_efficiency": (
                mean(signed_clinical_changes) / mean_absolute_clinical
                if mean_absolute_clinical > 0.0
                else None
            ),
            "clinical_selectivity_gap": bootstrap_mean(
                clinical_selectivity, args.bootstrap_draws, args.seed + layer
            ),
            "magnitude_selectivity_gap": bootstrap_mean(
                magnitude_selectivity, args.bootstrap_draws, args.seed + 100 + layer
            ),
            "opposite_state_pairwise_accuracy": mean(pairwise_correct),
            "wrong_direction_rate": mean(1.0 - value for value in pairwise_correct),
            "unsigned_responsive_rate": mean(unsigned_responsive),
            "misdirected_responsive_rate": mean(misdirected_responsive),
            "misdirection_given_unsigned_responsive": (
                sum(misdirected_responsive) / responsive_count
                if responsive_count > 0.0
                else None
            ),
            "anchor_error_count": sum(anchor_error_labels),
            "anchor_error_rate": mean(anchor_error_labels),
            "error_detection_auroc": {
                "low_output_margin": auc(anchor_error_labels, margin_error_scores),
                "low_signed_clinical_response": auc(
                    anchor_error_labels, signed_response_error_scores
                ),
                "low_unsigned_selectivity": auc(
                    anchor_error_labels, unsigned_selectivity_error_scores
                ),
                "low_clinical_selectivity_gap": auc(
                    anchor_error_labels, csg_error_scores
                ),
            },
            "csg_vs_output_margin_error_auroc_delta": bootstrap_auc_delta(
                anchor_error_labels,
                csg_error_scores,
                margin_error_scores,
                args.bootstrap_draws,
                args.seed + 1000 + layer,
            ),
            "same_state_answer_flip_rate": mean(same_state_flips),
            "opposite_state_answer_flip_rate": mean(opposite_state_flips),
            "answer_flip_selectivity_gap": mean(opposite_state_flips)
            - mean(same_state_flips),
            "by_finding": {
                finding: {
                    "n_triplets": len(values),
                    "anchor_vote_bin_counts": {
                        str(votes): anchor_vote_counts[finding].get(votes, 0)
                        for votes in range(4)
                    },
                    "mean_signed_clinical_change": mean(v[0] for v in values),
                    "mean_absolute_nuisance_change": mean(v[1] for v in values),
                    "clinical_selectivity_gap": mean(v[0] - v[1] for v in values),
                    "clinical_selectivity_gap_bootstrap": bootstrap_mean(
                        [v[0] - v[1] for v in values],
                        args.bootstrap_draws,
                        args.seed + 2000 + layer + sum(map(ord, finding)),
                    ),
                    "reader_support_spearman_bootstrap": bootstrap_spearman(
                        support_pairs_by_finding[finding],
                        args.bootstrap_draws,
                        args.seed + 3000 + layer + sum(map(ord, finding)),
                    ),
                    "vote_bin_mean_polarity": {
                        str(votes): (
                            mean(
                                polarity
                                for support, polarity in support_pairs_by_finding[finding]
                                if support == votes / 3
                            )
                            if any(
                                support == votes / 3
                                for support, _ in support_pairs_by_finding[finding]
                            )
                            else None
                        )
                        for votes in range(4)
                    },
                }
                for finding, values in sorted(by_finding.items())
            },
        }

    contracts = [contract for _, contract in joined]
    evidence_grades = sorted(
        {str(contract.get("evidence_grade", "ungraded")) for contract in contracts}
    )
    formal_reference = all(
        contract.get("formal_reference") is True
        and contract.get("reference_source") == "vindr_reader_votes"
        for contract in contracts
    )
    selected_layer = None
    dev_selection_eligible = None
    dev_analysis_sha256 = None
    if args.experiment_split == "dev":
        selected_layer, dev_selection_eligible = select_dev_layer(per_layer)
    elif args.experiment_split == "test":
        dev_result = json.loads(args.dev_analysis.read_text(encoding="utf-8"))
        if dev_result.get("experiment_split") != "dev":
            raise ValueError("--dev-analysis must be a dev-split analysis")
        if dev_result.get("formal_reference") is not True:
            raise ValueError("dev layer selection requires formal reader-vote references")
        if dev_result.get("model_id") != args.model_id:
            raise ValueError("dev/test model_id mismatch")
        selected_layer = int(dev_result["recommended_layer"])
        dev_selection_eligible = bool(dev_result["dev_selection_eligible"])
        if str(selected_layer) not in per_layer:
            raise ValueError(f"dev-selected layer absent from test: {selected_layer}")
        dev_analysis_sha256 = sha256_file(args.dev_analysis)

    result = {
        "version": VERSION,
        "model_id": args.model_id,
        "evidence_grades": evidence_grades,
        "evidence_grade": evidence_grades[0] if len(evidence_grades) == 1 else "mixed",
        "formal_reference": formal_reference,
        "experiment_split": args.experiment_split,
        "successful_records": len(joined),
        "complete_triplets": len(complete),
        "incomplete_triplets": sorted(set(triplets) - set(complete)),
        "interpretation": (
            "A model is clinically selective only when opposite-state swaps "
            "move claim polarity in the correct direction more than same-state "
            "patient/image swaps move it for nuisance reasons."
        ),
        "per_layer": per_layer,
    }
    if args.experiment_split == "dev":
        result.update(
            {
                "recommended_layer": selected_layer,
                "dev_selection_eligible": dev_selection_eligible,
                "layer_selection_rule": (
                    "maximum dev signed clinical-selectivity gap among layers with "
                    "positive dev support ordering and selectivity; ties use support "
                    "ordering then the earlier layer; locked test is never consulted"
                ),
            }
        )
    elif args.experiment_split == "test":
        gates, finding_summary = directional_admission_gates(
            per_layer[str(selected_layer)],
            formal_reference,
            bool(dev_selection_eligible),
            args.min_test_triplets_per_bin,
        )
        result.update(
            {
                "selected_layer": selected_layer,
                "dev_analysis": str(args.dev_analysis.resolve()),
                "dev_analysis_sha256": dev_analysis_sha256,
                "test_layer_selected_without_test_labels": True,
                "finding_gate_summary": finding_summary,
                "mechanism_gates": gates,
            }
        )
    result["config"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "raw": [str(path.resolve()) for path in args.raw],
        "raw_sha256": {str(path.resolve()): sha256_file(path) for path in args.raw},
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "command": shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]]),
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
