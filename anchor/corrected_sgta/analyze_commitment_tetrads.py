#!/usr/bin/env python3
"""Training-free layerwise audit of VinDr reader-support commitment tetrads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from corrected_sgta.clinical_claims import epistemic_coordinates


VERSION = "commitment-tetrad-analysis-v1"
ROLES = {"clear_a", "clear_b", "ambiguous_a", "ambiguous_b"}
MINIMUM_TEST_TETRADS_PER_BRANCH = 10


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(labels: list[int], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return None
    return sum(
        float(pos > neg) + 0.5 * float(pos == neg)
        for pos in positive
        for neg in negative
    ) / (len(positive) * len(negative))


def _coordinates(logits: Mapping[str, float]) -> tuple[float, float]:
    value = epistemic_coordinates(logits)
    return float(value["polarity"]), float(value["commitment"])


def _macro(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return mean(valid) if valid else None


def _layer_metrics(
    tetrads: list[dict[str, Any]], layer: int
) -> dict[str, float | dict[str, float | None] | None]:
    member_rows = [member for tetrad in tetrads for member in tetrad["members"]]
    support_scores: list[float] = []
    commitment_scores: list[float] = []
    labels: list[int] = []
    branches: list[str] = []
    definite: list[int] = []
    support_gaps = []
    support_nuisance = []
    commitment_gaps = []
    commitment_nuisance = []
    for tetrad in tetrads:
        by_role = {str(member["tetrad_role"]): member for member in tetrad["members"]}
        sign = 1.0 if tetrad["majority_polarity"] == "positive" else -1.0

        def coordinates(role: str) -> tuple[float, float]:
            logits = by_role[role]["measurement"]["trajectory"][str(layer)]["real_logits"]
            polarity, commitment = _coordinates(logits)
            return sign * polarity, commitment

        clear = [coordinates("clear_a"), coordinates("clear_b")]
        ambiguous = [coordinates("ambiguous_a"), coordinates("ambiguous_b")]
        support_gaps.append(mean(value[0] for value in clear) - mean(value[0] for value in ambiguous))
        support_nuisance.append(
            0.5 * (abs(clear[0][0] - clear[1][0]) + abs(ambiguous[0][0] - ambiguous[1][0]))
        )
        commitment_gaps.append(mean(value[1] for value in clear) - mean(value[1] for value in ambiguous))
        commitment_nuisance.append(
            0.5 * (abs(clear[0][1] - clear[1][1]) + abs(ambiguous[0][1] - ambiguous[1][1]))
        )

    for member in member_rows:
        branch = str(member["majority_polarity"])
        sign = 1.0 if branch == "positive" else -1.0
        trajectory = member["measurement"]["trajectory"][str(layer)]
        polarity, commitment = _coordinates(trajectory["real_logits"])
        is_clear = int(str(member["tetrad_role"]).startswith("clear"))
        support_scores.append(sign * polarity)
        commitment_scores.append(commitment)
        labels.append(is_clear)
        branches.append(branch)
        definite.append(int(str(trajectory["baseline_state"]) != "undetermined"))

    support_auc_by_branch = {}
    commitment_auc_by_branch = {}
    for branch in ("negative", "positive"):
        indices = [index for index, value in enumerate(branches) if value == branch]
        support_auc_by_branch[branch] = auc(
            [labels[index] for index in indices],
            [support_scores[index] for index in indices],
        )
        commitment_auc_by_branch[branch] = auc(
            [labels[index] for index in indices],
            [commitment_scores[index] for index in indices],
        )
    ambiguous_indices = [index for index, label in enumerate(labels) if label == 0]
    ambiguous_definite_by_branch = {}
    for branch in ("negative", "positive"):
        values = [
            definite[index]
            for index in ambiguous_indices
            if branches[index] == branch
        ]
        ambiguous_definite_by_branch[branch] = mean(values) if values else None
    return {
        "support_auroc_by_majority_polarity": support_auc_by_branch,
        "support_macro_auroc": _macro(support_auc_by_branch.values()),
        "commitment_auroc_by_majority_polarity": commitment_auc_by_branch,
        "commitment_macro_auroc": _macro(commitment_auc_by_branch.values()),
        "mean_directed_support_gap": mean(support_gaps),
        "mean_support_nuisance": mean(support_nuisance),
        "mean_support_selectivity_gap": mean(
            gap - nuisance for gap, nuisance in zip(support_gaps, support_nuisance)
        ),
        "directed_support_pair_accuracy": mean(float(value > 0.0) for value in support_gaps),
        "mean_commitment_gap": mean(commitment_gaps),
        "mean_commitment_nuisance": mean(commitment_nuisance),
        "mean_commitment_selectivity_gap": mean(
            gap - nuisance for gap, nuisance in zip(commitment_gaps, commitment_nuisance)
        ),
        "ambiguous_definite_rate": mean(definite[index] for index in ambiguous_indices),
        "ambiguous_definite_rate_by_majority_polarity": ambiguous_definite_by_branch,
    }


def _bootstrap(
    tetrads: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    draws: int,
    seed: int,
) -> dict[str, float | int | None]:
    observed = statistic(tetrads)
    rng = np.random.default_rng(seed)
    by_branch = {
        branch: [row for row in tetrads if row["majority_polarity"] == branch]
        for branch in sorted({str(row["majority_polarity"]) for row in tetrads})
    }
    values = []
    for _ in range(draws):
        sample = [
            row
            for rows in by_branch.values()
            for row in (
                rows[index]
                for index in rng.integers(0, len(rows), len(rows))
            )
        ]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return {
        "estimate": observed,
        "ci_low": float(np.quantile(values, 0.025)) if values else None,
        "ci_high": float(np.quantile(values, 0.975)) if values else None,
        "valid_draws": len(values),
    }


def analyze_commitment_tetrads(
    manifest_rows: Iterable[Mapping[str, Any]],
    raw_rows: Iterable[Mapping[str, Any]],
    bootstrap_draws: int,
    seed: int,
) -> dict[str, object]:
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    contracts: dict[tuple[str, str], dict[str, Any]] = {}
    for source in manifest_rows:
        row = dict(source)
        key = (str(row["finding"]), str(row["image_id"]))
        if key in contracts:
            raise ValueError(f"duplicate manifest key: {key}")
        if row.get("formal_reference") is not True or row.get("reference_source") != "vindr_reader_votes":
            raise ValueError("formal tetrad analysis requires VinDr reader-vote provenance")
        if str(row.get("tetrad_role")) not in ROLES:
            raise ValueError(f"invalid tetrad role for {key}")
        contracts[key] = row

    measured: dict[tuple[str, str], dict[str, Any]] = {}
    for source in raw_rows:
        if source.get("status") != "ok":
            continue
        key = (str(source["finding"]), str(source["image_id"]))
        if key in measured:
            raise ValueError(f"duplicate model record: {key}")
        if key not in contracts:
            raise ValueError(f"model record absent from tetrad manifest: {key}")
        measured[key] = dict(source)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, contract in contracts.items():
        if key not in measured:
            continue
        row = dict(measured[key])
        row.update(
            {
                "tetrad_id": contract["tetrad_id"],
                "tetrad_role": contract["tetrad_role"],
                "majority_polarity": contract["majority_polarity"],
                "experiment_split": contract["experiment_split"],
                "contract_finding": contract["finding"],
            }
        )
        grouped[str(contract["tetrad_id"])].append(row)

    tetrads = []
    incomplete = 0
    for tetrad_id, members in grouped.items():
        if {str(row["tetrad_role"]) for row in members} != ROLES:
            incomplete += 1
            continue
        splits = {str(row["experiment_split"]) for row in members}
        branches = {str(row["majority_polarity"]) for row in members}
        findings = {str(row["contract_finding"]) for row in members}
        if len(splits) != 1 or len(branches) != 1 or len(findings) != 1:
            raise ValueError(f"inconsistent tetrad contract: {tetrad_id}")
        tetrads.append(
            {
                "tetrad_id": tetrad_id,
                "split": next(iter(splits)),
                "majority_polarity": next(iter(branches)),
                "finding": next(iter(findings)),
                "members": members,
            }
        )
    if not tetrads:
        raise ValueError("no complete measured tetrads")
    layer_sets = {
        tuple(sorted(int(layer) for layer in member["measurement"]["trajectory"]))
        for tetrad in tetrads
        for member in tetrad["members"]
    }
    if len(layer_sets) != 1:
        raise ValueError(f"inconsistent layer trajectories: {layer_sets}")
    layers = list(next(iter(layer_sets)))
    if len(layers) < 2:
        raise ValueError("at least one intermediate and one final layer are required")
    dev = [row for row in tetrads if row["split"] == "dev"]
    test = [row for row in tetrads if row["split"] == "test"]
    for name, rows in (("dev", dev), ("test", test)):
        if {row["majority_polarity"] for row in rows} != {"negative", "positive"}:
            raise ValueError(f"{name} requires both majority-polarity branches")

    dev_metrics = {str(layer): _layer_metrics(dev, layer) for layer in layers}
    test_metrics = {str(layer): _layer_metrics(test, layer) for layer in layers}
    final_layer = layers[-1]
    selected_layer = max(
        layers[:-1],
        key=lambda layer: (
            float(dev_metrics[str(layer)]["support_macro_auroc"] or -math.inf),
            float(dev_metrics[str(layer)]["mean_support_selectivity_gap"]),
            -layer,
        ),
    )

    def metric(rows: list[dict[str, Any]], layer: int, name: str) -> float | None:
        value = _layer_metrics(rows, layer)[name]
        return None if value is None else float(value)

    early_signal = _bootstrap(
        test,
        lambda rows: (
            None
            if metric(rows, selected_layer, "support_macro_auroc") is None
            else metric(rows, selected_layer, "support_macro_auroc") - 0.5
        ),
        bootstrap_draws,
        seed,
    )
    erasure = _bootstrap(
        test,
        lambda rows: (
            None
            if metric(rows, selected_layer, "support_macro_auroc") is None
            or metric(rows, final_layer, "support_macro_auroc") is None
            else metric(rows, selected_layer, "support_macro_auroc")
            - metric(rows, final_layer, "support_macro_auroc")
        ),
        bootstrap_draws,
        seed + 1,
    )
    selectivity = _bootstrap(
        test,
        lambda rows: metric(rows, selected_layer, "mean_support_selectivity_gap"),
        bootstrap_draws,
        seed + 2,
    )
    branch_tests = {}
    for offset, branch in enumerate(("negative", "positive"), start=1):
        branch_rows = [row for row in test if row["majority_polarity"] == branch]
        branch_tests[branch] = {
            "selected_early_auroc_minus_chance": _bootstrap(
                branch_rows,
                lambda rows: (
                    None
                    if metric(rows, selected_layer, "support_macro_auroc") is None
                    else metric(rows, selected_layer, "support_macro_auroc") - 0.5
                ),
                bootstrap_draws,
                seed + 10 * offset,
            ),
            "selected_early_minus_final_support_auroc": _bootstrap(
                branch_rows,
                lambda rows: (
                    None
                    if metric(rows, selected_layer, "support_macro_auroc") is None
                    or metric(rows, final_layer, "support_macro_auroc") is None
                    else metric(rows, selected_layer, "support_macro_auroc")
                    - metric(rows, final_layer, "support_macro_auroc")
                ),
                bootstrap_draws,
                seed + 10 * offset + 1,
            ),
            "selected_early_support_selectivity": _bootstrap(
                branch_rows,
                lambda rows: metric(
                    rows, selected_layer, "mean_support_selectivity_gap"
                ),
                bootstrap_draws,
                seed + 10 * offset + 2,
            ),
        }
    final_overcommitment = float(
        test_metrics[str(final_layer)]["ambiguous_definite_rate"]
    )
    final_overcommitment_by_branch = test_metrics[str(final_layer)][
        "ambiguous_definite_rate_by_majority_polarity"
    ]

    def tests_pass(
        tests: Mapping[str, Mapping[str, float | int | None]],
        final_definite_rate: float,
        tetrad_count: int,
    ) -> bool:
        signal = tests["selected_early_auroc_minus_chance"]
        loss = tests["selected_early_minus_final_support_auroc"]
        selective = tests["selected_early_support_selectivity"]
        return bool(
            tetrad_count >= MINIMUM_TEST_TETRADS_PER_BRANCH
            and signal["ci_low"] is not None
            and float(signal["ci_low"]) > 0.0
            and selective["ci_low"] is not None
            and float(selective["ci_low"]) > 0.0
            and loss["estimate"] is not None
            and float(loss["estimate"]) >= 0.05
            and loss["ci_low"] is not None
            and float(loss["ci_low"]) > 0.0
            and final_definite_rate >= 0.5
        )

    branch_counts = Counter(row["majority_polarity"] for row in test)

    def branch_passes(branch: str) -> bool:
        return tests_pass(
            branch_tests[branch],
            float(final_overcommitment_by_branch[branch]),
            int(branch_counts[branch]),
        )

    finding_tests = {}
    for finding_index, finding in enumerate(
        sorted({str(row["finding"]) for row in test}), start=1
    ):
        finding_rows = [row for row in test if row["finding"] == finding]
        finding_branches = Counter(row["majority_polarity"] for row in finding_rows)
        finding_final = _layer_metrics(finding_rows, final_layer)[
            "ambiguous_definite_rate_by_majority_polarity"
        ]
        per_branch = {}
        passes = {}
        for branch_index, branch in enumerate(("negative", "positive"), start=1):
            rows = [
                row for row in finding_rows if row["majority_polarity"] == branch
            ]
            if not rows:
                per_branch[branch] = {"status": "missing_branch"}
                passes[branch] = False
                continue
            bundle = {
                "selected_early_auroc_minus_chance": _bootstrap(
                    rows,
                    lambda sample: (
                        None
                        if metric(sample, selected_layer, "support_macro_auroc") is None
                        else metric(sample, selected_layer, "support_macro_auroc") - 0.5
                    ),
                    bootstrap_draws,
                    seed + 100 * finding_index + 10 * branch_index,
                ),
                "selected_early_minus_final_support_auroc": _bootstrap(
                    rows,
                    lambda sample: (
                        None
                        if metric(sample, selected_layer, "support_macro_auroc") is None
                        or metric(sample, final_layer, "support_macro_auroc") is None
                        else metric(sample, selected_layer, "support_macro_auroc")
                        - metric(sample, final_layer, "support_macro_auroc")
                    ),
                    bootstrap_draws,
                    seed + 100 * finding_index + 10 * branch_index + 1,
                ),
                "selected_early_support_selectivity": _bootstrap(
                    rows,
                    lambda sample: metric(
                        sample, selected_layer, "mean_support_selectivity_gap"
                    ),
                    bootstrap_draws,
                    seed + 100 * finding_index + 10 * branch_index + 2,
                ),
            }
            per_branch[branch] = bundle
            rate = finding_final[branch]
            passes[branch] = bool(
                rate is not None
                and tests_pass(
                    bundle, float(rate), int(finding_branches[branch])
                )
            )
        finding_tests[finding] = {
            "test_tetrads_by_majority_polarity": dict(finding_branches),
            "minimum_test_tetrads_per_branch": MINIMUM_TEST_TETRADS_PER_BRANCH,
            "final_ambiguous_definite_rate_by_majority_polarity": finding_final,
            "heldout_tests_by_majority_polarity": per_branch,
            "branch_passed": passes,
            "finding_passed": all(passes.values()),
        }
    qualified_findings = [
        finding
        for finding, values in finding_tests.items()
        if all(
            int(values["test_tetrads_by_majority_polarity"].get(branch, 0))
            >= MINIMUM_TEST_TETRADS_PER_BRANCH
            for branch in ("negative", "positive")
        )
    ]
    passed_findings = [
        finding
        for finding in qualified_findings
        if bool(finding_tests[finding]["finding_passed"])
    ]
    majority_findings_passed = bool(
        qualified_findings and len(passed_findings) > len(qualified_findings) / 2
    )

    gates = {
        "early_reader_support_signal_ci_above_chance": bool(
            early_signal["ci_low"] is not None and float(early_signal["ci_low"]) > 0.0
        ),
        "early_support_selectivity_ci_above_zero": bool(
            selectivity["ci_low"] is not None and float(selectivity["ci_low"]) > 0.0
        ),
        "early_minus_final_support_auroc_ge_0.05_ci_above_zero": bool(
            erasure["estimate"] is not None
            and float(erasure["estimate"]) >= 0.05
            and erasure["ci_low"] is not None
            and float(erasure["ci_low"]) > 0.0
        ),
        "final_disagreement_definite_rate_ge_0.5": final_overcommitment >= 0.5,
        "negative_majority_branch_passed": branch_passes("negative"),
        "positive_majority_branch_passed": branch_passes("positive"),
        "majority_of_qualified_findings_passed": majority_findings_passed,
    }
    gates["observational_erasure_authorized"] = all(gates.values())
    return {
        "version": VERSION,
        "status": "complete",
        "formal_reference": True,
        "layers": layers,
        "final_layer": final_layer,
        "selected_early_layer": selected_layer,
        "layer_selection_rule": (
            "maximum dev macro AUROC for unanimous versus disagreement using "
            "majority-directed polarity; tie by support selectivity then earlier layer"
        ),
        "complete_tetrads": len(tetrads),
        "incomplete_measured_tetrads": incomplete,
        "dev_tetrads": len(dev),
        "test_tetrads": len(test),
        "bootstrap_unit": "tetrad_id",
        "dev_layer_metrics": dev_metrics,
        "test_layer_metrics": test_metrics,
        "heldout_tests": {
            "selected_early_auroc_minus_chance": early_signal,
            "selected_early_minus_final_support_auroc": erasure,
            "selected_early_support_selectivity": selectivity,
            "by_majority_polarity": branch_tests,
            "by_finding": finding_tests,
        },
        "finding_gate_summary": {
            "minimum_test_tetrads_per_branch": MINIMUM_TEST_TETRADS_PER_BRANCH,
            "qualified_findings": qualified_findings,
            "passed_findings": passed_findings,
            "pass_fraction": (
                len(passed_findings) / len(qualified_findings)
                if qualified_findings
                else None
            ),
        },
        "mechanism_gates": gates,
        "identification_note": (
            "The tetrad estimates a matched conditional response, not a literal "
            "image intervention. A causal erasure claim additionally requires "
            "same-polarity activation patching and DCR eligibility."
        ),
        "claim_ceiling": (
            "Passing establishes layerwise loss of reader-support ordering at "
            "fixed majority polarity. It does not by itself establish a causal "
            "decoder bias or a hallucination-mitigation method."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-id",
        help="stable model identifier required by the formal projection authorization",
    )
    args = parser.parse_args()
    result = analyze_commitment_tetrads(
        load_jsonl(args.manifest),
        [row for path in args.raw for row in load_jsonl(path)],
        args.bootstrap_draws,
        args.seed,
    )
    result["model_id"] = args.model_id
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "vindr-cxr-1.0.0-reader-votes",
        "method": "training-free-majority-polarity-commitment-tetrad-audit",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "raw": [str(path.resolve()) for path in args.raw],
        "raw_sha256": {str(path.resolve()): sha256_file(path) for path in args.raw},
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    result["config"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
