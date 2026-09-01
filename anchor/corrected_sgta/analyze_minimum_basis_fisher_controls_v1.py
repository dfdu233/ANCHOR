"""Fatal simple-baseline controls for minimum-intervention-basis Fisher claims.

The existing basis screen ranks every non-empty subset of five intervention
arms by a source-train Fisher distance and compares that ranking with target
balanced accuracy.  This CPU-only audit asks whether Fisher adds anything
after accounting for cheaper explanations available before target labels:

* subset size;
* the subset decoder's source-validation balanced accuracy;
* the mean and maximum source-validation performance of its member arms; and
* source-validation pairwise error diversity and error correlation.

The audit reads the existing JSON artifacts and cached expert outputs.  It does
not regenerate answers or write into any baseline directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, t as student_t

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import _split
from anchor.corrected_sgta.analyze_minimum_intervention_basis_v1 import load


ROOT = Path("corrected_runs")
FORWARD = ROOT / "minimum_intervention_basis_v1/result.json"
REVERSE = ROOT / "minimum_intervention_basis_v1/reverse_result.json"
INTERVENTION_CODE = ROOT / "intervention_code_cxr_v2/result.json"
OUT = ROOT / "minimum_intervention_basis_fisher_controls_v1/result.json"
PERMUTATIONS = 20_000
SEED = 20260810


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    positive = y == 1
    negative = y == 0
    tpr = float(np.mean(pred[positive] == 1)) if np.any(positive) else 0.0
    tnr = float(np.mean(pred[negative] == 0)) if np.any(negative) else 0.0
    return (tpr + tnr) / 2.0


def pairwise_error_features(
    subset: list[str],
    names: list[str],
    predictions: np.ndarray,
    target: np.ndarray,
    validation: np.ndarray,
) -> dict[str, Any]:
    indices = [names.index(name) for name in subset]
    if len(indices) < 2:
        return {
            "pair_count": 0,
            "mean_pairwise_error_diversity": None,
            "mean_pairwise_error_correlation": None,
            "pairwise": [],
        }
    errors = predictions[validation][:, indices] != target[validation, None]
    pairs = []
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            e1 = errors[:, left].astype(float)
            e2 = errors[:, right].astype(float)
            diversity = float(np.mean(e1 != e2))
            if float(np.std(e1)) == 0.0 or float(np.std(e2)) == 0.0:
                correlation = 0.0
                correlation_defined = False
            else:
                correlation = float(np.corrcoef(e1, e2)[0, 1])
                correlation_defined = True
            pairs.append({
                "arms": [subset[left], subset[right]],
                "error_diversity": diversity,
                "error_correlation": correlation,
                "error_correlation_defined": correlation_defined,
            })
    return {
        "pair_count": len(pairs),
        "mean_pairwise_error_diversity": float(np.mean([
            pair["error_diversity"] for pair in pairs
        ])),
        "mean_pairwise_error_correlation": float(np.mean([
            pair["error_correlation"] for pair in pairs
        ])),
        "pairwise": pairs,
    }


def size_dummies(sizes: np.ndarray) -> tuple[np.ndarray, list[str]]:
    levels = sorted(set(int(value) for value in sizes))
    reference = levels[0]
    columns = [
        (sizes == level).astype(float) for level in levels if level != reference
    ]
    matrix = np.column_stack(columns) if columns else np.empty((len(sizes), 0))
    return matrix, [f"subset_size_eq_{level}" for level in levels if level != reference]


def residualize(values: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, int]:
    design = np.column_stack([np.ones(len(values)), controls])
    fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    return values - fitted, int(np.linalg.matrix_rank(design) - 1)


def partial_rank(
    fisher: np.ndarray,
    target_bacc: np.ndarray,
    controls: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    fisher_rank = rankdata(fisher, method="average")
    target_rank = rankdata(target_bacc, method="average")
    ranked_controls = np.column_stack([
        rankdata(controls[:, column], method="average")
        for column in range(controls.shape[1])
    ]) if controls.shape[1] else controls
    fisher_residual, control_rank = residualize(fisher_rank, ranked_controls)
    target_residual, _ = residualize(target_rank, ranked_controls)
    rho = float(np.corrcoef(fisher_residual, target_residual)[0, 1])
    df = len(fisher) - control_rank - 2
    if df > 0 and abs(rho) < 1.0:
        statistic = rho * np.sqrt(df / max(1e-15, 1.0 - rho * rho))
        p_parametric = float(2.0 * student_t.sf(abs(statistic), df))
    else:
        statistic = None
        p_parametric = None
    observed = abs(rho)
    exceed = 0
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(target_residual)
        value = float(np.corrcoef(fisher_residual, permuted)[0, 1])
        exceed += abs(value) >= observed
    return {
        "partial_spearman": rho,
        "degrees_of_freedom": int(df),
        "t_statistic": statistic,
        "parametric_p_two_sided": p_parametric,
        "residual_permutation_p_two_sided": float((exceed + 1) / (PERMUTATIONS + 1)),
        "permutations": PERMUTATIONS,
    }


def standardize(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=float).copy()
    for column in range(output.shape[1]):
        mean = float(np.mean(output[:, column]))
        scale = float(np.std(output[:, column]))
        output[:, column] = (output[:, column] - mean) / scale if scale > 0 else 0.0
    return output


def ols(values: np.ndarray, predictors: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([np.ones(len(values)), predictors])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ beta
    sse = float(residual @ residual)
    centered = values - float(np.mean(values))
    sst = float(centered @ centered)
    rank = int(np.linalg.matrix_rank(design))
    df = len(values) - rank
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    adjusted = 1.0 - (1.0 - r2) * (len(values) - 1) / df if df > 0 else None
    covariance = (sse / df) * np.linalg.pinv(design.T @ design) if df > 0 else None
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance))) if covariance is not None else None
    return {
        "beta": beta,
        "standard_errors": standard_errors,
        "sse": sse,
        "r2": float(r2),
        "adjusted_r2": float(adjusted) if adjusted is not None else None,
        "degrees_of_freedom": int(df),
    }


def nested_regression(
    fisher: np.ndarray,
    target_bacc: np.ndarray,
    controls: np.ndarray,
) -> dict[str, Any]:
    ranked_controls = np.column_stack([
        rankdata(controls[:, column], method="average")
        for column in range(controls.shape[1])
    ]) if controls.shape[1] else controls
    ranked_controls = standardize(ranked_controls)
    ranked_fisher = standardize(rankdata(fisher, method="average")[:, None])[:, 0]
    ranked_target = standardize(rankdata(target_bacc, method="average")[:, None])[:, 0]
    base = ols(ranked_target, ranked_controls)
    full = ols(ranked_target, np.column_stack([ranked_controls, ranked_fisher]))
    beta = float(full["beta"][-1])
    standard_error = float(full["standard_errors"][-1]) if full["standard_errors"] is not None else None
    if standard_error is not None and standard_error > 0 and full["degrees_of_freedom"] > 0:
        statistic = beta / standard_error
        p_value = float(2.0 * student_t.sf(abs(statistic), full["degrees_of_freedom"]))
    else:
        statistic = None
        p_value = None
    return {
        "standardized_fisher_coefficient": beta,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "p_two_sided": p_value,
        "base_r2": base["r2"],
        "full_r2": full["r2"],
        "delta_r2_from_fisher": float(full["r2"] - base["r2"]),
        "base_adjusted_r2": base["adjusted_r2"],
        "full_adjusted_r2": full["adjusted_r2"],
        "degrees_of_freedom": full["degrees_of_freedom"],
    }


def analyze_direction(
    artifact: dict[str, Any],
    source_key: str,
    label: str,
) -> dict[str, Any]:
    source = load(source_key)
    names = list(source["names"])
    split = np.asarray([_split(cluster) for cluster in source["cluster"]])
    validation = np.flatnonzero(split == "validation")
    direct_single_bacc = {
        name: balanced_accuracy(
            source["target"][validation],
            source["pred"][validation, column],
        )
        for column, name in enumerate(names)
    }
    singleton_decoder_bacc = {
        row["arms"][0]: float(row["source_validation_bacc"])
        for row in artifact["all_subsets"] if len(row["arms"]) == 1
    }
    rows = []
    for original in artifact["all_subsets"]:
        arms = list(original["arms"])
        pairwise = pairwise_error_features(
            arms, names, source["pred"], source["target"], validation
        )
        member_decoder = [singleton_decoder_bacc[name] for name in arms]
        member_direct = [direct_single_bacc[name] for name in arms]
        rows.append({
            **original,
            "subset_size": len(arms),
            "mean_member_single_source_validation_bacc": float(np.mean(member_decoder)),
            "max_member_single_source_validation_bacc": float(np.max(member_decoder)),
            "mean_member_direct_arm_source_validation_bacc": float(np.mean(member_direct)),
            "max_member_direct_arm_source_validation_bacc": float(np.max(member_direct)),
            **pairwise,
        })

    # Pairwise controls are undefined for singleton subsets; the fatal full
    # comparison is therefore intentionally restricted to k>=2.
    multi = [row for row in rows if row["subset_size"] >= 2]
    sizes = np.asarray([row["subset_size"] for row in multi])
    size_matrix, size_names = size_dummies(sizes)
    performance_matrix = np.column_stack([
        [row["source_validation_bacc"] for row in multi],
        [row["mean_member_single_source_validation_bacc"] for row in multi],
        [row["max_member_single_source_validation_bacc"] for row in multi],
    ])
    diversity_matrix = np.column_stack([
        [row["mean_pairwise_error_diversity"] for row in multi],
        [row["mean_pairwise_error_correlation"] for row in multi],
    ])
    control_sets = {
        "subset_size_fixed_effects": (
            size_matrix,
            size_names,
        ),
        "plus_source_performance": (
            np.column_stack([size_matrix, performance_matrix]),
            size_names + [
                "source_validation_bacc",
                "mean_member_single_source_validation_bacc",
                "max_member_single_source_validation_bacc",
            ],
        ),
        "plus_pairwise_error_structure": (
            np.column_stack([size_matrix, performance_matrix, diversity_matrix]),
            size_names + [
                "source_validation_bacc",
                "mean_member_single_source_validation_bacc",
                "max_member_single_source_validation_bacc",
                "mean_pairwise_error_diversity",
                "mean_pairwise_error_correlation",
            ],
        ),
    }
    fisher = np.asarray([row["fisher_train"] for row in multi], dtype=float)
    target_bacc = np.asarray([row["target_bacc"] for row in multi], dtype=float)
    rng = np.random.default_rng(SEED + (0 if source_key == "source" else 1))
    adjusted = {
        name: {
            "controls": control_names,
            "partial_rank": partial_rank(fisher, target_bacc, controls, rng),
            "nested_rank_regression": nested_regression(fisher, target_bacc, controls),
        }
        for name, (controls, control_names) in control_sets.items()
    }

    full_controls = control_sets["plus_pairwise_error_structure"][0]
    jackknife = {}
    for arm in names:
        keep = np.asarray([arm not in row["arms"] for row in multi])
        if int(np.sum(keep)) < 8:
            continue
        jackknife[arm] = {
            "n": int(np.sum(keep)),
            **partial_rank(
                fisher[keep], target_bacc[keep], full_controls[keep], rng
            ),
        }

    return {
        "label": label,
        "source": artifact.get("source"),
        "target": artifact.get("target"),
        "n_all_subsets": len(rows),
        "n_multiarm_subsets_for_full_controls": len(multi),
        "source_validation_n": int(len(validation)),
        "direct_single_arm_source_validation_bacc": direct_single_bacc,
        "singleton_decoder_source_validation_bacc": singleton_decoder_bacc,
        "adjusted_fisher_tests": adjusted,
        "leave_one_arm_out_full_control_partial_rank": jackknife,
        "rows": rows,
    }


def main() -> None:
    forward = json.loads(FORWARD.read_text())
    reverse = json.loads(REVERSE.read_text())
    intervention = json.loads(INTERVENTION_CODE.read_text())
    intervention_arms = intervention["cohorts"]["full_five_arm_code"]["arms"]
    if list(forward["candidate_arms"]) != list(intervention_arms):
        raise RuntimeError("candidate-arm mismatch between basis and intervention-code artifacts")
    directions = {
        "knowledge_mimic_to_cxr_vishal": analyze_direction(
            forward, "source", "Knowledge-MIMIC CE -> CXR-VisHal"
        ),
        "cxr_vishal_to_knowledge_mimic": analyze_direction(
            reverse, "target", "CXR-VisHal -> Knowledge-MIMIC CE"
        ),
    }
    raw_rank = {
        "knowledge_mimic_to_cxr_vishal": float(
            forward["rank_prediction"]["fisher_vs_target_bacc_spearman"]
        ),
        "cxr_vishal_to_knowledge_mimic": float(
            reverse["fisher_vs_target_bacc_spearman"]
        ),
    }
    direction_decisions = {}
    for name, direction in directions.items():
        full = direction["adjusted_fisher_tests"]["plus_pairwise_error_structure"]
        partial = full["partial_rank"]
        regression = full["nested_rank_regression"]
        jackknife_rhos = [
            item["partial_spearman"]
            for item in direction["leave_one_arm_out_full_control_partial_rank"].values()
        ]
        direction_decisions[name] = {
            "raw_fisher_target_spearman": raw_rank[name],
            "full_control_partial_spearman": partial["partial_spearman"],
            "full_control_parametric_p": partial["parametric_p_two_sided"],
            "full_control_residual_permutation_p": partial[
                "residual_permutation_p_two_sided"
            ],
            "full_control_delta_r2": regression["delta_r2_from_fisher"],
            "positive_and_parametric_p_lt_0p05": bool(
                partial["partial_spearman"] > 0
                and partial["parametric_p_two_sided"] is not None
                and partial["parametric_p_two_sided"] < 0.05
            ),
            "leave_one_arm_out_sign_stable": bool(
                jackknife_rhos and min(jackknife_rhos) > 0
            ),
            "leave_one_arm_out_rho_range": [
                float(min(jackknife_rhos)), float(max(jackknife_rhos))
            ],
        }
    survives = all(
        item["positive_and_parametric_p_lt_0p05"]
        and item["leave_one_arm_out_sign_stable"]
        for item in direction_decisions.values()
    )
    result = {
        "status": "cached_cpu_fatal_simple_baseline_control",
        "version": "minimum-basis-fisher-controls-v1",
        "inputs": {
            str(FORWARD): sha256_file(FORWARD),
            str(REVERSE): sha256_file(REVERSE),
            str(INTERVENTION_CODE): sha256_file(INTERVENTION_CODE),
        },
        "definitions": {
            "single_arm_performance": (
                "source-validation BAcc of each singleton linear decoder; direct raw-arm "
                "BAcc is also reported as an audit"
            ),
            "pairwise_error_diversity": (
                "mean source-validation probability that exactly one of two arms is wrong"
            ),
            "pairwise_error_correlation": (
                "mean source-validation Pearson/phi correlation of binary error indicators"
            ),
            "target_labels_in_controls": False,
            "full_control_population": "all k>=2 subsets; singleton pairwise structure is undefined",
            "inference_limitation": (
                "The 26 multi-arm rows are overlapping combinations of only five arms, not "
                "independent experimental units. Parametric and row-residual permutation p-values "
                "are descriptive; direction replication and leave-one-arm-out sign stability are "
                "the fatal decision criteria."
            ),
        },
        "intervention_code_crosscheck": {
            "dataset": intervention.get("dataset"),
            "arms": intervention_arms,
            "five_fold_crossfit": intervention["cohorts"]["full_five_arm_code"][
                "five_fold_crossfit"
            ],
        },
        "verdict": {
            "fisher_survives_fatal_simple_controls": survives,
            "decision": (
                "RETAIN_FISHER_AS_CONDITIONAL_SELECTOR"
                if survives
                else "REJECT_FISHER_AS_INDEPENDENT_TRANSFERABLE_SELECTOR"
            ),
            "by_direction": direction_decisions,
            "interpretation": (
                "A positive raw subset ranking is insufficient. Fisher must add significant "
                "rank information after source-only strength/diversity controls in both "
                "directions and retain its sign when any one arm is removed."
            ),
        },
        "directions": directions,
        "decision_rule": (
            "Fisher survives only if its full-control partial rank and nested-regression "
            "increment remain directionally positive in both domain directions and are not "
            "carried by one arm in leave-one-arm-out sensitivity. This is a diagnostic "
            "screen, not an independent target confirmation, because target partitions were "
            "already used by the parent basis analyses."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(OUT),
        "directions": {
            key: value["adjusted_fisher_tests"]
            for key, value in result["directions"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
