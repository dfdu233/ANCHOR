#!/usr/bin/env python3
"""Outcome-blind power and split-provenance audit for CECD Stage 1.

This module deliberately reads only the VinDr reader-vote manifest, its
summary, and the CECD runner/analyzer *source*.  It never opens a model score,
packed factorial payload, sealed Stage-1 analysis, or clinician return.  The
paired-AUROC calculation uses an equal-variance binormal planning model and
deterministic Gauss-Hermite quadrature for the DeLong influence-function
variance of the candidate-minus-baseline AUROC.

The audit is a design artifact, not medical/model evidence and never
authorizes a GPU run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr, ndtri
from scipy.stats import norm


VERSION = "cecd-stage1-outcome-blind-power-audit-v1"
SELECTION_VERSION = "huatuo-dicom-render-pilot-v1"
DEFAULT_MANIFEST = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "manifests_v2/reader_vote_manifest_v2.jsonl"
)
DEFAULT_SUMMARY = DEFAULT_MANIFEST.with_name("summary_v2.json")
DEFAULT_RUNNER = Path("anchor/corrected_sgta/run_cecd_factorial_v1.py")
DEFAULT_ANALYZER = Path(
    "anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py"
)
DEFAULT_OUTPUT = Path(
    "corrected_runs/vindr_v2/cecd_stage1_power_audit_v1/power_audit.json"
)
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "pleural_effusion",
    "pulmonary_fibrosis",
)
VOTES = (0, 1, 2, 3)
SEED = 42
CI_Z = 1.96
MCID = 0.03
PLANNING_ALTERNATIVE = 0.05
BASELINE_AUC = 0.70
ERROR_PREVALENCE = 0.20
PAIRED_SCORE_CORRELATION = 0.95
CLUSTER_DESIGN_EFFECT = 1.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_record_key(row: Mapping[str, Any]) -> str:
    readable = f"{row['finding']}:{row['image_id']}"
    suffix = hashlib.sha256(readable.encode()).hexdigest()[:12]
    return f"{row['finding']}__{row['image_id']}__{suffix}"


def deterministic_selection(
    rows: Sequence[Mapping[str, Any]], split: str, per_bin: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for finding in FINDINGS:
        for vote in VOTES:
            group = [
                dict(row)
                for row in rows
                if str(row.get("experiment_split")) == split
                and str(row.get("finding")) == finding
                and int(row.get("positive_votes", -1)) == vote
            ]
            group.sort(
                key=lambda row: hashlib.sha256(
                    f"{SELECTION_VERSION}:{SEED}:{finding}:{vote}:{row['image_id']}".encode()
                ).hexdigest()
            )
            if len(group) < per_bin:
                raise ValueError(
                    f"{split}/{finding}/vote{vote} has {len(group)}, needs {per_bin}"
                )
            selected.extend(group[:per_bin])
    selected.sort(key=lambda row: (str(row["image_id"]), str(row["finding"])))
    return selected


def selection_audit(
    rows: Sequence[Mapping[str, Any]], stage: str, split: str, per_bin: int
) -> dict[str, Any]:
    selected = deterministic_selection(rows, split, per_bin)
    image_counts = Counter(str(row["image_id"]) for row in selected)
    bin_counts = Counter(
        (str(row["finding"]), int(row["positive_votes"])) for row in selected
    )
    keys = [safe_record_key(row) for row in selected]
    return {
        "stage": stage,
        "source_manifest_split": split,
        "per_finding_vote_bin": per_bin,
        "claim_count": len(selected),
        "unique_images": len(image_counts),
        "images_with_multiple_findings": sum(value > 1 for value in image_counts.values()),
        "maximum_claims_per_image": max(image_counts.values()),
        "image_cluster_size_histogram": {
            str(key): value for key, value in sorted(Counter(image_counts.values()).items())
        },
        "all_16_bins_exact": len(bin_counts) == 16
        and all(value == per_bin for value in bin_counts.values()),
        "selection_keys_sha256": canonical_sha256(keys),
        "selection_keys": keys,
        "image_ids": sorted(image_counts),
    }


def paired_auc_variance_constants(
    baseline_auc: float, delta_auc: float, score_correlation: float, order: int = 48
) -> tuple[float, float]:
    """Return positive/negative DeLong variance constants for paired AUCs.

    In the binormal model, negative scores have mean zero and positive scores
    have mean ``sqrt(2) * Phi^-1(AUC)``.  Candidate/baseline scores within an
    observation have the requested correlation.  The returned constants are
    variances of the paired positive- and negative-class influence functions.
    """

    if not 0 < baseline_auc < 1 or not 0 < baseline_auc + delta_auc < 1:
        raise ValueError("baseline and candidate AUCs must lie in (0, 1)")
    if not -1 < score_correlation < 1:
        raise ValueError("score correlation must lie in (-1, 1)")
    nodes, weights = hermgauss(order)
    z = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    z0 = z[:, None]
    z1 = z[None, :]
    joint_weights = weights[:, None] * weights[None, :]
    epsilon1 = score_correlation * z0 + math.sqrt(
        1.0 - score_correlation**2
    ) * z1
    mean0 = math.sqrt(2.0) * float(ndtri(baseline_auc))
    mean1 = math.sqrt(2.0) * float(ndtri(baseline_auc + delta_auc))
    positive_difference = ndtr(mean1 + epsilon1) - ndtr(mean0 + z0)
    negative_difference = ndtr(mean1 - epsilon1) - ndtr(mean0 - z0)

    def variance(values: np.ndarray) -> float:
        mean = float(np.sum(joint_weights * values))
        return float(np.sum(joint_weights * np.square(values - mean)))

    return variance(positive_difference), variance(negative_difference)


def auc_delta_se(
    *,
    n_per_vote_bin: int,
    finding_count: int,
    error_prevalence: float,
    positive_variance: float,
    negative_variance: float,
    design_effect: float,
) -> float:
    clear_total = 2.0 * n_per_vote_bin * finding_count
    error_count = clear_total * error_prevalence
    correct_count = clear_total * (1.0 - error_prevalence)
    return math.sqrt(
        design_effect
        * (positive_variance / error_count + negative_variance / correct_count)
    )


def normal_gate_power(
    true_delta: float, standard_error: float, *, require_point_mcid: bool
) -> float:
    threshold = CI_Z * standard_error
    if require_point_mcid:
        threshold = max(MCID, threshold)
    return float(norm.sf((threshold - true_delta) / standard_error))


def at_least_three_of_four(probability: float) -> float:
    return float(4.0 * probability**3 - 3.0 * probability**4)


def gate_row(
    n_per_bin: int,
    *,
    true_delta: float,
    error_prevalence: float = ERROR_PREVALENCE,
    score_correlation: float = PAIRED_SCORE_CORRELATION,
) -> dict[str, Any]:
    positive_variance, negative_variance = paired_auc_variance_constants(
        BASELINE_AUC, true_delta, score_correlation
    )
    finding_se = auc_delta_se(
        n_per_vote_bin=n_per_bin,
        finding_count=1,
        error_prevalence=error_prevalence,
        positive_variance=positive_variance,
        negative_variance=negative_variance,
        design_effect=CLUSTER_DESIGN_EFFECT,
    )
    pooled_se = finding_se / 2.0
    current_single = normal_gate_power(
        true_delta, finding_se, require_point_mcid=True
    )
    current_one_model = at_least_three_of_four(current_single)
    pooled_detect_one = normal_gate_power(
        true_delta, pooled_se, require_point_mcid=False
    )
    pooled_full_one = normal_gate_power(
        true_delta, pooled_se, require_point_mcid=True
    )
    direction_single = float(norm.cdf(true_delta / finding_se))
    direction_one_model = at_least_three_of_four(direction_single)
    return {
        "n_per_finding_vote_bin": n_per_bin,
        "claims_per_model": 16 * n_per_bin,
        "clear_claims_per_finding": 2 * n_per_bin,
        "expected_error_events_per_finding": 2
        * n_per_bin
        * error_prevalence,
        "per_finding_delta_auc_se": finding_se,
        "pooled_four_finding_delta_auc_se": pooled_se,
        "current_individual_significance_gate": {
            "single_finding": current_single,
            "one_model_three_of_four": current_one_model,
            "two_models_independent": current_one_model**2,
        },
        "recommended_hierarchical_components": {
            "pooled_detection_one_model": pooled_detect_one,
            "pooled_detection_two_models_independent": pooled_detect_one**2,
            "pooled_full_mcid_gate_one_model": pooled_full_one,
            "pooled_full_mcid_gate_two_models_independent": pooled_full_one**2,
            "three_of_four_positive_direction_one_model": direction_one_model,
            "three_of_four_positive_direction_two_models_independent": direction_one_model**2,
            "warning": "component powers are not multiplied; their events are positively dependent",
        },
    }


def hierarchical_joint_power(
    n_per_bin: int,
    *,
    true_delta: float,
    error_prevalence: float,
    score_correlation: float,
    require_point_mcid: bool,
    standard_normal_draws: np.ndarray,
) -> dict[str, float]:
    positive_variance, negative_variance = paired_auc_variance_constants(
        BASELINE_AUC, true_delta, score_correlation
    )
    se = auc_delta_se(
        n_per_vote_bin=n_per_bin,
        finding_count=1,
        error_prevalence=error_prevalence,
        positive_variance=positive_variance,
        negative_variance=negative_variance,
        design_effect=CLUSTER_DESIGN_EFFECT,
    )
    estimates = true_delta + se * standard_normal_draws
    pooled_se = se / 2.0
    threshold = CI_Z * pooled_se
    if require_point_mcid:
        threshold = max(MCID, threshold)
    one_model_event = (
        (estimates.mean(axis=1) >= threshold)
        & ((estimates > 0).sum(axis=1) >= 3)
        & ((estimates <= -MCID).sum(axis=1) == 0)
    )
    one_model = float(one_model_event.mean())
    return {
        "one_model": one_model,
        "two_models_independent": one_model**2,
        "two_models_perfect_positive_dependence": one_model,
    }


def minimum_n(
    goal: float,
    *,
    true_delta: float,
    error_prevalence: float,
    score_correlation: float,
    require_point_mcid: bool,
    standard_normal_draws: np.ndarray,
    maximum: int = 1000,
) -> dict[str, Any] | None:
    def evaluate(n_per_bin: int) -> dict[str, float]:
        return hierarchical_joint_power(
            n_per_bin,
            true_delta=true_delta,
            error_prevalence=error_prevalence,
            score_correlation=score_correlation,
            require_point_mcid=require_point_mcid,
            standard_normal_draws=standard_normal_draws,
        )

    if evaluate(maximum)["two_models_independent"] < goal:
        return None
    low, high = 1, maximum
    while low < high:
        middle = (low + high) // 2
        if evaluate(middle)["two_models_independent"] >= goal:
            high = middle
        else:
            low = middle + 1
    power = evaluate(low)
    # Monte Carlo power is nearly monotone but finite fixed draws can create a
    # one-count wobble.  Walk back only across the local boundary so the
    # reported integer is the first observed qualifying N.
    while low > 1:
        previous = evaluate(low - 1)
        if previous["two_models_independent"] < goal:
            break
        low -= 1
        power = previous
    return {"n_per_bin": low, "power": power}


def source_split_mismatch(runner_source: str, analyzer_source: str) -> dict[str, Any]:
    pack_alias = (
        '"split": "dev"' in runner_source
        and '"source_manifest_split": "pilot"' in runner_source
    )
    analyzer_dev_only = 'payload.get("split") != "dev"' in analyzer_source
    return {
        "runner_packs_pilot_as_dev": pack_alias,
        "analyzer_requires_dev_alias": analyzer_dev_only,
        "mismatch_present": pack_alias and analyzer_dev_only,
        "decision": (
            "MUST_FIX_BEFORE_NEW_FORMAL_OUTPUT"
            if pack_alias and analyzer_dev_only
            else "FIXED_THREE_STAGE_SOURCE_TRUTHFUL"
        ),
    }


def build_audit(
    manifest: Path, summary_path: Path, runner: Path, analyzer: Path
) -> dict[str, Any]:
    rows = load_jsonl(manifest)
    summary = json.loads(summary_path.read_text())
    stages = [
        selection_audit(rows, "pilot_screen", "pilot", 10),
        selection_audit(rows, "dev_fit", "dev", 20),
        selection_audit(rows, "confirmation_locked", "confirmation", 60),
    ]
    stage_images = {row["stage"]: set(row.pop("image_ids")) for row in stages}
    stage_keys = {row["stage"]: row.pop("selection_keys") for row in stages}
    overlap = {
        f"{left}__{right}": len(stage_images[left] & stage_images[right])
        for index, left in enumerate(stage_images)
        for right in list(stage_images)[index + 1 :]
    }
    if any(overlap.values()):
        raise ValueError(f"whole-image split leakage: {overlap}")

    rng = np.random.default_rng(20260803)
    standard_normal_draws = rng.standard_normal((300_000, 4))
    stage_power = {
        f"delta_{str(delta).replace('.', 'p')}": [
            gate_row(stage["per_finding_vote_bin"], true_delta=delta)
            for stage in stages
        ]
        for delta in (MCID, PLANNING_ALTERNATIVE)
    }
    hierarchical_joint_stage_power = {}
    for delta in (MCID, PLANNING_ALTERNATIVE):
        delta_key = f"delta_{str(delta).replace('.', 'p')}"
        hierarchical_joint_stage_power[delta_key] = []
        for stage in stages:
            n_per_bin = stage["per_finding_vote_bin"]
            hierarchical_joint_stage_power[delta_key].append(
                {
                    "stage": stage["stage"],
                    "n_per_bin": n_per_bin,
                    "detection_only": hierarchical_joint_power(
                        n_per_bin,
                        true_delta=delta,
                        error_prevalence=ERROR_PREVALENCE,
                        score_correlation=PAIRED_SCORE_CORRELATION,
                        require_point_mcid=False,
                        standard_normal_draws=standard_normal_draws,
                    ),
                    "full_gate_including_point_mcid": hierarchical_joint_power(
                        n_per_bin,
                        true_delta=delta,
                        error_prevalence=ERROR_PREVALENCE,
                        score_correlation=PAIRED_SCORE_CORRELATION,
                        require_point_mcid=True,
                        standard_normal_draws=standard_normal_draws,
                    ),
                }
            )
    sample_size = {
        "central": {
            "assumptions": {
                "baseline_auc": BASELINE_AUC,
                "error_prevalence": ERROR_PREVALENCE,
                "paired_score_correlation": PAIRED_SCORE_CORRELATION,
                "cluster_design_effect": CLUSTER_DESIGN_EFFECT,
            },
            "detect_mcid_delta_0p03": {
                str(goal): minimum_n(
                    goal,
                    true_delta=MCID,
                    error_prevalence=ERROR_PREVALENCE,
                    score_correlation=PAIRED_SCORE_CORRELATION,
                    require_point_mcid=False,
                    standard_normal_draws=standard_normal_draws,
                )
                for goal in (0.80, 0.90)
            },
            "full_gate_at_planning_delta_0p05": {
                str(goal): minimum_n(
                    goal,
                    true_delta=PLANNING_ALTERNATIVE,
                    error_prevalence=ERROR_PREVALENCE,
                    score_correlation=PAIRED_SCORE_CORRELATION,
                    require_point_mcid=True,
                    standard_normal_draws=standard_normal_draws,
                )
                for goal in (0.80, 0.90)
            },
        },
        "sensitivity_full_gate_delta_0p05": [],
    }
    for prevalence in (0.10, 0.20, 0.30):
        for correlation in (0.90, 0.95, 0.98, 0.99):
            sample_size["sensitivity_full_gate_delta_0p05"].append(
                {
                    "error_prevalence": prevalence,
                    "paired_score_correlation": correlation,
                    "n80": minimum_n(
                        0.80,
                        true_delta=PLANNING_ALTERNATIVE,
                        error_prevalence=prevalence,
                        score_correlation=correlation,
                        require_point_mcid=True,
                        standard_normal_draws=standard_normal_draws,
                    )["n_per_bin"],
                    "n90": minimum_n(
                        0.90,
                        true_delta=PLANNING_ALTERNATIVE,
                        error_prevalence=prevalence,
                        score_correlation=correlation,
                        require_point_mcid=True,
                        standard_normal_draws=standard_normal_draws,
                    )["n_per_bin"],
                }
            )

    current_cap = {
        "single_finding": 0.5,
        "one_model_three_of_four": at_least_three_of_four(0.5),
        "two_models_independent": at_least_three_of_four(0.5) ** 2,
        "reason": (
            "at true delta=MCID, estimate>=MCID has asymptotic probability 0.5; "
            "therefore no finite sample reaches 80% or 90% for the current gate"
        ),
    }

    # The current measured two-model 160-claim wall time was 45--75 minutes.
    # Scaling is deliberately linear and conservative; outputs are tiny JSON
    # shards and DICOMs already reside on the external dataset mount.
    runtime = []
    for stage in stages:
        scale = stage["claim_count"] / 160.0
        runtime.append(
            {
                "stage": stage["stage"],
                "two_model_wall_minutes": [45.0 * scale, 75.0 * scale],
                "cells_two_models": stage["claim_count"] * 19 * 2,
                "conservative_json_disk_gib": stage["claim_count"]
                * 2
                * 110_000
                / 1024**3,
            }
        )

    result: dict[str, Any] = {
        "version": VERSION,
        "status": "complete_outcome_blind_design_audit",
        "scientific_scope": (
            "prospective design only; no model score, sealed outcome, clinician return, "
            "or clinical-admission verdict was read"
        ),
        "gpu_authorized": False,
        "inputs": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "manifest_summary": str(summary_path),
            "manifest_summary_sha256": sha256_file(summary_path),
            "runner_source": str(runner),
            "runner_source_sha256": sha256_file(runner),
            "analyzer_source": str(analyzer),
            "analyzer_source_sha256": sha256_file(analyzer),
        },
        "manifest_contract": {
            "image_disjoint_declared": bool(
                summary["split_contract"].get("image_disjoint")
            ),
            "quotas_per_finding_vote_bin": summary["split_contract"][
                "quotas_per_finding_vote_bin"
            ],
            "stage_image_overlap_counts": overlap,
            "whole_image_split_verified": not any(overlap.values()),
        },
        "exact_stage_selections": stages,
        "private_selection_key_lists_sha256": {
            stage: canonical_sha256(keys) for stage, keys in stage_keys.items()
        },
        "provenance_mismatch": source_split_mismatch(
            runner.read_text(), analyzer.read_text()
        ),
        "power_model": {
            "method": (
                "paired equal-variance binormal AUROC; DeLong influence variance by "
                "48-point deterministic Gauss-Hermite quadrature; normal CI approximation"
            ),
            "minimum_clinically_meaningful_delta_auc": MCID,
            "planning_alternative_for_full_gate": PLANNING_ALTERNATIVE,
            "baseline_auc": BASELINE_AUC,
            "error_prevalence": ERROR_PREVALENCE,
            "paired_baseline_candidate_score_correlation": PAIRED_SCORE_CORRELATION,
            "image_cluster_design_effect": CLUSTER_DESIGN_EFFECT,
            "ci_z_two_sided_95": CI_Z,
            "four_finding_independence_is_planning_approximation": True,
            "two_model_independence_is_conservative_only_under_nonnegative_dependence": True,
            "oof_model_fitting_variance_not_included": (
                "yes; the calculation is an optimistic oracle-score approximation"
            ),
        },
        "current_gate_asymptotic_power_ceiling_at_mcid": current_cap,
        "stage_power": stage_power,
        "hierarchical_joint_stage_power": hierarchical_joint_stage_power,
        "sample_size": sample_size,
        "recommended_frozen_design": {
            "decision": "REPLACE_CURRENT_FORMAL_STAGE1_BEFORE_ANY_ADMITTED_OUTPUT",
            "pilot_screen": {
                "selection": "pilot split, first deterministic 10/bin, 160 claims/model",
                "role": "engineering canary only",
                "forbidden_decisions": [
                    "formal mechanism NO-GO",
                    "formal mechanism confirmation",
                    "method or hidden-state authorization",
                ],
                "payload_split_label": "pilot_screen",
            },
            "dev_fit": {
                "selection": "dev split, all preselected 20/bin, 320 claims/model",
                "role": "fit transforms/predictor, freeze coefficients and thresholds; estimate only",
                "payload_split_label": "dev_fit",
                "formal_null_decision_allowed": False,
            },
            "confirmation_locked": {
                "selection": "confirmation split, all preselected 60/bin, 960 claims/model",
                "role": "apply dev-frozen predictor once; no refit or threshold selection",
                "payload_split_label": "confirmation_locked",
                "primary_gate_per_model": [
                    "pooled four-finding image-cluster delta-AUROC point >= 0.03",
                    "pooled four-finding image-cluster 95% CI lower > 0",
                    "pooled harmful-interaction error-minus-correct 95% CI lower > 0",
                    "interaction RMS >= 0.25 reader-equivalents with CI lower > 0",
                    "identity render and duplicate prompt RMS each <= 0.1 x clinical interaction RMS",
                    "reader-slope CI lower > 0 for every finding",
                ],
                "heterogeneity_guard_per_model": [
                    "at least 3/4 per-finding delta-AUROC point estimates > 0",
                    "at least 3/4 per-finding harmful-alignment point estimates > 0",
                    "no finding delta-AUROC point estimate <= -0.03",
                    "no finding delta-AUROC bootstrap 95% CI upper < 0",
                ],
                "two_model_rule": "both Huatuo and Hulu pass all pooled and heterogeneity gates",
                "power_interpretation": (
                    "N=60/bin is about 94% for the full two-model hierarchical gate under "
                    "the frozen central planning alternative delta=0.05; at exact MCID=0.03 "
                    "the point>=MCID requirement remains a boundary and no powered-null claim is allowed"
                ),
            },
            "selection_firewall": [
                "three stages use different output directories",
                "three stages bind different selection-key hashes",
                "whole image_id, not image-claim, is the cluster and split unit",
                "no stage may be renamed dev/locked after scoring",
                "confirmation code and thresholds freeze before any confirmation shard exists",
            ],
        },
        "implementation_change_list": [
            {
                "component": "run_cecd_factorial_v1.py",
                "required": (
                    "accept and hash-bind explicit stage_label, manifest_split and per_bin; "
                    "pack pilot_screen/dev_fit/confirmation_locked truthfully; reject output-dir "
                    "reuse or selection-hash mismatch"
                ),
            },
            {
                "component": "analyze_clinical_equivalence_composition_defect_v1.py",
                "required": (
                    "replace one-payload dev OOF formal gate with dev_fit serialization and "
                    "confirmation_locked apply-only mode; make pooled clustered delta-AUROC primary; "
                    "retain per-finding direction/heterogeneity guard without individual significance"
                ),
            },
            {
                "component": "verify_cecd_two_model_stage1_v2.py",
                "required": (
                    "verify all three stage labels, source splits, disjoint image sets, exact claim/row "
                    "counts and distinct selection/output hashes; never treat pilot_screen failure as NO-GO"
                ),
            },
            {
                "component": "monitor_cecd_admission_pipeline.py",
                "required": (
                    "launch 160 canary, 320 dev_fit and 960 locked confirmation into distinct roots; "
                    "stop only on operational invalidity or prespecified dev futility, not low-power pilot AUROC"
                ),
            },
            {
                "component": "monitor_cecd_dual_semantics_transition_v1.py",
                "required": (
                    "consume only the two-model confirmation_locked verdict and its dev-fit provenance; "
                    "reject legacy pilot-as-dev Stage-1 artifacts"
                ),
            },
        ],
        "runtime_and_storage": {
            "stages": runtime,
            "all_three_stages_two_model_wall_hours": [6.75, 11.25],
            "all_three_stages_conservative_json_disk_gib": sum(
                row["conservative_json_disk_gib"] for row in runtime
            ),
            "new_dicom_download_required": False,
            "gpu_launch_permitted_by_this_artifact": False,
        },
    }
    result["fingerprint"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--analyzer", type=Path, default=DEFAULT_ANALYZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_audit(
        args.manifest, args.manifest_summary, args.runner, args.analyzer
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
