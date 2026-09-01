"""Fail-closed contracts for the Treble/CECD closest-work stage.

This module does not implement a scientific Treble reproduction.  The public
paper and repository disagree on several counterfactuals and the public entry
point is not executable as released.  The pure functions below make those
differences explicit, reproduce the unambiguous representation arithmetic and
test-time shift, and prevent a CECD factorial marginal from being relabelled as
a Treble Natural Direct Effect (NDE).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "treble-cecd-collision-contract-v1"
EXTERNAL_OUTCOME_SCHEMA = "cecd-treble-method-collision-v1"
DUAL_SEMANTICS_OUTCOME_SCHEMA = "cecd-treble-dual-semantics-envelope-v1"
DUAL_SEMANTICS_PREFLIGHT_SCHEMA = "cecd-treble-dual-semantics-preflight-v1"
TREBLE_ARXIV = "2503.06169v2"
TREBLE_ANTHOLOGY_ID = "2025.findings-emnlp.1000"
TREBLE_DOI = "10.18653/v1/2025.findings-emnlp.1000"
TREBLE_REPOSITORY = "TREE985/Treble-Counterfactual-VLMs"
TREBLE_REPOSITORY_COMMIT = "f52197e48bd34a54508afbb49da25a26cb74be3f"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

DUAL_SEMANTICS_VARIANTS: dict[str, dict[str, str]] = {
    "treble_proceedings": {
        "vision_delta": "original_minus_random_mask_mean",
        "text_delta": "factual_minus_hallucinated",
        "cross_modal_delta": "black_image_minus_no_image",
        "direction_fit": "rank1_pc1",
        "intervention": "paper_additive_a_b_c_0.9",
    },
    "treble_released": {
        "vision_delta": "gaussian_step500_mean_minus_original",
        "text_delta": "factual_minus_hallucinated",
        "cross_modal_delta": "no_image_minus_gaussian_step200_mean",
        "direction_fit": "mean_plus_rank1_pc_then_layer_reshape",
        "intervention": "source_norm_preserving_inner_step_0.1_lambda_0.9_fp16",
    },
}

DUAL_SEMANTICS_METHODS = (
    "unmitigated",
    "cecd_interaction_projection",
    "treble_proceedings",
    "treble_released",
    "full_orbit",
    "render_only",
    "prompt_only",
    "random_norm",
    "sign_permuted",
    "main_effect_removal",
)

PRIMARY_ENVELOPE_CONTROLS = (
    "treble_proceedings",
    "treble_released",
    "full_orbit",
)

# This is deliberately not a full mitigation-baseline closure.  The two Treble
# variants are static/global activation controls under a common protocol.  The
# ten arms contain no official-compatible query-adaptive or multimodal dynamic
# activation baseline and no representation-level PID control.
METHOD_CLOSURE_LIMITATIONS = (
    "no_official_compatible_dynamic_or_multimodal_activation_baseline",
    "no_representation_level_pid_synergy_control",
    "no_locked_test_behavioral_increment_analysis_in_this_outcome_schema",
    "treble_variants_are_common_protocol_not_exact_paper_native_reproductions",
)

METHOD_METRICS = (
    "ce_overcommitment_rate",
    "ce_clear_accuracy",
    "oe_hallucination_rate",
    "oe_omission_rate",
    "oe_claim_coverage",
    "oe_mean_claims",
    "oe_mean_length",
    "oe_refusal_rate",
    "reader_brier",
)

DUAL_SEMANTICS_THRESHOLDS = {
    "ce_overcommitment_relative_reduction_min": 0.20,
    "ce_clear_accuracy_absolute_loss_max": 0.01,
    "oe_hallucination_relative_reduction_min": 0.20,
    "oe_omission_increase_max": 0.0,
    "oe_coverage_absolute_difference_max": 0.01,
    "oe_claim_count_absolute_difference_max": 0.0,
    "oe_length_relative_difference_max": 0.05,
    "oe_refusal_increase_max": 0.0,
    "reader_brier_relative_improvement_min": 0.05,
    "paired_advantage_ci_lower_min_exclusive": 0.0,
    "minimum_clusters_per_task_model": 30,
}


class TrebleContractError(ValueError):
    """Raised when a purported exact reproduction violates the frozen contract."""


def _same_shape(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape:
        raise TrebleContractError(
            f"{label} operands must have equal shape, got {left.shape} and {right.shape}"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise TrebleContractError(f"{label} operands must be finite")


def paper_representation_deltas(
    *,
    original_vision: np.ndarray,
    corrupted_vision_mean: np.ndarray,
    original_text: np.ndarray,
    hallucinated_text: np.ndarray,
    black_image_text_state: np.ndarray,
    no_image_text_state: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the three representation differences stated in the paper.

    The names describe the paper's equations (5), (7), and (8).  Calling these
    arrays causally identified NDEs is intentionally outside this function's
    contract.
    """

    arrays = {
        "original_vision": np.asarray(original_vision, dtype=np.float64),
        "corrupted_vision_mean": np.asarray(corrupted_vision_mean, dtype=np.float64),
        "original_text": np.asarray(original_text, dtype=np.float64),
        "hallucinated_text": np.asarray(hallucinated_text, dtype=np.float64),
        "black_image_text_state": np.asarray(black_image_text_state, dtype=np.float64),
        "no_image_text_state": np.asarray(no_image_text_state, dtype=np.float64),
    }
    _same_shape(arrays["original_vision"], arrays["corrupted_vision_mean"], "vision")
    _same_shape(arrays["original_text"], arrays["hallucinated_text"], "text")
    _same_shape(
        arrays["black_image_text_state"], arrays["no_image_text_state"], "cross-modal"
    )
    return {
        "vision": arrays["original_vision"] - arrays["corrupted_vision_mean"],
        "text": arrays["original_text"] - arrays["hallucinated_text"],
        "cross_modal": arrays["black_image_text_state"] - arrays["no_image_text_state"],
    }


def released_code_representation_deltas(
    *,
    original_vision: np.ndarray,
    gaussian_step500_vision_mean: np.ndarray,
    factual_caption_text_state: np.ndarray,
    hallucinated_caption_text_state: np.ndarray,
    no_image_text_state: np.ndarray,
    gaussian_step200_image_text_state_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the differences computed by the released source, before PCA.

    These are kept separate from :func:`paper_representation_deltas` because
    the released vision direction has the opposite order to proceedings
    equation (5), and its cross-modal branch compares no-image against Gaussian
    degradation rather than black-image against no-image as in equation (8).
    The released text direction agrees with proceedings equation (7).
    """

    arrays = {
        "original_vision": np.asarray(original_vision, dtype=np.float64),
        "gaussian_step500_vision_mean": np.asarray(
            gaussian_step500_vision_mean, dtype=np.float64
        ),
        "factual_caption_text_state": np.asarray(
            factual_caption_text_state, dtype=np.float64
        ),
        "hallucinated_caption_text_state": np.asarray(
            hallucinated_caption_text_state, dtype=np.float64
        ),
        "no_image_text_state": np.asarray(no_image_text_state, dtype=np.float64),
        "gaussian_step200_image_text_state_mean": np.asarray(
            gaussian_step200_image_text_state_mean, dtype=np.float64
        ),
    }
    _same_shape(
        arrays["original_vision"], arrays["gaussian_step500_vision_mean"], "vision"
    )
    _same_shape(
        arrays["factual_caption_text_state"],
        arrays["hallucinated_caption_text_state"],
        "text",
    )
    _same_shape(
        arrays["no_image_text_state"],
        arrays["gaussian_step200_image_text_state_mean"],
        "cross-modal",
    )
    return {
        "vision": arrays["gaussian_step500_vision_mean"] - arrays["original_vision"],
        "text": arrays["factual_caption_text_state"]
        - arrays["hallucinated_caption_text_state"],
        "cross_modal": arrays["no_image_text_state"]
        - arrays["gaussian_step200_image_text_state_mean"],
    }


def _unit_last_axis(values: np.ndarray, label: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise TrebleContractError(f"{label} contains a zero or non-finite vector")
    return values / norms


def released_code_norm_preserving_shift(
    activations: np.ndarray,
    directions: Sequence[np.ndarray],
    lambdas: Sequence[float],
    *,
    inner_step: float = 0.1,
) -> np.ndarray:
    """Pure NumPy form of the released ``Shift_Layer`` arithmetic.

    For activation vectors ``x`` and directions ``d_j``, the source computes

    ``||x|| * normalize(normalize(x) + 0.1 * mean_j(lambda_j normalize(d_j)))``.

    The source later casts to FP16.  This function deliberately retains FP64
    so unit tests can isolate the mathematical contract from dtype effects.
    """

    x = np.asarray(activations, dtype=np.float64)
    if x.ndim < 1 or not np.isfinite(x).all():
        raise TrebleContractError("activations must be a finite array with a hidden axis")
    if not directions or len(directions) != len(lambdas):
        raise TrebleContractError("directions and lambdas must have equal non-zero length")
    if not np.isfinite(float(inner_step)) or inner_step < 0:
        raise TrebleContractError("inner_step must be finite and non-negative")
    original_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(original_norm <= 0):
        raise TrebleContractError("activations contain a zero vector")
    accumulated = np.zeros_like(x, dtype=np.float64)
    for index, (direction, weight) in enumerate(zip(directions, lambdas)):
        if not np.isfinite(float(weight)):
            raise TrebleContractError(f"lambda {index} is non-finite")
        d = np.asarray(direction, dtype=np.float64)
        try:
            d = np.broadcast_to(d, x.shape)
        except ValueError as error:
            raise TrebleContractError(
                f"direction {index} with shape {d.shape} cannot broadcast to {x.shape}"
            ) from error
        accumulated += float(weight) * _unit_last_axis(d, f"direction {index}")
    accumulated /= len(directions)
    shifted_unit = _unit_last_axis(
        _unit_last_axis(x, "activations") + float(inner_step) * accumulated,
        "shifted activations",
    )
    return shifted_unit * original_norm


@dataclass(frozen=True)
class TrebleComputeLedger:
    demonstrations: int
    perturbation_trials: int
    vision_counterfactual_encoder_forwards: int
    vision_original_encoder_forwards: int
    text_pair_multimodal_forwards: int
    cross_degraded_multimodal_forwards: int
    cross_no_image_language_forwards: int
    total_image_bearing_forwards: int
    target_generation_forwards_per_example: int


def released_code_compute_ledger(
    demonstrations: int = 50, perturbation_trials: int = 50
) -> TrebleComputeLedger:
    """Count released-path forward calls without collapsing unlike resources."""

    n, m = int(demonstrations), int(perturbation_trials)
    if n <= 0 or m <= 0:
        raise TrebleContractError("demonstrations and perturbation_trials must be positive")
    vision_cf = n * m
    vision_original = n
    text_pair = 2 * n
    cross_degraded = n * m
    cross_no_image = n
    return TrebleComputeLedger(
        demonstrations=n,
        perturbation_trials=m,
        vision_counterfactual_encoder_forwards=vision_cf,
        vision_original_encoder_forwards=vision_original,
        text_pair_multimodal_forwards=text_pair,
        cross_degraded_multimodal_forwards=cross_degraded,
        cross_no_image_language_forwards=cross_no_image,
        total_image_bearing_forwards=vision_cf
        + vision_original
        + text_pair
        + cross_degraded,
        target_generation_forwards_per_example=1,
    )


def proceedings_compute_ledger(
    demonstrations: int = 50, perturbation_trials: int = 50
) -> dict[str, int]:
    """Count the proceedings-stated calibration path without code-path inflation."""

    n, m = int(demonstrations), int(perturbation_trials)
    if n <= 0 or m <= 0:
        raise TrebleContractError("demonstrations and perturbation_trials must be positive")
    vision_cf = n * m
    vision_original = n
    text_pair = 2 * n
    cross_black = n
    cross_no_image = n
    return {
        "demonstrations": n,
        "perturbation_trials": m,
        "vision_counterfactual_encoder_forwards": vision_cf,
        "vision_original_encoder_forwards": vision_original,
        "text_pair_multimodal_forwards": text_pair,
        "cross_black_image_multimodal_forwards": cross_black,
        "cross_no_image_language_forwards": cross_no_image,
        "total_image_bearing_forwards": vision_cf
        + vision_original
        + text_pair
        + cross_black,
        "target_generation_forwards_per_example": 1,
    }


RELEASE_BLOCKERS = (
    "repository_has_no_explicit_root_license",
    "case_sensitive_module_filename_disagrees_with_import",
    "llava_parser_defines_sample_num_but_sampler_reads_num_demos",
    "entrypoint_reads_rankk_but_parser_defines_only_rank",
    "cross_modal_sampler_calls_add_gaussian_noise_with_an_unsupported_mask_argument",
    "proceedings_vision_delta_order_disagrees_with_released_source",
    "paper_black_vs_no_image_cross_modal_pair_disagrees_with_released_source",
    "paper_random_mask_description_disagrees_with_released_gaussian_degradation",
    "released_method_has_no_per_claim_scalar_nde_output",
)


def source_audit() -> dict[str, Any]:
    """Return the frozen public-source audit used by the external gate."""

    return {
        "schema_version": SCHEMA_VERSION,
        "paper": TREBLE_ANTHOLOGY_ID,
        "doi": TREBLE_DOI,
        "preprint": TREBLE_ARXIV,
        "repository": TREBLE_REPOSITORY,
        "source_repo_commit": TREBLE_REPOSITORY_COMMIT,
        "reproduction_fidelity": "blocked_unresolved_paper_code_semantics",
        "reproduction_authorized": False,
        "blockers": list(RELEASE_BLOCKERS),
        "forbidden_substitutions": [
            "CECD render main effect as vision NDE",
            "CECD prompt main effect as text NDE",
            "CECD two-way centered interaction as cross-modal NDE",
            "any post-hoc per-cell scalar named exact Treble NDE",
        ],
        "compute_ledger_defaults": {
            "treble_proceedings": proceedings_compute_ledger(),
            "treble_released": asdict(released_code_compute_ledger()),
        },
    }


def validate_external_method_outcome(payload: Mapping[str, Any]) -> None:
    """Validate and permanently fail closed the obsolete exact-v1 contract.

    The v1 schema has no independently hashed author/source adjudication field,
    so the string ``paper_and_code_semantics_resolved`` is only a self-claim.
    Contradictory primary sources cannot be reconciled by that string.  Retain
    parsing for old artifacts, but never let this function authorize one; use
    the dual-semantics common-protocol envelope instead.
    """

    required = {
        "schema_version",
        "source_repo_commit",
        "reproduction_fidelity",
        "model_fingerprint",
        "calibration_split",
        "evaluation_split",
        "record_keys_sha256",
        "compute_ledger",
        "paired_method_metrics",
        "paired_cluster_bootstrap",
        "collision_verdict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise TrebleContractError(f"external method outcome missing fields: {missing}")
    if payload["schema_version"] != EXTERNAL_OUTCOME_SCHEMA:
        raise TrebleContractError("external method outcome schema mismatch")
    if payload["source_repo_commit"] != TREBLE_REPOSITORY_COMMIT:
        raise TrebleContractError("Treble source commit is not the audited commit")
    if payload["reproduction_fidelity"] != "paper_and_code_semantics_resolved":
        raise TrebleContractError("paper/code counterfactual semantics remain unresolved")
    if payload["calibration_split"] != "dev" or payload["evaluation_split"] != "locked_test":
        raise TrebleContractError("Treble directions must use dev and evaluate on locked_test")
    if len(str(payload["record_keys_sha256"])) != 64:
        raise TrebleContractError("record_keys_sha256 must be a SHA-256 digest")
    if payload["collision_verdict"] not in {"cecd_survives", "direct_collision"}:
        raise TrebleContractError("collision_verdict must be preregistered and decisive")
    if not isinstance(payload["paired_method_metrics"], Mapping) or not isinstance(
        payload["paired_cluster_bootstrap"], Mapping
    ):
        raise TrebleContractError("paired metrics and image-cluster bootstrap are mandatory")
    if not payload["paired_method_metrics"] or not payload["paired_cluster_bootstrap"]:
        raise TrebleContractError("paired metrics and image-cluster bootstrap cannot be empty")
    if not isinstance(payload["compute_ledger"], Mapping) or not payload["compute_ledger"]:
        raise TrebleContractError("a non-empty heterogeneous compute ledger is mandatory")
    raise TrebleContractError(
        "exact-v1 self-attestation cannot resolve contradictory paper/code semantics; "
        "use the frozen dual-semantics envelope"
    )


def _require_hex64(value: Any, label: str) -> str:
    text = str(value)
    if HEX64.fullmatch(text) is None:
        raise TrebleContractError(f"{label} must be lowercase SHA-256 hex")
    return text


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TrebleContractError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TrebleContractError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise TrebleContractError(f"{label} must be a finite number")
    return number


def _rate(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if not 0.0 <= number <= 1.0:
        raise TrebleContractError(f"{label} must lie in [0, 1]")
    return number


def _validate_model_fingerprints(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"huatuo", "hulu"}:
        raise TrebleContractError("model_fingerprints must bind exactly Huatuo and Hulu")
    digest_fields = {
        "checkpoint_sha256",
        "processor_sha256",
        "template_sha256",
        "generation_contract_sha256",
        "hook_contract_sha256",
        "vision_token_transport_contract_sha256",
    }
    for family, record in value.items():
        if not isinstance(record, Mapping):
            raise TrebleContractError(f"{family} model fingerprint must be an object")
        if set(record) != {"model_id", *digest_fields}:
            raise TrebleContractError(f"{family} model fingerprint fields are not frozen")
        if not isinstance(record["model_id"], str) or not record["model_id"].strip():
            raise TrebleContractError(f"{family} model_id must be nonempty")
        for field in digest_fields:
            _require_hex64(record[field], f"{family}.{field}")
    model_ids = {str(record["model_id"]) for record in value.values()}
    if len(model_ids) != 2:
        raise TrebleContractError("Huatuo and Hulu must bind distinct model identities")


def _validate_compute_ledgers(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"huatuo", "hulu"}:
        raise TrebleContractError("compute_ledger must bind exactly Huatuo and Hulu")
    expected_calibrations = {
        "treble_proceedings": proceedings_compute_ledger(),
        "treble_released": asdict(released_code_compute_ledger()),
    }
    for family, ledger in value.items():
        if not isinstance(ledger, Mapping):
            raise TrebleContractError(f"{family} compute ledger must be an object")
        required = {
            "treble_proceedings",
            "treble_released",
            "target_examples",
            "cecd_target_generation_forwards",
            "full_orbit_target_generation_forwards",
        }
        if set(ledger) != required:
            raise TrebleContractError(f"{family} compute ledger fields are not frozen")
        for variant in DUAL_SEMANTICS_VARIANTS:
            if ledger[variant] != expected_calibrations[variant]:
                raise TrebleContractError(
                    f"{family}.{variant} must preserve its heterogeneous N=50,m=50 ledger"
                )
        target_examples = ledger["target_examples"]
        if (
            isinstance(target_examples, bool)
            or not isinstance(target_examples, int)
            or target_examples <= 0
        ):
            raise TrebleContractError(f"{family}.target_examples must be a positive integer")
        if ledger["cecd_target_generation_forwards"] != 4 * target_examples:
            raise TrebleContractError(f"{family} CECD online-call ledger is inconsistent")
        if ledger["full_orbit_target_generation_forwards"] != 4 * target_examples:
            raise TrebleContractError(
                f"{family} full-orbit online-call ledger is inconsistent"
            )


def _validate_method_metrics(value: Any) -> dict[str, dict[str, dict[str, float]]]:
    if not isinstance(value, Mapping) or set(value) != {"huatuo", "hulu"}:
        raise TrebleContractError("paired_method_metrics must bind Huatuo and Hulu")
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for family, record in value.items():
        if not isinstance(record, Mapping) or set(record) != {"n_clusters", "methods"}:
            raise TrebleContractError(f"{family} paired metrics fields are not frozen")
        clusters = record["n_clusters"]
        if (
            not isinstance(clusters, Mapping)
            or set(clusters) != {"ce", "oe"}
            or any(
                isinstance(number, bool) or not isinstance(number, int) or number < 30
                for number in clusters.values()
            )
        ):
            raise TrebleContractError(f"{family} requires at least 30 CE and OE clusters")
        methods = record["methods"]
        if not isinstance(methods, Mapping) or set(methods) != set(DUAL_SEMANTICS_METHODS):
            raise TrebleContractError(f"{family} method closure is incomplete")
        normalized[family] = {}
        for method, metrics in methods.items():
            if not isinstance(metrics, Mapping) or set(metrics) != set(METHOD_METRICS):
                raise TrebleContractError(f"{family}.{method} metric closure is incomplete")
            row: dict[str, float] = {}
            for metric in METHOD_METRICS:
                label = f"{family}.{method}.{metric}"
                if metric in {
                    "oe_mean_claims",
                    "oe_mean_length",
                }:
                    row[metric] = _finite_number(metrics[metric], label)
                    if row[metric] <= 0:
                        raise TrebleContractError(f"{label} must be positive")
                else:
                    row[metric] = _rate(metrics[metric], label)
            normalized[family][method] = row
    return normalized


def _validate_bootstrap(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"huatuo", "hulu"}:
        raise TrebleContractError("paired_cluster_bootstrap must bind Huatuo and Hulu")
    metrics = {
        "ce_overcommitment_control_minus_cecd",
        "oe_hallucination_control_minus_cecd",
        "reader_brier_control_minus_cecd",
    }
    for family, comparisons in value.items():
        expected = {f"cecd_vs_{control}" for control in PRIMARY_ENVELOPE_CONTROLS}
        if not isinstance(comparisons, Mapping) or set(comparisons) != expected:
            raise TrebleContractError(f"{family} bootstrap comparison closure is incomplete")
        for comparison, estimates in comparisons.items():
            if not isinstance(estimates, Mapping) or set(estimates) != metrics:
                raise TrebleContractError(f"{family}.{comparison} bootstrap metrics are incomplete")
            for metric, estimate in estimates.items():
                if not isinstance(estimate, Mapping) or set(estimate) != {
                    "point",
                    "ci_lower",
                    "ci_upper",
                    "replicates",
                    "unit",
                }:
                    raise TrebleContractError(
                        f"{family}.{comparison}.{metric} bootstrap contract is incomplete"
                    )
                point = _finite_number(estimate["point"], f"{comparison}.{metric}.point")
                lower = _finite_number(
                    estimate["ci_lower"], f"{comparison}.{metric}.ci_lower"
                )
                upper = _finite_number(
                    estimate["ci_upper"], f"{comparison}.{metric}.ci_upper"
                )
                if not lower <= point <= upper:
                    raise TrebleContractError("bootstrap point must lie inside its interval")
                if estimate["replicates"] != 10_000 or estimate["unit"] != "cluster_id":
                    raise TrebleContractError(
                        "paired comparisons require 10,000 patient/image-cluster replicates"
                    )


def _model_survives_envelope(
    methods: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    baseline = methods["unmitigated"]
    cecd = methods["cecd_interaction_projection"]
    failures: list[str] = []
    if baseline["ce_overcommitment_rate"] <= 0 or not (
        cecd["ce_overcommitment_rate"] <= 0.80 * baseline["ce_overcommitment_rate"]
    ):
        failures.append("ce_relative_reduction_below_20_percent")
    if baseline["oe_hallucination_rate"] <= 0 or not (
        cecd["oe_hallucination_rate"] <= 0.80 * baseline["oe_hallucination_rate"]
    ):
        failures.append("oe_relative_reduction_below_20_percent")
    if baseline["reader_brier"] <= 0 or not (
        cecd["reader_brier"] <= 0.95 * baseline["reader_brier"]
    ):
        failures.append("reader_brier_relative_improvement_below_5_percent")
    if cecd["ce_clear_accuracy"] < baseline["ce_clear_accuracy"] - 0.01:
        failures.append("clear_accuracy_loss_above_1pp")
    if cecd["oe_omission_rate"] > baseline["oe_omission_rate"] + 1e-12:
        failures.append("omission_increased")
    if cecd["oe_refusal_rate"] > baseline["oe_refusal_rate"] + 1e-12:
        failures.append("refusal_increased")
    for method, row in methods.items():
        if abs(row["oe_mean_claims"] - baseline["oe_mean_claims"]) > 1e-9:
            failures.append(f"{method}_claim_count_not_fixed")
        if abs(row["oe_claim_coverage"] - baseline["oe_claim_coverage"]) > 0.01:
            failures.append(f"{method}_coverage_not_matched")
        length_ratio = row["oe_mean_length"] / baseline["oe_mean_length"]
        if not 0.95 <= length_ratio <= 1.05:
            failures.append(f"{method}_length_not_matched")
    for control in ("treble_proceedings", "treble_released"):
        if cecd["reader_brier"] > methods[control]["reader_brier"] + 1e-12:
            failures.append(f"reader_brier_worse_than_{control}")
    for control in PRIMARY_ENVELOPE_CONTROLS:
        comparison = bootstrap[f"cecd_vs_{control}"]
        for metric in (
            "ce_overcommitment_control_minus_cecd",
            "oe_hallucination_control_minus_cecd",
        ):
            if comparison[metric]["ci_lower"] <= 0:
                failures.append(f"{control}_{metric}_ci_not_strictly_positive")
        if comparison["reader_brier_control_minus_cecd"]["ci_lower"] < 0:
            failures.append(f"{control}_reader_brier_ci_below_zero")
    return not failures, failures


def validate_dual_semantics_preflight_contract(payload: Mapping[str, Any]) -> None:
    """Validate an outcome-blind controlled-comparison plan.

    This function validates the frozen scientific contract only.  It never
    checks CECD Stage 1 and therefore never authorizes model or GPU execution;
    the runtime binder must independently verify and hash-bind the completed
    two-model Stage-1 artifact before emitting an authorization.
    """

    required = {
        "schema_version",
        "frozen_before_method_outputs",
        "source_repo_commit",
        "reproduction_fidelity",
        "paper_native_claimed",
        "exact_reproduction_claimed",
        "implementation_origin",
        "redistribution_policy",
        "variants",
        "model_fingerprints",
        "stage1_analysis_sha256",
        "stage1_input_gate_sha256",
        "admission_sha256",
        "calibration_split",
        "evaluation_split",
        "calibration_manifest_sha256",
        "evaluation_manifest_sha256",
        "record_keys_sha256",
        "claim_contract_sha256",
        "methods",
        "primary_envelope_controls",
        "method_metrics",
        "thresholds",
        "bootstrap_replicates",
        "bootstrap_unit",
        "compute_ledger",
        "method_output_root",
    }
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing or extra:
        raise TrebleContractError(
            f"dual-semantics preflight fields missing={missing} extra={extra}"
        )
    if payload["schema_version"] != DUAL_SEMANTICS_PREFLIGHT_SCHEMA:
        raise TrebleContractError("dual-semantics preflight schema mismatch")
    if payload["frozen_before_method_outputs"] is not True:
        raise TrebleContractError("method comparison must be frozen before any output")
    if payload["source_repo_commit"] != TREBLE_REPOSITORY_COMMIT:
        raise TrebleContractError("Treble source commit is not the audited commit")
    if payload["reproduction_fidelity"] != "dual_semantics_common_protocol_envelope":
        raise TrebleContractError("preflight must preserve the dual-semantics envelope")
    if payload["paper_native_claimed"] is not False or payload["exact_reproduction_claimed"] is not False:
        raise TrebleContractError("common-protocol preflight cannot claim exact paper-native Treble")
    if (
        payload["implementation_origin"]
        != "independent_clean_room_from_public_equations_and_audited_arithmetic"
        or payload["redistribution_policy"]
        != "local_evaluation_only_no_official_source_or_demo_redistribution"
    ):
        raise TrebleContractError("preflight implementation/license boundary is not frozen")
    if payload["variants"] != DUAL_SEMANTICS_VARIANTS:
        raise TrebleContractError("preflight source semantics are not exact")
    _validate_model_fingerprints(payload["model_fingerprints"])
    for field in (
        "stage1_analysis_sha256",
        "stage1_input_gate_sha256",
        "admission_sha256",
        "calibration_manifest_sha256",
        "evaluation_manifest_sha256",
        "record_keys_sha256",
        "claim_contract_sha256",
    ):
        _require_hex64(payload[field], field)
    if payload["calibration_split"] != "dev" or payload["evaluation_split"] != "locked_test":
        raise TrebleContractError("preflight must fit on dev and evaluate on locked_test")
    if payload["methods"] != list(DUAL_SEMANTICS_METHODS):
        raise TrebleContractError("preflight method order/closure is not frozen")
    if payload["primary_envelope_controls"] != list(PRIMARY_ENVELOPE_CONTROLS):
        raise TrebleContractError("preflight primary control envelope is incomplete")
    if payload["method_metrics"] != list(METHOD_METRICS):
        raise TrebleContractError("preflight clinical metric closure is incomplete")
    if payload["thresholds"] != DUAL_SEMANTICS_THRESHOLDS:
        raise TrebleContractError("preflight no-exchange or efficacy thresholds drifted")
    if payload["bootstrap_replicates"] != 10_000 or payload["bootstrap_unit"] != "cluster_id":
        raise TrebleContractError("preflight bootstrap must use 10,000 cluster replicates")
    _validate_compute_ledgers(payload["compute_ledger"])
    output_root = payload["method_output_root"]
    if not isinstance(output_root, str) or not output_root.startswith("/"):
        raise TrebleContractError("method_output_root must be a frozen absolute path")


def validate_dual_semantics_envelope_outcome(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the only non-fabricated fallback for contradictory Treble sources.

    This common-protocol envelope is deliberately not paper-native Treble.  It
    requires separately frozen proceedings-faithful and released-source-faithful
    variants and evaluates CECD against the stronger observed variant in both
    Huatuo and Hulu.  It is a post-run collision decision, never a pre-run GPU
    authorization.
    """

    required = {
        "schema_version",
        "source_repo_commit",
        "reproduction_fidelity",
        "paper_native_claimed",
        "exact_reproduction_claimed",
        "implementation_origin",
        "redistribution_policy",
        "variants",
        "model_fingerprints",
        "calibration_split",
        "evaluation_split",
        "calibration_manifest_sha256",
        "evaluation_manifest_sha256",
        "record_keys_sha256",
        "claim_contract_sha256",
        "preflight_sha256",
        "compute_ledger",
        "paired_method_metrics",
        "paired_cluster_bootstrap",
        "collision_verdict",
    }
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing or extra:
        raise TrebleContractError(
            f"dual-semantics outcome fields missing={missing} extra={extra}"
        )
    if payload["schema_version"] != DUAL_SEMANTICS_OUTCOME_SCHEMA:
        raise TrebleContractError("dual-semantics outcome schema mismatch")
    if payload["source_repo_commit"] != TREBLE_REPOSITORY_COMMIT:
        raise TrebleContractError("Treble source commit is not the audited commit")
    if payload["reproduction_fidelity"] != "dual_semantics_common_protocol_envelope":
        raise TrebleContractError("contradictory sources require the frozen dual-semantics envelope")
    if payload["paper_native_claimed"] is not False or payload["exact_reproduction_claimed"] is not False:
        raise TrebleContractError("the dual-semantics envelope cannot claim paper-native exactness")
    if (
        payload["implementation_origin"]
        != "independent_clean_room_from_public_equations_and_audited_arithmetic"
        or payload["redistribution_policy"]
        != "local_evaluation_only_no_official_source_or_demo_redistribution"
    ):
        raise TrebleContractError("implementation/license boundary is not frozen")
    if payload["variants"] != DUAL_SEMANTICS_VARIANTS:
        raise TrebleContractError("proceedings and released-source semantics are not exact")
    if payload["calibration_split"] != "dev" or payload["evaluation_split"] != "locked_test":
        raise TrebleContractError("directions must use dev and evaluate on locked_test")
    for field in (
        "calibration_manifest_sha256",
        "evaluation_manifest_sha256",
        "record_keys_sha256",
        "claim_contract_sha256",
        "preflight_sha256",
    ):
        _require_hex64(payload[field], field)
    _validate_model_fingerprints(payload["model_fingerprints"])
    _validate_compute_ledgers(payload["compute_ledger"])
    metrics = _validate_method_metrics(payload["paired_method_metrics"])
    _validate_bootstrap(payload["paired_cluster_bootstrap"])
    failures: dict[str, list[str]] = {}
    for family in ("huatuo", "hulu"):
        survives, reasons = _model_survives_envelope(
            metrics[family], payload["paired_cluster_bootstrap"][family]
        )
        if not survives:
            failures[family] = reasons
    computed = (
        "cecd_survives_dual_semantics_envelope"
        if not failures
        else "collision_or_no_specific_advantage"
    )
    if payload["collision_verdict"] != computed:
        raise TrebleContractError(
            f"declared collision verdict {payload['collision_verdict']!r} disagrees with {computed!r}"
        )
    return {
        "schema_version": DUAL_SEMANTICS_OUTCOME_SCHEMA,
        "valid": True,
        "computed_collision_verdict": computed,
        "family_failures": failures,
        "paper_native_treble_reproduced": False,
        "exact_treble_reproduced": False,
        "cecd_treble_envelope_advantage_established": computed
        == "cecd_survives_dual_semantics_envelope",
        "cecd_causal_claim_authorized": False,
        "full_method_gate_authorized": False,
        "oral_baseline_closure_authorized": False,
        "static_activation_control_status": (
            "dual Treble common-protocol variants present; neither is exact paper-native"
        ),
        "method_closure_limitations": list(METHOD_CLOSURE_LIMITATIONS),
        "paper_claim_authorized": False,
    }
