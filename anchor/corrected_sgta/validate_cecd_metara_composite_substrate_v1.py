#!/usr/bin/env python3
"""Validate a synthetic MetaRA-style CE composite-control substrate.

This module deliberately has no file-reading or command-line entry point.  It
accepts an in-memory, explicitly synthetic payload and constructs only
reader-label-free behavioral features.  It cannot authorize CECD, inspect real
results, use a GPU, establish PAEL absorption, or provide standalone mechanism
evidence.  The interaction features below are collision/readout covariates only.

The unit of validation is a complete 2 x 2 metamorphic orbit for one image,
finding, image transformation, and question transformation:

    clean, image-only, question-only, joint.

Every unit must be paired exactly across all declared models.  Development and
confirmation must use disjoint image and question transformation *families*,
not merely different transform IDs from the same family.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "cecd-metara-style-ce-composite-synthetic-substrate-v1"
SCIENTIFIC_STATUS = "synthetic_only_non_authorizing_collision_control_substrate"
TRANSFORM_REGISTRY_SCHEMA_VERSION = "cecd-transform-implementation-registry-v1"
ADMISSION_SCHEMA_VERSION = "cecd-pathology-proposition-preservation-admission-v1"
REAL_ADAPTER_SCHEMA_VERSION = "cecd-metara-real-adapter-record-v1"
ANSWER_SPACE = "tristate_support_refute_undetermined"
STAGES = ("dev_fit", "confirmation_locked")
CELL_ROLES = ("clean", "image_only", "question_only", "joint")
LOGIT_KEYS = ("support", "refute", "undetermined")
TRANSFORM_AXES = ("image", "question")
MAX_ABSOLUTE_LOGIT = 1_000_000.0
HEX = frozenset("0123456789abcdef")
FORBIDDEN_READER_KEYS = frozenset(
    {
        "reader_vote",
        "reader_votes",
        "reader_label",
        "reader_labels",
        "target",
        "targets",
        "label",
        "labels",
        "gold",
        "ground_truth",
        "correct",
        "correctness",
    }
)

# This is a schema handoff, not an executable adapter.  In particular it has no
# paths, callbacks, imports, or permission to read real data.  A later adapter
# must emit these fields into an in-memory payload that is independently bound
# before this validator may be reused.
REAL_ADAPTER_INPUT_SCHEMA = {
    "schema_version": REAL_ADAPTER_SCHEMA_VERSION,
    "execution_enabled": False,
    "required_payload_bindings": [
        "dataset_manifest_sha256",
        "patient_image_split_manifest_sha256",
        "finding_stage_quotas",
        "transform_registry",
        "transform_registry_sha256",
    ],
    "required_record_fields": [
        "stage",
        "unit_id",
        "model",
        "model_checkpoint_sha256",
        "tokenizer_sha256",
        "score_extractor_sha256",
        "patient_id",
        "image_id",
        "source_record_sha256",
        "finding",
        "image_transform_registry_entry_sha256",
        "question_transform_registry_entry_sha256",
        "admission_artifact",
        "cells",
    ],
    "required_cell_fields": [
        "role",
        "image_sha256",
        "prompt_text",
        "prompt_sha256",
        "tristate_logits",
    ],
    "prohibited_capabilities": [
        "filesystem_read",
        "model_inference",
        "gpu_execution",
        "authorization",
    ],
}


class SubstrateError(RuntimeError):
    """The synthetic collision-control contract was violated."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SubstrateError(f"payload is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_clone(value: Any) -> Any:
    return json.loads(_json_bytes(value).decode("utf-8"))


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SubstrateError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], context: str
) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise SubstrateError(
            f"{context} keys differ: missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubstrateError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value: Any, context: str) -> str:
    digest = _require_nonempty_string(value, context)
    if len(digest) != 64 or any(char not in HEX for char in digest):
        raise SubstrateError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _reject_reader_information(value: Any, context: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_READER_KEYS:
                raise SubstrateError(
                    f"{context} contains forbidden outcome key {key!r}"
                )
            _reject_reader_information(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_reader_information(nested, f"{context}[{index}]")


def _validate_transform_registry(
    raw: Any,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    context = "transform_registry"
    registry = _require_mapping(raw, context)
    _require_exact_keys(
        registry, ("schema_version", "entries", "registry_sha256"), context
    )
    if registry["schema_version"] != TRANSFORM_REGISTRY_SCHEMA_VERSION:
        raise SubstrateError("transform_registry.schema_version mismatch")
    entries_raw = registry["entries"]
    if not isinstance(entries_raw, list) or not entries_raw:
        raise SubstrateError("transform_registry.entries must be a non-empty list")
    entries: list[dict[str, Any]] = []
    by_axis_family: dict[tuple[str, str], dict[str, Any]] = {}
    implementation_to_family: dict[tuple[str, str], str] = {}
    for index, raw_entry in enumerate(entries_raw):
        entry_context = f"{context}.entries[{index}]"
        entry = _require_mapping(raw_entry, entry_context)
        _require_exact_keys(
            entry,
            (
                "axis",
                "family",
                "implementation_name",
                "implementation_version",
                "implementation_sha256",
                "entry_sha256",
            ),
            entry_context,
        )
        canonical = {
            "axis": _require_nonempty_string(entry["axis"], f"{entry_context}.axis"),
            "family": _require_nonempty_string(
                entry["family"], f"{entry_context}.family"
            ),
            "implementation_name": _require_nonempty_string(
                entry["implementation_name"],
                f"{entry_context}.implementation_name",
            ),
            "implementation_version": _require_nonempty_string(
                entry["implementation_version"],
                f"{entry_context}.implementation_version",
            ),
            "implementation_sha256": _require_sha256(
                entry["implementation_sha256"],
                f"{entry_context}.implementation_sha256",
            ),
        }
        if canonical["axis"] not in TRANSFORM_AXES:
            raise SubstrateError(f"{entry_context}.axis is not image or question")
        claimed_entry_sha = _require_sha256(
            entry["entry_sha256"], f"{entry_context}.entry_sha256"
        )
        if object_sha256(canonical) != claimed_entry_sha:
            raise SubstrateError(f"{entry_context}.entry_sha256 mismatch")
        key = (canonical["axis"], canonical["family"])
        if key in by_axis_family:
            raise SubstrateError(f"duplicate transform registry family {key!r}")
        implementation_key = (
            canonical["axis"],
            canonical["implementation_sha256"],
        )
        previous_family = implementation_to_family.setdefault(
            implementation_key, canonical["family"]
        )
        if previous_family != canonical["family"]:
            raise SubstrateError(
                "transform implementation SHA is renamed across families: "
                f"{previous_family!r} vs {canonical['family']!r}"
            )
        normalized = {**canonical, "entry_sha256": claimed_entry_sha}
        by_axis_family[key] = normalized
        entries.append(normalized)
    entries.sort(key=lambda entry: (entry["axis"], entry["family"]))
    canonical_registry = {
        "schema_version": TRANSFORM_REGISTRY_SCHEMA_VERSION,
        "entries": entries,
    }
    claimed_registry_sha = _require_sha256(
        registry["registry_sha256"], "transform_registry.registry_sha256"
    )
    if object_sha256(canonical_registry) != claimed_registry_sha:
        raise SubstrateError("transform_registry.registry_sha256 mismatch")
    return (
        {**canonical_registry, "registry_sha256": claimed_registry_sha},
        by_axis_family,
    )


def _validate_transform(
    raw: Any,
    context: str,
    axis: str,
    declared_families: set[str],
    transform_registry: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    transform = _require_mapping(raw, context)
    _require_exact_keys(
        transform,
        (
            "id",
            "family",
            "implementation_sha256",
            "registry_entry_sha256",
            "parameters",
        ),
        context,
    )
    transform_id = _require_nonempty_string(transform["id"], f"{context}.id")
    family = _require_nonempty_string(transform["family"], f"{context}.family")
    if family not in declared_families:
        raise SubstrateError(f"{context}.family {family!r} is undeclared")
    registry_entry = transform_registry.get((axis, family))
    if registry_entry is None:
        raise SubstrateError(f"{context}.family {family!r} is absent from registry")
    implementation_sha = _require_sha256(
        transform["implementation_sha256"], f"{context}.implementation_sha256"
    )
    registry_entry_sha = _require_sha256(
        transform["registry_entry_sha256"], f"{context}.registry_entry_sha256"
    )
    if implementation_sha != registry_entry["implementation_sha256"]:
        raise SubstrateError(f"{context}.implementation_sha256 differs from registry")
    if registry_entry_sha != registry_entry["entry_sha256"]:
        raise SubstrateError(f"{context}.registry_entry_sha256 differs from registry")
    parameters = _require_mapping(transform["parameters"], f"{context}.parameters")
    # Canonicalization catches NaN, infinity, bytes, and non-JSON parameters.
    _json_bytes(parameters)
    return {
        "id": transform_id,
        "family": family,
        "implementation_sha256": implementation_sha,
        "registry_entry_sha256": registry_entry_sha,
        "parameters": dict(parameters),
    }


def _validate_proposition(raw: Any, finding: str, context: str) -> dict[str, Any]:
    proposition = _require_mapping(raw, context)
    _require_exact_keys(
        proposition,
        (
            "canonical_claim_id",
            "finding",
            "answer_space",
            "supported_token",
            "refuted_token",
            "undetermined_token",
            "mapping_sha256",
        ),
        context,
    )
    canonical = {
        key: proposition[key] for key in proposition if key != "mapping_sha256"
    }
    for key in canonical:
        canonical[key] = _require_nonempty_string(
            canonical[key], f"{context}.{key}"
        )
    if canonical["finding"] != finding:
        raise SubstrateError(f"{context}.finding does not match unit finding")
    if canonical["answer_space"] != ANSWER_SPACE:
        raise SubstrateError(f"{context}.answer_space is not the frozen trinary space")
    if len(
        {
            canonical["supported_token"],
            canonical["refuted_token"],
            canonical["undetermined_token"],
        }
    ) != 3:
        raise SubstrateError(f"{context} maps multiple states to one token")
    claimed = _require_sha256(proposition["mapping_sha256"], f"{context}.mapping_sha256")
    if object_sha256(canonical) != claimed:
        raise SubstrateError(f"{context}.mapping_sha256 mismatch")
    return {**canonical, "mapping_sha256": claimed}


def _validate_preservation_decision(
    raw: Any, *, pathology: bool, context: str
) -> dict[str, Any]:
    decision = _require_mapping(raw, context)
    common_keys = (
        "decision",
        "review_blinded_to_model_outputs",
        "review_protocol_sha256",
    )
    domain_keys = (
        "finding_evidence_preserved",
        "anatomy_visibility_preserved",
        "attribute_visibility_preserved",
    ) if pathology else (
        "canonical_proposition_preserved",
        "speech_act_preserved",
        "certainty_demand_preserved",
        "answer_space_preserved",
        "output_grammar_preserved",
    )
    _require_exact_keys(decision, (*common_keys, *domain_keys), context)
    if decision["decision"] != "preserved":
        raise SubstrateError(f"{context}.decision must be 'preserved'")
    if decision["review_blinded_to_model_outputs"] is not True:
        raise SubstrateError(f"{context} review must be blinded to model outputs")
    output: dict[str, Any] = {
        "decision": "preserved",
        "review_blinded_to_model_outputs": True,
        "review_protocol_sha256": _require_sha256(
            decision["review_protocol_sha256"],
            f"{context}.review_protocol_sha256",
        ),
    }
    for key in domain_keys:
        if decision[key] is not True:
            raise SubstrateError(f"{context}.{key} must be true")
        output[key] = True
    return output


def _validate_admission_artifact(
    raw: Any,
    *,
    patient_id: str,
    image_id: str,
    finding: str,
    proposition: Mapping[str, Any],
    image_transform: Mapping[str, Any],
    question_transform: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
    context: str,
) -> dict[str, Any]:
    artifact = _require_mapping(raw, context)
    _require_exact_keys(
        artifact,
        (
            "schema_version",
            "artifact_id",
            "patient_id",
            "image_id",
            "finding",
            "canonical_claim_id",
            "source_image_sha256",
            "transformed_image_sha256",
            "canonical_prompt_sha256",
            "transformed_prompt_sha256",
            "image_transform_registry_entry_sha256",
            "question_transform_registry_entry_sha256",
            "pathology_preservation",
            "proposition_preservation",
            "artifact_sha256",
        ),
        context,
    )
    canonical = {
        "schema_version": artifact["schema_version"],
        "artifact_id": _require_nonempty_string(
            artifact["artifact_id"], f"{context}.artifact_id"
        ),
        "patient_id": _require_nonempty_string(
            artifact["patient_id"], f"{context}.patient_id"
        ),
        "image_id": _require_nonempty_string(
            artifact["image_id"], f"{context}.image_id"
        ),
        "finding": _require_nonempty_string(
            artifact["finding"], f"{context}.finding"
        ),
        "canonical_claim_id": _require_nonempty_string(
            artifact["canonical_claim_id"], f"{context}.canonical_claim_id"
        ),
        "source_image_sha256": _require_sha256(
            artifact["source_image_sha256"], f"{context}.source_image_sha256"
        ),
        "transformed_image_sha256": _require_sha256(
            artifact["transformed_image_sha256"],
            f"{context}.transformed_image_sha256",
        ),
        "canonical_prompt_sha256": _require_sha256(
            artifact["canonical_prompt_sha256"],
            f"{context}.canonical_prompt_sha256",
        ),
        "transformed_prompt_sha256": _require_sha256(
            artifact["transformed_prompt_sha256"],
            f"{context}.transformed_prompt_sha256",
        ),
        "image_transform_registry_entry_sha256": _require_sha256(
            artifact["image_transform_registry_entry_sha256"],
            f"{context}.image_transform_registry_entry_sha256",
        ),
        "question_transform_registry_entry_sha256": _require_sha256(
            artifact["question_transform_registry_entry_sha256"],
            f"{context}.question_transform_registry_entry_sha256",
        ),
        "pathology_preservation": _validate_preservation_decision(
            artifact["pathology_preservation"],
            pathology=True,
            context=f"{context}.pathology_preservation",
        ),
        "proposition_preservation": _validate_preservation_decision(
            artifact["proposition_preservation"],
            pathology=False,
            context=f"{context}.proposition_preservation",
        ),
    }
    if canonical["schema_version"] != ADMISSION_SCHEMA_VERSION:
        raise SubstrateError(f"{context}.schema_version mismatch")
    expected = {
        "patient_id": patient_id,
        "image_id": image_id,
        "finding": finding,
        "canonical_claim_id": proposition["canonical_claim_id"],
        "source_image_sha256": cells["clean"]["image_sha256"],
        "transformed_image_sha256": cells["image_only"]["image_sha256"],
        "canonical_prompt_sha256": cells["clean"]["prompt_sha256"],
        "transformed_prompt_sha256": cells["question_only"]["prompt_sha256"],
        "image_transform_registry_entry_sha256": image_transform[
            "registry_entry_sha256"
        ],
        "question_transform_registry_entry_sha256": question_transform[
            "registry_entry_sha256"
        ],
    }
    for key, expected_value in expected.items():
        if canonical[key] != expected_value:
            raise SubstrateError(f"{context}.{key} does not bind the scientific orbit")
    claimed = _require_sha256(
        artifact["artifact_sha256"], f"{context}.artifact_sha256"
    )
    if object_sha256(canonical) != claimed:
        raise SubstrateError(f"{context}.artifact_sha256 mismatch")
    return {**canonical, "artifact_sha256": claimed}


def _validate_logits(raw: Any, context: str) -> dict[str, float]:
    logits = _require_mapping(raw, context)
    _require_exact_keys(logits, LOGIT_KEYS, context)
    output: dict[str, float] = {}
    for key in LOGIT_KEYS:
        value = logits[key]
        if isinstance(value, bool):
            raise SubstrateError(f"{context}.{key} must be numeric")
        try:
            output[key] = float(value)
        except (TypeError, ValueError) as error:
            raise SubstrateError(f"{context}.{key} must be numeric") from error
        if not math.isfinite(output[key]):
            raise SubstrateError(f"{context}.{key} must be finite")
        if abs(output[key]) > MAX_ABSOLUTE_LOGIT:
            raise SubstrateError(
                f"{context}.{key} exceeds the numerical stability contract"
            )
    return output


def _validate_cell(raw: Any, role: str, context: str) -> dict[str, Any]:
    cell = _require_mapping(raw, context)
    _require_exact_keys(
        cell,
        ("role", "image_sha256", "prompt_text", "prompt_sha256", "tristate_logits"),
        context,
    )
    if cell["role"] != role:
        raise SubstrateError(f"{context}.role must equal {role!r}")
    image_digest = _require_sha256(cell["image_sha256"], f"{context}.image_sha256")
    prompt = _require_nonempty_string(cell["prompt_text"], f"{context}.prompt_text")
    prompt_digest = _require_sha256(cell["prompt_sha256"], f"{context}.prompt_sha256")
    if text_sha256(prompt) != prompt_digest:
        raise SubstrateError(f"{context}.prompt_sha256 mismatch")
    return {
        "role": role,
        "image_sha256": image_digest,
        "prompt_text": prompt,
        "prompt_sha256": prompt_digest,
        "tristate_logits": _validate_logits(
            cell["tristate_logits"], f"{context}.tristate_logits"
        ),
    }


def _softmax_with_log(logits: Mapping[str, float]) -> tuple[list[float], list[float]]:
    values = [float(logits[key]) for key in LOGIT_KEYS]
    center = max(values)
    exponential = [math.exp(value - center) for value in values]
    log_normalizer = center + math.log(math.fsum(exponential))
    log_probability = [value - log_normalizer for value in values]
    probability = [math.exp(value) for value in log_probability]
    # The log probabilities, rather than log(probability), are retained so a
    # finite but very small class probability can underflow to zero harmlessly.
    return probability, log_probability


def _kl_from_log(
    left: Sequence[float], left_log: Sequence[float], right_log: Sequence[float]
) -> float:
    terms = [
        probability * (log_left - log_right)
        for probability, log_left, log_right in zip(left, left_log, right_log)
        if probability > 0.0
    ]
    value = math.fsum(terms)
    if not math.isfinite(value):
        raise SubstrateError("KL divergence is non-finite under stability contract")
    # Round-off at identical/extremely concentrated distributions may create a
    # tiny negative value even though KL is nonnegative.
    return max(0.0, value)


def _js(
    left: Sequence[float],
    left_log: Sequence[float],
    right: Sequence[float],
    right_log: Sequence[float],
) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]
    midpoint_log = [math.log(value) if value > 0.0 else -math.inf for value in midpoint]
    return (
        _kl_from_log(left, left_log, midpoint_log)
        + _kl_from_log(right, right_log, midpoint_log)
    ) / 2.0


def _symmetric_kl(
    left: Sequence[float],
    left_log: Sequence[float],
    right: Sequence[float],
    right_log: Sequence[float],
) -> float:
    return (
        _kl_from_log(left, left_log, right_log)
        + _kl_from_log(right, right_log, left_log)
    ) / 2.0


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _behavioral_features(unit: Mapping[str, Any]) -> dict[str, Any]:
    probability_and_log = {
        role: _softmax_with_log(unit["cells"][role]["tristate_logits"])
        for role in CELL_ROLES
    }
    probability = {role: value[0] for role, value in probability_and_log.items()}
    log_probability = {
        role: value[1] for role, value in probability_and_log.items()
    }
    prediction = {
        role: LOGIT_KEYS[max(range(3), key=lambda index: probability[role][index])]
        for role in CELL_ROLES
    }
    clean = probability["clean"]
    image = probability["image_only"]
    question = probability["question_only"]
    joint = probability["joint"]
    image_flip = prediction["image_only"] != prediction["clean"]
    question_flip = prediction["question_only"] != prediction["clean"]
    joint_flip = prediction["joint"] != prediction["clean"]
    probability_interaction = [
        joint[index] - image[index] - question[index] + clean[index]
        for index in range(3)
    ]
    centered_logits: dict[str, list[float]] = {}
    for role in CELL_ROLES:
        vector = [unit["cells"][role]["tristate_logits"][key] for key in LOGIT_KEYS]
        mean = sum(vector) / len(vector)
        centered_logits[role] = [value - mean for value in vector]
    logit_interaction = [
        centered_logits["joint"][index]
        - centered_logits["image_only"][index]
        - centered_logits["question_only"][index]
        + centered_logits["clean"][index]
        for index in range(3)
    ]
    clean_log = log_probability["clean"]
    image_log = log_probability["image_only"]
    question_log = log_probability["question_only"]
    joint_log = log_probability["joint"]
    js_image = _js(clean, clean_log, image, image_log)
    js_question = _js(clean, clean_log, question, question_log)
    js_joint = _js(clean, clean_log, joint, joint_log)
    skl_image = _symmetric_kl(clean, clean_log, image, image_log)
    skl_question = _symmetric_kl(clean, clean_log, question, question_log)
    skl_joint = _symmetric_kl(clean, clean_log, joint, joint_log)
    return {
        "argmax": prediction,
        "image_only_argmax_flip": image_flip,
        "question_only_argmax_flip": question_flip,
        "joint_argmax_flip": joint_flip,
        "singles_stable_but_joint_flips": bool(
            not image_flip and not question_flip and joint_flip
        ),
        "js_clean_to_image_only": js_image,
        "js_clean_to_question_only": js_question,
        "js_clean_to_joint": js_joint,
        "joint_js_excess_over_singles": js_joint - js_image - js_question,
        "symmetric_kl_clean_to_image_only": skl_image,
        "symmetric_kl_clean_to_question_only": skl_question,
        "symmetric_kl_clean_to_joint": skl_joint,
        "joint_symmetric_kl_excess_over_singles": (
            skl_joint - skl_image - skl_question
        ),
        "probability_interaction": probability_interaction,
        "probability_interaction_l2": _l2(probability_interaction),
        "centered_logit_interaction": logit_interaction,
        "centered_logit_interaction_l2": _l2(logit_interaction),
    }


def _validate_stage(
    stage_name: str,
    raw: Any,
    models: Sequence[str],
    findings: set[str],
    transform_registry: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage = _require_mapping(raw, f"stages.{stage_name}")
    _require_exact_keys(
        stage,
        (
            "declared_image_transform_families",
            "declared_question_transform_families",
            "units",
        ),
        f"stages.{stage_name}",
    )
    image_families_raw = stage["declared_image_transform_families"]
    question_families_raw = stage["declared_question_transform_families"]
    if not isinstance(image_families_raw, list) or not image_families_raw:
        raise SubstrateError(f"stages.{stage_name} needs image transform families")
    if not isinstance(question_families_raw, list) or not question_families_raw:
        raise SubstrateError(f"stages.{stage_name} needs question transform families")
    image_families = {
        _require_nonempty_string(value, f"stages.{stage_name}.image_family")
        for value in image_families_raw
    }
    question_families = {
        _require_nonempty_string(value, f"stages.{stage_name}.question_family")
        for value in question_families_raw
    }
    if len(image_families) != len(image_families_raw):
        raise SubstrateError(f"stages.{stage_name} has duplicate image families")
    if len(question_families) != len(question_families_raw):
        raise SubstrateError(f"stages.{stage_name} has duplicate question families")

    units_raw = stage["units"]
    if not isinstance(units_raw, list) or not units_raw:
        raise SubstrateError(f"stages.{stage_name}.units must be a non-empty list")
    units: list[dict[str, Any]] = []
    observed_image_families: set[str] = set()
    observed_question_families: set[str] = set()
    transform_id_registry: dict[tuple[str, str], bytes] = {}
    canonical_registry: dict[tuple[str, str], bytes] = {}
    image_application_registry: dict[tuple[str, str], bytes] = {}
    question_application_registry: dict[tuple[str, str], bytes] = {}
    admission_artifact_registry: dict[str, bytes] = {}
    seen_unit_ids: set[str] = set()
    by_pair_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for index, raw_unit in enumerate(units_raw):
        context = f"stages.{stage_name}.units[{index}]"
        unit = _require_mapping(raw_unit, context)
        _require_exact_keys(
            unit,
            (
                "unit_id",
                "model",
                "patient_id",
                "image_id",
                "finding",
                "image_transform",
                "question_transform",
                "proposition_mapping",
                "admission_artifact",
                "cells",
            ),
            context,
        )
        unit_id = _require_nonempty_string(unit["unit_id"], f"{context}.unit_id")
        if unit_id in seen_unit_ids:
            raise SubstrateError(f"duplicate unit_id {unit_id!r} in {stage_name}")
        seen_unit_ids.add(unit_id)
        model = _require_nonempty_string(unit["model"], f"{context}.model")
        if model not in models:
            raise SubstrateError(f"{context}.model {model!r} is undeclared")
        patient_id = _require_nonempty_string(
            unit["patient_id"], f"{context}.patient_id"
        )
        image_id = _require_nonempty_string(unit["image_id"], f"{context}.image_id")
        finding = _require_nonempty_string(unit["finding"], f"{context}.finding")
        if finding not in findings:
            raise SubstrateError(f"{context}.finding {finding!r} is undeclared")
        image_transform = _validate_transform(
            unit["image_transform"],
            f"{context}.image_transform",
            "image",
            image_families,
            transform_registry,
        )
        question_transform = _validate_transform(
            unit["question_transform"],
            f"{context}.question_transform",
            "question",
            question_families,
            transform_registry,
        )
        observed_image_families.add(image_transform["family"])
        observed_question_families.add(question_transform["family"])
        for axis, transform in (
            ("image", image_transform),
            ("question", question_transform),
        ):
            registry_key = (axis, transform["id"])
            signature = _json_bytes(transform)
            previous = transform_id_registry.setdefault(registry_key, signature)
            if previous != signature:
                raise SubstrateError(
                    f"transform ID {transform['id']!r} changes metadata within {stage_name}"
                )
        proposition = _validate_proposition(
            unit["proposition_mapping"], finding, f"{context}.proposition_mapping"
        )
        cells_raw = _require_mapping(unit["cells"], f"{context}.cells")
        _require_exact_keys(cells_raw, CELL_ROLES, f"{context}.cells")
        cells = {
            role: _validate_cell(cells_raw[role], role, f"{context}.cells.{role}")
            for role in CELL_ROLES
        }
        source_image = cells["clean"]["image_sha256"]
        transformed_image = cells["image_only"]["image_sha256"]
        canonical_prompt = cells["clean"]["prompt_text"]
        transformed_prompt = cells["question_only"]["prompt_text"]
        if transformed_image == source_image:
            raise SubstrateError(f"{context} image transformation has identical bytes")
        if transformed_prompt == canonical_prompt:
            raise SubstrateError(f"{context} question transformation has identical text")
        if cells["question_only"]["image_sha256"] != source_image:
            raise SubstrateError(f"{context} question-only cell changed the image")
        if cells["joint"]["image_sha256"] != transformed_image:
            raise SubstrateError(f"{context} joint cell does not reuse image-only bytes")
        if cells["image_only"]["prompt_text"] != canonical_prompt:
            raise SubstrateError(f"{context} image-only cell changed the question")
        if cells["joint"]["prompt_text"] != transformed_prompt:
            raise SubstrateError(f"{context} joint cell does not reuse question-only text")
        admission = _validate_admission_artifact(
            unit["admission_artifact"],
            patient_id=patient_id,
            image_id=image_id,
            finding=finding,
            proposition=proposition,
            image_transform=image_transform,
            question_transform=question_transform,
            cells=cells,
            context=f"{context}.admission_artifact",
        )
        admission_signature = _json_bytes(admission)
        previous_admission = admission_artifact_registry.setdefault(
            admission["artifact_id"], admission_signature
        )
        if previous_admission != admission_signature:
            raise SubstrateError(
                f"admission artifact ID {admission['artifact_id']!r} changes content"
            )

        normalized = {
            "unit_id": unit_id,
            "model": model,
            "patient_id": patient_id,
            "image_id": image_id,
            "finding": finding,
            "image_transform": image_transform,
            "question_transform": question_transform,
            "proposition_mapping": proposition,
            "admission_artifact": admission,
            "cells": cells,
        }
        canonical_key = (image_id, finding)
        canonical_signature = _json_bytes(
            {
                "source_image_sha256": source_image,
                "canonical_prompt_text": canonical_prompt,
                "canonical_prompt_sha256": cells["clean"]["prompt_sha256"],
                "proposition_mapping": proposition,
                "patient_id": patient_id,
            }
        )
        previous = canonical_registry.setdefault(canonical_key, canonical_signature)
        if previous != canonical_signature:
            raise SubstrateError(
                f"canonical anchor {canonical_key!r} changes across transform pairs"
            )
        image_application_key = (image_id, image_transform["id"])
        image_application_signature = _json_bytes(
            {
                "source_image_sha256": source_image,
                "transformed_image_sha256": transformed_image,
                "image_transform": image_transform,
            }
        )
        previous = image_application_registry.setdefault(
            image_application_key, image_application_signature
        )
        if previous != image_application_signature:
            raise SubstrateError(
                f"image transform application {image_application_key!r} is not exact"
            )
        question_application_key = (finding, question_transform["id"])
        question_application_signature = _json_bytes(
            {
                "canonical_prompt_text": canonical_prompt,
                "canonical_prompt_sha256": cells["clean"]["prompt_sha256"],
                "transformed_prompt_text": transformed_prompt,
                "transformed_prompt_sha256": cells["question_only"]["prompt_sha256"],
                "question_transform": question_transform,
                "proposition_mapping": proposition,
            }
        )
        previous = question_application_registry.setdefault(
            question_application_key, question_application_signature
        )
        if previous != question_application_signature:
            raise SubstrateError(
                f"question transform application {question_application_key!r} is not exact"
            )
        pair_key = (
            image_id,
            finding,
            image_transform["id"],
            question_transform["id"],
        )
        by_pair_key[pair_key].append(normalized)
        units.append(normalized)

    if observed_image_families != image_families:
        raise SubstrateError(
            f"stages.{stage_name} declared/observed image families differ"
        )
    if observed_question_families != question_families:
        raise SubstrateError(
            f"stages.{stage_name} declared/observed question families differ"
        )

    model_set = set(models)
    for pair_key, paired_units in by_pair_key.items():
        observed_models = [unit["model"] for unit in paired_units]
        if len(observed_models) != len(model_set) or set(observed_models) != model_set:
            raise SubstrateError(
                f"orbit {pair_key!r} is not paired exactly across declared models"
            )
        signatures = set()
        for paired in paired_units:
            model_free = {key: value for key, value in paired.items() if key != "model"}
            model_free["unit_id"] = "<model-specific-id>"
            model_free["cells"] = {
                role: {
                    key: value
                    for key, value in paired["cells"][role].items()
                    if key != "tristate_logits"
                }
                for role in CELL_ROLES
            }
            signatures.add(_json_bytes(model_free))
        if len(signatures) != 1:
            raise SubstrateError(
                f"orbit {pair_key!r} input metadata differs across models"
            )

    pairs_by_item: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for image_id, finding, image_transform_id, question_transform_id in by_pair_key:
        pairs_by_item[(image_id, finding)].add(
            (image_transform_id, question_transform_id)
        )
    for item_key, observed_pairs in pairs_by_item.items():
        image_transform_ids = {pair[0] for pair in observed_pairs}
        question_transform_ids = {pair[1] for pair in observed_pairs}
        expected_pairs = {
            (image_transform_id, question_transform_id)
            for image_transform_id in image_transform_ids
            for question_transform_id in question_transform_ids
        }
        if observed_pairs != expected_pairs:
            raise SubstrateError(
                f"item {item_key!r} lacks the exact image-by-question transform product"
            )

    normalized_stage = {
        "declared_image_transform_families": sorted(image_families),
        "declared_question_transform_families": sorted(question_families),
        "units": units,
    }
    return normalized_stage, units


def _validate_finding_stage_quotas(
    raw: Any, findings: Sequence[str]
) -> dict[str, dict[str, int]]:
    quotas = _require_mapping(raw, "finding_stage_quotas")
    _require_exact_keys(quotas, STAGES, "finding_stage_quotas")
    normalized: dict[str, dict[str, int]] = {}
    expected_findings = set(findings)
    for stage_name in STAGES:
        stage_quotas = _require_mapping(
            quotas[stage_name], f"finding_stage_quotas.{stage_name}"
        )
        _require_exact_keys(
            stage_quotas, findings, f"finding_stage_quotas.{stage_name}"
        )
        normalized[stage_name] = {}
        for finding in findings:
            quota = stage_quotas[finding]
            if isinstance(quota, bool) or not isinstance(quota, int) or quota < 1:
                raise SubstrateError(
                    f"finding_stage_quotas.{stage_name}.{finding} must be a positive integer"
                )
            normalized[stage_name][finding] = quota
        if set(normalized[stage_name]) != expected_findings:
            raise SubstrateError(f"{stage_name} finding quota coverage is incomplete")
    return normalized


def validate_and_extract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed and return only generic, reader-label-free control features."""

    root = _require_mapping(payload, "payload")
    _reject_reader_information(root)
    _require_exact_keys(
        root,
        (
            "schema_version",
            "scientific_status",
            "authorized",
            "execution_scope",
            "models",
            "findings",
            "finding_stage_quotas",
            "transform_registry",
            "real_adapter_input_schema",
            "stages",
        ),
        "payload",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise SubstrateError("schema_version mismatch")
    if root["scientific_status"] != SCIENTIFIC_STATUS:
        raise SubstrateError("scientific_status must remain synthetic and non-authorizing")
    if root["authorized"] is not False:
        raise SubstrateError("synthetic substrate can never be authorizing")
    scope = _require_mapping(root["execution_scope"], "execution_scope")
    _require_exact_keys(
        scope,
        ("synthetic_only", "real_results_read", "gpu_used", "reader_labels_used"),
        "execution_scope",
    )
    if dict(scope) != {
        "synthetic_only": True,
        "real_results_read": False,
        "gpu_used": False,
        "reader_labels_used": False,
    }:
        raise SubstrateError("execution_scope must attest synthetic CPU-only label-free use")

    real_adapter_schema = _require_mapping(
        root["real_adapter_input_schema"], "real_adapter_input_schema"
    )
    if _json_bytes(real_adapter_schema) != _json_bytes(REAL_ADAPTER_INPUT_SCHEMA):
        raise SubstrateError(
            "real_adapter_input_schema must remain the frozen, disabled schema handoff"
        )

    models_raw = root["models"]
    findings_raw = root["findings"]
    if not isinstance(models_raw, list) or len(models_raw) < 2:
        raise SubstrateError("at least two declared models are required for exact pairing")
    if not isinstance(findings_raw, list) or not findings_raw:
        raise SubstrateError("at least one finding is required")
    models = [_require_nonempty_string(value, "models[]") for value in models_raw]
    findings = [_require_nonempty_string(value, "findings[]") for value in findings_raw]
    if len(set(models)) != len(models):
        raise SubstrateError("models contains duplicates")
    if len(set(findings)) != len(findings):
        raise SubstrateError("findings contains duplicates")
    finding_stage_quotas = _validate_finding_stage_quotas(
        root["finding_stage_quotas"], findings
    )
    normalized_transform_registry, transform_registry = _validate_transform_registry(
        root["transform_registry"]
    )

    stages_raw = _require_mapping(root["stages"], "stages")
    _require_exact_keys(stages_raw, STAGES, "stages")
    normalized_stages: dict[str, Any] = {}
    units_by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage_name in STAGES:
        normalized_stages[stage_name], units_by_stage[stage_name] = _validate_stage(
            stage_name,
            stages_raw[stage_name],
            models,
            set(findings),
            transform_registry,
        )

    for stage_name in STAGES:
        unique_images_by_finding: dict[str, set[str]] = defaultdict(set)
        for unit in units_by_stage[stage_name]:
            unique_images_by_finding[unit["finding"]].add(unit["image_id"])
        for finding in findings:
            observed = len(unique_images_by_finding[finding])
            required = finding_stage_quotas[stage_name][finding]
            if observed < required:
                raise SubstrateError(
                    f"{stage_name} finding {finding!r} has {observed} unique images; "
                    f"quota requires {required}"
                )

    dev_image_families = set(
        normalized_stages["dev_fit"]["declared_image_transform_families"]
    )
    confirmation_image_families = set(
        normalized_stages["confirmation_locked"][
            "declared_image_transform_families"
        ]
    )
    dev_question_families = set(
        normalized_stages["dev_fit"]["declared_question_transform_families"]
    )
    confirmation_question_families = set(
        normalized_stages["confirmation_locked"][
            "declared_question_transform_families"
        ]
    )
    image_family_leak = dev_image_families & confirmation_image_families
    question_family_leak = dev_question_families & confirmation_question_families
    if image_family_leak:
        raise SubstrateError(
            f"image transform families are not held out: {sorted(image_family_leak)}"
        )
    if question_family_leak:
        raise SubstrateError(
            f"question transform families are not held out: {sorted(question_family_leak)}"
        )
    for axis, declared_key in (
        ("image", "declared_image_transform_families"),
        ("question", "declared_question_transform_families"),
    ):
        dev_implementations = {
            transform_registry[(axis, family)]["implementation_sha256"]
            for family in normalized_stages["dev_fit"][declared_key]
        }
        confirmation_implementations = {
            transform_registry[(axis, family)]["implementation_sha256"]
            for family in normalized_stages["confirmation_locked"][declared_key]
        }
        implementation_leak = dev_implementations & confirmation_implementations
        if implementation_leak:
            raise SubstrateError(
                f"{axis} transform implementations are not held out; "
                "family rename cannot hide leakage"
            )

    dev_images = {unit["image_id"] for unit in units_by_stage["dev_fit"]}
    confirmation_images = {
        unit["image_id"] for unit in units_by_stage["confirmation_locked"]
    }
    if dev_images & confirmation_images:
        raise SubstrateError("image IDs cross dev and confirmation stages")
    dev_patients = {unit["patient_id"] for unit in units_by_stage["dev_fit"]}
    confirmation_patients = {
        unit["patient_id"] for unit in units_by_stage["confirmation_locked"]
    }
    if dev_patients & confirmation_patients:
        raise SubstrateError("patient IDs cross dev and confirmation stages")
    patient_by_image: dict[str, str] = {}
    for stage_name in STAGES:
        for unit in units_by_stage[stage_name]:
            previous_patient = patient_by_image.setdefault(
                unit["image_id"], unit["patient_id"]
            )
            if previous_patient != unit["patient_id"]:
                raise SubstrateError(
                    f"image {unit['image_id']!r} maps to multiple patients"
                )
    dev_source_hashes = {
        unit["cells"]["clean"]["image_sha256"]
        for unit in units_by_stage["dev_fit"]
    }
    confirmation_source_hashes = {
        unit["cells"]["clean"]["image_sha256"]
        for unit in units_by_stage["confirmation_locked"]
    }
    if dev_source_hashes & confirmation_source_hashes:
        raise SubstrateError("source image bytes cross dev and confirmation stages")
    dev_all_image_hashes = {
        unit["cells"][role]["image_sha256"]
        for unit in units_by_stage["dev_fit"]
        for role in CELL_ROLES
    }
    confirmation_all_image_hashes = {
        unit["cells"][role]["image_sha256"]
        for unit in units_by_stage["confirmation_locked"]
        for role in CELL_ROLES
    }
    if dev_all_image_hashes & confirmation_all_image_hashes:
        raise SubstrateError("image bytes cross dev and confirmation stages")

    feature_rows = []
    for stage_name in STAGES:
        for unit in units_by_stage[stage_name]:
            feature_rows.append(
                {
                    "stage": stage_name,
                    "unit_id": unit["unit_id"],
                    "model": unit["model"],
                    "patient_id": unit["patient_id"],
                    "image_id": unit["image_id"],
                    "finding": unit["finding"],
                    "image_transform_id": unit["image_transform"]["id"],
                    "image_transform_family": unit["image_transform"]["family"],
                    "question_transform_id": unit["question_transform"]["id"],
                    "question_transform_family": unit["question_transform"]["family"],
                    "features": _behavioral_features(unit),
                }
            )
    feature_rows.sort(
        key=lambda row: (
            row["stage"],
            row["image_id"],
            row["finding"],
            row["image_transform_id"],
            row["question_transform_id"],
            row["model"],
        )
    )
    normalized_manifest = {
        "schema_version": SCHEMA_VERSION,
        "models": models,
        "findings": findings,
        "finding_stage_quotas": finding_stage_quotas,
        "transform_registry": normalized_transform_registry,
        "real_adapter_input_schema": _canonical_clone(real_adapter_schema),
        "stages": normalized_stages,
    }
    return {
        "schema_version": f"{SCHEMA_VERSION}.audit",
        "scientific_status": SCIENTIFIC_STATUS,
        "authorized": False,
        "gpu_authorized": False,
        "real_results_read": False,
        "reader_labels_used": False,
        "paper_faithfulness_claim": (
            "MetaRA-style CE adaptation; not a faithful MetaRA reproduction"
        ),
        "scientific_interpretation": {
            "interaction_role": "collision_control_readout_only",
            "standalone_metric_novelty": False,
            "standalone_mechanism_evidence": False,
            "pael_absorption_established": False,
            "absorption_requires_separate_preregistered_external_analysis": True,
        },
        "real_adapter_contract": {
            "schema": _canonical_clone(REAL_ADAPTER_INPUT_SCHEMA),
            "schema_sha256": object_sha256(REAL_ADAPTER_INPUT_SCHEMA),
            "execution_enabled": False,
            "adapter_implemented": False,
        },
        "gates": {
            "four_cell_orbits_complete": True,
            "model_orbit_pairing_exact": True,
            "held_out_image_transform_families": True,
            "held_out_question_transform_families": True,
            "transform_families_bound_to_immutable_implementations": True,
            "held_out_transform_implementations": True,
            "admission_content_and_preservation_schema_bound": True,
            "dev_confirmation_images_disjoint": True,
            "dev_confirmation_patients_disjoint": True,
            "two_stage_finding_coverage_and_quotas_met": True,
            "kl_numerical_stability_contract_met": True,
            "generic_collision_features_are_reader_label_free": True,
            "standalone_mechanism_claim_authorized": False,
            "pael_absorption_claim_authorized": False,
            "cecd_authorized": False,
        },
        "counts": {
            stage_name: {
                "units": len(units_by_stage[stage_name]),
                "paired_scientific_orbits": len(units_by_stage[stage_name])
                // len(models),
                "unique_patients": len(
                    {unit["patient_id"] for unit in units_by_stage[stage_name]}
                ),
                "unique_images_by_finding": {
                    finding: len(
                        {
                            unit["image_id"]
                            for unit in units_by_stage[stage_name]
                            if unit["finding"] == finding
                        }
                    )
                    for finding in findings
                },
            }
            for stage_name in STAGES
        },
        "normalized_manifest_sha256": object_sha256(normalized_manifest),
        "feature_rows_sha256": object_sha256(feature_rows),
        "feature_rows": feature_rows,
    }
