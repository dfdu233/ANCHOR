#!/usr/bin/env python3
"""Fail-closed teacher-forced Clinical Autoregressive Lock-in probe.

This is model-independent plumbing.  A production Huatuo adapter must expose
gold continuation probabilities from the exact multimodal chat serialization
at declared decoder layers.  The runtime never discovers claims, prefixes,
splits, thresholds, or layers from development scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import numpy as np


RUNTIME_PROTOCOL_ID = "clinical-autoregressive-lockin-probe-v1"
MANIFEST_PROTOCOL_ID = "clinical-autoregressive-lockin-manifest-v5-natural-tokenwise"
F6_REJECTION_ID = "clinical-lockin-f6-unnatural-prefix-continuation-rejection-v1"
CURRENT_RUNTIME_GPU_AUTHORIZED = False
Condition = Literal["image", "text_only"]
OffsetUnit = Literal["unicode_character", "utf8_byte"]
IMAGE_VARIANTS = ("original", "same_support_swap", "opposite_support_swap")


class ContractError(RuntimeError):
    """The manifest, adapter, or resume artifact violates the frozen contract."""


class RowExclusion(ContractError):
    """One row is unscoreable exactly; no token or template fallback is allowed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"resume fingerprint drift at {path}")
        return
    _atomic_write(path, payload)


@dataclass(frozen=True)
class ContextualContinuationTrace:
    """Exact-context log probabilities for one assistant continuation.

    Prefix and continuation offsets are relative to those exact raw strings;
    chat delimiters, image placeholders, EOS, and assistant suffixes are not
    included.  ``layer_gold_logp`` is layer x continuation-token.  Each layer
    must use the adapter-declared logit-lens rule (normally the model's final
    norm followed by its tied LM head); the final row must equal normal
    teacher-forced logits.
    """

    condition: Condition
    prompt: str
    prefix: str
    continuation: str
    prefix_token_ids: list[int]
    prefix_token_offsets: list[tuple[int, int]]
    continuation_token_ids: list[int]
    continuation_token_offsets: list[tuple[int, int]]
    offset_unit: OffsetUnit
    layer_ids: list[str]
    layer_fractions: list[float]
    layer_gold_logp: list[list[float]]
    serialized_input_sha256: str
    prompt_sha256: str
    prefix_sha256: str
    continuation_sha256: str
    image_sha256: str | None
    template_id: str
    contextual_offsets_certified: bool
    final_layer_matches_standard_logits: bool


@dataclass(frozen=True)
class PromptEndTrace:
    """Hidden readout before the first assistant response token is consumed.

    This trace is the *only* admissible input to the reader-polarity admission
    probe.  Teacher-forced continuations and full-sequence likelihoods cannot
    be substituted for it.
    """

    condition: Condition
    prompt: str
    layer_ids: list[str]
    layer_fractions: list[float]
    layer_prompt_end_hidden: list[list[float]]
    serialized_prompt_sha256: str
    prompt_sha256: str
    image_sha256: str | None
    template_id: str
    prompt_end_position_contract: str
    first_response_token_consumed: bool
    multimodal_expansion_certified: bool


@dataclass(frozen=True)
class GreedyGenerationTrace:
    """Actual generation-only endpoint; exact surface diagnostic, never truth."""

    text: str
    generated_token_ids: list[int]
    image_sha256: str
    prompt_sha256: str
    serialized_prompt_sha256: str
    template_id: str
    decode_contract: str
    hit_max_new_tokens: bool


class LockinAdapter(Protocol):
    """Production boundary; no model-specific assumptions enter this module."""

    def fingerprint(self) -> dict[str, Any]: ...

    def prompt_end(
        self,
        *,
        image_path: Path | None,
        prompt: str,
        condition: Condition,
    ) -> PromptEndTrace: ...

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
    ) -> GreedyGenerationTrace: ...

    def score(
        self,
        *,
        image_path: Path | None,
        prompt: str,
        prefix: str,
        continuation: str,
        condition: Condition,
    ) -> ContextualContinuationTrace: ...


def _validate_prompt_end_trace(
    trace: PromptEndTrace,
    *,
    prompt: str,
    condition: Condition,
    expected_image_sha256: str | None,
) -> np.ndarray:
    if trace.condition != condition or trace.prompt != prompt:
        raise ContractError("prompt-end adapter changed condition or prompt")
    if trace.prompt_sha256 != _sha(prompt.encode()):
        raise ContractError("prompt-end prompt hash mismatch")
    if trace.image_sha256 != expected_image_sha256:
        raise ContractError("prompt-end image hash mismatch")
    if trace.first_response_token_consumed:
        raise ContractError("prompt-end trace consumed an assistant response token")
    if not trace.multimodal_expansion_certified:
        raise RowExclusion("prompt-end multimodal expansion was not certified")
    if not trace.serialized_prompt_sha256 or not trace.template_id:
        raise ContractError("prompt-end serialization/template provenance missing")
    if trace.prompt_end_position_contract not in {
        "last_expanded_prompt_token_before_first_assistant_response_token",
        "assistant_boundary_token_before_first_response_content_token",
    }:
        raise ContractError("unsupported prompt-end position contract")
    if len(trace.layer_ids) != len(trace.layer_fractions) or len(trace.layer_ids) < 4:
        raise ContractError("prompt-end trace needs at least four declared layers")
    fractions = np.asarray(trace.layer_fractions, dtype=float)
    if (
        not np.isfinite(fractions).all()
        or not np.all(np.diff(fractions) > 0)
        or fractions[0] <= 0
        or not math.isclose(float(fractions[-1]), 1.0, abs_tol=1e-9)
    ):
        raise ContractError("invalid prompt-end layer fractions")
    values = np.asarray(trace.layer_prompt_end_hidden, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(trace.layer_ids) or values.shape[1] < 2:
        raise ContractError("prompt-end hidden array must be layer x hidden-dimension")
    if not np.isfinite(values).all():
        raise ContractError("prompt-end hidden array is non-finite")
    return values


def _validate_generation_trace(
    trace: GreedyGenerationTrace,
    *,
    prompt: str,
    expected_image_sha256: str,
) -> None:
    if trace.image_sha256 != expected_image_sha256:
        raise ContractError("generation endpoint image hash mismatch")
    if trace.prompt_sha256 != _sha(prompt.encode()) or not trace.serialized_prompt_sha256:
        raise ContractError("generation endpoint prompt provenance mismatch")
    if trace.decode_contract != "greedy-num_beams1-sampling_false-max_new_tokens256":
        raise ContractError("generation endpoint decode contract drifted")
    if not trace.template_id or not trace.generated_token_ids or not trace.text.strip():
        raise RowExclusion("generation endpoint is empty or lacks provenance")
    if trace.hit_max_new_tokens:
        raise RowExclusion("generation endpoint hit max_new_tokens")


def _char_boundaries(text: str) -> list[int]:
    output = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        output.append(total)
    return output


def _validate_offsets(
    text: str,
    offsets: Sequence[tuple[int, int]],
    unit: OffsetUnit,
    *,
    allow_empty_text: bool,
) -> list[tuple[int, int]]:
    if not text:
        if allow_empty_text and not offsets:
            return []
        raise ContractError("empty text has non-empty contextual offsets")
    boundaries = _char_boundaries(text)
    byte_length = boundaries[-1]
    converted = []
    for index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ContractError(f"offset {index} does not have two values")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end <= start:
            raise ContractError(f"offset {index} is empty, negative, or reversed")
        if unit == "unicode_character":
            if end > len(text):
                raise ContractError("character offset exceeds contextual string")
            start, end = boundaries[start], boundaries[end]
        elif unit == "utf8_byte":
            if end > byte_length or start not in boundaries or end not in boundaries:
                raise ContractError("offset is not aligned to a UTF-8 boundary")
        else:
            raise ContractError(f"unsupported offset unit: {unit}")
        converted.append((start, end))
    for left, right in zip(converted, converted[1:]):
        if left[0] > right[0] or left[1] > right[0]:
            raise ContractError("contextual offsets overlap or are not monotone")
    covered = {position for start, end in converted for position in range(start, end)}
    content = set()
    cursor = 0
    for character in text:
        encoded = character.encode("utf-8")
        if not character.isspace():
            content.update(range(cursor, cursor + len(encoded)))
        cursor += len(encoded)
    if content - covered:
        raise ContractError("contextual offsets do not cover all non-whitespace bytes")
    return converted


def _validate_trace(
    trace: ContextualContinuationTrace,
    *,
    prompt: str,
    prefix: str,
    continuation: str,
    condition: Condition,
    expected_image_sha256: str | None,
) -> np.ndarray:
    if not trace.contextual_offsets_certified:
        raise RowExclusion("adapter did not certify exact contextual offsets")
    if not trace.final_layer_matches_standard_logits:
        raise RowExclusion("final layer was not proven equal to standard teacher-forced logits")
    if (trace.prompt, trace.prefix, trace.continuation, trace.condition) != (
        prompt,
        prefix,
        continuation,
        condition,
    ):
        raise ContractError("adapter returned a different prompt/prefix/continuation/condition")
    if trace.prompt_sha256 != _sha(prompt.encode()):
        raise ContractError("prompt hash mismatch")
    if trace.prefix_sha256 != _sha(prefix.encode()):
        raise ContractError("prefix hash mismatch")
    if trace.continuation_sha256 != _sha(continuation.encode()):
        raise ContractError("continuation hash mismatch")
    if trace.image_sha256 != expected_image_sha256:
        raise ContractError("adapter image hash mismatch")
    if not trace.serialized_input_sha256 or not trace.template_id:
        raise ContractError("adapter omitted serialized/template provenance")
    if len(trace.prefix_token_ids) != len(trace.prefix_token_offsets):
        raise ContractError("prefix token IDs/offsets differ in length")
    if len(trace.continuation_token_ids) != len(trace.continuation_token_offsets):
        raise ContractError("continuation token IDs/offsets differ in length")
    if not trace.continuation_token_ids:
        raise RowExclusion("continuation has no contextual content tokens")
    _validate_offsets(
        prefix,
        trace.prefix_token_offsets,
        trace.offset_unit,
        allow_empty_text=True,
    )
    _validate_offsets(
        continuation,
        trace.continuation_token_offsets,
        trace.offset_unit,
        allow_empty_text=False,
    )
    if len(trace.layer_ids) != len(trace.layer_fractions) or len(trace.layer_ids) < 4:
        raise ContractError("at least four declared decoder layers/fractions are required")
    fractions = np.asarray(trace.layer_fractions, dtype=float)
    if not np.isfinite(fractions).all() or not np.all(np.diff(fractions) > 0):
        raise ContractError("layer fractions must be finite and strictly increasing")
    if fractions[0] <= 0 or not math.isclose(float(fractions[-1]), 1.0, abs_tol=1e-9):
        raise ContractError("layer fractions must lie in (0,1] and include exact final=1")
    values = np.asarray(trace.layer_gold_logp, dtype=float)
    expected = (len(trace.layer_ids), len(trace.continuation_token_ids))
    if values.shape != expected:
        raise ContractError(f"layer log-probability shape {values.shape} != {expected}")
    if not np.isfinite(values).all() or bool((values > 1e-6).any()):
        raise ContractError("gold log probabilities must be finite and <= 0")
    return values


def load_manifest(manifest: Path, metadata: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not manifest.is_file() or not metadata.is_file():
        raise ContractError("manifest and inseparable metadata are required")
    meta = json.loads(metadata.read_text())
    if meta.get("manifest_protocol_id") != MANIFEST_PROTOCOL_ID:
        raise ContractError("wrong manifest protocol")
    if meta.get("status") != "dev_frozen_gpu_not_run" or meta.get("split") != "dev":
        raise ContractError("only the untouched frozen development manifest is admissible")
    if meta.get("dev_model_output_used_for_selection") is not False:
        raise ContractError("development model outputs influenced selection")
    if meta.get("confirmation_split_locked") is not True:
        raise ContractError("confirmation split was not certified locked")
    if _sha_file(manifest) != meta.get("manifest_sha256"):
        raise ContractError("manifest/metadata SHA-256 mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != meta.get("anchor_rows") or not rows:
        raise ContractError("manifest row count mismatch")
    ids: set[str] = set()
    blocks: dict[str, set[int]] = {}
    required = {
        "sample_id",
        "block_id",
        "split",
        "finding",
        "positive_votes",
        "prompt_end_probe_role",
        "prompt_condition",
        "prompt",
        "prompt_utf8_sha256",
        "embedded_claim",
        "embedded_polarity",
        "prefix_ladder",
        "non_attractor_preclaim_template_control",
        "image_conditions",
    }
    for row in rows:
        if row.get("manifest_protocol_id") != MANIFEST_PROTOCOL_ID or required - row.keys():
            raise ContractError("manifest row schema/protocol mismatch")
        if row["sample_id"] in ids or row["split"] != "dev":
            raise ContractError("duplicate sample or non-development row")
        ids.add(row["sample_id"])
        if row["positive_votes"] not in {0, 3}:
            raise ContractError("lock-in admission uses only 0/3 and 3/3 reader support")
        expected_condition = {
            "pleural_effusion": "existential",
            "lung_opacity": "negative_obligation",
        }.get(row["finding"])
        if expected_condition is None or row["prompt_condition"] != expected_condition:
            raise ContractError("claim-specific pilot prompt condition drifted")
        if row["prompt_utf8_sha256"] != _sha(str(row["prompt"]).encode()):
            raise ContractError("manifest prompt hash mismatch")
        if row["embedded_polarity"] != "present":
            raise ContractError("frozen embedded claim is not positive/present")
        if row["prompt_end_probe_role"] not in {"probe_fit", "probe_eval"}:
            raise ContractError("prompt-end probe block role is absent or invalid")
        blocks.setdefault(row["block_id"], set()).add(int(row["positive_votes"]))
        ladder = row["prefix_ladder"]
        if [step.get("step") for step in ladder] != list(range(5)):
            raise ContractError("prefix ladder must have the frozen five ordered steps")
        if any(not step.get("claim_begins_after_prefix") for step in ladder):
            raise ContractError("a ladder step crosses the claim onset")
        conditions = row["image_conditions"]
        if set(conditions) != set(IMAGE_VARIANTS):
            raise ContractError("image intervention set drifted")
        if conditions["original"]["positive_votes"] != row["positive_votes"]:
            raise ContractError("original support differs from row support")
        if conditions["same_support_swap"]["positive_votes"] != row["positive_votes"]:
            raise ContractError("same-support swap is not same-support")
        if conditions["opposite_support_swap"]["positive_votes"] != 3 - row["positive_votes"]:
            raise ContractError("opposite-support swap is not opposite-support")
    if any(values != {0, 3} for values in blocks.values()):
        raise ContractError("each independent block needs one 0/3 and one 3/3 anchor")
    block_roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        block_roles[row["block_id"]].add(row["prompt_end_probe_role"])
    if any(len(roles) != 1 for roles in block_roles.values()):
        raise ContractError("one independent block crosses prompt-end fit/eval roles")
    for finding in {row["finding"] for row in rows}:
        finding_rows = [row for row in rows if row["finding"] == finding]
        if len({row["prompt"] for row in finding_rows}) != 1:
            raise ContractError("one finding uses multiple prompts")
        if len({_canonical(row["prefix_ladder"]) for row in finding_rows}) != 1:
            raise ContractError("one finding uses multiple prefix ladders")
        fit = {
            row["block_id"]
            for row in rows
            if row["finding"] == finding and row["prompt_end_probe_role"] == "probe_fit"
        }
        evaluate = {
            row["block_id"]
            for row in rows
            if row["finding"] == finding and row["prompt_end_probe_role"] == "probe_eval"
        }
        if not fit or len(fit) != len(evaluate) or fit & evaluate:
            raise ContractError("prompt-end probe block split is not balanced/disjoint")
    return rows, meta


def _resolve_image(root: Path, relative: str, expected_hash: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"image escapes --image-root: {relative}") from exc
    if not path.is_file() or _sha_file(path) != expected_hash:
        raise ContractError(f"image missing or hash drifted: {path}")
    return path


def _trace_set(
    adapter: LockinAdapter,
    *,
    row: dict[str, Any],
    image_root: Path,
    prefix: str,
    continuation: str,
) -> dict[str, ContextualContinuationTrace]:
    traces = {}
    for variant in IMAGE_VARIANTS:
        reference = row["image_conditions"][variant]
        path = _resolve_image(image_root, reference["dicom_relpath"], reference["dicom_sha256"])
        traces[variant] = adapter.score(
            image_path=path,
            prompt=row["prompt"],
            prefix=prefix,
            continuation=continuation,
            condition="image",
        )
    traces["text_only"] = adapter.score(
        image_path=None,
        prompt=row["prompt"],
        prefix=prefix,
        continuation=continuation,
        condition="text_only",
    )
    return traces


def _validate_trace_set(
    traces: dict[str, ContextualContinuationTrace],
    *,
    row: dict[str, Any],
    prefix: str,
    continuation: str,
) -> tuple[dict[str, np.ndarray], ContextualContinuationTrace]:
    values = {}
    for variant, trace in traces.items():
        condition: Condition = "text_only" if variant == "text_only" else "image"
        expected = (
            None
            if variant == "text_only"
            else row["image_conditions"][variant]["dicom_sha256"]
        )
        values[variant] = _validate_trace(
            trace,
            prompt=row["prompt"],
            prefix=prefix,
            continuation=continuation,
            condition=condition,
            expected_image_sha256=expected,
        )
    reference = traces["original"]
    identity = (
        reference.prefix_token_ids,
        reference.prefix_token_offsets,
        reference.continuation_token_ids,
        reference.continuation_token_offsets,
        reference.offset_unit,
        reference.layer_ids,
        reference.layer_fractions,
        reference.template_id,
    )
    for variant, trace in traces.items():
        other = (
            trace.prefix_token_ids,
            trace.prefix_token_offsets,
            trace.continuation_token_ids,
            trace.continuation_token_offsets,
            trace.offset_unit,
            trace.layer_ids,
            trace.layer_fractions,
            trace.template_id,
        )
        if other != identity:
            raise RowExclusion(
                f"contextual token/layer/template identity differs for {variant}"
            )
    return values, reference


def compute_row(adapter: LockinAdapter, row: dict[str, Any], image_root: Path) -> dict[str, Any]:
    sign = 1.0 if int(row["positive_votes"]) == 3 else -1.0
    generations: dict[str, GreedyGenerationTrace] = {}
    for variant in IMAGE_VARIANTS:
        reference = row["image_conditions"][variant]
        path = _resolve_image(
            image_root, reference["dicom_relpath"], reference["dicom_sha256"]
        )
        trace = adapter.generate(image_path=path, prompt=row["prompt"])
        _validate_generation_trace(
            trace,
            prompt=row["prompt"],
            expected_image_sha256=reference["dicom_sha256"],
        )
        generations[variant] = trace
    prompt_end_values: dict[str, np.ndarray] = {}
    prompt_end_traces: dict[str, PromptEndTrace] = {}
    prompt_end_identity: tuple[list[str], list[float], str, int, str] | None = None
    for variant in (*IMAGE_VARIANTS, "text_only"):
        if variant == "text_only":
            path = None
            expected_image_sha256 = None
            condition: Condition = "text_only"
        else:
            condition = "image"
            reference = row["image_conditions"][variant]
            path = _resolve_image(
                image_root, reference["dicom_relpath"], reference["dicom_sha256"]
            )
            expected_image_sha256 = reference["dicom_sha256"]
        trace = adapter.prompt_end(
            image_path=path, prompt=row["prompt"], condition=condition
        )
        values = _validate_prompt_end_trace(
            trace,
            prompt=row["prompt"],
            condition=condition,
            expected_image_sha256=expected_image_sha256,
        )
        identity = (
            trace.layer_ids,
            trace.layer_fractions,
            trace.template_id,
            int(values.shape[1]),
            trace.prompt_end_position_contract,
        )
        if variant == "text_only":
            # Text-only is an explicit nuisance control.  Removing the image
            # changes serialization and may change the boundary/template; we
            # record those facts and never demand contextual-token identity.
            pass
        elif prompt_end_identity is None:
            prompt_end_identity = identity
        elif identity != prompt_end_identity:
            raise RowExclusion("prompt-end layer/template/hidden identity differs by image")
        prompt_end_values[variant] = values
        prompt_end_traces[variant] = trace
    assert prompt_end_identity is not None
    text_trace = prompt_end_traces["text_only"]
    text_values = prompt_end_values["text_only"]
    if (
        text_trace.layer_ids != prompt_end_identity[0]
        or text_trace.layer_fractions != prompt_end_identity[1]
        or int(text_values.shape[1]) != prompt_end_identity[3]
    ):
        raise RowExclusion(
            "text-only prompt-end layers/hidden dimension are not comparable to image path"
        )
    ladder_payload = []
    common_identity: tuple[list[str], list[float], str] | None = None
    for step in row["prefix_ladder"]:
        prefix = str(step["prefix"])
        continuation = str(row["embedded_claim"])
        traces = _trace_set(
            adapter,
            row=row,
            image_root=image_root,
            prefix=prefix,
            continuation=continuation,
        )
        values, reference = _validate_trace_set(
            traces, row=row, prefix=prefix, continuation=continuation
        )
        identity = (reference.layer_ids, reference.layer_fractions, reference.template_id)
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise RowExclusion("layers or chat template changed across prefix steps")
        means = {name: matrix.mean(axis=1) for name, matrix in values.items()}
        opposite = sign * (means["original"] - means["opposite_support_swap"])
        same = np.abs(means["original"] - means["same_support_swap"])
        token_opposite = sign * (
            values["original"] - values["opposite_support_swap"]
        )
        token_same = np.abs(values["original"] - values["same_support_swap"])
        ladder_payload.append(
            {
                "step": int(step["step"]),
                "phase": step["phase"],
                "prefix_utf8_sha256": step["prefix_utf8_sha256"],
                "prefix_token_ids": reference.prefix_token_ids,
                "prefix_token_offsets": reference.prefix_token_offsets,
                "continuation_token_ids": reference.continuation_token_ids,
                "continuation_token_offsets": reference.continuation_token_offsets,
                "offset_unit": reference.offset_unit,
                "layer_gold_logp": {
                    name: matrix.tolist() for name, matrix in values.items()
                },
                "layer_mean_logp": {
                    name: vector.astype(float).tolist() for name, vector in means.items()
                },
                "effects": {
                    "oriented_original_minus_opposite": opposite.astype(float).tolist(),
                    "absolute_original_minus_same": same.astype(float).tolist(),
                    "causal_excess_over_same_support": (opposite - same).astype(float).tolist(),
                    "token_oriented_original_minus_opposite": token_opposite.astype(float).tolist(),
                    "token_absolute_original_minus_same": token_same.astype(float).tolist(),
                },
                "serialized_input_sha256": {
                    name: trace.serialized_input_sha256 for name, trace in traces.items()
                },
            }
        )

    polarity = row["non_attractor_preclaim_template_control"]
    polarity_values: dict[str, dict[str, Any]] = {}
    for label, continuation in (
        ("present", polarity["present_continuation"]),
        ("absent", polarity["absent_continuation"]),
    ):
        traces = _trace_set(
            adapter,
            row=row,
            image_root=image_root,
            prefix=polarity["prefix"],
            continuation=continuation,
        )
        values, reference = _validate_trace_set(
            traces, row=row, prefix=polarity["prefix"], continuation=continuation
        )
        if (reference.layer_ids, reference.layer_fractions, reference.template_id) != common_identity:
            raise RowExclusion("polarity control uses different layers or chat template")
        polarity_values[label] = {
            "means": {name: matrix.mean(axis=1) for name, matrix in values.items()},
            "continuation_token_ids": reference.continuation_token_ids,
            "continuation_token_offsets": reference.continuation_token_offsets,
            "serialized_input_sha256": {
                name: trace.serialized_input_sha256 for name, trace in traces.items()
            },
        }
    margins = {
        variant: polarity_values["present"]["means"][variant]
        - polarity_values["absent"]["means"][variant]
        for variant in (*IMAGE_VARIANTS, "text_only")
    }
    polarity_opposite = sign * (margins["original"] - margins["opposite_support_swap"])
    polarity_same = np.abs(margins["original"] - margins["same_support_swap"])
    assert common_identity is not None
    layer_ids, layer_fractions, template_id = common_identity
    if (
        prompt_end_identity[0] != layer_ids
        or prompt_end_identity[1] != layer_fractions
        or prompt_end_identity[2] != template_id
    ):
        raise RowExclusion("prompt-end and teacher-forced layer/template identity differs")
    if any(trace.template_id != template_id for trace in generations.values()):
        raise RowExclusion("generation and mechanistic traces use different chat templates")
    normalized_generation = {
        name: " ".join(trace.text.casefold().split())
        for name, trace in generations.items()
    }
    embedded = str(row["embedded_claim"]).casefold()
    return {
        "sample_id": row["sample_id"],
        "block_id": row["block_id"],
        "finding": row["finding"],
        "positive_votes": row["positive_votes"],
        "prompt_end_probe_role": row["prompt_end_probe_role"],
        "layer_ids": layer_ids,
        "layer_fractions": layer_fractions,
        "template_id": template_id,
        "generation_endpoint": {
            "role": "exact surface endpoint only; never a clinical truth label",
            "decode_contract": "greedy-num_beams1-sampling_false-max_new_tokens256",
            "text": {name: trace.text for name, trace in generations.items()},
            "generated_token_ids": {
                name: trace.generated_token_ids for name, trace in generations.items()
            },
            "normalized_text_sha256": {
                name: _sha(text.encode()) for name, text in normalized_generation.items()
            },
            "contains_frozen_embedded_claim_surface": {
                name: embedded in text for name, text in normalized_generation.items()
            },
            "original_opposite_same_embedded_claim_surface": (
                embedded in normalized_generation["original"]
                and embedded in normalized_generation["opposite_support_swap"]
            ),
            "original_opposite_exact_full_text_collision": (
                normalized_generation["original"]
                == normalized_generation["opposite_support_swap"]
            ),
            "clinical_correctness_assigned": False,
        },
        "prompt_end_readout": {
            "role": (
                "only admissible source for pre-response reader-polarity decoding; "
                "no assistant response content token consumed"
            ),
            "position_contract": prompt_end_identity[4],
            "hidden_dimension": prompt_end_identity[3],
            "layer_hidden": {
                name: values.astype(float).tolist()
                for name, values in prompt_end_values.items()
            },
            "serialized_prompt_sha256": {
                name: trace.serialized_prompt_sha256
                for name, trace in prompt_end_traces.items()
            },
            "text_only_control": {
                "used_to_fit_or_select_prompt_end_probe": False,
                "same_prompt_no_image": True,
                "template_id": prompt_end_traces["text_only"].template_id,
                "position_contract": prompt_end_traces["text_only"].prompt_end_position_contract,
                "template_matches_image_path": (
                    prompt_end_traces["text_only"].template_id == prompt_end_identity[2]
                ),
                "position_contract_matches_image_path": (
                    prompt_end_traces["text_only"].prompt_end_position_contract
                    == prompt_end_identity[4]
                ),
                "token_identity_required": False,
                "layer_and_hidden_dimension_alignment_required": True,
            },
        },
        "prefix_ladder": ladder_payload,
        "non_attractor_preclaim_template_control": {
            "prefix_utf8_sha256": polarity["prefix_utf8_sha256"],
            "present_continuation_token_ids": polarity_values["present"]["continuation_token_ids"],
            "absent_continuation_token_ids": polarity_values["absent"]["continuation_token_ids"],
            "margin_present_minus_absent": {
                name: value.astype(float).tolist() for name, value in margins.items()
            },
            "oriented_original_minus_opposite": polarity_opposite.astype(float).tolist(),
            "absolute_original_minus_same": polarity_same.astype(float).tolist(),
            "causal_excess_over_same_support": (polarity_opposite - polarity_same)
            .astype(float)
            .tolist(),
            "serialized_input_sha256": {
                label: values["serialized_input_sha256"]
                for label, values in polarity_values.items()
            },
            "template_role": polarity["role"],
            "not_pre_response_hidden_decoding": True,
        },
    }


def _row_sha(row: dict[str, Any]) -> str:
    return _sha(_canonical(row))


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _validate_shard(path: Path, fingerprint: str, row_sha: str) -> dict[str, Any]:
    try:
        shard = json.loads(path.read_text())
    except Exception as exc:
        raise ContractError(f"corrupt resume shard {path}: {exc}") from exc
    if shard.get("runtime_protocol_id") != RUNTIME_PROTOCOL_ID:
        raise ContractError("resume shard protocol drift")
    if shard.get("config_fingerprint") != fingerprint or shard.get("row_sha256") != row_sha:
        raise ContractError("resume shard fingerprint drift")
    if shard.get("payload_sha256") != _sha(_canonical(shard.get("payload"))):
        raise ContractError("resume shard payload checksum mismatch")
    return shard["payload"]


def build_controls(payloads: Sequence[dict[str, Any]], manifest_sha256: str) -> dict[str, Any]:
    ok = [payload for payload in payloads if payload.get("status") == "ok"]
    random_mapping: dict[str, str] = {}
    for finding in sorted({payload["finding"] for payload in ok}):
        group = [payload for payload in ok if payload["finding"] == finding]
        ordered = sorted(
            group,
            key=lambda row: _sha(
                f"{manifest_sha256}|random-pair|{row['sample_id']}".encode()
            ),
        )
        for shift in range(1, len(ordered)):
            candidate = {
                target["sample_id"]: ordered[(index + shift) % len(ordered)]["sample_id"]
                for index, target in enumerate(ordered)
            }
            by_id = {row["sample_id"]: row for row in ordered}
            if all(
                by_id[source]["block_id"] != by_id[target]["block_id"]
                for target, source in candidate.items()
            ):
                random_mapping.update(candidate)
                break
    length_rows = []
    for payload in ok:
        for step in payload["prefix_ladder"]:
            length_rows.append(
                {
                    "sample_id": payload["sample_id"],
                    "finding": payload["finding"],
                    "step": step["step"],
                    "prefix_token_count": len(step["prefix_token_ids"]),
                    "continuation_token_count": len(step["continuation_token_ids"]),
                }
            )
    cross_claim_matches = {}
    for step in range(5):
        counts = {
            finding: sorted(
                {
                    row["prefix_token_count"]
                    for row in length_rows
                    if row["finding"] == finding and row["step"] == step
                }
            )
            for finding in {row["finding"] for row in length_rows}
        }
        common = sorted(set.intersection(*(set(values) for values in counts.values()))) if counts else []
        cross_claim_matches[str(step)] = {
            "per_finding_prefix_token_counts": counts,
            "exact_common_counts": common,
            "eligible": bool(common),
        }
    return {
        "protocol": "clinical-autoregressive-lockin-controls-v1",
        "random_pair_control": {
            "mapping_target_to_source": dict(sorted(random_mapping.items())),
            "different_independent_block_required": True,
            "support_not_used_in_pairing": True,
            "mapping_frozen_by_manifest_hash": True,
        },
        "length_control": {
            "policy": (
                "within-claim continuation-token count fixed; common-step linear prefix-token "
                "trend extrapolated to modifier; cross-claim exact counts diagnostic only"
            ),
            "cross_claim_step_matches": cross_claim_matches,
            "continuation_token_count_retained_as_fixed_claim stratum": True,
            "smooth_length_null": "within-finding linear prefix-token-count trend; preregistered modifier residual must remain",
            "cross_claim_caliper_widening_forbidden": True,
        },
        "template_control": {
            "policy": "same user prompt and claim identity; non-attractor present-vs-absent template",
            "required_for_admission": True,
            "not_a_second_prompt_condition": True,
        },
    }


def run_runtime(
    *,
    manifest: Path,
    metadata: Path,
    image_root: Path,
    output_dir: Path,
    adapter: LockinAdapter,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    raise ContractError(
        f"{F6_REJECTION_ID}: the fixed-continuation prefix-ladder runtime is rejected; "
        "v4 is not GPU-authorized and a separate natural-sequence tokenwise v5 runtime "
        "has not been implemented"
    )
    # Unreachable legacy implementation is intentionally retained temporarily
    # for exact forensic comparison with the F6 audit.  Removing this guard is
    # forbidden; v5 requires a new tokenwise entry point and conformance suite.
    rows, meta = load_manifest(manifest, metadata)
    fingerprint_payload = adapter.fingerprint()
    if not isinstance(fingerprint_payload, dict) or not fingerprint_payload:
        raise ContractError("adapter fingerprint must be a non-empty object")
    required_adapter_fields = {
        "model_family",
        "model_artifact_fingerprint",
        "tokenizer_fingerprint",
        "chat_template_sha256",
        "multimodal_expansion_contract",
        "prompt_end_position_contract",
        "layer_logit_lens_contract",
        "generation_decode_contract",
    }
    if required_adapter_fields - fingerprint_payload.keys():
        raise ContractError("adapter fingerprint omits required scientific identity")
    _canonical(fingerprint_payload)
    config = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": meta["manifest_sha256"],
        "metadata": str(metadata.resolve()),
        "metadata_sha256": _sha_file(metadata),
        "image_root": str(image_root.resolve()),
        "adapter_fingerprint": fingerprint_payload,
        "runtime_source_sha256": _sha_file(Path(__file__)),
        "command": list(command or []),
        "scientific_split": "dev",
        "thresholds_selected_from_this_run": False,
        "teacher_forcing": "exact serialized assistant prefix and continuation",
    }
    fingerprint = _sha(_canonical(config))
    config["config_fingerprint"] = fingerprint
    _write_once_or_equal(
        output_dir / "config.json",
        json.dumps(config, indent=2, sort_keys=True).encode() + b"\n",
    )
    ordered = [row["sample_id"] for row in rows]
    _write_once_or_equal(
        output_dir / "ordered_keys.json",
        json.dumps(
            {"sample_ids": ordered, "sample_ids_sha256": _sha(_canonical(ordered))},
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n",
    )
    payloads = []
    resumed = 0
    for index, row in enumerate(rows):
        row_hash = _row_sha(row)
        shard_path = output_dir / "shards" / f"{index:04d}-{_safe(row['sample_id'])}.json"
        if shard_path.exists():
            payloads.append(_validate_shard(shard_path, fingerprint, row_hash))
            resumed += 1
            continue
        try:
            payload = {"status": "ok", **compute_row(adapter, row, image_root)}
        except RowExclusion as exc:
            payload = {
                "status": "excluded",
                "sample_id": row["sample_id"],
                "block_id": row["block_id"],
                "finding": row["finding"],
                "positive_votes": row["positive_votes"],
                "reason": str(exc),
                "no_fallback_used": True,
            }
        shard = {
            "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
            "config_fingerprint": fingerprint,
            "row_sha256": row_hash,
            "payload_sha256": _sha(_canonical(payload)),
            "payload": payload,
        }
        _atomic_write(
            shard_path,
            json.dumps(shard, indent=2, sort_keys=True).encode() + b"\n",
        )
        payloads.append(payload)
    controls = build_controls(payloads, meta["manifest_sha256"])
    _write_once_or_equal(
        output_dir / "controls.json",
        json.dumps(controls, indent=2, sort_keys=True).encode() + b"\n",
    )
    ok = [payload for payload in payloads if payload["status"] == "ok"]
    per_cell = {
        f"{finding}:{support}": sum(
            row["finding"] == finding and row["positive_votes"] == support for row in ok
        )
        for finding in sorted({row["finding"] for row in rows})
        for support in (0, 3)
    }
    completion = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "status": "complete" if len(ok) == len(rows) else "complete_with_exclusions",
        "config_fingerprint": fingerprint,
        "rows": len(rows),
        "analyzable_rows": len(ok),
        "excluded_rows": len(rows) - len(ok),
        "per_finding_support_cell": per_cell,
        "analysis_input_complete": (
            len(ok) >= math.ceil(0.8 * len(rows)) and min(per_cell.values()) >= 10
        ),
        "scientific_gate_authorized": False,
        "scientific_gate_status": "pending_preregistered_analysis",
        "confirmation_or_patching_authorized": False,
    }
    _write_once_or_equal(
        output_dir / "COMPLETE.json",
        json.dumps(completion, indent=2, sort_keys=True).encode() + b"\n",
    )
    return {**completion, "resumed_rows": resumed}


def _load_factory(specification: str, config: dict[str, Any]) -> LockinAdapter:
    if ":" not in specification:
        raise ContractError("--adapter-factory must be module:function")
    module_name, function_name = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--adapter-config-json", default="{}")
    args = parser.parse_args()
    adapter = _load_factory(args.adapter_factory, json.loads(args.adapter_config_json))
    result = run_runtime(
        manifest=args.manifest,
        metadata=args.metadata,
        image_root=args.image_root,
        output_dir=args.output_dir,
        adapter=adapter,
        command=os.sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
