#!/usr/bin/env python3
"""Fail-closed teacher-forcing runtime for Specificity Ratchet v1.

This module intentionally contains no Huatuo/Hulu model assumptions.  A model
adapter must return gold-token log probabilities and offsets produced by its
*exact serialized teacher-forcing path*.  Standalone target tokenization is not
an admissible substitute.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import numpy as np


RUNTIME_PROTOCOL_ID = "specificity-ratchet-teacher-forcing-v1"
MANIFEST_PROTOCOL_ID = "specificity-ratchet-mechanism-v1"
F6_REJECTION_ARTIFACT = (
    "corrected_runs/specificity_ratchet/isolated_target_runtime_v1_NOT_AUTHORIZED_F6.json"
)
OffsetUnit = Literal["unicode_character", "utf8_byte"]
Condition = Literal["image", "text_only"]


class ContractError(RuntimeError):
    """A scientific input or adapter violated the frozen runtime contract."""


class RowExclusion(ContractError):
    """A target is not exactly scoreable and must be excluded, never repaired."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
class TeacherForcedTrace:
    """Model-independent output of one exact-template teacher-forcing pass.

    ``token_offsets`` are relative to ``target`` and cover content tokens only:
    assistant delimiters, EOS, and other template tokens must be removed.  Each
    row of ``layer_gold_logp`` corresponds to a declared ``layer_id`` and each
    column to one response content token.
    """

    condition: Condition
    target: str
    token_ids: list[int]
    token_offsets: list[tuple[int, int]]
    offset_unit: OffsetUnit
    layer_ids: list[str]
    layer_gold_logp: list[list[float]]
    serialized_input_sha256: str
    prompt_sha256: str
    target_sha256: str
    image_sha256: str | None
    template_id: str
    contextual_offsets_certified: bool


class TeacherForcingAdapter(Protocol):
    """Required production boundary; implementations may use any model family."""

    def fingerprint(self) -> dict[str, Any]: ...

    def score(
        self,
        *,
        image_path: Path | None,
        question: str,
        target: str,
        condition: Condition,
    ) -> TeacherForcedTrace: ...


def _char_to_byte_boundaries(text: str) -> list[int]:
    boundaries = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        boundaries.append(total)
    return boundaries


def _offsets_as_bytes(
    target: str, offsets: Sequence[tuple[int, int]], unit: OffsetUnit
) -> list[tuple[int, int]]:
    boundaries = _char_to_byte_boundaries(target)
    byte_length = boundaries[-1]
    output: list[tuple[int, int]] = []
    for index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ContractError(f"token {index}: offset must have two integers")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end <= start:
            raise ContractError(f"token {index}: empty/special/reversed target offset {pair}")
        if unit == "unicode_character":
            if end > len(target):
                raise ContractError(f"token {index}: character offset exceeds target")
            start, end = boundaries[start], boundaries[end]
        elif unit == "utf8_byte":
            if end > byte_length or start not in boundaries or end not in boundaries:
                raise ContractError(f"token {index}: offset is not on a UTF-8 boundary")
        else:
            raise ContractError(f"unsupported offset unit: {unit!r}")
        output.append((start, end))
    for left, right in zip(output, output[1:]):
        if left[0] > right[0] or left[1] > right[0]:
            raise ContractError("target token offsets overlap or are not monotone")
    return output


def _non_whitespace_bytes(text: str) -> set[int]:
    result: set[int] = set()
    cursor = 0
    for character in text:
        encoded = character.encode("utf-8")
        if not character.isspace():
            result.update(range(cursor, cursor + len(encoded)))
        cursor += len(encoded)
    return result


def validate_full_target_coverage(
    target: str, offsets: Sequence[tuple[int, int]], unit: OffsetUnit
) -> list[tuple[int, int]]:
    byte_offsets = _offsets_as_bytes(target, offsets, unit)
    covered = {position for start, end in byte_offsets for position in range(start, end)}
    missing = _non_whitespace_bytes(target) - covered
    if missing:
        raise ContractError(
            f"contextual offsets do not cover {len(missing)} non-whitespace target bytes"
        )
    return byte_offsets


def map_constraint_spans(
    target: str,
    spans: Sequence[dict[str, Any]],
    offsets: Sequence[tuple[int, int]],
    unit: OffsetUnit,
) -> list[int]:
    """Map every frozen character span to the exact contextual-token union.

    Tokens may spill over adjacent whitespace (common for leading-space BPEs),
    but any spill into another non-whitespace character is a hard boundary
    failure.  Every occurrence remains independently checked before unioning.
    """

    byte_offsets = validate_full_target_coverage(target, offsets, unit)
    char_boundaries = _char_to_byte_boundaries(target)
    frozen_intervals: list[tuple[int, int]] = []
    for index, span in enumerate(spans):
        start = span.get("char_start")
        end = span.get("char_end_exclusive")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(target)):
            raise RowExclusion(f"constraint span {index}: invalid character boundary")
        exact = target[start:end]
        if exact != span.get("text"):
            raise RowExclusion(f"constraint span {index}: frozen text mismatch")
        if _sha256_bytes(exact.encode("utf-8")) != span.get("utf8_sha256"):
            raise RowExclusion(f"constraint span {index}: UTF-8 hash mismatch")
        frozen_intervals.append((char_boundaries[start], char_boundaries[end]))
    if not frozen_intervals:
        raise RowExclusion("child has no frozen constraint spans")
    for left, right in zip(frozen_intervals, frozen_intervals[1:]):
        if left[1] > right[0]:
            raise RowExclusion("frozen constraint spans overlap or are unsorted")

    selected: set[int] = set()
    for span_index, (span_start, span_end) in enumerate(frozen_intervals):
        occurrence = {
            token_index
            for token_index, (token_start, token_end) in enumerate(byte_offsets)
            if token_start < span_end and span_start < token_end
        }
        if not occurrence:
            raise RowExclusion(f"constraint span {span_index}: no contextual token overlaps")
        covered = {
            position
            for token_index in occurrence
            for position in range(*byte_offsets[token_index])
        }
        required = set(range(span_start, span_end)) & _non_whitespace_bytes(target)
        if required - covered:
            raise RowExclusion(f"constraint span {span_index}: incomplete token coverage")
        selected.update(occurrence)

    union_bytes = {
        position for start, end in frozen_intervals for position in range(start, end)
    }
    nonwhite = _non_whitespace_bytes(target)
    spill = {
        position
        for token_index in selected
        for position in range(*byte_offsets[token_index])
        if position not in union_bytes and position in nonwhite
    }
    if spill:
        raise RowExclusion(
            "constraint token boundary spills into non-whitespace text; exact span is not token-identifiable"
        )
    return sorted(selected)


def _validate_trace(
    trace: TeacherForcedTrace,
    *,
    target: str,
    question: str,
    condition: Condition,
    expected_image_sha256: str | None,
) -> np.ndarray:
    if not trace.contextual_offsets_certified:
        raise ContractError("adapter did not certify exact-template contextual offsets")
    if trace.condition != condition or trace.target != target:
        raise ContractError("adapter trace condition/target mismatch")
    if trace.prompt_sha256 != _sha256_bytes(question.encode("utf-8")):
        raise ContractError("adapter prompt hash mismatch")
    if trace.target_sha256 != _sha256_bytes(target.encode("utf-8")):
        raise ContractError("adapter target hash mismatch")
    if trace.image_sha256 != expected_image_sha256:
        raise ContractError("adapter image hash mismatch or image leaked into text-only condition")
    if not re.fullmatch(r"[0-9a-f]{64}", trace.serialized_input_sha256):
        raise ContractError("adapter serialized-input fingerprint is absent")
    if not trace.template_id.strip():
        raise ContractError("adapter template_id is absent")
    count = len(trace.token_ids)
    if count == 0 or len(trace.token_offsets) != count:
        raise ContractError("empty or misaligned response token trace")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in trace.token_ids):
        raise ContractError("token ids must be non-negative integers")
    validate_full_target_coverage(target, trace.token_offsets, trace.offset_unit)
    values = np.asarray(trace.layer_gold_logp, dtype=np.float64)
    if values.shape != (len(trace.layer_ids), count) or len(set(trace.layer_ids)) != len(trace.layer_ids):
        raise ContractError("layer/token log-probability shape or layer ids are invalid")
    if values.shape[0] < 2:
        raise ContractError("mechanism trace needs at least two decoder layers")
    if not np.isfinite(values).all() or bool((values > 1e-6).any()):
        raise ContractError("gold log probabilities must be finite and <= 0")
    return values


def _matched_indices(source_indices: Sequence[int], anchors: Sequence[int], source_length: int, anchor_length: int) -> list[int]:
    if len(source_indices) < len(anchors) or not anchors:
        raise RowExclusion("insufficient tokens for an exact-count matched signal")
    remaining = set(int(value) for value in source_indices)
    chosen: list[int] = []
    for anchor in sorted(anchors):
        anchor_position = (anchor + 0.5) / anchor_length
        selected = min(
            remaining,
            key=lambda candidate: (abs((candidate + 0.5) / source_length - anchor_position), candidate),
        )
        chosen.append(selected)
        remaining.remove(selected)
    return sorted(chosen)


def _mean_by_layer(values: np.ndarray, indices: Sequence[int]) -> list[float]:
    return values[:, list(indices)].mean(axis=1).astype(float).tolist()


def compute_signals(
    row: dict[str, Any],
    parent_image: TeacherForcedTrace,
    child_image: TeacherForcedTrace,
    parent_text: TeacherForcedTrace,
    child_text: TeacherForcedTrace,
    expected_image_sha256: str,
) -> dict[str, Any]:
    question, parent, child = row["question"], row["parent_target"], row["child_target"]
    pi = _validate_trace(parent_image, target=parent, question=question, condition="image", expected_image_sha256=expected_image_sha256)
    ci = _validate_trace(child_image, target=child, question=question, condition="image", expected_image_sha256=expected_image_sha256)
    pt = _validate_trace(parent_text, target=parent, question=question, condition="text_only", expected_image_sha256=None)
    ct = _validate_trace(child_text, target=child, question=question, condition="text_only", expected_image_sha256=None)
    traces = (parent_image, child_image, parent_text, child_text)
    if any(trace.layer_ids != parent_image.layer_ids for trace in traces[1:]):
        raise ContractError("image/text and parent/child layer ids differ")
    if parent_image.token_ids != parent_text.token_ids or parent_image.token_offsets != parent_text.token_offsets:
        raise ContractError("parent image/text-only response tokenization differs")
    if child_image.token_ids != child_text.token_ids or child_image.token_offsets != child_text.token_offsets:
        raise ContractError("child image/text-only response tokenization differs")
    if len({trace.template_id for trace in traces}) != 1:
        raise ContractError("image/text-only or parent/child teacher-forcing templates differ")

    constraint = map_constraint_spans(
        child,
        row["constraint_char_spans_in_child"],
        child_image.token_offsets,
        child_image.offset_unit,
    )
    child_nonconstraint = [index for index in range(len(child_image.token_ids)) if index not in set(constraint)]
    parent_matched = _matched_indices(
        range(len(parent_image.token_ids)), constraint, len(parent_image.token_ids), len(child_image.token_ids)
    )
    child_matched = _matched_indices(
        child_nonconstraint, constraint, len(child_image.token_ids), len(child_image.token_ids)
    )
    constraint_logp = np.asarray(_mean_by_layer(ci, constraint))
    parent_logp = np.asarray(_mean_by_layer(pi, range(pi.shape[1])))
    parent_matched_logp = np.asarray(_mean_by_layer(pi, parent_matched))
    child_matched_logp = np.asarray(_mean_by_layer(ci, child_matched))
    child_logp = np.asarray(_mean_by_layer(ci, range(ci.shape[1])))
    final = -1
    return {
        "layer_ids": parent_image.layer_ids,
        "token_counts": {
            "constraint": len(constraint),
            "parent_target": len(parent_image.token_ids),
            "child_target": len(child_image.token_ids),
            "matched_parent": len(parent_matched),
            "matched_child_nonconstraint": len(child_matched),
        },
        "token_indices": {
            "constraint_in_child": constraint,
            "matched_parent": parent_matched,
            "matched_child_nonconstraint": child_matched,
        },
        "image_layer_signals": {
            "constraint_logp": constraint_logp.tolist(),
            "parent_sequence_logp": parent_logp.tolist(),
            "child_sequence_logp": child_logp.tolist(),
            "matched_parent_logp": parent_matched_logp.tolist(),
            "matched_child_nonconstraint_logp": child_matched_logp.tolist(),
            "constraint_minus_parent_sequence": (constraint_logp - parent_logp).tolist(),
            "constraint_minus_matched_parent": (constraint_logp - parent_matched_logp).tolist(),
            "constraint_minus_matched_child": (constraint_logp - child_matched_logp).tolist(),
            "child_minus_parent_sequence": (child_logp - parent_logp).tolist(),
        },
        "text_only_nuisance": {
            "constraint_mean_nll": float(-ct[final, constraint].mean()),
            "parent_sequence_mean_nll": float(-pt[final].mean()),
            "child_sequence_mean_nll": float(-ct[final].mean()),
            "matched_parent_mean_nll": float(-pt[final, parent_matched].mean()),
            "matched_child_nonconstraint_mean_nll": float(-ct[final, child_matched].mean()),
            "lexical_proxy_only_not_clinical_evidence": True,
        },
        "trace_provenance": {
            name: {
                "serialized_input_sha256": trace.serialized_input_sha256,
                "template_id": trace.template_id,
                "offset_unit": trace.offset_unit,
                "target_token_ids_sha256": _sha256_bytes(_canonical(trace.token_ids)),
            }
            for name, trace in (
                ("parent_image", parent_image),
                ("child_image", child_image),
                ("parent_text_only", parent_text),
                ("child_text_only", child_text),
            )
        },
    }


def load_admitted_manifest(manifest: Path, metadata: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not manifest.is_file() or not metadata.is_file():
        raise ContractError("physician-admitted manifest and inseparable metadata are required")
    meta = json.loads(metadata.read_text())
    if meta.get("manifest_protocol_id") != MANIFEST_PROTOCOL_ID or meta.get("status") != "physician_admitted":
        raise ContractError("metadata is not a physician-admitted Specificity Ratchet manifest")
    if meta.get("image_disjoint") is not True:
        raise ContractError("manifest metadata does not certify image-disjoint splits")
    actual_hash = _sha256_file(manifest)
    if meta.get("manifest_sha256") != actual_hash:
        raise ContractError("manifest/metadata SHA-256 mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != meta.get("n_scientific_edges"):
        raise ContractError("empty or count-mismatched physician manifest")
    sample_ids: set[str] = set()
    case_splits: dict[str, str] = {}
    required = {
        "sample_id", "case_id", "edge_id", "image_relpath", "question", "parent_target",
        "child_target", "constraint_char_spans_in_child", "scientific_role", "split", "edge_type",
    }
    for row in rows:
        if row.get("manifest_protocol_id") != MANIFEST_PROTOCOL_ID or required - row.keys():
            raise ContractError("manifest row schema/protocol mismatch")
        if row["sample_id"] in sample_ids:
            raise ContractError("duplicate sample_id")
        sample_ids.add(row["sample_id"])
        previous = case_splits.setdefault(row["case_id"], row["split"])
        if previous != row["split"] or row["split"] not in {"dev", "test"}:
            raise ContractError("case leakage or invalid split")
    return rows, meta


def _row_key(row: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical(row))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _resolve_image(image_root: Path, relative: str) -> Path:
    root = image_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"admitted image escapes --image-root: {relative}") from exc
    return candidate


def _validate_shard(path: Path, *, config_fingerprint: str, row_sha256: str) -> dict[str, Any]:
    try:
        shard = json.loads(path.read_text())
    except Exception as exc:
        raise ContractError(f"corrupt resume shard {path}: {exc}") from exc
    if shard.get("runtime_protocol_id") != RUNTIME_PROTOCOL_ID:
        raise ContractError(f"wrong shard protocol: {path}")
    if shard.get("config_fingerprint") != config_fingerprint or shard.get("row_sha256") != row_sha256:
        raise ContractError(f"resume shard fingerprint drift: {path}")
    if shard.get("payload_sha256") != _sha256_bytes(_canonical(shard.get("payload"))):
        raise ContractError(f"resume shard payload checksum mismatch: {path}")
    return shard


def _cyclic_derangement(rows: Sequence[dict[str, Any]], seed_key: str, *, require_role_change: bool) -> dict[str, str]:
    if len(rows) < 2:
        return {}
    ordered = sorted(rows, key=lambda row: _sha256_bytes(f"{seed_key}|{row['sample_id']}".encode()))
    for shift in range(1, len(ordered)):
        mapping: dict[str, str] = {}
        valid = True
        for index, target in enumerate(ordered):
            source = ordered[(index + shift) % len(ordered)]
            if source["case_id"] == target["case_id"] or source["sample_id"] == target["sample_id"]:
                valid = False
                break
            if require_role_change and source["scientific_role"] == target["scientific_role"]:
                valid = False
                break
            mapping[target["sample_id"]] = source["sample_id"]
        if valid:
            return mapping
    return {}


def build_control_plan(payloads: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Freeze exact-bin shuffled-pair and sequence-length permutations.

    Sparse bins are reported as ineligible; bins are never widened post hoc.
    """

    from collections import defaultdict

    pair_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    length_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        if payload.get("status") != "ok":
            continue
        counts = payload["signals"]["token_counts"]
        pair_groups[(payload["split"], payload["edge_type"], counts["parent_target"], counts["constraint"])].append(payload)
        length_groups[(payload["split"], counts["parent_target"], counts["child_target"], counts["constraint"])].append(payload)
    shuffled: dict[str, str] = {}
    length_permutation: dict[str, str] = {}
    pair_eligible_bins = 0
    length_eligible_bins = 0
    for key, group in sorted(pair_groups.items(), key=lambda item: repr(item[0])):
        mapping = _cyclic_derangement(group, f"pair|{seed}|{key!r}", require_role_change=False)
        if mapping:
            pair_eligible_bins += 1
            shuffled.update(mapping)
    for key, group in sorted(length_groups.items(), key=lambda item: repr(item[0])):
        mapping = _cyclic_derangement(group, f"length|{seed}|{key!r}", require_role_change=True)
        if mapping:
            length_eligible_bins += 1
            length_permutation.update(mapping)
    total = sum(payload.get("status") == "ok" for payload in payloads)
    return {
        "protocol": "specificity-ratchet-negative-controls-v1",
        "seed": seed,
        "shuffled_parent_pairing": {
            "exact_bin_fields": ["split", "edge_type", "parent_target_token_count", "constraint_token_count"],
            "different_case_required": True,
            "mapping_target_to_source": dict(sorted(shuffled.items())),
            "eligible_rows": len(shuffled),
            "coverage": len(shuffled) / total if total else 0.0,
            "eligible_bins": pair_eligible_bins,
            "no_caliper_relaxation": True,
        },
        "sequence_length_role_permutation": {
            "exact_bin_fields": ["split", "parent_target_token_count", "child_target_token_count", "constraint_token_count"],
            "different_case_and_role_required": True,
            "mapping_target_to_role_source": dict(sorted(length_permutation.items())),
            "eligible_rows": len(length_permutation),
            "coverage": len(length_permutation) / total if total else 0.0,
            "eligible_bins": length_eligible_bins,
            "no_caliper_relaxation": True,
        },
        "interpretation": "Ineligible exact bins reduce control coverage; they never justify wider post-hoc bins.",
    }


def run_runtime(
    *,
    manifest: Path,
    metadata: Path,
    image_root: Path,
    output_dir: Path,
    adapter: TeacherForcingAdapter,
    split: str = "dev",
    seed: int = 20260802,
    command: Sequence[str] | None = None,
    _historical_contract_test_only: bool = False,
) -> dict[str, Any]:
    if not _historical_contract_test_only:
        raise ContractError(
            "isolated parent/child runtime is F6-rejected; use full-visible-answer replay; "
            f"decision={F6_REJECTION_ARTIFACT}"
        )
    rows, meta = load_admitted_manifest(manifest, metadata)
    if split not in {"dev", "test", "all"}:
        raise ContractError("split must be dev, test, or all")
    selected = [row for row in rows if split == "all" or row["split"] == split]
    if not selected:
        raise ContractError(f"physician manifest has no rows for split={split}")
    selected.sort(key=lambda row: row["sample_id"])
    adapter_fingerprint = adapter.fingerprint()
    if not isinstance(adapter_fingerprint, dict) or not adapter_fingerprint:
        raise ContractError("adapter fingerprint must be a non-empty JSON object")
    try:
        _canonical(adapter_fingerprint)
    except (TypeError, ValueError) as exc:
        raise ContractError("adapter fingerprint is not canonical-JSON serializable") from exc
    source_sha = _sha256_file(Path(__file__))
    config = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": meta["manifest_sha256"],
        "metadata_path": str(metadata.resolve()),
        "metadata_sha256": _sha256_file(metadata),
        "image_root": str(image_root.resolve()),
        "adapter_fingerprint": adapter_fingerprint,
        "split": split,
        "seed": seed,
        "command": list(command or []),
        "runtime_source_sha256": source_sha,
        "offset_policy": "exact-contextual-offsets; whitespace-only spill; no sentence fallback",
        "text_only_policy": "identical response ids/offsets/template; final-layer NLL is nuisance only",
    }
    config_fingerprint = _sha256_bytes(_canonical(config))
    config["config_fingerprint"] = config_fingerprint
    ordered = {
        "sample_ids": [row["sample_id"] for row in selected],
        "sample_ids_sha256": _sha256_bytes(_canonical([row["sample_id"] for row in selected])),
    }
    _write_once_or_equal(output_dir / "config.json", (json.dumps(config, indent=2, sort_keys=True) + "\n").encode())
    _write_once_or_equal(output_dir / "ordered_keys.json", (json.dumps(ordered, indent=2, sort_keys=True) + "\n").encode())

    payloads: list[dict[str, Any]] = []
    resumed = 0
    for index, row in enumerate(selected):
        row_sha = _row_key(row)
        shard_path = output_dir / "shards" / f"{index:06d}-{_safe_name(row['sample_id'])}.json"
        if shard_path.exists():
            shard = _validate_shard(shard_path, config_fingerprint=config_fingerprint, row_sha256=row_sha)
            payloads.append(shard["payload"])
            resumed += 1
            continue
        image_path = _resolve_image(image_root, row["image_relpath"])
        if not image_path.is_file():
            raise ContractError(f"missing admitted image: {image_path}")
        image_sha = _sha256_file(image_path)
        traces = {
            "parent_image": adapter.score(image_path=image_path, question=row["question"], target=row["parent_target"], condition="image"),
            "child_image": adapter.score(image_path=image_path, question=row["question"], target=row["child_target"], condition="image"),
            "parent_text": adapter.score(image_path=None, question=row["question"], target=row["parent_target"], condition="text_only"),
            "child_text": adapter.score(image_path=None, question=row["question"], target=row["child_target"], condition="text_only"),
        }
        base_payload = {
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "edge_id": row["edge_id"],
            "split": row["split"],
            "scientific_role": row["scientific_role"],
            "edge_type": row["edge_type"],
            "modality_stratum": row.get("modality_stratum"),
            "anatomy_stratum": row.get("anatomy_stratum"),
            "prompt_requested_increment": row.get("prompt_requested_increment"),
            "image_sha256": image_sha,
        }
        try:
            signals = compute_signals(row, traces["parent_image"], traces["child_image"], traces["parent_text"], traces["child_text"], image_sha)
            payload = {**base_payload, "status": "ok", "signals": signals}
        except RowExclusion as exc:
            payload = {
                **base_payload,
                "status": "excluded",
                "exclusion_stage": "exact_token_mapping_or_matched_signal",
                "exclusion_reason": str(exc),
                "no_fallback_used": True,
            }
        shard = {
            "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
            "config_fingerprint": config_fingerprint,
            "row_sha256": row_sha,
            "payload_sha256": _sha256_bytes(_canonical(payload)),
            "payload": payload,
        }
        _atomic_write(shard_path, (json.dumps(shard, indent=2, sort_keys=True) + "\n").encode())
        payloads.append(payload)
    controls = build_control_plan(payloads, seed)
    _write_once_or_equal(output_dir / "controls.json", (json.dumps(controls, indent=2, sort_keys=True) + "\n").encode())
    ok_rows = sum(payload.get("status") == "ok" for payload in payloads)
    excluded_rows = len(payloads) - ok_rows
    if ok_rows == 0:
        raise ContractError("all admitted rows failed exact token mapping; no COMPLETE artifact written")
    completion_file = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "status": "complete",
        "config_fingerprint": config_fingerprint,
        "rows": len(payloads),
        "analyzable_rows": ok_rows,
        "excluded_rows": excluded_rows,
        "control_coverage": {
            "shuffled_parent": controls["shuffled_parent_pairing"]["coverage"],
            "length_permutation": controls["sequence_length_role_permutation"]["coverage"],
        },
    }
    _write_once_or_equal(output_dir / "COMPLETE.json", (json.dumps(completion_file, indent=2, sort_keys=True) + "\n").encode())
    return {**completion_file, "resumed_rows": resumed}


def _load_factory(specification: str, config: dict[str, Any]) -> TeacherForcingAdapter:
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
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--adapter-factory", help="Audited module:function; required unless --preflight-only")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        raise ContractError(
            "this CLI is F6-rejected and cannot authorize scoring; "
            f"use full-visible-answer replay; decision={F6_REJECTION_ARTIFACT}"
        )
        rows, _ = load_admitted_manifest(args.manifest, args.metadata)
        selected = [row for row in rows if args.split == "all" or row["split"] == args.split]
        resolved = [_resolve_image(args.image_root, row["image_relpath"]) for row in selected]
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise ContractError(f"{len(missing)} admitted images are missing; first={missing[0]}")
        if args.preflight_only:
            print(json.dumps({"status": "preflight_passed", "rows": len(selected), "gpu_started": False}, indent=2))
            return
        if not args.adapter_factory:
            raise ContractError("an audited --adapter-factory is required; Huatuo/Hulu are not guessed")
        adapter_config = json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
        adapter = _load_factory(args.adapter_factory, adapter_config)
        result = run_runtime(
            manifest=args.manifest,
            metadata=args.metadata,
            image_root=args.image_root,
            output_dir=args.output_dir,
            adapter=adapter,
            split=args.split,
            seed=args.seed,
            command=[shlex.join(sys.argv)],
        )
    except (ContractError, OSError, ValueError, ImportError, AttributeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
