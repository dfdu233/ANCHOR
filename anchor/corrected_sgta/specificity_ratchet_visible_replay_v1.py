#!/usr/bin/env python3
"""Case-cached full-visible-answer runtime for Specificity Ratchet.

Scientific execution requires a direct native-generation capture for every
selected case.  The capture must prove exact visible-text reproduction and
identity between native generation IDs and the contextual teacher-forcing IDs.
Text-only traces are retained only as lexical sensitivity; the primary visual
contrast is own image versus two frozen matched image swaps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from corrected_sgta.compile_specificity_ratchet_replay_manifest_v1 import (
    PROTOCOL_ID as MANIFEST_PROTOCOL_ID,
    SWAPS_PER_CASE,
    TARGET_MODEL_FAMILY,
)
from corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    ContractError,
    TeacherForcedTrace,
    TeacherForcingAdapter,
    _atomic_write,
    _canonical,
    _matched_indices,
    _resolve_image,
    _safe_name,
    _sha256_bytes,
    _sha256_file,
    _validate_trace,
    _write_once_or_equal,
    map_constraint_spans,
)


RUNTIME_PROTOCOL_ID = "specificity-ratchet-visible-replay-runtime-v1"
CAPTURE_PROTOCOL_ID = "huatuo-specificity-native-capture-v1"


def load_replay_manifest(
    manifest: Path, metadata: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not manifest.is_file() or not metadata.is_file():
        raise ContractError("visible-replay manifest and metadata are required")
    meta = json.loads(metadata.read_text())
    if (
        meta.get("manifest_protocol_id") != MANIFEST_PROTOCOL_ID
        or meta.get("status") != "physician_admitted_visible_answer_replay"
    ):
        raise ContractError("wrong or non-admitted visible-replay metadata")
    if meta.get("target_model_family") != TARGET_MODEL_FAMILY:
        raise ContractError("visible-replay manifest targets the wrong model family")
    if meta.get("native_capture_required_before_scientific_runtime") is not True:
        raise ContractError("manifest does not require native generation capture")
    if meta.get("manifest_sha256") != _sha256_file(manifest):
        raise ContractError("visible-replay manifest/metadata hash mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != meta.get("n_scientific_edges"):
        raise ContractError("visible-replay manifest is empty or count-mismatched")
    sample_ids: set[str] = set()
    case_contract: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        if row.get("manifest_protocol_id") != MANIFEST_PROTOCOL_ID:
            raise ContractError("visible-replay row protocol mismatch")
        if row.get("target_model_family") != TARGET_MODEL_FAMILY:
            raise ContractError("Huatuo replay cannot contain another model's target")
        if row.get("model_input_contract") != "complete frozen visible OE answer only":
            raise ContractError("isolated parent/child target leaked into replay manifest")
        if row.get("native_generation_ids_certified") is not False:
            raise ContractError("manifest must not self-certify native generation IDs")
        if row["sample_id"] in sample_ids:
            raise ContractError("duplicate replay sample ID")
        sample_ids.add(row["sample_id"])
        swaps = row.get("matched_image_swaps")
        if not isinstance(swaps, list) or len(swaps) != SWAPS_PER_CASE:
            raise ContractError("every replay row needs exactly two frozen swaps")
        if len({swap.get("case_id") for swap in swaps}) != SWAPS_PER_CASE:
            raise ContractError("replay swaps are not distinct")
        if row["case_id"] in {swap.get("case_id") for swap in swaps}:
            raise ContractError("target case appears in its own swap set")
        contract = (
            row["split"],
            row["image_relpath"],
            row["question"],
            row["full_visible_answer_sha256"],
            _sha256_bytes(_canonical(swaps)),
        )
        prior = case_contract.setdefault(row["case_id"], contract)
        if prior != contract:
            raise ContractError("case-level replay target or swap plan is inconsistent")
    split_by_case: dict[str, str] = {}
    for row in rows:
        prior = split_by_case.setdefault(row["case_id"], row["split"])
        if prior != row["split"] or row["split"] not in {"dev", "test"}:
            raise ContractError("case leakage or invalid replay split")
    return rows, meta


def load_native_capture(
    path: Path,
    *,
    manifest_sha256: str,
    metadata_sha256: str,
    adapter_fingerprint: dict[str, Any],
    selected_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ContractError("direct native-generation capture is required")
    payload = json.loads(path.read_text())
    if payload.get("capture_protocol_id") != CAPTURE_PROTOCOL_ID:
        raise ContractError("wrong native-capture protocol")
    if payload.get("status") != "complete_passed":
        raise ContractError("native capture is partial or contains identity failures")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ContractError("native capture belongs to another replay manifest")
    if payload.get("metadata_sha256") != metadata_sha256:
        raise ContractError("native capture belongs to different replay metadata")
    if payload.get("target_model_family") != TARGET_MODEL_FAMILY:
        raise ContractError("native capture targets another model family")
    if payload.get("adapter_fingerprint") != adapter_fingerprint:
        raise ContractError("native capture/scoring adapter fingerprint mismatch")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ContractError("native capture has no case list")
    by_case = {str(row.get("case_id")): row for row in cases}
    if len(by_case) != len(cases) or "" in by_case:
        raise ContractError("native capture case IDs are empty or duplicated")
    expected = {row["case_id"] for row in selected_rows}
    missing = sorted(expected - set(by_case))
    if missing:
        raise ContractError(f"native capture misses {len(missing)} selected cases")
    for row in selected_rows:
        capture = by_case[row["case_id"]]
        if capture.get("source_question_id") != row["source_question_id"]:
            raise ContractError(f"{row['case_id']}: capture question identity mismatch")
        if capture.get("frozen_visible_answer_sha256") != row["full_visible_answer_sha256"]:
            raise ContractError(f"{row['case_id']}: capture answer identity mismatch")
        if capture.get("decoded_text_exact_frozen_match") is not True:
            raise ContractError(f"{row['case_id']}: native decoded text did not reproduce")
        if capture.get("directly_captured_output_sequences") is not True:
            raise ContractError(f"{row['case_id']}: output.sequences was not captured directly")
        if capture.get("native_ids_equal_contextual_target_ids") is not True:
            raise ContractError(f"{row['case_id']}: native/contextual token IDs differ")
        counts = capture.get("visual_token_counts_own_swap1_swap2")
        if (
            capture.get("visual_token_count_equal_across_own_swaps") is not True
            or not isinstance(counts, list)
            or len(counts) != SWAPS_PER_CASE + 1
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts)
            or len(set(counts)) != 1
        ):
            raise ContractError(f"{row['case_id']}: own/swap visual token lengths differ")
        native_ids = capture.get("native_generation_token_ids")
        if (
            not isinstance(native_ids, list)
            or not native_ids
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in native_ids)
        ):
            raise ContractError(f"{row['case_id']}: invalid native token IDs")
        if capture.get("native_generation_token_ids_sha256") != _sha256_bytes(
            _canonical(native_ids)
        ):
            raise ContractError(f"{row['case_id']}: native token-ID hash mismatch")
    return by_case


def _mean(values: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return values[:, list(indices)].mean(axis=1)


def compute_replay_signals(
    row: dict[str, Any],
    *,
    own: TeacherForcedTrace,
    swaps: list[TeacherForcedTrace],
    text_only: TeacherForcedTrace,
    own_image_sha256: str,
    swap_image_sha256: list[str],
    native_token_ids: list[int],
) -> dict[str, Any]:
    target, question = row["full_visible_answer"], row["question"]
    own_values = _validate_trace(
        own,
        target=target,
        question=question,
        condition="image",
        expected_image_sha256=own_image_sha256,
    )
    swap_values = [
        _validate_trace(
            trace,
            target=target,
            question=question,
            condition="image",
            expected_image_sha256=image_hash,
        )
        for trace, image_hash in zip(swaps, swap_image_sha256)
    ]
    text_values = _validate_trace(
        text_only,
        target=target,
        question=question,
        condition="text_only",
        expected_image_sha256=None,
    )
    traces = [own, *swaps, text_only]
    if any(trace.layer_ids != own.layer_ids for trace in traces[1:]):
        raise ContractError("own/swap/text layer identities differ")
    if any(
        trace.token_ids != own.token_ids or trace.token_offsets != own.token_offsets
        for trace in traces[1:]
    ):
        raise ContractError("own/swap/text contextual target tokenization differs")
    if len({trace.template_id for trace in traces}) != 1:
        raise ContractError("own/swap/text teacher-forcing templates differ")
    if own.token_ids != native_token_ids:
        raise ContractError("contextual teacher-forcing IDs differ from direct native IDs")

    constraint = map_constraint_spans(
        target,
        row["constraint_char_spans_in_full_answer"],
        own.token_offsets,
        own.offset_unit,
    )
    child = map_constraint_spans(
        target,
        [row["child_char_span_in_full_answer"]],
        own.token_offsets,
        own.offset_unit,
    )
    child_nonconstraint = sorted(set(child) - set(constraint))
    matched = _matched_indices(
        child_nonconstraint,
        constraint,
        len(own.token_ids),
        len(own.token_ids),
    )
    own_constraint = _mean(own_values, constraint)
    own_matched = _mean(own_values, matched)
    per_swap_constraint = [_mean(values, constraint) for values in swap_values]
    per_swap_matched = [_mean(values, matched) for values in swap_values]
    mean_swap_constraint = np.mean(per_swap_constraint, axis=0)
    mean_swap_matched = np.mean(per_swap_matched, axis=0)
    text_constraint = _mean(text_values, constraint)
    text_matched = _mean(text_values, matched)
    own_swap_constraint = own_constraint - mean_swap_constraint
    own_swap_matched = own_matched - mean_swap_matched
    primary_did = own_swap_constraint - own_swap_matched
    text_constraint_residual = own_constraint - text_constraint
    text_matched_residual = own_matched - text_matched
    per_swap_did = [
        ((own_constraint - constraint_values) - (own_matched - matched_values)).tolist()
        for constraint_values, matched_values in zip(
            per_swap_constraint, per_swap_matched
        )
    ]
    return {
        "layer_ids": list(own.layer_ids),
        "token_counts": {
            "full_visible_answer": len(own.token_ids),
            "child_surface": len(child),
            "constraint": len(constraint),
            "matched_child_nonconstraint": len(matched),
        },
        "token_indices": {
            "constraint": constraint,
            "child_surface": child,
            "matched_child_nonconstraint": matched,
        },
        "primary_own_minus_matched_swaps": {
            "constraint": own_swap_constraint.tolist(),
            "matched_nonconstraint": own_swap_matched.tolist(),
            "constraint_minus_matched_difference_in_differences": primary_did.tolist(),
            "per_swap_difference_in_differences": per_swap_did,
        },
        "raw_commitment": {
            "own_constraint_logp": own_constraint.tolist(),
            "own_matched_nonconstraint_logp": own_matched.tolist(),
            "constraint_minus_matched": (own_constraint - own_matched).tolist(),
            "mean_swap_constraint_logp": mean_swap_constraint.tolist(),
            "mean_swap_matched_nonconstraint_logp": mean_swap_matched.tolist(),
        },
        "text_only_secondary": {
            "text_constraint_logp": text_constraint.tolist(),
            "text_matched_nonconstraint_logp": text_matched.tolist(),
            "own_minus_text_constraint": text_constraint_residual.tolist(),
            "own_minus_text_matched_nonconstraint": text_matched_residual.tolist(),
            "difference_in_differences": (
                text_constraint_residual - text_matched_residual
            ).tolist(),
            "not_primary_visual_evidence": True,
        },
        "trace_provenance": {
            "own": own.serialized_input_sha256,
            "swaps": [trace.serialized_input_sha256 for trace in swaps],
            "text_only": text_only.serialized_input_sha256,
            "native_generation_token_ids_sha256": _sha256_bytes(
                _canonical(native_token_ids)
            ),
            "template_id": own.template_id,
        },
    }


def _validate_shard(
    path: Path, *, config_fingerprint: str, row_sha256: str
) -> dict[str, Any]:
    try:
        shard = json.loads(path.read_text())
    except Exception as exc:
        raise ContractError(f"corrupt replay shard {path}: {exc}") from exc
    if shard.get("runtime_protocol_id") != RUNTIME_PROTOCOL_ID:
        raise ContractError(f"wrong replay shard protocol: {path}")
    if (
        shard.get("config_fingerprint") != config_fingerprint
        or shard.get("row_sha256") != row_sha256
    ):
        raise ContractError(f"replay resume fingerprint drift: {path}")
    if shard.get("payload_sha256") != _sha256_bytes(_canonical(shard.get("payload"))):
        raise ContractError(f"replay shard payload checksum mismatch: {path}")
    return shard


def run_runtime(
    *,
    manifest: Path,
    metadata: Path,
    native_capture: Path,
    image_root: Path,
    output_dir: Path,
    adapter: TeacherForcingAdapter,
    split: str = "dev",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows, meta = load_replay_manifest(manifest, metadata)
    if split not in {"dev", "test", "all"}:
        raise ContractError("split must be dev, test, or all")
    selected = [row for row in rows if split == "all" or row["split"] == split]
    if not selected:
        raise ContractError(f"replay manifest has no rows for split={split}")
    selected.sort(key=lambda row: row["sample_id"])
    adapter_fingerprint = adapter.fingerprint()
    if adapter_fingerprint.get("model_family") != TARGET_MODEL_FAMILY:
        raise ContractError("Huatuo replay refuses another model-family adapter")
    captures = load_native_capture(
        native_capture,
        manifest_sha256=meta["manifest_sha256"],
        metadata_sha256=_sha256_file(metadata),
        adapter_fingerprint=adapter_fingerprint,
        selected_rows=selected,
    )
    config = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": meta["manifest_sha256"],
        "metadata": str(metadata.resolve()),
        "metadata_sha256": _sha256_file(metadata),
        "native_capture": str(native_capture.resolve()),
        "native_capture_sha256": _sha256_file(native_capture),
        "image_root": str(image_root.resolve()),
        "adapter_fingerprint": adapter_fingerprint,
        "split": split,
        "command": list(command or []),
        "runtime_source_sha256": _sha256_file(Path(__file__).resolve()),
        "primary_control": "own image minus mean of two frozen exact-cell matched swaps",
        "text_only_role": "secondary lexical sensitivity only",
    }
    config_fingerprint = _sha256_bytes(_canonical(config))
    config["config_fingerprint"] = config_fingerprint
    _write_once_or_equal(
        output_dir / "config.json",
        (json.dumps(config, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_once_or_equal(
        output_dir / "ordered_keys.json",
        (
            json.dumps(
                {"sample_ids": [row["sample_id"] for row in selected]},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_case[row["case_id"]].append(row)
    payloads: list[dict[str, Any]] = []
    resumed = 0
    scored_cases = 0
    for case_id, case_rows in sorted(by_case.items()):
        missing_rows: list[tuple[dict[str, Any], Path, str]] = []
        for row in sorted(case_rows, key=lambda item: item["sample_id"]):
            row_sha = _sha256_bytes(_canonical(row))
            shard_path = output_dir / "shards" / f"{_safe_name(row['sample_id'])}.json"
            if shard_path.exists():
                payloads.append(
                    _validate_shard(
                        shard_path,
                        config_fingerprint=config_fingerprint,
                        row_sha256=row_sha,
                    )["payload"]
                )
                resumed += 1
            else:
                missing_rows.append((row, shard_path, row_sha))
        if not missing_rows:
            continue
        exemplar = case_rows[0]
        own_path = _resolve_image(image_root, exemplar["image_relpath"])
        swap_paths = [
            _resolve_image(image_root, swap["image_relpath"])
            for swap in exemplar["matched_image_swaps"]
        ]
        for path in [own_path, *swap_paths]:
            if not path.is_file():
                raise ContractError(f"missing replay image: {path}")
        target, question = exemplar["full_visible_answer"], exemplar["question"]
        own = adapter.score(
            image_path=own_path,
            question=question,
            target=target,
            condition="image",
        )
        swaps = [
            adapter.score(
                image_path=path,
                question=question,
                target=target,
                condition="image",
            )
            for path in swap_paths
        ]
        text_only = adapter.score(
            image_path=None,
            question=question,
            target=target,
            condition="text_only",
        )
        scored_cases += 1
        own_hash = _sha256_file(own_path)
        swap_hashes = [_sha256_file(path) for path in swap_paths]
        native_ids = captures[case_id]["native_generation_token_ids"]
        for row, shard_path, row_sha in missing_rows:
            signals = compute_replay_signals(
                row,
                own=own,
                swaps=swaps,
                text_only=text_only,
                own_image_sha256=own_hash,
                swap_image_sha256=swap_hashes,
                native_token_ids=native_ids,
            )
            payload = {
                "sample_id": row["sample_id"],
                "case_id": case_id,
                "edge_id": row["edge_id"],
                "split": row["split"],
                "scientific_role": row["scientific_role"],
                "edge_type": row["edge_type"],
                "modality_stratum": row["modality_stratum"],
                "anatomy_stratum": row["anatomy_stratum"],
                "prompt_requested_increment": row["prompt_requested_increment"],
                "constraint_lexical_key_sha256": _sha256_bytes(
                    _canonical(
                        [
                            str(span["text"]).strip().casefold()
                            for span in row["constraint_char_spans_in_full_answer"]
                        ]
                    )
                ),
                "status": "ok",
                "signals": signals,
            }
            shard = {
                "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
                "config_fingerprint": config_fingerprint,
                "row_sha256": row_sha,
                "payload_sha256": _sha256_bytes(_canonical(payload)),
                "payload": payload,
            }
            _atomic_write(
                shard_path,
                (json.dumps(shard, indent=2, sort_keys=True) + "\n").encode(),
            )
            payloads.append(payload)
    payloads.sort(key=lambda row: row["sample_id"])
    completion_file = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "status": "complete",
        "rows": len(payloads),
        "cases": len(by_case),
        "config_fingerprint": config_fingerprint,
        "native_capture_enforced": True,
        "primary_swap_count": SWAPS_PER_CASE,
    }
    _write_once_or_equal(
        output_dir / "COMPLETE.json",
        (json.dumps(completion_file, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        **completion_file,
        "scored_cases_this_invocation": scored_cases,
        "resumed_rows": resumed,
    }


def _load_factory(specification: str, config: dict[str, Any]) -> TeacherForcingAdapter:
    if ":" not in specification:
        raise ContractError("--adapter-factory must be module:function")
    module_name, function_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--native-capture", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--adapter-factory")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        rows, _ = load_replay_manifest(args.manifest, args.metadata)
        selected = [row for row in rows if args.split == "all" or row["split"] == args.split]
        paths = []
        for row in selected:
            paths.append(_resolve_image(args.image_root, row["image_relpath"]))
            paths.extend(
                _resolve_image(args.image_root, swap["image_relpath"])
                for swap in row["matched_image_swaps"]
            )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ContractError(f"replay preflight misses images; first={missing[0]}")
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "rows": len(selected),
                        "native_capture_not_yet_validated_without_adapter": True,
                        "gpu_started": False,
                    },
                    indent=2,
                )
            )
            return
        if not args.adapter_factory:
            raise ContractError("audited Huatuo adapter factory is required")
        adapter_config = (
            json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
        )
        adapter = _load_factory(args.adapter_factory, adapter_config)
        result = run_runtime(
            manifest=args.manifest,
            metadata=args.metadata,
            native_capture=args.native_capture,
            image_root=args.image_root,
            output_dir=args.output_dir,
            adapter=adapter,
            split=args.split,
            command=[shlex.join(sys.argv)],
        )
    except (ContractError, OSError, ValueError, ImportError, AttributeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
