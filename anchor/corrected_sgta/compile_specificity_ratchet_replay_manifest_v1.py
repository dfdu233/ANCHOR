#!/usr/bin/env python3
"""Compile physician-admitted full-visible-answer replay samples.

Unlike the superseded isolated-target manifest, this compiler never presents
an automatically shortened parent or child proposal to the model.  It binds an
admitted edge to the model's complete frozen OE answer, translates the exact
added-constraint span into that answer, and freezes two different-case image
swaps matched on split, modality, and anatomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from corrected_sgta.compile_specificity_ratchet_mechanism_manifest_v1 import (
    _scientific_role,
    exact_constraint_spans,
)
from corrected_sgta.huatuo_lockin_adapter_v1 import (
    ASSISTANT_SUFFIX,
    partition_answer_tokens,
)
from corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    RowExclusion,
    map_constraint_spans,
)
from corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
    AdjudicationValidationError,
    validate_adjudication,
)


PROTOCOL_ID = "specificity-ratchet-visible-replay-v1"
TARGET_MODEL_FAMILY = "huatuogpt-vision-7b"
SPLIT_SEED = "specificity-ratchet-visible-replay-split-v1"
SWAP_SEED = "specificity-ratchet-visible-replay-swaps-v1"
SWAPS_PER_CASE = 2
REQUIRED_PRIMARY_ROLES = {
    "supported_specificity_control",
    "causal_escalation_error",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _translated_span(span: dict[str, Any], offset: int) -> dict[str, Any]:
    return {
        **span,
        "char_start": int(span["char_start"]) + offset,
        "char_end_exclusive": int(span["char_end_exclusive"]) + offset,
    }


def _label_blind_case_pool(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Assign every frozen candidate image before reading physician outcomes."""

    case_info: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        info = {
            "case_id": candidate["case_id"],
            "image_relpath": candidate["image_relpath"],
            "modality_stratum": candidate["modality_stratum"],
            "anatomy_stratum": candidate["anatomy_stratum"],
        }
        prior = case_info.setdefault(candidate["case_id"], info)
        if prior != info:
            raise ValueError(f"{candidate['case_id']}: inconsistent frozen case strata")
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = {}
    for info in case_info.values():
        by_cell.setdefault(
            (info["modality_stratum"], info["anatomy_stratum"]), []
        ).append(info)
    output: list[dict[str, str]] = []
    for cell, items in sorted(by_cell.items()):
        ordered = sorted(
            items,
            key=lambda row: _sha_bytes(
                f"{SPLIT_SEED}|{cell!r}|{row['case_id']}".encode()
            ),
        )
        start = int(_sha_bytes(f"{SPLIT_SEED}|{cell!r}".encode()), 16) % 2
        for index, row in enumerate(ordered):
            output.append(
                {**row, "split": "dev" if (index + start) % 2 == 0 else "test"}
            )
    return output


def _case_swap_plan(
    rows: list[dict[str, Any]],
    pool_cases: list[dict[str, str]] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """Freeze exact-cell, same-split swaps without post-hoc relaxation."""

    case_info: dict[str, dict[str, str]] = {}
    for row in rows:
        info = {
            "case_id": row["case_id"],
            "image_relpath": row["image_relpath"],
            "split": row["split"],
            "modality_stratum": row["modality_stratum"],
            "anatomy_stratum": row["anatomy_stratum"],
        }
        prior = case_info.setdefault(row["case_id"], info)
        if prior != info:
            raise ValueError(f"{row['case_id']}: inconsistent case-level swap strata")
    pool = pool_cases or list(case_info.values())
    if len({row["case_id"] for row in pool}) != len(pool):
        raise ValueError("swap pool has duplicate case IDs")
    plan: dict[str, list[dict[str, str]]] = {}
    exclusions: list[dict[str, str]] = []
    for case_id, info in sorted(case_info.items()):
        eligible = [
            candidate
            for candidate in pool
            if candidate["case_id"] != case_id
            and candidate["split"] == info["split"]
            and candidate["modality_stratum"] == info["modality_stratum"]
            and candidate["anatomy_stratum"] == info["anatomy_stratum"]
        ]
        eligible.sort(
            key=lambda candidate: _sha_bytes(
                f"{SWAP_SEED}|{case_id}|{candidate['case_id']}".encode()
            )
        )
        if len(eligible) < SWAPS_PER_CASE:
            exclusions.append(
                {
                    "case_id": case_id,
                    "reason": "fewer_than_two_exact_cell_same_split_swap_cases",
                }
            )
            continue
        plan[case_id] = [
            {
                "case_id": candidate["case_id"],
                "image_relpath": candidate["image_relpath"],
            }
            for candidate in eligible[:SWAPS_PER_CASE]
        ]
    return plan, exclusions


def _require_g0_role_closure(rows: list[dict[str, Any]]) -> None:
    """Do not turn a one-sided human return into an authorized GPU canary."""
    missing: dict[str, list[str]] = {}
    for split in ("dev", "test"):
        observed = {row["scientific_role"] for row in rows if row["split"] == split}
        absent = sorted(REQUIRED_PRIMARY_ROLES - observed)
        if absent:
            missing[split] = absent
    if missing:
        raise ValueError(
            "G0 failed: supported-child controls and observable parent-only errors "
            f"must both survive in each frozen split; missing={missing}"
        )


def build_replay_rows(
    *,
    validated: Any,
    pack: Path,
    repo: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, str]]:
    candidate_rows = list(validated.candidates)
    candidates = {row["edge_id"]: row for row in candidate_rows}
    case_pool = _label_blind_case_pool(candidate_rows)
    case_split = {row["case_id"]: row["split"] for row in case_pool}
    provenance_path = pack / "provenance.private.jsonl"
    provenance_rows = _read_jsonl(provenance_path)
    provenance = {row["edge_id"]: row for row in provenance_rows}
    if len(provenance) != len(provenance_rows) or set(provenance) != set(candidates):
        raise ValueError("private provenance does not exactly cover frozen candidate edges")
    answer_paths = {row["source_answer_path"] for row in provenance_rows}
    if len(answer_paths) != 1:
        raise ValueError("replay manifest requires one frozen source-answer file")
    answer_path = (repo / next(iter(answer_paths))).resolve()
    source_rows = _read_jsonl(answer_path)
    source_by_qid = {row["question_id"]: row for row in source_rows}
    if len(source_by_qid) != len(source_rows):
        raise ValueError("source answers have duplicate question IDs")

    scientific: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for edge_id, candidate in candidates.items():
        final = validated.final_rows[edge_id]
        role, reason = _scientific_role(final)
        if role is None:
            exclusions.append(
                {"case_id": candidate["case_id"], "edge_id": edge_id, "reason": reason}
            )
            continue
        private = provenance[edge_id]
        if private.get("source_model") != "huatuo":
            raise ValueError(f"{edge_id}: non-Huatuo output cannot enter Huatuo replay")
        qid = str(private["question_id"])
        source = source_by_qid.get(qid)
        line = int(private["source_answer_line"])
        if (
            source is None
            or line < 1
            or line > len(source_rows)
            or source_rows[line - 1]["question_id"] != qid
        ):
            raise ValueError(f"{edge_id}: frozen source answer identity mismatch")
        if source.get("model_id") != "huatuo":
            raise ValueError(f"{edge_id}: source answer model identity mismatch")
        full_answer = str(source["text"])
        child = str(candidate["child_proposal"])
        if full_answer.count(child) != 1:
            raise ValueError(f"{edge_id}: child must be one exact full-answer substring")
        child_start = full_answer.index(child)
        visible_ids = tokenizer(full_answer, add_special_tokens=False).input_ids
        recorded_ids = source.get("metadata", {}).get("generated_token_ids")
        if visible_ids != recorded_ids:
            raise ValueError(f"{edge_id}: visible-text token provenance drift")
        encoded = tokenizer(
            full_answer + ASSISTANT_SUFFIX,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        mapping = partition_answer_tokens(
            answer_text=full_answer + ASSISTANT_SUFFIX,
            prefix="",
            continuation=full_answer,
            token_ids=encoded["input_ids"],
            offsets=encoded["offset_mapping"],
        )
        constraint_spans = [
            _translated_span(span, child_start)
            for span in exact_constraint_spans(candidate)
        ]
        child_span = {
            "char_start": child_start,
            "char_end_exclusive": child_start + len(child),
            "text": child,
            "utf8_sha256": _sha_bytes(child.encode()),
        }
        try:
            map_constraint_spans(
                full_answer,
                constraint_spans,
                mapping["continuation_token_offsets"],
                "unicode_character",
            )
            map_constraint_spans(
                full_answer,
                [child_span],
                mapping["continuation_token_offsets"],
                "unicode_character",
            )
        except RowExclusion as exc:
            exclusions.append(
                {
                    "case_id": candidate["case_id"],
                    "edge_id": edge_id,
                    "reason": f"exact_full_answer_token_mapping_failed: {exc}",
                }
            )
            continue
        scientific.append(
            {
                "manifest_protocol_id": PROTOCOL_ID,
                "sample_id": "SRR1-" + _sha_bytes(edge_id.encode())[:16],
                "case_id": candidate["case_id"],
                "edge_id": edge_id,
                "source_question_id": qid,
                "target_model_family": TARGET_MODEL_FAMILY,
                "image_relpath": candidate["image_relpath"],
                "question": candidate["question"],
                "full_visible_answer": full_answer,
                "full_visible_answer_sha256": _sha_bytes(full_answer.encode()),
                "visible_text_reencoded_token_ids": visible_ids,
                "visible_text_reencoded_token_ids_sha256": _sha_bytes(_canonical(visible_ids)),
                "native_generation_ids_certified": False,
                "child_char_span_in_full_answer": child_span,
                "constraint_char_spans_in_full_answer": constraint_spans,
                "annotation_child_surface": child,
                "mitigation_parent_surface_only": candidate["parent_proposal"],
                "model_input_contract": "complete frozen visible OE answer only",
                "edge_type": candidate["edge_type"],
                "modality_stratum": candidate["modality_stratum"],
                "anatomy_stratum": candidate["anatomy_stratum"],
                "prompt_requested_increment": candidate["prompt_requested_increment"],
                "scientific_role": role,
                "adjudicated_parent_visual_support": final[
                    "final_parent_visual_support"
                ],
                "adjudicated_child_visual_support": final[
                    "final_child_visual_support"
                ],
                "adjudicated_increment_observability": final[
                    "final_increment_observability"
                ],
                "mitigation_claim_count_delta": 0,
            }
        )
    if not scientific:
        raise ValueError("no physician-admitted exactly scoreable replay edges")
    for row in scientific:
        row["split"] = case_split[row["case_id"]]
    swap_plan, swap_exclusions = _case_swap_plan(scientific, case_pool)
    excluded_cases = {row["case_id"] for row in swap_exclusions}
    exclusions.extend(swap_exclusions)
    scientific = [row for row in scientific if row["case_id"] not in excluded_cases]
    for row in scientific:
        row["matched_image_swaps"] = swap_plan[row["case_id"]]
        if len({swap["case_id"] for swap in row["matched_image_swaps"]}) != SWAPS_PER_CASE:
            raise AssertionError("swap cases are not distinct")
    if not scientific or len({row["split"] for row in scientific}) != 2:
        raise ValueError("exact-swap exclusions collapsed the replay split")
    _require_g0_role_closure(scientific)
    source_hashes = {
        "private_provenance": _sha_file(provenance_path),
        "source_answers": _sha_file(answer_path),
    }
    return scientific, exclusions, source_hashes


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite replay artifact: {path}")
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


def compile_replay_manifest(
    *,
    pack: Path,
    repo: Path,
    tokenizer_dir: Path,
    output: Path,
    metadata_output: Path,
    attestations: Path | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    validated = validate_adjudication(pack, attestations)
    active_tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        tokenizer_dir, use_fast=True, local_files_only=True
    )
    if not active_tokenizer.is_fast:
        raise ValueError("replay manifest requires a fast tokenizer")
    rows, exclusions, source_hashes = build_replay_rows(
        validated=validated,
        pack=pack,
        repo=repo,
        tokenizer=active_tokenizer,
    )
    payload = b"".join(_canonical(row) + b"\n" for row in rows)
    role_counts = Counter(row["scientific_role"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    metadata = {
        "manifest_protocol_id": PROTOCOL_ID,
        "status": "physician_admitted_visible_answer_replay",
        "target_model_family": TARGET_MODEL_FAMILY,
        "n_scientific_edges": len(rows),
        "n_scientific_cases": len({row["case_id"] for row in rows}),
        "n_exclusions": len(exclusions),
        "role_counts": dict(sorted(role_counts.items())),
        "split_edge_counts": dict(sorted(split_counts.items())),
        "image_disjoint": True,
        "split_seed": SPLIT_SEED,
        "swap_seed": SWAP_SEED,
        "swaps_per_case": SWAPS_PER_CASE,
        "swap_match_fields": ["split", "modality_stratum", "anatomy_stratum"],
        "swap_caliper_relaxation": False,
        "native_generation_sequence_certified": False,
        "native_capture_required_before_scientific_runtime": True,
        "manifest_sha256": _sha_bytes(payload),
        "input_sha256": {**validated.input_sha256, **source_hashes},
        "tokenizer_json_sha256": _sha_file(tokenizer_dir / "tokenizer.json"),
        "compiler_sha256": _sha_file(Path(__file__).resolve()),
        "analysis_protocol_id": "specificity-ratchet-visible-replay-analysis-v1",
        "analysis_source_sha256": _sha_file(
            Path(__file__).with_name(
                "analyze_specificity_ratchet_visible_replay_v1.py"
            )
        ),
        "estimand": (
            "Joint error-versus-supported signature from the first recorded to final "
            "decoder layer: weaker early own-minus-swap constraint evidence, positive "
            "own-image constraint commitment growth, and positive commitment growth "
            "that survives the mean of two swaps after frozen nuisance adjustment."
        ),
        "frozen_analysis_gates": {
            "swap_language_ratchet_adjusted_ci_lower_gt": 0.0,
            "own_commitment_ratchet_adjusted_ci_lower_gt": 0.0,
            "early_visual_evidence_error_minus_control_ci_upper_lt": 0.0,
            "swap_minus_half_own_ci_lower_gt": 0.0,
            "each_individual_swap_adjusted_ci_lower_gt": 0.0,
            "image_specific_transition_adjusted_ci_upper_lt": 0.0,
            "minimum_cases_per_role_per_split": 12,
            "minimum_total_cases_per_split": 24,
            "minimum_edge_types_per_split": 3,
            "minimum_exact_lexical_overlap_blocks": 10,
            "minimum_role_effective_clusters": 12.0,
            "maximum_role_cluster_leverage": 0.20,
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 7319,
            "layer_choice": "first recorded and final; no post-data selection",
            "constraint_lexical_fixed_effect": (
                "strict sensitivity and scope gate, not primary nuisance"
            ),
            "minimum_valid_bootstrap_fraction": 0.95,
        },
        "text_only_role": "secondary lexical sensitivity; never primary visual evidence",
        "exclusions": exclusions,
        "truth_prohibitions": [
            "source model output as support truth",
            "VQA-RAD short reference answer",
            "LLM judge",
            "RadGraph",
            "cross-model agreement",
        ],
    }
    metadata_payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    if output.exists() or metadata_output.exists():
        raise FileExistsError("refusing to overwrite replay manifest or metadata")
    _atomic_write(output, payload)
    try:
        _atomic_write(metadata_output, metadata_payload)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2"),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    )
    parser.add_argument("--attestations", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/replay_manifest_v1/samples.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/replay_manifest_v1/metadata.json"),
    )
    args = parser.parse_args()
    try:
        result = compile_replay_manifest(
            pack=args.pack.resolve(),
            repo=args.repo.resolve(),
            tokenizer_dir=args.tokenizer_dir.resolve(),
            output=args.output.resolve(),
            metadata_output=args.metadata_output.resolve(),
            attestations=args.attestations,
        )
    except (AdjudicationValidationError, FileExistsError, OSError, ValueError) as exc:
        issues = exc.issues if isinstance(exc, AdjudicationValidationError) else [str(exc)]
        print(json.dumps({"status": "refused", "issues": issues}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
