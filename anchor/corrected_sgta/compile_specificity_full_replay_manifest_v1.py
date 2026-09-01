#!/usr/bin/env python3
"""Compile physician-admitted full-visible-answer Specificity replay rows.

This is the F6-corrected successor to the isolated parent/child manifest.  It
never scores automatically shortened text.  Each row replays the complete
frozen Huatuo OE answer, localizes the physician-admitted added constraint in
that answer, and freezes same-split modality/anatomy swap candidates.  Native
generation-token identity remains false until the separate post-admission GPU
canary directly captures ``output.sequences``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from anchor.corrected_sgta.compile_specificity_ratchet_mechanism_manifest_v1 import (
        PRIMARY_ERROR_STATES,
        assign_grouped_splits,
        exact_constraint_spans,
    )
    from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
        SOURCE_REQUIRED_STATES,
        validate_adjudication,
    )
except ModuleNotFoundError:
    from compile_specificity_ratchet_mechanism_manifest_v1 import (  # type: ignore[no-redef]
        PRIMARY_ERROR_STATES,
        assign_grouped_splits,
        exact_constraint_spans,
    )
    from validate_specificity_ratchet_adjudication_v1 import (  # type: ignore[no-redef]
        SOURCE_REQUIRED_STATES,
        validate_adjudication,
    )


PROTOCOL_ID = "specificity-ratchet-full-visible-replay-v1"
SUBSTRATE_PROTOCOL = "specificity-ratchet-visible-answer-replay-substrate-v1"
SWAP_SEED = "specificity-ratchet-same-split-swap-pool-v1"
MIN_SWAP_CANDIDATES = 2


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "fingerprint"}
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scientific_role(final: dict[str, str]) -> tuple[str | None, str]:
    if final["final_edge_entailment_admitted"] != "yes":
        return None, "edge_not_admitted"
    if final["final_parent_visual_support"] != "supported":
        return None, "parent_not_visually_supported"
    child = final["final_child_visual_support"]
    source = final["final_increment_observability"]
    if child == "supported" and source == "observable_on_supplied_image":
        return "supported_specificity_control", "admitted_supported_child"
    if child in PRIMARY_ERROR_STATES and source == "observable_on_supplied_image":
        return "causal_escalation_error", f"admitted_child_{child}"
    if child == "unobservable" and source in SOURCE_REQUIRED_STATES:
        return "evidence_source_boundary", f"admitted_{source}"
    return None, "uncertain_or_incoherent_evidence_boundary"


def _safe_repo_path(repo: Path, relative: str) -> Path:
    candidate = (repo / relative).resolve()
    if not candidate.is_relative_to(repo.resolve()):
        raise ValueError(f"source answer path escapes repository: {relative}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_and_validate_substrate(
    *, pack: Path, repo: Path, substrate_audit: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, list[dict[str, Any]]]:
    payload = json.loads(substrate_audit.read_text())
    if payload.get("protocol") != SUBSTRATE_PROTOCOL:
        raise ValueError("full-replay substrate protocol mismatch")
    if payload.get("status") != "passed_with_declared_edge_exclusions":
        raise ValueError("full-replay substrate audit did not pass")
    if payload.get("native_generation_sequence_certified") is not False:
        raise ValueError("substrate must not pre-certify native generation IDs")
    if payload.get("gpu_identity_canary_required_after_physician_admission") is not True:
        raise ValueError("substrate omitted the native-ID identity canary")
    candidates_path = pack / "candidates.blinded.jsonl"
    private_path = pack / "provenance.private.jsonl"
    expected = payload.get("input_sha256", {})
    if expected.get("candidates") != _sha256(candidates_path):
        raise ValueError("substrate candidate hash drift")
    if expected.get("private_provenance") != _sha256(private_path):
        raise ValueError("substrate private-provenance hash drift")
    audit_source = Path(__file__).with_name("audit_specificity_native_replay_substrate_v1.py")
    if expected.get("audit_source") != _sha256(audit_source):
        raise ValueError("substrate audit-source hash drift")
    private_rows = _read_jsonl(private_path)
    source_relpaths = {str(row["source_answer_path"]) for row in private_rows}
    if len(source_relpaths) != 1:
        raise ValueError("full replay requires one frozen source-answer file")
    source_path = _safe_repo_path(repo, next(iter(source_relpaths)))
    if expected.get("source_answers") != _sha256(source_path):
        raise ValueError("substrate source-answer hash drift")
    source_rows = _read_jsonl(source_path)
    return payload, private_rows, source_path, source_rows


def _freeze_swap_pools(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    case_records: dict[str, dict[str, str]] = {}
    by_stratum: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        case_id = row["case_id"]
        record = {"image_relpath": row["image_relpath"], "split": row["split"]}
        previous = case_records.setdefault(case_id, record)
        if previous != record:
            raise ValueError(f"case {case_id} has inconsistent image or split")
        by_stratum[(row["split"], row["modality_stratum"], row["anatomy_stratum"])].add(case_id)

    kept: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for row in rows:
        key = (row["split"], row["modality_stratum"], row["anatomy_stratum"])
        candidates = by_stratum[key] - {row["case_id"]}
        ordered = sorted(
            candidates,
            key=lambda case: hashlib.sha256(
                f"{SWAP_SEED}|{row['sample_id']}|{case}".encode()
            ).hexdigest(),
        )
        if len(ordered) < MIN_SWAP_CANDIDATES:
            exclusions.append(
                {
                    "case_id": row["case_id"],
                    "edge_id": row["edge_id"],
                    "reason": "fewer_than_two_same_split_modality_anatomy_swap_candidates",
                }
            )
            continue
        row = dict(row)
        row["swap_pool_protocol"] = SWAP_SEED
        row["swap_candidates"] = [
            {
                "case_id": case,
                "image_relpath": case_records[case]["image_relpath"],
                "split": row["split"],
            }
            for case in ordered
        ]
        row["minimum_exact_visual_length_swaps_required"] = MIN_SWAP_CANDIDATES
        kept.append(row)
    return kept, exclusions


def compile_replay_rows(
    *,
    validated: Any,
    pack: Path,
    repo: Path,
    substrate_audit: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    audit, private_rows, source_path, source_rows = _load_and_validate_substrate(
        pack=pack, repo=repo, substrate_audit=substrate_audit
    )
    private_by_edge = {str(row["edge_id"]): row for row in private_rows}
    if set(private_by_edge) != {str(row["edge_id"]) for row in validated.candidates}:
        raise ValueError("private provenance edge set differs from physician candidates")
    source_by_qid = {str(row["question_id"]): row for row in source_rows}
    if len(source_by_qid) != len(source_rows):
        raise ValueError("source answers contain duplicate question IDs")
    audit_exclusions = {
        str(row["edge_id"]): str(row["reason"])
        for row in audit.get("exclusions", [])
    }
    generation_config_path = source_path.parent / "generation_config.json"
    if not generation_config_path.is_file():
        raise FileNotFoundError(
            f"full replay requires source generation_config.json: {generation_config_path}"
        )
    generation_config = json.loads(generation_config_path.read_text())
    source_generation_fingerprint = str(generation_config.get("fingerprint", ""))
    if not source_generation_fingerprint:
        raise ValueError("source generation config has no fingerprint")
    if _canonical_fingerprint(generation_config) != source_generation_fingerprint:
        raise ValueError("source generation config fingerprint is not self-consistent")

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for candidate in validated.candidates:
        edge_id = str(candidate["edge_id"])
        role, reason = _scientific_role(validated.final_rows[edge_id])
        if role is None:
            exclusions.append(
                {"case_id": candidate["case_id"], "edge_id": edge_id, "reason": reason}
            )
            continue
        if edge_id in audit_exclusions:
            exclusions.append(
                {
                    "case_id": candidate["case_id"],
                    "edge_id": edge_id,
                    "reason": "frozen_exact_token_boundary_exclusion: " + audit_exclusions[edge_id],
                }
            )
            continue
        private = private_by_edge[edge_id]
        if private.get("source_model") != "huatuo":
            raise ValueError(f"{edge_id}: current full-replay substrate must be Huatuo-native")
        qid = str(private["question_id"])
        source = source_by_qid.get(qid)
        if source is None or source.get("model_id") != "huatuo":
            raise ValueError(f"{edge_id}: missing Huatuo source answer")
        if source.get("metadata", {}).get("fingerprint") != source_generation_fingerprint:
            raise ValueError(f"{edge_id}: source answer generation fingerprint drift")
        line = int(private["source_answer_line"])
        if not (1 <= line <= len(source_rows)) or source_rows[line - 1].get("question_id") != qid:
            raise ValueError(f"{edge_id}: frozen source-answer line identity mismatch")
        full_answer = str(source.get("text", ""))
        child = str(candidate["child_proposal"])
        if not full_answer.strip() or full_answer.count(child) != 1:
            raise ValueError(f"{edge_id}: child is not unique in non-empty full visible answer")
        child_start = full_answer.index(child)
        local_spans = exact_constraint_spans(candidate)
        full_spans = [
            {
                **span,
                "char_start": child_start + int(span["char_start"]),
                "char_end_exclusive": child_start + int(span["char_end_exclusive"]),
            }
            for span in local_spans
        ]
        for span in full_spans:
            exact = full_answer[span["char_start"] : span["char_end_exclusive"]]
            if exact != span["text"] or _sha256_bytes(exact.encode()) != span["utf8_sha256"]:
                raise ValueError(f"{edge_id}: translated constraint span drift")
        final = validated.final_rows[edge_id]
        rows.append(
            {
                "manifest_protocol_id": PROTOCOL_ID,
                "sample_id": "SRF1-" + hashlib.sha256(edge_id.encode()).hexdigest()[:16],
                "case_id": candidate["case_id"],
                "edge_id": edge_id,
                "image_relpath": candidate["image_relpath"],
                "question": candidate["question"],
                "source_model": "huatuo",
                "source_question_id": qid,
                "source_answer_path": str(source_path.relative_to(repo.resolve())),
                "source_answer_line": line,
                "source_generation_fingerprint": source_generation_fingerprint,
                "full_visible_answer": full_answer,
                "full_visible_answer_sha256": _sha256_bytes(full_answer.encode()),
                "child_span_in_visible_answer": {
                    "char_start": child_start,
                    "char_end_exclusive": child_start + len(child),
                    "utf8_sha256": _sha256_bytes(child.encode()),
                },
                "constraint_char_spans_in_visible_answer": full_spans,
                "constraint_occurrences": len(full_spans),
                "scientific_role": role,
                "edge_type": candidate["edge_type"],
                "modality_stratum": candidate["modality_stratum"],
                "anatomy_stratum": candidate["anatomy_stratum"],
                "prompt_requested_increment": candidate["prompt_requested_increment"],
                "adjudicated_parent_visual_support": final["final_parent_visual_support"],
                "adjudicated_child_visual_support": final["final_child_visual_support"],
                "adjudicated_increment_observability": final["final_increment_observability"],
                "native_generation_sequence_certified": False,
                "native_generation_identity_canary_required": True,
                "isolated_parent_child_targets_prohibited": True,
                "text_only_role": "secondary_lexical_sensitivity_only",
            }
        )
    if not rows:
        raise ValueError("no physician-admitted full-visible replay rows")
    assignment = assign_grouped_splits(rows)
    for row in rows:
        row["split"] = assignment[row["case_id"]]
    rows, swap_exclusions = _freeze_swap_pools(rows)
    exclusions.extend(swap_exclusions)
    if not rows:
        raise ValueError("all physician-admitted rows lack frozen swap controls")
    if {row["case_id"] for row in rows if row["split"] == "dev"} & {
        row["case_id"] for row in rows if row["split"] == "test"
    }:
        raise AssertionError("full-replay manifest leaks cases across dev/test")
    return rows, exclusions, {
        "substrate_audit_sha256": _sha256(substrate_audit),
        "source_answers_sha256": _sha256(source_path),
        "source_generation_config": str(
            generation_config_path.relative_to(repo.resolve())
        ),
        "source_generation_config_sha256": _sha256(generation_config_path),
        "source_generation_fingerprint": source_generation_fingerprint,
        "substrate_protocol": audit["protocol"],
    }


def _write_pair(
    *, output: Path, metadata_output: Path, rows_payload: bytes, metadata_payload: bytes
) -> None:
    for destination in (output, metadata_output):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite full-replay artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[tuple[Path, str]] = []
    try:
        for destination, payload in (
            (output, rows_payload),
            (metadata_output, metadata_payload),
        ):
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporaries.append((destination, temporary))
        for destination, temporary in temporaries:
            os.replace(temporary, destination)
    finally:
        for _, temporary in temporaries:
            if os.path.exists(temporary):
                os.unlink(temporary)


def compile_full_replay_manifest(
    *,
    pack: Path,
    repo: Path,
    substrate_audit: Path,
    output: Path,
    metadata_output: Path,
    attestations: Path | None = None,
) -> dict[str, Any]:
    validated = validate_adjudication(pack, attestations)
    rows, exclusions, provenance = compile_replay_rows(
        validated=validated,
        pack=pack,
        repo=repo,
        substrate_audit=substrate_audit,
    )
    rows_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode()
    role_counts = Counter(row["scientific_role"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    metadata = {
        "manifest_protocol_id": PROTOCOL_ID,
        "status": "physician_admitted_full_visible_replay",
        "rows": len(rows),
        "manifest_sha256": _sha256_bytes(rows_payload),
        "role_counts": dict(sorted(role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "image_disjoint": True,
        "source_model": "huatuo",
        "source_generation_fingerprint": provenance["source_generation_fingerprint"],
        "native_generation_sequence_certified": False,
        "native_generation_identity_canary_required": True,
        "gpu_scoring_authorized": False,
        "authorization_next_gate": (
            "one admitted Huatuo deterministic regeneration with directly captured "
            "output.sequences and decoded visible-text identity"
        ),
        "primary_estimand": (
            "conjunctive signature: error-minus-control late-minus-early change in the "
            "own-image constraint-versus-relative-position-matched nonconstraint contrast, "
            "plus an early supported-control image-specific advantage and no corresponding "
            "late increase in own-minus-mean-swap visual residual; at least two same-split "
            "modality/anatomy swaps with exactly matched visual-token length"
        ),
        "text_only_role": "secondary_lexical_sensitivity_only",
        "isolated_parent_child_runtime_prohibited": True,
        "swap_pool_protocol": SWAP_SEED,
        "minimum_exact_visual_length_swaps_required": MIN_SWAP_CANDIDATES,
        "physician_input_sha256": validated.input_sha256,
        "provenance": provenance,
        "exclusions": exclusions,
    }
    metadata_payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    _write_pair(
        output=output,
        metadata_output=metadata_output,
        rows_payload=rows_payload,
        metadata_payload=metadata_payload,
    )
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
        "--substrate-audit",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/native_replay_substrate_audit_v1.json"),
    )
    parser.add_argument("--attestations", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/full_replay_manifest_v1/samples.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/full_replay_manifest_v1/metadata.json"),
    )
    args = parser.parse_args()
    try:
        result = compile_full_replay_manifest(
            pack=args.pack.resolve(),
            repo=args.repo.resolve(),
            substrate_audit=args.substrate_audit.resolve(),
            output=args.output.resolve(),
            metadata_output=args.metadata_output.resolve(),
            attestations=args.attestations.resolve() if args.attestations else None,
        )
    except (OSError, ValueError) as exc:
        issues = getattr(exc, "issues", None)
        print(json.dumps({"status": "refused", "reason": str(exc), "issues": issues}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
