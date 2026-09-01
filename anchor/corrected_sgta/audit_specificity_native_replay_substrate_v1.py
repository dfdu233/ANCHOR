#!/usr/bin/env python3
"""CPU-only substrate audit for full-visible-answer Specificity Ratchet replay.

This audit does not claim that saved token IDs are native generation IDs.  The
source runner decoded text and then re-tokenized it; those IDs are checked only
as visible-text provenance.  Exact generation-sequence identity remains a
separate GPU canary gate after physician admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from corrected_sgta.compile_specificity_ratchet_mechanism_manifest_v1 import (
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


PROTOCOL = "specificity-ratchet-visible-answer-replay-substrate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(pack: Path, repo: Path, tokenizer_dir: Path) -> dict[str, Any]:
    candidates_path = pack / "candidates.blinded.jsonl"
    provenance_path = pack / "provenance.private.jsonl"
    candidates = _read_jsonl(candidates_path)
    provenance = _read_jsonl(provenance_path)
    if not candidates or len(candidates) != len(provenance):
        raise ValueError("candidate/private provenance count mismatch")
    private_by_edge = {row["edge_id"]: row for row in provenance}
    if set(private_by_edge) != {row["edge_id"] for row in candidates}:
        raise ValueError("candidate/private provenance edge IDs differ")
    source_paths = {row["source_answer_path"] for row in provenance}
    if len(source_paths) != 1:
        raise ValueError("native replay requires one frozen Huatuo source file")
    source_path = (repo / next(iter(source_paths))).resolve()
    source_rows = _read_jsonl(source_path)
    source_by_qid = {row["question_id"]: row for row in source_rows}
    if len(source_by_qid) != len(source_rows):
        raise ValueError("source answer question IDs are duplicated")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, use_fast=True, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise ValueError("exact full-answer offsets require a fast tokenizer")

    counts: Counter[str] = Counter()
    unique_cases: set[str] = set()
    exclusions: list[dict[str, str]] = []
    for candidate in candidates:
        edge_id = candidate["edge_id"]
        private = private_by_edge[edge_id]
        qid = private["question_id"]
        source = source_by_qid.get(qid)
        if source is None:
            raise ValueError(f"{edge_id}: missing source question ID")
        line = int(private["source_answer_line"])
        if line < 1 or line > len(source_rows) or source_rows[line - 1]["question_id"] != qid:
            raise ValueError(f"{edge_id}: source line identity mismatch")
        target = str(source["text"])
        child = str(candidate["child_proposal"])
        if target.count(child) != 1:
            raise ValueError(f"{edge_id}: child is not unique in full visible answer")
        child_start = target.index(child)
        visible_ids = tokenizer(target, add_special_tokens=False).input_ids
        recorded_ids = source.get("metadata", {}).get("generated_token_ids")
        if visible_ids != recorded_ids:
            raise ValueError(f"{edge_id}: visible-text re-tokenization provenance drift")
        counts["visible_text_ids_match_record"] += 1

        encoded = tokenizer(
            target + ASSISTANT_SUFFIX,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        mapping = partition_answer_tokens(
            answer_text=target + ASSISTANT_SUFFIX,
            prefix="",
            continuation=target,
            token_ids=encoded["input_ids"],
            offsets=encoded["offset_mapping"],
        )
        if mapping["continuation_token_ids"] == recorded_ids:
            counts["assistant_context_ids_match_visible_record"] += 1
        else:
            counts["assistant_context_ids_differ_from_visible_record"] += 1

        local_spans = exact_constraint_spans(candidate)
        full_spans = [
            {
                **span,
                "char_start": child_start + span["char_start"],
                "char_end_exclusive": child_start + span["char_end_exclusive"],
            }
            for span in local_spans
        ]
        try:
            mapped = map_constraint_spans(
                target,
                full_spans,
                mapping["continuation_token_offsets"],
                "unicode_character",
            )
        except RowExclusion as exc:
            counts["constraint_not_exactly_scoreable"] += 1
            exclusions.append({"edge_id": edge_id, "reason": str(exc)})
        else:
            if not mapped:
                raise AssertionError("exact constraint mapping returned no token")
            counts["constraint_exactly_scoreable"] += 1
        counts["edges"] += 1
        unique_cases.add(candidate["case_id"])

    return {
        "protocol": PROTOCOL,
        "status": "passed_with_declared_edge_exclusions",
        "scientific_interpretation": (
            "Full visible model answers are a valid natural-text replay substrate for "
            "scoreable edges; exact native generation-token identity is not established."
        ),
        "native_generation_sequence_certified": False,
        "gpu_identity_canary_required_after_physician_admission": True,
        "unique_cases": len(unique_cases),
        "counts": dict(sorted(counts.items())),
        "exclusions": exclusions,
        "input_sha256": {
            "candidates": _sha256(candidates_path),
            "private_provenance": _sha256(provenance_path),
            "source_answers": _sha256(source_path),
            "tokenizer_json": _sha256(tokenizer_dir / "tokenizer.json"),
            "audit_source": _sha256(Path(__file__).resolve()),
        },
        "truth_created": False,
    }


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pack.resolve(), args.repo.resolve(), args.tokenizer_dir.resolve())
    result["command"] = [
        "python",
        str(Path(__file__).resolve()),
        "--pack",
        str(args.pack.resolve()),
        "--repo",
        str(args.repo.resolve()),
        "--tokenizer-dir",
        str(args.tokenizer_dir.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    _write_once(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

