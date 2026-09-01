#!/usr/bin/env python3
"""Outcome-blind audit of observable parent-before-constraint structure.

This audit reads only the frozen blinded candidate JSONL.  It deliberately
does not open reviewer or adjudication files and never assigns clinical truth.

The strict automatic certificate uses a conservative surface criterion: the
proposed parent must occur verbatim in the observed answer and close with
sentence-final punctuation before the earliest added-constraint character.
Anything weaker may be useful for a later, independently validated parent
readout, but it cannot make a token-score curve a parent-to-child crossing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from anchor.corrected_sgta.compile_specificity_ratchet_mechanism_manifest_v1 import (
        exact_constraint_spans,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` entry point.
    from compile_specificity_ratchet_mechanism_manifest_v1 import (  # type: ignore[no-redef]
        exact_constraint_spans,
    )


PROTOCOL_ID = "specificity-ratchet-parent-before-constraint-audit-v1"
SPLIT_SEED = "specificity-ratchet-visible-replay-split-v1"
MIN_REPEATED_BLOCKS_PER_SPLIT = 10
MIN_EDGE_TYPES = 3


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label_blind_case_splits(candidates: list[dict[str, Any]]) -> dict[str, str]:
    """Reproduce the frozen image-disjoint split without importing transformers."""

    case_info: dict[str, dict[str, str]] = {}
    for row in candidates:
        info = {
            "case_id": str(row["case_id"]),
            "image_relpath": str(row["image_relpath"]),
            "modality_stratum": str(row["modality_stratum"]),
            "anatomy_stratum": str(row["anatomy_stratum"]),
        }
        prior = case_info.setdefault(info["case_id"], info)
        if prior != info:
            raise ValueError(f"{info['case_id']}: inconsistent frozen case strata")

    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for info in case_info.values():
        by_cell[(info["modality_stratum"], info["anatomy_stratum"])].append(info)

    assignment: dict[str, str] = {}
    for cell, items in sorted(by_cell.items()):
        ordered = sorted(
            items,
            key=lambda row: _sha_bytes(
                f"{SPLIT_SEED}|{cell!r}|{row['case_id']}".encode()
            ),
        )
        start = int(_sha_bytes(f"{SPLIT_SEED}|{cell!r}".encode()), 16) % 2
        for index, row in enumerate(ordered):
            assignment[row["case_id"]] = (
                "dev" if (index + start) % 2 == 0 else "test"
            )
    return assignment


def classify_parent_realization(candidate: dict[str, Any]) -> dict[str, Any]:
    """Classify only directly observed surface order; infer no clinical state."""

    answer = str(candidate["answer_span"])
    child = str(candidate["child_proposal"])
    parent = str(candidate["parent_proposal"])
    child_starts = [index for index in range(len(answer)) if answer.startswith(child, index)]
    if len(child_starts) != 1:
        raise ValueError(
            f"{candidate['edge_id']}: child must occur exactly once in answer_span"
        )
    child_start = child_starts[0]
    constraint_spans = exact_constraint_spans(candidate)
    earliest_constraint = child_start + min(
        int(span["char_start"]) for span in constraint_spans
    )

    parent_occurrences: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = answer.find(parent, cursor)
        if start < 0:
            break
        end = start + len(parent)
        if end <= earliest_constraint:
            parent_occurrences.append((start, end))
        cursor = start + 1

    exact_before = bool(parent_occurrences)
    sentence_closed = exact_before and parent.rstrip().endswith((".", "!", "?"))
    if sentence_closed:
        state = "strict_sentence_closed_parent_before_constraint"
    elif exact_before:
        state = "exact_surface_parent_but_not_sentence_closed"
    else:
        state = "counterfactual_parent_only"

    block_key = " || ".join(
        str(span["text"]).strip().casefold() for span in constraint_spans
    )
    return {
        "case_id": str(candidate["case_id"]),
        "edge_id": str(candidate["edge_id"]),
        "edge_type": str(candidate["edge_type"]),
        "realization_state": state,
        "exact_parent_before_constraint": exact_before,
        "strict_parent_before_constraint": sentence_closed,
        "earliest_constraint_char_in_answer_span": earliest_constraint,
        "parent_occurrences_before_constraint": [
            {"char_start": start, "char_end_exclusive": end}
            for start, end in parent_occurrences
        ],
        "exact_constraint_block_key": block_key,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_block: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_block[row["exact_constraint_block_key"]].add(row["case_id"])
    repeated = {
        key: cases for key, cases in by_block.items() if len(cases) >= 2
    }
    return {
        "edges": len(rows),
        "cases": len({row["case_id"] for row in rows}),
        "edge_type_counts": dict(sorted(Counter(row["edge_type"] for row in rows).items())),
        "edge_types": len({row["edge_type"] for row in rows}),
        "exact_constraint_blocks": len(by_block),
        "repeated_exact_constraint_blocks": len(repeated),
        "edges_in_repeated_exact_blocks": sum(
            1 for row in rows if row["exact_constraint_block_key"] in repeated
        ),
    }


def audit(candidates_path: Path) -> dict[str, Any]:
    candidates = _read_candidates(candidates_path)
    splits = label_blind_case_splits(candidates)
    rows = []
    for candidate in candidates:
        row = classify_parent_realization(candidate)
        row["split"] = splits[row["case_id"]]
        rows.append(row)

    strict = [row for row in rows if row["strict_parent_before_constraint"]]
    strict_summaries = {
        split: _summary(
            strict if split == "all" else [row for row in strict if row["split"] == split]
        )
        for split in ("all", "dev", "test")
    }
    state_counts = Counter(row["realization_state"] for row in rows)
    exact_surface = [row for row in rows if row["exact_parent_before_constraint"]]
    exact_summaries = {
        split: _summary(
            exact_surface
            if split == "all"
            else [row for row in exact_surface if row["split"] == split]
        )
        for split in ("all", "dev", "test")
    }

    split_block_gate = all(
        strict_summaries[split]["repeated_exact_constraint_blocks"]
        >= MIN_REPEATED_BLOCKS_PER_SPLIT
        for split in ("dev", "test")
    )
    type_gate = strict_summaries["all"]["edge_types"] >= MIN_EDGE_TYPES
    current_pack_certifiable = split_block_gate and type_gate
    return {
        "protocol_id": PROTOCOL_ID,
        "status": (
            "surface_construct_certificate_possible"
            if current_pack_certifiable
            else "no_go_current_pack"
        ),
        "dataset": "VQA-RAD public OE candidate pack",
        "model": "HuatuoGPT-Vision-7B frozen natural OE outputs",
        "split_seed": SPLIT_SEED,
        "candidate_sha256": _sha_file(candidates_path),
        "source_sha256": _sha_file(Path(__file__).resolve()),
        "outcome_blind_contract": {
            "files_read": [str(candidates_path)],
            "physician_reviews_read": False,
            "adjudication_read": False,
            "clinical_support_inferred": False,
        },
        "strict_surface_definition": (
            "The proposed parent occurs verbatim and closes with sentence-final "
            "punctuation before the earliest exact added-constraint character in "
            "the observed answer span."
        ),
        "state_counts": dict(sorted(state_counts.items())),
        "exact_surface_parent_summaries": exact_summaries,
        "strict_parent_summaries": strict_summaries,
        "gates": {
            "minimum_repeated_exact_constraint_blocks_per_split": MIN_REPEATED_BLOCKS_PER_SPLIT,
            "minimum_edge_types": MIN_EDGE_TYPES,
            "dev_repeated_block_gate": (
                strict_summaries["dev"]["repeated_exact_constraint_blocks"]
                >= MIN_REPEATED_BLOCKS_PER_SPLIT
            ),
            "test_repeated_block_gate": (
                strict_summaries["test"]["repeated_exact_constraint_blocks"]
                >= MIN_REPEATED_BLOCKS_PER_SPLIT
            ),
            "edge_type_gate": type_gate,
            "current_pack_surface_construct_certifiable": current_pack_certifiable,
        },
        "semantic_block_limitation": (
            "No pre-frozen physician-independent semantic_block_id exists. Exact "
            "constraint repetition is therefore the only auditable block count; "
            "the audit refuses to invent semantic equivalence after selection."
        ),
        "scientific_naming_gate": {
            "crossing_authorized": False,
            "permitted_name_for_constraint_only_layer_curve": "late constraint amplification",
            "reason": (
                "A surface audit cannot establish physician-admitted parent support, "
                "and the current strict subset also fails prevalence and edge-type gates."
            ),
        },
        "rows": rows,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.candidates.resolve())
    result["command"] = [shlex.join([str(Path(__file__)), *os.sys.argv[1:]])]
    _atomic_write(
        args.output.resolve(),
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
