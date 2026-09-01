#!/usr/bin/env python3
"""Analyze the bounded natural-OE diagnostic-completion pilot fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.audit_diagnostic_completion_substrate_v1 import (
    extract_events,
    sha256_file,
    wilson,
)


VERSION = "natural-oe-diagnostic-completion-analysis-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def prefix(text: str, tokens: int = 10) -> str:
    return " ".join(normalize_text(text).split()[:tokens])


def sentence_count(text: str) -> int:
    return len([chunk for chunk in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if chunk.strip()])


def concentration(values: Sequence[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = len(values)
    top_value, top_count = counts.most_common(1)[0] if counts else (None, 0)
    return {
        "total": total,
        "unique": len(counts),
        "top1_value": top_value,
        "top1_count": top_count,
        "top1_share": top_count / total if total else None,
    }


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_generation(
    run_dir: Path, contract_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(contract["generation_manifest"]))
    manifest = load_jsonl(manifest_path)
    if sha256_file(manifest_path) != contract["generation_manifest_sha256"]:
        raise ValueError("pilot generation manifest hash drift")
    generation_path = run_dir / "generations.jsonl"
    summary = json.loads((run_dir / "generation_summary.json").read_text())
    if sha256_file(generation_path) != summary.get("generations_sha256"):
        raise ValueError("generation aggregate hash drift")
    rows = load_jsonl(generation_path)
    if len(rows) != len(manifest) or len(rows) != int(contract["images"]):
        raise ValueError("generation does not cover the exact frozen manifest")
    by_item = {str(row["item_id"]): row for row in rows}
    if len(by_item) != len(rows):
        raise ValueError("duplicate generated item IDs")
    if set(by_item) != {str(row["item_id"]) for row in manifest}:
        raise ValueError("generated item IDs differ from frozen manifest")
    config = json.loads((run_dir / "generation_config.json").read_text())
    if config.get("pilot_contract_sha256") != sha256_file(contract_path):
        raise ValueError("generation config is not bound to this pilot contract")
    fingerprint = str(config["fingerprint"])
    for row in rows:
        if row.get("fingerprint") != fingerprint:
            raise ValueError("generation shard fingerprint drift")
        if row.get("reader_labels_available_to_generation") is not False:
            raise ValueError("reader labels leaked into generation")
        if row.get("target_edge_available_to_generation") is not False:
            raise ValueError("target edge leaked into generation")
        if row.get("clinical_claim_evaluation_status") != (
            "pending_physician_construct_audit"
        ):
            raise ValueError("generation assigned clinical truth")
    return contract, manifest, rows


def analyze(
    run_dir: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output_path}")
    contract, manifest, rows = validate_generation(run_dir, contract_path)
    stop_rule = contract["predeclared_pilot_stop_rule"]

    # Phase A is deliberately reader-label free. Event text is frozen before
    # the sealed design file is opened below.
    geometry = {
        "images": len(rows),
        "nonempty": sum(bool(str(row["text"]).strip()) for row in rows),
        "cap_hits": sum(bool(row["hit_max_new_tokens"]) for row in rows),
        "surface_refusal_matches": sum(
            bool(row["surface_refusal_match"]) for row in rows
        ),
        "generated_tokens": {
            "mean": sum(int(row["generated_token_count"]) for row in rows) / len(rows),
            "minimum": min(int(row["generated_token_count"]) for row in rows),
            "maximum": max(int(row["generated_token_count"]) for row in rows),
        },
        "whitespace_words": {
            "mean": sum(len(str(row["text"]).split()) for row in rows) / len(rows),
            "maximum": max(len(str(row["text"]).split()) for row in rows),
        },
        "sentence_count": Counter(sentence_count(str(row["text"])) for row in rows),
        "one_sentence_adherence": sum(
            sentence_count(str(row["text"])) == 1 for row in rows
        )
        / len(rows),
        "exact_text": concentration([normalize_text(str(row["text"])) for row in rows]),
        "prefix_10": concentration([prefix(str(row["text"]), 10) for row in rows]),
    }
    geometry["sentence_count"] = {
        str(key): value for key, value in sorted(geometry["sentence_count"].items())
    }
    frozen_events = []
    for row in rows:
        for event in extract_events(str(row["text"])):
            frozen_events.append(
                {
                    **event,
                    "item_id": str(row["item_id"]),
                    "image_id": str(row["image_id"]),
                }
            )
    frozen_events.sort(
        key=lambda row: (str(row["item_id"]), str(row["edge_id"]), str(row["target_key"]))
    )
    event_fingerprint = canonical_sha(frozen_events)

    # Phase B opens the pre-frozen reader design only after extraction.
    sealed_path = Path(str(contract["sealed_reader_design"]))
    if sha256_file(sealed_path) != contract["sealed_reader_design_sha256"]:
        raise ValueError("sealed reader design hash drift")
    design = load_jsonl(sealed_path)
    if len(design) != len(rows):
        raise ValueError("sealed design does not cover every generated image")
    design_by_image = {str(row["image_id"]): row for row in design}
    if len(design_by_image) != len(design):
        raise ValueError("sealed design is not image-disjoint")
    events_by_image_edge: dict[tuple[str, str], int] = Counter(
        (str(row["image_id"]), str(row["edge_id"])) for row in frozen_events
    )
    cells: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"images": 0, "events": 0})
    )
    joined = []
    for image_id, design_row in sorted(design_by_image.items()):
        edge_id = str(design_row["edge_id"])
        stratum = str(design_row["design_stratum"])
        has_event = int(events_by_image_edge.get((image_id, edge_id), 0) > 0)
        cells[edge_id][stratum]["images"] += 1
        cells[edge_id][stratum]["events"] += has_event
        joined.append(
            {
                **design_row,
                "assigned_edge_event": bool(has_event),
            }
        )

    cell_summary: dict[str, Any] = {}
    passing_edges = []
    minimum_events = int(stop_rule["minimum_events_per_extreme_per_edge"])
    for edge_id in sorted(cells):
        cell_summary[edge_id] = {}
        extreme_counts = []
        for stratum in ("child_0of3", "child_3of3"):
            cell = cells[edge_id][stratum]
            cell_summary[edge_id][stratum] = {
                **cell,
                "wilson_95": wilson(cell["events"], cell["images"]),
            }
            extreme_counts.append(cell["events"])
        if min(extreme_counts) >= minimum_events:
            passing_edges.append(edge_id)

    nonempty_rate = geometry["nonempty"] / len(rows)
    cap_hit_rate = geometry["cap_hits"] / len(rows)
    prefix_share = float(geometry["prefix_10"]["top1_share"] or 0.0)
    geometry_gate = (
        nonempty_rate >= float(stop_rule["minimum_nonempty_rate"])
        and cap_hit_rate <= float(stop_rule["maximum_cap_hit_rate"])
        and prefix_share <= float(stop_rule["maximum_dominant_prefix_10_share"])
    )
    substrate_gate = len(passing_edges) >= int(
        stop_rule["minimum_semantic_edges_with_events_in_both_extremes"]
    )
    summary = {
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pilot_analyzed_no_clinical_truth",
        "integrity_passed": True,
        "phase_a_reader_label_free_geometry": geometry,
        "phase_a_frozen_events": {
            "events": len(frozen_events),
            "images": len({row["image_id"] for row in frozen_events}),
            "fingerprint": event_fingerprint,
        },
        "phase_b_assigned_edge_cells": cell_summary,
        "gates": {
            "response_geometry_passed": geometry_gate,
            "passing_semantic_edges": passing_edges,
            "repeated_extreme_event_substrate_passed": substrate_gate,
            "physician_construct_review_authorized": bool(
                geometry_gate and substrate_gate
            ),
            "hidden_state_replay_authorized": False,
            "larger_generation_authorized": False,
        },
        "clinical_claims": {
            "hallucination_claim_authorized": False,
            "unsupported_diagnosis_claim_authorized": False,
            "reason": "VinDr reader-panel image labels are not complete clinical diagnostic truth; physician construct admission remains required.",
        },
        "run_dir": str(run_dir),
        "generations_sha256": sha256_file(run_dir / "generations.jsonl"),
        "pilot_contract": str(contract_path),
        "pilot_contract_sha256": sha256_file(contract_path),
        "sealed_reader_design_sha256": sha256_file(sealed_path),
        "analyzer_source_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_events": frozen_events,
        "joined_assigned_edges": joined,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pilot-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.pilot_contract, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase_a_frozen_events": result["phase_a_frozen_events"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
