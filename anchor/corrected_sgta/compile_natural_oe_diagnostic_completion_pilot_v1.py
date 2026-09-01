#!/usr/bin/env python3
"""Compile a bounded, reader-stratified natural-OE completion pilot.

This compiler uses the VinDr R8/R9/R10 panel to obtain design balance, but it
never reads model generations.  The public generation manifest contains no
reader labels or target edge.  A separate sealed design file retains the
predeclared edge/stratum assignment for analysis after generation freezes.

This is deliberately called reader-stratified, not label-blind.  Generation is
blind to reader labels; sample selection is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.audit_diagnostic_completion_substrate_v1 import (
    EDGE_SPECS,
    EdgeSpec,
    load_exact_panel_votes,
    sha256_file,
)


VERSION = "natural-oe-diagnostic-completion-pilot-v1"
PROMPT = (
    "What abnormalities are visible in this chest X-ray? "
    "Answer in one concise sentence."
)
DEFAULT_LABELS = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "annotations/image_labels_train.csv"
)
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank(seed: int, edge_id: str, child_votes: int, image_id: str) -> str:
    key = f"{VERSION}:{seed}:{edge_id}:{child_votes}:{image_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def candidate_counts(
    votes: Mapping[str, Mapping[str, int]],
    edge_specs: Sequence[EdgeSpec] = EDGE_SPECS,
    *,
    minimum_parent_votes: int = 3,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for spec in edge_specs:
        result[spec.edge_id] = {
            f"child_{child_votes}of3": sum(
                1
                for row in votes.values()
                if int(row[spec.parent_label]) >= minimum_parent_votes
                and int(row[spec.child_label]) == child_votes
            )
            for child_votes in (0, 3)
        }
    return result


def compile_design(
    votes: Mapping[str, Mapping[str, int]],
    *,
    seed: int = 20260802,
    edge_types: int = 2,
    per_extreme: int = 12,
    maximum_images: int = 48,
    minimum_parent_votes: int = 3,
    edge_specs: Sequence[EdgeSpec] = EDGE_SPECS,
) -> dict[str, Any]:
    if edge_types < 2:
        raise ValueError("at least two semantic edge types are required")
    if per_extreme < 1 or maximum_images < 1:
        raise ValueError("per_extreme and maximum_images must be positive")
    if minimum_parent_votes not in (1, 2, 3):
        raise ValueError("minimum_parent_votes must be 1, 2, or 3")
    required = edge_types * 2 * per_extreme
    if required > maximum_images:
        raise ValueError(
            f"design requests {required} distinct images, above maximum {maximum_images}"
        )

    counts = candidate_counts(
        votes, edge_specs, minimum_parent_votes=minimum_parent_votes
    )
    eligible = [
        spec
        for spec in edge_specs
        if min(counts[spec.edge_id].values()) >= per_extreme
    ]
    eligible.sort(
        key=lambda spec: (
            -min(counts[spec.edge_id].values()),
            spec.edge_id,
        )
    )
    chosen = eligible[:edge_types]
    if len(chosen) < edge_types:
        raise ValueError(
            f"only {len(chosen)} edges have >= {per_extreme} candidates in both extremes"
        )

    used: set[str] = set()
    assignments: list[dict[str, Any]] = []
    for spec in chosen:
        for child_votes in (0, 3):
            candidates = [
                image_id
                for image_id, row in votes.items()
                if int(row[spec.parent_label]) >= minimum_parent_votes
                and int(row[spec.child_label]) == child_votes
            ]
            candidates.sort(key=lambda image_id: _rank(seed, spec.edge_id, child_votes, image_id))
            selected = [image_id for image_id in candidates if image_id not in used][
                :per_extreme
            ]
            if len(selected) != per_extreme:
                raise ValueError(
                    f"distinct-image allocation failed for {spec.edge_id} child={child_votes}/3: "
                    f"{len(selected)} < {per_extreme}"
                )
            for image_id in selected:
                used.add(image_id)
                row = votes[image_id]
                assignments.append(
                    {
                        "image_id": image_id,
                        "edge_id": spec.edge_id,
                        "parent_label": spec.parent_label,
                        "child_label": spec.child_label,
                        "parent_votes": int(row[spec.parent_label]),
                        "child_votes": child_votes,
                        "design_stratum": f"child_{child_votes}of3",
                    }
                )

    assignments.sort(key=lambda row: str(row["image_id"]))
    if len(assignments) != required or len(used) != required:
        raise RuntimeError("design is not image-disjoint")
    return {
        "assignments": assignments,
        "candidate_counts": counts,
        "chosen_edges": [spec.edge_id for spec in chosen],
        "requested_images": required,
        "minimum_parent_votes": minimum_parent_votes,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def write_design(
    result: Mapping[str, Any],
    *,
    output_dir: Path,
    labels_path: Path,
    image_root: Path,
    seed: int,
    per_extreme: int,
    minimum_parent_votes: int,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix or overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = list(result["assignments"])
    public_rows = []
    for row in assignments:
        image_id = str(row["image_id"])
        dicom = image_root / f"{image_id}.dicom"
        if not dicom.is_file():
            raise FileNotFoundError(dicom)
        public_rows.append(
            {
                "item_id": hashlib.sha256(
                    f"{VERSION}:{seed}:{image_id}".encode("utf-8")
                ).hexdigest()[:32],
                "image_id": image_id,
                "dicom_path": str(dicom),
                "prompt_id": "natural_abnormality_listing_v1",
                "selection_uses_reader_labels": True,
                "selection_uses_model_outputs": False,
                "generation_receives_reader_labels": False,
                "generation_receives_target_edge": False,
            }
        )
    public_rows.sort(key=lambda row: str(row["item_id"]))

    public_path = output_dir / "generation_manifest.jsonl"
    sealed_path = output_dir / "sealed_reader_design.jsonl"
    _atomic_write(public_path, _jsonl(public_rows))
    _atomic_write(sealed_path, _jsonl(assignments))
    source_path = Path(__file__).resolve()
    contract = {
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "manifest_frozen_generation_not_authorized",
        "prompt": PROMPT,
        "prompt_id": "natural_abnormality_listing_v1",
        "reader_stratified_selection": True,
        "label_blind_selection_claimed": False,
        "selection_uses_model_outputs": False,
        "generation_receives_reader_labels": False,
        "generation_receives_target_edge": False,
        "seed": seed,
        "images": len(public_rows),
        "per_edge_per_extreme": per_extreme,
        "minimum_parent_votes": minimum_parent_votes,
        "chosen_edges": list(result["chosen_edges"]),
        "candidate_counts": result["candidate_counts"],
        "labels_path": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "generation_manifest": str(public_path),
        "generation_manifest_sha256": sha256_file(public_path),
        "sealed_reader_design": str(sealed_path),
        "sealed_reader_design_sha256": sha256_file(sealed_path),
        "compiler_source": str(source_path),
        "compiler_source_sha256": sha256_file(source_path),
        "progression_gate": (
            "corrected_runs/specificity_ratchet/"
            "diagnostic_completion_progression_gate_v1.json"
        ),
        "gpu_generation_authorized": False,
        "hidden_state_replay_authorized": False,
        "physician_construct_review_required_before_replay": True,
        "predeclared_pilot_stop_rule": {
            "minimum_semantic_edges_with_events_in_both_extremes": 2,
            "minimum_events_per_extreme_per_edge": 4,
            "maximum_cap_hit_rate": 0.05,
            "minimum_nonempty_rate": 0.95,
            "maximum_dominant_prefix_10_share": 0.80,
            "clinical_claim_status": "reader-panel diagnostic support only until physician admission"
        },
    }
    contract["fingerprint"] = canonical_sha(
        {key: value for key, value in contract.items() if key != "created_at_utc"}
    )
    _atomic_write(
        output_dir / "pilot_contract.json",
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--edge-types", type=int, default=2)
    parser.add_argument("--per-extreme", type=int, default=12)
    parser.add_argument("--maximum-images", type=int, default=48)
    parser.add_argument("--minimum-parent-votes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    votes = load_exact_panel_votes(args.labels)
    result = compile_design(
        votes,
        seed=args.seed,
        edge_types=args.edge_types,
        per_extreme=args.per_extreme,
        maximum_images=args.maximum_images,
        minimum_parent_votes=args.minimum_parent_votes,
    )
    contract = write_design(
        result,
        output_dir=args.output_dir,
        labels_path=args.labels,
        image_root=args.image_root,
        seed=args.seed,
        per_extreme=args.per_extreme,
        minimum_parent_votes=args.minimum_parent_votes,
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
