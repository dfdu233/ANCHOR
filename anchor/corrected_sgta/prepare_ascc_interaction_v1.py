#!/usr/bin/env python3
"""Freeze the reader-matched substrate for controlled commitment scoring.

This replaces the invalid free-generation confirmatory design.  The model is
never allowed to choose an observation, diagnosis, polarity, or wording.  For
each image it receives a fixed observation/diagnosis prefix and we compare the
single-token continuations ``unlikely``, ``possible``, and ``present`` under
two task framings.  Selection uses only the independent VinDr R8/R9/R10 labels
and acquisition-shape metadata, never model output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit_diagnostic_completion_substrate_v1 import (
    DEFAULT_GENERATIONS,
    DEFAULT_LABELS,
    FROZEN_PANEL,
    load_jsonl,
    sha256_file,
)
from .prepare_pragmatic_commitment_confirmatory_v1 import canonical_hash


VERSION = "ascc-reader-interaction-substrate-v1"
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")
MARKERS = (" unlikely", " possible", " present")
PROMPTS = (
    {
        "name": "neutral_findings_a",
        "prompt_pair_id": "a",
        "framing": "neutral",
        "prompt": (
            "Describe the image-grounded findings on this chest X-ray. Present, absent, "
            "and uncertain states are all valid. Complete the fixed report fragment "
            "using exactly one of: unlikely, possible, or present."
        ),
    },
    {
        "name": "existential_abnormalities_a",
        "prompt_pair_id": "a",
        "framing": "existential",
        "prompt": (
            "List the image-grounded abnormalities on this chest X-ray. Present, absent, "
            "and uncertain states are all valid. Complete the fixed report fragment "
            "using exactly one of: unlikely, possible, or present."
        ),
    },
    {
        "name": "neutral_findings_b",
        "prompt_pair_id": "b",
        "framing": "neutral",
        "prompt": (
            "Provide an image-grounded description of this chest X-ray. Present, absent, "
            "and uncertain states are all valid. Complete the fixed report fragment "
            "using exactly one of: unlikely, possible, or present."
        ),
    },
    {
        "name": "existential_abnormalities_b",
        "prompt_pair_id": "b",
        "framing": "existential",
        "prompt": (
            "Name the image-grounded abnormalities on this chest X-ray. Present, absent, "
            "and uncertain states are all valid. Complete the fixed report fragment "
            "using exactly one of: unlikely, possible, or present."
        ),
    },
)


@dataclass(frozen=True)
class Edge:
    edge_id: str
    parent_label: str
    child_label: str
    prefix: str


EDGES = (
    Edge(
        edge_id="lung_opacity_to_pneumonia",
        parent_label="Lung Opacity",
        child_label="Pneumonia",
        prefix="Findings: Lung opacity is present. Impression: Pneumonia is",
    ),
    Edge(
        edge_id="infiltration_to_pneumonia",
        parent_label="Infiltration",
        child_label="Pneumonia",
        prefix="Findings: Pulmonary infiltration is present. Impression: Pneumonia is",
    ),
    Edge(
        edge_id="nodule_mass_to_lung_tumor",
        parent_label="Nodule/Mass",
        child_label="Lung tumor",
        prefix=(
            "Findings: A pulmonary nodule or mass is present. "
            "Impression: Lung tumor is"
        ),
    ),
)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def deterministic_key(seed: int, *parts: object) -> str:
    joined = ":".join(str(part) for part in (VERSION, seed, *parts))
    return hashlib.sha256(joined.encode()).hexdigest()


def load_reader_panel(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    labels = {edge.parent_label for edge in EDGES} | {
        edge.child_label for edge in EDGES
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = labels - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"VinDr label CSV is missing fields: {sorted(missing)}")
        for row in reader:
            grouped[str(row["image_id"])].append(row)
    output: dict[str, dict[str, dict[str, int]]] = {}
    for image_id, rows in grouped.items():
        if len(rows) != 3 or {row["rad_id"] for row in rows} != FROZEN_PANEL:
            continue
        output[image_id] = {
            str(row["rad_id"]): {label: int(row[label]) for label in labels}
            for row in rows
        }
    return output


def acquisition_bucket(image_path: Path) -> dict[str, Any]:
    import pydicom

    data = pydicom.dcmread(
        str(image_path), stop_before_pixels=True, specific_tags=["Rows", "Columns"]
    )
    rows, columns = int(data.Rows), int(data.Columns)
    ratio = columns / rows
    aspect = "square" if ratio >= 0.93 else "wide" if ratio >= 0.80 else "portrait"
    return {
        "dicom_rows": rows,
        "dicom_columns": columns,
        "aspect_bucket": aspect,
        "view_position": "unavailable_in_released_dicom",
    }


def candidate_rows(
    panel: Mapping[str, Mapping[str, Mapping[str, int]]],
    excluded: set[str],
    image_root: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for image_id, readers in panel.items():
        if image_id in excluded:
            continue
        image_path = image_root / f"{image_id}.dicom"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        metadata: dict[str, Any] | None = None
        for edge in EDGES:
            parent_by_reader = {
                reader: int(values[edge.parent_label])
                for reader, values in readers.items()
            }
            child_by_reader = {
                reader: int(values[edge.child_label])
                for reader, values in readers.items()
            }
            parent_votes = sum(parent_by_reader.values())
            child_votes = sum(child_by_reader.values())
            if parent_votes < 2:
                continue
            if metadata is None:
                metadata = acquisition_bucket(image_path)
            output.append(
                {
                    "image_id": image_id,
                    "item_id": f"{image_id}:{edge.edge_id}",
                    "dicom_relpath": f"train/{image_id}.dicom",
                    "edge_id": edge.edge_id,
                    "parent_label": edge.parent_label,
                    "child_label": edge.child_label,
                    "fixed_prefix": edge.prefix,
                    "parent_votes": parent_votes,
                    "child_votes": child_votes,
                    "child_support_stratum": f"reader_{child_votes}of3",
                    "parent_by_reader": parent_by_reader,
                    "child_by_reader": child_by_reader,
                    "within_reader_joint_support": sum(
                        parent_by_reader[reader] and child_by_reader[reader]
                        for reader in sorted(readers)
                    ),
                    **metadata,
                }
            )
    return output


def _allocate_pairs(
    available: Mapping[tuple[int, str], int], target: int, seed: int, edge_id: str, family: str
) -> dict[tuple[int, str], int]:
    total = sum(available.values())
    raw = [
        (key, count, target * count / total if total else 0.0)
        for key, count in sorted(available.items())
    ]
    allocation = {key: min(count, int(value)) for key, count, value in raw}
    remaining = target - sum(allocation.values())
    order = sorted(
        raw,
        key=lambda item: (
            -(item[2] - int(item[2])),
            deterministic_key(seed, edge_id, family, *item[0]),
        ),
    )
    for key, count, _ in order:
        if remaining <= 0:
            break
        if allocation[key] < count:
            allocation[key] += 1
            remaining -= 1
    return allocation


def exact_local_matched_sample(
    candidates: Iterable[Mapping[str, Any]], seed: int, maximum_pairs: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        row = dict(raw)
        key = (
            str(row["edge_id"]),
            int(row["parent_votes"]),
            str(row["aspect_bucket"]),
            int(row["child_votes"]),
        )
        cells[key].append(row)
    selected: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for edge in EDGES:
        edge_rows: list[dict[str, Any]] = []
        family_diagnostics: dict[str, Any] = {}
        for family, vote_pair in (("negative_boundary", (0, 1)), ("positive_boundary", (2, 3))):
            available = {}
            for parent_votes in (2, 3):
                for aspect in ("portrait", "wide", "square"):
                    available[(parent_votes, aspect)] = min(
                        len(cells[(edge.edge_id, parent_votes, aspect, vote_pair[0])]),
                        len(cells[(edge.edge_id, parent_votes, aspect, vote_pair[1])]),
                    )
            available_pairs = sum(available.values())
            target_pairs = min(maximum_pairs, available_pairs)
            allocation = _allocate_pairs(
                available, target_pairs, seed, edge.edge_id, family
            )
            for (parent_votes, aspect), take in sorted(allocation.items()):
                pools = []
                for child_votes in vote_pair:
                    pool = cells[(edge.edge_id, parent_votes, aspect, child_votes)]
                    pools.append(
                        sorted(
                            pool,
                            key=lambda row: deterministic_key(
                                seed,
                                edge.edge_id,
                                family,
                                parent_votes,
                                aspect,
                                child_votes,
                                row["image_id"],
                            ),
                        )[:take]
                    )
                for pair_index, (left, right) in enumerate(zip(*pools)):
                    pair_id = hashlib.sha256(
                        f"{VERSION}:{edge.edge_id}:{family}:{parent_votes}:{aspect}:"
                        f"{left['image_id']}:{right['image_id']}".encode()
                    ).hexdigest()
                    for row in (left, right):
                        row["comparison_family"] = family
                        row["matched_pair_id"] = pair_id
                        row["ambiguity_state"] = "ambiguous" if row["child_votes"] in {1, 2} else "clear"
                        edge_rows.append(row)
            family_diagnostics[family] = {
                "vote_pair": list(vote_pair),
                "available_exact_matched_pairs": available_pairs,
                "selected_exact_matched_pairs": target_pairs,
                "allocation_by_parent_votes_and_aspect": {
                    f"{key[0]}of3:{key[1]}": value
                    for key, value in sorted(allocation.items())
                },
            }
        selected.extend(edge_rows)
        diagnostics[edge.edge_id] = {
            "selected_rows": len(edge_rows),
            "families": family_diagnostics,
        }
    return sorted(selected, key=lambda row: (row["edge_id"], row["image_id"])), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--discovery-generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--seed", type=int, default=19031)
    parser.add_argument("--maximum-pairs-per-edge", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    excluded = {str(row["image_id"]) for row in load_jsonl(args.discovery_generations)}
    candidates = candidate_rows(load_reader_panel(args.labels), excluded, args.image_root)
    rows, diagnostics = exact_local_matched_sample(
        candidates, args.seed, args.maximum_pairs_per_edge
    )
    primary_families = diagnostics[EDGES[0].edge_id]["families"]
    if min(value["selected_exact_matched_pairs"] for value in primary_families.values()) < 90:
        raise ValueError("primary edge has fewer than 90 exact local matched pairs")
    for edge in EDGES[1:]:
        families = diagnostics[edge.edge_id]["families"]
        if min(value["selected_exact_matched_pairs"] for value in families.values()) < 30:
            raise ValueError(f"replication edge {edge.edge_id} has fewer than 30 local pairs")
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    config = {
        "version": VERSION,
        "status": "untouched_confirmatory_census_gpu_not_run",
        "seed": args.seed,
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256_file(args.labels),
        "discovery_generations": str(args.discovery_generations.resolve()),
        "discovery_generations_sha256": sha256_file(args.discovery_generations),
        "discovery_images_excluded": len(excluded),
        "image_root": str(args.image_root.resolve()),
        "reader_panel": sorted(FROZEN_PANEL),
        "selection_uses_model_output": False,
        "selection": (
            "parent support >=2/3; all four child-support bins retained; exact local "
            "0/3-vs-1/3 and 2/3-vs-3/3 matching on edge, parent vote count, and "
            "released-DICOM aspect bucket"
        ),
        "view_position_note": (
            "ViewPosition is absent from the released VinDr DICOMs; aspect bucket is "
            "frozen as the available acquisition-shape control"
        ),
        "prompts": list(PROMPTS),
        "markers": list(MARKERS),
        "edges": [edge.__dict__ for edge in EDGES],
        "primary_commitment_coordinate": (
            "0.5*(final_logit[' present']+final_logit[' unlikely'])-"
            "final_logit[' possible']"
        ),
        "polarity_control": "final_logit[' present']-final_logit[' unlikely']",
        "primary_estimand": (
            "0.5 * ((deltaK_1of3-deltaK_0of3) + "
            "(deltaK_2of3-deltaK_3of3)), where deltaK is the within-image "
            "existential-minus-neutral commitment shift averaged over frozen prompt pairs"
        ),
        "primary_edge": EDGES[0].edge_id,
        "replication_edge": EDGES[1].edge_id,
        "gates": {
            "primary_census_image_cluster_bootstrap_ci_excludes_zero": True,
            "both_local_ambiguity_contrasts_positive": True,
            "both_prompt_pairs_same_direction": True,
            "at_least_one_replication_edge_same_direction": True,
            "polarity_interaction_90pct_ci_within_log_odds_minus0p2_plus0p2": True,
            "generic_prompt_shift_without_support_interaction_is_failure": True,
            "text_only_or_same_support_swap_equal_effect_is_failure": True,
            "minimum_parser_precision_for_oe_external_validity": 0.90,
            "minimum_physician_weighted_kappa_for_oe_external_validity": 0.80,
        },
        "diagnostics": diagnostics,
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
    }
    config["fingerprint"] = canonical_hash(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "selected_manifest.jsonl"
    config_path = args.output_dir / "substrate_config.json"
    if manifest_path.exists() or config_path.exists():
        raise FileExistsError("controlled substrate is write-once")
    atomic_write(manifest_path, manifest_text)
    atomic_write(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
