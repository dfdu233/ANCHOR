#!/usr/bin/env python3
"""Freeze a discovery-disjoint VinDr confirmatory generation substrate.

Selection uses only the independent R8/R9/R10 Lung Opacity and Pneumonia
votes.  It never uses model output.  Every eligible image is assigned to a
single balanced dev/test split before generation, and every selected image is
run under both frozen prompt conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit_diagnostic_completion_substrate_v1 import (
    DEFAULT_GENERATIONS,
    DEFAULT_LABELS,
    load_exact_panel_votes,
    load_jsonl,
    sha256_file,
)


VERSION = "pragmatic-commitment-confirmatory-substrate-v1"
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")
COMMON_RESPONSE_FORM = (
    "Answer in one concise sentence of at most 30 words. "
    "State uncertainty explicitly rather than guessing."
)
CONDITIONS = (
    {
        "name": "neutral",
        "prompt": (
            "Summarize the image-grounded findings on this chest X-ray, including "
            "clinically relevant present, absent, or uncertain observations. "
            + COMMON_RESPONSE_FORM
        ),
        "pragmatic_task": "unrestricted_image_grounded_summary",
        "answer_space_focus": "present_absent_or_uncertain_findings",
    },
    {
        "name": "existential",
        "prompt": (
            "What abnormalities are present on this chest X-ray? Mention the "
            "present or uncertain abnormalities that are clinically relevant. "
            + COMMON_RESPONSE_FORM
        ),
        "pragmatic_task": "existential_abnormality_listing",
        "answer_space_focus": "present_or_uncertain_abnormalities",
    },
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def discovery_ids(generations: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row["image_id"]) for row in generations}


def balanced_split(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(int(row["lung_opacity_votes"]), int(row["pneumonia_votes"]))].append(
            row
        )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    assigned = []
    for stratum, items in sorted(strata.items()):
        ordered = sorted(
            items,
            key=lambda row: hashlib.sha256(
                f"{VERSION}:{seed}:{row['image_id']}".encode()
            ).hexdigest(),
        )
        offset = int(
            hashlib.sha256(f"{VERSION}:{seed}:{stratum}".encode()).hexdigest(), 16
        ) % 2
        for index, row in enumerate(ordered):
            split = "dev" if (index + offset) % 2 == 0 else "test"
            item = {**row, "experiment_split": split}
            assigned.append(item)
            counts[f"{stratum[0]}of3_{stratum[1]}of3"][split] += 1
    diagnostics = {
        stratum: dict(counter) for stratum, counter in sorted(counts.items())
    }
    for stratum, counter in diagnostics.items():
        if abs(int(counter.get("dev", 0)) - int(counter.get("test", 0))) > 1:
            raise AssertionError(f"unbalanced reader stratum: {stratum}")
    return sorted(assigned, key=lambda row: row["image_id"]), diagnostics


def prepare(
    votes: Mapping[str, Mapping[str, int]],
    excluded_ids: set[str],
    image_root: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    missing_images = []
    for image_id, image_votes in votes.items():
        opacity = int(image_votes["Lung Opacity"])
        pneumonia = int(image_votes["Pneumonia"])
        if image_id in excluded_ids or (opacity == 0 and pneumonia == 0):
            continue
        image_path = image_root / f"{image_id}.dicom"
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        candidates.append(
            {
                "image_id": image_id,
                "item_id": image_id,
                "dicom_relpath": f"train/{image_id}.dicom",
                "lung_opacity_votes": opacity,
                "pneumonia_votes": pneumonia,
                "reader_panel": ["R8", "R9", "R10"],
                "selection_reason": "lung_opacity_or_pneumonia_at_least_1of3",
                "selection_uses_reader_labels": True,
                "discovery_image_excluded": True,
            }
        )
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} eligible DICOMs are missing; first={missing_images[0]}"
        )
    assigned, strata = balanced_split(candidates, seed)
    split_counts = Counter(str(row["experiment_split"]) for row in assigned)
    if min(split_counts.values(), default=0) < 500:
        raise ValueError(f"confirmatory splits unexpectedly small: {dict(split_counts)}")
    diagnostics = {
        "eligible_after_discovery_exclusion": len(assigned),
        "split_counts": dict(split_counts),
        "reader_stratum_split_counts": strata,
        "discovery_excluded_count": len(excluded_ids),
    }
    return assigned, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--discovery-generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    discovery = load_jsonl(args.discovery_generations)
    rows, diagnostics = prepare(
        load_exact_panel_votes(args.labels),
        discovery_ids(discovery),
        args.image_root,
        args.seed,
    )
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    config = {
        "version": VERSION,
        "seed": args.seed,
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256_file(args.labels),
        "discovery_generations": str(args.discovery_generations.resolve()),
        "discovery_generations_sha256": sha256_file(args.discovery_generations),
        "image_root": str(args.image_root.resolve()),
        "reader_panel": ["R8", "R9", "R10"],
        "selection": (
            "exact-panel images with Lung Opacity>=1/3 or Pneumonia>=1/3, "
            "excluding every image in the discovery generation"
        ),
        "selection_uses_reader_labels": True,
        "selection_uses_model_output": False,
        "split": "reader-stratum-balanced image-disjoint dev/test",
        "prompt_conditions": list(CONDITIONS),
        "generation_jobs": len(rows) * len(CONDITIONS),
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "diagnostics": diagnostics,
        "frozen_primary_identity": (
            "same image + Lung Opacity parent + Pneumonia child + normalized "
            "observation prefix across neutral/existential outputs"
        ),
        "frozen_primary_gates": {
            "minimum_admitted_pairs_per_split": 20,
            "maximum_local_sentence_word_gap": 4,
            "physician_blinded_commitment_order_required": True,
            "upward_minus_downward_pair_rate_ci_low_above_zero": True,
            "effect_must_hold_in_dev_and_untouched_test": True,
            "whole_answer_length_and_claim_count_reported_as_nuisance": True,
            "confirmatory_hidden_state_replay_before_behavioral_gate": False,
        },
        "claim_ceiling": (
            "radiograph-attributable reader diagnostic impression; no patient-level "
            "etiology or clinical-context truth"
        ),
    }
    config["fingerprint"] = canonical_hash(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "selected_manifest.jsonl"
    config_path = args.output_dir / "substrate_config.json"
    if manifest_path.exists() or config_path.exists():
        raise FileExistsError("confirmatory substrate is write-once")
    atomic_write(manifest_path, manifest_text)
    atomic_write(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
