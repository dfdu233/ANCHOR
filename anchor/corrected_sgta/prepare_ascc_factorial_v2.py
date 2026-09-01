#!/usr/bin/env python3
"""Freeze ASCC-v2: symmetric three-state markers and a true 2x2 prompt design."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .audit_diagnostic_completion_substrate_v1 import DEFAULT_LABELS, sha256_file
from .prepare_ascc_interaction_v1 import (
    DEFAULT_IMAGE_ROOT,
    EDGES,
    atomic_write,
    candidate_rows,
    load_reader_panel,
)
from .prepare_pragmatic_commitment_confirmatory_v1 import canonical_hash


VERSION = "ascc-symmetric-factorial-substrate-v2"
DEFAULT_DISCOVERY_MANIFEST = Path(
    "/home/dbw/ANCHOR/corrected_runs/vindr_v2/"
    "clinical_presupposition_huatuo_generation_v1/selected_manifest.jsonl"
)
MARKERS = (" absent", " uncertain", " present")
COMMON_INSTRUCTION = (
    " Present, absent, and uncertain states are all valid. Complete the fixed "
    "report fragment using exactly one of: absent, uncertain, or present."
)
PROMPTS = (
    {
        "name": "describe_findings",
        "speech_act": "describe",
        "clinical_noun": "findings",
        "prompt": "Describe the image-grounded findings on this chest X-ray."
        + COMMON_INSTRUCTION,
    },
    {
        "name": "describe_abnormalities",
        "speech_act": "describe",
        "clinical_noun": "abnormalities",
        "prompt": "Describe the image-grounded abnormalities on this chest X-ray."
        + COMMON_INSTRUCTION,
    },
    {
        "name": "list_findings",
        "speech_act": "list",
        "clinical_noun": "findings",
        "prompt": "List the image-grounded findings on this chest X-ray."
        + COMMON_INSTRUCTION,
    },
    {
        "name": "list_abnormalities",
        "speech_act": "list",
        "clinical_noun": "abnormalities",
        "prompt": "List the image-grounded abnormalities on this chest X-ray."
        + COMMON_INSTRUCTION,
    },
)


def load_discovery_ids(path: Path) -> set[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = {str(row["image_id"]) for row in rows}
    if len(ids) != 200:
        raise ValueError(f"frozen discovery manifest must contain 200 unique images, got {len(ids)}")
    return ids


def validate_factorial() -> None:
    cells = {(row["speech_act"], row["clinical_noun"]) for row in PROMPTS}
    expected = {
        (speech_act, clinical_noun)
        for speech_act in ("describe", "list")
        for clinical_noun in ("findings", "abnormalities")
    }
    if cells != expected or len(PROMPTS) != 4:
        raise ValueError("ASCC-v2 prompts are not a complete 2x2 factorial")
    for speech_act in ("describe", "list"):
        pair = [row for row in PROMPTS if row["speech_act"] == speech_act]
        left, right = sorted(pair, key=lambda row: row["clinical_noun"])
        normalized_left = left["prompt"].replace(left["clinical_noun"], "<NOUN>")
        normalized_right = right["prompt"].replace(right["clinical_noun"], "<NOUN>")
        if normalized_left != normalized_right:
            raise ValueError("noun contrast changes more than the frozen clinical noun")
    for clinical_noun in ("findings", "abnormalities"):
        pair = [row for row in PROMPTS if row["clinical_noun"] == clinical_noun]
        describe = next(row for row in pair if row["speech_act"] == "describe")
        listed = next(row for row in pair if row["speech_act"] == "list")
        normalized_describe = describe["prompt"].replace("Describe", "<ACT>", 1)
        normalized_list = listed["prompt"].replace("List", "<ACT>", 1)
        if normalized_describe != normalized_list:
            raise ValueError("speech-act contrast changes more than the frozen initial verb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--discovery-manifest", type=Path, default=DEFAULT_DISCOVERY_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    validate_factorial()
    excluded = load_discovery_ids(args.discovery_manifest)
    rows = candidate_rows(load_reader_panel(args.labels), excluded, args.image_root)
    rows = sorted(rows, key=lambda row: (row["edge_id"], row["image_id"]))
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate ASCC-v2 image-edge identity")
    counts = Counter((row["edge_id"], int(row["child_votes"])) for row in rows)
    primary = EDGES[0].edge_id
    if min(counts[(primary, vote)] for vote in range(4)) < 100:
        raise ValueError("primary edge requires at least 100 images in every reader bin")
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    config = {
        "version": VERSION,
        "status": "untouched_confirmatory_census_gpu_not_run",
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256_file(args.labels),
        "discovery_manifest": str(args.discovery_manifest.resolve()),
        "discovery_manifest_sha256": sha256_file(args.discovery_manifest),
        "discovery_images_excluded": len(excluded),
        "discovery_exclusion_uses_generation_output": False,
        "selection_uses_model_output": False,
        "image_root": str(args.image_root.resolve()),
        "selection": (
            "complete discovery-disjoint exact R8/R9/R10 census with parent support "
            ">=2/3; all child vote bins 0/3,1/3,2/3,3/3 retained"
        ),
        "edges": [edge.__dict__ for edge in EDGES],
        "prompts": list(PROMPTS),
        "factorial": {
            "primary_factor": "clinical_noun: abnormalities minus findings",
            "orthogonal_control_factor": "speech_act: list minus describe",
            "complete_2x2": True,
        },
        "markers": list(MARKERS),
        "marker_semantics": {
            " absent": "definite negative state",
            " uncertain": "explicit third/undetermined state",
            " present": "definite positive state",
        },
        "primary_commitment": (
            "logsumexp(z_present,z_absent)-z_uncertain over the restricted "
            "symmetric three-state instrument"
        ),
        "polarity_control": "z_present-z_absent",
        "primary_estimand": (
            "0.5*((deltaC_noun_1-deltaC_noun_0)+"
            "(deltaC_noun_2-deltaC_noun_3))"
        ),
        "mandatory_admission": (
            "under findings prompts, uncertainty preference must rise from 0/3 to 1/3 "
            "and from 3/3 to 2/3, with separate stratified bootstrap CIs above zero"
        ),
        "mandatory_controls": [
            "separate local interactions and ambiguous-bin noun shifts",
            "polarity local-equivalence without cancellation",
            "clear-bin cross-fit affine/temperature residual interaction",
            "text-only and same/cross-support image swaps after primary screen",
            "restricted-marker mass/top-1 conformance diagnostics",
        ],
        "promotion_ceiling": (
            "primary screen only; no global ASCC, hallucination, OE, or mitigation "
            "claim before second model, controls, natural output, and physician review"
        ),
        "counts_by_edge_and_child_votes": {
            f"{edge}:{vote}of3": counts[(edge, vote)]
            for edge in sorted({row["edge_id"] for row in rows})
            for vote in range(4)
        },
        "registered_rows": len(rows),
        "registered_jobs_per_model": len(rows) * len(PROMPTS),
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
    }
    config["fingerprint"] = canonical_hash(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "substrate_config.json"
    manifest_path = args.output_dir / "selected_manifest.jsonl"
    if config_path.exists() or manifest_path.exists():
        raise FileExistsError("ASCC-v2 substrate is write-once")
    atomic_write(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    atomic_write(manifest_path, manifest_text)
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
