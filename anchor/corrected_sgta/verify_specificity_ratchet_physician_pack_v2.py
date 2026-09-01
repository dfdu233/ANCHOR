#!/usr/bin/env python3
"""Fail-closed verifier for the Specificity Ratchet physician pack v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


GENERATED_FILES = (
    "candidates.blinded.jsonl",
    "provenance.private.jsonl",
    "annotations.reviewer_1.csv",
    "annotations.reviewer_2.csv",
    "adjudication.csv",
    "annotation_schema.json",
    "summary.json",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2"),
    )
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    pack = (repo / args.pack).resolve() if not args.pack.is_absolute() else args.pack

    candidates = read_jsonl(pack / "candidates.blinded.jsonl")
    private = read_jsonl(pack / "provenance.private.jsonl")
    summary = json.loads((pack / "summary.json").read_text())
    schema = json.loads((pack / "annotation_schema.json").read_text())
    if len(candidates) != len(private) or not candidates:
        raise ValueError("candidate/private count mismatch or empty pack")
    if len({row["edge_id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate reviewer edge_id")
    if len({row["edge_id"] for row in private}) != len(private):
        raise ValueError("duplicate private edge_id")
    private_by_edge = {row["edge_id"]: row for row in private}
    per_case = Counter(row["case_id"] for row in candidates)
    if max(per_case.values()) > 3:
        raise ValueError("more than three edges on one image")
    if len(per_case) != summary["selection"]["selected_images"]:
        raise ValueError("selected image count mismatch")
    if len(candidates) != summary["n_edges"]:
        raise ValueError("edge count mismatch")

    forbidden_reviewer_keys = {
        "model_id",
        "source_model",
        "question_id",
        "gt_ans",
        "answer",
        "reference",
        "source_row",
        "same_type_models_screening_only",
    }
    answer_cache: dict[Path, list[dict]] = {}
    case_to_image: dict[str, str] = {}
    for candidate in candidates:
        leaked = forbidden_reviewer_keys.intersection(candidate)
        if leaked:
            raise ValueError(f"reviewer leak {candidate['edge_id']}: {sorted(leaked)}")
        if candidate.get("proposal_only") is not True:
            raise ValueError(f"proposal_only missing: {candidate['edge_id']}")
        source = private_by_edge[candidate["edge_id"]]
        if source["case_id"] != candidate["case_id"]:
            raise ValueError("private case join mismatch")
        source_path = repo / source["source_answer_path"]
        if source_path not in answer_cache:
            answer_cache[source_path] = read_jsonl(source_path)
        source_answer = answer_cache[source_path][source["source_answer_line"] - 1]
        if source_answer["question_id"] != source["question_id"]:
            raise ValueError(f"line/qid mismatch: {candidate['edge_id']}")
        for key in ("answer_span", "child_proposal"):
            if candidate[key] not in source_answer["text"]:
                raise ValueError(f"non-exact {key}: {candidate['edge_id']}")
        image_name = Path(candidate["image_relpath"]).name
        if image_name != source["image_name"]:
            raise ValueError(f"image join mismatch: {candidate['edge_id']}")
        image_path = repo.parent / "datasets/public/vqa_rad_hf" / candidate["image_relpath"]
        if not image_path.is_file():
            raise ValueError(f"missing image: {image_path}")
        prior_image = case_to_image.setdefault(candidate["case_id"], image_name)
        if prior_image != image_name:
            raise ValueError("case_id maps to multiple images")

    for model, expected in summary["source_hashes"].items():
        if model == "manifest":
            path = repo / "corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json"
        else:
            path = {
                "huatuo": repo / "corrected_runs/unified_eval/full/huatuo_native_vqa_rad_oe_v3_512/answers.jsonl",
                "hulu": repo / "corrected_runs/unified_eval/full/hulu_native_vqa_rad_oe_v1/answers.jsonl",
                "llava": repo / "corrected_runs/unified_eval/full/llava_native_vqa_rad_oe_v1/answers.jsonl",
            }[model]
        if sha256(path) != expected:
            raise ValueError(f"source hash mismatch: {model}")

    for reviewer in (1, 2):
        with (pack / f"annotations.reviewer_{reviewer}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(candidates):
            raise ValueError(f"reviewer {reviewer} row count mismatch")
        if [row["edge_id"] for row in rows] != [row["edge_id"] for row in candidates]:
            raise ValueError(f"reviewer {reviewer} ordering mismatch")
        for row in rows:
            for field in (
                "reviewer_id",
                "edge_entailment_admitted",
                "parent_visual_support",
                "child_visual_support",
                "increment_observability",
                "reviewer_confidence",
                "rationale",
            ):
                if row[field]:
                    raise ValueError(f"reviewer template is not blank: {field}")

    with (pack / "adjudication.csv").open(newline="") as handle:
        adjudication = list(csv.DictReader(handle))
    if len(adjudication) != len(candidates):
        raise ValueError("adjudication row count mismatch")
    if "undetermined" not in schema["fields"]["parent_visual_support"]:
        raise ValueError("annotation schema omits undetermined")
    if "unobservable" not in schema["fields"]["child_visual_support"]:
        raise ValueError("annotation schema omits unobservable")

    if not args.skip_rebuild:
        with tempfile.TemporaryDirectory(prefix="specificity-ratchet-v2-") as tmp:
            rebuilt = Path(tmp) / "pack"
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "anchor/corrected_sgta/build_specificity_ratchet_physician_pack_v2.py"),
                    "--repo-root",
                    str(repo),
                    "--output",
                    str(rebuilt),
                    "--target-images",
                    str(summary["selection"]["requested_images"]),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            mismatched = [
                name for name in GENERATED_FILES if sha256(pack / name) != sha256(rebuilt / name)
            ]
            if mismatched:
                raise ValueError(f"non-reproducible generated files: {mismatched}")

    print(
        json.dumps(
            {
                "status": "passed",
                "unique_images": len(per_case),
                "edges": len(candidates),
                "max_edges_per_image": max(per_case.values()),
                "source_hashes_verified": True,
                "exact_substrings_verified": True,
                "reviewer_blinding_verified": True,
                "templates_verified": True,
                "deterministic_rebuild_verified": not args.skip_rebuild,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
