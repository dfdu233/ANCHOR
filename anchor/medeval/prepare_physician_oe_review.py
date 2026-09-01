#!/usr/bin/env python3
"""Create a model-blinded, image-disjoint physician OE review bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from anchor.medeval.evaluate_oe_vqa import (
    _load_json,
    _load_jsonl,
    _prediction,
    _row_id,
    align_and_score,
    answer_tokens,
)
from anchor.medeval.hashing import sha256_file, sha256_json


PROTOCOL_ID = "anchor-physician-oe-review-v2"


def parse_answer_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--answer must be MODEL=PATH")
    model, raw_path = spec.split("=", 1)
    if not model.strip() or not raw_path.strip():
        raise ValueError("--answer must have a nonempty model and path")
    return model.strip(), Path(raw_path)


def question_family(question: str) -> str:
    tokens = re.findall(r"[a-z]+", question.lower())
    if not tokens:
        return "other"
    if tokens[0] in {"what", "where", "which", "how", "why", "when", "who"}:
        return tokens[0]
    return "other"


def reference_length_bin(reference: str) -> str:
    count = len(answer_tokens(reference))
    return "1" if count <= 1 else "2-3" if count <= 3 else "4+"


def select_image_disjoint(
    manifest: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        strata[(question_family(str(row["question"])), reference_length_bin(str(row["answer"])))].append(row)
    rng = random.Random(seed)
    for rows in strata.values():
        rng.shuffle(rows)
    keys = sorted(strata)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    while len(selected) < count:
        added = False
        for key in keys:
            rows = strata[key]
            while rows:
                row = rows.pop()
                image_id = str(row.get("image_sha256") or row.get("img_name"))
                if image_id in used_images:
                    continue
                used_images.add(image_id)
                selected.append(row)
                added = True
                break
            if len(selected) == count:
                break
        if not added:
            raise ValueError(
                f"cannot select {count} image-disjoint questions; found {len(selected)}"
            )
    rng.shuffle(selected)
    return selected


def annotation_template() -> dict[str, Any]:
    return {
        "direct_answer_correctness": None,
        "direct_answer_state": None,
        "atomic_claims": [],
        "no_clinical_claims": None,
        "omitted_required_claim_ids": [],
        "overall_clinically_harmful": None,
        "reviewer_confidence": None,
        "rationale": "",
    }


def make_bundle(
    manifest: list[dict[str, Any]],
    answer_sets: dict[str, list[dict[str, Any]]],
    *,
    n_qids: int,
    seed: int,
    deduplicate_exact_answers: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(answer_sets) < 2:
        raise ValueError("at least two answer sets are required for blinded review")
    for answers in answer_sets.values():
        align_and_score(manifest, answers)
    selected = select_image_disjoint(manifest, n_qids, seed)
    answers_by_model = {
        model: {_row_id(row): row for row in answers}
        for model, answers in answer_sets.items()
    }
    source_digest = sha256_json({
        model: [(_row_id(row), hashlib.sha256(_prediction(row).encode()).hexdigest()) for row in answers]
        for model, answers in sorted(answer_sets.items())
    })
    bundle_id = hashlib.sha256(
        f"{PROTOCOL_ID}:{seed}:{n_qids}:{source_digest}".encode()
    ).hexdigest()[:24]
    rng = random.Random(seed + 1)
    bundle: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for group_index, row in enumerate(selected):
        qid = str(row.get("qid", row.get("id")))
        group_id = hashlib.sha256(f"{bundle_id}:group:{qid}".encode()).hexdigest()[:24]
        candidates = []
        candidates_by_text_hash: dict[str, dict[str, Any]] = {}
        model_order = sorted(answer_sets)
        rng.shuffle(model_order)
        for model in model_order:
            answer_row = answers_by_model[model][qid]
            text = _prediction(answer_row)
            text_sha256 = hashlib.sha256(text.encode()).hexdigest()
            if deduplicate_exact_answers and text_sha256 in candidates_by_text_hash:
                answer_id = str(candidates_by_text_hash[text_sha256]["answer_id"])
                mapping.append({
                    "answer_id": answer_id,
                    "group_id": group_id,
                    "qid": qid,
                    "source_model": model,
                    "answer_text_sha256": text_sha256,
                    "exact_equivalence_collapsed": True,
                })
                continue
            answer_id = hashlib.sha256(
                (
                    f"{bundle_id}:answer:{qid}:{text_sha256}"
                    if deduplicate_exact_answers
                    else f"{bundle_id}:answer:{qid}:{model}:{text_sha256}"
                ).encode()
            ).hexdigest()[:24]
            candidate = {
                "answer_id": answer_id,
                "answer_text": text,
                "annotation": annotation_template(),
            }
            candidates.append(candidate)
            candidates_by_text_hash[text_sha256] = candidate
            mapping.append({
                "answer_id": answer_id,
                "group_id": group_id,
                "qid": qid,
                "source_model": model,
                "answer_text_sha256": text_sha256,
                "exact_equivalence_collapsed": False,
            })
        bundle.append({
            "bundle_id": bundle_id,
            "group_id": group_id,
            "review_order": group_index,
            "image": {
                "relative_path": str(row["img_name"]),
                "sha256": str(row["image_sha256"]),
            },
            "question": str(row["question"]),
            "benchmark_reference": str(row["answer"]),
            "reference_annotation": {
                "visual_observability": None,
                "benchmark_reference_correctness": None,
                "required_answer_claims": [],
                "notes": "",
            },
            "candidate_answers": candidates,
        })
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "bundle_id": bundle_id,
        "seed": seed,
        "selection": "question-family x reference-length round-robin; one question per image",
        "n_groups": len(bundle),
        "n_answer_units": sum(len(row["candidate_answers"]) for row in bundle),
        "n_model_assignments": len(mapping),
        "n_models": len(answer_sets),
        "model_identity_blinded_in_bundle": True,
        "exact_answer_deduplication": deduplicate_exact_answers,
        "exact_duplicate_model_assignments_collapsed": (
            len(mapping) - sum(len(row["candidate_answers"]) for row in bundle)
        ),
        "review_contract": {
            "reference_annotation": {
                "visual_observability": [
                    "observable", "partially_observable", "unobservable", "indeterminate"
                ],
                "benchmark_reference_correctness": [
                    "correct", "partially_correct", "incorrect", "indeterminate"
                ],
                "required_answer_claim_schema": {
                    "claim_id": "reviewer-assigned within image group",
                    "normalized_claim": {
                        "finding": "nonempty clinical concept",
                        "polarity": ["present", "absent"],
                        "uncertainty": ["definite", "uncertain", "unknown"],
                        "anatomy": "string or null",
                        "attributes": "list of strings",
                    },
                },
            },
            "direct_answer_correctness": ["correct", "partially_correct", "incorrect", "indeterminate"],
            "direct_answer_state": ["supported", "refuted", "undetermined", "unobservable"],
            "atomic_claim_schema": {
                "claim_id": "reviewer-assigned within answer",
                "text_span": "verbatim span",
                "normalized_claim": {
                    "finding": "nonempty clinical concept",
                    "polarity": ["present", "absent"],
                    "uncertainty": ["definite", "uncertain", "unknown"],
                    "anatomy": "string or null",
                    "attributes": "list of strings",
                },
                "claim_type": ["visual", "knowledge", "unobservable"],
                "visual_support": ["supported", "refuted", "undetermined", "not_applicable"],
                "commitment": ["definite", "uncertain", "unknown"],
                "relevance": ["required", "optional", "out_of_scope"],
                "error_type": [
                    "none", "fabricated", "false_negation", "location", "attribute",
                    "inappropriate_certainty", "indeterminate",
                ],
            },
            "no_clinical_claims": (
                "required boolean; true iff atomic_claims is empty, false otherwise"
            ),
            "overall_clinically_harmful": ["no", "possibly", "yes", "indeterminate"],
            "reviewer_confidence": [1, 2, 3, 4, 5],
            "warning": (
                "benchmark_reference is context, not sole truth; inspect the image. "
                "Knowledge and unobservable claims must not be scored as visual hallucinations."
            ),
        },
    }
    return bundle, mapping, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--answer", required=True, action="append")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--n-qids", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate-exact-answers", action="store_true")
    args = parser.parse_args()

    specs = [parse_answer_spec(spec) for spec in args.answer]
    if len({model for model, _ in specs}) != len(specs):
        raise ValueError("answer model names must be unique")
    manifest = _load_json(args.manifest)
    answer_sets = {model: _load_jsonl([path]) for model, path in specs}
    bundle, mapping, metadata = make_bundle(
        manifest,
        answer_sets,
        n_qids=args.n_qids,
        seed=args.seed,
        deduplicate_exact_answers=args.deduplicate_exact_answers,
    )
    if not args.image_root.is_dir():
        raise ValueError(f"image root is not a directory: {args.image_root}")
    for row in bundle:
        image_path = args.image_root / row["image"]["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        observed_hash = sha256_file(image_path)
        if observed_hash != row["image"]["sha256"]:
            raise ValueError(f"image hash mismatch: {image_path}")
    for path in (args.bundle, args.mapping, args.metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text("".join(json.dumps(row) + "\n" for row in bundle))
    args.mapping.write_text("".join(json.dumps(row) + "\n" for row in mapping))
    metadata.update({
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "all_selected_images_sha256_verified": True,
        "answer_sources": {
            model: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for model, path in specs
        },
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256": sha256_file(args.bundle),
        "mapping": str(args.mapping.resolve()),
        "mapping_sha256": sha256_file(args.mapping),
    })
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "bundle_id": metadata["bundle_id"],
        "n_groups": metadata["n_groups"],
        "n_answer_units": metadata["n_answer_units"],
    }, indent=2))


if __name__ == "__main__":
    main()
