#!/usr/bin/env python3
"""Resumable, task-agnostic ANCHOR candidate generation and transport scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from corrected_sgta.anchor_models import NULL_RGB, load_anchor_adapter
from corrected_sgta.anchor_transport import (
    DEFAULT_NEIGHBORS,
    DEFAULT_PROJECTIONS,
    DEFAULT_PROJECTION_SEED,
    DEFAULT_QUANTILES,
    FEATURE_NAMES,
    VERSION,
    deterministic_directions,
    file_sha256,
    load_json_or_jsonl,
    model_artifact_fingerprint,
    nearest_source_indices,
    normalize_manifest_record,
    normalize_trajectory,
    resolve_image_path,
    select_candidate,
    source_frechet_energy,
    stable_json_sha256,
)
from corrected_sgta.cache import repair_truncated_jsonl_tail


RUN_VERSION = "anchor-transport-inference-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_bank(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("method_version") != VERSION:
        raise RuntimeError("source bank method-version mismatch")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("source bank evidence-feature mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source bank contains no records")
    if payload.get("source_answer_text_stored") is not False:
        raise RuntimeError("source bank must not retain source answer text")
    return payload


def per_item_seed(base_seed: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{identifier}".encode()).hexdigest()
    return int(digest[:8], 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate complete natural-language candidates and score them with "
            "one source-native evidence geometry for every task."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--model", choices=("hulu", "llava"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--candidate-batch", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--projections", type=int, default=DEFAULT_PROJECTIONS)
    parser.add_argument("--quantiles", type=int, default=DEFAULT_QUANTILES)
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.0,
        help="Provisional offline selection only; all lambda-independent terms are cached.",
    )
    parser.add_argument(
        "--exclude-source-domain",
        help="Source-only LODO validation control; never set for unknown target deployment.",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_budget != 8:
        raise ValueError("ANCHOR v1 fixes the equal generation budget at 8")
    if (
        args.max_new_tokens <= 0
        or args.candidate_batch <= 0
        or args.neighbors <= 0
        or args.projections <= 0
        or args.quantiles <= 0
        or args.max_samples < 0
        or args.lambda_value < 0
    ):
        raise ValueError("generation, geometry, and lambda parameters are invalid")
    bank = load_bank(args.bank)
    if bank.get("model") != args.model:
        raise RuntimeError("source bank and inference model differ")
    rows = [
        normalize_manifest_record(
            row, require_answer=False, default_domain="unknown"
        )
        for row in load_json_or_jsonl(args.manifest)
    ]
    if args.max_samples:
        rows = rows[: args.max_samples]
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("target manifest contains duplicate ids")
    resolved_images = {}
    image_hash_cache = {}
    target_image_sha256 = []
    for row in rows:
        image_path = resolve_image_path(row["image"], args.image_root)
        resolved_images[row["id"]] = image_path
        path_key = str(image_path)
        if path_key not in image_hash_cache:
            image_hash_cache[path_key] = file_sha256(image_path)
        target_image_sha256.append(
            {"id": row["id"], "sha256": image_hash_cache[path_key]}
        )

    model_path = args.model_path
    if model_path is None:
        from corrected_sgta.models import HULU_PATH, LLAVA_PATH

        model_path = HULU_PATH if args.model == "hulu" else LLAVA_PATH
    artifact = model_artifact_fingerprint(model_path)
    if artifact["fingerprint"] != bank["model_artifacts"]["fingerprint"]:
        raise RuntimeError("loaded model artifacts do not match the source bank")

    retained_bank_indices = [
        index
        for index, record in enumerate(bank["records"])
        if record["domain"] != args.exclude_source_domain
    ]
    if not retained_bank_indices:
        raise ValueError("source-domain exclusion removed the entire bank")
    bank_records = [bank["records"][index] for index in retained_bank_indices]
    source_embeddings = np.asarray(
        [record["embedding"] for record in bank_records], dtype=np.float64
    )
    source_trajectories = [
        np.asarray(record["trajectory"], dtype=np.float64)
        for record in bank_records
    ]
    directions = deterministic_directions(
        dimension=len(FEATURE_NAMES),
        count=args.projections,
        seed=args.projection_seed,
    )
    fingerprint_payload = {
        "version": RUN_VERSION,
        "method_version": VERSION,
        "manifest_sha256": file_sha256(args.manifest),
        "target_image_sha256": target_image_sha256,
        "bank_sha256": file_sha256(args.bank),
        "bank_fingerprint": bank["fingerprint"],
        "model": args.model,
        "model_artifact_fingerprint": artifact["fingerprint"],
        "candidate_budget": args.candidate_budget,
        "candidate_policy": "one_greedy_plus_seven_seeded_nucleus_samples",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "candidate_batch": args.candidate_batch,
        "max_sequence_tokens": bank["max_sequence_tokens"],
        "null_rgb": NULL_RGB,
        "neighbors": args.neighbors,
        "projections": args.projections,
        "quantiles": args.quantiles,
        "projection_seed": args.projection_seed,
        "exclude_source_domain": args.exclude_source_domain,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "target_domain_used_for_selection": False,
        "target_labels_used_for_selection": False,
        "source_answer_text_retrieved": False,
        "code_sha256": {
            "runner": file_sha256(Path(__file__)),
            "core": file_sha256(
                Path(__file__).with_name("anchor_transport.py")
            ),
            "models": file_sha256(Path(__file__).with_name("anchor_models.py")),
        },
    }
    fingerprint = stable_json_sha256(fingerprint_payload)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if args.raw.exists():
        repair_truncated_jsonl_tail(args.raw)
    completed: dict[str, dict[str, Any]] = {}
    if args.raw.exists():
        for line in args.raw.read_text().splitlines():
            record = json.loads(line)
            if record.get("fingerprint") != fingerprint:
                raise RuntimeError("ANCHOR raw-cache fingerprint mismatch")
            if record["id"] in completed:
                raise RuntimeError(f"duplicate cached id={record['id']}")
            completed[record["id"]] = record
    if completed and not args.resume:
        raise FileExistsError("raw cache exists; use --resume")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)

    adapter = load_anchor_adapter(args.model, model_path)
    try:
        with args.raw.open("a") as handle:
            for row in tqdm(
                rows,
                desc=f"ANCHOR transport ({args.model})",
                initial=len(completed),
                total=len(rows),
            ):
                if row["id"] in completed:
                    continue
                image_path = resolved_images[row["id"]]
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                query_embedding = adapter.input_embedding(image, row["prompt"])
                neighbor_indices, similarities = nearest_source_indices(
                    query_embedding, source_embeddings, args.neighbors
                )
                retrieved_trajectories = [
                    source_trajectories[int(index)] for index in neighbor_indices
                ]
                item_seed = per_item_seed(args.seed, row["id"])
                candidates = adapter.generate_candidates(
                    image,
                    row["prompt"],
                    candidate_budget=args.candidate_budget,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    seed=item_seed,
                    candidate_batch=args.candidate_batch,
                )
                for candidate in candidates:
                    evidence = adapter.sequence_evidence(
                        image,
                        row["prompt"],
                        candidate["text"],
                        bank["max_sequence_tokens"],
                    )
                    normalized = normalize_trajectory(
                        evidence.trajectory, bank["feature_statistics"]
                    )
                    source_distance, individual_distances = source_frechet_energy(
                        normalized,
                        retrieved_trajectories,
                        directions=directions,
                        quantiles=args.quantiles,
                    )
                    candidate.update(evidence.to_json())
                    candidate["normalized_trajectory"] = normalized.astype(
                        float
                    ).tolist()
                    candidate["source_distance"] = source_distance
                    candidate["neighbor_distances"] = individual_distances
                selected_index, scores = select_candidate(
                    candidates, args.lambda_value
                )
                for candidate, score in zip(candidates, scores):
                    candidate["provisional_anchor_score"] = score
                neighbors = [
                    {
                        "domain": bank_records[int(index)]["domain"],
                        "id": bank_records[int(index)]["id"],
                        "similarity": float(similarity),
                    }
                    for index, similarity in zip(neighbor_indices, similarities)
                ]
                record = {
                    "version": RUN_VERSION,
                    "fingerprint": fingerprint,
                    "status": "ok",
                    "id": row["id"],
                    "image": row["image"],
                    "image_sha256": image_hash_cache[str(image_path)],
                    "prompt_sha256": stable_json_sha256(row["prompt"]),
                    "evaluation_group": row["domain"],
                    "item_seed": item_seed,
                    "neighbors": neighbors,
                    "candidates": candidates,
                    "unique_candidate_count": len(
                        {candidate["text"] for candidate in candidates}
                    ),
                    "provisional_lambda": args.lambda_value,
                    "selected_index": selected_index,
                    "selected_text": candidates[selected_index]["text"],
                    "target_domain_used_for_selection": False,
                    "target_labels_used_for_generation_or_selection": False,
                    "source_answer_text_retrieved": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[row["id"]] = record
    finally:
        adapter.close()

    ordered = [completed[row["id"]] for row in rows]
    payload = {
        "version": RUN_VERSION,
        "method_version": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "status": "final",
        "model": args.model,
        "model_artifacts": artifact,
        "source_bank_fingerprint": bank["fingerprint"],
        "lambda_independent_candidate_cache": True,
        "target_domain_used_for_selection": False,
        "target_labels_used_for_generation_or_selection": False,
        "source_answer_text_retrieved": False,
        "records": ordered,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fingerprint": fingerprint,
                "records": len(ordered),
                "candidate_budget": args.candidate_budget,
                "provisional_lambda": args.lambda_value,
                "lambda_independent_candidate_cache": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
