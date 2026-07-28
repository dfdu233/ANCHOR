#!/usr/bin/env python3
"""Resumable ANCHOR-Riemann gate runner.

This runner deliberately scores complete generated sentences only.  It never
enumerates yes/no labels and never uses target labels during generation or
candidate selection.  Target references are carried only for the downstream
gate analysis.
"""

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
    stable_json_sha256,
)
from corrected_sgta.cache import repair_truncated_jsonl_tail
from corrected_sgta.riemann_geometry import (
    diagonal_fisher_metric,
    dirichlet_energy,
    nearest_manifold_distance,
    riemann_code_identity,
    stable_ranked_permutation,
    zscore,
)


RUN_VERSION = "anchor-riemann-gate-runner-v1"
REPORT_PROMPT = (
    "You are a professional radiologist. You are provided with a chest X-ray "
    "image. Please generate a report based on the image. Please only include "
    "the content of the report in your response."
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


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
        raise RuntimeError("source bank must not store source answer text")
    return payload


def per_item_seed(base_seed: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{identifier}".encode()).hexdigest()
    return int(digest[:8], 16)


def normalize_target_record(
    row: dict[str, Any],
    *,
    default_domain: str,
    default_prompt: str,
    require_answer: bool = True,
) -> dict[str, Any]:
    try:
        return normalize_manifest_record(
            row, require_answer=require_answer, default_domain=default_domain
        )
    except ValueError:
        image = row.get("image", row.get("image_path"))
        if isinstance(image, list):
            image = image[0] if image else None
        answer = row.get("answer", row.get("reference", row.get("report")))
        identifier = row.get("id", row.get("qid", row.get("study_id")))
        domain = row.get("domain", row.get("dataset", default_domain))
        patient = row.get("patient_id", row.get("subject_id", identifier))
        missing = [
            name
            for name, value in (
                ("id", identifier),
                ("image", image),
                ("answer", answer if require_answer else "ok"),
            )
            if value is None or not str(value).strip()
        ]
        if missing:
            raise ValueError(f"target row missing fields after report fallback: {missing}")
        output = {
            "id": str(identifier),
            "image": str(image),
            "prompt": str(row.get("prompt") or row.get("question") or default_prompt),
            "domain": str(domain),
            "patient_id": str(patient),
        }
        if require_answer:
            output["answer"] = str(answer).strip()
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--default-domain", default="unknown")
    parser.add_argument("--default-prompt", default=REPORT_PROMPT)
    parser.add_argument("--task", choices=("ce", "oe"), required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--model", choices=("hulu", "llava"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--candidate-batch", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--projections", type=int, default=32)
    parser.add_argument("--quantiles", type=int, default=DEFAULT_QUANTILES)
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--mu-value", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_budget < 2:
        raise ValueError("candidate budget must include greedy plus samples")
    if (
        args.max_new_tokens <= 0
        or args.candidate_batch <= 0
        or args.neighbors <= 0
        or args.projections <= 0
        or args.quantiles <= 0
        or args.max_samples < 0
        or args.lambda_value < 0
        or args.mu_value < 0
    ):
        raise ValueError("invalid generation or geometry parameters")

    bank = load_bank(args.bank)
    if bank.get("model") != args.model:
        raise RuntimeError("source bank and inference model differ")
    rows = [
        normalize_target_record(
            row,
            default_domain=args.default_domain,
            default_prompt=args.default_prompt,
            require_answer=True,
        )
        for row in load_json_or_jsonl(args.manifest)
    ]
    if args.max_samples:
        rows = rows[: args.max_samples]
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("target manifest contains duplicate ids")

    resolved_images: dict[str, Path] = {}
    image_hash_cache: dict[str, str] = {}
    target_image_sha256: list[dict[str, str]] = []
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

    bank_records = bank["records"]
    source_embeddings = np.asarray(
        [record["embedding"] for record in bank_records], dtype=np.float64
    )
    source_trajectories = [
        np.asarray(record["trajectory"], dtype=np.float64)
        for record in bank_records
    ]
    metric = diagonal_fisher_metric(source_trajectories)
    directions = deterministic_directions(
        dimension=len(FEATURE_NAMES),
        count=args.projections,
        seed=args.projection_seed,
    )
    random_order = stable_ranked_permutation(
        len(source_trajectories), args.seed ^ 0xA17C0DE
    )
    random_source_trajectories = [
        source_trajectories[int(index)] for index in random_order
    ]

    fingerprint_payload = {
        "version": RUN_VERSION,
        "geometry_identity": riemann_code_identity(),
        "method_version": VERSION,
        "task": args.task,
        "manifest_sha256": file_sha256(args.manifest),
        "target_image_sha256": target_image_sha256,
        "bank_sha256": file_sha256(args.bank),
        "bank_fingerprint": bank["fingerprint"],
        "model": args.model,
        "model_artifact_fingerprint": artifact["fingerprint"],
        "candidate_budget": args.candidate_budget,
        "candidate_policy": "one_greedy_plus_seeded_nucleus_samples",
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
        "lambda_value": args.lambda_value,
        "mu_value": args.mu_value,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "target_domain_used_for_selection": False,
        "target_labels_used_for_generation_or_selection": False,
        "uses_canonical_label_logits_for_prediction": False,
        "source_answer_text_retrieved": False,
        "code_sha256": {
            "runner": file_sha256(Path(__file__)),
            "core": file_sha256(Path(__file__).with_name("anchor_transport.py")),
            "geometry": file_sha256(Path(__file__).with_name("riemann_geometry.py")),
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
                raise RuntimeError("Riemann raw-cache fingerprint mismatch")
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
                desc=f"ANCHOR-Riemann gate ({args.task}/{args.model})",
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
                nll_values: list[float] = []
                sw_values: list[float] = []
                dirichlet_values: list[float] = []
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
                    nearest_distance, nearest_index, individual_distances = (
                        nearest_manifold_distance(
                            normalized,
                            retrieved_trajectories,
                            directions=directions,
                            quantiles=args.quantiles,
                        )
                    )
                    random_distance, random_index, _ = nearest_manifold_distance(
                        normalized,
                        random_source_trajectories[: len(retrieved_trajectories)],
                        directions=directions,
                        quantiles=args.quantiles,
                    )
                    path_energy = dirichlet_energy(normalized, metric)
                    candidate.update(evidence.to_json())
                    candidate["normalized_trajectory"] = normalized.astype(float).tolist()
                    candidate["source_manifold_distance"] = nearest_distance
                    candidate["source_neighbor_local_index"] = nearest_index
                    candidate["source_neighbor_distances"] = individual_distances
                    candidate["random_manifold_distance"] = random_distance
                    candidate["random_neighbor_local_index"] = random_index
                    candidate["dirichlet_energy"] = path_energy
                    nll_values.append(-float(evidence.mean_image_log_probability))
                    sw_values.append(nearest_distance)
                    dirichlet_values.append(path_energy)
                znll = zscore(nll_values)
                zsw = zscore(sw_values)
                zdir = zscore(dirichlet_values)
                combined = [
                    znll[index]
                    + args.lambda_value * zsw[index]
                    + args.mu_value * zdir[index]
                    for index in range(len(candidates))
                ]
                nll_selected = int(np.argmin(np.asarray(nll_values)))
                riemann_selected = int(np.argmin(np.asarray(combined)))
                random_combined = [
                    znll[index]
                    + args.lambda_value
                    * zscore(
                        [
                            float(candidate["random_manifold_distance"])
                            for candidate in candidates
                        ]
                    )[index]
                    + args.mu_value * zdir[index]
                    for index in range(len(candidates))
                ]
                random_selected = int(np.argmin(np.asarray(random_combined)))
                for index, candidate in enumerate(candidates):
                    candidate["sequence_nll"] = nll_values[index]
                    candidate["z_sequence_nll"] = znll[index]
                    candidate["z_source_manifold_distance"] = zsw[index]
                    candidate["z_dirichlet_energy"] = zdir[index]
                    candidate["riemann_energy"] = combined[index]
                    candidate["random_manifold_energy"] = random_combined[index]
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
                    "task": args.task,
                    "id": row["id"],
                    "image": row["image"],
                    "image_sha256": image_hash_cache[str(image_path)],
                    "prompt": row["prompt"],
                    "prompt_sha256": stable_json_sha256(row["prompt"]),
                    "reference": row["answer"],
                    "reference_sha256": stable_json_sha256(row["answer"]),
                    "evaluation_group": row["domain"],
                    "patient_id": row["patient_id"],
                    "item_seed": item_seed,
                    "neighbors": neighbors,
                    "candidates": candidates,
                    "unique_candidate_count": len({c["text"] for c in candidates}),
                    "greedy_index": 0,
                    "nll_selected_index": nll_selected,
                    "nll_selected_text": candidates[nll_selected]["text"],
                    "riemann_selected_index": riemann_selected,
                    "riemann_selected_text": candidates[riemann_selected]["text"],
                    "random_selected_index": random_selected,
                    "random_selected_text": candidates[random_selected]["text"],
                    "target_domain_used_for_selection": False,
                    "target_labels_used_for_generation_or_selection": False,
                    "uses_canonical_label_logits_for_prediction": False,
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
        "geometry_identity": riemann_code_identity(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "status": "final",
        "model": args.model,
        "task": args.task,
        "model_artifacts": artifact,
        "source_bank_fingerprint": bank["fingerprint"],
        "target_domain_used_for_selection": False,
        "target_labels_used_for_generation_or_selection": False,
        "uses_canonical_label_logits_for_prediction": False,
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
                "lambda_value": args.lambda_value,
                "mu_value": args.mu_value,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
