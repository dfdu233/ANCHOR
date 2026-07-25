"""Run the minimal QLS-TR identifiability probe on binary CXR questions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_surface import load_adapter
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    build_prompt,
    file_sha256,
    ground_truth_index,
    labels_for_sample,
    protocol_fingerprint,
    resolve_image,
    task_kind,
    validate_dataset,
)
from corrected_sgta.qls_tr import (
    PCAIndex,
    categorical_js,
    fisher_quadratic,
    kde_neighbors,
    reconstruct_mean_shift_view,
    spectral_descriptor,
)
from corrected_sgta.source_bank_v2 import load_descriptor_image, load_index, sha256_file


VERSION = "sgta-qls-tr-probe-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qls-index", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matched-source", default="pubmedvision_xray_formal")
    parser.add_argument("--wrong-source", default="radimagenet_ct_leaksafe")
    return parser.parse_args()


def softmax_negative_nll(values: np.ndarray) -> np.ndarray:
    logits = -np.asarray(values, dtype=np.float64)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def load_source(index_path: Path, source_id: str) -> tuple[PCAIndex, list[dict], dict]:
    metadata = json.loads(index_path.with_suffix(index_path.suffix + ".meta.json").read_text())
    entry = next(item for item in metadata["entries"] if item["source_id"] == source_id)
    arrays = np.load(index_path, allow_pickle=False)
    index = PCAIndex(
        mean=arrays[f"{source_id}__mean"],
        components=arrays[f"{source_id}__components"],
        coordinates=arrays[f"{source_id}__coordinates"],
        bandwidth=float(entry["bandwidth"]),
        median_nn_distance=float(entry["median_nn_distance"]),
    )
    descriptors = load_index(Path(entry["retained_index"]))
    return index, descriptors, entry


def build_views(image: Image.Image, index: PCAIndex, descriptors: list[dict]) -> tuple[list[Image.Image], dict]:
    feature = spectral_descriptor(image)
    ids, weights, mean_shift = kde_neighbors(feature, index, k=8)
    radius = 0.25 * index.median_nn_distance
    euclidean_blend = min(1.0, radius / max(float(np.linalg.norm(mean_shift)), 1e-12))
    neighbor_images = [load_descriptor_image(descriptors[int(i)]) for i in ids]
    blends = [euclidean_blend, euclidean_blend / 2.0, euclidean_blend / 4.0]
    views = [
        reconstruct_mean_shift_view(image, neighbor_images, weights, blend=value)
        for value in blends
    ]
    query_coordinate = (feature - index.mean) @ index.components.T
    original_nn_distance = float(
        np.min(np.linalg.norm(index.coordinates - query_coordinate, axis=1))
    )
    candidate_nn_distances = []
    for view in views:
        coordinate = (spectral_descriptor(view) - index.mean) @ index.components.T
        candidate_nn_distances.append(
            float(np.min(np.linalg.norm(index.coordinates - coordinate, axis=1)))
        )
    return views, {
        "neighbor_ids": ids.tolist(),
        "neighbor_weights": weights.tolist(),
        "mean_shift_norm": float(np.linalg.norm(mean_shift)),
        "euclidean_radius": float(radius),
        "candidate_blends": blends,
        "original_nn_distance": original_nn_distance,
        "candidate_nn_distances": candidate_nn_distances,
        "candidate_density_gain": [
            original_nn_distance - value for value in candidate_nn_distances
        ],
    }


def choose_trust_region(original, reference, candidates) -> tuple[int | None, dict]:
    probability = softmax_negative_nll(original.sequence_nll)
    reference_delta = -reference.sequence_nll - (-original.sequence_nll)
    radius = fisher_quadratic(probability, reference_delta)
    energies = []
    divergences = []
    for candidate in candidates:
        delta = -candidate.sequence_nll - (-original.sequence_nll)
        energies.append(fisher_quadratic(probability, delta))
        divergences.append(
            categorical_js(
                probability, softmax_negative_nll(candidate.sequence_nll)
            )
        )
    eligible = [i for i, energy in enumerate(energies) if energy <= radius + 1e-12]
    selected = eligible[0] if eligible else None
    return selected, {
        "fisher_radius": radius,
        "candidate_fisher_energy": energies,
        "candidate_js": divergences,
        "fallback": selected is None,
    }


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    matched_index, matched_descriptors, matched_entry = load_source(
        args.qls_index, args.matched_source
    )
    wrong_index, wrong_descriptors, wrong_entry = load_source(
        args.qls_index, args.wrong_source
    )
    config = {
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "qls_index": str(args.qls_index.resolve()),
        "qls_index_sha256": sha256_file(args.qls_index),
        "qls_index_meta_sha256": sha256_file(
            args.qls_index.with_suffix(args.qls_index.suffix + ".meta.json")
        ),
        "max_samples": args.max_samples,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid)",
        "matched_source": args.matched_source,
        "wrong_source": args.wrong_source,
        "kde_k": 8,
        "euclidean_radius": "0.25 * held-out source median 1-NN distance",
        "fisher_radius": "per-question deterministic 1% contrast perturbation",
        "candidate_blends": "r_E clipped mean shift times [1, 1/2, 1/4]",
        "selection": "largest blend within categorical Fisher radius; else original",
        "label_interface": "surface-robust complete Yes/No label NLL",
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
        "matched_index_entry": matched_entry,
        "wrong_index_entry": wrong_entry,
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("metadata mismatch; choose a new output path")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))
    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    selected_rows = []
    for row in rows:
        try:
            if task_kind(row) != "binary":
                continue
            labels = labels_for_sample(row)
            if tuple(labels) != ("Yes", "No"):
                continue
            ground_truth_index(row)
            if resolve_image(row.get("img_name", "")) is not None:
                selected_rows.append(row)
        except ProtocolError:
            continue
    selected_rows.sort(
        key=lambda row: hashlib.sha256(f"{args.seed}:{row['qid']}".encode()).hexdigest()
    )
    selected_rows = selected_rows[: args.max_samples]
    eligible = [row for row in selected_rows if str(row["qid"]) not in saved]
    print(f"QLS-TR fingerprint={fingerprint[:12]} eligible={len(eligible)}", flush=True)
    if not eligible:
        return
    adapter = load_adapter(args.model)
    try:
        with args.output.open("a") as output:
            for sample in tqdm(eligible, desc=f"QLS-TR {args.model}"):
                try:
                    path = resolve_image(sample.get("img_name", ""))
                    with Image.open(path) as raw:
                        image = resize_image(raw.convert("RGB"), args.max_image_side)
                    matched_views, matched_geometry = build_views(
                        image, matched_index, matched_descriptors
                    )
                    wrong_views, wrong_geometry = build_views(
                        image, wrong_index, wrong_descriptors
                    )
                    reference = ImageEnhance.Contrast(image).enhance(1.01)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    evidence = adapter.forward_ce(
                        [image, reference] + matched_views + wrong_views, prompt, labels
                    )
                    original, reference_evidence = evidence[:2]
                    matched_evidence = evidence[2:5]
                    wrong_evidence = evidence[5:8]
                    matched_selected, matched_tr = choose_trust_region(
                        original, reference_evidence, matched_evidence
                    )
                    wrong_selected, wrong_tr = choose_trust_region(
                        original, reference_evidence, wrong_evidence
                    )
                    chosen_matched = (
                        original
                        if matched_selected is None
                        else matched_evidence[matched_selected]
                    )
                    chosen_wrong = (
                        original if wrong_selected is None else wrong_evidence[wrong_selected]
                    )
                    unconditioned = matched_evidence[0]
                    gt = ground_truth_index(sample)
                    row = {
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name"),
                        "labels": list(labels),
                        "gt_index": gt,
                        "original_nll": original.sequence_nll.tolist(),
                        "matched_nll": chosen_matched.sequence_nll.tolist(),
                        "wrong_nll": chosen_wrong.sequence_nll.tolist(),
                        "null_unconditioned_nll": unconditioned.sequence_nll.tolist(),
                        "matched_candidate_nll": [
                            value.sequence_nll.tolist() for value in matched_evidence
                        ],
                        "wrong_candidate_nll": [
                            value.sequence_nll.tolist() for value in wrong_evidence
                        ],
                        "original_prediction": int(np.argmin(original.sequence_nll)),
                        "matched_prediction": int(np.argmin(chosen_matched.sequence_nll)),
                        "wrong_prediction": int(np.argmin(chosen_wrong.sequence_nll)),
                        "null_unconditioned_prediction": int(
                            np.argmin(unconditioned.sequence_nll)
                        ),
                        "matched_selected_candidate": matched_selected,
                        "wrong_selected_candidate": wrong_selected,
                        "matched_geometry": matched_geometry,
                        "wrong_geometry": wrong_geometry,
                        "matched_trust_region": matched_tr,
                        "wrong_trust_region": wrong_tr,
                    }
                except Exception as exc:
                    traceback.print_exc()
                    row = {
                        "fingerprint": fingerprint,
                        "status": "error",
                        "qid": sample.get("qid"),
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        gc.collect()
                        torch.cuda.empty_cache()
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
