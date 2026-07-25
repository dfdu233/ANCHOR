"""Probe exact versus proxy LLaVA-Med source support in model feature spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.build_visual_centers_v2 import ordered_descriptors
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_local_source import LlavaLocalSourceAdapter
from corrected_sgta.protocol_v2 import file_sha256, resolve_image
from corrected_sgta.provenance_release2 import model_identity
from corrected_sgta.source_bank_v2 import load_index, sha256_file
from corrected_sgta.source_bank_v3 import load_descriptor_image, verify_descriptor


VERSION = "llava-native-support-probe-v1"
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-index", required=True, type=Path)
    parser.add_argument("--proxy-index", required=True, type=Path)
    parser.add_argument("--target-dataset", required=True, type=Path)
    parser.add_argument("--greedy-eval", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-source", type=int, default=64)
    parser.add_argument("--max-target", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(
        np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None
    )


def center(values: np.ndarray) -> np.ndarray:
    return normalize_rows(normalize_rows(values).mean(axis=0, keepdims=True))[0]


def cosine_distances(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return 1.0 - normalize_rows(values) @ normalize_rows(reference[None])[0]


def error_auroc(scores: np.ndarray, errors: np.ndarray) -> float | None:
    positives = scores[errors == 1]
    negatives = scores[errors == 0]
    if not len(positives) or not len(negatives):
        return None
    greater = (positives[:, None] > negatives[None, :]).sum()
    equal = (positives[:, None] == negatives[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(positives) * len(negatives)))


def source_descriptors(path: Path, maximum: int, seed: int) -> list[dict]:
    rows = ordered_descriptors(load_index(path), seed)[:maximum]
    for row in rows:
        verify_descriptor(row)
    return rows


def target_descriptors(path: Path, maximum: int) -> list[dict]:
    rows = []
    for item in json.loads(path.read_text())[:maximum]:
        image = resolve_image(item.get("img_name", ""))
        if image is None:
            continue
        rows.append({"qid": item["qid"], "path": str(image.resolve())})
    return rows


@torch.inference_mode()
def extract_features(
    adapter: LlavaLocalSourceAdapter,
    descriptors: Sequence[dict],
    batch_size: int,
    max_image_side: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from llava.mm_utils import process_images

    raw_pooled = []
    projected_pooled = []
    kept = []
    for start in tqdm(
        range(0, len(descriptors), batch_size), desc="native-support features"
    ):
        batch = descriptors[start : start + batch_size]
        images = []
        batch_kept = []
        for row in batch:
            try:
                image = (
                    load_descriptor_image(row)
                    if row.get("kind")
                    else Image.open(row["path"]).convert("RGB")
                )
                images.append(resize_image(image, max_image_side))
                image.close()
                batch_kept.append(str(row.get("qid", row.get("pair_id", row["path"]))))
            except OSError:
                continue
        if not images:
            continue
        tensors = process_images(images, adapter.image_processor, adapter.model.config)
        if isinstance(tensors, list):
            tensors = torch.stack(tensors)
        tensors = tensors.to(adapter.model.device, dtype=adapter.model.dtype)
        raw = adapter.model.get_vision_tower()(tensors)
        projected = adapter.model.get_model().mm_projector(raw)
        raw_pooled.append(raw.float().mean(dim=1).cpu().numpy())
        projected_pooled.append(projected.float().mean(dim=1).cpu().numpy())
        kept.extend(batch_kept)
        del tensors, raw, projected
        for image in images:
            image.close()
    if not kept:
        raise RuntimeError("no readable images")
    return (
        np.concatenate(raw_pooled).astype(np.float32),
        np.concatenate(projected_pooled).astype(np.float32),
        kept,
    )


def support_metrics(
    exact: np.ndarray,
    proxy: np.ndarray,
    target: np.ndarray,
    errors: np.ndarray,
) -> dict:
    exact_center = center(exact)
    proxy_center = center(proxy)
    exact_distance = cosine_distances(target, exact_center)
    proxy_distance = cosine_distances(target, proxy_center)
    exact_unit = normalize_rows(exact)
    proxy_unit = normalize_rows(proxy)
    target_unit = normalize_rows(target)
    exact_knn = 1.0 - np.sort(target_unit @ exact_unit.T, axis=1)[:, -5:]
    proxy_knn = 1.0 - np.sort(target_unit @ proxy_unit.T, axis=1)[:, -5:]
    exact_1nn = exact_knn.min(axis=1)
    proxy_1nn = proxy_knn.min(axis=1)
    exact_half_a = center(exact[::2])
    exact_half_b = center(exact[1::2])
    proxy_half_a = center(proxy[::2])
    proxy_half_b = center(proxy[1::2])
    return {
        "center_cosine_distance_exact_proxy": float(
            cosine_distances(exact_center[None], proxy_center)[0]
        ),
        "split_half_center_distance": {
            "exact": float(cosine_distances(exact_half_a[None], exact_half_b)[0]),
            "proxy": float(cosine_distances(proxy_half_a[None], proxy_half_b)[0]),
        },
        "within_source_distance_to_center": {
            "exact_mean": float(cosine_distances(exact, exact_center).mean()),
            "proxy_mean": float(cosine_distances(proxy, proxy_center).mean()),
        },
        "target_to_global_center": {
            "exact_mean": float(exact_distance.mean()),
            "proxy_mean": float(proxy_distance.mean()),
            "exact_minus_proxy_mean": float(
                (exact_distance - proxy_distance).mean()
            ),
            "exact_closer_fraction": float(
                (exact_distance < proxy_distance).mean()
            ),
        },
        "target_to_local_support": {
            "exact_1nn_mean": float(exact_1nn.mean()),
            "proxy_1nn_mean": float(proxy_1nn.mean()),
            "exact_5nn_mean": float(exact_knn.mean(axis=1).mean()),
            "proxy_5nn_mean": float(proxy_knn.mean(axis=1).mean()),
            "exact_1nn_closer_fraction": float(
                (exact_1nn < proxy_1nn).mean()
            ),
        },
        "decoded_error_risk": {
            "n_errors": int(errors.sum()),
            "exact_center_auroc": error_auroc(exact_distance, errors),
            "proxy_center_auroc": error_auroc(proxy_distance, errors),
            "exact_1nn_auroc": error_auroc(exact_1nn, errors),
            "proxy_1nn_auroc": error_auroc(proxy_1nn, errors),
            "exact_5nn_auroc": error_auroc(exact_knn.mean(axis=1), errors),
            "proxy_5nn_auroc": error_auroc(proxy_knn.mean(axis=1), errors),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact_rows = source_descriptors(args.exact_index, args.max_source, args.seed)
    proxy_rows = source_descriptors(args.proxy_index, args.max_source, args.seed)
    target_rows = target_descriptors(args.target_dataset, args.max_target)
    adapter = LlavaLocalSourceAdapter()
    try:
        exact_raw, exact_projected, exact_ids = extract_features(
            adapter, exact_rows, args.batch_size, args.max_image_side
        )
        proxy_raw, proxy_projected, proxy_ids = extract_features(
            adapter, proxy_rows, args.batch_size, args.max_image_side
        )
        target_raw, target_projected, target_ids = extract_features(
            adapter, target_rows, args.batch_size, args.max_image_side
        )
    finally:
        adapter.close()

    evaluation = json.loads(args.greedy_eval.read_text())
    correctness = {
        str(row["question_id"]): bool(row["correct"])
        for row in evaluation["details"]
    }
    errors = np.asarray(
        [not correctness.get(qid, False) for qid in target_ids], dtype=np.int64
    )
    arrays_path = args.output_dir / "features.npz"
    np.savez_compressed(
        arrays_path,
        exact_raw=exact_raw,
        exact_projected=exact_projected,
        proxy_raw=proxy_raw,
        proxy_projected=proxy_projected,
        target_raw=target_raw,
        target_projected=target_projected,
    )
    config = {
        "version": VERSION,
        "exact_index": str(args.exact_index.resolve()),
        "exact_index_sha256": file_sha256(args.exact_index),
        "proxy_index": str(args.proxy_index.resolve()),
        "proxy_index_sha256": file_sha256(args.proxy_index),
        "target_dataset": str(args.target_dataset.resolve()),
        "target_dataset_sha256": file_sha256(args.target_dataset),
        "greedy_eval": str(args.greedy_eval.resolve()),
        "greedy_eval_sha256": file_sha256(args.greedy_eval),
        "max_source": args.max_source,
        "max_target": args.max_target,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "source_order": "sha256(seed:full descriptor JSON)",
        "labels_used_for": "post-hoc error-risk analysis only",
        "model_identity": model_identity("llava"),
    }
    payload = {
        "fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "config": config,
        "features": str(arrays_path.resolve()),
        "features_sha256": sha256_file(arrays_path),
        "n": {
            "exact": len(exact_ids),
            "proxy": len(proxy_ids),
            "target": len(target_ids),
        },
        "ids": {
            "exact": exact_ids,
            "proxy": proxy_ids,
            "target": target_ids,
        },
        "raw_clip": support_metrics(exact_raw, proxy_raw, target_raw, errors),
        "projected": support_metrics(
            exact_projected, proxy_projected, target_projected, errors
        ),
    }
    output = args.output_dir / "analysis.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
