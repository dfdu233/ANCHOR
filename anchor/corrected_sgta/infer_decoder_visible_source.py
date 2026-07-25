"""Probe a question-conditioned, decoder-visible source direction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
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
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "decoder-visible-source-projection-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matched-features", required=True, type=Path)
    parser.add_argument("--matched-key", default="exact_projected")
    parser.add_argument("--wrong-features", required=True, type=Path)
    parser.add_argument("--wrong-key", default="ct_source_projected")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=8)
    return parser.parse_args()


def source_geometry(values: np.ndarray, neighbors: int) -> dict:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or len(source) <= neighbors:
        raise ValueError("source features must be [N,D] with N > neighbors")
    distances = np.linalg.norm(source[:, None] - source[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    ordered = np.sort(distances, axis=1)
    return {
        "features": source,
        "bandwidth": float(np.median(ordered[:, neighbors - 1])),
        "median_nn": float(np.median(ordered[:, 0])),
    }


def local_score(point: np.ndarray, geometry: dict, neighbors: int) -> tuple[np.ndarray, dict]:
    source = geometry["features"]
    squared = np.sum((source - point[None, :]) ** 2, axis=1)
    ids = np.argsort(squared)[:neighbors]
    bandwidth = max(float(geometry["bandwidth"]), 1e-8)
    logits = -squared[ids] / (2.0 * bandwidth**2)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    local_mean = np.sum(weights[:, None] * source[ids], axis=0)
    return (local_mean - point).astype(np.float32), {
        "neighbor_ids": ids.tolist(),
        "neighbor_weights": weights.tolist(),
        "nearest_distance_before": float(np.sqrt(squared[ids[0]])),
    }


def projected_direction(score: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, dict]:
    denominator = float(np.dot(gradient, gradient))
    coefficient = float(np.dot(score, gradient)) / max(denominator, 1e-20)
    projection = coefficient * gradient
    norm = float(np.linalg.norm(projection))
    return projection.astype(np.float32), {
        "score_norm": float(np.linalg.norm(score)),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "score_gradient_inner_product": float(np.dot(score, gradient)),
        "projection_coefficient": coefficient,
        "projection_norm_before_rescale": norm,
    }


def equal_clipped_steps(
    matched_projection: np.ndarray,
    wrong_projection: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    matched_norm = float(np.linalg.norm(matched_projection))
    wrong_norm = float(np.linalg.norm(wrong_projection))
    shared = min(float(radius), matched_norm, wrong_norm)
    if shared <= 1e-12:
        return np.zeros_like(matched_projection), np.zeros_like(wrong_projection), 0.0
    return (
        (shared * matched_projection / matched_norm).astype(np.float32),
        (shared * wrong_projection / wrong_norm).astype(np.float32),
        shared,
    )


def kde_log_density(point: np.ndarray, geometry: dict) -> float:
    source = geometry["features"]
    bandwidth = max(float(geometry["bandwidth"]), 1e-8)
    squared = np.sum((source - point[None, :]) ** 2, axis=1)
    logits = -squared / (2.0 * bandwidth**2)
    maximum = float(np.max(logits))
    return maximum + float(np.log(np.mean(np.exp(logits - maximum))))


@contextmanager
def capture_projected_tokens(adapter: LlavaMedAlignmentAdapter):
    original_encode = adapter.model.encode_images
    captured: list[torch.Tensor] = []

    def capture(*args, **kwargs):
        value = original_encode(*args, **kwargs).detach().requires_grad_(True)
        captured.append(value)
        return value

    adapter.model.encode_images = capture
    try:
        yield captured
    finally:
        adapter.model.encode_images = original_encode


@contextmanager
def shift_projected_tokens(adapter: LlavaMedAlignmentAdapter, step: np.ndarray):
    original_encode = adapter.model.encode_images

    def shifted(*args, **kwargs):
        value = original_encode(*args, **kwargs)
        delta = torch.as_tensor(step, device=value.device, dtype=value.dtype)
        return value + delta.view(1, 1, -1)

    adapter.model.encode_images = shifted
    try:
        yield
    finally:
        adapter.model.encode_images = original_encode


def differentiable_margin(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    labels: tuple[str, ...],
) -> dict:
    from llava.mm_utils import process_images

    input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    image_tensor = process_images([image], adapter.image_processor, adapter.model.config)
    if isinstance(image_tensor, list):
        image_tensor = [
            item.to(adapter.model.device, dtype=adapter.model.dtype)
            for item in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(adapter.model.device, dtype=adapter.model.dtype)
    with capture_projected_tokens(adapter) as captured:
        _, position_ids, attention_mask, _, inputs_embeds, _ = (
            adapter.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_tensor,
                image_sizes=[image.size],
            )
        )
        output = adapter.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[:, -1]
        weight = adapter.model.get_output_embeddings().weight
        class_logits = []
        for group in adapter.label_id_groups(labels):
            values = hidden.to(weight.dtype) @ weight[group].T
            class_logits.append(values.max(-1).values)
        logits = torch.stack(class_logits, dim=-1)[0]
        prediction = int(logits.argmax())
        margin = logits[prediction] - logits[1 - prediction]
        token_features = captured[0]
        gradient = torch.autograd.grad(margin, token_features)[0]
        vocabulary_logits = hidden.to(weight.dtype) @ weight.T
        log_probability = torch.log_softmax(vocabulary_logits.float(), dim=-1)
        sequence_nll = torch.stack(
            [-log_probability[:, group].max(-1).values[0] for group in adapter.label_id_groups(labels)]
        )
        result = {
            "logits": logits.detach().float().cpu().numpy(),
            "sequence_nll": sequence_nll.detach().cpu().numpy(),
            "prediction": prediction,
            "predicted_margin": float(margin.detach()),
            "pooled_visual": token_features.detach()[0].mean(0).float().cpu().numpy(),
            "pooled_margin_gradient": gradient.detach()[0].sum(0).float().cpu().numpy(),
            "visual_token_count": int(token_features.shape[1]),
        }
    del output, hidden, inputs_embeds, image_tensor, input_ids
    return result


def nearest_distance(point: np.ndarray, source: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(source - point[None, :], axis=1)))


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    with np.load(args.matched_features) as arrays:
        matched_source = np.asarray(arrays[args.matched_key], dtype=np.float32)
    with np.load(args.wrong_features) as arrays:
        wrong_source = np.asarray(arrays[args.wrong_key], dtype=np.float32)
    matched_geometry = source_geometry(matched_source, args.neighbors)
    wrong_geometry = source_geometry(wrong_source, args.neighbors)
    radius = 0.25 * matched_geometry["median_nn"]
    config = {
        "version": VERSION,
        "code_identity": {
            "infer_decoder_visible_source.py": sha256_file(Path(__file__)),
            "models_alignment.py": sha256_file(
                Path(__file__).with_name("models_alignment.py")
            ),
            "models_surface.py": sha256_file(
                Path(__file__).with_name("models_surface.py")
            ),
        },
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": "llava",
        "model_identity": model_identity("llava"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "matched_features": str(args.matched_features.resolve()),
        "matched_features_sha256": sha256_file(args.matched_features),
        "matched_key": args.matched_key,
        "wrong_features": str(args.wrong_features.resolve()),
        "wrong_features_sha256": sha256_file(args.wrong_features),
        "wrong_key": args.wrong_key,
        "max_samples": args.max_samples,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "neighbors": args.neighbors,
        "fixed_radius": radius,
        "operator": "KDE local source score projected onto predicted-margin gradient",
        "control": "matched and wrong directions use identical Euclidean step radius",
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
        "matched_source": {
            "n": len(matched_source),
            "bandwidth": matched_geometry["bandwidth"],
            "median_nn": matched_geometry["median_nn"],
        },
        "wrong_source": {
            "n": len(wrong_source),
            "bandwidth": wrong_geometry["bandwidth"],
            "median_nn": wrong_geometry["median_nn"],
        },
    }
    if meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("metadata mismatch; choose a new output path")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2))
    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    selected = []
    for row in rows:
        try:
            if task_kind(row) != "binary" or tuple(labels_for_sample(row)) != ("Yes", "No"):
                continue
            ground_truth_index(row)
            if resolve_image(row.get("img_name", "")) is not None:
                selected.append(row)
        except ProtocolError:
            continue
    selected.sort(
        key=lambda row: hashlib.sha256(f"{args.seed}:{row['qid']}".encode()).hexdigest()
    )
    selected = selected[: args.max_samples]
    eligible = [row for row in selected if str(row["qid"]) not in saved]
    print(f"DVS fingerprint={fingerprint[:12]} eligible={len(eligible)} radius={radius:.6g}")
    if not eligible:
        return
    adapter = LlavaMedAlignmentAdapter()
    adapter.model.requires_grad_(False)
    try:
        with args.output.open("a") as output:
            for sample in tqdm(eligible, desc="decoder-visible source"):
                try:
                    path = resolve_image(sample.get("img_name", ""))
                    with Image.open(path) as raw:
                        image = resize_image(raw.convert("RGB"), args.max_image_side)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    original = differentiable_margin(adapter, image, prompt, labels)
                    point = original["pooled_visual"]
                    gradient = original["pooled_margin_gradient"]
                    matched_score, matched_info = local_score(
                        point, matched_geometry, args.neighbors
                    )
                    wrong_score, wrong_info = local_score(
                        point, wrong_geometry, args.neighbors
                    )
                    matched_direction, matched_projection = projected_direction(
                        matched_score, gradient
                    )
                    wrong_direction, wrong_projection = projected_direction(
                        wrong_score, gradient
                    )
                    matched_step, wrong_step, shared_step_norm = equal_clipped_steps(
                        matched_direction, wrong_direction, radius
                    )
                    with shift_projected_tokens(adapter, matched_step):
                        matched = adapter.forward_ce([image], prompt, labels)[0]
                    with shift_projected_tokens(adapter, wrong_step):
                        wrong = adapter.forward_ce([image], prompt, labels)[0]
                    matched_after = point + matched_step
                    wrong_after = point + wrong_step
                    row = {
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name"),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "original_logits": original["logits"].tolist(),
                        "original_nll": original["sequence_nll"].tolist(),
                        "matched_nll": matched.sequence_nll.tolist(),
                        "wrong_nll": wrong.sequence_nll.tolist(),
                        "original_prediction": original["prediction"],
                        "matched_prediction": int(np.argmin(matched.sequence_nll)),
                        "wrong_prediction": int(np.argmin(wrong.sequence_nll)),
                        "predicted_margin": original["predicted_margin"],
                        "visual_token_count": original["visual_token_count"],
                        "matched": {
                            **matched_info,
                            **matched_projection,
                            "shared_step_norm": shared_step_norm,
                            "step_norm": float(np.linalg.norm(matched_step)),
                            "source_ascent_inner_product": float(
                                np.dot(matched_score, matched_step)
                            ),
                            "kde_log_density_before": kde_log_density(
                                point, matched_geometry
                            ),
                            "kde_log_density_after": kde_log_density(
                                matched_after, matched_geometry
                            ),
                            "nearest_distance_after": nearest_distance(
                                matched_after, matched_source
                            ),
                        },
                        "wrong": {
                            **wrong_info,
                            **wrong_projection,
                            "shared_step_norm": shared_step_norm,
                            "step_norm": float(np.linalg.norm(wrong_step)),
                            "source_ascent_inner_product": float(
                                np.dot(wrong_score, wrong_step)
                            ),
                            "kde_log_density_before": kde_log_density(
                                point, wrong_geometry
                            ),
                            "kde_log_density_after": kde_log_density(
                                wrong_after, wrong_geometry
                            ),
                            "nearest_distance_after": nearest_distance(
                                wrong_after, wrong_source
                            ),
                        },
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
