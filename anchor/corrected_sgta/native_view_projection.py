"""DDA-like native-aligned image views for frozen medical VLMs.

This is a deliberately small experiment driver:

1. estimate a native support from calibration images that the VLM answers
   correctly (competence-native support);
2. generate input-side native-aligned views with lightweight FDA / histogram
   matching;
3. accept views only when they move the VLM visual representation closer to the
   native support while preserving structure;
4. evaluate original, best accepted view, and simple fused logits.

It is a practical approximation of "project test inputs back to the source"
without training a diffusion model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFile, ImageOps
from tqdm import tqdm

from corrected_sgta.infer_ce import resize_image
from corrected_sgta.methods import (
    feddg_frequency_interpolation,
    lame_rbf_affinity,
    laplacian_optimization,
    softmax_np,
)
from corrected_sgta.models_local_source import load_local_source_adapter
from corrected_sgta.protocol_v2 import (
    ProtocolError,
    build_prompt,
    file_sha256,
    ground_truth_index,
    labels_for_sample,
    resolve_image,
    task_kind,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "native-view-projection-v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--greedy-eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="llava", choices=("llava", "hulu"))
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-type", default="binary")
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--l-values", type=float, nargs="*", default=(0.003, 0.01, 0.03, 0.06))
    parser.add_argument("--source-ratios", type=float, nargs="*", default=(0.5, 0.8))
    parser.add_argument("--top-native", type=int, default=2)
    parser.add_argument("--min-psnr", type=float, default=18.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.85)
    parser.add_argument("--min-closure", type=float, default=0.0)
    parser.add_argument("--max-views-forward", type=int, default=4)
    parser.add_argument("--transport-betas", type=float, nargs="*", default=(0.05, 0.1, 0.2, 0.4))
    parser.add_argument("--prototype-tokens-per-image", type=int, default=8)
    parser.add_argument("--fusion-temperature", type=float, default=0.2)
    parser.add_argument("--laplacian-lambda", type=float, default=0.5)
    parser.add_argument("--transport-confidence-power", type=float, default=2.0)
    return parser.parse_args()


def load_eval(path: Path):
    payload = json.loads(path.read_text())
    return {str(d["question_id"]): d for d in payload["details"]}, payload


def unit(x: np.ndarray, axis=-1, eps=1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = unit(np.asarray(a, dtype=np.float32).reshape(1, -1))[0]
    b = unit(np.asarray(b, dtype=np.float32).reshape(1, -1))[0]
    return float(1.0 - np.dot(a, b))


def acc(values):
    return float(np.mean(values)) if values else None


def margin(logits: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(logits, dtype=np.float64), axis=-1)
    if values.shape[-1] < 2:
        return values[..., -1]
    return values[..., -1] - values[..., -2]


def sample_token_prototypes(tokens: np.ndarray, count: int) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    if count <= 0 or tokens.shape[0] <= count:
        return tokens
    indices = np.linspace(0, tokens.shape[0] - 1, num=count, dtype=np.int64)
    return tokens[indices]


def weighted_nll_prediction(nll: np.ndarray, weights: np.ndarray) -> int:
    scores = np.sum(np.asarray(weights)[:, None] * np.asarray(nll), axis=0)
    return int(np.argmin(scores))


def laplacian_fused_probability(probabilities: np.ndarray, features: np.ndarray, bound_lambda: float) -> np.ndarray:
    if len(probabilities) <= 1:
        return np.asarray(probabilities)[0]
    with torch.inference_mode():
        probs = torch.as_tensor(probabilities, dtype=torch.float32)
        feats = torch.as_tensor(features, dtype=torch.float32)
        kernel = lame_rbf_affinity(feats, knn=min(3, len(probabilities)), force_symmetry=True)
        refined = laplacian_optimization(
            probs,
            kernel,
            bound_lambda=float(bound_lambda),
            max_steps=50,
            tolerance=1e-7,
        )
    return refined.mean(dim=0).cpu().numpy()


def eligible_rows(rows, question_type: str | None, seed: int, max_samples: int):
    kept = []
    for row in rows:
        try:
            if task_kind(row) == "open":
                continue
            if question_type and row.get("question_type") != question_type:
                continue
            labels_for_sample(row)
            ground_truth_index(row)
            if resolve_image(row.get("img_name", "")) is not None:
                kept.append(row)
        except ProtocolError:
            continue
    kept.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['qid']}".encode()).hexdigest())
    return kept[:max_samples] if max_samples else kept


def fft_amplitude(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    return np.abs(np.fft.fft2(rgb, axes=(-2, -1))).astype(np.float32)


def structure_metrics(source: Image.Image, transformed: Image.Image) -> dict[str, float | None]:
    left = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    right = np.asarray(transformed.convert("L").resize(source.size), dtype=np.float64) / 255.0
    mse = float(np.mean((left - right) ** 2))
    source_edge = np.hypot(*np.gradient(left))
    target_edge = np.hypot(*np.gradient(right))
    sc = source_edge.ravel() - source_edge.mean()
    tc = target_edge.ravel() - target_edge.mean()
    denom = float(np.linalg.norm(sc) * np.linalg.norm(tc))
    edge = float(np.clip(sc @ tc / denom, -1.0, 1.0)) if denom > 1e-12 else 1.0
    return {
        "pixel_mse": mse,
        "psnr": None if mse <= 1e-12 else float(-10.0 * math.log10(mse)),
        "edge_correlation": edge,
    }




def blend_images(source: Image.Image, target: Image.Image, strength: float) -> Image.Image:
    src = np.asarray(source.convert("RGB"), dtype=np.float32)
    tgt = np.asarray(target.convert("RGB").resize(source.size), dtype=np.float32)
    out = (1.0 - float(strength)) * src + float(strength) * tgt
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def gamma_view(image: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    out = np.power(np.clip(arr, 0.0, 1.0), float(gamma)) * 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def photometric_views(image: Image.Image):
    views = []
    for factor in (0.75, 0.9, 1.1, 1.25, 1.5):
        transformed = ImageEnhance.Contrast(image).enhance(factor)
        views.append((f"contrast_{factor:g}", transformed, {"factor": float(factor)}))
    for factor in (0.5, 0.8, 1.25, 1.75, 2.25):
        transformed = ImageEnhance.Sharpness(image).enhance(factor)
        views.append((f"sharpness_{factor:g}", transformed, {"factor": float(factor)}))
    for gamma in (0.75, 0.9, 1.1, 1.25):
        transformed = gamma_view(image, gamma)
        views.append((f"gamma_{gamma:g}", transformed, {"gamma": float(gamma)}))
    auto = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    for strength in (0.25, 0.5, 0.75):
        transformed = blend_images(image, auto, strength)
        views.append((f"autocontrast_{strength:g}", transformed, {"strength": float(strength), "cutoff": 1}))
    return views

def match_mean_std(image: Image.Image, mean: np.ndarray, std: np.ndarray, strength: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    cur_mean = arr.reshape(-1, 3).mean(axis=0)
    cur_std = arr.reshape(-1, 3).std(axis=0) + 1e-6
    target = (arr - cur_mean) / cur_std * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    out = (1.0 - strength) * arr + strength * target
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def build_native_support(adapter, selected_rows, train_qids, greedy, max_side, prototype_tokens_per_image: int):
    native = []
    for row in tqdm(selected_rows, desc="native support scan"):
        qid = str(row["qid"])
        if qid not in train_qids or qid not in greedy or not greedy[qid]["correct"]:
            continue
        path = resolve_image(row.get("img_name", ""))
        if path is None:
            continue
        with Image.open(path) as raw:
            image = resize_image(raw.convert("RGB"), max_side)
        tokens = adapter.visual_tokens([image])[0].astype(np.float32)
        native.append(
            {
                "qid": qid,
                "image": image,
                "pooled": tokens.mean(axis=0).astype(np.float32),
                "prototypes": sample_token_prototypes(tokens, prototype_tokens_per_image),
                "amplitude": fft_amplitude(image),
                "rgb_mean": np.asarray(image).reshape(-1, 3).mean(axis=0).astype(np.float32),
                "rgb_std": np.asarray(image).reshape(-1, 3).std(axis=0).astype(np.float32),
            }
        )
    if len(native) < 2:
        raise RuntimeError("not enough competence-native support images")
    center = np.mean([r["pooled"] for r in native], axis=0)
    rgb_mean = np.mean([r["rgb_mean"] for r in native], axis=0)
    rgb_std = np.mean([r["rgb_std"] for r in native], axis=0)
    # Amplitudes may have different spatial sizes; resize by reusing the
    # per-neighbor amplitudes for FDA and use nearest-native instead of a brittle
    # global pixel center.
    prototypes = np.concatenate([r["prototypes"] for r in native], axis=0).astype(np.float32)
    return native, center.astype(np.float32), prototypes, rgb_mean, rgb_std


def candidate_views(image: Image.Image, original_pooled: np.ndarray, native, native_center, rgb_mean, rgb_std, args):
    native_ranked = sorted(
        native,
        key=lambda r: cosine_distance(original_pooled, r["pooled"]),
    )[: max(1, args.top_native)]
    views = [
        {
            "name": "original",
            "image": image,
            "family": "original",
            "params": {},
            "structure": {"pixel_mse": 0.0, "psnr": None, "edge_correlation": 1.0},
        }
    ]
    for nidx, item in enumerate(native_ranked):
        for l_value in args.l_values:
            for source_ratio in args.source_ratios:
                transformed = feddg_frequency_interpolation(
                    image,
                    item["amplitude"],
                    low_frequency_ratio=float(l_value),
                    source_ratio=float(source_ratio),
                )
                views.append(
                    {
                        "name": f"fda_nn{nidx}_l{l_value:g}_sr{source_ratio:g}",
                        "image": transformed,
                        "family": "fda_nearest_native",
                        "params": {
                            "native_qid": item["qid"],
                            "low_frequency_ratio": float(l_value),
                            "source_ratio": float(source_ratio),
                        },
                        "structure": structure_metrics(image, transformed),
                    }
                )
    for strength in (0.25, 0.5, 0.75):
        transformed = match_mean_std(image, rgb_mean, rgb_std, strength)
        views.append(
            {
                "name": f"meanstd_s{strength:g}",
                "image": transformed,
                "family": "mean_std_match",
                "params": {"strength": float(strength)},
                "structure": structure_metrics(image, transformed),
            }
        )
    for name, transformed, params in photometric_views(image):
        views.append(
            {
                "name": name,
                "image": transformed,
                "family": "photometric_native",
                "params": params,
                "structure": structure_metrics(image, transformed),
            }
        )
    return views


def is_structure_safe(view, args) -> bool:
    if view["name"] == "original":
        return True
    s = view.get("structure", {})
    psnr = s.get("psnr")
    edge = s.get("edge_correlation")
    return (
        psnr is not None
        and float(psnr) >= args.min_psnr
        and edge is not None
        and float(edge) >= args.min_edge_correlation
    )


def main():
    args = parse_args()
    rows_raw = json.loads(args.dataset.read_text())
    greedy, greedy_meta = load_eval(args.greedy_eval)
    selected_rows = eligible_rows(rows_raw, args.question_type, args.seed, args.max_samples)
    qids = [str(r["qid"]) for r in selected_rows]
    train_cut = int(round(args.train_frac * len(qids)))
    train_qids = set(qids[:train_cut])

    adapter = load_local_source_adapter(args.model)
    native, native_center, native_prototypes, rgb_mean, rgb_std = build_native_support(
        adapter,
        selected_rows,
        train_qids,
        greedy,
        args.max_image_side,
        args.prototype_tokens_per_image,
    )

    out_rows = []
    for row in tqdm(selected_rows, desc="native aligned views"):
        qid = str(row["qid"])
        if qid not in greedy:
            continue
        path = resolve_image(row.get("img_name", ""))
        if path is None:
            continue
        with Image.open(path) as raw:
            image = resize_image(raw.convert("RGB"), args.max_image_side)
        labels = labels_for_sample(row)
        gt = ground_truth_index(row)
        prompt = build_prompt(row)
        original_tokens = adapter.visual_tokens([image])[0].astype(np.float32)
        original_pooled = original_tokens.mean(axis=0).astype(np.float32)
        original_distance = cosine_distance(original_pooled, native_center)
        views = candidate_views(image, original_pooled, native, native_center, rgb_mean, rgb_std, args)

        # Measure closure for every view, then select a small set for expensive CE.
        measured = []
        for view in views:
            if view["name"] == "original":
                pooled = original_pooled
            else:
                pooled = adapter.visual_tokens([view["image"]])[0].astype(np.float32).mean(axis=0)
            dist = cosine_distance(pooled, native_center)
            closure = original_distance - dist
            measured.append(
                {
                    **{k: v for k, v in view.items() if k != "image"},
                    "native_distance": dist,
                    "native_closure": closure,
                    "structure_safe": is_structure_safe(view, args),
                }
            )

        accepted_idx = [
            i
            for i, m in enumerate(measured)
            if i > 0 and m["structure_safe"] and float(m["native_closure"]) > args.min_closure
        ]
        accepted_idx = sorted(
            accepted_idx,
            key=lambda i: (measured[i]["native_closure"], -measured[i]["native_distance"]),
            reverse=True,
        )[: max(0, args.max_views_forward - 1)]
        forward_idx = [0] + accepted_idx
        forward_images = [views[i]["image"] for i in forward_idx]
        pixel_evidence = adapter.forward_ce(forward_images, prompt, labels)

        candidates = []
        for local_index, evidence in enumerate(pixel_evidence):
            view_index = forward_idx[local_index]
            view_meta = measured[view_index]
            candidates.append(
                {
                    "name": view_meta["name"],
                    "family": view_meta["family"],
                    "params": view_meta.get("params", {}),
                    "view_index": int(view_index),
                    "native_distance": view_meta.get("native_distance"),
                    "native_closure": view_meta.get("native_closure"),
                    "structure": view_meta.get("structure"),
                    "structure_safe": bool(view_meta.get("structure_safe", False)),
                    "logits": evidence.logits.astype(np.float32),
                    "features": evidence.features.astype(np.float32),
                    "sequence_nll": evidence.sequence_nll.astype(np.float32),
                }
            )

        for beta in args.transport_betas:
            evidence = adapter.forward_ce_local_transport(
                image,
                prompt,
                labels,
                native_prototypes,
                beta=float(beta),
                confidence_power=float(args.transport_confidence_power),
            )
            candidates.append(
                {
                    "name": f"token_native_beta{beta:g}",
                    "family": "feature_token_native",
                    "params": {
                        "beta": float(beta),
                        "prototype_count": int(native_prototypes.shape[0]),
                        "confidence_power": float(args.transport_confidence_power),
                    },
                    "view_index": None,
                    "native_distance": None,
                    "native_closure": None,
                    "structure": {"pixel_mse": 0.0, "psnr": None, "edge_correlation": 1.0},
                    "structure_safe": True,
                    "logits": evidence.logits.astype(np.float32),
                    "features": evidence.features.astype(np.float32),
                    "sequence_nll": evidence.sequence_nll.astype(np.float32),
                }
            )

        logits = np.stack([c["logits"] for c in candidates])
        features = np.stack([c["features"] for c in candidates])
        sequence_nll = np.stack([c["sequence_nll"] for c in candidates])
        pred = logits.argmax(axis=1)
        nll_pred = sequence_nll.argmin(axis=1)
        probs = softmax_np(logits, axis=-1)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=-1)
        margins = margin(logits)

        temperature = max(float(args.fusion_temperature), 1e-6)
        entropy_weights = softmax_np(-entropy / temperature, axis=0)
        margin_weights = softmax_np(margins / temperature, axis=0)
        nll_weights = softmax_np(-sequence_nll.min(axis=1) / temperature, axis=0)
        entropy_fused_logits = np.sum(entropy_weights[:, None] * logits, axis=0)
        margin_fused_logits = np.sum(margin_weights[:, None] * logits, axis=0)
        laplacian_probs = laplacian_fused_probability(probs, features, args.laplacian_lambda)

        best_entropy_local = int(np.argmin(entropy))
        best_margin_local = int(np.argmax(margins))
        best_nll_local = int(np.argmin(sequence_nll.min(axis=1)))
        entropy_fusion_prediction = int(np.argmax(entropy_fused_logits))
        margin_fusion_prediction = int(np.argmax(margin_fused_logits))
        nll_fusion_prediction = weighted_nll_prediction(sequence_nll, nll_weights)
        laplacian_fusion_prediction = int(np.argmax(laplacian_probs))

        candidate_public = []
        for idx, candidate in enumerate(candidates):
            candidate_public.append(
                {
                    "name": candidate["name"],
                    "family": candidate["family"],
                    "params": candidate["params"],
                    "view_index": candidate["view_index"],
                    "native_distance": candidate["native_distance"],
                    "native_closure": candidate["native_closure"],
                    "structure": candidate["structure"],
                    "structure_safe": candidate["structure_safe"],
                    "logits": candidate["logits"].tolist(),
                    "sequence_nll": candidate["sequence_nll"].tolist(),
                    "logit_prediction": int(pred[idx]),
                    "nll_prediction": int(nll_pred[idx]),
                    "entropy": float(entropy[idx]),
                    "margin": float(margins[idx]),
                }
            )

        original_correct = bool(pred[0] == gt)
        best_entropy_correct = bool(pred[best_entropy_local] == gt)
        best_margin_correct = bool(pred[best_margin_local] == gt)
        best_nll_correct = bool(nll_pred[best_nll_local] == gt)
        entropy_fusion_correct = bool(entropy_fusion_prediction == gt)
        margin_fusion_correct = bool(margin_fusion_prediction == gt)
        nll_fusion_correct = bool(nll_fusion_prediction == gt)
        laplacian_fusion_correct = bool(laplacian_fusion_prediction == gt)

        out_rows.append(
            {
                "qid": qid,
                "split": "train" if qid in train_qids else "test",
                "img_name": row.get("img_name"),
                "gt_index": int(gt),
                "labels": list(labels),
                "greedy_eval_correct": bool(greedy[qid]["correct"]),
                "original_distance": original_distance,
                "views": measured,
                "forward_indices": forward_idx,
                "accepted_indices": accepted_idx,
                "accepted_view_count": len(accepted_idx),
                "candidate_names": [c["name"] for c in candidate_public],
                "candidates": candidate_public,
                "style_names": [c["name"] for c in candidate_public],
                "style_logits": logits.tolist(),
                "style_predictions": pred.astype(int).tolist(),
                "style_nll_predictions": nll_pred.astype(int).tolist(),
                "style_correct": (pred == gt).astype(bool).tolist(),
                "style_nll_correct": (nll_pred == gt).astype(bool).tolist(),
                "best_view_index": int(candidates[best_entropy_local]["view_index"] or 0),
                "best_view_name": candidates[best_entropy_local]["name"],
                "best_view_prediction": int(pred[best_entropy_local]),
                "best_view_correct": best_entropy_correct,
                "best_entropy_name": candidates[best_entropy_local]["name"],
                "best_entropy_prediction": int(pred[best_entropy_local]),
                "best_entropy_correct": best_entropy_correct,
                "best_margin_name": candidates[best_margin_local]["name"],
                "best_margin_prediction": int(pred[best_margin_local]),
                "best_margin_correct": best_margin_correct,
                "best_nll_name": candidates[best_nll_local]["name"],
                "best_nll_prediction": int(nll_pred[best_nll_local]),
                "best_nll_correct": best_nll_correct,
                "entropy_fusion_weights": entropy_weights.tolist(),
                "entropy_fusion_logits": entropy_fused_logits.tolist(),
                "entropy_fusion_prediction": entropy_fusion_prediction,
                "entropy_fusion_correct": entropy_fusion_correct,
                "margin_fusion_weights": margin_weights.tolist(),
                "margin_fusion_logits": margin_fused_logits.tolist(),
                "margin_fusion_prediction": margin_fusion_prediction,
                "margin_fusion_correct": margin_fusion_correct,
                "nll_fusion_weights": nll_weights.tolist(),
                "nll_fusion_prediction": nll_fusion_prediction,
                "nll_fusion_correct": nll_fusion_correct,
                "laplacian_fusion_probs": laplacian_probs.tolist(),
                "laplacian_fusion_prediction": laplacian_fusion_prediction,
                "laplacian_fusion_correct": laplacian_fusion_correct,
                "fused_weights": entropy_weights.tolist(),
                "fused_logits": entropy_fused_logits.tolist(),
                "fused_prediction": entropy_fusion_prediction,
                "fused_correct": entropy_fusion_correct,
            }
        )

    summary = {
        "version": VERSION,
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "greedy_eval": str(args.greedy_eval.resolve()),
        "n": len(out_rows),
        "train_frac": args.train_frac,
        "support_n": len(native),
        "max_samples": args.max_samples,
        "grid": {
            "l_values": list(args.l_values),
            "source_ratios": list(args.source_ratios),
            "top_native": args.top_native,
            "min_psnr": args.min_psnr,
            "min_edge_correlation": args.min_edge_correlation,
            "min_closure": args.min_closure,
            "transport_betas": list(args.transport_betas),
            "prototype_tokens_per_image": args.prototype_tokens_per_image,
            "native_prototype_count": int(native_prototypes.shape[0]),
            "fusion_temperature": args.fusion_temperature,
            "laplacian_lambda": args.laplacian_lambda,
            "transport_confidence_power": args.transport_confidence_power,
        },
    }

    def summarize_subset(subset):
        methods = {
            "original": [r["style_correct"][0] for r in subset],
            "entropy_best": [r["best_entropy_correct"] for r in subset],
            "margin_best": [r["best_margin_correct"] for r in subset],
            "nll_best": [r["best_nll_correct"] for r in subset],
            "entropy_fusion": [r["entropy_fusion_correct"] for r in subset],
            "margin_fusion": [r["margin_fusion_correct"] for r in subset],
            "nll_fusion": [r["nll_fusion_correct"] for r in subset],
            "laplacian_fusion": [r["laplacian_fusion_correct"] for r in subset],
        }
        block = {
            "n": len(subset),
            "accepted_rate": acc([r["accepted_view_count"] > 0 for r in subset]),
            "mean_accepted": float(np.mean([r["accepted_view_count"] for r in subset])) if subset else None,
        }
        correctness_key = {
            "entropy_best": "best_entropy_correct",
            "margin_best": "best_margin_correct",
            "nll_best": "best_nll_correct",
            "entropy_fusion": "entropy_fusion_correct",
            "margin_fusion": "margin_fusion_correct",
            "nll_fusion": "nll_fusion_correct",
            "laplacian_fusion": "laplacian_fusion_correct",
        }
        for name, values in methods.items():
            block[f"{name}_accuracy"] = acc(values)
            if name != "original":
                key = correctness_key[name]
                block[f"rescues_{name}"] = sum((not r["style_correct"][0]) and r[key] for r in subset)
                block[f"harmful_{name}"] = sum(r["style_correct"][0] and (not r[key]) for r in subset)
        # Backward-compatible aliases for older notebooks.
        block["best_view_accuracy"] = block["entropy_best_accuracy"]
        block["fused_accuracy"] = block["entropy_fusion_accuracy"]
        block["rescues_best"] = block.get("rescues_entropy_best", 0)
        block["harmful_best"] = block.get("harmful_entropy_best", 0)
        block["rescues_fused"] = block.get("rescues_entropy_fusion", 0)
        block["harmful_fused"] = block.get("harmful_entropy_fusion", 0)
        return block

    summary["overall"] = summarize_subset(out_rows)
    for split in ("train", "test"):
        split_rows = [r for r in out_rows if r["split"] == split]
        summary[split] = summarize_subset(split_rows)
    payload = {"summary": summary, "rows": out_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    adapter.close()


if __name__ == "__main__":
    main()
