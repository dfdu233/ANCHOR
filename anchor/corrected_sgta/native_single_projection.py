"""One-shot native-source image projection for frozen medical VLMs.

The goal is intentionally narrower than multi-view SGTA: estimate a VLM-native
visual support, project each test image once toward that support with a small
parameterized image transform, then run the VLM on the single aligned image.

This is a DDA/SDA-inspired diagnostic without a diffusion model: input-side
source projection, frozen VLM, and explicit structure preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFile
from tqdm import tqdm

from corrected_sgta.infer_ce import resize_image
from corrected_sgta.methods import feddg_frequency_interpolation, softmax_np
from corrected_sgta.models_local_source import LlavaLocalSourceAdapter
from corrected_sgta.native_view_projection import (
    cosine_distance,
    fft_amplitude,
    match_mean_std,
    structure_metrics,
)
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
VERSION = "native-single-projection-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--greedy-eval", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-type", default="binary")
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--support-mode", choices=("competence", "self_conf"), default="self_conf")
    parser.add_argument("--support-top-frac", type=float, default=0.8)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--max-support", type=int, default=64)
    parser.add_argument("--projection-mode", choices=("improve_only", "forced", "distance_only"), default="forced")
    parser.add_argument("--families", nargs="*", default=("bary_fda", "nearest_fda", "meanstd", "photo", "combo"))
    parser.add_argument("--l-values", type=float, nargs="*", default=(0.003, 0.01, 0.03, 0.06, 0.1))
    parser.add_argument("--source-ratios", type=float, nargs="*", default=(0.0, 0.2, 0.5, 0.8))
    parser.add_argument("--top-native", type=int, default=3)
    parser.add_argument("--meanstd-strengths", type=float, nargs="*", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--gamma-values", type=float, nargs="*", default=(0.75, 0.9, 1.1, 1.25))
    parser.add_argument("--contrast-values", type=float, nargs="*", default=(0.8, 0.9, 1.1, 1.25))
    parser.add_argument("--brightness-values", type=float, nargs="*", default=(0.9, 1.1))
    parser.add_argument("--sharpness-values", type=float, nargs="*", default=(0.8, 1.2))
    parser.add_argument("--min-psnr", type=float, default=16.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.80)
    parser.add_argument("--structure-weight", type=float, default=0.05)
    parser.add_argument("--edge-weight", type=float, default=0.02)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--save-preview-dir", type=Path, default=None)
    parser.add_argument("--save-preview-count", type=int, default=12)
    return parser.parse_args()


def load_eval(path: Path | None):
    if path is None or not path.exists():
        return {}, None
    payload = json.loads(path.read_text())
    return {str(d["question_id"]): d for d in payload.get("details", [])}, payload


def entropy_from_logits(logits: np.ndarray) -> float:
    probs = softmax_np(np.asarray(logits, dtype=np.float64))
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))


def margin_from_logits(logits: np.ndarray) -> float:
    values = np.sort(np.asarray(logits, dtype=np.float64))
    return float(values[-1] - values[-2]) if len(values) >= 2 else float(values[-1])


def acc(values):
    return float(np.mean(values)) if values else None


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


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def resize_amplitude_to(target_amplitudes: list[np.ndarray], size_hw: tuple[int, int]) -> np.ndarray:
    resized = []
    height, width = size_hw
    for amp in target_amplitudes:
        tensor = torch.as_tensor(np.asarray(amp, dtype=np.float32)).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        resized.append(tensor.squeeze(0).numpy())
    return np.mean(resized, axis=0).astype(np.float32)


def gamma_transform(image: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), 1.0 / float(gamma))
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def enhance_image(image: Image.Image, kind: str, factor: float) -> Image.Image:
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(factor))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(factor))
    if kind == "sharpness":
        return ImageEnhance.Sharpness(image).enhance(float(factor))
    raise ValueError(kind)


def build_support(adapter, selected_rows, train_qids, greedy, args):
    scanned = []
    for row in tqdm(selected_rows, desc=f"support scan [{args.support_mode}]"):
        qid = str(row["qid"])
        if qid not in train_qids:
            continue
        path = resolve_image(row.get("img_name", ""))
        if path is None:
            continue
        with Image.open(path) as raw:
            image = resize_image(raw.convert("RGB"), args.max_image_side)
        labels = labels_for_sample(row)
        prompt = build_prompt(row)
        if args.support_mode == "competence":
            if qid not in greedy or not bool(greedy[qid].get("correct")):
                continue
            score = 1.0
            entropy = None
            pred = None
        else:
            evidence = adapter.forward_ce([image], prompt, labels)[0]
            entropy = entropy_from_logits(evidence.logits)
            margin = margin_from_logits(evidence.logits)
            pred = int(np.argmax(evidence.logits))
            # label-free native reliability: confident, low-entropy samples form
            # the proxy native support.  The small margin term avoids ties in
            # near-uniform Yes/No outputs.
            score = -entropy + 0.05 * margin
        scanned.append({"qid": qid, "image": image, "score": float(score), "entropy": entropy, "pred": pred})

    if args.support_mode == "self_conf":
        scanned.sort(key=lambda x: x["score"], reverse=True)
        count = int(round(float(args.support_top_frac) * len(scanned)))
        count = min(max(count, int(args.min_support)), int(args.max_support), len(scanned))
        scanned = scanned[:count]
    else:
        scanned = scanned[: int(args.max_support)]
    if len(scanned) < int(args.min_support):
        raise RuntimeError(f"not enough native support samples: {len(scanned)} < {args.min_support}")

    support = []
    for item in tqdm(scanned, desc="native token/amplitude bank"):
        tokens = adapter.visual_tokens([item["image"]])[0].astype(np.float32)
        arr = image_array(item["image"])
        support.append(
            {
                "qid": item["qid"],
                "image": item["image"],
                "pooled": tokens.mean(axis=0).astype(np.float32),
                "amplitude": fft_amplitude(item["image"]),
                "rgb_mean": arr.reshape(-1, 3).mean(axis=0).astype(np.float32),
                "rgb_std": arr.reshape(-1, 3).std(axis=0).astype(np.float32),
                "support_score": item["score"],
                "support_entropy": item["entropy"],
            }
        )
    center = np.mean([x["pooled"] for x in support], axis=0).astype(np.float32)
    rgb_mean = np.mean([x["rgb_mean"] for x in support], axis=0).astype(np.float32)
    rgb_std = np.mean([x["rgb_std"] for x in support], axis=0).astype(np.float32)
    return support, center, rgb_mean, rgb_std


def candidate_pool(image: Image.Image, original_pooled: np.ndarray, support, native_center, rgb_mean, rgb_std, args):
    families = set(args.families)
    height, width = image.size[1], image.size[0]
    pool = [
        {
            "name": "original",
            "family": "original",
            "params": {},
            "image": image,
            "structure": {"pixel_mse": 0.0, "psnr": None, "edge_correlation": 1.0},
        }
    ]
    nearest = sorted(support, key=lambda x: cosine_distance(original_pooled, x["pooled"]))[: max(1, args.top_native)]

    if "bary_fda" in families:
        bary_amp = resize_amplitude_to([x["amplitude"] for x in support], (height, width))
        for l_value in args.l_values:
            for source_ratio in args.source_ratios:
                transformed = feddg_frequency_interpolation(image, bary_amp, l_value, source_ratio)
                pool.append({"name": f"bary_fda_l{l_value:g}_sr{source_ratio:g}", "family": "bary_fda", "params": {"low_frequency_ratio": float(l_value), "source_ratio": float(source_ratio)}, "image": transformed, "structure": structure_metrics(image, transformed)})

    if "nearest_fda" in families:
        for nidx, item in enumerate(nearest):
            for l_value in args.l_values:
                for source_ratio in args.source_ratios:
                    transformed = feddg_frequency_interpolation(image, item["amplitude"], l_value, source_ratio)
                    pool.append({"name": f"nearest{nidx}_fda_l{l_value:g}_sr{source_ratio:g}", "family": "nearest_fda", "params": {"native_qid": item["qid"], "low_frequency_ratio": float(l_value), "source_ratio": float(source_ratio)}, "image": transformed, "structure": structure_metrics(image, transformed)})

    if "meanstd" in families:
        for strength in args.meanstd_strengths:
            transformed = match_mean_std(image, rgb_mean, rgb_std, strength)
            pool.append({"name": f"meanstd_s{strength:g}", "family": "meanstd", "params": {"strength": float(strength)}, "image": transformed, "structure": structure_metrics(image, transformed)})

    if "photo" in families:
        for gamma in args.gamma_values:
            transformed = gamma_transform(image, gamma)
            pool.append({"name": f"gamma_{gamma:g}", "family": "photo", "params": {"gamma": float(gamma)}, "image": transformed, "structure": structure_metrics(image, transformed)})
        for kind, values in (("contrast", args.contrast_values), ("brightness", args.brightness_values), ("sharpness", args.sharpness_values)):
            for value in values:
                transformed = enhance_image(image, kind, value)
                pool.append({"name": f"{kind}_{value:g}", "family": "photo", "params": {kind: float(value)}, "image": transformed, "structure": structure_metrics(image, transformed)})

    if "combo" in families:
        # A tiny, interpretable combo set: source-spectrum projection followed
        # by native first/second-order intensity matching.  This is still one
        # final image, not an ensemble.
        for l_value in (0.03, 0.06, 0.1):
            bary_amp = resize_amplitude_to([x["amplitude"] for x in nearest], (height, width))
            base = feddg_frequency_interpolation(image, bary_amp, l_value, 0.2)
            for strength in (0.25, 0.5, 0.75):
                transformed = match_mean_std(base, rgb_mean, rgb_std, strength)
                pool.append({"name": f"combo_l{l_value:g}_ms{strength:g}", "family": "combo", "params": {"low_frequency_ratio": float(l_value), "source_ratio": 0.2, "meanstd_strength": float(strength)}, "image": transformed, "structure": structure_metrics(image, transformed)})

    # Deterministic cap: keep broad family diversity and avoid runaway per-image
    # visual-token scoring cost.
    if args.max_candidates and len(pool) > args.max_candidates:
        original = pool[:1]
        rest = pool[1:]
        rest.sort(key=lambda c: (c["family"], c["name"]))
        idx = np.linspace(0, len(rest) - 1, num=args.max_candidates - 1, dtype=int)
        pool = original + [rest[i] for i in idx]
    return pool


def structure_penalty(structure, args):
    psnr = structure.get("psnr")
    edge = structure.get("edge_correlation")
    value = 0.0
    if psnr is not None and float(psnr) < float(args.min_psnr):
        value += (float(args.min_psnr) - float(psnr)) / max(float(args.min_psnr), 1e-6)
    if edge is not None and float(edge) < float(args.min_edge_correlation):
        value += float(args.edge_weight) * (float(args.min_edge_correlation) - float(edge))
    return value


def choose_projection(adapter, image, support, native_center, rgb_mean, rgb_std, args):
    original_tokens = adapter.visual_tokens([image])[0].astype(np.float32)
    original_pooled = original_tokens.mean(axis=0).astype(np.float32)
    original_distance = cosine_distance(original_pooled, native_center)
    pool = candidate_pool(image, original_pooled, support, native_center, rgb_mean, rgb_std, args)
    scored = []
    for candidate in pool:
        if candidate["family"] == "original":
            pooled = original_pooled
        else:
            pooled = adapter.visual_tokens([candidate["image"]])[0].astype(np.float32).mean(axis=0)
        distance = cosine_distance(pooled, native_center)
        penalty = structure_penalty(candidate["structure"], args)
        score = distance + float(args.structure_weight) * penalty
        scored.append({k: v for k, v in candidate.items() if k != "image"} | {"native_distance": distance, "native_closure": original_distance - distance, "structure_penalty": penalty, "objective": score})

    original_score = scored[0]["objective"]
    non_original = list(range(1, len(scored)))
    if args.projection_mode == "distance_only":
        chosen_idx = min(non_original or [0], key=lambda i: scored[i]["native_distance"])
    elif args.projection_mode == "forced":
        chosen_idx = min(non_original or [0], key=lambda i: scored[i]["objective"])
    else:
        best_non_original = min(non_original or [0], key=lambda i: scored[i]["objective"])
        chosen_idx = best_non_original if scored[best_non_original]["objective"] < original_score else 0
    chosen = pool[chosen_idx]
    return chosen["image"], scored, int(chosen_idx)


def save_preview(source: Image.Image, aligned: Image.Image, row, output_dir: Path, rank: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = source.size
    canvas = Image.new("RGB", (width * 2, height), "white")
    canvas.paste(source.convert("RGB"), (0, 0))
    canvas.paste(aligned.convert("RGB"), (width, 0))
    safe_qid = str(row.get("qid", rank)).replace("/", "_")
    canvas.save(output_dir / f"{rank:03d}_qid{safe_qid}_orig_aligned.jpg", quality=92)


def main():
    args = parse_args()
    rows_raw = json.loads(args.dataset.read_text())
    greedy, _ = load_eval(args.greedy_eval)
    selected_rows = eligible_rows(rows_raw, args.question_type, args.seed, args.max_samples)
    qids = [str(r["qid"]) for r in selected_rows]
    train_cut = int(round(args.train_frac * len(qids)))
    train_qids = set(qids[:train_cut])

    adapter = LlavaLocalSourceAdapter()
    support, native_center, rgb_mean, rgb_std = build_support(adapter, selected_rows, train_qids, greedy, args)

    out_rows = []
    preview_count = 0
    for row in tqdm(selected_rows, desc=f"single native projection [{args.projection_mode}]"):
        qid = str(row["qid"])
        path = resolve_image(row.get("img_name", ""))
        if path is None:
            continue
        with Image.open(path) as raw:
            image = resize_image(raw.convert("RGB"), args.max_image_side)
        labels = labels_for_sample(row)
        gt = ground_truth_index(row)
        prompt = build_prompt(row)
        aligned_image, scored, chosen_idx = choose_projection(adapter, image, support, native_center, rgb_mean, rgb_std, args)
        evidence = adapter.forward_ce([image, aligned_image], prompt, labels)
        original_logits = evidence[0].logits.astype(np.float32)
        aligned_logits = evidence[1].logits.astype(np.float32)
        original_pred = int(np.argmax(original_logits))
        aligned_pred = int(np.argmax(aligned_logits))
        original_correct = bool(original_pred == gt)
        aligned_correct = bool(aligned_pred == gt)
        best_candidate = min(range(len(scored)), key=lambda i: scored[i]["objective"])
        if args.save_preview_dir and preview_count < args.save_preview_count and chosen_idx != 0:
            save_preview(image, aligned_image, row, args.save_preview_dir, preview_count)
            preview_count += 1
        out_rows.append(
            {
                "qid": qid,
                "split": "train" if qid in train_qids else "test",
                "img_name": row.get("img_name"),
                "gt_index": int(gt),
                "labels": list(labels),
                "greedy_eval_correct": bool(greedy.get(qid, {}).get("correct", original_correct)),
                "original_logits": original_logits.tolist(),
                "aligned_logits": aligned_logits.tolist(),
                "original_prediction": original_pred,
                "aligned_prediction": aligned_pred,
                "original_correct": original_correct,
                "aligned_correct": aligned_correct,
                "chosen_index": chosen_idx,
                "chosen_name": scored[chosen_idx]["name"],
                "chosen_family": scored[chosen_idx]["family"],
                "chosen_params": scored[chosen_idx].get("params", {}),
                "chosen_native_distance": scored[chosen_idx]["native_distance"],
                "chosen_native_closure": scored[chosen_idx]["native_closure"],
                "chosen_objective": scored[chosen_idx]["objective"],
                "original_native_distance": scored[0]["native_distance"],
                "original_objective": scored[0]["objective"],
                "best_candidate_name": scored[best_candidate]["name"],
                "best_candidate_family": scored[best_candidate]["family"],
                "candidate_count": len(scored),
                "candidate_family_counts": dict(Counter(x["family"] for x in scored)),
                "candidates": scored,
            }
        )

    def summarize(subset):
        block = {
            "n": len(subset),
            "original_accuracy": acc([r["original_correct"] for r in subset]),
            "aligned_accuracy": acc([r["aligned_correct"] for r in subset]),
            "rescues": sum((not r["original_correct"]) and r["aligned_correct"] for r in subset),
            "harmful": sum(r["original_correct"] and (not r["aligned_correct"]) for r in subset),
            "changed_prediction": sum(r["original_prediction"] != r["aligned_prediction"] for r in subset),
            "aligned_rate": acc([r["chosen_index"] != 0 for r in subset]),
            "mean_native_closure": float(np.mean([r["chosen_native_closure"] for r in subset])) if subset else None,
            "mean_logit_delta": float(np.mean([float(np.linalg.norm(np.asarray(r["aligned_logits"]) - np.asarray(r["original_logits"]))) for r in subset])) if subset else None,
        }
        families = Counter(r["chosen_family"] for r in subset)
        block["chosen_families"] = dict(families)
        return block

    summary = {
        "version": VERSION,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "greedy_eval": str(args.greedy_eval.resolve()) if args.greedy_eval else None,
        "n": len(out_rows),
        "max_samples": args.max_samples,
        "train_frac": args.train_frac,
        "support_mode": args.support_mode,
        "projection_mode": args.projection_mode,
        "support_n": len(support),
        "support_qids": [x["qid"] for x in support],
        "grid": {
            "families": list(args.families),
            "l_values": list(args.l_values),
            "source_ratios": list(args.source_ratios),
            "top_native": args.top_native,
            "meanstd_strengths": list(args.meanstd_strengths),
            "gamma_values": list(args.gamma_values),
            "contrast_values": list(args.contrast_values),
            "brightness_values": list(args.brightness_values),
            "sharpness_values": list(args.sharpness_values),
            "min_psnr": args.min_psnr,
            "min_edge_correlation": args.min_edge_correlation,
            "structure_weight": args.structure_weight,
            "edge_weight": args.edge_weight,
            "max_candidates": args.max_candidates,
        },
    }
    summary["overall"] = summarize(out_rows)
    for split in ("train", "test"):
        summary[split] = summarize([r for r in out_rows if r["split"] == split])

    payload = {"summary": summary, "rows": out_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    adapter.close()


if __name__ == "__main__":
    main()
