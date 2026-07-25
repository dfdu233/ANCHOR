"""Gradient one-shot native projection through the frozen VLM vision encoder.

This probes whether input-side native-source alignment can move the VLM decision
boundary when the image is optimized directly in the VLM visual feature space.
Only a low-resolution smooth residual is optimized; the VLM weights remain
frozen and the final prediction uses a single aligned image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_local_source import LlavaLocalSourceAdapter
from corrected_sgta.native_single_projection import acc, build_support, eligible_rows, entropy_from_logits
from corrected_sgta.native_view_projection import cosine_distance, structure_metrics
from corrected_sgta.protocol_v2 import build_prompt, file_sha256, ground_truth_index, labels_for_sample, resolve_image

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "native-gradient-projection-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--greedy-eval", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-type", default="binary")
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--support-mode", choices=("competence", "self_conf"), default="self_conf")
    parser.add_argument("--support-top-frac", type=float, default=0.8)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--max-support", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--l2-weight", type=float, default=0.5)
    parser.add_argument("--tv-weight", type=float, default=0.05)
    parser.add_argument("--target", choices=("center", "nearest_support"), default="center")
    parser.add_argument("--save-preview-dir", type=Path, default=None)
    parser.add_argument("--save-preview-count", type=int, default=8)
    return parser.parse_args()


def load_eval(path: Path | None):
    if path is None or not path.exists():
        return {}, None
    payload = json.loads(path.read_text())
    return {str(d["question_id"]): d for d in payload.get("details", [])}, payload


def tensor_from_image(adapter, image: Image.Image) -> torch.Tensor:
    from llava.mm_utils import process_images

    tensor = process_images([image], adapter.image_processor, adapter.model.config)
    if isinstance(tensor, list):
        tensor = tensor[0].unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.to(adapter.model.device, dtype=torch.float32)


def image_from_pixel(pixel: torch.Tensor) -> Image.Image:
    arr = pixel.detach().float().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def tv_loss(delta: torch.Tensor) -> torch.Tensor:
    return (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean() + (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()


def encode_pooled(adapter, normalized: torch.Tensor) -> torch.Tensor:
    # LLaVA-Med wraps CLIPVisionTower.forward in @torch.no_grad().  For input
    # projection we bypass that wrapper while keeping all model weights frozen,
    # so gradients flow only to the image residual.
    vision_tower = adapter.model.get_model().get_vision_tower()
    images = normalized.to(device=vision_tower.device, dtype=vision_tower.dtype)
    image_forward_outs = vision_tower.vision_tower(images, output_hidden_states=True)
    image_features = vision_tower.feature_select(image_forward_outs).to(images.dtype)
    projected = adapter.model.get_model().mm_projector(image_features)
    if projected.ndim == 2:
        pooled = projected.float().mean(dim=0)
    elif projected.ndim == 3:
        pooled = projected.float()[0].mean(dim=0)
    else:
        raise RuntimeError(f"unexpected feature shape: {tuple(projected.shape)}")
    return pooled


def choose_target(original_pooled_np: np.ndarray, support, native_center: np.ndarray, mode: str) -> np.ndarray:
    if mode == "center":
        return native_center
    best = min(support, key=lambda item: cosine_distance(original_pooled_np, item["pooled"]))
    return best["pooled"].astype(np.float32)


def gradient_project(adapter, image: Image.Image, support, native_center: np.ndarray, args):
    base_norm = tensor_from_image(adapter, image)
    mean = torch.tensor(adapter.image_processor.image_mean, device=base_norm.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(adapter.image_processor.image_std, device=base_norm.device, dtype=torch.float32).view(1, 3, 1, 1)
    base_pixel = (base_norm * std + mean).clamp(0, 1).detach()
    with torch.no_grad():
        original_pooled = encode_pooled(adapter, base_norm).detach().float()
    target_np = choose_target(original_pooled.detach().cpu().numpy(), support, native_center, args.target)
    target = torch.tensor(target_np, device=base_norm.device, dtype=torch.float32)
    target = F.normalize(target, dim=0)
    delta_low = torch.zeros((1, 3, int(args.grid_size), int(args.grid_size)), device=base_norm.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([delta_low], lr=float(args.lr))
    history = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        delta = torch.tanh(delta_low) * float(args.epsilon)
        delta = F.interpolate(delta, size=base_pixel.shape[-2:], mode="bicubic", align_corners=False)
        pixel = (base_pixel + delta).clamp(0, 1)
        normed = (pixel - mean) / std
        pooled = encode_pooled(adapter, normed)
        cosine = 1.0 - torch.sum(F.normalize(pooled, dim=0) * target)
        l2 = delta.pow(2).mean()
        tv = tv_loss(delta)
        loss = cosine + float(args.l2_weight) * l2 + float(args.tv_weight) * tv
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach().cpu()), "cosine_distance": float(cosine.detach().cpu()), "l2": float(l2.detach().cpu()), "tv": float(tv.detach().cpu())})
    with torch.no_grad():
        delta = torch.tanh(delta_low) * float(args.epsilon)
        delta = F.interpolate(delta, size=base_pixel.shape[-2:], mode="bicubic", align_corners=False)
        pixel = (base_pixel + delta).clamp(0, 1)
        final_norm = (pixel - mean) / std
        final_pooled = encode_pooled(adapter, final_norm).detach().cpu().numpy().astype(np.float32)
    original_np = original_pooled.detach().cpu().numpy().astype(np.float32)
    aligned = image_from_pixel(pixel)
    return aligned, {
        "original_native_distance": cosine_distance(original_np, target_np),
        "aligned_native_distance": cosine_distance(final_pooled, target_np),
        "native_closure": cosine_distance(original_np, target_np) - cosine_distance(final_pooled, target_np),
        "history": history,
        "target_mode": args.target,
        "epsilon": args.epsilon,
        "grid_size": args.grid_size,
        "steps": args.steps,
    }


def save_preview(source: Image.Image, aligned: Image.Image, row, output_dir: Path, rank: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_resized = aligned.resize(source.size)
    canvas = Image.new("RGB", (source.size[0] * 2, source.size[1]), "white")
    canvas.paste(source.convert("RGB"), (0, 0))
    canvas.paste(aligned_resized.convert("RGB"), (source.size[0], 0))
    canvas.save(output_dir / f"{rank:03d}_qid{str(row.get('qid')).replace('/', '_')}_orig_aligned.jpg", quality=92)


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
    rows = []
    preview_count = 0
    for row in tqdm(selected_rows, desc="gradient native projection"):
        qid = str(row["qid"])
        path = resolve_image(row.get("img_name", ""))
        if path is None:
            continue
        with Image.open(path) as raw:
            image = resize_image(raw.convert("RGB"), args.max_image_side)
        labels = labels_for_sample(row)
        gt = ground_truth_index(row)
        prompt = build_prompt(row)
        aligned, projection = gradient_project(adapter, image, support, native_center, args)
        metrics = structure_metrics(image, aligned)
        evidence = adapter.forward_ce([image, aligned], prompt, labels)
        orig_logits = evidence[0].logits.astype(np.float32)
        align_logits = evidence[1].logits.astype(np.float32)
        orig_pred = int(np.argmax(orig_logits))
        align_pred = int(np.argmax(align_logits))
        orig_correct = bool(orig_pred == gt)
        align_correct = bool(align_pred == gt)
        if args.save_preview_dir and preview_count < args.save_preview_count:
            save_preview(image, aligned, row, args.save_preview_dir, preview_count)
            preview_count += 1
        rows.append({
            "qid": qid,
            "split": "train" if qid in train_qids else "test",
            "img_name": row.get("img_name"),
            "gt_index": int(gt),
            "labels": list(labels),
            "original_logits": orig_logits.tolist(),
            "aligned_logits": align_logits.tolist(),
            "original_prediction": orig_pred,
            "aligned_prediction": align_pred,
            "original_correct": orig_correct,
            "aligned_correct": align_correct,
            "original_entropy": entropy_from_logits(orig_logits),
            "aligned_entropy": entropy_from_logits(align_logits),
            "logit_delta": float(np.linalg.norm(align_logits - orig_logits)),
            "structure": metrics,
            "projection": projection,
        })
    def summarize(subset):
        return {
            "n": len(subset),
            "original_accuracy": acc([r["original_correct"] for r in subset]),
            "aligned_accuracy": acc([r["aligned_correct"] for r in subset]),
            "rescues": sum((not r["original_correct"]) and r["aligned_correct"] for r in subset),
            "harmful": sum(r["original_correct"] and (not r["aligned_correct"]) for r in subset),
            "changed_prediction": sum(r["original_prediction"] != r["aligned_prediction"] for r in subset),
            "mean_native_closure": float(np.mean([r["projection"]["native_closure"] for r in subset])) if subset else None,
            "mean_logit_delta": float(np.mean([r["logit_delta"] for r in subset])) if subset else None,
            "mean_psnr": float(np.mean([r["structure"]["psnr"] for r in subset if r["structure"].get("psnr") is not None])) if subset else None,
            "mean_edge_correlation": float(np.mean([r["structure"]["edge_correlation"] for r in subset])) if subset else None,
        }
    summary = {
        "version": VERSION,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "greedy_eval": str(args.greedy_eval.resolve()) if args.greedy_eval else None,
        "n": len(rows),
        "support_mode": args.support_mode,
        "support_n": len(support),
        "support_qids": [x["qid"] for x in support],
        "target": args.target,
        "params": {"steps": args.steps, "lr": args.lr, "grid_size": args.grid_size, "epsilon": args.epsilon, "l2_weight": args.l2_weight, "tv_weight": args.tv_weight},
        "overall": summarize(rows),
    }
    for split in ("train", "test"):
        summary[split] = summarize([r for r in rows if r["split"] == split])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    adapter.close()


if __name__ == "__main__":
    main()
