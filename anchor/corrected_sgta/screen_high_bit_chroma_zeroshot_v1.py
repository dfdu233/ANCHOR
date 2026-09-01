#!/usr/bin/env python3
"""CPU fatal screen for high-bit DICOM residual coding in frozen RGB towers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from anchor.corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    canonical_polarity,
    read_dicom_pixels,
)
from anchor.corrected_sgta.screen_external_visual_increment_v1 import load_claims, sha256_file


FINDINGS = ("cardiomegaly", "pleural_effusion", "pleural_thickening", "pulmonary_fibrosis")
DISPLAY = {
    "cardiomegaly": "cardiomegaly",
    "pleural_effusion": "pleural effusion",
    "pleural_thickening": "pleural thickening",
    "pulmonary_fibrosis": "pulmonary fibrosis",
}
VIEWS = (
    "base", "true_residual", "spatial_shuffle", "cross_image", "random_sign",
    "gray_residual", "multi_contrast_rgb",
)
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
CHROMA = np.asarray([1.0, -0.2126 / 0.7152, 0.0], dtype=np.float32)


def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def float_window(path: Path, side: int) -> np.ndarray:
    pixels = read_dicom_pixels(path)
    finite = pixels.modality[pixels.valid]
    lo, hi = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    x = np.clip((pixels.modality - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    x = canonical_polarity(x, pixels.photometric)
    # Resize in float before 8-bit quantization: r is exactly the information
    # lost at the ordinary uint8 RGB interface, not a resize artifact.
    import cv2

    return cv2.resize(x.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)


def to_uint8(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")


def construct_views(image_id: str, x: np.ndarray, cross_r: np.ndarray, gain: float, seed: int):
    q = np.rint(x * 255.0) / 255.0
    r = x - q
    rng = np.random.default_rng(stable_seed(image_id, seed))
    shuffled = r.reshape(-1).copy()
    rng.shuffle(shuffled)
    shuffled = shuffled.reshape(r.shape)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=r.shape)
    cross_r = cross_r * (np.linalg.norm(r) / max(np.linalg.norm(cross_r), 1e-12))

    def chroma(residual: np.ndarray) -> Image.Image:
        rgb = q[..., None] + gain * residual[..., None] * CHROMA
        return to_uint8(rgb)

    base = np.repeat(q[..., None], 3, axis=-1)
    multi_contrast = np.stack(
        [np.clip((x - 0.5) * scale + 0.5, 0, 1) for scale in (0.75, 1.0, 1.25)],
        axis=-1,
    )
    return {
        "base": to_uint8(base),
        "true_residual": chroma(r),
        "spatial_shuffle": chroma(shuffled),
        "cross_image": chroma(cross_r),
        "random_sign": chroma(np.abs(r) * signs),
        "gray_residual": to_uint8(np.repeat((q + gain * r)[..., None], 3, axis=-1)),
        "multi_contrast_rgb": to_uint8(multi_contrast),
    }


class BiomedTower:
    name = "biomedclip_vit_b16"

    def __init__(self, root: Path):
        import open_clip
        from open_clip.factory import _MODEL_CONFIGS, create_model_and_transforms, get_tokenizer

        config = json.loads((root / "open_clip_config.json").read_text())
        config["model_cfg"]["text_cfg"]["hf_model_name"] = str(root / "text_encoder")
        config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(root)
        model_name = "biomedclip_high_bit_chroma_v1"
        _MODEL_CONFIGS[model_name] = config["model_cfg"]
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=str(root / "open_clip_pytorch_model.bin"),
            **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
        )
        self.model.eval().to("cpu")
        self.tokenizer = get_tokenizer(model_name)
        self.provenance = {"root": str(root.resolve()), "weights_sha256": sha256_file(root / "open_clip_pytorch_model.bin")}

    def text(self, prompts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(prompts, context_length=256)
        with torch.inference_mode():
            z = self.model.encode_text(tokens, normalize=True)
        return z.cpu().numpy()

    def image(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(image) for image in images])
        with torch.inference_mode():
            z = self.model.encode_image(batch, normalize=True)
        return z.cpu().numpy()


class ClipTower:
    name = "clip_vit_l14_336"

    def __init__(self, root: Path):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(root, local_files_only=True).eval().to("cpu")
        self.processor = CLIPProcessor.from_pretrained(root, local_files_only=True)
        self.provenance = {"root": str(root.resolve()), "config_sha256": sha256_file(root / "config.json")}

    def text(self, prompts: list[str]) -> np.ndarray:
        batch = self.processor(text=prompts, return_tensors="pt", padding=True)
        with torch.inference_mode():
            z = self.model.get_text_features(**batch)
            z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy()

    def image(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            z = self.model.get_image_features(**batch)
            z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy()


def text_directions(tower) -> np.ndarray:
    prompts = []
    for finding in FINDINGS:
        prompts.extend(
            [
                f"a frontal chest radiograph showing {DISPLAY[finding]}",
                f"a frontal chest radiograph without {DISPLAY[finding]}",
            ]
        )
    encoded = tower.text(prompts).reshape(len(FINDINGS), 2, -1)
    return encoded[:, 0] - encoded[:, 1]


def macro_auroc(rows, scores: dict[str, np.ndarray]) -> dict[str, float]:
    output = {}
    for view, matrix in scores.items():
        values = []
        for finding_index, finding in enumerate(FINDINGS):
            selected = [row for row in rows if row["finding"] == finding]
            if not selected:
                continue
            y = np.asarray([row["label"] for row in selected])
            p = np.asarray([matrix[row["image_index"], finding_index] for row in selected])
            values.append(roc_auc_score(y, p))
        output[view] = float(np.mean(values))
    return output


def balanced_cap(rows: list[dict[str, Any]], per_finding: int, seed: int) -> list[dict[str, Any]]:
    """Freeze an equal-class pilot without inspecting any model score."""
    if per_finding % 2:
        raise ValueError("per-finding cap must be even")
    output = []
    for finding in FINDINGS:
        for label in (0, 1):
            candidates = [row for row in rows if row["finding"] == finding and row["label"] == label]
            rng = np.random.default_rng(stable_seed(f"{finding}:{label}", seed))
            order = rng.permutation(len(candidates))[: per_finding // 2]
            output.extend(candidates[i] for i in order)
    return output


def bootstrap(rows, scores, draws: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    comparisons = {view: [] for view in VIEWS if view != "true_residual"}
    base_delta = []
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        sampled_rows = [dict(row) for image_id in sampled for row in groups[image_id]]
        try:
            point = macro_auroc(sampled_rows, scores)
        except ValueError:
            continue
        base_delta.append(point["true_residual"] - point["base"])
        for view in comparisons:
            comparisons[view].append(point["true_residual"] - point[view])

    def summary(values):
        x = np.asarray(values)
        return {"mean": float(x.mean()), "ci95": [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]}

    return {"true_minus_base": summary(base_delta), "true_minus_controls": {k: summary(v) for k, v in comparisons.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--biomedclip-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=336)
    parser.add_argument("--gain", type=float, default=96.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--per-finding-dev", type=int, default=80)
    parser.add_argument("--per-finding-confirmation", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' to preserve the baseline GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = balanced_cap(
        [row for row in load_claims(args.dev, "dev", "label") if row["finding"] in FINDINGS],
        args.per_finding_dev,
        args.seed,
    )
    confirmation = balanced_cap(
        [row for row in load_claims(args.confirmation, "confirmation", "label") if row["finding"] in FINDINGS],
        args.per_finding_confirmation,
        args.seed + 1,
    )
    image_ids = sorted({row["image_id"] for row in dev + confirmation})
    index = {image_id: i for i, image_id in enumerate(image_ids)}
    for row in dev + confirmation:
        row["image_index"] = index[row["image_id"]]
    # A cyclic image-id permutation is fixed before reading pixels and forms
    # the cross-image placebo without using labels.
    shifted = image_ids[1:] + image_ids[:1]
    raw = {image_id: float_window(args.image_root / f"{image_id}.dicom", args.side) for image_id in image_ids}
    residual = {image_id: raw[image_id] - np.rint(raw[image_id] * 255.0) / 255.0 for image_id in image_ids}

    towers = [BiomedTower(args.biomedclip_root), ClipTower(args.clip_root)]
    analyses = {}
    passes = []
    for tower in towers:
        directions = text_directions(tower)
        score_lists = {view: [] for view in VIEWS}
        for start in range(0, len(image_ids), args.batch_size):
            batch_ids = image_ids[start : start + args.batch_size]
            view_images = {view: [] for view in VIEWS}
            for image_id in batch_ids:
                other = shifted[index[image_id]]
                views = construct_views(image_id, raw[image_id], residual[other], args.gain, args.seed)
                for view in VIEWS:
                    view_images[view].append(views[view])
            for view in VIEWS:
                embedding = tower.image(view_images[view])
                score_lists[view].append(embedding @ directions.T)
        scores = {view: np.concatenate(parts, axis=0) for view, parts in score_lists.items()}
        dev_auc = macro_auroc(dev, scores)
        test_auc = macro_auroc(confirmation, scores)
        boot = bootstrap(confirmation, scores, args.bootstrap_draws, args.seed)
        gate = (
            test_auc["true_residual"] - test_auc["base"] >= 0.02
            and boot["true_minus_base"]["ci95"][0] > 0
            and all(
                boot["true_minus_controls"][view]["ci95"][0] > 0
                for view in ("spatial_shuffle", "cross_image", "random_sign")
            )
        )
        passes.append(gate)
        analyses[tower.name] = {
            "provenance": tower.provenance,
            "dev_macro_auroc": dev_auc,
            "confirmation_macro_auroc": test_auc,
            "paired_image_bootstrap": boot,
            "passes": gate,
        }
        del tower.model

    result = {
        "status": "complete",
        "decision": "PASS_L0" if all(passes) else "NO_GO_L0",
        "command": shlex.join(sys.argv),
        "n_images": len(image_ids),
        "n_dev_claims": len(dev),
        "n_confirmation_claims": len(confirmation),
        "findings": list(FINDINGS),
        "views": list(VIEWS),
        "encoding": {
            "decomposition": "float-resize x = round(255x)/255 + residual",
            "luma": LUMA.tolist(),
            "chroma_direction": CHROMA.tolist(),
            "gain": args.gain,
            "output": "clipped and rounded uint8 RGB",
        },
        "input_sha256": {"dev": sha256_file(args.dev), "confirmation": sha256_file(args.confirmation)},
        "preregistered_gate": {
            "models": "both frozen RGB towers must pass",
            "true_minus_base_macro_auroc": ">=0.02 and image-bootstrap CI lower >0",
            "true_minus_placebos": "CI lower >0 vs spatial shuffle, cross-image residual, and equal-amplitude random-sign chroma",
            "failure_action": "close high-bit chroma residual candidate without a VLM generation run",
        },
        "analyses": analyses,
        "boundary": "A PASS is only evidence that frozen RGB towers read clinical low-bit structure. It is not yet a hallucination mitigation result or an ICLR contribution.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
