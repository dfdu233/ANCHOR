#!/usr/bin/env python3
"""Precompute AGLA prompt-matched views with the licensed LAVIS BLIP-ITM.

This is a clean-room implementation of the paper's disclosed augmentation:
BLIP-ITM GradCAM is thresholded by ``1 - similarity / 2`` and non-selected
pixels are zeroed.  Precomputation isolates BLIP's old Transformers runtime
from each target VLM and makes the same augmented view available to every
backbone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from anchor.medeval.hashing import sha256_file


LAVIS_ROOT = Path("/home/dbw/ANCHOR/third_party/LAVIS/lavis")
DEFAULT_CHECKPOINT = Path("/home/dbw/models/BLIP-ITM-large/model_large_retrieval_coco.pth")


def question_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r'([.!"()*#:;~])', " ", text)
    return " ".join(text.split()[:50])


def agla_binary_mask(attention: np.ndarray, similarity: float) -> tuple[np.ndarray, float]:
    """Return the paper's top-attention binary mask and retained ratio."""
    values = np.asarray(attention, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("attention must be a finite non-empty 2D array")
    ratio = float(np.clip(1.0 - float(similarity) / 2.0, 0.0, 1.0 - 1e-5))
    ordered = np.sort(values.reshape(-1))[::-1]
    threshold = ordered[min(int(values.size * ratio), values.size - 1)]
    return (values >= threshold).astype(np.uint8), ratio


def _load_blip(checkpoint: Path, device: torch.device):
    """Load only LAVIS BLIP modules, avoiding unrelated diffusion imports."""
    for name, path in (("lavis", LAVIS_ROOT), ("lavis.models", LAVIS_ROOT / "models")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    from lavis.models.base_model import BaseModel

    sys.modules["lavis.models"].BaseModel = BaseModel
    from lavis.common.registry import registry

    registry.register_path("library_root", str(LAVIS_ROOT))
    registry.register_path("cache_root", "/home/dbw/.cache/lavis")
    from lavis.models.blip_models.blip_image_text_matching import BlipITM, compute_gradcam
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(LAVIS_ROOT / "configs/models/blip_itm_large.yaml").model
    cfg.finetuned = str(checkpoint)
    model = BlipITM.from_config(cfg).to(device).eval()
    return model, compute_gradcam


def _attention_map(raw: Image.Image, gradcam: torch.Tensor) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    from skimage.transform import resize

    values = gradcam.detach().float().cpu().numpy()
    values -= values.min()
    maximum = float(values.max())
    if maximum > 0:
        values /= maximum
    values = resize(values, (384, 384), order=3, mode="constant")
    values = gaussian_filter(values, 0.02 * 384)
    values -= values.min()
    maximum = float(values.max())
    return values / maximum if maximum > 0 else np.zeros((384, 384), dtype=np.float32)


def _tensor(raw: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F

    resized = F.resize(raw.convert("RGB"), [384, 384], interpolation=InterpolationMode.BICUBIC)
    plain = F.to_tensor(resized)
    normalized = F.normalize(
        plain,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    return plain, normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = json.loads(args.manifest.read_text())
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    output_manifest = args.output_dir / "manifest.json"
    device = torch.device(args.device)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    model, compute_gradcam = _load_blip(args.checkpoint, device)
    derived: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        qid = str(row.get("qid", row.get("question_id", index)))
        source = Path(str(row["img_name"]))
        if not source.is_absolute():
            source = args.image_root / source
        question = question_text(row["question"])
        key = hashlib.sha256(
            f"{source.resolve()}\0{question}\0{checkpoint_sha256}".encode()
        ).hexdigest()
        target = image_dir / f"{key}.png"
        audit_path = image_dir / f"{key}.json"
        if target.is_file() and audit_path.is_file():
            audit = json.loads(audit_path.read_text())
        else:
            with Image.open(source) as opened:
                raw = opened.convert("RGB")
            plain, normalized = _tensor(raw)
            image_batch = normalized.unsqueeze(0).to(device)
            tokenized = model.tokenizer(
                question, padding="longest", truncation=True,
                max_length=model.max_txt_len, return_tensors="pt",
            ).to(device)
            with torch.enable_grad():
                gradcams, _ = compute_gradcam(
                    model, image_batch, [question], tokenized, block_num=6
                )
            with torch.no_grad():
                similarity = float(model(
                    {"image": image_batch, "text_input": [question]}, match_head="itc"
                )[0, 0].item())
            attention = _attention_map(raw, gradcams[0][1])
            mask, ratio = agla_binary_mask(attention, similarity)
            augmented = plain * torch.from_numpy(mask).unsqueeze(0)
            array = np.uint8(np.clip(augmented.permute(1, 2, 0).numpy(), 0, 1) * 255)
            Image.fromarray(array, mode="RGB").save(target)
            audit = {
                "qid": qid, "source": str(source.resolve()), "target": str(target.resolve()),
                "question": question, "itc_similarity": similarity,
                "retained_ratio_parameter": ratio, "retained_pixel_fraction": float(mask.mean()),
                "gradcam_block": 6, "checkpoint": str(args.checkpoint.resolve()),
            }
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        derived.append({**row, "agla_img_name": str(target.resolve())})
        audits.append(audit)
        print(f"[{index + 1}/{len(rows)}] {qid}", flush=True)
    output_manifest.write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "protocol": "agla-blip-itm-prompt-match-v1", "rows": len(rows),
        "source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": sha256_file(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha256,
        "output_manifest": str(output_manifest.resolve()), "output_manifest_sha256": sha256_file(output_manifest),
        "mean_retained_pixel_fraction": float(np.mean([x["retained_pixel_fraction"] for x in audits])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
