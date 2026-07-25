#!/usr/bin/env python3
"""Generate reusable per-style CE logits/features for Hulu-Med or LLaVA-Med."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from functools import lru_cache
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import (
    encode_array,
    load_successful_qids,
    repair_truncated_jsonl_tail,
)
from corrected_sgta.methods import feddg_frequency_interpolation, gamma_transform
from corrected_sgta.models_surface import load_adapter
from corrected_sgta.models import LLAVA_IMAGE_PREPROCESS_VERSION
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    build_prompt,
    choices_for_sample,
    file_sha256,
    ground_truth_index,
    labels_for_sample,
    normalize_text,
    protocol_fingerprint,
    resolve_image,
    task_kind,
    validate_dataset,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
CENTER_ROOT = Path("/root/autodl-tmp/multimodal_center_report/centers")
CENTER_FILES = {
    "overall": "pubmedvision_overall.npy",
    "xray": "pubmedvision_xray.npy",
    "ct": "pubmedvision_ct.npy",
    "mri": "pubmedvision_mri.npy",
}


@lru_cache(maxsize=None)
def load_center(path: str) -> np.ndarray:
    return np.load(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-labels", action="store_true")
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    parser.add_argument("--no-feddg", action="store_true")
    parser.add_argument("--gammas", type=float, nargs="*", default=(0.8, 1.2))
    parser.add_argument(
        "--center-policy",
        choices=("matched", "inferred", "all"),
        default="matched",
        help=(
            "matched/inferred use exactly one modality-matched center; all is retained "
            "only as an explicit ablation"
        ),
    )
    parser.add_argument("--feddg-l", type=float, default=0.003)
    parser.add_argument("--feddg-l-values", type=float, nargs="*", default=None)
    parser.add_argument("--feddg-source-ratio", type=float, default=0.0)
    parser.add_argument("--feddg-source-ratios", type=float, nargs="*", default=None)
    parser.add_argument("--min-style-psnr", type=float, default=20.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument(
        "--keep-unsafe-styles",
        action="store_true",
        help="explicit ablation: forward styles that fail the pre-forward structure gate",
    )
    return parser.parse_args()


def decoded_label_index(
    text: str, labels: list[str] | tuple[str, ...], sample: dict | None = None
) -> int | None:
    normalized = " ".join(str(text).strip().lower().split())
    matches = []
    for index, label in enumerate(labels):
        candidate = " ".join(str(label).strip().lower().split())
        if re.match(rf"^{re.escape(candidate)}(?:$|[^a-z0-9])", normalized):
            matches.append((len(candidate), index))
    if matches:
        return max(matches)[1]
    if sample is None or task_kind(sample) != "multichoice":
        return None
    try:
        choices = choices_for_sample(sample)
    except ProtocolError:
        return None
    answer = normalize_text(text)
    option_matches: list[int] = []
    for index, option_text in enumerate(choices.texts):
        option = normalize_text(option_text)
        if not option:
            continue
        if re.search(rf"(?:^|\s){re.escape(option)}(?:$|\s)", answer):
            option_matches.append(index)
            continue
        tokens = [token for token in option.split() if len(token) >= 3]
        if tokens and all(
            re.search(rf"(?:^|\s){re.escape(token)}(?:$|\s)", answer)
            for token in tokens
        ):
            option_matches.append(index)
    return option_matches[0] if len(option_matches) == 1 else None


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= max_side:
        return image
    ratio = max_side / max(image.size)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def center_name(sample: dict, dataset: Path) -> str:
    modality = str(sample.get("modality", "")).lower()
    image_name = str(sample.get("img_name", "")).lower()
    if "ct" in modality:
        return "pubmedvision_ct.npy"
    if "mri" in modality or "mr " in modality:
        return "pubmedvision_mri.npy"
    if (
        "x-ray" in modality
        or "xray" in modality
        or image_name.startswith("p")
        or "cxr" in str(dataset).lower()
    ):
        return "pubmedvision_xray.npy"
    return "pubmedvision_overall.npy"


def center_candidates(sample: dict, dataset: Path, policy: str) -> list[tuple[str, str]]:
    inferred = center_name(sample, dataset)
    inferred_key = next(
        (name for name, filename in CENTER_FILES.items() if filename == inferred),
        "inferred",
    )
    if policy in {"matched", "inferred"}:
        return [(inferred_key, inferred)]
    if policy != "all":
        raise ValueError(f"unknown center policy: {policy}")
    ordered = [(inferred_key, inferred)]
    for name, filename in CENTER_FILES.items():
        if filename != inferred:
            ordered.append((name, filename))
    return ordered


def _center_distance(
    image: Image.Image, target_amplitude: np.ndarray, low_frequency_ratio: float
) -> dict[str, float]:
    """Scale-robust source-to-center distance in the transferred FFT window."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64).transpose(2, 0, 1)
    target = np.asarray(target_amplitude, dtype=np.float64)
    if target.ndim == 2:
        target = np.repeat(target[None, ...], 3, axis=0)
    if target.shape[0] == 1:
        target = np.repeat(target, 3, axis=0)
    if target.shape[-2:] != rgb.shape[-2:]:
        resized = torch.from_numpy(target).unsqueeze(0).float()
        target = torch.nn.functional.interpolate(
            resized, size=rgb.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(0).numpy()
    source = np.fft.fftshift(np.abs(np.fft.fft2(rgb, axes=(-2, -1))), axes=(-2, -1))
    target = np.fft.fftshift(target, axes=(-2, -1))
    radius = int(math.floor(min(rgb.shape[-2:]) * low_frequency_ratio))
    center_h, center_w = rgb.shape[-2] // 2, rgb.shape[-1] // 2
    window = (
        slice(None),
        slice(center_h - radius, center_h + radius + 1),
        slice(center_w - radius, center_w + radius + 1),
    )
    left = np.log1p(np.clip(source[window], 0.0, None)).ravel()
    right = np.log1p(np.clip(target[window], 0.0, None)).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(left @ right / denominator) if denominator > 1e-12 else 1.0
    relative_rmse = float(
        np.sqrt(np.mean((left - right) ** 2))
        / max(float(np.sqrt(np.mean(left**2))), 1e-12)
    )
    return {
        "log_amplitude_cosine_distance": float(np.clip(1.0 - cosine, 0.0, 2.0)),
        "log_amplitude_relative_rmse": relative_rmse,
    }


def _structure_metrics(source: Image.Image, transformed: Image.Image) -> dict[str, float | None]:
    """Dependency-free structure checks for cached style auditing."""

    left = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    right = np.asarray(transformed.convert("L"), dtype=np.float64) / 255.0
    mse = float(np.mean((left - right) ** 2))
    source_edge = np.hypot(*np.gradient(left))
    target_edge = np.hypot(*np.gradient(right))
    source_centered = source_edge.ravel() - source_edge.mean()
    target_centered = target_edge.ravel() - target_edge.mean()
    denominator = float(
        np.linalg.norm(source_centered) * np.linalg.norm(target_centered)
    )
    edge_correlation = (
        float(
            np.clip(source_centered @ target_centered / denominator, -1.0, 1.0)
        )
        if denominator > 1e-12
        else 1.0
    )
    return {
        "pixel_mse": mse,
        "psnr": None if mse <= 1e-12 else float(-10.0 * math.log10(mse)),
        "edge_correlation": edge_correlation,
    }


def make_styles_with_metadata(
    image: Image.Image, sample: dict, args: argparse.Namespace
) -> tuple[list[str], list[Image.Image], list[dict]]:
    names = ["original"]
    images = [image]
    metadata = [
        {
            "family": "original",
            "domain_id": "original",
            "parameters": {},
            "structure": {
                "pixel_mse": 0.0,
                "psnr": None,
                "edge_correlation": 1.0,
            },
        }
    ]
    if not args.no_feddg:
        l_values = list(getattr(args, "feddg_l_values", None) or [args.feddg_l])
        source_ratios = list(
            getattr(args, "feddg_source_ratios", None) or [args.feddg_source_ratio]
        )
        center_policy = getattr(args, "center_policy", "inferred")
        for center_key, filename in center_candidates(sample, args.dataset, center_policy):
            center_path = CENTER_ROOT / filename
            if not center_path.is_file():
                raise FileNotFoundError(
                    f"missing external amplitude center: {center_path}"
                )
            center = load_center(str(center_path))
            for low_frequency_ratio in l_values:
                for source_ratio in source_ratios:
                    transformed = feddg_frequency_interpolation(
                        image,
                        center,
                        low_frequency_ratio=low_frequency_ratio,
                        source_ratio=source_ratio,
                    )
                    images.append(transformed)
                    if (
                        center_policy == "inferred"
                        and len(l_values) == 1
                        and len(source_ratios) == 1
                    ):
                        names.append("feddg_center")
                    else:
                        names.append(
                            "feddg_"
                            f"{center_key}_l{low_frequency_ratio:g}_sr{source_ratio:g}"
                        )
                    metadata.append(
                        {
                            "family": "feddg",
                            "domain_id": center_key,
                            "center_file": filename,
                            "parameters": {
                                "low_frequency_ratio": float(low_frequency_ratio),
                                "source_ratio": float(source_ratio),
                            },
                            "structure": _structure_metrics(image, transformed),
                            "center_distance": _center_distance(
                                image, center, float(low_frequency_ratio)
                            ),
                        }
                    )
    for gamma in args.gammas:
        names.append(f"gamma_{gamma:g}")
        transformed = gamma_transform(image, gamma)
        images.append(transformed)
        metadata.append(
            {
                "family": "gamma",
                "domain_id": "photometric",
                "parameters": {"gamma": float(gamma)},
                "structure": _structure_metrics(image, transformed),
            }
        )
    return names, images, metadata


def make_styles(image: Image.Image, sample: dict, args: argparse.Namespace):
    """Backward-compatible style API used by existing callers."""

    names, images, _ = make_styles_with_metadata(image, sample, args)
    return names, images


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    config = {
        "model": args.model,
        "seed": args.seed,
        "dataset_sha256": file_sha256(args.dataset),
        "max_image_side": args.max_image_side,
        "decode_labels": args.decode_labels,
        "decode_max_new_tokens": args.decode_max_new_tokens,
        "gammas": list(args.gammas),
        "feddg": not args.no_feddg,
        "center_policy": args.center_policy,
        "feddg_l_values": list(args.feddg_l_values or [args.feddg_l]),
        "feddg_source_ratios": list(args.feddg_source_ratios or [args.feddg_source_ratio]),
        "structure_gate": {
            "min_psnr": args.min_style_psnr,
            "min_edge_correlation": args.min_edge_correlation,
            "keep_unsafe_styles": args.keep_unsafe_styles,
        },
        "label_policy": "surface_max_single_token_constrained",
        "label_sequence_score": (
            "full-vocabulary NLL for each accepted complete one-token surface form; "
            "minimum NLL within semantic class"
        ),
        "feature": "last_multimodal_prompt_hidden_state",
    }
    if args.model == "llava":
        config["image_preprocessing_version"] = LLAVA_IMAGE_PREPROCESS_VERSION
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"metadata mismatch; choose a new output path: {metadata_path}"
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))

    repair = repair_truncated_jsonl_tail(args.output)
    if repair["action"] != "none":
        print(f"cache tail repair: {repair}", flush=True)
    saved = load_successful_qids(args.output, fingerprint)
    target_rows = []
    for sample in rows:
        try:
            if task_kind(sample) == "open":
                continue
            labels_for_sample(sample)
            ground_truth_index(sample)
            if resolve_image(sample.get("img_name", "")) is None:
                continue
            target_rows.append(sample)
        except ProtocolError:
            continue
    target_rows.sort(
        key=lambda sample: hashlib.sha256(
            f"{args.seed}:{sample['qid']}".encode()
        ).hexdigest()
    )
    if args.max_samples:
        target_rows = target_rows[: args.max_samples]
    eligible = [sample for sample in target_rows if str(sample["qid"]) not in saved]
    print(
        f"protocol={PROTOCOL_VERSION} fingerprint={fingerprint[:12]} "
        f"eligible={len(eligible)} cached={len(saved)} validation={validation}",
        flush=True,
    )
    if not eligible:
        return

    print(f"Loading {args.model}...", flush=True)
    adapter = load_adapter(args.model)
    print(f"Loaded {adapter.name}", flush=True)
    started = time.time()
    errors = 0
    with args.output.open("a") as output:
        for sample in tqdm(eligible, desc=f"CE {args.model}"):
            try:
                image_path = resolve_image(sample.get("img_name", ""))
                assert image_path is not None
                with Image.open(image_path) as source:
                    image = resize_image(source, args.max_image_side)
                labels = labels_for_sample(sample)
                style_names, style_images, style_metadata = make_styles_with_metadata(
                    image, sample, args
                )
                rejected_style_metadata = []
                if not args.keep_unsafe_styles:
                    kept = [0]
                    for style_index, metadata_item in enumerate(style_metadata[1:], start=1):
                        structure = metadata_item.get("structure") or {}
                        psnr = structure.get("psnr")
                        edge = structure.get("edge_correlation")
                        safe = (
                            psnr is not None
                            and float(psnr) >= args.min_style_psnr
                            and edge is not None
                            and float(edge) >= args.min_edge_correlation
                        )
                        if safe:
                            kept.append(style_index)
                        else:
                            rejected_style_metadata.append(
                                {**metadata_item, "style_name": style_names[style_index]}
                            )
                    style_names = [style_names[index] for index in kept]
                    style_images = [style_images[index] for index in kept]
                    style_metadata = [style_metadata[index] for index in kept]
                prompt = build_prompt(sample)
                evidence = adapter.forward_ce(style_images, prompt, labels)
                decoded_text = (
                    adapter.decode_ce(
                        style_images, prompt, max_new_tokens=args.decode_max_new_tokens
                    )
                    if args.decode_labels
                    else None
                )
                decoded_prediction = (
                    [decoded_label_index(text, labels, sample) for text in decoded_text]
                    if decoded_text is not None
                    else None
                )
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "status": "ok",
                    "qid": sample["qid"],
                    "img_name": sample.get("img_name", ""),
                    "question_type": task_kind(sample),
                    "labels": list(labels),
                    "gt_index": ground_truth_index(sample),
                    "style_names": style_names,
                    "style_metadata": style_metadata,
                    "rejected_style_metadata": rejected_style_metadata,
                    "style_logits": [value.logits.tolist() for value in evidence],
                    "style_decoded_text": decoded_text,
                    "style_decoded_prediction": decoded_prediction,
                    "decoded_prediction": (
                        None if decoded_prediction is None else decoded_prediction[0]
                    ),
                    "style_sequence_nll": [
                        None if value.sequence_nll is None else value.sequence_nll.tolist()
                        for value in evidence
                    ],
                    "style_features": encode_array(
                        np.stack([value.features for value in evidence])
                    ),
                }
            except Exception as exc:
                errors += 1
                traceback.print_exc()
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
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

    elapsed = time.time() - started
    print(
        f"Finished {len(eligible)} rows in {elapsed / 60:.2f} min; errors={errors}",
        flush=True,
    )
    adapter.close()


if __name__ == "__main__":
    main()
