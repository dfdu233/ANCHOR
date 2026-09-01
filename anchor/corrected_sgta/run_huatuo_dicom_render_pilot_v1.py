#!/usr/bin/env python3
"""Paired reader-grounded DICOM-render pilot for HuatuoGPT-Vision.

The experimental unit is one atomic VinDr finding claim.  Every claim is
scored on several clinically plausible renderings of the *same* DICOM.  The
polarity-toggle rendering is deliberately marked secondary and can never be
used as evidence for the primary clinical-render gate.

Results are written as one atomic JSON shard per claim.  A crash therefore
loses at most one claim, and ``--resume`` never recomputes a complete shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    atomic_json,
    import_huatuo,
    label_ids,
    load_jsonl,
    prepared_embeddings,
    prompt_for,
    resolve_image,
    sha256_file,
)


VERSION = "huatuo-dicom-render-pilot-v1"
DEFAULT_FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "pleural_effusion",
    "pulmonary_fibrosis",
)
BASELINE_VIEW = "baseline_percentile"
SECONDARY_NAMES = frozenset({"polarity_toggle", "content_loss_downsample"})


@dataclass(frozen=True)
class DicomPixels:
    modality: np.ndarray
    valid: np.ndarray
    photometric: str
    window_center: float
    window_width: float
    window_source: str
    metadata: dict[str, Any]


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_artifact_fingerprint(model_dir: Path) -> dict[str, Any]:
    """Cheap, stable model identity without re-hashing ~15 GB every resume."""

    metadata_names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "vit/clip_vit_large_patch14_336/config.json",
        "vit/clip_vit_large_patch14_336/preprocessor_config.json",
    )
    metadata = {
        name: sha256_file(model_dir / name)
        for name in metadata_names
        if (model_dir / name).is_file()
    }
    weights = [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in sorted(model_dir.glob("*.safetensors"))
    ]
    payload = {"metadata_sha256": metadata, "weight_inventory": weights}
    return {**payload, "fingerprint": canonical_json_sha256(payload)}


def safe_record_key(row: Mapping[str, Any]) -> str:
    readable = f"{row['finding']}:{row['image_id']}"
    suffix = hashlib.sha256(readable.encode()).hexdigest()[:12]
    return f"{row['finding']}__{row['image_id']}__{suffix}"


def first_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (str, bytes)):
            return float(value)
        values = np.asarray(value).reshape(-1)
        return None if not values.size else float(values[0])
    except (TypeError, ValueError):
        try:
            return float(value)  # pydicom DSfloat
        except (TypeError, ValueError):
            return None


def read_dicom_pixels(path: Path) -> DicomPixels:
    import pydicom

    dataset = pydicom.dcmread(str(path))
    stored = np.asarray(dataset.pixel_array)
    if stored.ndim != 2:
        raise ValueError(f"expected one 2-D radiograph, got {stored.shape}: {path}")
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    modality = stored.astype(np.float32) * slope + intercept
    valid = np.isfinite(modality)
    padding = first_float(getattr(dataset, "PixelPaddingValue", None))
    padding_limit = first_float(getattr(dataset, "PixelPaddingRangeLimit", None))
    if padding is not None:
        if padding_limit is None:
            valid &= stored != padding
        else:
            lo, hi = sorted((padding, padding_limit))
            valid &= ~((stored >= lo) & (stored <= hi))
    finite = modality[valid]
    if finite.size == 0:
        raise ValueError(f"DICOM contains no finite non-padding pixels: {path}")
    p005, p995 = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    if p995 <= p005:
        p005, p995 = float(finite.min()), float(finite.max())
    if p995 <= p005:
        raise ValueError(f"DICOM has constant pixels: {path}")
    center = first_float(getattr(dataset, "WindowCenter", None))
    width = first_float(getattr(dataset, "WindowWidth", None))
    if center is None or width is None or not math.isfinite(width) or width <= 1.0:
        center, width = (p005 + p995) / 2.0, p995 - p005
        source = "percentile_fallback"
    else:
        source = "dicom_window_center_width"
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(f"unsupported PhotometricInterpretation={photometric!r}")
    metadata = {
        "rows": int(stored.shape[0]),
        "columns": int(stored.shape[1]),
        "photometric_interpretation": photometric,
        "bits_stored": int(getattr(dataset, "BitsStored", stored.dtype.itemsize * 8)),
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "pixel_padding_value": padding,
        "pixel_padding_range_limit": padding_limit,
        "percentile_0_5": p005,
        "percentile_99_5": p995,
        "window_center": float(center),
        "window_width": float(width),
        "window_source": source,
        "voi_lut_function": str(getattr(dataset, "VOILUTFunction", "LINEAR")),
    }
    return DicomPixels(
        modality=modality,
        valid=valid,
        photometric=photometric,
        window_center=float(center),
        window_width=float(width),
        window_source=source,
        metadata=metadata,
    )


def percentile_render(pixels: DicomPixels) -> np.ndarray:
    finite = pixels.modality[pixels.valid]
    lower, upper = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    output = np.clip((pixels.modality - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    return canonical_polarity(output, pixels.photometric)


def linear_voi(values: np.ndarray, center: float, width: float) -> np.ndarray:
    """DICOM C.11.2 LINEAR windowing, normalized to [0,1]."""

    if width <= 1.0:
        raise ValueError("DICOM LINEAR WindowWidth must be greater than one")
    output = (values - (center - 0.5)) / (width - 1.0) + 0.5
    return np.clip(output, 0.0, 1.0)


def sigmoid_voi(values: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0.0:
        raise ValueError("DICOM SIGMOID WindowWidth must be positive")
    argument = np.clip(-4.0 * (values - center) / width, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(argument))


def canonical_polarity(values: np.ndarray, photometric: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if photometric == "MONOCHROME1":
        result = 1.0 - result
    return np.clip(result, 0.0, 1.0)


def gradient_magnitude(values: np.ndarray) -> np.ndarray:
    y = np.zeros_like(values, dtype=np.float32)
    x = np.zeros_like(values, dtype=np.float32)
    y[:-1] = np.abs(np.diff(values, axis=0))
    x[:, :-1] = np.abs(np.diff(values, axis=1))
    return np.hypot(x, y)


def edge_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = gradient_magnitude(reference).reshape(-1).astype(np.float64)
    right = gradient_magnitude(candidate).reshape(-1).astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 1.0 if denominator <= 1e-12 and np.allclose(left, right) else (
        0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)
    )


def normalized_box(box: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width, int(math.floor(float(box["x_min"])))))
    y0 = max(0, min(height, int(math.floor(float(box["y_min"])))))
    x1 = max(0, min(width, int(math.ceil(float(box["x_max"])))))
    y1 = max(0, min(height, int(math.ceil(float(box["y_max"])))))
    return x0, y0, x1, y1


def roi_mask(
    shape: tuple[int, int], boxes: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, str]:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        x0, y0, x1, y1 = normalized_box(box, width, height)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    if mask.any():
        return mask, "claim_reader_boxes"
    # A fixed central thorax proxy avoids selecting a region from model output.
    x0, x1 = int(round(0.15 * width)), int(round(0.85 * width))
    y0, y1 = int(round(0.10 * height)), int(round(0.90 * height))
    mask[y0:y1, x0:x1] = True
    return mask, "fixed_central_thorax_proxy"


def saturation_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    if selected.size == 0:
        raise ValueError("empty audit ROI")
    return float(np.mean((selected <= 1.0 / 255.0) | (selected >= 254.0 / 255.0)))


def conservative_blank_crop(
    baseline: np.ndarray, all_boxes: Sequence[Mapping[str, Any]]
) -> tuple[tuple[int, int, int, int], dict[str, float | bool]]:
    """Find only near-uniform edge strips, capped at five percent per edge."""

    height, width = baseline.shape
    max_y, max_x = int(0.05 * height), int(0.05 * width)
    border = np.concatenate((baseline[0], baseline[-1], baseline[:, 0], baseline[:, -1]))
    background = float(np.median(border))

    def blank_line(line: np.ndarray) -> bool:
        return bool(
            np.std(line) < 0.015
            and abs(float(np.mean(line)) - background) < 0.04
            and float(np.mean(np.abs(np.diff(line)))) < 0.008
        )

    top = next((index for index in range(max_y + 1) if not blank_line(baseline[index])), max_y)
    bottom_trim = next(
        (index for index in range(max_y + 1) if not blank_line(baseline[height - 1 - index])),
        max_y,
    )
    left = next((index for index in range(max_x + 1) if not blank_line(baseline[:, index])), max_x)
    right_trim = next(
        (index for index in range(max_x + 1) if not blank_line(baseline[:, width - 1 - index])),
        max_x,
    )
    bottom, right = height - bottom_trim, width - right_trim

    # Every annotation on the image, not just the target finding, is protected.
    for box in all_boxes:
        x0, y0, x1, y1 = normalized_box(box, width, height)
        top = min(top, max(0, y0 - 2))
        left = min(left, max(0, x0 - 2))
        bottom = max(bottom, min(height, y1 + 2))
        right = max(right, min(width, x1 + 2))
    top, left = max(0, top), max(0, left)
    bottom, right = min(height, bottom), min(width, right)
    if bottom <= top or right <= left:
        top, left, bottom, right = 0, 0, height, width

    edge = gradient_magnitude(baseline)
    retained_edge = float(edge[top:bottom, left:right].sum() / max(float(edge.sum()), 1e-12))
    body = np.abs(baseline - background) > 0.03
    body_total = int(body.sum())
    retained_body = (
        1.0
        if body_total == 0
        else float(body[top:bottom, left:right].sum() / body_total)
    )
    bbox_retained = all(
        normalized_box(box, width, height)[0] >= left
        and normalized_box(box, width, height)[1] >= top
        and normalized_box(box, width, height)[2] <= right
        and normalized_box(box, width, height)[3] <= bottom
        for box in all_boxes
    )
    return (left, top, right, bottom), {
        "bbox_retention": bool(bbox_retained),
        "edge_energy_retained": retained_edge,
        "body_mask_retained": retained_body,
        "cropped_area_fraction": 1.0 - (right - left) * (bottom - top) / (width * height),
    }


def resize_crop(values: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bounds
    image = Image.fromarray(np.round(values[top:bottom, left:right] * 65535).astype(np.uint16))
    resized = image.resize((values.shape[1], values.shape[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 65535.0


def content_loss_downsample(values: np.ndarray) -> np.ndarray:
    """Deliberately remove fine detail as a secondary positive control."""

    height, width = values.shape
    image = Image.fromarray(np.round(values * 65535).astype(np.uint16))
    small = image.resize((32, 32), resample=Image.Resampling.BILINEAR)
    restored = small.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.float32) / 65535.0


def pixel_sha256(values: np.ndarray) -> str:
    pixels = np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype=np.int64).tobytes())
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def to_pil(values: np.ndarray) -> Image.Image:
    pixels = np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def transform_boxes_for_crop(
    boxes: Sequence[Mapping[str, Any]],
    bounds: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[dict[str, float]]:
    left, top, right, bottom = bounds
    height, width = shape
    sx, sy = width / (right - left), height / (bottom - top)
    output = []
    for box in boxes:
        x0, y0, x1, y1 = normalized_box(box, width, height)
        output.append(
            {
                "x_min": (x0 - left) * sx,
                "y_min": (y0 - top) * sy,
                "x_max": (x1 - left) * sx,
                "y_max": (y1 - top) * sy,
            }
        )
    return output


def build_render_views(
    pixels: DicomPixels,
    claim_boxes: Sequence[Mapping[str, Any]],
    all_boxes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    # ``claim_boxes`` is accepted only for API provenance.  It is deliberately
    # not used for validity selection: reader outcomes must not determine which
    # renderings are admitted.  Boxes are used solely below as part of the
    # all-lesion crop-retention audit.
    del claim_boxes
    baseline = percentile_render(pixels)
    center, width = pixels.window_center, pixels.window_width
    voi_function = str(pixels.metadata.get("voi_lut_function", "LINEAR")).upper()
    sigmoid_is_native = voi_function == "SIGMOID"
    arrays: list[tuple[str, np.ndarray, str, dict[str, Any]]] = [
        (BASELINE_VIEW, baseline, "reference", {}),
        (
            "identity_lossless_duplicate",
            baseline.copy(),
            "identity_control",
            {"expected_pixel_identity_with_baseline": True},
        ),
        (
            "native_linear",
            canonical_polarity(linear_voi(pixels.modality, center, width), pixels.photometric),
            "primary_continuous",
            {"center": center, "width": width, "window_source": pixels.window_source},
        ),
        (
            "center_minus_0p05w",
            canonical_polarity(
                linear_voi(pixels.modality, center - 0.05 * width, width), pixels.photometric
            ),
            "primary_continuous",
            {"center": center - 0.05 * width, "width": width},
        ),
        (
            "center_plus_0p05w",
            canonical_polarity(
                linear_voi(pixels.modality, center + 0.05 * width, width), pixels.photometric
            ),
            "primary_continuous",
            {"center": center + 0.05 * width, "width": width},
        ),
        (
            "width_x0p8",
            canonical_polarity(linear_voi(pixels.modality, center, 0.8 * width), pixels.photometric),
            "primary_continuous",
            {"center": center, "width": 0.8 * width},
        ),
        (
            "width_x1p25",
            canonical_polarity(linear_voi(pixels.modality, center, 1.25 * width), pixels.photometric),
            "primary_continuous",
            {"center": center, "width": 1.25 * width},
        ),
        (
            "native_sigmoid",
            canonical_polarity(sigmoid_voi(pixels.modality, center, width), pixels.photometric),
            "primary_continuous" if sigmoid_is_native else "secondary_physician_validation",
            {
                "center": center,
                "width": width,
                "voi_lut_function": "SIGMOID",
                "dicom_declares_sigmoid": sigmoid_is_native,
                "cannot_drive_primary_gate": not sigmoid_is_native,
            },
        ),
    ]
    crop_bounds, crop_retention = conservative_blank_crop(baseline, all_boxes)
    crop = resize_crop(baseline, crop_bounds)
    arrays.append(
        (
            "blank_border_zoom",
            crop,
            "primary_continuous",
            {"crop_bounds_xyxy": list(crop_bounds), **crop_retention},
        )
    )
    arrays.append(
        (
            "polarity_toggle",
            1.0 - baseline,
            "secondary_polarity_toggle",
            {"secondary_only": True, "cannot_drive_primary_gate": True},
        )
    )

    arrays.append(
        (
            "content_loss_downsample",
            content_loss_downsample(baseline),
            "secondary_positive_control",
            {
                "downsample_size": [32, 32],
                "secondary_only": True,
                "cannot_drive_primary_gate": True,
            },
        )
    )

    # A fixed, label-independent region is the only saturation ROI.  Reader
    # boxes would make clinical-validity admission depend on the target outcome.
    baseline_mask, roi_source = roi_mask(baseline.shape, [])
    baseline_saturation = saturation_fraction(baseline, baseline_mask)
    views: list[dict[str, Any]] = []
    for name, values, track, parameters in arrays:
        audit_mask = baseline_mask
        if name == "blank_border_zoom":
            audit_mask, roi_source = roi_mask(baseline.shape, [])
        saturation = saturation_fraction(values, audit_mask)
        correlation = edge_correlation(baseline, values)
        bbox_retention = bool(parameters.get("bbox_retention", True))
        guard = (
            np.isfinite(values).all()
            and bbox_retention
            and saturation <= min(0.35, baseline_saturation + 0.15)
        )
        if name == "blank_border_zoom":
            guard = guard and float(parameters["body_mask_retained"]) >= 0.995
            guard = guard and float(parameters["edge_energy_retained"]) >= 0.98
            # A no-op crop is a valid identity export, but not a perturbation
            # and therefore cannot enter the clinical-render orbit.
            guard = guard and float(parameters["cropped_area_fraction"]) > 1e-6
        elif track != "secondary_polarity_toggle":
            guard = guard and correlation >= 0.75
        audit = {
            "pixel_sha256": pixel_sha256(values),
            "finite_fraction": float(np.mean(np.isfinite(values))),
            "roi_source": roi_source,
            "roi_saturation_fraction": saturation,
            "baseline_roi_saturation_fraction": baseline_saturation,
            "roi_saturation_increase": saturation - baseline_saturation,
            "display_edge_correlation_with_baseline": correlation,
            "bbox_retention": bbox_retention,
            "clinical_guard_pass": bool(guard),
        }
        views.append(
            {
                "name": name,
                "track": track,
                "is_primary": track == "primary_continuous",
                "parameters": parameters,
                "audit": audit,
                "image": to_pil(values),
            }
        )
    hashes = {view["name"]: view["audit"]["pixel_sha256"] for view in views}
    identity = next(view for view in views if view["name"] == "identity_lossless_duplicate")
    identity_matches = hashes[BASELINE_VIEW] == hashes["identity_lossless_duplicate"]
    identity["audit"]["identity_hash_matches_baseline"] = identity_matches
    identity["audit"]["clinical_guard_pass"] = bool(
        identity["audit"]["clinical_guard_pass"] and identity_matches
    )
    if not identity_matches:
        raise RuntimeError("lossless identity control changed rendered pixels")
    return views


@torch.inference_mode()
def score_image(bot: Any, image: Image.Image, prompt: str, ids: Mapping[str, int]) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, _ = prepared_embeddings(bot, prompt, tensor)
    output = bot.model.model(
        input_ids=None,
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state[0, -1].float()
    weight = bot.model.get_output_embeddings().weight
    token_ids = torch.tensor([ids[state] for state in VERBALIZERS], device=weight.device)
    logits_tensor = hidden @ weight.index_select(0, token_ids).float().T
    logits = {
        state: float(logits_tensor[index].cpu())
        for index, state in enumerate(VERBALIZERS)
    }
    polarity = logits["supported"] - logits["refuted"]
    commitment = max(logits["supported"], logits["refuted"]) - logits["undetermined"]
    prediction = max(logits, key=logits.get)
    return {
        "logits": logits,
        "polarity": float(polarity),
        "commitment": float(commitment),
        "prediction": prediction,
        "readout": "FP32 hidden @ FP32 Yes/No/Maybe lm-head rows at exact answer position",
    }


def balanced_rows(
    manifest: Path,
    split: str,
    findings: Sequence[str],
    votes: Sequence[int],
    per_bin: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in load_jsonl(manifest)
        if str(row.get("experiment_split")) == split
        and str(row["finding"]) in set(findings)
        and int(row["positive_votes"]) in set(votes)
    ]
    selected: list[dict[str, Any]] = []
    for finding in findings:
        for vote in votes:
            group = [
                row
                for row in eligible
                if str(row["finding"]) == finding and int(row["positive_votes"]) == vote
            ]
            group.sort(
                key=lambda row: hashlib.sha256(
                    f"{VERSION}:{seed}:{finding}:{vote}:{row['image_id']}".encode()
                ).hexdigest()
            )
            if len(group) < per_bin:
                raise ValueError(
                    f"{finding} vote={vote} has {len(group)} rows, needs {per_bin}"
                )
            selected.extend(group[:per_bin])
    selected.sort(key=lambda row: (str(row["image_id"]), str(row["finding"])))
    keys = [safe_record_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("balanced selection contains duplicate record keys")
    return selected


def freeze_config(candidate: dict[str, Any], path: Path, resume: bool) -> dict[str, Any]:
    if not resume:
        if path.exists():
            raise FileExistsError(path)
        candidate["fingerprint"] = canonical_json_sha256(
            {key: value for key, value in candidate.items() if key not in {"created_at", "command"}}
        )
        atomic_json(path, candidate)
        return candidate
    if not path.is_file():
        raise FileNotFoundError("--resume requires the original config.json")
    existing = json.loads(path.read_text(encoding="utf-8"))
    ignored = {"created_at", "command", "fingerprint"}
    left = {key: value for key, value in existing.items() if key not in ignored}
    right = {key: value for key, value in candidate.items() if key not in ignored}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise ValueError(f"refusing resume after config drift: {changed}")
    return existing


def valid_completed_shard(
    path: Path,
    record_key: str,
    fingerprint: str,
    require_scores: bool = True,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        BASELINE_VIEW,
        "identity_lossless_duplicate",
        "native_linear",
        "center_minus_0p05w",
        "center_plus_0p05w",
        "width_x0p8",
        "width_x1p25",
        "native_sigmoid",
        "blank_border_zoom",
        "polarity_toggle",
        "content_loss_downsample",
    }
    names = {str(view.get("name")) for view in row.get("views", [])}
    return bool(
        row.get("status") == ("ok" if require_scores else "audit_only")
        and row.get("record_key") == record_key
        and row.get("config_fingerprint") == fingerprint
        and names == expected
        and all(
            "audit" in view and (not require_scores or "scores" in view)
            for view in row.get("views", [])
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--split", choices=("pilot", "dev", "confirmation"), default="pilot")
    parser.add_argument("--findings", nargs="+", default=list(DEFAULT_FINDINGS))
    parser.add_argument("--votes", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--per-bin", type=int, default=10)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--render-audit-only",
        action="store_true",
        help="CPU-only DICOM/render canary; writes separate audit shards and never loads Huatuo",
    )
    args = parser.parse_args()
    if args.per_bin <= 0 or any(value not in {0, 1, 2, 3} for value in args.votes):
        raise ValueError("per-bin must be positive and votes must lie in 0..3")
    full_rows = balanced_rows(
        args.manifest, args.split, args.findings, args.votes, args.per_bin, args.seed
    )
    rows = full_rows if args.max_cases is None else full_rows[: args.max_cases]
    if not rows:
        raise ValueError("selection is empty")
    bbox_rows = load_jsonl(args.bboxes)
    boxes_by_claim = {
        (str(row["image_id"]), str(row["finding"])): list(row.get("boxes", []))
        for row in bbox_rows
    }
    boxes_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in bbox_rows:
        boxes_by_image.setdefault(str(row["image_id"]), []).extend(row.get("boxes", []))

    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    selection_keys = [safe_record_key(row) for row in rows]
    config_candidate: dict[str, Any] = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "vindr-cxr-1.0.0-reader-votes",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "bboxes": str(args.bboxes.resolve()),
        "bboxes_sha256": sha256_file(args.bboxes),
        "image_root": str(args.image_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "model_artifact_fingerprint": model_artifact_fingerprint(args.model_dir),
        "split": args.split,
        "findings": list(args.findings),
        "votes": list(args.votes),
        "per_bin_before_max_cases": args.per_bin,
        "max_cases": args.max_cases,
        "selected_claims": len(rows),
        "selection_keys_sha256": canonical_json_sha256(selection_keys),
        "selection_keys": selection_keys,
        "seed": args.seed,
        "device": args.device,
        "render_audit_only": args.render_audit_only,
        "baseline_view": BASELINE_VIEW,
        "primary_track": "primary_continuous",
        "secondary_views_excluded_from_gate": sorted(SECONDARY_NAMES),
        "render_contract": (
            "correct modality transform and MONOCHROME1 normalization precede all views; "
            "one-factor clinical render orbit, label-independent fixed thorax audit ROI, "
            "identity control, and secondary polarity/content-loss controls; SIGMOID is "
            "primary only when declared by the source DICOM"
        ),
        "clinical_guard_thresholds": {
            "roi_saturation_ceiling": 0.35,
            "roi_saturation_increase_ceiling": 0.15,
            "intensity_view_edge_correlation_floor": 0.75,
            "crop_body_mask_retention_floor": 0.995,
            "crop_edge_energy_retention_floor": 0.98,
        },
        "logit_readout": "FP32 Yes/No/Maybe rows at the exact next-token answer position",
        "code_sha256": sha256_file(Path(__file__)),
        "command": " ".join(sys.argv),
    }
    config = freeze_config(config_candidate, args.output_dir / "config.json", args.resume)
    fingerprint = str(config["fingerprint"])

    bot = None
    ids: dict[str, int] = {}
    if not args.render_audit_only:
        klass = import_huatuo(args.huatuo_root)
        bot = klass(str(args.model_dir), device=args.device)
        ids = label_ids(bot)
        if set(ids) != set(VERBALIZERS):
            raise RuntimeError("unexpected Huatuo verbalizer mapping")
    timings: list[float] = []
    completed, failures = 0, 0
    cached_image_id: str | None = None
    cached_pixels: DicomPixels | None = None
    cached_path: Path | None = None
    cached_file_hash: str | None = None
    for index, row in enumerate(rows, start=1):
        record_key = safe_record_key(row)
        shard_path = shards_dir / f"{record_key}.json"
        if valid_completed_shard(
            shard_path,
            record_key,
            fingerprint,
            require_scores=not args.render_audit_only,
        ):
            completed += 1
            print(f"[{index}/{len(rows)}] resume skip {record_key}", flush=True)
            continue
        started = time.perf_counter()
        try:
            image_id, finding = str(row["image_id"]), str(row["finding"])
            path = resolve_image(row, args.image_root)
            if not path.is_file():
                raise FileNotFoundError(path)
            if cached_image_id != image_id:
                cached_pixels = read_dicom_pixels(path)
                cached_path = path
                cached_file_hash = sha256_file(path)
                cached_image_id = image_id
            assert cached_pixels is not None and cached_path == path and cached_file_hash is not None
            views = build_render_views(
                cached_pixels,
                boxes_by_claim.get((image_id, finding), []),
                boxes_by_image.get(image_id, []),
            )
            prompt = prompt_for(finding)
            output_views = []
            for view in views:
                image = view.pop("image")
                output = dict(view)
                if bot is not None:
                    output["scores"] = score_image(bot, image, prompt, ids)
                output_views.append(output)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            payload = {
                "version": VERSION,
                "status": "audit_only" if args.render_audit_only else "ok",
                "record_key": record_key,
                "config_fingerprint": fingerprint,
                "dataset": row.get("dataset", "vindr-cxr-1.0.0"),
                "image_id": image_id,
                "finding": finding,
                "positive_votes": int(row["positive_votes"]),
                "reader_votes": row.get("reader_votes", []),
                "reader_support": float(row["reader_support"]),
                "experiment_split": row.get("experiment_split"),
                "reference_source": row.get("reference_source"),
                "evidence_grade": row.get("evidence_grade"),
                "dicom": {
                    "relative_path": row.get("dicom_relpath"),
                    "file_sha256": cached_file_hash,
                    "metadata": cached_pixels.metadata,
                },
                "prompt": prompt,
                "verbalizer_token_ids": ids,
                "views": output_views,
                "elapsed_seconds": elapsed,
            }
            atomic_json(shard_path, payload)
            completed += 1
            by_hash: dict[str, list[str]] = {}
            for view in output_views:
                by_hash.setdefault(str(view["audit"]["pixel_sha256"]), []).append(str(view["name"]))
            unexpected_collisions = [
                names
                for names in by_hash.values()
                if len(names) > 1 and set(names) != {BASELINE_VIEW, "identity_lossless_duplicate"}
            ]
            print(
                f"[{index}/{len(rows)}] ok {record_key} {elapsed:.1f}s "
                f"unexpected_pixel_collisions={unexpected_collisions}",
                flush=True,
            )
        except Exception as error:  # preserve evidence without treating it complete
            failures += 1
            atomic_json(
                shard_path,
                {
                    "version": VERSION,
                    "status": "error",
                    "record_key": record_key,
                    "config_fingerprint": fingerprint,
                    "image_id": row.get("image_id"),
                    "finding": row.get("finding"),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{index}/{len(rows)}] ERROR {record_key}: {error!r}", file=sys.stderr, flush=True)
    run_state = {
        "version": VERSION,
        "config_fingerprint": fingerprint,
        "selected_claims": len(rows),
        "complete_shards": completed,
        "error_shards_this_invocation": failures,
        "mean_seconds_per_computed_claim": None if not timings else float(np.mean(timings)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_dir / "run_state.json", run_state)
    print(json.dumps(run_state, indent=2))
    if failures:
        raise RuntimeError(f"{failures} claim shards failed; rerun with --resume after diagnosis")


if __name__ == "__main__":
    main()
