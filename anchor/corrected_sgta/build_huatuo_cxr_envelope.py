#!/usr/bin/env python3
"""Build a high-precision PubMedVision CXR radial source envelope for Huatuo."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corrected_sgta.mosec import (
    DEFAULT_BINS,
    DEFAULT_SIZE,
    model_visible_image,
    radial_log_amplitude,
    stable_sha256,
)


POSITIVE = re.compile(
    r"\b(?:chest\s*(?:x[- ]?ray|radiograph(?:y|s|ic)?|CXR|PA\s+view|AP\s+view)"
    r"|portable\s+(?:chest\s+)?radiograph)\b",
    re.IGNORECASE,
)
COMPETING = re.compile(
    r"\b(?:CT|computed tomography|MRI|magnetic resonance|ultrasound|sonograph"
    r"|microscop|histolog|patholog|PET|endoscop|fundus|OCT)\b",
    re.IGNORECASE,
)
MULTIPANEL = re.compile(
    r"(?:\([A-Ha-h]\)|\b[A-Ha-h][.:]\s|\bfigures?\s+\d+[A-Za-z]?\s*[-–]\s*"
    r"\d+[A-Za-z]?|\bmultiple\s+(?:images|panels)|\bpanels?\b)",
    re.IGNORECASE,
)
NON_CLINICAL = re.compile(
    r"\b(?:confusion matrix|classification model|neural network|heatmap|workflow"
    r"|architecture|dataset examples?|saliency map)\b",
    re.IGNORECASE,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def stable_split(identifier: str, train_fraction: float) -> str:
    digest = hashlib.sha256(identifier.encode()).hexdigest()
    value = int(digest[:16], 16) / float(16**16 - 1)
    return "train" if value < train_fraction else "validation"


def source_rows(metadata_path: Path, train_fraction: float) -> tuple[list[dict], Counter]:
    rows = json.loads(metadata_path.read_text())
    selected: list[dict] = []
    counts: Counter = Counter()
    for row in rows:
        counts["metadata_rows"] += 1
        images = row.get("image") or []
        if isinstance(images, str):
            images = [images]
        caption = str(row.get("Original_Caption") or "")
        if len(images) != 1:
            counts["reject_image_count"] += 1
            continue
        if str(row.get("body_part") or "").strip().lower() != "chest":
            counts["reject_body_part"] += 1
            continue
        if not POSITIVE.search(caption):
            counts["reject_no_cxr_marker"] += 1
            continue
        if COMPETING.search(caption):
            counts["reject_competing_modality"] += 1
            continue
        if MULTIPANEL.search(caption):
            counts["reject_multipanel"] += 1
            continue
        if NON_CLINICAL.search(caption):
            counts["reject_non_clinical"] += 1
            continue
        identifier = str(row.get("id") or images[0])
        selected.append(
            {
                "id": identifier,
                "image": str(images[0]),
                "caption": caption,
                "raw_modality": row.get("modality"),
                "body_part": row.get("body_part"),
                "split": stable_split(identifier, train_fraction),
            }
        )
        counts["selected"] += 1
    return selected, counts


def zip_index(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for archive in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if Path(member).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    output.setdefault(member, archive)
    return output


def load_image(handle: zipfile.ZipFile, member: str) -> Image.Image:
    with Image.open(BytesIO(handle.read(member))) as image:
        return image.convert("RGB").copy()


def pixel_rejection_reason(image: Image.Image, raw_modality: object) -> str | None:
    """Conservative non-learned guards against obvious non-CXR source figures."""
    modality = str(raw_modality or "").strip().lower()
    if any(
        marker in modality
        for marker in (
            "ultrasound",
            "magnetic resonance",
            "microscopy",
            "endoscopy",
            "fundus",
            "optical coherence",
        )
    ):
        return "raw_modality_conflict"
    width, height = image.size
    aspect = max(width, height) / max(min(width, height), 1)
    if aspect > 1.25:
        return "extreme_aspect_or_montage"
    visible = model_visible_image(image, size=DEFAULT_SIZE)
    gray = np.asarray(visible.convert("L"), dtype=np.float64) / 255.0
    start = int(0.15 * DEFAULT_SIZE)
    stop = int(0.85 * DEFAULT_SIZE)
    if float(gray[start:stop].std(axis=1).min()) < 0.02:
        return "internal_uniform_row_or_montage"
    return None


def audit_sheet(items: list[tuple[dict, Image.Image]], output: Path) -> None:
    if not items:
        return
    thumb = 224
    columns = 8
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb, rows * (thumb + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (record, image) in enumerate(items):
        x = (index % columns) * thumb
        y = (index // columns) * (thumb + 28)
        shown = image.copy()
        shown.thumbnail((thumb, thumb), Image.Resampling.BICUBIC)
        canvas.paste(shown, (x + (thumb - shown.width) // 2, y))
        draw.text((x + 4, y + thumb + 4), str(record["id"])[:28], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/dbw/data/PubMedVision"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr"),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    metadata_path = args.dataset_root / "PubMedVision_Original_Caption.json"
    records, filter_counts = source_rows(metadata_path, args.train_fraction)
    index = zip_index(args.dataset_root)
    if args.max_images:
        records = records[: args.max_images]
    handles = {path: zipfile.ZipFile(path) for path in set(index.values())}
    descriptors: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
    used: list[dict] = []
    errors: list[dict] = []
    sheet_items: list[tuple[dict, Image.Image]] = []
    try:
        for position, record in enumerate(records, start=1):
            archive = index.get(record["image"])
            if archive is None:
                errors.append({"id": record["id"], "image": record["image"], "error": "missing archive"})
                continue
            try:
                image = load_image(handles[archive], record["image"])
                rejection = pixel_rejection_reason(image, record.get("raw_modality"))
                if rejection is not None:
                    filter_counts[f"reject_pixel_{rejection}"] += 1
                    continue
                visible = model_visible_image(image, size=DEFAULT_SIZE)
                descriptor = radial_log_amplitude(visible, DEFAULT_BINS)
                descriptors[record["split"]].append(descriptor)
                used_record = {
                    **record,
                    "archive": str(archive),
                    "descriptor_sha256": stable_sha256(descriptor.tolist()),
                }
                used.append(used_record)
                if record["split"] == "validation" and len(sheet_items) < 64:
                    sheet_items.append((record, visible))
            except Exception as exc:  # keep the source scan moving
                errors.append({"id": record["id"], "image": record["image"], "error": repr(exc)})
            if position == 1 or position % args.progress_every == 0 or position == len(records):
                print(
                    json.dumps(
                        {
                            "scanned": position,
                            "selected": len(records),
                            "used": len(used),
                            "errors": len(errors),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    train = np.stack(descriptors["train"])
    validation = np.stack(descriptors["validation"])
    center_mean = train.mean(axis=0)
    center_median = np.median(train, axis=0)
    scale = 1.4826 * np.median(np.abs(train - center_median), axis=0)
    scale = np.maximum(scale, 1e-4)
    base_lower = np.quantile(train, 0.05, axis=0)
    base_upper = np.quantile(train, 0.95, axis=0)
    exceedance = np.maximum(
        (base_lower[None] - validation) / scale[None],
        (validation - base_upper[None]) / scale[None],
    )
    simultaneous_score = np.maximum(exceedance, 0.0).max(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_banks: dict[str, dict[str, object]] = {}
    for coverage in (0.50, 0.80, 0.95):
        inflation = float(
            np.quantile(simultaneous_score, coverage, method="higher")
        )
        lower = base_lower - inflation * scale
        upper = base_upper + inflation * scale
        identity = np.all(
            (validation >= lower[None]) & (validation <= upper[None]), axis=1
        )
        label = f"c{int(coverage * 100)}"
        bank_path = args.output_dir / f"cxr_radial_envelope_{label}.npz"
        np.savez_compressed(
            bank_path,
            mean=center_mean.astype(np.float32),
            median=center_median.astype(np.float32),
            scale=scale.astype(np.float32),
            lower=lower.astype(np.float32),
            upper=upper.astype(np.float32),
        )
        coverage_banks[label] = {
            "coverage": coverage,
            "simultaneous_inflation": inflation,
            "validation_identity_rate": float(identity.mean()),
            "bank": str(bank_path),
            "bank_sha256": file_sha256(bank_path),
        }
    bank_path = args.output_dir / "cxr_radial_envelope.npz"
    bank_path.write_bytes(
        (args.output_dir / "cxr_radial_envelope_c95.npz").read_bytes()
    )
    index_path = args.output_dir / "source_index.jsonl"
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in used))
    if errors:
        write_json(args.output_dir / "errors.json", errors)
    audit_sheet(sheet_items, args.output_dir / "validation_audit_64.jpg")
    metadata = {
        "version": "huatuo-mosec-cxr-bank-v2",
        "dataset": "PubMedVision",
        "model": "HuatuoGPT-Vision-7B",
        "source_scope": "partial_actual_training_source",
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "bank": str(bank_path),
        "bank_sha256": file_sha256(bank_path),
        "selection": {
            "single_image": True,
            "body_part": "chest",
            "positive_pattern": POSITIVE.pattern,
            "competing_pattern": COMPETING.pattern,
            "multipanel_pattern": MULTIPANEL.pattern,
            "non_clinical_pattern": NON_CLINICAL.pattern,
            "filter_counts": dict(filter_counts),
        },
        "preprocess": {
            "square_pad": "CLIP image mean",
            "resize": [DEFAULT_SIZE, DEFAULT_SIZE],
            "resample": "bicubic",
            "range": "[0,1] before CLIP normalization",
        },
        "descriptor": {
            "type": "radial_median_log_fft_amplitude",
            "bins": DEFAULT_BINS,
            "luminance": "BT.709",
        },
        "n_selected": len(records),
        "n_used": len(used),
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_errors": len(errors),
        "envelope": {
            "base_quantiles": [0.05, 0.95],
            "coverage_banks": coverage_banks,
            "default_coverage": 0.95,
        },
        "source_index": str(index_path),
        "source_index_sha256": file_sha256(index_path),
        "audit_sheet": str(args.output_dir / "validation_audit_64.jpg"),
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
