#!/usr/bin/env python3
"""Image-activation probe for SITH-style VO singular directions on VinDr.

This first probe asks whether weight-space singular directions carry real CXR
label information beyond random hidden directions.  It does not use the result
as clinical truth, shortcut causality, or mitigation evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

from anchor.corrected_sgta.sith_medical_preference_probe_v1 import (
    PROTOCOL_VERSION as WEIGHT_PROBE_PROTOCOL,
)
from anchor.corrected_sgta.sith_medical_preference_probe_v1 import (
    deterministic_svd,
    get_vo_matrix,
    parse_layers,
    sha256_file,
)


PROTOCOL_VERSION = "sith-vindr-activation-probe-v2-response-and-top-random-controls"


DEFAULT_FINDINGS = (
    "Cardiomegaly",
    "Pleural effusion",
    "Aortic enlargement",
    "Pulmonary fibrosis",
)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_labels(csv_path: Path) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            class_name = row["class_name"]
            labels[image_id].add(class_name)
    return labels


def build_manifest(
    labels: dict[str, set[str]],
    image_root: Path,
    findings: tuple[str, ...],
    per_group: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    no_finding = sorted(
        image_id
        for image_id, image_labels in labels.items()
        if image_labels == {"No finding"} and (image_root / f"{image_id}.dicom").is_file()
    )
    for image_id in no_finding[:per_group]:
        used.add(image_id)
        rows.append(
            {
                "image_id": image_id,
                "sampling_group": "No finding",
                "labels": sorted(labels[image_id]),
            }
        )
    for finding in findings:
        candidates = sorted(
            image_id
            for image_id, image_labels in labels.items()
            if finding in image_labels
            and image_id not in used
            and (image_root / f"{image_id}.dicom").is_file()
        )
        for image_id in candidates[:per_group]:
            used.add(image_id)
            rows.append(
                {
                    "image_id": image_id,
                    "sampling_group": finding,
                    "labels": sorted(labels[image_id]),
                }
            )
    return rows


def render_dicom(path: Path) -> Image.Image:
    import pydicom

    ds = pydicom.dcmread(str(path))
    array = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        array = array.max() - array
    lo, hi = np.percentile(array, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(array.min()), float(array.max())
    scaled = np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    u8 = (scaled * 255.0).round().astype(np.uint8)
    return Image.fromarray(u8, mode="L").convert("RGB")


def auc_binary(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[labels.astype(bool)].sum()
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def stack_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def compute_direction_tables(
    state: dict[str, torch.Tensor],
    layers: list[int],
    num_heads: int,
    rank: int,
    vector_type: str,
) -> dict[int, torch.Tensor]:
    tables: dict[int, torch.Tensor] = {}
    for layer in layers:
        vo = get_vo_matrix(state, layer=layer, num_heads=num_heads)
        u, _s, vh = deterministic_svd(vo, rank=rank)
        if vector_type == "right":
            raw = vh.reshape(num_heads * rank, vh.shape[-1])
        elif vector_type == "left":
            raw = u.permute(0, 2, 1).reshape(num_heads * rank, u.shape[-2])
        else:
            raise ValueError(f"unsupported vector_type={vector_type!r}")
        tables[layer] = F.normalize(raw.float(), dim=-1)
    return tables


def direction_responses(tokens: torch.Tensor, directions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return sign-invariant image-level responses for one direction table.

    The SVD sign is arbitrary, so all primary statistics use absolute response.
    We keep three granularities to avoid a false negative caused by choosing only
    a local-patch or only a CLS statistic.
    """

    normalized = F.normalize(tokens.float(), dim=-1)
    cls = normalized[:, :1, :]
    patches = normalized[:, 1:, :]
    cls_dot = torch.einsum("btd,rd->btr", cls, directions).abs().squeeze(1)
    patch_dot = torch.einsum("btd,rd->btr", patches, directions).abs()
    return {
        "cls_abs_dot_layernorm1_input": cls_dot,
        "mean_abs_patch_dot_layernorm1_input": patch_dot.mean(dim=1),
        "max_abs_patch_dot_layernorm1_input": patch_dot.amax(dim=1),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    image_root = Path(args.image_root)
    annotation_csv = Path(args.annotation_csv)
    layers = parse_layers(args.layers)
    findings = tuple(item.strip() for item in args.findings.split(",") if item.strip())
    device = torch.device(args.device)

    labels = read_labels(annotation_csv)
    manifest = build_manifest(labels, image_root, findings, args.per_group)
    if not manifest:
        raise RuntimeError("empty manifest")

    model = CLIPModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval().requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    state = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
    num_heads = int(model.config.vision_config.num_attention_heads)
    rank = int(args.rank)
    directions = compute_direction_tables(state, layers, num_heads, rank, args.vector_type)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_directions = {
        layer: F.normalize(
            torch.randn(directions[layer].shape, generator=generator),
            dim=-1,
        )
        for layer in layers
    }

    response_names = [
        "cls_abs_dot_layernorm1_input",
        "mean_abs_patch_dot_layernorm1_input",
        "max_abs_patch_dot_layernorm1_input",
    ]
    observed_scores: dict[int, dict[str, list[np.ndarray]]] = {
        layer: {name: [] for name in response_names} for layer in layers
    }
    random_scores: dict[int, dict[str, list[np.ndarray]]] = {
        layer: {name: [] for name in response_names} for layer in layers
    }
    for batch in stack_batches(manifest, args.batch_size):
        images = [render_dicom(image_root / f"{row['image_id']}.dicom") for row in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.vision_model(**inputs, output_hidden_states=True)
        for layer in layers:
            hidden = output.hidden_states[layer]
            normalized = model.vision_model.encoder.layers[layer].layer_norm1(hidden).cpu()
            direction = directions[layer]
            random_direction = random_directions[layer]
            observed = direction_responses(normalized, direction)
            random = direction_responses(normalized, random_direction)
            for name in response_names:
                observed_scores[layer][name].append(observed[name].numpy())
                random_scores[layer][name].append(random[name].numpy())

    y = {
        finding: np.asarray([finding in set(row["labels"]) for row in manifest], dtype=np.int32)
        for finding in findings
    }
    layer_reports = []
    for layer in layers:
        response_reports = {}
        for response_name in response_names:
            observed = np.concatenate(observed_scores[layer][response_name], axis=0)
            random = np.concatenate(random_scores[layer][response_name], axis=0)
            findings_report = {}
            for finding in findings:
                labels_binary = y[finding]
                observed_aucs = np.asarray(
                    [
                        value
                        for value in (
                            auc_binary(observed[:, idx], labels_binary)
                            for idx in range(observed.shape[1])
                        )
                        if value is not None
                    ],
                    dtype=np.float64,
                )
                random_aucs = np.asarray(
                    [
                        value
                        for value in (
                            auc_binary(random[:, idx], labels_binary)
                            for idx in range(random.shape[1])
                        )
                        if value is not None
                    ],
                    dtype=np.float64,
                )
                observed_abs = np.abs(observed_aucs - 0.5)
                random_abs = np.abs(random_aucs - 0.5)
                top_indices = np.argsort(observed_abs)[::-1][: args.top_directions]
                findings_report[finding] = {
                    "positive_count": int(labels_binary.sum()),
                    "negative_count": int(len(labels_binary) - labels_binary.sum()),
                    "observed_mean_abs_auc_minus_half": float(observed_abs.mean()),
                    "random_mean_abs_auc_minus_half": float(random_abs.mean()),
                    "observed_minus_random_mean_abs": float(
                        observed_abs.mean() - random_abs.mean()
                    ),
                    "observed_top_abs_auc_minus_half": float(observed_abs.max()),
                    "random_top_abs_auc_minus_half": float(random_abs.max()),
                    "observed_minus_random_top_abs": float(observed_abs.max() - random_abs.max()),
                    "random_p95_abs_auc_minus_half": float(np.quantile(random_abs, 0.95)),
                    "top_observed_directions": [
                        {
                            "flat_direction_index": int(idx),
                            "head": int(idx // rank),
                            "rank_index": int(idx % rank),
                            "auc": float(observed_aucs[idx]),
                            "abs_auc_minus_half": float(observed_abs[idx]),
                        }
                        for idx in top_indices
                    ],
                }
            response_reports[response_name] = {"findings": findings_report}
        layer_reports.append({"layer": layer, "responses": response_reports})

    counts = defaultdict(int)
    for row in manifest:
        counts[row["sampling_group"]] += 1
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "weight_probe_protocol": WEIGHT_PROBE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "claim_boundary": (
            "Exploratory image-activation screen on VinDr labels. It can reject "
            "or motivate SITH-style directions, but cannot establish clinical "
            "shortcut causality, source preference, or mitigation efficacy."
        ),
        "model": {
            "path": str(model_path),
            "weights_sha256": sha256_file(model_path / "pytorch_model.bin"),
            "vision_layers": int(model.config.vision_config.num_hidden_layers),
            "vision_heads": num_heads,
            "hidden_size": int(model.config.vision_config.hidden_size),
        },
        "data": {
            "annotation_csv": str(annotation_csv),
            "annotation_sha256": sha256_file(annotation_csv),
            "image_root": str(image_root),
            "n_images": len(manifest),
            "sampling_counts": dict(sorted(counts.items())),
            "findings": list(findings),
        },
        "settings": {
            "layers": layers,
            "rank": rank,
            "vector_type": args.vector_type,
            "batch_size": args.batch_size,
            "per_group": args.per_group,
            "seed": args.seed,
            "direction_responses": response_names,
            "random_control": "same-count isotropic hidden directions through same response statistic",
            "top_hit_control": "observed best direction is compared to random best direction over the same number of directions",
        },
        "layers": layer_reports,
        "manifest": manifest,
    }
    result["fingerprint"] = canonical_json_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/home/dbw/models/HuatuoGPT-Vision-7B/vit/clip_vit_large_patch14_336",
    )
    parser.add_argument("--annotation-csv", default="/workspace/vinbigdata/train.csv")
    parser.add_argument("--image-root", default="/workspace/vinbigdata/train")
    parser.add_argument("--findings", default=",".join(DEFAULT_FINDINGS))
    parser.add_argument("--layers", default="20-23")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--vector-type", choices=("right", "left"), default="right")
    parser.add_argument("--per-group", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-directions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="corrected_runs/sith_medical_preference_probe_v1/vindr_activation_huatuo_l20_23_rank16_n200.json",
    )
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {
        "output": str(output),
        "n_images": result["data"]["n_images"],
        "sampling_counts": result["data"]["sampling_counts"],
        "layers": [
            {
                "layer": row["layer"],
                "responses": {
                    response_name: {
                        "mean_observed_minus_random_abs_auc": float(
                            np.mean(
                                [
                                    finding["observed_minus_random_mean_abs"]
                                    for finding in response["findings"].values()
                                ]
                            )
                        ),
                        "mean_observed_minus_random_top_abs_auc": float(
                            np.mean(
                                [
                                    finding["observed_minus_random_top_abs"]
                                    for finding in response["findings"].values()
                                ]
                            )
                        ),
                    }
                    for response_name, response in row["responses"].items()
                },
            }
            for row in result["layers"]
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
