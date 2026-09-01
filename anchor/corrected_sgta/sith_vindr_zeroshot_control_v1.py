#!/usr/bin/env python3
"""Positive controls for SITH-style VinDr probes.

This script asks a narrower sanity question before interpreting SITH direction
results: does the local CLIP tower contain usable VinDr pathology information
at all?  It reports both prompt-based zero-shot scores and a tiny deterministic
ridge probe on pooled image features.  The ridge probe is a positive control,
not a method baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

from anchor.corrected_sgta.sith_medical_preference_probe_v1 import sha256_file
from anchor.corrected_sgta.sith_vindr_activation_probe_v1 import (
    DEFAULT_FINDINGS,
    auc_binary,
    build_manifest,
    read_labels,
    render_dicom,
    stack_batches,
)


PROTOCOL_VERSION = "sith-vindr-zeroshot-control-v1"


POSITIVE_TEMPLATES = (
    "a chest x-ray showing {}",
    "a frontal chest radiograph with {}",
    "radiographic evidence of {}",
    "{} is present on this chest x-ray",
)

NEGATIVE_TEMPLATES = (
    "a chest x-ray without {}",
    "a frontal chest radiograph with no {}",
    "no radiographic evidence of {}",
    "{} is absent on this chest x-ray",
)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_or_build_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest_json:
        payload = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
        if "manifest" not in payload:
            raise KeyError(f"{args.manifest_json} does not contain a top-level manifest")
        return list(payload["manifest"])
    labels = read_labels(Path(args.annotation_csv))
    findings = tuple(item.strip() for item in args.findings.split(",") if item.strip())
    return build_manifest(labels, Path(args.image_root), findings, args.per_group)


def encode_prompt_pair(
    model: CLIPModel,
    tokenizer: CLIPTokenizerFast,
    finding: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for templates in (POSITIVE_TEMPLATES, NEGATIVE_TEMPLATES):
        prompts = [template.format(finding.lower()) for template in templates]
        tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            features = F.normalize(model.get_text_features(**tokens).float(), dim=-1)
        rows.append(F.normalize(features.mean(dim=0), dim=-1).cpu())
    return rows[0], rows[1]


def deterministic_stratified_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for value in (0, 1):
        indices = np.flatnonzero(labels == value)
        rng.shuffle(indices)
        split = max(1, len(indices) // 2)
        train_indices.extend(indices[:split].tolist())
        test_indices.extend(indices[split:].tolist())
    return np.asarray(sorted(train_indices)), np.asarray(sorted(test_indices))


def ridge_probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives < 2 or negatives < 2:
        return {"status": "skipped", "reason": "not enough positive/negative samples"}
    train_idx, test_idx = deterministic_stratified_split(labels, seed)
    x_train = features[train_idx].astype(np.float64)
    x_test = features[test_idx].astype(np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    y_train = labels[train_idx].astype(np.float64) * 2.0 - 1.0
    regularizer = np.eye(x_train.shape[1], dtype=np.float64) * alpha
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + regularizer, x_train.T @ y_train)
    scores = x_test @ weights
    auc = auc_binary(scores, labels[test_idx].astype(np.int32))
    return {
        "status": "complete",
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "test_positive_count": int(labels[test_idx].sum()),
        "test_negative_count": int(len(test_idx) - labels[test_idx].sum()),
        "auc": auc,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    image_root = Path(args.image_root)
    findings = tuple(item.strip() for item in args.findings.split(",") if item.strip())
    manifest = load_or_build_manifest(args)
    if not manifest:
        raise RuntimeError("empty manifest")

    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval().requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(str(model_path), local_files_only=True)

    image_features: list[np.ndarray] = []
    for batch in stack_batches(manifest, args.batch_size):
        images = [render_dicom(image_root / f"{row['image_id']}.dicom") for row in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            features = F.normalize(model.get_image_features(**inputs).float(), dim=-1)
        image_features.append(features.cpu().numpy())
    x = np.concatenate(image_features, axis=0)

    labels_by_finding = {
        finding: np.asarray([finding in set(row["labels"]) for row in manifest], dtype=np.int32)
        for finding in findings
    }
    text_pairs = {
        finding: encode_prompt_pair(model, tokenizer, finding, device)
        for finding in findings
    }
    finding_reports: dict[str, Any] = {}
    for finding in findings:
        labels = labels_by_finding[finding]
        pos_text, neg_text = text_pairs[finding]
        prompt_scores = x @ (pos_text - neg_text).numpy()
        zero_shot_auc = auc_binary(prompt_scores, labels)
        finding_seed_offset = int(hashlib.sha256(finding.encode()).hexdigest()[:8], 16) % 100000
        ridge = ridge_probe_auc(
            x,
            labels,
            alpha=float(args.ridge_alpha),
            seed=int(args.seed) + finding_seed_offset,
        )
        finding_reports[finding] = {
            "positive_count": int(labels.sum()),
            "negative_count": int(len(labels) - labels.sum()),
            "zero_shot_pos_minus_neg_auc": zero_shot_auc,
            "ridge_probe": ridge,
            "positive_prompts": [template.format(finding.lower()) for template in POSITIVE_TEMPLATES],
            "negative_prompts": [template.format(finding.lower()) for template in NEGATIVE_TEMPLATES],
        }

    counts = defaultdict(int)
    for row in manifest:
        counts[row["sampling_group"]] += 1
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "claim_boundary": (
            "Positive-control screen only. Zero-shot and ridge-probe AUCs show "
            "whether pooled CLIP features carry label information on this sample; "
            "they do not validate SITH directions, shortcuts, or mitigation."
        ),
        "model": {
            "path": str(model_path),
            "weights_sha256": sha256_file(model_path / "pytorch_model.bin"),
            "projection_dim": int(model.config.projection_dim),
        },
        "data": {
            "annotation_csv": str(Path(args.annotation_csv)),
            "annotation_sha256": sha256_file(Path(args.annotation_csv)),
            "image_root": str(image_root),
            "manifest_source": str(args.manifest_json) if args.manifest_json else "built_from_annotation_csv",
            "n_images": len(manifest),
            "sampling_counts": dict(sorted(counts.items())),
            "findings": list(findings),
        },
        "settings": {
            "batch_size": args.batch_size,
            "per_group": args.per_group,
            "seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
        },
        "findings": finding_reports,
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
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--findings", default=",".join(DEFAULT_FINDINGS))
    parser.add_argument("--per-group", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="corrected_runs/sith_medical_preference_probe_v1/vindr_zeroshot_control_huatuo_n200.json",
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
        "findings": {
            finding: {
                "zero_shot_auc": report["zero_shot_pos_minus_neg_auc"],
                "ridge_probe_auc": report["ridge_probe"].get("auc"),
            }
            for finding, report in result["findings"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
