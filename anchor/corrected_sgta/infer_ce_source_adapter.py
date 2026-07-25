#!/usr/bin/env python3
"""Cache paired MedHEval CE evidence for identity and Source-ERM.

The label NLL stored here is the next-token NLL of accepted single-token
surface forms.  It is deliberately named ``style_label_surface_nll`` and must
not be interpreted as full-answer sequence NLL.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import (
    encode_array,
    load_successful_qids,
    repair_truncated_jsonl_tail,
)
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.infer_rule_dg_adapter import checkpoint_adapter_spec
from corrected_sgta.models import LLAVA_IMAGE_PREPROCESS_VERSION
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    build_prompt,
    file_sha256,
    ground_truth_index,
    labels_for_sample,
    protocol_fingerprint,
    resolve_image,
    task_kind,
    validate_dataset,
)
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.train_rule_dg_adapter import (
    BoundedResidualBottleneck,
    attach_postprojector_adapter,
)
from corrected_sgta.train_rule_source_group_adapter import (
    VERSION as SOURCE_GROUP_VERSION,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "medheval-source-erm-ce-cache-v1"
MODEL = "llava"
CONV_MODE = "mistral_instruct"
MAX_IMAGE_SIDE = 384
SOURCE_ERM_OBJECTIVE = "pooled_erm"
SOURCE_INVARIANT_OBJECTIVE = "source_invariant"
SUPPORTED_OBJECTIVES = (SOURCE_ERM_OBJECTIVE, SOURCE_INVARIANT_OBJECTIVE)


class SourceAdapterError(RuntimeError):
    """Raised when a checkpoint or cache violates the frozen protocol."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_source_erm_checkpoint(
    payload: dict[str, Any], expected_objective: str = SOURCE_ERM_OBJECTIVE
) -> dict[str, Any]:
    if payload.get("version") != SOURCE_GROUP_VERSION:
        raise SourceAdapterError(
            f"checkpoint must use {SOURCE_GROUP_VERSION}, got {payload.get('version')}"
        )
    spec = checkpoint_adapter_spec(payload)
    if spec.get("location") != "post":
        raise SourceAdapterError("Source-ERM checkpoint must be post-projector")
    if expected_objective not in SUPPORTED_OBJECTIVES:
        raise SourceAdapterError(f"unsupported expected objective: {expected_objective}")
    if spec.get("objective") != expected_objective:
        raise SourceAdapterError(
            f"checkpoint objective must be {expected_objective}, "
            f"got {spec.get('objective')}"
        )
    width = int(payload.get("width", 0))
    if width != 4096:
        raise SourceAdapterError(f"checkpoint width must be 4096, got {width}")
    if not isinstance(payload.get("state_dict"), dict):
        raise SourceAdapterError("checkpoint lacks adapter state_dict")
    return spec


def select_binary_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    seed: int,
    resolver: Callable[[str], Path | None] = resolve_image,
) -> list[dict[str, Any]]:
    """Match the fixed protocol_v2 qid-hash subset, restricted to Yes/No."""

    selected = []
    for sample in rows:
        try:
            if task_kind(sample) != "binary":
                continue
            labels = labels_for_sample(sample)
            if tuple(label.lower() for label in labels) != ("yes", "no"):
                raise ProtocolError(f"noncanonical binary labels: {labels}")
            ground_truth_index(sample)
            if resolver(str(sample.get("img_name", ""))) is None:
                continue
            selected.append(sample)
        except ProtocolError:
            continue
    selected.sort(
        key=lambda sample: hashlib.sha256(
            f"{seed}:{sample['qid']}".encode()
        ).hexdigest()
    )
    return selected[:max_samples] if max_samples else selected


def paired_forward(
    adapter: LlavaMedAlignmentAdapter,
    module: BoundedResidualBottleneck,
    image: Image.Image,
    prompt: str,
    labels: tuple[str, ...],
) -> list[Any]:
    """Run the same surface-form interface with only the adapter arm changed."""

    evidence = []
    for active in (None, module):
        context = (
            nullcontext()
            if active is None
            else attach_postprojector_adapter(adapter.model, active)
        )
        with context:
            evidence.append(adapter.forward_ce([image], prompt, labels)[0])
    return evidence


def code_identity(project_root: Path) -> dict[str, str]:
    names = (
        "corrected_sgta/infer_ce_source_adapter.py",
        "corrected_sgta/infer_ce.py",
        "corrected_sgta/models_surface.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/infer_rule_dg_adapter.py",
        "corrected_sgta/train_rule_dg_adapter.py",
        "corrected_sgta/train_rule_source_group_adapter.py",
        "corrected_sgta/protocol_v2.py",
    )
    return {name: sha256_file(project_root / name) for name in names}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-objective",
        choices=SUPPORTED_OBJECTIVES,
        default=SOURCE_ERM_OBJECTIVE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise SourceAdapterError("max-samples must be non-negative")
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SourceAdapterError("dataset must be a JSON array")
    validation = validate_dataset(rows)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise SourceAdapterError("checkpoint must be a mapping")
    module_spec = validate_source_erm_checkpoint(
        checkpoint, args.expected_objective
    )
    adapted_style = (
        "anchor"
        if args.expected_objective == SOURCE_INVARIANT_OBJECTIVE
        else "source_erm"
    )
    style_names = ("original", adapted_style)
    project_root = Path(__file__).resolve().parents[1]
    config = {
        "cache_version": VERSION,
        "model": MODEL,
        "model_identity": model_identity(MODEL),
        "conv_mode": CONV_MODE,
        "image_preprocessing_version": LLAVA_IMAGE_PREPROCESS_VERSION,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_version": checkpoint["version"],
        "checkpoint_fingerprint": checkpoint.get("fingerprint"),
        "adapter_spec": module_spec,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid)",
        "question_type": "binary",
        "labels": ["Yes", "No"],
        "style_names": list(style_names),
        "style_logits": "max over equal single-token semantic surface-form groups",
        "style_label_surface_nll": (
            "minimum next-token full-vocabulary NLL within each semantic "
            "single-token surface-form group; not complete-sequence NLL"
        ),
        "style_features": "last multimodal prompt hidden state; [2,4096]",
        "max_image_side": MAX_IMAGE_SIDE,
        "code_identity": code_identity(project_root),
    }
    fingerprint = protocol_fingerprint(config)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise SourceAdapterError(
                f"metadata fingerprint mismatch; choose a new output: {metadata_path}"
            )
    else:
        if args.output.exists() and args.output.stat().st_size:
            raise SourceAdapterError("output exists without matching metadata")
        atomic_json(metadata_path, metadata)

    repair = repair_truncated_jsonl_tail(args.output)
    if repair["action"] != "none":
        print(f"cache tail repair: {repair}", flush=True)
    saved = load_successful_qids(args.output, fingerprint)
    selected = select_binary_rows(rows, args.max_samples, args.seed)
    eligible = [sample for sample in selected if str(sample["qid"]) not in saved]
    print(
        f"protocol={PROTOCOL_VERSION} cache={VERSION} "
        f"fingerprint={fingerprint[:12]} selected={len(selected)} "
        f"eligible={len(eligible)} cached={len(saved)}",
        flush=True,
    )
    if not eligible:
        return

    adapter = LlavaMedAlignmentAdapter(conv_mode=CONV_MODE)
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    module = BoundedResidualBottleneck(
        int(checkpoint["width"]),
        int(module_spec["rank"]),
        float(module_spec["max_relative_update"]),
    ).to(adapter.model.device)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()

    started = time.time()
    errors = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("a", encoding="utf-8") as stream:
            for sample in tqdm(eligible, desc="MedHEval Source-ERM CE"):
                try:
                    image_path = resolve_image(str(sample.get("img_name", "")))
                    if image_path is None:
                        raise FileNotFoundError(sample.get("img_name"))
                    with Image.open(image_path) as source:
                        image = resize_image(source, MAX_IMAGE_SIDE)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    evidence = paired_forward(adapter, module, image, prompt, labels)
                    features = np.stack([item.features for item in evidence])
                    if features.shape != (2, 4096):
                        raise SourceAdapterError(
                            f"unexpected style feature shape: {features.shape}"
                        )
                    if any(item.sequence_nll is None for item in evidence):
                        raise SourceAdapterError("surface adapter returned no label NLL")
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                        "cache_version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "question_type": "binary",
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "style_names": list(style_names),
                        "style_metadata": [
                            {"family": "original", "adapter": None},
                            {
                                "family": adapted_style,
                                "checkpoint_version": checkpoint["version"],
                                "checkpoint_fingerprint": checkpoint.get("fingerprint"),
                                "adapter_location": "post",
                                "objective": module_spec["objective"],
                            },
                        ],
                        "style_logits": [item.logits.tolist() for item in evidence],
                        "style_label_surface_nll": [
                            item.sequence_nll.tolist() for item in evidence
                        ],
                        "style_features": encode_array(features),
                    }
                except Exception as error:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                        "cache_version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "error",
                        "qid": sample.get("qid"),
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del module, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata["n_selected"] = len(selected)
    metadata["n_complete"] = len(load_successful_qids(args.output, fingerprint))
    metadata["errors_this_invocation"] = errors
    metadata["elapsed_minutes_this_invocation"] = (time.time() - started) / 60.0
    atomic_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
