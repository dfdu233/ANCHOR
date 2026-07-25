#!/usr/bin/env python3
"""Cache paired RULE Yes/No evidence for Original and a DG adapter.

This runner follows the released RULE dataset-specific no-reference prompts and
LLaVA-Med ``vicuna_v1`` conversation template.  It stores surface logits and
last-prompt hidden features for a paired Original/DG SCA-T comparison; it does
not use generated-text parsing as a hidden substitute for the fixed classes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import encode_array, load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.infer_ce_source_adapter import (
    atomic_json,
    paired_forward,
    validate_source_erm_checkpoint,
)
from corrected_sgta.models import LLAVA_IMAGE_PREPROCESS_VERSION
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.protocol_v2 import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION, file_sha256, protocol_fingerprint
from corrected_sgta.train_rule_dg_adapter import BoundedResidualBottleneck, rule_no_reference_prompt
from corrected_sgta.infer_rule_dg_adapter import official_prompt

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "rule-dg-scat-cache-v1"
CONV_MODE = "vicuna_v1"
LABELS = ("Yes", "No")
MAX_IMAGE_SIDE = 384

class RuleScatCacheError(RuntimeError):
    pass

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qids = [str(row.get("question_id")) for row in rows]
    if any(qid in {"", "None"} for qid in qids) or len(qids) != len(set(qids)):
        raise RuleScatCacheError("missing or duplicate RULE question_id")
    return rows

def label_index(answer: object) -> int | None:
    value = str(answer).strip().lower().rstrip(".! ")
    return 0 if value == "yes" else 1 if value == "no" else None

def rule_prompt(row: dict[str, Any], dataset: str) -> str:
    if dataset == "mimic":
        return official_prompt(row, "rule_mimic")
    if dataset == "iuxray":
        return rule_no_reference_prompt(row.get("question", ""))
    raise RuleScatCacheError(f"unsupported RULE dataset: {dataset}")

def select_rows(rows: list[dict[str, Any]], image_root: Path, maximum: int, seed: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        gt = label_index(row.get("answer"))
        image = image_root / str(row.get("image", ""))
        if gt is not None and image.is_file():
            selected.append(row)
    selected.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['question_id']}".encode()).hexdigest())
    return selected[:maximum] if maximum else selected

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-dataset", choices=("mimic", "iuxray"), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-objective", choices=("pooled_erm", "source_invariant"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise RuleScatCacheError("max-samples must be nonnegative")
    rows = load_jsonl(args.questions)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    spec = validate_source_erm_checkpoint(checkpoint, args.expected_objective)
    adapted_style = "anchor" if args.expected_objective == "source_invariant" else "source_erm"
    style_names = ("original", adapted_style)
    selected = select_rows(rows, args.image_root, args.max_samples, args.seed)
    if not selected:
        raise RuleScatCacheError("no usable binary rows with local images")
    config = {
        "cache_version": VERSION,
        "model": "llava",
        "conv_mode": CONV_MODE,
        "rule_dataset": args.rule_dataset,
        "questions": str(args.questions.resolve()),
        "questions_sha256": file_sha256(args.questions),
        "image_root": str(args.image_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_fingerprint": checkpoint.get("fingerprint"),
        "adapter_spec": spec,
        "expected_objective": args.expected_objective,
        "style_names": list(style_names),
        "labels": list(LABELS),
        "prompt_protocol": f"rule_official_no_reference_{args.rule_dataset}",
        "image_preprocessing_version": LLAVA_IMAGE_PREPROCESS_VERSION,
        "max_image_side": MAX_IMAGE_SIDE,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "subset_order": "sha256(seed:question_id)",
        "style_features": "last multimodal prompt hidden state; [2,4096]",
        "style_logits": "fixed Yes/No semantic surface logits",
    }
    fingerprint = protocol_fingerprint(config)
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "n_source_rows": len(rows),
        "n_binary_local": len(selected),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuleScatCacheError("existing metadata fingerprint mismatch")
    else:
        if args.output.exists() and args.output.stat().st_size:
            raise RuleScatCacheError("output exists without matching metadata")
        atomic_json(meta_path, meta)
    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    eligible = [row for row in selected if str(row["question_id"]) not in saved]
    print(f"protocol={PROTOCOL_VERSION} fingerprint={fingerprint[:12]} selected={len(selected)} eligible={len(eligible)}", flush=True)
    if not eligible:
        return
    adapter = LlavaMedAlignmentAdapter(conv_mode=CONV_MODE)
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    module = BoundedResidualBottleneck(int(checkpoint["width"]), int(spec["rank"]), float(spec["max_relative_update"])).to(adapter.model.device)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    started = time.time()
    try:
        with args.output.open("a", encoding="utf-8") as stream:
            for row in tqdm(eligible, desc=f"RULE {args.rule_dataset} DG+SCA-T cache"):
                qid = str(row["question_id"])
                try:
                    image_path = args.image_root / str(row["image"])
                    with Image.open(image_path) as source:
                        image = resize_image(source, MAX_IMAGE_SIDE)
                    evidence = paired_forward(adapter, module, image, rule_prompt(row, args.rule_dataset), LABELS)
                    features = np.stack([item.features for item in evidence])
                    if features.shape != (2, 4096):
                        raise RuleScatCacheError(f"invalid feature shape: {features.shape}")
                    output = {
                        "protocol_version": PROTOCOL_VERSION,
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                        "cache_version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": qid,
                        "image": str(row["image"]),
                        "group_id": str(row["image"]),
                        "question_type": "binary",
                        "labels": list(LABELS),
                        "gt_index": int(label_index(row["answer"])),
                        "style_names": list(style_names),
                        "style_logits": [item.logits.tolist() for item in evidence],
                        "style_label_surface_nll": [item.sequence_nll.tolist() for item in evidence],
                        "style_features": encode_array(features),
                    }
                except Exception as error:
                    errors += 1
                    traceback.print_exc()
                    output = {"protocol_version": PROTOCOL_VERSION, "cache_version": VERSION, "fingerprint": fingerprint, "status": "error", "qid": qid, "error": f"{type(error).__name__}: {error}"[:500]}
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
    finally:
        del module, adapter
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    meta.update({"n_selected": len(selected), "n_complete": len(load_successful_qids(args.output, fingerprint)), "errors_this_invocation": errors, "elapsed_minutes_this_invocation": (time.time()-started)/60.0})
    atomic_json(meta_path, meta)

if __name__ == "__main__":
    main()
