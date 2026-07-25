#!/usr/bin/env python3
"""Audit whether CE predictions depend on real visual evidence.

The audit compares a normal image forward pass with a same-size blank-image
control.  The subset is label-free to define: a row is visual-dependent when
the prediction changes or the constrained class probability moves enough.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import decoded_label_index, resize_image
from corrected_sgta.models_surface import load_adapter
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


ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "visual-dependency-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-type", choices=("binary", "multichoice", "all"), default="binary")
    parser.add_argument("--blank-color", type=int, default=127)
    parser.add_argument("--prob-delta-threshold", type=float, default=0.10)
    parser.add_argument("--logit-delta-threshold", type=float, default=1.0)
    parser.add_argument("--decode-labels", action="store_true")
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


def eligible_rows(rows: list[dict], question_type: str, seed: int, max_samples: int) -> list[dict]:
    selected: list[dict] = []
    for row in rows:
        try:
            kind = task_kind(row)
            if kind == "open":
                continue
            if question_type != "all" and kind != question_type:
                continue
            labels_for_sample(row)
            ground_truth_index(row)
            if resolve_image(row.get("img_name", "")) is None:
                continue
            selected.append(row)
        except ProtocolError:
            continue
    selected.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['qid']}".encode()).hexdigest())
    return selected[:max_samples] if max_samples else selected


def summarize(records: list[dict], config: dict, fingerprint: str) -> dict:
    ok = [row for row in records if row.get("status") == "ok" and row.get("fingerprint") == fingerprint]
    if not ok:
        return {"version": VERSION, "fingerprint": fingerprint, "config": config, "n": 0}
    gt = np.asarray([row["gt_index"] for row in ok], dtype=np.int64)
    image_pred = np.asarray([row["image_pred"] for row in ok], dtype=np.int64)
    blank_pred = np.asarray([row["blank_pred"] for row in ok], dtype=np.int64)
    dependent = np.asarray([row["visual_dependent"] for row in ok], dtype=bool)
    image_correct = image_pred == gt
    blank_correct = blank_pred == gt
    helpful = image_correct & ~blank_correct
    harmful = ~image_correct & blank_correct
    both_wrong_changed = (~image_correct) & (~blank_correct) & (image_pred != blank_pred)
    return {
        "version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "n": len(ok),
        "image_accuracy": float(image_correct.mean()),
        "blank_accuracy": float(blank_correct.mean()),
        "prediction_change_rate": float(np.mean(image_pred != blank_pred)),
        "visual_dependent_rate": float(dependent.mean()),
        "visual_dependent_n": int(dependent.sum()),
        "image_helpful_n": int(helpful.sum()),
        "image_harmful_n": int(harmful.sum()),
        "both_wrong_changed_n": int(both_wrong_changed.sum()),
        "mean_max_prob_delta": float(np.mean([row["max_prob_delta"] for row in ok])),
        "p90_max_prob_delta": float(np.quantile([row["max_prob_delta"] for row in ok], 0.9)),
        "mean_max_logit_delta": float(np.mean([row["max_logit_delta"] for row in ok])),
        "p90_max_logit_delta": float(np.quantile([row["max_logit_delta"] for row in ok], 0.9)),
    }


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    config = {
        "cache_version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "max_samples": args.max_samples,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid)",
        "question_type": args.question_type,
        "blank_control": "same resized size, constant RGB value",
        "blank_color": args.blank_color,
        "prob_delta_threshold": args.prob_delta_threshold,
        "logit_delta_threshold": args.logit_delta_threshold,
        "decode_labels": args.decode_labels,
        "decode_max_new_tokens": args.decode_max_new_tokens,
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(f"metadata mismatch; choose a new output: {args.output}")
    else:
        atomic_json(meta_path, metadata)

    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    target = eligible_rows(rows, args.question_type, args.seed, args.max_samples)
    todo = [row for row in target if str(row["qid"]) not in saved]
    print(f"visual-dep fingerprint={fingerprint[:12]} eligible={len(todo)} cached={len(saved)}", flush=True)
    adapter = None
    errors = 0
    started = time.time()
    try:
        if todo:
            adapter = load_adapter(args.model)
            with args.output.open("a") as output:
                for sample in tqdm(todo, desc=f"visual dependency {args.model}"):
                    try:
                        image_path = resolve_image(sample.get("img_name", ""))
                        assert image_path is not None
                        with Image.open(image_path) as source:
                            image = resize_image(source, args.max_image_side)
                        blank = Image.new("RGB", image.size, (args.blank_color,) * 3)
                        labels = labels_for_sample(sample)
                        prompt = build_prompt(sample)
                        evidence = adapter.forward_ce([image, blank], prompt, labels)
                        logits = np.stack([item.logits for item in evidence]).astype(np.float32)
                        probs = np.stack([softmax(item.logits) for item in evidence]).astype(np.float32)
                        preds = logits.argmax(axis=1).astype(int)
                        max_prob_delta = float(np.max(np.abs(probs[0] - probs[1])))
                        max_logit_delta = float(np.max(np.abs(logits[0] - logits[1])))
                        decoded = None
                        decoded_pred = None
                        if args.decode_labels:
                            decoded = adapter.decode_ce(
                                [image, blank],
                                prompt,
                                max_new_tokens=args.decode_max_new_tokens,
                            )
                            decoded_pred = [
                                decoded_label_index(text, labels, sample) for text in decoded
                            ]
                        row = {
                            "protocol_version": PROTOCOL_VERSION,
                            "cache_schema_version": CACHE_SCHEMA_VERSION,
                            "cache_version": VERSION,
                            "fingerprint": fingerprint,
                            "status": "ok",
                            "qid": sample["qid"],
                            "img_name": sample.get("img_name", ""),
                            "question_type": task_kind(sample),
                            "labels": list(labels),
                            "gt_index": ground_truth_index(sample),
                            "image_size": list(image.size),
                            "style_names": ["image", "blank"],
                            "style_logits": logits.tolist(),
                            "style_probabilities": probs.tolist(),
                            "image_pred": int(preds[0]),
                            "blank_pred": int(preds[1]),
                            "prediction_changed": bool(preds[0] != preds[1]),
                            "max_prob_delta": max_prob_delta,
                            "max_logit_delta": max_logit_delta,
                            "visual_dependent": bool(
                                preds[0] != preds[1]
                                or max_prob_delta >= args.prob_delta_threshold
                                or max_logit_delta >= args.logit_delta_threshold
                            ),
                            "image_correct": bool(int(preds[0]) == ground_truth_index(sample)),
                            "blank_correct": bool(int(preds[1]) == ground_truth_index(sample)),
                            "style_decoded_text": decoded,
                            "style_decoded_prediction": decoded_pred,
                        }
                    except Exception as exc:
                        errors += 1
                        traceback.print_exc()
                        row = {
                            "protocol_version": PROTOCOL_VERSION,
                            "cache_schema_version": CACHE_SCHEMA_VERSION,
                            "cache_version": VERSION,
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
    finally:
        if adapter is not None:
            adapter.close()

    records = [
        json.loads(line)
        for line in args.output.read_text().splitlines()
        if line.strip()
    ]
    summary = summarize(records, config, fingerprint)
    summary["elapsed_minutes"] = (time.time() - started) / 60.0
    summary["errors"] = errors
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
