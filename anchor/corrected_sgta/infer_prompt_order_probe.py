#!/usr/bin/env python3
"""Resumable diagnostic for semantically equivalent VLM modality orders.

This is a strong-control experiment, not an SGTA method.  It tests whether the
standard image-first prompt is dominated by question-first or question-echo
layouts before spending compute on visual-domain transformations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import resize_image
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
VERSION = "prompt-order-probe-v1"
LAYOUTS = ("image_first", "question_first", "question_echo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--question-type",
        choices=("binary", "multichoice", "all"),
        default="all",
    )
    return parser.parse_args()


def layout_text(prompt: str, image_token: str, layout: str) -> str:
    if layout == "image_first":
        return f"{image_token}\n{prompt}"
    if layout == "question_first":
        return f"{prompt}\n{image_token}"
    if layout == "question_echo":
        return f"{prompt}\n{image_token}\n{prompt}"
    raise ValueError(f"unknown layout: {layout}")


def hulu_content(prompt: str, layout: str) -> list[dict[str, str]]:
    image = {"type": "image"}
    text = {"type": "text", "text": prompt}
    if layout == "image_first":
        return [image, text]
    if layout == "question_first":
        return [text, image]
    if layout == "question_echo":
        return [text, image, text.copy()]
    raise ValueError(f"unknown layout: {layout}")


def hulu_inputs(adapter, image: Image.Image, prompt: str, layout: str):
    conversation = [
        {"role": "user", "content": hulu_content(prompt, layout)}
    ]
    inputs = adapter.processor(
        images=[image],
        conversation=conversation,
        add_system_prompt=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if key == "pixel_values":
                value = value.to(dtype=adapter.model.dtype)
            inputs[key] = value.to(adapter.model.device)
    return inputs


def llava_prompt_ids(adapter, prompt: str, layout: str) -> torch.Tensor:
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(adapter.model.config, "mm_use_im_start_end", False):
        image_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
    conversation = conv_templates["mistral_instruct"].copy()
    conversation.append_message(
        conversation.roles[0], layout_text(prompt, image_token, layout)
    )
    conversation.append_message(conversation.roles[1], None)
    return tokenizer_image_token(
        conversation.get_prompt(),
        adapter.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)


@torch.inference_mode()
def forward_layout(
    adapter,
    model_name: str,
    image: Image.Image,
    prompt: str,
    labels: Sequence[str],
    layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    groups = adapter.label_id_groups(labels)
    if model_name == "hulu":
        inputs = hulu_inputs(adapter, image, prompt, layout)
        result = adapter.model(
            **inputs,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
            num_logits_to_keep=1,
        )
        vocabulary_logits = result.logits[0, -1]
        del result, inputs
    else:
        input_ids = llava_prompt_ids(adapter, prompt, layout).to(
            adapter.model.device
        )
        image_tensor = adapter._process_images([image])
        if isinstance(image_tensor, list):
            image_tensor = [
                item.to(adapter.model.device, dtype=adapter.model.dtype)
                for item in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(
                adapter.model.device, dtype=adapter.model.dtype
            )
        _, position_ids, attention_mask, _, inputs_embeds, _ = (
            adapter.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_tensor,
                image_sizes=[image.size],
            )
        )
        base_output = adapter.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        hidden = base_output.last_hidden_state[:, -1]
        vocabulary_weight = adapter.model.get_output_embeddings().weight
        vocabulary_logits = (
            hidden.to(vocabulary_weight.dtype) @ vocabulary_weight.T
        )[0]
        del base_output, input_ids, image_tensor, inputs_embeds

    class_logits = torch.stack(
        [vocabulary_logits[group].max() for group in groups]
    )
    log_probability = torch.log_softmax(vocabulary_logits.float(), dim=-1)
    sequence_nll = torch.stack(
        [-log_probability[group].max() for group in groups]
    )
    return (
        class_logits.float().cpu().numpy(),
        sequence_nll.cpu().numpy(),
    )


def eligible_rows(
    rows: list[dict], question_type: str, seed: int, max_samples: int
) -> list[dict]:
    selected = []
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
    selected.sort(
        key=lambda row: hashlib.sha256(f"{seed}:{row['qid']}".encode()).hexdigest()
    )
    return selected[:max_samples] if max_samples else selected


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def summarize(records: list[dict], fingerprint: str) -> dict:
    rows = [
        row
        for row in records
        if row.get("status") == "ok" and row.get("fingerprint") == fingerprint
    ]
    result = {"n": len(rows), "layouts": {}}
    if not rows:
        return result
    gt = np.asarray([row["gt_index"] for row in rows], dtype=np.int64)
    baseline = np.asarray(
        [row["layout_predictions"]["image_first"] for row in rows],
        dtype=np.int64,
    )
    baseline_correct = baseline == gt
    for layout in LAYOUTS:
        prediction = np.asarray(
            [row["layout_predictions"][layout] for row in rows],
            dtype=np.int64,
        )
        correct = prediction == gt
        result["layouts"][layout] = {
            "accuracy": float(correct.mean()),
            "delta_pp_vs_image_first": float(
                100.0 * (correct.mean() - baseline_correct.mean())
            ),
            "rescues": int(np.sum(~baseline_correct & correct)),
            "harms": int(np.sum(baseline_correct & ~correct)),
            "prediction_change_rate": float(np.mean(prediction != baseline)),
        }
    return result


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    config = {
        "version": VERSION,
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
        "layouts": list(LAYOUTS),
        "scoring": "max over equal surface-form sets",
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"metadata mismatch; choose a new output: {args.output}"
            )
    else:
        atomic_json(metadata_path, metadata)

    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    target = eligible_rows(
        rows, args.question_type, args.seed, args.max_samples
    )
    todo = [row for row in target if str(row["qid"]) not in saved]
    print(
        f"prompt-order fingerprint={fingerprint[:12]} "
        f"eligible={len(todo)} cached={len(saved)}",
        flush=True,
    )
    adapter = None
    errors = 0
    started = time.time()
    try:
        if todo:
            adapter = load_adapter(args.model)
            with args.output.open("a") as output:
                for sample in tqdm(todo, desc=f"prompt order {args.model}"):
                    try:
                        image_path = resolve_image(sample.get("img_name", ""))
                        assert image_path is not None
                        with Image.open(image_path) as source:
                            image = resize_image(source, args.max_image_side)
                        labels = labels_for_sample(sample)
                        prompt = build_prompt(sample)
                        logits = {}
                        nll = {}
                        predictions = {}
                        for layout in LAYOUTS:
                            layout_logits, layout_nll = forward_layout(
                                adapter,
                                args.model,
                                image,
                                prompt,
                                labels,
                                layout,
                            )
                            logits[layout] = layout_logits.tolist()
                            nll[layout] = layout_nll.tolist()
                            predictions[layout] = int(np.argmax(layout_logits))
                        record = {
                            "protocol_version": PROTOCOL_VERSION,
                            "cache_schema_version": CACHE_SCHEMA_VERSION,
                            "version": VERSION,
                            "fingerprint": fingerprint,
                            "status": "ok",
                            "qid": sample["qid"],
                            "img_name": sample.get("img_name", ""),
                            "question_type": task_kind(sample),
                            "labels": list(labels),
                            "gt_index": ground_truth_index(sample),
                            "layouts": list(LAYOUTS),
                            "layout_logits": logits,
                            "layout_sequence_nll": nll,
                            "layout_predictions": predictions,
                        }
                    except Exception as exc:
                        errors += 1
                        traceback.print_exc()
                        record = {
                            "fingerprint": fingerprint,
                            "status": "error",
                            "qid": sample.get("qid"),
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        }
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            gc.collect()
                            torch.cuda.empty_cache()
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    output.flush()
    finally:
        if adapter is not None:
            adapter.close()

    records = [
        json.loads(line)
        for line in args.output.read_text().splitlines()
        if line.strip()
    ]
    summary = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        **summarize(records, fingerprint),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "errors": errors,
    }
    atomic_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
