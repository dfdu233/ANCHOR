#!/usr/bin/env python3
"""Trace prompt, image preprocessing, and first-token logits for LLaVA paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from anchor.medeval.hashing import sha256_file
from anchor.medeval.store import atomic_write_json


VERSION = "llava-backend-trace-v1"


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().float().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "minimum": float(cpu.min().item()),
        "maximum": float(cpu.max().item()),
        "mean": float(cpu.mean().item()),
        "sha256_float32": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }


def trace(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.backend_root.resolve()))
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model_path),
        None,
        get_model_name_from_path(str(args.model_path)),
        device_map="auto",
        load_8bit=False,
        load_4bit=False,
    )
    model.eval()
    rows = json.loads(args.manifest.read_text())
    wanted = set(args.qids)
    selected = [row for row in rows if str(row.get("qid", row.get("id"))) in wanted]
    if {str(row.get("qid", row.get("id"))) for row in selected} != wanted:
        raise ValueError("one or more requested qids are absent from the manifest")
    traces = []
    try:
        for row in selected:
            item_id = str(row.get("qid", row.get("id")))
            image_path = args.image_root / str(row["img_name"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            image_token = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, "mm_use_im_start_end", False):
                image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            conversation = conv_templates[args.conv_mode].copy()
            conversation.append_message(
                conversation.roles[0], image_token + "\n" + str(row["question"])
            )
            conversation.append_message(conversation.roles[1], None)
            rendered_prompt = conversation.get_prompt()
            input_ids = tokenizer_image_token(
                rendered_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(model.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [item.to(model.device, dtype=model.dtype) for item in image_tensor]
                summary = [tensor_summary(item) for item in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=model.dtype)
                summary = tensor_summary(image_tensor)
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            logits = output.scores[0][0].float()
            values, indices = torch.topk(logits, k=10)
            generated = output.sequences[:, -len(output.scores) :]
            generated_token_ids = [int(value) for value in generated[0].tolist()]
            step_top = []
            for step, step_scores in enumerate(output.scores):
                step_values, step_indices = torch.topk(step_scores[0].float(), k=2)
                step_top.append(
                    {
                        "step": step,
                        "top1_token_id": int(step_indices[0]),
                        "top1_logit": float(step_values[0]),
                        "top2_token_id": int(step_indices[1]),
                        "top2_logit": float(step_values[1]),
                        "margin": float(step_values[0] - step_values[1]),
                    }
                )
            traces.append(
                {
                    "question_id": item_id,
                    "question": str(row["question"]),
                    "rendered_prompt": rendered_prompt,
                    "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode()).hexdigest(),
                    "input_ids": [int(value) for value in input_ids[0].tolist()],
                    "input_ids_sha256": hashlib.sha256(
                        input_ids.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "image_path": str(image_path.resolve()),
                    "image_sha256": sha256_file(image_path),
                    "image_mode": image.mode,
                    "image_size": list(image.size),
                    "processed_image": summary,
                    "first_token_topk": [
                        {
                            "token_id": int(token_id),
                            "token": tokenizer.decode([int(token_id)]),
                            "logit": float(logit),
                        }
                        for logit, token_id in zip(values.tolist(), indices.tolist())
                    ],
                    "first_generated_token_id": generated_token_ids[0],
                    "generated_token_ids": generated_token_ids,
                    "generated_text": tokenizer.decode(
                        generated_token_ids, skip_special_tokens=True
                    ).strip(),
                    "step_top": step_top,
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_write_json(
        args.output,
        {
            "protocol_version": VERSION,
            "backend_root": str(args.backend_root.resolve()),
            "model_path": str(args.model_path.resolve()),
            "conv_mode": args.conv_mode,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "traces": traces,
        },
    )


def compare(args: argparse.Namespace) -> None:
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    left_rows = {row["question_id"]: row for row in left["traces"]}
    right_rows = {row["question_id"]: row for row in right["traces"]}
    fields = (
        "rendered_prompt_sha256",
        "input_ids_sha256",
        "image_sha256",
        "image_mode",
        "image_size",
        "processed_image",
        "first_generated_token_id",
        "generated_token_ids",
    )
    rows = []
    for item_id in sorted(set(left_rows) | set(right_rows)):
        a, b = left_rows.get(item_id), right_rows.get(item_id)
        equality = {
            field: bool(a is not None and b is not None and a.get(field) == b.get(field))
            for field in fields
        }
        rows.append(
            {
                "question_id": item_id,
                "field_equal": equality,
                "first_divergent_field": next(
                    (field for field in fields if not equality[field]), None
                ),
                "first_divergent_token_step": next(
                    (
                        index
                        for index, (left_id, right_id) in enumerate(
                            zip(
                                [] if a is None else a.get("generated_token_ids", []),
                                [] if b is None else b.get("generated_token_ids", []),
                            )
                        )
                        if left_id != right_id
                    ),
                    None,
                ),
                "left_first_token_topk": None if a is None else a["first_token_topk"],
                "right_first_token_topk": None if b is None else b["first_token_topk"],
            }
        )
    atomic_write_json(
        args.output,
        {
            "protocol_version": VERSION,
            "left": str(args.left.resolve()),
            "right": str(args.right.resolve()),
            "rows": rows,
            "all_equal": all(
                all(row["field_equal"].values()) for row in rows
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("trace")
    run.add_argument("--backend-root", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--image-root", type=Path, required=True)
    run.add_argument("--qids", nargs="+", required=True)
    run.add_argument("--conv-mode", default="mistral_instruct")
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--output", type=Path, required=True)
    diff = sub.add_parser("compare")
    diff.add_argument("--left", type=Path, required=True)
    diff.add_argument("--right", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace(args) if args.command == "trace" else compare(args)


if __name__ == "__main__":
    main()
