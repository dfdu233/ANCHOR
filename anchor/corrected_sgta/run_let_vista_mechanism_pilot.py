#!/usr/bin/env python3
"""Small controlled comparison of LET and VISTA-style self-logit augmentation.

Every method uses the same RULE prompt, image, greedy full-sentence decoding,
token budget, and deterministic parser.  The variants change only whether the
intermediate state is final-normalized and whether one layer or a layer window
is transported.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.anchor_transport import resolve_image_path, stable_json_sha256
from corrected_sgta.models_oe import Generation, LlavaMedOEAdapter
from corrected_sgta.run_anchor_let_rule75 import (
    evaluate_sentence,
    load_jsonl,
    patient_id,
    sha256_file,
)
from corrected_sgta.rule_mitigation_backend import (
    render_rule_model_prompt,
    render_rule_prompt,
)

VERSION = "let-vista-mechanism-pilot-v1"


@torch.inference_mode()
def generate_variant(
    adapter: LlavaMedOEAdapter,
    image: Image.Image,
    prompt: str,
    *,
    variant: str,
    alpha: float,
    expert_layer: int,
    vista_start: int,
    vista_end: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[Generation, dict[str, float]]:
    from transformers import set_seed
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )

    set_seed(seed)
    model_prompt = render_rule_model_prompt(
        prompt,
        image_token=DEFAULT_IMAGE_TOKEN,
        image_start_token=DEFAULT_IM_START_TOKEN,
        image_end_token=DEFAULT_IM_END_TOKEN,
        use_image_start_end=bool(adapter.model.config.mm_use_im_start_end),
    )
    input_ids = tokenizer_image_token(
        model_prompt,
        adapter.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(adapter.model.device)
    attention_mask = torch.ones_like(input_ids)
    image_tensor = process_images(
        [image], adapter.image_processor, adapter.model.config
    )[0].unsqueeze(0).to(adapter.model.device, dtype=adapter.model.dtype)
    stopping = KeywordsStoppingCriteria(["</s>"], adapter.tokenizer, input_ids)

    original_forward = adapter.model.forward
    diagnostics: list[dict[str, float]] = []

    if variant != "greedy":
        @functools.wraps(original_forward)
        def mixed_forward(_self, *forward_args: Any, **forward_kwargs: Any):
            forward_kwargs["output_hidden_states"] = True
            output = original_forward(*forward_args, **forward_kwargs)
            hidden = output.hidden_states
            final_logits = output.logits.float()

            if variant.startswith("let_"):
                index = expert_layer if expert_layer >= 0 else len(hidden) + expert_layer
                selected = hidden[max(0, min(index, len(hidden) - 1))]
                if variant == "let_norm":
                    selected = adapter.model.model.norm(selected)
                augmented = adapter.model.lm_head(
                    selected.to(adapter.model.lm_head.weight.dtype)
                ).float()
            else:
                # VISTA ignores the embedding state and indexes the remaining
                # transformer-layer states with the inclusive [start, end]
                # interval.  Its released SLA applies lm_head without the
                # model's final RMSNorm.
                layer_states = list(hidden[1:])
                chosen = layer_states[vista_start : vista_end + 1]
                if variant == "vista_norm_window":
                    chosen = [adapter.model.model.norm(state) for state in chosen]
                augmented = torch.stack(
                    [
                        adapter.model.lm_head(
                            state.to(adapter.model.lm_head.weight.dtype)
                        ).float()
                        for state in chosen
                    ]
                ).mean(dim=0)

            mixed = (1.0 - alpha) * final_logits + alpha * augmented
            # Centering removes the softmax-invariant additive direction.
            final_centered = final_logits[:, -1] - final_logits[:, -1].mean(-1, keepdim=True)
            aug_centered = augmented[:, -1] - augmented[:, -1].mean(-1, keepdim=True)
            perturb = mixed[:, -1] - final_logits[:, -1]
            diagnostics.append(
                {
                    "aug_to_final_centered_norm": float(
                        aug_centered.norm() / final_centered.norm().clamp_min(1e-8)
                    ),
                    "relative_perturbation_norm": float(
                        perturb.norm() / final_centered.norm().clamp_min(1e-8)
                    ),
                    "top1_agreement": float(
                        (augmented[:, -1].argmax(-1) == final_logits[:, -1].argmax(-1))
                        .float()
                        .mean()
                    ),
                }
            )
            output.logits = mixed
            return output

        adapter.model.forward = types.MethodType(mixed_forward, adapter.model)

    try:
        output_ids = adapter.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=adapter.tokenizer.eos_token_id,
            images=image_tensor,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping],
            do_sample=False,
        )
    finally:
        adapter.model.forward = original_forward

    text = adapter.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    summary = {
        key: float(np.mean([row[key] for row in diagnostics]))
        for key in (
            "aug_to_final_centered_norm",
            "relative_perturbation_norm",
            "top1_agreement",
        )
    } if diagnostics else {}
    return (
        Generation(
            text=text,
            uncertainty=float("nan"),
            token_count=int(output_ids.shape[-1]),
        ),
        summary,
    )


def summarize(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    greedy = [bool(row["methods"]["greedy"]["eval"]["rule_normalized_correct"]) for row in rows]
    output: dict[str, Any] = {"version": VERSION, "n": len(rows), "methods": {}}
    for method in methods:
        payloads = [row["methods"][method] for row in rows]
        correct = [bool(item["eval"]["rule_normalized_correct"]) for item in payloads]
        confusion: Counter[str] = Counter(
            f"{item['eval']['ground_truth']}->{item['eval']['rule_normalized_prediction']}"
            for item in payloads
        )
        tp, tn = confusion["yes->yes"], confusion["no->no"]
        fp, fn = confusion["no->yes"], confusion["yes->no"]
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        diagnostics = [item.get("diagnostics", {}) for item in payloads]
        keys = sorted({key for item in diagnostics for key in item})
        output["methods"][method] = {
            "accuracy": float(np.mean(correct)),
            "balanced_accuracy": float((sensitivity + specificity) / 2),
            "rescue_vs_greedy": int(sum((not b) and c for b, c in zip(greedy, correct))),
            "harm_vs_greedy": int(sum(b and (not c) for b, c in zip(greedy, correct))),
            "output_change_rate": float(
                np.mean(
                    [
                        item["text"] != row["methods"]["greedy"]["text"]
                        for item, row in zip(payloads, rows)
                    ]
                )
            ),
            "mean_words": float(np.mean([len(item["text"].split()) for item in payloads])),
            "diagnostics": {
                key: float(np.mean([item[key] for item in diagnostics if key in item]))
                for key in keys
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/rule/test/mimic_test.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/medheval/images"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corrected_runs/final_let_vista_mechanism_pilot_v1"),
    )
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--expert-layer", type=int, default=-12)
    parser.add_argument("--vista-start", type=int, default=25)
    parser.add_argument("--vista-end", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    methods = [
        "greedy",
        "let_norm",
        "let_unnorm",
        "vista_exact_window",
        "vista_norm_window",
    ]
    all_questions = load_jsonl(args.manifest)
    questions = all_questions[
        args.start_index : args.start_index + args.max_samples
    ]
    fingerprint = stable_json_sha256(
        {
            "version": VERSION,
            "manifest_sha256": sha256_file(args.manifest),
            "methods": methods,
            "conv_mode": "vicuna_v1",
            "max_samples": args.max_samples,
            "start_index": args.start_index,
            "max_new_tokens": args.max_new_tokens,
            "alpha": args.alpha,
            "expert_layer": args.expert_layer,
            "vista_window": [args.vista_start, args.vista_end],
            "seed": args.seed,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    summary_path = args.output_dir / "summary.json"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and raw_path.exists():
        for row in load_jsonl(raw_path):
            if row.get("fingerprint") != fingerprint:
                raise ValueError("resume fingerprint mismatch")
            existing[str(row["question_id"])] = row

    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    records = list(existing.values())
    with raw_path.open("a" if args.resume else "w") as handle:
        for index, question in enumerate(tqdm(questions, desc="LET-vs-VISTA")):
            qid = str(question["question_id"])
            if qid in existing:
                continue
            prompt = render_rule_prompt("mimic", question)
            with Image.open(resolve_image_path(str(question["image"]), args.image_root)) as source:
                image = source.convert("RGB")
            payload: dict[str, Any] = {}
            for method in methods:
                generated, diagnostics = generate_variant(
                    adapter,
                    image,
                    prompt,
                    variant=method,
                    alpha=args.alpha,
                    expert_layer=args.expert_layer,
                    vista_start=args.vista_start,
                    vista_end=args.vista_end,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed + index,
                )
                payload[method] = {
                    "text": generated.text,
                    "token_count": generated.token_count,
                    "diagnostics": diagnostics,
                    "eval": evaluate_sentence(generated.text, question["answer"]),
                }
            record = {
                "version": VERSION,
                "fingerprint": fingerprint,
                "question_id": qid,
                "patient_id": patient_id(str(question["image"])),
                "image": question["image"],
                "question": question["question"],
                "reference": question["answer"],
                "methods": payload,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    report = summarize(records, methods)
    report["fingerprint"] = fingerprint
    report["raw"] = str(raw_path)
    report["status"] = "complete"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
