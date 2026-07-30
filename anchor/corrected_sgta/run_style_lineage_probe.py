"""Generate full-sentence disease answers on fixed-content style prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from anchor.corrected_sgta.evaluate_medheval_answers import (
    parse_answer,
    rule_pope_prediction,
)
from anchor.corrected_sgta.run_center_native_qwen import messages_for


VERSION = "style-lineage-probe-v1"
QUESTIONS = {
    "pneumothorax": "Is there evidence of pneumothorax in this chest radiograph? Answer in one complete sentence.",
    "effusion": "Is there evidence of pleural effusion in this chest radiograph? Answer in one complete sentence.",
    "opacity": "Is there evidence of pulmonary opacity or consolidation in this chest radiograph? Answer in one complete sentence.",
    "cardiomegaly": "Is there evidence of cardiomegaly in this chest radiograph? Answer in one complete sentence.",
    "edema": "Is there evidence of pulmonary edema in this chest radiograph? Answer in one complete sentence.",
    "device": "Is there an indwelling medical device in this chest radiograph? Answer in one complete sentence.",
}
CONDITION_NAMES = {
    "pneumothorax": "pneumothorax",
    "effusion": "pleural effusion",
    "opacity": "pulmonary opacity or consolidation",
    "cardiomegaly": "cardiomegaly",
    "edema": "pulmonary edema",
    "device": "an indwelling medical device",
}
REPORT_PROMPT = (
    "Write a complete radiology report for this chest radiograph, including "
    "findings and impression. Do not invent details not supported by the image."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--task", choices=("binary", "report"), default="binary"
    )
    parser.add_argument(
        "--binary-frame",
        choices=("positive", "neutral", "negative", "contrast"),
        default="positive",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prototypes = read_jsonl(args.manifest)
    done = {}
    if args.output.exists():
        done = {row["id"]: row for row in read_jsonl(args.output)}

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()
    processor.tokenizer.padding_side = "left"
    config_sha = sha256(args.model / "config.json")
    if args.task == "report":
        question_items = [("report", "report", REPORT_PROMPT)]
    elif args.binary_frame == "positive":
        question_items = [
            (disease, "positive", question)
            for disease, question in QUESTIONS.items()
        ]
    else:
        neutral_questions = [
            (
                disease,
                "neutral",
                "Determine from this chest radiograph whether "
                f"{condition} is present or absent, and state the conclusion "
                "in one complete sentence.",
            )
            for disease, condition in CONDITION_NAMES.items()
        ]
        negative_questions = [
            (
                disease,
                "negative",
                f"This chest radiograph does not show {condition}, correct? "
                "Answer in one complete sentence.",
            )
            for disease, condition in CONDITION_NAMES.items()
        ]
        question_items = (
            neutral_questions
            if args.binary_frame == "neutral"
            else negative_questions
            if args.binary_frame == "negative"
            else neutral_questions + negative_questions
        )
    for prototype in prototypes:
        with Image.open(prototype["image"]) as handle:
            image = handle.convert("RGB")
        pending = [
            (disease, frame, question)
            for disease, frame, question in question_items
            if (
                f"{prototype['id']}::{disease}"
                if len({item[1] for item in question_items}) == 1
                else f"{prototype['id']}::{disease}::{frame}"
            )
            not in done
        ]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [
                processor.apply_chat_template(
                    messages_for(question),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for _, _, question in batch
            ]
            inputs = processor(
                text=prompts,
                images=[image] * len(batch),
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            generated = output[:, inputs["input_ids"].shape[1] :]
            texts = processor.batch_decode(
                generated, skip_special_tokens=True
            )
            for (disease, frame, question), text in zip(
                batch, texts, strict=True
            ):
                text = text.strip()
                identifier = (
                    f"{prototype['id']}::{disease}"
                    if len({item[1] for item in question_items}) == 1
                    else f"{prototype['id']}::{disease}::{frame}"
                )
                explicit = (
                    parse_answer(text, answer_type="binary")
                    if args.task == "binary"
                    else None
                )
                record = {
                    "version": VERSION,
                    "id": identifier,
                    "condition": args.condition,
                    "prototype_id": prototype["id"],
                    "cluster": prototype["cluster"],
                    "replicate": prototype["replicate"],
                    "disease": disease,
                    "task": args.task,
                    "prompt_frame": frame,
                    "question": question,
                    "text": text,
                    "rule_prediction": (
                        rule_pope_prediction(text)
                        if args.task == "binary"
                        else None
                    ),
                    "explicit_prediction": (
                        explicit.labels[0]
                        if explicit is not None and explicit.labels
                        else None
                    ),
                    "model": str(args.model.resolve()),
                    "model_config_sha256": config_sha,
                    "prototype_manifest_sha256": sha256(args.manifest),
                }
                with args.output.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                done[identifier] = record
                if len(done) % 18 == 0:
                    print(
                        json.dumps(
                            {
                                "condition": args.condition,
                                "completed": len(done),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
