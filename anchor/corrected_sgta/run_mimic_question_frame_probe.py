"""Generate HuatuoGPT answers for paired MIMIC question frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from anchor.corrected_sgta.evaluate_medheval_answers import (
    parse_answer,
    rule_pope_prediction,
)
from anchor.corrected_sgta.run_center_native_qwen import (
    messages_for,
    pad_392,
)


VERSION = "mimic-question-frame-probe-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def clean_question(question: str) -> str:
    return question.replace("<image>", "").strip()


def framed_questions(question: str) -> dict[str, str]:
    clean = clean_question(question)
    return {
        "original": f"{clean} Answer in one complete sentence.",
        "neutral": (
            "Independently assess the image without assuming that the finding "
            f"named in the question is present. {clean} State the conclusion "
            "in one complete sentence."
        ),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(args.questions)[: args.limit]
    images = {
        row["relative_path"]: row["path_in_repo"]
        for row in read_jsonl(args.image_manifest)
    }
    done = (
        {row["id"]: row for row in read_jsonl(args.output)}
        if args.output.exists()
        else {}
    )
    records = []
    for row in questions:
        for frame, question in framed_questions(row["question"]).items():
            identifier = f"{row['question_id']}::{frame}"
            if identifier in done:
                continue
            image = Path(images[row["image"]])
            if not image.is_absolute():
                image = Path.cwd() / image
            records.append(
                {
                    "id": identifier,
                    "question_id": str(row["question_id"]),
                    "frame": frame,
                    "question": question,
                    "ground_truth": str(row["answer"]),
                    "image_relative": row["image"],
                    "image": image,
                }
            )

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()
    config_sha = sha256(args.model / "config.json")
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts = [
            processor.apply_chat_template(
                messages_for(row["question"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        batch_images = [pad_392(row["image"]) for row in batch]
        inputs = processor(
            text=prompts,
            images=batch_images,
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
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(batch, texts, strict=True):
            text = text.strip()
            explicit = parse_answer(text, answer_type="binary")
            record = {
                "version": VERSION,
                "id": row["id"],
                "question_id": row["question_id"],
                "frame": row["frame"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "image_relative": row["image_relative"],
                "text": text,
                "rule_prediction": rule_pope_prediction(text),
                "explicit_prediction": (
                    explicit.labels[0] if explicit.labels else None
                ),
                "model": str(args.model.resolve()),
                "model_config_sha256": config_sha,
                "questions_sha256": sha256(args.questions),
                "image_manifest_sha256": sha256(args.image_manifest),
            }
            with args.output.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            done[row["id"]] = record
        print(
            json.dumps(
                {"completed": len(done), "total": args.limit * 2}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
