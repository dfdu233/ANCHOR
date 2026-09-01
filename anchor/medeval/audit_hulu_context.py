#!/usr/bin/env python3
"""Audit Hulu's expanded multimodal context for paired RAG prompts.

Hulu expands one image placeholder into a resolution-dependent visual-token
sequence before generation.  Its tokenizer advertises 16,384 tokens even
though the language model checkpoint supports a larger positional window, so
the tokenizer warning alone is neither a failure nor admissible evidence that
the run was safe.  This audit measures the actual expanded inputs and applies
the checkpoint's max_position_embeddings as the hard boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .audit_retrieval_split import read_rows
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "hulu-expanded-context-audit-v1"


def qid(row: dict, index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def summarize(
    measurements: list[dict],
    *,
    max_position_embeddings: int,
    tokenizer_model_max_length: int,
    max_new_tokens: int,
) -> dict:
    if max_position_embeddings <= 0 or max_new_tokens <= 0:
        raise ValueError("context and generation limits must be positive")
    paired = all(
        row["baseline"]["image_tokens"] == row["candidate"]["image_tokens"]
        for row in measurements
    )
    values = [value for row in measurements for value in row.values() if isinstance(value, dict)]
    for value in values:
        value["required_context_tokens"] = value["input_tokens"] + max_new_tokens
        value["model_context_overflow"] = (
            value["required_context_tokens"] > max_position_embeddings
        )
        value["tokenizer_metadata_warning"] = (
            value["input_tokens"] > tokenizer_model_max_length
        )
    result = {
        "protocol_version": VERSION,
        "n": len(measurements),
        "max_position_embeddings": max_position_embeddings,
        "tokenizer_model_max_length": tokenizer_model_max_length,
        "max_new_tokens": max_new_tokens,
        "paired_visual_token_count": paired,
        "maximum_input_tokens": max((v["input_tokens"] for v in values), default=0),
        "maximum_required_context_tokens": max(
            (v["required_context_tokens"] for v in values), default=0
        ),
        "model_context_overflow_count": sum(v["model_context_overflow"] for v in values),
        "tokenizer_metadata_warning_count": sum(
            v["tokenizer_metadata_warning"] for v in values
        ),
        "measurements": measurements,
    }
    result["passed"] = bool(
        measurements
        and paired
        and result["model_context_overflow_count"] == 0
        and all(v["input_tokens"] > 0 and v["image_tokens"] > 0 for v in values)
    )
    result["interpretation"] = (
        "tokenizer metadata warnings are diagnostic only; admission uses the "
        "checkpoint language model positional window after visual-token expansion"
    )
    return result


def measure(
    baseline: list[dict],
    candidate: list[dict],
    image_root: Path,
    model_root: Path,
    max_new_tokens: int,
) -> dict:
    from transformers import AutoProcessor

    if len(baseline) != len(candidate):
        raise ValueError("paired manifests have different lengths")
    processor = AutoProcessor.from_pretrained(
        str(model_root), trust_remote_code=True, local_files_only=True
    )
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    max_positions = int(config.get("max_position_embeddings", 0))
    tokenizer_limit = int(processor.tokenizer.model_max_length)
    measurements = []
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        sample_id = qid(left, index)
        if sample_id != qid(right, index):
            raise ValueError(f"paired qid mismatch at row {index}")
        left_image = str(left.get("img_name", left.get("image", "")))
        right_image = str(right.get("img_name", right.get("image", "")))
        if not left_image or left_image != right_image:
            raise ValueError(f"paired image mismatch for qid {sample_id}")
        with Image.open(image_root / left_image) as source:
            image = source.convert("RGB")
        row = {"qid": sample_id, "image": left_image}
        for name, item in (("baseline", left), ("candidate", right)):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": str(item["question"])},
                    ],
                }
            ]
            inputs = processor(
                images=[image],
                conversation=conversation,
                add_system_prompt=False,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"][0]
            image_tokens = int((input_ids == processor.image_token_id).sum().item())
            row[name] = {
                "input_tokens": int(input_ids.numel()),
                "image_tokens": image_tokens,
                "nonimage_tokens": int(input_ids.numel()) - image_tokens,
            }
        measurements.append(row)
    result = summarize(
        measurements,
        max_position_embeddings=max_positions,
        tokenizer_model_max_length=tokenizer_limit,
        max_new_tokens=max_new_tokens,
    )
    result.update(
        {
            "baseline_manifest_sha256": None,
            "candidate_manifest_sha256": None,
            "model_config_sha256": sha256_file(model_root / "config.json"),
            "model_root": str(model_root.resolve()),
            "image_root": str(image_root.resolve()),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = read_rows(args.baseline_manifest), read_rows(args.candidate_manifest)
    if args.limit:
        baseline, candidate = baseline[: args.limit], candidate[: args.limit]
    result = measure(
        baseline, candidate, args.image_root, args.model_root, args.max_new_tokens
    )
    result["baseline_manifest_sha256"] = sha256_file(args.baseline_manifest)
    result["candidate_manifest_sha256"] = sha256_file(args.candidate_manifest)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
