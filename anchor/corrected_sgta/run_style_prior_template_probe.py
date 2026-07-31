"""Probe whether style-conditioned clinical evidence survives rephrasing.

The probe keeps images, style views, diseases, and complete-sentence scoring
fixed while replacing both the question frame and the positive/negative answer
frame.  It is a diagnostic only: teacher-forced likelihoods are never used as
predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from anchor.corrected_sgta.run_visual_evidence_chord_probe import (
    CONDITIONS,
    build_views,
    read_jsonl,
    sequence_nll,
    sha256,
    unique_cases,
)


VERSION = "style-prior-template-probe-v1"
TEMPLATES = {
    "evidence": {
        "question": (
            "Assess this chest radiograph for {condition}. Determine whether "
            "the finding is present or absent, and answer in one complete "
            "sentence based only on the image."
        ),
        "positive": "Radiographic evidence of {condition} is present.",
        "negative": "Radiographic evidence of {condition} is absent.",
    },
    "demonstrates": {
        "question": (
            "Does this chest radiograph demonstrate {condition}? Give one "
            "complete affirmative or negative sentence based only on the "
            "visual evidence."
        ),
        "positive": "This chest radiograph demonstrates {condition}.",
        "negative": "This chest radiograph demonstrates no {condition}.",
    },
}


def render_template(template_id: str, condition: str) -> dict[str, str]:
    """Render a frozen semantically opposed complete-sentence template."""
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown template: {template_id}")
    template = TEMPLATES[template_id]
    return {
        key: value.format(condition=condition)
        for key, value in template.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--style-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--templates",
        nargs="+",
        choices=sorted(TEMPLATES),
        default=sorted(TEMPLATES),
    )
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.12)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = unique_cases(args.questions, args.image_manifest, args.limit)
    prototypes = [
        row
        for row in read_jsonl(args.style_manifest)
        if int(row["replicate"]) == 0
    ]
    if len(prototypes) < 3:
        raise RuntimeError("style manifest must contain at least three clusters")

    config_sha = sha256(args.model / "config.json")
    template_payload = {
        template_id: TEMPLATES[template_id]
        for template_id in args.templates
    }
    pending: list[dict] = []
    metadata: dict[tuple[str, str], dict] = {}
    for case in cases:
        views, view_metrics = build_views(
            case,
            prototypes,
            args.radius,
            args.strength,
            args.output,
        )
        for view_name, image in views.items():
            metadata[(case["case_id"], view_name)] = view_metrics[view_name]
            for disease, condition in CONDITIONS.items():
                for template_id in args.templates:
                    rendered = render_template(template_id, condition)
                    for polarity in ("positive", "negative"):
                        pending.append(
                            {
                                **case,
                                "view": view_name,
                                "image": image,
                                "disease": disease,
                                "condition": condition,
                                "template_id": template_id,
                                "polarity": polarity,
                                "question": rendered["question"],
                                "answer": rendered[polarity],
                            }
                        )

    expected = {
        "version": VERSION,
        "model_config_sha256": config_sha,
        "questions_sha256": sha256(args.questions),
        "image_manifest_sha256": sha256(args.image_manifest),
        "style_manifest_sha256": sha256(args.style_manifest),
        "templates": template_payload,
        "radius": args.radius,
        "strength": args.strength,
    }
    existing_keys: set[tuple[str, str, str, str, str]] = set()
    if args.resume and args.output.exists():
        for record in read_jsonl(args.output):
            for field, value in expected.items():
                if record.get(field) != value:
                    raise RuntimeError(
                        f"resume fingerprint mismatch for {field}: "
                        f"{record.get(field)!r} != {value!r}"
                    )
            existing_keys.add(
                (
                    record["case_id"],
                    record["view"],
                    record["disease"],
                    record["template_id"],
                    record["polarity"],
                )
            )
        pending = [
            row
            for row in pending
            if (
                row["case_id"],
                row["view"],
                row["disease"],
                row["template_id"],
                row["polarity"],
            )
            not in existing_keys
        ]
    if not pending:
        print(
            json.dumps(
                {
                    "completed": len(existing_keys),
                    "remaining": 0,
                    "output": str(args.output),
                }
            )
        )
        return

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()
    mode = "a" if args.resume and existing_keys else "w"
    with args.output.open(mode) as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            scores = sequence_nll(model, processor, batch)
            for row, (nll, token_count) in zip(batch, scores, strict=True):
                record = {
                    **expected,
                    "case_id": row["case_id"],
                    "image_relative": row["image_relative"],
                    "source_question_id": row["source_question_id"],
                    "view": row["view"],
                    "disease": row["disease"],
                    "condition": row["condition"],
                    "template_id": row["template_id"],
                    "polarity": row["polarity"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sequence_nll": nll,
                    "answer_token_count": token_count,
                    "image_metrics": metadata[
                        (row["case_id"], row["view"])
                    ],
                    "model": str(args.model.resolve()),
                }
                handle.write(json.dumps(record) + "\n")
            print(
                json.dumps(
                    {
                        "completed": min(start + len(batch), len(pending)),
                        "remaining_total": len(pending),
                        "previously_completed": len(existing_keys),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
