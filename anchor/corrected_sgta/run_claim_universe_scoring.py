#!/usr/bin/env python3
"""Score a grouped medical claim universe with optional uniform-null controls."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageStat
import torch

from corrected_sgta.evaluate_medheval_answers import parse_answer
from corrected_sgta.run_huatuo_rule_feddg import generate_with_nll
from corrected_sgta.run_huatuo_vindr_commitment_probe import import_huatuo, sha256_file
from corrected_sgta.run_hulu_vindr_commitment_probe import HuluRuntime
from corrected_sgta.run_llava_vindr_commitment_probe import LlavaRuntime
from corrected_sgta.run_slake_quantifier_coverage_probe import (
    score_huatuo,
    score_hulu,
    score_llava,
)


VERSION = "claim-universe-original-null-scoring-v3"
PROMPT = (
    "{question}\nAnswer based only on the medical image. "
    "Answer exactly Yes, No, or Maybe."
)


def margin(score: dict) -> float:
    return float(score["logits"]["supported"] - score["logits"]["refuted"])


@torch.inference_mode()
def generate_hulu(runtime: HuluRuntime, image: Image.Image, prompt: str) -> dict:
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    inputs = runtime.processor(
        images=[image],
        conversation=conversation,
        add_system_prompt=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if key == "pixel_values":
                value = value.to(dtype=runtime.model.dtype)
            inputs[key] = value.to(runtime.model.device)
    output_ids = runtime.model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=8,
        use_cache=True,
        pad_token_id=runtime.tokenizer.eos_token_id,
    )
    text = runtime.processor.batch_decode(
        output_ids, skip_special_tokens=True, use_think=False
    )[0].strip()
    return {"text": text, "generated_token_count": int(output_ids.shape[-1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("llava_med", "huatuo", "hulu"), default="llava_med")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument(
        "--bare-question",
        action="store_true",
        help="Score the exact source question without appending answer instructions.",
    )
    parser.add_argument(
        "--generate-draft",
        action="store_true",
        help="Also generate a greedy Yes/No/Maybe draft (Huatuo or Hulu).",
    )
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help=(
            "Score only the original image. Use this for preregistered raw-evidence "
            "experiments after the uniform-null diagnostic has been ruled out."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible partial raw.jsonl and retry only unfinished/error rows.",
    )
    parser.add_argument(
        "--draft-bare-question",
        action="store_true",
        help="Use the exact source question for draft generation while keeping the canonical scoring prompt.",
    )
    parser.add_argument(
        "--draft-answer-type",
        choices=("binary", "ternary"),
        default="binary",
        help="Strict parser used only for the optional generated draft.",
    )
    parser.add_argument(
        "--llava-root",
        type=Path,
        default=Path(
            "/home/dbw/ANCHOR/data/medheval/code/baselines/"
            "Med-LVLMs/llava-med-1.5"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    rows = json.loads(args.questions.read_text())
    default_paths = {
        "llava_med": Path("/home/dbw/models/LLaVA-Med-v1.5-mistral-7b"),
        "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
        "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
    }
    model_path = args.model_path or default_paths[args.model]
    if args.generate_draft and args.model not in {"huatuo", "hulu"}:
        raise ValueError("--generate-draft is implemented for Huatuo and Hulu")
    prompt_template = "{question}" if args.bare_question else PROMPT
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "image_root": str(args.image_root.resolve()),
        "model": args.model,
        "model_path": str(model_path.resolve()),
        "prompt": prompt_template,
        "null": (
            "not computed"
            if args.skip_null
            else "uniform RGB image using the per-image channel mean"
        ),
        "code_sha256": sha256_file(Path(__file__)),
        "claim_ceiling": (
            "The uniform image is a diagnostic visual null, not a causal absence "
            "of every image token and not a reader-disagreement target."
        ),
    }
    config_path = args.output_dir / "config.json"
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"resume requires {config_path}")
        prior = json.loads(config_path.read_text())
        stable_keys = (
            "questions_sha256", "image_root", "model", "model_path", "prompt", "null"
        )
        mismatches = {
            key: {"prior": prior.get(key), "current": config.get(key)}
            for key in stable_keys
            if prior.get(key) != config.get(key)
        }
        if mismatches:
            raise ValueError(f"incompatible resume config: {mismatches}")
        config = prior
        config.setdefault("resume_code_sha256", []).append(sha256_file(Path(__file__)))
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if args.model == "llava_med":
        runtime = LlavaRuntime(model_path, args.llava_root, "mistral_instruct")
        scorer = score_llava
    elif args.model == "huatuo":
        constructor = import_huatuo(Path("/home/dbw/HuatuoGPT-Vision"))
        runtime = constructor(str(model_path), device="cuda:0")
        scorer = score_huatuo
    else:
        runtime = HuluRuntime(model_path, args.max_visual_tokens)
        scorer = score_hulu
    raw_path = args.output_dir / "raw.jsonl"
    completed_qids: set[int] = set()
    if args.resume and raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            prior_record = json.loads(line)
            if prior_record.get("status") == "ok":
                completed_qids.add(int(prior_record["question_id"]))
    completed = len(completed_qids)
    for index, row in enumerate(rows):
        if int(row["qid"]) in completed_qids:
            print(
                f"[{index + 1}/{len(rows)}] qid={row['qid']} cached",
                flush=True,
            )
            continue
        record = {
            "version": VERSION,
            "question_id": int(row["qid"]),
            "image": str(row["img_name"]),
            "question": str(row["question"]),
            "truth": str(row["answer"]),
            "status": "error",
        }
        try:
            path = args.image_root / row["img_name"]
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            prompt = prompt_template.format(question=row["question"])
            original_score = scorer(runtime, image, prompt)
            record.update({
                "status": "ok",
                "image_sha256": sha256_file(path),
                "original": original_score,
                "scores": {
                    "original_margin": margin(original_score),
                },
            })
            if not args.skip_null:
                means = tuple(round(value) for value in ImageStat.Stat(image).mean[:3])
                null = Image.new("RGB", image.size, means)
                null_score = scorer(runtime, null, prompt)
                record["uniform_null"] = null_score
                record["scores"].update({
                    "null_margin": margin(null_score),
                    "null_centered_margin": margin(original_score) - margin(null_score),
                })
            if args.generate_draft:
                draft_prompt = str(row["question"]) if args.draft_bare_question else prompt
                if args.model == "huatuo":
                    draft = generate_with_nll(
                        runtime, draft_prompt, image,
                        max_new_tokens=8, repetition_penalty=1.0,
                    )
                else:
                    draft = generate_hulu(runtime, image, draft_prompt)
                parsed = parse_answer(
                    draft["text"], answer_type=args.draft_answer_type
                )
                record["draft"] = {
                    **draft,
                    "prediction": parsed.labels[0] if parsed.labels else "invalid",
                    "parse_status": parsed.status,
                }
            completed += 1
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
        with raw_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[{index + 1}/{len(rows)}] qid={row['qid']} {record['status']}", flush=True)
    summary = {
        "version": VERSION,
        "status": "complete" if completed == len(rows) else "partial",
        "n": len(rows),
        "completed": completed,
        "errors": len(rows) - completed,
        "config": config,
        "raw_sha256": sha256_file(raw_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
