"""Resumable paired base/adapted RULE inference for the sequence DG adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.train_rule_dg_adapter import (
    VERSION, BoundedResidualBottleneck, attach_postprojector_adapter,
    attach_preprojector_adapter, file_sha256, rule_no_reference_prompt,
)
from corrected_sgta.train_rule_source_group_adapter import (
    VERSION as SOURCE_GROUP_VERSION,
)
from corrected_sgta.train_anchor_dg import VERSION as ANCHOR_DG_VERSION

INFERENCE_VERSION = "rule-sequence-dg-inference-v2"
RULE_MIMIC_NO_REFERENCE_SUFFIX = (
    "Please answer the question based on the image and report and choose from "
    "the following two options: [yes, no]."
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    qids = [str(row.get("question_id", row.get("qid"))) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate question ids")
    return rows


def repair_jsonl_tail(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    boundary = data.rfind(b"\n")
    path.write_bytes(data[: boundary + 1] if boundary >= 0 else b"")


def successful_qids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row["question_id"])
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        if row.get("status") == "ok"
    }


def cached_base_answers(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_rows(path)
    output = {}
    for row in rows:
        qid = str(row.get("question_id", row.get("qid")))
        if qid in output:
            raise ValueError(f"duplicate cached baseline qid={qid}")
        output[qid] = row
    return output


def official_prompt(
    row: dict[str, Any], prompt_protocol: str = "no_reference"
) -> str:
    question = str(row.get("question", row.get("text", ""))).replace("<image>", "").strip()
    if not question:
        raise ValueError("missing RULE question")
    if prompt_protocol not in {"no_reference", "rule_mimic"}:
        raise ValueError(f"unknown prompt protocol: {prompt_protocol}")
    reference = row.get("reference_report")
    if reference is None:
        if prompt_protocol == "rule_mimic":
            return question + " " + RULE_MIMIC_NO_REFERENCE_SUFFIX
        return rule_no_reference_prompt(question)
    reports = reference if isinstance(reference, list) else [reference]
    formatted = str(reports[0]) if len(reports) == 1 else "".join(
        f"{index + 1}. {value} " for index, value in enumerate(reports)
    )
    return (
        f"You are provided with a chest X-ray image, a image-related question and {len(reports)} reference report(s): "
        f"{formatted}\nPlease answer the question based on the image and report and choose from the following two options: "
        "[yes, no]. It should be noted that the diagnostic information in the reference reports cannot be directly "
        "used as the basis for diagnosis, but should only be used for reference and comparison. Question: " + question
    )


def inference_fingerprint_data(
    questions: Path, checkpoint: Path, max_new_tokens: int,
    base_answers: Path | None, prompt_protocol: str,
) -> dict[str, Any]:
    if prompt_protocol not in {"no_reference", "rule_mimic"}:
        raise ValueError(f"unknown prompt protocol: {prompt_protocol}")
    return {
        "version": INFERENCE_VERSION,
        "questions_sha256": file_sha256(questions),
        "checkpoint_sha256": file_sha256(checkpoint),
        "max_new_tokens": max_new_tokens,
        "base_answers_sha256": file_sha256(base_answers) if base_answers else None,
        "prompt_protocol": prompt_protocol,
    }


def checkpoint_adapter_spec(payload: dict[str, Any]) -> dict[str, Any]:
    version = payload.get("version")
    if version == VERSION:
        config = payload["config"]
        return {
            "location": "pre", "rank": int(config["rank"]),
            "max_relative_update": float(config["max_relative_update"]),
            "objective": config.get("mode"),
        }
    if version in {SOURCE_GROUP_VERSION, ANCHOR_DG_VERSION}:
        config = payload.get("config")
        if config is None:
            config = payload["fingerprint_payload"]["config"]
        return {
            "location": "post", "rank": int(config["rank"]),
            "max_relative_update": float(config["max_relative_update"]),
            "objective": config.get("objective"),
        }
    raise RuntimeError(f"unsupported checkpoint version: {version}")


@torch.inference_mode()
def decode(adapter, image, prompt, max_new_tokens, module, module_location="pre") -> str:
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria
    input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    pixels = adapter._process_images([image])
    if isinstance(pixels, list):
        pixels = [item.to(adapter.model.device, dtype=adapter.model.dtype) for item in pixels]
    else:
        pixels = pixels.to(adapter.model.device, dtype=adapter.model.dtype)
    conversation = conv_templates[adapter.conv_mode].copy()
    stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
    stopping = KeywordsStoppingCriteria([stop], adapter.tokenizer, input_ids)
    if module is None:
        context = nullcontext()
    elif module_location == "pre":
        context = attach_preprojector_adapter(adapter.model, module)
    elif module_location == "post":
        context = attach_postprojector_adapter(adapter.model, module)
    else:
        raise ValueError(f"unknown module location: {module_location}")
    with context:
        output_ids = adapter.model.generate(
            input_ids, images=pixels, image_sizes=[image.size],
            attention_mask=torch.ones_like(input_ids, dtype=torch.long),
            do_sample=False, temperature=0.0, num_beams=1,
            max_new_tokens=max_new_tokens, use_cache=True,
            stopping_criteria=[stopping], pad_token_id=adapter.tokenizer.eos_token_id,
        )
    text = adapter.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text[: -len(stop)].strip() if stop and text.endswith(stop) else text


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-answers", type=Path, help="Optional verified RULE cache used to avoid duplicate base decoding.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--prompt-protocol", choices=("no_reference", "rule_mimic"),
        default="no_reference",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module_spec = checkpoint_adapter_spec(payload)
    rows = load_rows(args.questions)
    rows = rows[: args.max_samples] if args.max_samples is not None else rows
    for row in rows:
        if row.get("answer") is None:
            raise ValueError(f"missing ground truth for qid={row.get('question_id')}")
    fingerprint_data = inference_fingerprint_data(
        args.questions, args.checkpoint, args.max_new_tokens,
        args.base_answers, args.prompt_protocol,
    )
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta = fingerprint_data | {
        "fingerprint": fingerprint, "questions": str(args.questions.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "n_requested": len(rows),
    }
    if args.resume and meta_path.is_file():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("existing output fingerprint mismatch")
    elif args.output.exists() and args.output.stat().st_size:
        raise FileExistsError("output exists; use --resume only for identical fingerprint")
    atomic_json(meta_path, meta)
    repair_jsonl_tail(args.output)
    completed = successful_qids(args.output) if args.resume else set()
    base_cache = cached_base_answers(args.base_answers)

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    module = BoundedResidualBottleneck(
        int(payload["width"]), module_spec["rank"],
        module_spec["max_relative_update"],
    ).to(adapter.model.device)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    with args.output.open("a") as handle:
        for row in tqdm(rows, desc="paired RULE decode"):
            qid = str(row.get("question_id", row.get("qid")))
            if qid in completed:
                continue
            record = {
                "question_id": row.get("question_id", row.get("qid")), "image": row["image"],
                "gt_answer": row["answer"], "fingerprint": fingerprint,
            }
            try:
                with Image.open(args.image_root / row["image"]) as source:
                    image = source.convert("RGB")
                prompt = official_prompt(row, args.prompt_protocol)
                cached = base_cache.get(qid)
                if cached is not None:
                    if str(cached.get("image")) != str(row.get("image")):
                        raise ValueError(f"cached baseline image mismatch for qid={qid}")
                    if str(cached.get("gt_answer")) != str(row.get("answer")):
                        raise ValueError(f"cached baseline GT mismatch for qid={qid}")
                    if str(cached.get("prompt")) != prompt:
                        raise ValueError(f"cached baseline prompt mismatch for qid={qid}")
                    base_text = str(cached.get("answer", cached.get("text", ""))).strip()
                    if not base_text:
                        raise ValueError(f"empty cached baseline answer for qid={qid}")
                else:
                    base_text = decode(adapter, image, prompt, args.max_new_tokens, None)
                adapted_text = decode(
                    adapter, image, prompt, args.max_new_tokens, module,
                    module_location=module_spec["location"],
                )
                record.update({
                    "status": "ok", "prompt": prompt, "base_text": base_text,
                    "adapted_text": adapted_text, "text": adapted_text, "answer": adapted_text,
                    "model_id": f"llava-med-{module_spec['location']}projector-sequence-dg",
                    "metadata": {
                        "checkpoint_version": payload["version"],
                        "objective": module_spec["objective"],
                        "adapter_location": module_spec["location"],
                    },
                })
            except Exception as error:
                record.update({"status": "error", "error_type": type(error).__name__, "error": str(error)})
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    meta["n_complete"] = len(successful_qids(args.output))
    atomic_json(meta_path, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
