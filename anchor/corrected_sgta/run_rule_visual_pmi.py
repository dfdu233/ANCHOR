"""Resumable binary full-sequence visual DC-PMI pilot for RULE/LLaVA-Med.

For candidate y in {Yes., No.}, the fixed score is

    mean_t log p(y_t | I, q, y_<t) - mean_t log p(y_t | I_null, q, y_<t),

where I_null is a same-sized constant RGB image whose channels equal the CLIP
normalization mean.  The script is diagnostic: it performs no calibration or
hyper-parameter search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from corrected_sgta.infer_rule_dg_adapter import (
    atomic_json,
    load_rows,
    official_prompt,
    repair_jsonl_tail,
    successful_qids,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.train_rule_dg_adapter import (
    VERSION as CHECKPOINT_VERSION,
    BoundedResidualBottleneck,
    attach_preprojector_adapter,
    build_teacher_forcing,
    file_sha256,
    process_image,
    rule_label,
)

VERSION = "rule-visual-dc-pmi-v1"
CANDIDATES = ("Yes.", "No.")
CLIP_CHANNEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
NULL_RGB = tuple(round(value * 255) for value in CLIP_CHANNEL_MEAN)


def parse_variant(value: str) -> tuple[str, Path | None]:
    name, separator, checkpoint = value.partition("=")
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise ValueError(f"invalid variant name: {name!r}")
    if not separator:
        if name != "base":
            raise ValueError("only the base variant may omit =CHECKPOINT")
        return name, None
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def load_module(adapter: LlavaMedAlignmentAdapter, checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(f"unsupported checkpoint version: {payload.get('version')}")
    config = payload.get("config", {})
    module = BoundedResidualBottleneck(
        int(payload["width"]),
        int(config["rank"]),
        float(config["max_relative_update"]),
    ).to(adapter.model.device)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return module, {
        "checkpoint_version": payload["version"],
        "training_mode": config.get("mode"),
    }


@torch.inference_mode()
def mean_sequence_log_probability(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    candidate: str,
    module,
) -> tuple[float, int, list[float]]:
    input_ids, labels = build_teacher_forcing(adapter, prompt, candidate)
    ids = input_ids.to(adapter.model.device)
    targets = labels.to(adapter.model.device)
    pixels = process_image(adapter, image)
    context = attach_preprojector_adapter(adapter.model, module) if module is not None else nullcontext()
    with context:
        _, position_ids, attention_mask, _, inputs_embeds, expanded_labels = (
            adapter.model.prepare_inputs_labels_for_multimodal(
                ids,
                None,
                None,
                None,
                targets,
                pixels,
                image_sizes=[image.size],
            )
        )
        output = adapter.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        vocabulary_weight = adapter.model.get_output_embeddings().weight
        logits = output.last_hidden_state.to(vocabulary_weight.dtype) @ vocabulary_weight.T
    if expanded_labels is None:
        raise RuntimeError("multimodal preparation did not return expanded labels")
    shifted_labels = expanded_labels[:, 1:]
    mask = shifted_labels.ne(-100)
    token_count = int(mask.sum())
    if token_count < 2:
        raise RuntimeError("candidate sequence must contain label and separator/EOS tokens")
    selected_labels = shifted_labels[mask]
    selected_logits = logits[:, :-1][mask].float()
    token_log_probabilities = -F.cross_entropy(
        selected_logits, selected_labels, reduction="none"
    )
    if not bool(torch.isfinite(token_log_probabilities).all()):
        raise FloatingPointError("non-finite candidate sequence log probability")
    values = token_log_probabilities.cpu().tolist()
    return float(token_log_probabilities.mean().cpu()), token_count, values


def score_variant(adapter, image, null_image, prompt, module) -> dict[str, Any]:
    candidate_rows = []
    for candidate in CANDIDATES:
        image_logp, image_tokens, image_token_logp = mean_sequence_log_probability(
            adapter, image, prompt, candidate, module
        )
        null_logp, null_tokens, null_token_logp = mean_sequence_log_probability(
            adapter, null_image, prompt, candidate, module
        )
        if image_tokens != null_tokens:
            raise RuntimeError("image/null candidate token counts differ")
        candidate_rows.append({
            "label": rule_label(candidate),
            "candidate": candidate,
            "token_count": image_tokens,
            "image_mean_token_logp": image_logp,
            "null_mean_token_logp": null_logp,
            "visual_dc_pmi": image_logp - null_logp,
            "image_token_logp": image_token_logp,
            "null_token_logp": null_token_logp,
        })
    prediction = max(candidate_rows, key=lambda row: row["visual_dc_pmi"])["label"]
    return {"prediction": prediction, "candidates": candidate_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME[=CHECKPOINT]",
        help="Use --variant base for the unmodified model and NAME=PATH for adapters.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    variants = dict(parse_variant(value) for value in args.variant)
    if len(variants) != len(args.variant):
        raise ValueError("duplicate variant name")
    rows = load_rows(args.questions)
    rows = rows[: args.max_samples] if args.max_samples is not None else rows
    for row in rows:
        if row.get("answer") is None:
            raise ValueError(f"missing ground truth for qid={row.get('question_id')}")

    fingerprint_data = {
        "version": VERSION,
        "questions_sha256": file_sha256(args.questions),
        "variants": {
            name: file_sha256(path) if path is not None else None
            for name, path in variants.items()
        },
        "candidate_sequences": CANDIDATES,
        "score": "mean-token-log-p(image)-mean-token-log-p(null)",
        "null_image": {
            "kind": "same-size-constant-clip-channel-mean",
            "clip_channel_mean": CLIP_CHANNEL_MEAN,
            "uint8_rgb": NULL_RGB,
        },
        "prompt": "RULE official no-reference/reference-aware prompt",
        "parser": "RULE/LLaVA POPE first-period no/not convention",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta = fingerprint_data | {
        "fingerprint": fingerprint,
        "questions": str(args.questions.resolve()),
        "image_root": str(args.image_root.resolve()),
        "n_requested": len(rows),
    }
    if args.resume and meta_path.is_file():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("existing output fingerprint mismatch")
    elif args.output.exists() and args.output.stat().st_size:
        raise FileExistsError("output exists; use --resume only for identical fingerprint")
    atomic_json(meta_path, meta)
    repair_jsonl_tail(args.output)
    completed = successful_qids(args.output) if args.resume else set()

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    modules: dict[str, Any] = {}
    variant_metadata: dict[str, dict[str, Any]] = {}
    for name, checkpoint in variants.items():
        if checkpoint is None:
            modules[name] = None
            variant_metadata[name] = {"checkpoint_version": None, "training_mode": None}
        else:
            modules[name], variant_metadata[name] = load_module(adapter, checkpoint)

    with args.output.open("a") as handle:
        for row in tqdm(rows, desc="RULE visual DC-PMI"):
            qid = str(row.get("question_id", row.get("qid")))
            if qid in completed:
                continue
            record = {
                "question_id": row.get("question_id", row.get("qid")),
                "image": row["image"],
                "gt_answer": rule_label(row["answer"]),
                "fingerprint": fingerprint,
            }
            try:
                with Image.open(args.image_root / row["image"]) as source:
                    image = source.convert("RGB")
                null_image = Image.new("RGB", image.size, NULL_RGB)
                prompt = official_prompt(row)
                scores = {
                    name: score_variant(adapter, image, null_image, prompt, modules[name])
                    for name in variants
                }
                record.update({
                    "status": "ok",
                    "prompt": prompt,
                    "null_rgb": NULL_RGB,
                    "variants": scores,
                    "variant_metadata": variant_metadata,
                })
            except Exception as error:
                record.update({
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    meta["n_complete"] = len(successful_qids(args.output))
    atomic_json(meta_path, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
