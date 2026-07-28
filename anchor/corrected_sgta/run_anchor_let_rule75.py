#!/usr/bin/env python3
"""Evaluate layer evidence transport under the frozen RULE/MIMIC protocol.

The runner deliberately generates complete sentences.  The published local
RULE-compatible score is computed only after generation with the normalized
first-sentence parser.  A frozen greedy cache may be joined by question id so
the expensive baseline decode is not repeated.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import re
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.anchor_transport import resolve_image_path, stable_json_sha256
from corrected_sgta.evaluate_medheval_answers import parse_answer
from corrected_sgta.evaluate_rule_vqa import (
    parse_rule_ground_truth,
    rule_normalized_prediction,
)
from corrected_sgta.models_oe import Generation, LlavaMedOEAdapter
from corrected_sgta.rule_mitigation_backend import render_rule_prompt
from corrected_sgta.rule_mitigation_backend import render_rule_model_prompt
from corrected_sgta.run_anchor_layer_expert_pilot import generate_layer_expert

VERSION = "anchor-let-rule75-v1"


@torch.inference_mode()
def generate_layer_expert_standard(
    adapter: LlavaMedOEAdapter,
    image: Image.Image,
    prompt: str,
    *,
    alpha: float,
    expert_layer: int,
    max_new_tokens: int,
    seed: int,
) -> Generation:
    """Use RULE's standard ``model.generate`` path and alter only its logits."""
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
    if alpha > 0:
        @functools.wraps(original_forward)
        def mixed_forward(_self, *forward_args: Any, **forward_kwargs: Any):
            forward_kwargs["output_hidden_states"] = True
            output = original_forward(*forward_args, **forward_kwargs)
            hidden_states = output.hidden_states
            layer_index = (
                expert_layer
                if expert_layer >= 0
                else len(hidden_states) + expert_layer
            )
            layer_index = max(0, min(layer_index, len(hidden_states) - 1))
            expert_hidden = adapter.model.model.norm(hidden_states[layer_index])
            expert_logits = adapter.model.lm_head(
                expert_hidden.to(adapter.model.lm_head.weight.dtype)
            ).float()
            output.logits = (
                (1.0 - alpha) * output.logits.float() + alpha * expert_logits
            )
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
    text = adapter.tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()
    return Generation(
        text=text,
        uncertainty=float("nan"),
        token_count=int(output_ids.shape[-1]),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_id(image: str) -> str:
    # MIMIC paths contain both a two-digit shard (for example ``p15``) and
    # the actual subject id (for example ``p15518538``).  Prefer the longest
    # patient-like path component so cluster inference is not performed over
    # only the ten top-level shards.
    matches = re.findall(r"(?:^|/)(p\d+)(?=/|$)", image)
    return max(matches, key=len) if matches else image


def evaluate_sentence(text: str, reference: str) -> dict[str, Any]:
    gt, gt_status = parse_rule_ground_truth(reference)
    prediction = rule_normalized_prediction(text)
    strict = parse_answer(text, answer_type="binary")
    strict_prediction = strict.labels[0] if strict.labels else None
    return {
        "ground_truth": gt,
        "ground_truth_status": gt_status,
        "rule_normalized_prediction": prediction,
        "rule_normalized_correct": bool(gt is not None and prediction == gt),
        "strict_prediction": strict_prediction,
        "strict_status": strict.status,
        "strict_correct": bool(gt is not None and strict_prediction == gt),
        "strict_parseable": strict_prediction is not None,
    }


def binary_summary(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    confusion: Counter[str] = Counter()
    correct = 0
    strict_correct = 0
    strict_parseable = 0
    for row in records:
        evaluation = row[prefix]
        gt = evaluation["ground_truth"]
        pred = evaluation["rule_normalized_prediction"]
        confusion[f"{gt}->{pred}"] += 1
        correct += int(evaluation["rule_normalized_correct"])
        strict_correct += int(evaluation["strict_correct"])
        strict_parseable += int(evaluation["strict_parseable"])
    n = len(records)
    tp = confusion["yes->yes"]
    tn = confusion["no->no"]
    fp = confusion["no->yes"]
    fn = confusion["yes->no"]
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": sensitivity,
        "f1": (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        ),
        "strict_accuracy_invalid_as_error": strict_correct / n if n else 0.0,
        "strict_parse_rate": strict_parseable / n if n else 0.0,
    }


def paired_summary(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    baseline = binary_summary(records, "baseline_eval")
    let = binary_summary(records, "let_eval")
    rescue = sum(
        (not row["baseline_eval"]["rule_normalized_correct"])
        and row["let_eval"]["rule_normalized_correct"]
        for row in records
    )
    harm = sum(
        row["baseline_eval"]["rule_normalized_correct"]
        and (not row["let_eval"]["rule_normalized_correct"])
        for row in records
    )
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        # Re-derive the patient id so resumed records produced before the
        # patient-path fix are summarized correctly without being regenerated.
        by_patient[patient_id(str(row["image"]))].append(row)
    patient_keys = sorted(by_patient)
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    if patient_keys:
        for _ in range(2000):
            chosen = rng.choice(patient_keys, size=len(patient_keys), replace=True)
            sampled = [row for key in chosen for row in by_patient[str(key)]]
            delta = np.mean(
                [
                    int(row["let_eval"]["rule_normalized_correct"])
                    - int(row["baseline_eval"]["rule_normalized_correct"])
                    for row in sampled
                ]
            )
            boot.append(float(delta))
    discordant = rescue + harm
    # Exact two-sided McNemar p-value without a scipy dependency.
    if discordant:
        smaller = min(rescue, harm)
        tail = sum(
            math.comb(discordant, k) for k in range(smaller + 1)
        ) / (2**discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    return {
        "version": VERSION,
        "n": len(records),
        "patients": len(patient_keys),
        "baseline": baseline,
        "let": let,
        "delta_accuracy": let["accuracy"] - baseline["accuracy"],
        "delta_balanced_accuracy": (
            let["balanced_accuracy"] - baseline["balanced_accuracy"]
        ),
        "rescue": int(rescue),
        "harm": int(harm),
        "mcnemar_exact_p": float(mcnemar_p),
        "patient_cluster_bootstrap_delta_95ci": (
            [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if boot
            else [0.0, 0.0]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/rule/test/mimic_test.jsonl")
    )
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        default=Path(
            "../Hulu-Med/MedUniEval/corrected_runs/rule_source_hull_v1/"
            "fullsource_erm.mimic_locked.canonical.jsonl"
        ),
    )
    parser.add_argument("--image-root", type=Path, default=Path("data/medheval/images"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corrected_runs/final_anchor_let_rule75_v1"),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--layer", type=int, default=-12)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--generation-backend", choices=("standard", "manual"), default="standard"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = load_jsonl(args.manifest)
    if args.max_samples:
        questions = questions[: args.max_samples]
    baseline_rows = load_jsonl(args.baseline_cache)
    baseline_by_qid = {
        str(row["question_id"]): row for row in baseline_rows
    }
    missing = [
        str(row["question_id"])
        for row in questions
        if str(row["question_id"]) not in baseline_by_qid
    ]
    if missing:
        raise ValueError(f"baseline cache misses question ids: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "let.raw.jsonl"
    summary_path = args.output_dir / "summary.json"
    fingerprint = stable_json_sha256(
        {
            "version": VERSION,
            "manifest_sha256": sha256_file(args.manifest),
            "baseline_sha256": sha256_file(args.baseline_cache),
            "conv_mode": "vicuna_v1",
            "rule_dataset": "mimic",
            "max_new_tokens": args.max_new_tokens,
            "layer": args.layer,
            "alpha": args.alpha,
            "seed": args.seed,
            "generation_backend": args.generation_backend,
            "generated_sentence_evaluation": True,
            "label_logits_as_prediction": False,
        }
    )
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and raw_path.exists():
        for row in load_jsonl(raw_path):
            if row.get("fingerprint") != fingerprint:
                raise ValueError("resume fingerprint mismatch")
            existing[str(row["question_id"])] = row

    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    # The vendored RULE scripts render this exact one-turn Vicuna string.
    # LLaVA's generic conversation helper differs slightly across repository
    # revisions, so bind the audited renderer directly before tokenization.
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import tokenizer_image_token

    def exact_rule_prompt_ids(current_prompt: str):
        model_prompt = render_rule_model_prompt(
            current_prompt,
            image_token=DEFAULT_IMAGE_TOKEN,
            image_start_token=DEFAULT_IM_START_TOKEN,
            image_end_token=DEFAULT_IM_END_TOKEN,
            use_image_start_end=bool(adapter.model.config.mm_use_im_start_end),
        )
        return tokenizer_image_token(
            model_prompt,
            adapter.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)

    adapter._prompt_ids = exact_rule_prompt_ids
    mode = "a" if args.resume else "w"
    records = list(existing.values())
    with raw_path.open(mode) as handle:
        for index, question in enumerate(tqdm(questions, desc="LET RULE-75")):
            qid = str(question["question_id"])
            if qid in existing:
                continue
            current_prompt = render_rule_prompt("mimic", question)
            with Image.open(
                resolve_image_path(str(question["image"]), args.image_root)
            ) as source:
                image = source.convert("RGB")
            generator = (
                generate_layer_expert_standard
                if args.generation_backend == "standard"
                else generate_layer_expert
            )
            generation = generator(
                adapter,
                image,
                current_prompt,
                alpha=args.alpha,
                expert_layer=args.layer,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + index,
            )
            baseline_text = str(baseline_by_qid[qid]["base_text"])
            record = {
                "version": VERSION,
                "fingerprint": fingerprint,
                "question_id": qid,
                "patient_id": patient_id(str(question["image"])),
                "image": question["image"],
                "question": question["question"],
                "prompt": current_prompt,
                "reference": question["answer"],
                "baseline_text": baseline_text,
                "let_text": generation.text,
                "layer": args.layer,
                "alpha": args.alpha,
                "mean_token_nll": generation.uncertainty,
                "token_count": generation.token_count,
                "baseline_eval": evaluate_sentence(baseline_text, question["answer"]),
                "let_eval": evaluate_sentence(generation.text, question["answer"]),
                "target_labels_used_for_generation": False,
                "uses_label_logits_for_prediction": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            if len(records) % 32 == 0:
                progress = paired_summary(records, args.seed)
                progress["fingerprint"] = fingerprint
                progress["status"] = "running"
                summary_path.write_text(
                    json.dumps(progress, indent=2, ensure_ascii=False) + "\n"
                )
    summary = paired_summary(records, args.seed)
    summary["fingerprint"] = fingerprint
    summary["status"] = "complete"
    summary["raw"] = str(raw_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
