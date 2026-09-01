#!/usr/bin/env python3
"""Inference-only pilot for multimodal under-identification on VQA-RAD.

The scientific variables are prompt-orbit dispersion and visual causal
response.  Body region is used only to diversify the small pilot sample; it is
not an explanatory image taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.clinical_claims import softmax_states
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    append_jsonl,
    atomic_json,
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    sha256_file,
)


VERSION = "vqa-rad-underidentification-pilot-v1"
PROMPT_TEMPLATES = {
    "canonical": "{question} Answer with exactly one word: Yes, No, or Maybe.",
    "polite": "Please answer this question about the medical image: {question} Respond with exactly one word: Yes, No, or Maybe.",
    "topic_fronted": "Regarding the medical image, {question} Give exactly one word: Yes, No, or Maybe.",
    "question_marked": "Question about the medical image: {question} Your answer must be exactly Yes, No, or Maybe.",
}


def stable_key(seed: int, *parts: object) -> str:
    return hashlib.sha256(":".join((str(seed), *(str(x) for x in parts))).encode()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("dataset must be a JSON list")
    return [dict(row) for row in payload]


def select_rows(rows: Sequence[Mapping[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in rows
        if str(row.get("question_type", "")).lower() == "binary"
        and str(row.get("answer", "")).strip().lower() in {"yes", "no"}
    ]
    # One question per image avoids pseudo-replication.  Round-robin over broad
    # acquisition regions and labels makes the small pilot diverse without
    # turning those metadata fields into a scientific taxonomy.
    per_image: dict[str, dict[str, Any]] = {}
    for row in sorted(eligible, key=lambda x: stable_key(seed, x.get("img_name"), x.get("qid"))):
        per_image.setdefault(str(row["img_name"]), row)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_image.values():
        key = (str(row.get("location", "unknown")), str(row["answer"]).lower())
        buckets.setdefault(key, []).append(row)
    for key in buckets:
        buckets[key].sort(key=lambda x: stable_key(seed, key, x.get("img_name"), x.get("qid")))
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda key: stable_key(seed, "bucket", key))
    while len(selected) < limit and any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
    if len(selected) != limit:
        raise ValueError(f"requested {limit} image-disjoint rows but found {len(selected)}")
    return sorted(selected, key=lambda x: stable_key(seed, "run", x.get("img_name"), x.get("qid")))


def entropy(values: Sequence[float]) -> float:
    return -sum(float(x) * math.log(max(float(x), 1e-12)) for x in values)


def js_divergence(probabilities: Sequence[Sequence[float]]) -> float:
    array = np.asarray(probabilities, dtype=np.float64)
    mean = array.mean(axis=0)
    return float(entropy(mean) - np.mean([entropy(row) for row in array]))


@torch.inference_mode()
def score_conditions(bot: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, (start, end) = prepared_embeddings(bot, prompt, tensor)
    replacement = embeddings[:, start:end].mean(dim=1, keepdim=True)
    mean_null = embeddings.clone()
    mean_null[:, start:end] = replacement
    replacement_norm = torch.linalg.vector_norm(replacement.float(), dim=-1, keepdim=True)
    original_norms = torch.linalg.vector_norm(
        embeddings[:, start:end].float(), dim=-1, keepdim=True
    )
    if float(replacement_norm.min()) <= 1e-12:
        raise ValueError("visual mean direction has degenerate norm")
    norm_null = embeddings.clone()
    norm_null[:, start:end] = (
        replacement.float() / replacement_norm * original_norms
    ).to(embeddings.dtype)
    ids = label_ids(bot)
    output: dict[str, Any] = {}
    for name, condition in (
        ("real", embeddings),
        ("mean_token_null", mean_null),
        ("norm_matched_null", norm_null),
    ):
        hidden = hidden_trajectory(bot, condition, attention, positions)
        final = len(hidden) - 1
        logits = layer_logits(bot, hidden, [final], ids)[final]
        probabilities = softmax_states(logits)
        output[name] = {
            "logits": logits,
            "probabilities": probabilities,
            "state": max(probabilities, key=probabilities.get),
        }
    output["visual_tokens"] = end - start
    return output


def auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    sorted_scores = np.asarray(scores, dtype=float)[order]
    begin = 0
    while begin < len(order):
        finish = begin + 1
        while finish < len(order) and sorted_scores[finish] == sorted_scores[begin]:
            finish += 1
        ranks[order[begin:finish]] = (begin + 1 + finish) / 2.0
        begin = finish
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    scale = float(array.std())
    return (array - array.mean()) / (scale if scale > 1e-12 else 1.0)


def analyze(records: Sequence[Mapping[str, Any]], seed: int, draws: int) -> dict[str, Any]:
    rows = [dict(row) for row in records if row.get("status") == "ok"]
    if not rows:
        raise ValueError("no successful records")
    derived = []
    state_order = ("supported", "refuted", "undetermined")
    for row in rows:
        reference = str(row["answer"]).lower()
        sign = 1.0 if reference == "yes" else -1.0
        real_margins, null_margins, real_prob_vectors = [], [], []
        real_states = []
        for prompt_name in PROMPT_TEMPLATES:
            score = row["scores"][prompt_name]
            real = score["real"]
            null = score["norm_matched_null"]
            real_margins.append(float(real["logits"]["supported"] - real["logits"]["refuted"]))
            null_margins.append(float(null["logits"]["supported"] - null["logits"]["refuted"]))
            real_prob_vectors.append([float(real["probabilities"][state]) for state in state_order])
            real_states.append(str(real["state"]))
        base = row["scores"]["canonical"]["real"]
        prediction = str(base["state"])
        expected = "supported" if reference == "yes" else "refuted"
        delta = np.asarray(real_margins) - np.asarray(null_margins)
        derived.append({
            "qid": row["qid"],
            "image": row["img_name"],
            "error": int(prediction != expected),
            "baseline_state": prediction,
            "baseline_entropy": entropy([float(base["probabilities"][state]) for state in state_order]),
            "baseline_abs_margin": abs(real_margins[0]),
            "language_js": js_divergence(real_prob_vectors),
            "language_margin_std": float(np.std(real_margins)),
            "prompt_flip": int(len(set(real_states)) > 1),
            "visual_response": float(abs(np.mean(delta))),
            "visual_response_mean_abs": float(np.mean(np.abs(delta))),
            "truth_aligned_visual_lift": float(sign * np.mean(delta)),
        })
    errors = [int(row["error"]) for row in derived]
    language = np.asarray([row["language_js"] for row in derived])
    visual = np.asarray([row["visual_response"] for row in derived])
    underid = zscore(language) - zscore(visual)
    for row, value in zip(derived, underid):
        row["underidentification_score"] = float(value)
    metrics = {
        "n": len(derived),
        "errors": sum(errors),
        "accuracy": float(1.0 - np.mean(errors)),
        "prompt_flip_rate": float(np.mean([row["prompt_flip"] for row in derived])),
        "mean_language_js": float(language.mean()),
        "mean_visual_response": float(visual.mean()),
        "mean_truth_aligned_visual_lift": float(np.mean([row["truth_aligned_visual_lift"] for row in derived])),
        "error_auroc": {
            "baseline_entropy": auc(errors, [row["baseline_entropy"] for row in derived]),
            "language_js": auc(errors, language),
            "negative_visual_response": auc(errors, -visual),
            "underidentification_score": auc(errors, underid),
        },
    }
    order = np.argsort(underid)
    quartile = max(1, len(order) // 4)
    metrics["underidentification_quartiles"] = {
        "lowest_error_rate": float(np.mean([errors[index] for index in order[:quartile]])),
        "highest_error_rate": float(np.mean([errors[index] for index in order[-quartile:]])),
        "quartile_n": quartile,
    }
    rng = np.random.default_rng(seed)
    deltas = []
    valid_draws = 0
    for _ in range(draws):
        indices = rng.integers(0, len(derived), len(derived))
        sampled_errors = [errors[index] for index in indices]
        candidate = auc(sampled_errors, [underid[index] for index in indices])
        baseline = auc(sampled_errors, [derived[index]["baseline_entropy"] for index in indices])
        if candidate is not None and baseline is not None:
            deltas.append(candidate - baseline)
            valid_draws += 1
    metrics["underidentification_minus_entropy_auroc_bootstrap"] = {
        "valid_draws": valid_draws,
        "estimate": (
            metrics["error_auroc"]["underidentification_score"]
            - metrics["error_auroc"]["baseline_entropy"]
            if metrics["error_auroc"]["underidentification_score"] is not None
            and metrics["error_auroc"]["baseline_entropy"] is not None
            else None
        ),
        "ci_low": float(np.quantile(deltas, 0.025)) if deltas else None,
        "ci_high": float(np.quantile(deltas, 0.975)) if deltas else None,
    }
    return {"metrics": metrics, "derived_rows": derived}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260814)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    source_rows = load_rows(args.dataset)
    available_rows = [
        row for row in source_rows
        if (args.image_root / str(row.get("img_name", ""))).is_file()
    ]
    selected = select_rows(available_rows, args.limit, args.seed)
    completed = {
        str(json.loads(line)["qid"])
        for line in raw_path.read_text().splitlines()
        if line.strip() and json.loads(line).get("status") == "ok"
    } if args.resume and raw_path.exists() else set()
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or a new output directory")
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "image_root": str(args.image_root.resolve()),
        "model": str(args.model_dir.resolve()),
        "method": "inference-only prompt-orbit x visual-null under-identification pilot",
        "prompt_templates": PROMPT_TEMPLATES,
        "visual_conditions": ["real", "per-image mean-token null", "per-token norm-matched mean-direction null"],
        "selection": "image-disjoint deterministic round-robin over broad region x reference polarity; metadata are sampling controls only",
        "source_rows": len(source_rows),
        "rows_with_local_images": len(available_rows),
        "limit": args.limit,
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(args.output_dir / "config.json", config)
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    for index, row in enumerate(selected):
        qid = str(row["qid"])
        if qid in completed:
            continue
        record: dict[str, Any] = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "qid": qid,
            "img_name": row["img_name"],
            "question": row["question"],
            "answer": row["answer"],
            "sampling_region": row.get("location"),
            "status": "error",
        }
        try:
            image_path = args.image_root / str(row["img_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            record["scores"] = {
                name: score_conditions(bot, image, template.format(question=str(row["question"]).strip()))
                for name, template in PROMPT_TEMPLATES.items()
            }
            record["image_path"] = str(image_path.resolve())
            record["status"] = "ok"
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record["error"] = f"CUDA OOM: {error}"
            record["traceback"] = traceback.format_exc()
        except Exception as error:
            record["error"] = repr(error)
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(json.dumps({"progress": f"{index + 1}/{len(selected)}", "qid": qid, "status": record["status"], "error": record.get("error")}), flush=True)
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    summary = analyze(records, args.seed, args.bootstrap_draws)
    summary["version"] = VERSION
    summary["config"] = config
    summary["errors_in_runtime"] = sum(row.get("status") != "ok" for row in records)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
