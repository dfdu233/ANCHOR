#!/usr/bin/env python3
"""Natural-image counterfactual test of multimodal under-identification.

Each experimental unit uses the same normalized binary question on one Yes
image and one No image. Images are globally disjoint. Diagnostic categories
are not explanatory variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.clinical_claims import softmax_states
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    append_jsonl,
    atomic_json,
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    sha256_file,
)
from corrected_sgta.run_vqa_rad_underidentification_pilot import (
    PROMPT_TEMPLATES,
    auc,
    entropy,
    js_divergence,
    load_rows,
    stable_key,
    zscore,
)


VERSION = "vqa-rad-natural-counterfactual-underidentification-v1"


def normalize_question(value: object) -> str:
    return re.sub(r"\W+", " ", str(value).lower()).strip()


def select_pairs(
    rows: Sequence[Mapping[str, Any]], image_root: Path, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        if (
            str(row.get("question_type", "")).lower() != "binary"
            or str(row.get("answer", "")).strip().lower() not in {"yes", "no"}
            or not (image_root / str(row.get("img_name", ""))).is_file()
        ):
            continue
        grouped[normalize_question(row["question"])].append(row)

    candidates: list[tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = []
    for question, members in grouped.items():
        positives = [row for row in members if str(row["answer"]).lower() == "yes"]
        negatives = [row for row in members if str(row["answer"]).lower() == "no"]
        pairs = [
            (positive, negative)
            for positive in positives
            for negative in negatives
            if positive["img_name"] != negative["img_name"]
        ]
        if pairs:
            candidates.append((question, pairs))
    candidates.sort(key=lambda item: (len(item[1]), stable_key(seed, "question", item[0])))

    used_images: set[str] = set()
    selected = []
    for question, pairs in candidates:
        ordered = sorted(
            pairs,
            key=lambda pair: stable_key(
                seed, question, pair[0]["img_name"], pair[1]["img_name"]
            ),
        )
        for positive, negative in ordered:
            images = {str(positive["img_name"]), str(negative["img_name"])}
            if images.isdisjoint(used_images):
                used_images.update(images)
                selected.append({
                    "pair_id": hashlib.sha256(f"{seed}:{question}".encode()).hexdigest()[:16],
                    "normalized_question": question,
                    "question": str(positive["question"]),
                    "positive": positive,
                    "negative": negative,
                })
                break
    if not selected:
        raise ValueError("no image-disjoint mixed-label exact-question pairs")
    return sorted(selected, key=lambda row: stable_key(seed, "run", row["pair_id"]))


@torch.inference_mode()
def score_real(bot: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, (start, end) = prepared_embeddings(bot, prompt, tensor)
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    final = len(hidden) - 1
    logits = layer_logits(bot, hidden, [final], label_ids(bot))[final]
    probabilities = softmax_states(logits)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "state": max(probabilities, key=probabilities.get),
        "visual_tokens": end - start,
    }


def analyze(records: Sequence[Mapping[str, Any]], seed: int, draws: int) -> dict[str, Any]:
    rows = [dict(row) for row in records if row.get("status") == "ok"]
    if not rows:
        raise ValueError("no successful pairs")
    states = ("supported", "refuted", "undetermined")
    derived = []
    for row in rows:
        margins: dict[str, list[float]] = {"positive": [], "negative": []}
        probability_vectors: dict[str, list[list[float]]] = {"positive": [], "negative": []}
        prompt_states: dict[str, list[str]] = {"positive": [], "negative": []}
        for role in ("positive", "negative"):
            for prompt_name in PROMPT_TEMPLATES:
                score = row["scores"][role][prompt_name]
                margins[role].append(
                    float(score["logits"]["supported"] - score["logits"]["refuted"])
                )
                probability_vectors[role].append(
                    [float(score["probabilities"][state]) for state in states]
                )
                prompt_states[role].append(str(score["state"]))
        contrast = np.asarray(margins["positive"]) - np.asarray(margins["negative"])
        positive_base = row["scores"]["positive"]["canonical"]
        negative_base = row["scores"]["negative"]["canonical"]
        positive_error = int(positive_base["state"] != "supported")
        negative_error = int(negative_base["state"] != "refuted")
        language_positive = js_divergence(probability_vectors["positive"])
        language_negative = js_divergence(probability_vectors["negative"])
        derived.append({
            "pair_id": row["pair_id"],
            "positive_qid": row["positive_qid"],
            "negative_qid": row["negative_qid"],
            "positive_error": positive_error,
            "negative_error": negative_error,
            "any_error": int(positive_error or negative_error),
            "mean_error": (positive_error + negative_error) / 2.0,
            "baseline_mean_entropy": float(np.mean([
                entropy([float(positive_base["probabilities"][state]) for state in states]),
                entropy([float(negative_base["probabilities"][state]) for state in states]),
            ])),
            "language_js": float((language_positive + language_negative) / 2.0),
            "natural_directional_response": float(np.mean(contrast)),
            "natural_response_instability": float(np.std(contrast)),
            "contrast_nonpositive": int(float(np.mean(contrast)) <= 0.0),
            "prompt_flip": int(
                len(set(prompt_states["positive"])) > 1
                or len(set(prompt_states["negative"])) > 1
            ),
        })

    language = np.asarray([row["language_js"] for row in derived])
    visual = np.asarray([row["natural_directional_response"] for row in derived])
    underid = zscore(language) - zscore(visual)
    for row, value in zip(derived, underid):
        row["underidentification_score"] = float(value)
    errors = [int(row["any_error"]) for row in derived]
    metrics = {
        "n_pairs": len(derived),
        "n_images": 2 * len(derived),
        "pair_any_error_rate": float(np.mean(errors)),
        "image_error_rate": float(np.mean([row["mean_error"] for row in derived])),
        "prompt_flip_rate": float(np.mean([row["prompt_flip"] for row in derived])),
        "nonpositive_natural_contrast_rate": float(np.mean([row["contrast_nonpositive"] for row in derived])),
        "mean_natural_directional_response": float(visual.mean()),
        "mean_language_js": float(language.mean()),
        "pair_error_auroc": {
            "baseline_mean_entropy": auc(errors, [row["baseline_mean_entropy"] for row in derived]),
            "language_js": auc(errors, language),
            "negative_natural_response": auc(errors, -visual),
            "natural_response_instability": auc(errors, [row["natural_response_instability"] for row in derived]),
            "underidentification_score": auc(errors, underid),
        },
    }
    order = np.argsort(underid)
    quartile = max(1, len(order) // 4)
    metrics["underidentification_quartiles"] = {
        "quartile_n": quartile,
        "lowest_pair_error_rate": float(np.mean([errors[index] for index in order[:quartile]])),
        "highest_pair_error_rate": float(np.mean([errors[index] for index in order[-quartile:]])),
    }
    rng = np.random.default_rng(seed)
    deltas = []
    component_deltas = []
    for _ in range(draws):
        indices = rng.integers(0, len(derived), len(derived))
        sampled_errors = [errors[index] for index in indices]
        candidate = auc(sampled_errors, [underid[index] for index in indices])
        baseline = auc(
            sampled_errors,
            [derived[index]["baseline_mean_entropy"] for index in indices],
        )
        visual_component = auc(
            sampled_errors,
            [-derived[index]["natural_directional_response"] for index in indices],
        )
        if candidate is not None and baseline is not None:
            deltas.append(candidate - baseline)
        if candidate is not None and visual_component is not None:
            component_deltas.append(candidate - visual_component)
    candidate_auc = metrics["pair_error_auroc"]["underidentification_score"]
    baseline_auc = metrics["pair_error_auroc"]["baseline_mean_entropy"]
    metrics["underidentification_minus_entropy_auroc_bootstrap"] = {
        "valid_draws": len(deltas),
        "estimate": candidate_auc - baseline_auc if candidate_auc is not None and baseline_auc is not None else None,
        "ci_low": float(np.quantile(deltas, 0.025)) if deltas else None,
        "ci_high": float(np.quantile(deltas, 0.975)) if deltas else None,
    }
    visual_auc = metrics["pair_error_auroc"]["negative_natural_response"]
    metrics["underidentification_minus_visual_component_auroc_bootstrap"] = {
        "valid_draws": len(component_deltas),
        "estimate": candidate_auc - visual_auc if candidate_auc is not None and visual_auc is not None else None,
        "ci_low": float(np.quantile(component_deltas, 0.025)) if component_deltas else None,
        "ci_high": float(np.quantile(component_deltas, 0.975)) if component_deltas else None,
    }
    return {"metrics": metrics, "derived_pairs": derived}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--seed", type=int, default=260814)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pairs = select_pairs(load_rows(args.dataset), args.image_root, args.seed)
    if args.limit_pairs is not None:
        pairs = pairs[: args.limit_pairs]
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "image_root": str(args.image_root.resolve()),
        "model": str(args.model_dir.resolve()),
        "method": "inference-only exact-question natural-image counterfactual x semantic prompt orbit",
        "prompt_templates": PROMPT_TEMPLATES,
        "selection": "exact normalized question with both Yes and No natural images; globally image-disjoint deterministic greedy matching",
        "pairs": len(pairs),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(args.output_dir / "config.json", config)
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    raw_path = args.output_dir / "raw.jsonl"
    for index, pair in enumerate(pairs):
        record: dict[str, Any] = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "pair_id": pair["pair_id"],
            "question": pair["question"],
            "normalized_question": pair["normalized_question"],
            "positive_qid": str(pair["positive"]["qid"]),
            "negative_qid": str(pair["negative"]["qid"]),
            "positive_image": pair["positive"]["img_name"],
            "negative_image": pair["negative"]["img_name"],
            "status": "error",
        }
        try:
            scores: dict[str, Any] = {}
            for role in ("positive", "negative"):
                image_path = args.image_root / str(pair[role]["img_name"])
                image = Image.open(image_path).convert("RGB")
                scores[role] = {
                    name: score_real(
                        bot,
                        image,
                        template.format(question=str(pair["question"]).strip()),
                    )
                    for name, template in PROMPT_TEMPLATES.items()
                }
            record["scores"] = scores
            record["status"] = "ok"
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record["error"] = f"CUDA OOM: {error}"
            record["traceback"] = traceback.format_exc()
        except Exception as error:
            record["error"] = repr(error)
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(json.dumps({"progress": f"{index + 1}/{len(pairs)}", "pair_id": pair["pair_id"], "status": record["status"], "error": record.get("error")}), flush=True)
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    summary = analyze(records, args.seed, args.bootstrap_draws)
    summary["version"] = VERSION
    summary["config"] = config
    summary["runtime_errors"] = sum(row.get("status") != "ok" for row in records)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
