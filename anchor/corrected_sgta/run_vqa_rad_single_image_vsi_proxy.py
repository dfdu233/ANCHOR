#!/usr/bin/env python3
"""Single-image, label-free proxy for visual solution identifiability.

The canonical prompt selects image regions whose removal most reduces the
model's own chosen semantic answer.  Disjoint prompt paraphrases then measure
whether those regions are selectively necessary (more influential than a
matched random region set) and sufficient (retain the full-image solution when
shown alone).  Labels and the paired counterfactual image are evaluation-only.
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
from PIL import Image, ImageFilter

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    append_jsonl,
    atomic_json,
    import_huatuo,
    sha256_file,
)
from corrected_sgta.run_vqa_rad_natural_counterfactual_pilot import score_real
from corrected_sgta.run_vqa_rad_underidentification_pilot import (
    PROMPT_TEMPLATES,
    auc,
    entropy,
    js_divergence,
    stable_key,
)


VERSION = "vqa-rad-single-image-vsi-proxy-v1"
STATES = ("supported", "refuted", "undetermined")
SELECTION_PROMPT = "canonical"
EVALUATION_PROMPTS = tuple(name for name in PROMPT_TEMPLATES if name != SELECTION_PROMPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--limit-pairs", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--blur-fraction", type=float, default=1 / 24)
    parser.add_argument("--seed", type=int, default=260814)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def vector(score: Mapping[str, Any]) -> list[float]:
    return [float(score["probabilities"][state]) for state in STATES]


def log_probability(score: Mapping[str, Any], state: str) -> float:
    return math.log(max(float(score["probabilities"][state]), 1e-12))


def boxes(width: int, height: int, grid_size: int) -> list[tuple[int, int, int, int]]:
    xs = [round(index * width / grid_size) for index in range(grid_size + 1)]
    ys = [round(index * height / grid_size) for index in range(grid_size + 1)]
    return [
        (xs[column], ys[row], xs[column + 1], ys[row + 1])
        for row in range(grid_size)
        for column in range(grid_size)
    ]


def replace_regions(base: Image.Image, donor: Image.Image, regions: Sequence[tuple[int, int, int, int]]) -> Image.Image:
    output = base.copy()
    for box in regions:
        output.paste(donor.crop(box), box)
    return output


def matched_random_indices(total: int, excluded: Sequence[int], count: int, key: str) -> list[int]:
    candidates = [index for index in range(total) if index not in set(excluded)]
    return sorted(candidates, key=lambda index: stable_key(key, index))[:count]


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None

    def ranks(values: Sequence[float]) -> np.ndarray:
        values_array = np.asarray(values, dtype=np.float64)
        order = np.argsort(values_array, kind="mergesort")
        result = np.empty(len(values_array), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values_array[order[end]] == values_array[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2
            start = end
        return result

    xr, yr = ranks(x), ranks(y)
    if xr.std() == 0 or yr.std() == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def analyze(records: Sequence[Mapping[str, Any]], source: Mapping[str, Mapping[str, Any]], seed: int, draws: int) -> dict[str, Any]:
    images = []
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if row.get("status") != "ok":
            continue
        baseline = row["baseline_scores"][SELECTION_PROMPT]
        error = int(baseline["state"] != row["expected_state"])
        target_js, random_js, sufficiency_js = [], [], []
        target_drop, random_drop, sufficiency_drop = [], [], []
        held_state_agreement = []
        for name in EVALUATION_PROMPTS:
            full = row["baseline_scores"][name]
            chosen = str(full["state"])
            interventions = row["evaluation_scores"][name]
            target_js.append(js_divergence((vector(full), vector(interventions["target_removed"]))))
            random_js.append(js_divergence((vector(full), vector(interventions["random_removed"]))))
            sufficiency_js.append(js_divergence((vector(full), vector(interventions["target_only"]))))
            target_drop.append(log_probability(full, chosen) - log_probability(interventions["target_removed"], chosen))
            random_drop.append(log_probability(full, chosen) - log_probability(interventions["random_removed"], chosen))
            sufficiency_drop.append(log_probability(full, chosen) - log_probability(interventions["target_only"], chosen))
            held_state_agreement.append(int(chosen == baseline["state"]))
        selective_necessity = float(np.mean(target_js) - np.mean(random_js))
        sufficiency_distance = float(np.mean(sufficiency_js))
        identifiability_proxy = selective_necessity - sufficiency_distance
        item = {
            "pair_id": row["pair_id"],
            "role": row["role"],
            "image": row["image"],
            "error": error,
            "entropy": entropy(vector(baseline)),
            "mean_target_removal_js": float(np.mean(target_js)),
            "mean_random_removal_js": float(np.mean(random_js)),
            "selective_necessity_js": selective_necessity,
            "mean_target_only_js": sufficiency_distance,
            "identifiability_proxy": identifiability_proxy,
            "selective_chosen_logprob_drop": float(np.mean(target_drop) - np.mean(random_drop)),
            "target_only_chosen_logprob_drop": float(np.mean(sufficiency_drop)),
            "held_prompt_state_agreement": float(np.mean(held_state_agreement)),
        }
        images.append(item)
        by_pair.setdefault(row["pair_id"], []).append(item)

    pair_rows = []
    for pair_id, members in by_pair.items():
        if len(members) != 2:
            continue
        source_row = source[pair_id]
        positive_vectors = np.asarray([vector(source_row["scores"]["positive"][name]) for name in PROMPT_TEMPLATES])
        negative_vectors = np.asarray([vector(source_row["scores"]["negative"][name]) for name in PROMPT_TEMPLATES])
        oracle = js_divergence((positive_vectors.mean(axis=0), negative_vectors.mean(axis=0)))
        pair_rows.append({
            "pair_id": pair_id,
            "any_error": int(any(member["error"] for member in members)),
            "oracle_between_image_js": oracle,
            "mean_identifiability_proxy": float(np.mean([member["identifiability_proxy"] for member in members])),
            "minimum_identifiability_proxy": float(np.min([member["identifiability_proxy"] for member in members])),
            "mean_entropy": float(np.mean([member["entropy"] for member in members])),
        })
    image_labels = [row["error"] for row in images]
    pair_labels = [row["any_error"] for row in pair_rows]
    metrics: dict[str, Any] = {
        "n_images": len(images),
        "n_pairs": len(pair_rows),
        "image_error_rate": float(np.mean(image_labels)),
        "pair_any_error_rate": float(np.mean(pair_labels)),
        "image_error_auroc": {
            "negative_identifiability_proxy": auc(image_labels, [-row["identifiability_proxy"] for row in images]),
            "negative_selective_necessity": auc(image_labels, [-row["selective_necessity_js"] for row in images]),
            "sufficiency_distance": auc(image_labels, [row["mean_target_only_js"] for row in images]),
            "entropy": auc(image_labels, [row["entropy"] for row in images]),
        },
        "pair_error_auroc": {
            "negative_mean_proxy": auc(pair_labels, [-row["mean_identifiability_proxy"] for row in pair_rows]),
            "negative_minimum_proxy": auc(pair_labels, [-row["minimum_identifiability_proxy"] for row in pair_rows]),
            "entropy": auc(pair_labels, [row["mean_entropy"] for row in pair_rows]),
        },
        "proxy_oracle_spearman": {
            "mean_proxy": spearman([row["mean_identifiability_proxy"] for row in pair_rows], [row["oracle_between_image_js"] for row in pair_rows]),
            "minimum_proxy": spearman([row["minimum_identifiability_proxy"] for row in pair_rows], [row["oracle_between_image_js"] for row in pair_rows]),
        },
    }
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(draws):
        indices = rng.integers(0, len(pair_rows), len(pair_rows))
        value = spearman(
            [pair_rows[index]["mean_identifiability_proxy"] for index in indices],
            [pair_rows[index]["oracle_between_image_js"] for index in indices],
        )
        if value is not None:
            correlations.append(value)
    metrics["mean_proxy_oracle_spearman_bootstrap"] = {
        "valid_draws": len(correlations),
        "ci_low": float(np.quantile(correlations, 0.025)) if correlations else None,
        "ci_high": float(np.quantile(correlations, 0.975)) if correlations else None,
    }
    return {"metrics": metrics, "derived_images": images, "derived_pairs": pair_rows}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.grid_size < 2 or not 0 < args.top_k < args.grid_size ** 2:
        raise ValueError("top-k must be positive and smaller than number of grid cells")
    args.output_dir.mkdir(parents=True)
    source_records = [row for row in read_jsonl(args.source_raw) if row.get("status") == "ok"][: args.limit_pairs]
    source = {str(row["pair_id"]): row for row in source_records}
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VQA-RAD natural exact-question counterfactual panel",
        "source_raw": str(args.source_raw.resolve()),
        "source_raw_sha256": sha256_file(args.source_raw),
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256_file(args.source_config),
        "image_root": str(args.image_root.resolve()),
        "model": str(args.model_dir.resolve()),
        "method": "inference-only label-free single-image selective-necessity and sufficiency proxy",
        "selection_prompt": SELECTION_PROMPT,
        "evaluation_prompts": EVALUATION_PROMPTS,
        "grid_size": args.grid_size,
        "top_k": args.top_k,
        "blur_fraction_of_short_side": args.blur_fraction,
        "pairs": len(source_records),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(args.output_dir / "config.json", config)
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    raw_path = args.output_dir / "raw.jsonl"
    tasks = [(pair, role) for pair in source_records for role in ("positive", "negative")]
    for task_index, (pair, role) in enumerate(tasks):
        image_name = str(pair[f"{role}_image"])
        record: dict[str, Any] = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "pair_id": pair["pair_id"],
            "question": pair["question"],
            "role": role,
            "image": image_name,
            "expected_state": "supported" if role == "positive" else "refuted",
            "status": "error",
        }
        try:
            image = Image.open(args.image_root / image_name).convert("RGB")
            blur = image.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(image.size) * args.blur_fraction)))
            regions = boxes(*image.size, args.grid_size)
            canonical = pair["scores"][role][SELECTION_PROMPT]
            selected_state = str(canonical["state"])
            selection_prompt = PROMPT_TEMPLATES[SELECTION_PROMPT].format(question=str(pair["question"]).strip())
            region_scores = []
            for index, box in enumerate(regions):
                ablated = replace_regions(image, blur, [box])
                score = score_real(bot, ablated, selection_prompt)
                region_scores.append({
                    "index": index,
                    "box": box,
                    "chosen_logprob_drop": log_probability(canonical, selected_state) - log_probability(score, selected_state),
                    "score": score,
                })
            target_indices = sorted(
                [entry["index"] for entry in sorted(region_scores, key=lambda entry: (-entry["chosen_logprob_drop"], entry["index"]))[: args.top_k]]
            )
            random_indices = matched_random_indices(len(regions), target_indices, args.top_k, stable_key(args.seed, pair["pair_id"], role))
            target_regions = [regions[index] for index in target_indices]
            random_regions = [regions[index] for index in random_indices]
            variants = {
                "target_removed": replace_regions(image, blur, target_regions),
                "random_removed": replace_regions(image, blur, random_regions),
                "target_only": replace_regions(blur, image, target_regions),
            }
            evaluation_scores = {}
            for name in EVALUATION_PROMPTS:
                prompt = PROMPT_TEMPLATES[name].format(question=str(pair["question"]).strip())
                evaluation_scores[name] = {variant: score_real(bot, altered, prompt) for variant, altered in variants.items()}
            record.update({
                "image_size": image.size,
                "blur_radius": max(1.0, min(image.size) * args.blur_fraction),
                "selected_state": selected_state,
                "region_scores": region_scores,
                "target_indices": target_indices,
                "random_indices": random_indices,
                "baseline_scores": pair["scores"][role],
                "evaluation_scores": evaluation_scores,
                "status": "ok",
            })
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record["error"] = f"CUDA OOM: {error}"
            record["traceback"] = traceback.format_exc()
        except Exception as error:
            record["error"] = repr(error)
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(json.dumps({"progress": f"{task_index + 1}/{len(tasks)}", "pair_id": pair["pair_id"], "role": role, "status": record["status"], "error": record.get("error")}), flush=True)
    records = read_jsonl(raw_path)
    summary = analyze(records, source, args.seed, args.bootstrap_draws)
    summary.update({"version": VERSION, "config": config, "runtime_errors": sum(row.get("status") != "ok" for row in records)})
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
