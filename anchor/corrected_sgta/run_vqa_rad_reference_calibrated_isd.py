#!/usr/bin/env python3
"""Reference-calibrated invariant solution decoding (RC-ISD) pilot.

For each target image/question, score the same prompt orbit with several
unlabelled, real in-domain reference images.  Answer scores are calibrated by
the answer distribution that survives image replacement, then intersected
across equivalent prompts.  This estimates and removes question-conditioned
language prior without corrupted/null images, training, or clinical rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from corrected_sgta.run_huatuo_vindr_commitment_probe import append_jsonl, atomic_json, import_huatuo, sha256_file
from corrected_sgta.run_vqa_rad_natural_counterfactual_pilot import score_real
from corrected_sgta.run_vqa_rad_underidentification_pilot import PROMPT_TEMPLATES, auc, stable_key


VERSION = "vqa-rad-reference-calibrated-isd-v1"
STATES = ("supported", "refuted", "undetermined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--limit-pairs", type=int, default=8)
    parser.add_argument("--references", type=int, default=3)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260814)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def vector(score: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(score["probabilities"][state]) for state in STATES], dtype=np.float64)


def select_references(records: list[dict[str, Any]], pair_id: str, image_name: str, count: int, seed: int) -> list[str]:
    candidates = sorted({
        str(row[f"{role}_image"])
        for row in records
        if str(row["pair_id"]) != pair_id
        for role in ("positive", "negative")
        if str(row[f"{role}_image"]) != image_name
    })
    ordered = sorted(candidates, key=lambda candidate: stable_key(seed, pair_id, image_name, candidate))
    if len(ordered) < count:
        raise ValueError(f"need {count} reference images, found {len(ordered)}")
    return ordered[:count]


def derive(record: Mapping[str, Any], prior_strength: float) -> dict[str, Any]:
    target = np.stack([vector(record["target_scores"][name]) for name in PROMPT_TEMPLATES])
    references = np.stack([
        np.stack([vector(scores[name]) for name in PROMPT_TEMPLATES])
        for scores in record["reference_scores"].values()
    ])
    target_log = np.log(np.maximum(target, 1e-12))
    reference_log = np.log(np.maximum(references, 1e-12))
    target_mean = target_log.mean(axis=0)
    reference_mean = reference_log.mean(axis=(0, 1))
    calibrated = target_mean - prior_strength * reference_mean
    blended = target_mean + calibrated
    promptwise_calibrated = target_log - prior_strength * reference_log.mean(axis=0)
    calibrated_index = int(np.argmax(calibrated))
    blended_index = int(np.argmax(blended))
    geometric_index = int(np.argmax(target_mean))
    factual_prompt_margins = promptwise_calibrated[:, 0] - promptwise_calibrated[:, 1]
    calibrated_margin = float(calibrated[0] - calibrated[1])
    robust_visual_margin = float(abs(factual_prompt_margins.mean()) - factual_prompt_margins.std())
    return {
        "canonical": STATES[int(np.argmax(target[0]))],
        "geometric_intersection": STATES[geometric_index],
        "pure_reference_calibrated": STATES[calibrated_index],
        "blended_reference_calibrated": STATES[blended_index],
        "calibrated_factual_candidate": "supported" if calibrated_margin >= 0 else "refuted",
        "calibrated_factual_margin": calibrated_margin,
        "promptwise_calibrated_margin_std": float(factual_prompt_margins.std()),
        "robust_visual_margin": robust_visual_margin,
        "target_reference_js_proxy": float(np.mean([
            0.5 * (
                np.sum(target[p] * np.log(np.maximum(target[p], 1e-12) / np.maximum(0.5 * (target[p] + references[:, p].mean(axis=0)), 1e-12)))
                + np.sum(references[:, p].mean(axis=0) * np.log(np.maximum(references[:, p].mean(axis=0), 1e-12) / np.maximum(0.5 * (target[p] + references[:, p].mean(axis=0)), 1e-12)))
            ) for p in range(len(PROMPT_TEMPLATES))
        ])),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    all_records = [row for row in read_jsonl(args.source_raw) if row.get("status") == "ok"]
    targets = all_records[: args.limit_pairs]
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VQA-RAD natural-image exact-question panel",
        "source_raw": str(args.source_raw.resolve()),
        "source_raw_sha256": sha256_file(args.source_raw),
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256_file(args.source_config),
        "image_root": str(args.image_root.resolve()),
        "model": str(args.model_dir.resolve()),
        "method": "inference-only domain-reference prior calibration x prompt-orbit invariant decoding",
        "reference_selection": "deterministic unlabelled images from different question pairs; labels never inspected",
        "references_per_target": args.references,
        "prompt_templates": PROMPT_TEMPLATES,
        "prior_strength": args.prior_strength,
        "pairs": len(targets),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(args.output_dir / "config.json", config)
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    raw_path = args.output_dir / "raw.jsonl"
    tasks = [(pair, role) for pair in targets for role in ("positive", "negative")]
    for index, (pair, role) in enumerate(tasks):
        image_name = str(pair[f"{role}_image"])
        record: dict[str, Any] = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "pair_id": pair["pair_id"],
            "role": role,
            "image": image_name,
            "question": pair["question"],
            "expected_state": "supported" if role == "positive" else "refuted",
            "status": "error",
        }
        try:
            reference_names = select_references(all_records, str(pair["pair_id"]), image_name, args.references, args.seed)
            reference_scores = {}
            for reference_name in reference_names:
                reference_image = Image.open(args.image_root / reference_name).convert("RGB")
                reference_scores[reference_name] = {
                    name: score_real(bot, reference_image, template.format(question=str(pair["question"]).strip()))
                    for name, template in PROMPT_TEMPLATES.items()
                }
            record.update({
                "reference_images": reference_names,
                "target_scores": pair["scores"][role],
                "reference_scores": reference_scores,
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
        print(json.dumps({"progress": f"{index + 1}/{len(tasks)}", "pair_id": pair["pair_id"], "role": role, "status": record["status"], "error": record.get("error")}), flush=True)

    raw = read_jsonl(raw_path)
    derived = []
    for row in raw:
        if row.get("status") != "ok":
            continue
        item = derive(row, args.prior_strength)
        item.update({key: row[key] for key in ("pair_id", "role", "image", "expected_state")})
        derived.append(item)
    methods = ("canonical", "geometric_intersection", "pure_reference_calibrated", "blended_reference_calibrated", "calibrated_factual_candidate")
    canonical_errors = [int(row["canonical"] != row["expected_state"]) for row in derived]
    metrics = {
        "n_images": len(derived),
        "runtime_errors": sum(row.get("status") != "ok" for row in raw),
        "accuracy": {method: float(np.mean([row[method] == row["expected_state"] for row in derived])) for method in methods},
        "canonical_error_detection_auroc": {
            "negative_robust_visual_margin": auc(canonical_errors, [-row["robust_visual_margin"] for row in derived]),
            "negative_absolute_calibrated_margin": auc(canonical_errors, [-abs(row["calibrated_factual_margin"]) for row in derived]),
            "target_reference_js_proxy": auc(canonical_errors, [-row["target_reference_js_proxy"] for row in derived]),
            "calibrated_prompt_instability": auc(canonical_errors, [row["promptwise_calibrated_margin_std"] for row in derived]),
        },
        "scientific_role": "small discovery probe; held-out questions, datasets, and models are required before any generality claim",
    }
    summary = {"version": VERSION, "config": config, "metrics": metrics, "derived_images": derived}
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
