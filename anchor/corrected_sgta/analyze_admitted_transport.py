#!/usr/bin/env python3
"""Frozen-branch, paired holdout audit for fixed-K claim transport.

The admission branch is selected on calibration data and supplied explicitly.
This module never selects a branch from the holdout records.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_claim_reranking import aggregate, align, evaluate_image
from corrected_sgta.analyze_no_free_grounding import _binary, sha256_file


VERSION = "frozen-admitted-fixed-k-transport-holdout-v1"
ALLOWED_BRANCHES = {"original_margin", "null_centered_margin"}
METRICS = ("precision", "recall", "f1")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def images_from_embedded_drafts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [record for record in records if record.get("status") == "ok"]
    if not successful or any("draft" not in record for record in successful):
        raise ValueError("successful records with embedded bare-question drafts are required")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[int] = set()
    for record in successful:
        qid = int(record["question_id"])
        if qid in seen:
            raise ValueError(f"duplicate question_id={qid}")
        seen.add(qid)
        truth = _binary(record["truth"])
        if truth == "invalid":
            raise ValueError(f"invalid truth for qid={qid}")
        row = {
            "question_id": qid,
            "image": str(record["image"]),
            "truth": truth,
            "baseline": _binary(record["draft"].get("prediction")),
            "scores": {
                name: float(record["scores"][name])
                for name in ("original_margin", "null_margin", "null_centered_margin")
            },
        }
        grouped.setdefault(row["image"], []).append(row)
    images = []
    for name in sorted(grouped):
        image = evaluate_image(grouped[name])
        image["claim_signature"] = [
            [row["question_id"], row["truth"]]
            for row in sorted(grouped[name], key=lambda row: row["question_id"])
        ]
        images.append(image)
    return images


def images_from_external_draft(
    baseline_report: dict[str, Any],
    score_records: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = align(baseline_report, score_records, questions)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append(row)
    images = []
    for name in sorted(grouped):
        image = evaluate_image(grouped[name])
        image["claim_signature"] = [
            [row["question_id"], row["truth"]]
            for row in sorted(grouped[name], key=lambda row: row["question_id"])
        ]
        images.append(image)
    return images


def _index_images(images: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(image["image"]): image for image in images}
    if len(indexed) != len(images):
        raise ValueError("duplicate image clusters")
    return indexed


def _delta(sample: list[dict[str, Any]], branch: str) -> tuple[float, float, float]:
    candidate = aggregate(sample, branch)
    baseline = aggregate(sample, "baseline")
    return tuple(float(candidate[name] - baseline[name]) for name in METRICS)


def _bootstrap(
    indexed_by_model: dict[str, dict[str, dict[str, Any]]],
    branches: dict[str, str],
    *,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    image_names = sorted(next(iter(indexed_by_model.values())))

    def pooled(names: list[str]) -> tuple[float, float, float]:
        candidate_rows = []
        baseline_rows = []
        for name in names:
            for model, indexed in indexed_by_model.items():
                image = indexed[name]
                candidate_rows.append({"selected": image[branches[model]]})
                baseline_rows.append({"selected": image["baseline"]})

        def sum_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
            counts = {key: 0 for key in ("tp", "fp", "fn")}
            for row in rows:
                for key in counts:
                    counts[key] += int(row["selected"][key])
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            return {
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
            }

        left, right = sum_metrics(candidate_rows), sum_metrics(baseline_rows)
        return tuple(left[name] - right[name] for name in METRICS)

    observed = pooled(image_names)
    rng = np.random.default_rng(seed)
    samples = [[] for _ in METRICS]
    for _ in range(draws):
        chosen = rng.integers(0, len(image_names), len(image_names))
        values = pooled([image_names[index] for index in chosen])
        for target, value in zip(samples, values):
            if np.isfinite(value):
                target.append(float(value))
    return {
        name: {
            "estimate": float(observed[index]),
            "ci_low": float(np.quantile(samples[index], 0.025)),
            "ci_high": float(np.quantile(samples[index], 0.975)),
        }
        for index, name in enumerate(METRICS)
    }


def analyze(
    images_by_model: dict[str, list[dict[str, Any]]],
    branches: dict[str, str],
    *,
    draws: int,
    seed: int,
    material_drop: float = 0.01,
) -> dict[str, Any]:
    if set(images_by_model) != set(branches):
        raise ValueError("each model must have exactly one frozen branch")
    if any(branch not in ALLOWED_BRANCHES for branch in branches.values()):
        raise ValueError(f"branches must be among {sorted(ALLOWED_BRANCHES)}")
    indexed = {model: _index_images(images) for model, images in images_by_model.items()}
    image_sets = [set(rows) for rows in indexed.values()]
    if not image_sets or any(names != image_sets[0] for names in image_sets[1:]):
        raise ValueError("model holdouts are not aligned by image")
    if not image_sets[0]:
        raise ValueError("holdout images are required")
    for image_name in image_sets[0]:
        reference = next(iter(indexed.values()))[image_name]
        reference_shape = (
            reference["baseline"]["n_claims"],
            reference["baseline"]["n_true_claims"],
        )
        signatures = []
        for model, model_images in indexed.items():
            image = model_images[image_name]
            shape = (
                image["baseline"]["n_claims"],
                image["baseline"]["n_true_claims"],
            )
            if shape != reference_shape:
                raise ValueError(
                    f"claim universe differs for image={image_name}, model={model}"
                )
            signatures.append(image.get("claim_signature"))
        if any(signature is not None for signature in signatures):
            if any(signature is None for signature in signatures):
                raise ValueError("claim signatures are missing for some models")
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise ValueError(f"claim identities differ for image={image_name}")

    per_model = {}
    nonnegative = 0
    no_material_drop = True
    for model, images in images_by_model.items():
        branch = branches[model]
        baseline = aggregate(images, "baseline")
        candidate = aggregate(images, branch)
        delta = _delta(images, branch)
        if delta[0] >= 0 and delta[1] >= 0:
            nonnegative += 1
        if delta[0] < -material_drop or delta[1] < -material_drop:
            no_material_drop = False
        per_model[model] = {
            "frozen_branch": branch,
            "baseline": baseline,
            "candidate": candidate,
            "candidate_minus_baseline": dict(zip(METRICS, delta)),
            "tp_delta": int(candidate["tp"] - baseline["tp"]),
        }

    pooled = _bootstrap(indexed, branches, draws=draws, seed=seed)
    passed = (
        pooled["precision"]["ci_low"] > 0
        and pooled["recall"]["ci_low"] > 0
        and nonnegative >= 2
        and no_material_drop
    )
    return {
        "n_models": len(images_by_model),
        "n_paired_images": len(image_sets[0]),
        "frozen_branches": branches,
        "per_model": per_model,
        "pooled_paired_image_bootstrap": pooled,
        "holdout_gate": {
            "passed": passed,
            "rule": (
                "pooled precision and recall 95% CI lower bounds > 0; at least two "
                "models have nonnegative precision and recall point deltas; no model "
                f"drops by more than {material_drop:.3f} absolute on either metric"
            ),
            "nonnegative_models": nonnegative,
            "no_material_drop": no_material_drop,
            "material_drop_threshold": material_drop,
        },
        "resampling_unit": (
            "image; every sampled image contributes all model results, preserving "
            "cross-model dependence"
        ),
        "identity": "Per image and model, K is fixed, so delta(FP)=delta(FN)=-delta(TP).",
        "scope_ceiling": (
            "Grouped SLAKE binary claims are an OE claim-universe proxy, not natural "
            "free-text OE extraction or report generation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--llava-baseline", type=Path, required=True)
    parser.add_argument("--llava-scores", type=Path, required=True)
    parser.add_argument("--huatuo-scores", type=Path, required=True)
    parser.add_argument("--hulu-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=941)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    branches = protocol["frozen_branches"]
    questions = json.loads(args.questions.read_text())
    llava_scores = _load_jsonl(args.llava_scores)
    huatuo_scores = _load_jsonl(args.huatuo_scores)
    hulu_scores = _load_jsonl(args.hulu_scores)
    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": analyze(
            {
                "llava_med": images_from_external_draft(
                    json.loads(args.llava_baseline.read_text()), llava_scores, questions
                ),
                "huatuo": images_from_embedded_drafts(huatuo_scores),
                "hulu": images_from_embedded_drafts(hulu_scores),
            },
            branches,
            draws=args.bootstrap_draws,
            seed=args.seed,
            material_drop=float(protocol["material_drop_threshold"]),
        ),
        "provenance": {
            "protocol_sha256": sha256_file(args.protocol),
            "questions_sha256": sha256_file(args.questions),
            "llava_baseline_sha256": sha256_file(args.llava_baseline),
            "llava_scores_sha256": sha256_file(args.llava_scores),
            "huatuo_scores_sha256": sha256_file(args.huatuo_scores),
            "hulu_scores_sha256": sha256_file(args.hulu_scores),
            "code_sha256": sha256_file(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
