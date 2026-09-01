#!/usr/bin/env python3
"""Test whether logical claim involution removes response-format bias.

The same atomic finding is presented as an affirmative statement ``c`` and
its logical complement ``not c``.  A semantic presence score averages the
affirmative Yes-vs-No margin with the sign-reversed complement margin.  This
crossing cancels a fixed Yes/No response-token preference and exposes whether
the image orders positive and negative cases after that nuisance is removed.

This is a screening measurement.  SLAKE labels are binary and cannot validate
reader uncertainty, OE mitigation, or clinical report generation.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    import_huatuo,
    sha256_file,
)
from corrected_sgta.run_hulu_vindr_commitment_probe import HuluRuntime
from corrected_sgta.run_llava_vindr_commitment_probe import LlavaRuntime
from corrected_sgta.run_slake_quantifier_coverage_probe import (
    DEFAULT_SLAKE,
    score_huatuo,
    score_hulu,
    score_llava,
    slake_rows,
)


VERSION = "slake-claim-involution-probe-v1"
PROMPTS = {
    "affirmative": (
        "Statement: This chest X-ray shows {finding}. Based only on the image, "
        "is this statement correct? Answer exactly Yes, No, or Maybe."
    ),
    "complement": (
        "Statement: This chest X-ray shows no {finding}. Based only on the image, "
        "is this statement correct? Answer exactly Yes, No, or Maybe."
    ),
}


def log_probability(score: Mapping[str, Any], state: str) -> float:
    return math.log(max(float(score["probabilities"][state]), 1e-30))


def involution_coordinates(
    affirmative: Mapping[str, Any], complement: Mapping[str, Any]
) -> dict[str, float]:
    affirmative_margin = (
        log_probability(affirmative, "supported")
        - log_probability(affirmative, "refuted")
    )
    # Yes supports absence and No refutes it, so the semantic-presence sign is
    # reversed for the complement statement.
    complement_presence_margin = (
        log_probability(complement, "refuted")
        - log_probability(complement, "supported")
    )
    semantic_margin = 0.5 * (affirmative_margin + complement_presence_margin)
    framing_disagreement = 0.5 * abs(
        affirmative_margin - complement_presence_margin
    )
    affirmative_commitment = (
        max(
            log_probability(affirmative, "supported"),
            log_probability(affirmative, "refuted"),
        )
        - log_probability(affirmative, "undetermined")
    )
    complement_commitment = (
        max(
            log_probability(complement, "supported"),
            log_probability(complement, "refuted"),
        )
        - log_probability(complement, "undetermined")
    )
    return {
        "affirmative_presence_margin": affirmative_margin,
        "complement_presence_margin": complement_presence_margin,
        "semantic_presence_margin": semantic_margin,
        "framing_disagreement": framing_disagreement,
        "mean_commitment": 0.5 * (
            affirmative_commitment + complement_commitment
        ),
    }


def binary_auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def bootstrap_auc_delta(
    labels: list[int],
    baseline: list[float],
    involution: list[float],
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(labels)
    for _ in range(draws):
        indices = rng.integers(0, n, n)
        sampled_labels = [labels[index] for index in indices]
        if len(set(sampled_labels)) < 2:
            continue
        deltas.append(
            binary_auc(
                sampled_labels, [involution[index] for index in indices]
            )
            - binary_auc(
                sampled_labels, [baseline[index] for index in indices]
            )
        )
    if not deltas:
        raise ValueError("no valid bootstrap resamples")
    return {
        "n": n,
        "estimate": binary_auc(labels, involution) - binary_auc(labels, baseline),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "valid_draws": len(deltas),
    }


def analyze(
    records: list[dict[str, Any]], seed: int, draws: int
) -> dict[str, Any]:
    rows = [row for row in records if row.get("status") == "ok"]
    if not rows:
        raise ValueError("no successful records")
    labels = [int(row["reference_polarity"] == "positive") for row in rows]
    baseline = [
        float(row["coordinates"]["affirmative_presence_margin"]) for row in rows
    ]
    involution = [
        float(row["coordinates"]["semantic_presence_margin"]) for row in rows
    ]
    delta = bootstrap_auc_delta(labels, baseline, involution, seed, draws)
    per_finding = {}
    for finding in sorted({str(row["finding"]) for row in rows}):
        subset = [row for row in rows if row["finding"] == finding]
        finding_labels = [
            int(row["reference_polarity"] == "positive") for row in subset
        ]
        finding_baseline = [
            float(row["coordinates"]["affirmative_presence_margin"])
            for row in subset
        ]
        finding_involution = [
            float(row["coordinates"]["semantic_presence_margin"])
            for row in subset
        ]
        per_finding[finding] = {
            "n": len(subset),
            "baseline_auroc": binary_auc(finding_labels, finding_baseline),
            "involution_auroc": binary_auc(finding_labels, finding_involution),
            "mean_framing_disagreement": float(np.mean([
                row["coordinates"]["framing_disagreement"] for row in subset
            ])),
        }
    baseline_auc = binary_auc(labels, baseline)
    involution_auc = binary_auc(labels, involution)
    return {
        "version": VERSION,
        "status": "complete",
        "n": len(rows),
        "n_errors": len(records) - len(rows),
        "baseline_affirmative_auroc": baseline_auc,
        "involution_auroc": involution_auc,
        "involution_minus_baseline_auroc": delta,
        "mean_framing_disagreement": float(np.mean([
            row["coordinates"]["framing_disagreement"] for row in rows
        ])),
        "sign_accuracy": {
            "baseline": float(np.mean([
                (score >= 0.0) == bool(label)
                for label, score in zip(labels, baseline)
            ])),
            "involution": float(np.mean([
                (score >= 0.0) == bool(label)
                for label, score in zip(labels, involution)
            ])),
        },
        "per_finding": per_finding,
        "screening_gate": {
            "involution_auroc_at_least_0_70": involution_auc >= 0.70,
            "improvement_at_least_0_05": delta["estimate"] >= 0.05,
            "improvement_ci_above_zero": delta["ci_low"] > 0.0,
        },
        "claim_ceiling": (
            "A positive result only shows that logical-frame symmetrization "
            "improves binary finding ordering on SLAKE. It does not establish "
            "reader uncertainty, OE hallucination reduction, or report quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("huatuo", "hulu", "llava_med"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slake-root", type=Path, default=DEFAULT_SLAKE)
    parser.add_argument(
        "--findings", nargs="+", default=["Effusion", "Pneumothorax"]
    )
    parser.add_argument("--per-finding", type=int, default=8)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=149)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = slake_rows(
        args.slake_root, args.findings, args.per_finding, args.seed, padding=0.10
    )
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "slake_root": str(args.slake_root.resolve()),
        "findings": args.findings,
        "per_finding": args.per_finding,
        "selection": "stable image-disjoint SLAKE X-ray positives and negatives",
        "prompts": PROMPTS,
        "seed": args.seed,
        "code_sha256": sha256_file(Path(__file__)),
        "evidence_grade": "screening: SLAKE disease boxes and detection labels",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.model == "huatuo":
        constructor = import_huatuo(Path("/home/dbw/HuatuoGPT-Vision"))
        runtime = constructor(
            "/home/dbw/models/HuatuoGPT-Vision-7B", device="cuda:0"
        )
        scorer = score_huatuo
    elif args.model == "hulu":
        runtime = HuluRuntime(
            Path("/home/dbw/models/Hulu-Med-4B"), args.max_visual_tokens
        )
        scorer = score_hulu
    else:
        runtime = LlavaRuntime(
            Path("/home/dbw/models/LLaVA-Med-v1.5-mistral-7b"),
            Path(
                "/home/dbw/ANCHOR/data/medheval/code/baselines/"
                "Med-LVLMs/llava-med-1.5"
            ),
            "mistral_instruct",
        )
        scorer = score_llava

    raw_path = args.output_dir / "raw.jsonl"
    records = []
    for index, row in enumerate(rows):
        record = {
            "version": VERSION,
            "case_id": row["case_id"],
            "image_path": row["image_path"],
            "finding": row["finding"],
            "finding_label": row["finding_label"],
            "reference_polarity": row["reference_polarity"],
            "status": "error",
        }
        try:
            with Image.open(row["image_path"]) as opened:
                image = opened.convert("RGB")
            scores = {
                name: scorer(runtime, image, prompt.format(finding=row["finding"]))
                for name, prompt in PROMPTS.items()
            }
            record["scores"] = scores
            record["coordinates"] = involution_coordinates(
                scores["affirmative"], scores["complement"]
            )
            record["status"] = "ok"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print(
            f"[{index + 1}/{len(rows)}] {row['case_id']} {row['finding']} "
            f"{record['status']}",
            flush=True,
        )
    summary = analyze(records, args.seed, args.bootstrap_draws)
    summary["config"] = config
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
