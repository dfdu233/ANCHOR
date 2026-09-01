#!/usr/bin/env python3
"""Screen whether a medical VLM applies prior-separable visual updates.

For a fixed clinical claim, explicit low/neutral/high pre-image priors are
crossed with real positive and negative images.  If language prior and image
evidence combine additively in log-odds, the positive-minus-negative image
contrast is invariant to the stated prior.  A prior-dependent contrast rejects
that separability, without pretending that any single model logit is an
identified clinical likelihood ratio.

This is a black-box evidence-update-law probe, not a calibration or mitigation
result.  SLAKE binary labels do not define reader uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

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


VERSION = "slake-prior-titration-probe-v3"
PRIORS = {"low": 0.10, "neutral": 0.50, "high": 0.90}
PROMPT = (
    "Before viewing the image, the stated background probability of {finding} "
    "in comparable patients is {percent}%. Update that background probability "
    "using the current chest X-ray. Is {finding} present in this image? "
    "Answer exactly Yes, No, or Maybe."
)


def polarity_margin(score: dict[str, Any]) -> float:
    logits = score["logits"]
    return float(logits["supported"]) - float(logits["refuted"])


def mean_bootstrap(
    values: list[float], seed: int, draws: int
) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap requires observations")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        array[rng.integers(0, len(array), len(array))].mean()
        for _ in range(draws)
    ])
    return {
        "n": len(values),
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def _strata(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return balanced finding x polarity strata for patient-level resampling."""
    ok = [row for row in records if row.get("status") == "ok"]
    keys = sorted({
        (str(row["finding"]), str(row["reference_polarity"])) for row in ok
    })
    strata = [
        [
            row
            for row in ok
            if (str(row["finding"]), str(row["reference_polarity"])) == key
        ]
        for key in keys
    ]
    if not strata or any(not stratum for stratum in strata):
        raise ValueError("stratified bootstrap requires non-empty strata")
    return strata


def stratified_mean_bootstrap(
    records: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    """Estimate an equal-stratum mean while resampling independent images."""
    strata = _strata(records)
    observed = float(np.mean([
        np.mean([value(row) for row in stratum]) for stratum in strata
    ]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        stratum_means = []
        for stratum in strata:
            indices = rng.integers(0, len(stratum), len(stratum))
            stratum_means.append(np.mean([value(stratum[index]) for index in indices]))
        samples.append(float(np.mean(stratum_means)))
    return {
        "n": sum(len(stratum) for stratum in strata),
        "n_strata": len(strata),
        "estimate": observed,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def stratified_contrast_bootstrap(
    records: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    """Macro-average positive-minus-negative effects over findings.

    Positive and negative patients are resampled independently. This avoids
    treating an arbitrary cross-patient pairing as a matched observation.
    """
    ok = [row for row in records if row.get("status") == "ok"]
    findings = sorted({str(row["finding"]) for row in ok})
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for finding in findings:
        grouped[finding] = {}
        for polarity in ("positive", "negative"):
            subset = [
                row for row in ok
                if str(row["finding"]) == finding
                and row["reference_polarity"] == polarity
            ]
            if not subset:
                raise ValueError(f"missing {polarity} rows for {finding}")
            grouped[finding][polarity] = subset

    def effect(sampled: dict[str, dict[str, list[dict[str, Any]]]]) -> float:
        per_finding = []
        for finding in findings:
            positive = sampled[finding]["positive"]
            negative = sampled[finding]["negative"]
            per_finding.append(
                np.mean([value(row) for row in positive])
                - np.mean([value(row) for row in negative])
            )
        return float(np.mean(per_finding))

    observed = effect(grouped)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        sampled: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for finding in findings:
            sampled[finding] = {}
            for polarity in ("positive", "negative"):
                rows = grouped[finding][polarity]
                indices = rng.integers(0, len(rows), len(rows))
                sampled[finding][polarity] = [rows[index] for index in indices]
        samples.append(effect(sampled))
    return {
        "n": len(ok),
        "n_findings": len(findings),
        "estimate": observed,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def pair_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in records if row.get("status") == "ok"]
    pairs = []
    for finding in sorted({str(row["finding"]) for row in rows}):
        positives = sorted(
            [
                row
                for row in rows
                if row["finding"] == finding
                and row["reference_polarity"] == "positive"
            ],
            key=lambda row: str(row["case_id"]),
        )
        negatives = sorted(
            [
                row
                for row in rows
                if row["finding"] == finding
                and row["reference_polarity"] == "negative"
            ],
            key=lambda row: str(row["case_id"]),
        )
        if len(positives) != len(negatives):
            raise ValueError(f"unbalanced positive/negative rows for {finding}")
        for positive, negative in zip(positives, negatives):
            contrast = {
                prior_name: polarity_margin(positive["scores"][prior_name])
                - polarity_margin(negative["scores"][prior_name])
                for prior_name in PRIORS
            }
            pairs.append({
                "finding": finding,
                "positive_case_id": positive["case_id"],
                "negative_case_id": negative["case_id"],
                "contrast": contrast,
                "low_to_high_interaction": contrast["high"] - contrast["low"],
                "curvature": (
                    contrast["high"] + contrast["low"]
                    - 2.0 * contrast["neutral"]
                ),
            })
    return pairs


def analyze(
    records: list[dict[str, Any]],
    blank_scores: dict[str, dict[str, Any]],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    pairs = pair_rows(records)
    blank_margins = {
        name: polarity_margin(score) for name, score in blank_scores.items()
    }
    slope_denominator = math.log(0.9 / 0.1) - math.log(0.1 / 0.9)

    def margin(row: dict[str, Any], prior: str) -> float:
        return polarity_margin(row["scores"][prior])

    contrast_stats = {
        name: stratified_contrast_bootstrap(
            records,
            lambda row, prior=name: margin(row, prior),
            seed + 10 + index,
            draws,
        )
        for index, name in enumerate(PRIORS)
    }
    clinical = contrast_stats["neutral"]
    interaction = stratified_contrast_bootstrap(
        records,
        lambda row: margin(row, "high") - margin(row, "low"),
        seed + 2,
        draws,
    )
    curvature = stratified_contrast_bootstrap(
        records,
        lambda row: (
            margin(row, "high") + margin(row, "low")
            - 2.0 * margin(row, "neutral")
        ),
        seed + 3,
        draws,
    )
    slope_gap = {
        key: (value / slope_denominator if key in {"estimate", "ci_low", "ci_high"} else value)
        for key, value in interaction.items()
    }

    prior_response = {
        "low_to_neutral": stratified_mean_bootstrap(
            records,
            lambda row: margin(row, "neutral") - margin(row, "low"),
            seed + 20,
            draws,
        ),
        "neutral_to_high": stratified_mean_bootstrap(
            records,
            lambda row: margin(row, "high") - margin(row, "neutral"),
            seed + 21,
            draws,
        ),
        "low_to_high": stratified_mean_bootstrap(
            records,
            lambda row: margin(row, "high") - margin(row, "low"),
            seed + 22,
            draws,
        ),
    }
    class_prior_response = {
        polarity: stratified_mean_bootstrap(
            [
                row for row in records
                if row.get("status") == "ok"
                and row["reference_polarity"] == polarity
            ],
            lambda row: margin(row, "high") - margin(row, "low"),
            seed + 23 + index,
            draws,
        )
        for index, polarity in enumerate(("positive", "negative"))
    }

    per_finding = {}
    for finding in sorted({str(pair["finding"]) for pair in pairs}):
        subset = [pair for pair in pairs if pair["finding"] == finding]
        per_finding[finding] = {
            "n_pairs": len(subset),
            "mean_contrast": {
                name: float(np.mean([
                    pair["contrast"][name] for pair in subset
                ]))
                for name in PRIORS
            },
            "mean_low_to_high_interaction": float(np.mean([
                pair["low_to_high_interaction"] for pair in subset
            ])),
            "mean_absolute_interaction": float(np.mean([
                abs(pair["low_to_high_interaction"]) for pair in subset
            ])),
        }

    absolute_interaction = float(np.mean([
        abs(float(pair["low_to_high_interaction"])) for pair in pairs
    ]))
    prior_check = blank_margins["high"] - blank_margins["low"]
    tolerance = 0.25 * max(abs(float(clinical["estimate"])), 1.0)
    manipulation_detectable = (
        prior_response["low_to_high"]["ci_low"] > 0.0
        or prior_response["low_to_high"]["ci_high"] < 0.0
    )
    semantically_aligned = (
        prior_response["low_to_high"]["ci_low"] > 0.0
        and prior_response["low_to_neutral"]["estimate"] >= 0.0
        and prior_response["neutral_to_high"]["estimate"] >= 0.0
    )
    return {
        "version": VERSION,
        "status": "complete",
        "n": sum(row.get("status") == "ok" for row in records),
        "n_errors": sum(row.get("status") != "ok" for row in records),
        "n_image_pairs": len(pairs),
        "blank_prior_manipulation": {
            "margins": blank_margins,
            "high_minus_low": prior_check,
            "monotonic": (
                blank_margins["low"] <= blank_margins["neutral"]
                <= blank_margins["high"]
            ),
            "role": (
                "secondary prompt-compliance diagnostic only; the OOD gray "
                "image is not a visual null and does not define the gate"
            ),
        },
        "real_image_prior_response": {
            **prior_response,
            "by_reference_polarity_low_to_high": class_prior_response,
            "manipulation_detectable": manipulation_detectable,
            "semantically_aligned": semantically_aligned,
            "direction_reversed": prior_response["low_to_high"]["ci_high"] < 0.0,
            "definition": (
                "equal-stratum mean response to stated prior over real images; "
                "95% CI uses patient-level resampling within finding x polarity"
            ),
        },
        "positive_minus_negative_image_contrast": contrast_stats,
        "neutral_prior_clinical_contrast": clinical,
        "prior_by_image_interaction": {
            "low_to_high_contrast_change": interaction,
            "positive_minus_negative_prior_slope_gap": slope_gap,
            "contrast_curvature": curvature,
            "mean_absolute_arbitrarily_paired_interaction_diagnostic": absolute_interaction,
            "predeclared_screening_tolerance": tolerance,
            "equivalence_ci_within_tolerance": (
                interaction["ci_low"] > -tolerance
                and interaction["ci_high"] < tolerance
                and curvature["ci_low"] > -tolerance
                and curvature["ci_high"] < tolerance
            ),
        },
        "per_finding": per_finding,
        "pairs": pairs,
        "screening_gate": {
            "prior_manipulation_is_detectable": manipulation_detectable,
            "prior_prompt_is_semantically_aligned": semantically_aligned,
            "clinical_contrast_is_positive": clinical["ci_low"] > 0.0,
            "prior_separability_supported": (
                semantically_aligned
                and clinical["ci_low"] > 0.0
                and interaction["ci_low"] > -tolerance
                and interaction["ci_high"] < tolerance
                and curvature["ci_low"] > -tolerance
                and curvature["ci_high"] < tolerance
            ),
        },
        "claim_ceiling": (
            "This screen can reject or provisionally support an additive "
            "prior-plus-image log-odds law. It does not identify a biological "
            "likelihood ratio, reader ambiguity, or hallucination mitigation."
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
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = slake_rows(
        args.slake_root, args.findings, args.per_finding, args.seed, padding=0.10
    )
    prompts = {
        name: PROMPT.format(
            finding="{finding}", percent=int(round(probability * 100))
        )
        for name, probability in PRIORS.items()
    }
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "slake_root": str(args.slake_root.resolve()),
        "findings": args.findings,
        "per_finding": args.per_finding,
        "priors": PRIORS,
        "prompts": prompts,
        "selection": "stable image-disjoint SLAKE X-ray positives and negatives",
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

    blank = Image.new("RGB", (512, 512), (128, 128, 128))
    blank_finding = str(rows[0]["finding"])
    blank_scores = {
        name: scorer(runtime, blank, template.format(finding=blank_finding))
        for name, template in prompts.items()
    }
    (args.output_dir / "blank_scores.json").write_text(
        json.dumps(blank_scores, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

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
            record["scores"] = {
                name: scorer(
                    runtime, image, template.format(finding=row["finding"])
                )
                for name, template in prompts.items()
            }
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
    summary = analyze(records, blank_scores, args.seed, args.bootstrap_draws)
    summary["config"] = config
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
