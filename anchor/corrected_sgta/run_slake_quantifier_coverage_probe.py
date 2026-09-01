#!/usr/bin/env python3
"""Screen the existential/universal evidence asymmetry on real lesion boxes.

For a positive medical finding, a localized region can witness presence.  A
definite absence claim is different: it requires the relevant anatomy to be
observable, not merely the disappearance of that witness.  This probe uses
SLAKE chest-X-ray lesion boxes to create a target occlusion and an equal-area
non-lesion control.  It asks whether a VLM turns loss of the positive witness
into ``No`` instead of the epistemically valid ``Maybe``.

This is a screening manipulation, not a clinical mitigation result.  The
occluded image is deliberately incomplete and the prompt states that fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image

from corrected_sgta.clinical_claims import epistemic_coordinates, softmax_states
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    sha256_file,
)
from corrected_sgta.run_hulu_vindr_commitment_probe import (
    HuluRuntime,
    prepared_embeddings_hulu,
)
from corrected_sgta.run_llava_vindr_commitment_probe import (
    LlavaRuntime,
    prepared_embeddings_llava,
)


VERSION = "slake-quantifier-coverage-probe-v1"
DEFAULT_SLAKE = Path("/home/dbw/ANCHOR/data/medheval/images/Slake")
FINDING_TEXT = {
    "Effusion": "pleural effusion",
    "Pneumothorax": "pneumothorax",
}
PROMPTS = {
    "ordinary": (
        "Does this chest X-ray show {finding}? "
        "Answer with exactly one word: Yes, No, or Maybe."
    ),
    "coverage_explicit": (
        "This chest X-ray may contain an intentionally obscured region. "
        "What is the status of {finding} in the underlying study, based only on "
        "the available evidence? Answer Yes if it is present, No if it is absent, "
        "or Maybe if obscured anatomy prevents determination. Answer exactly "
        "Yes, No, or Maybe."
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def normalize_box(
    raw_box: Iterable[float], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    values = [float(value) for value in raw_box]
    if len(values) != 4:
        raise ValueError(f"SLAKE boxes must be [x,y,w,h], got {values}")
    x, y, box_width, box_height = values
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"invalid non-positive box: {values}")
    pad_x = box_width * padding
    pad_y = box_height * padding
    left = max(0, math.floor(x - pad_x))
    top = max(0, math.floor(y - pad_y))
    right = min(width, math.ceil(x + box_width + pad_x))
    bottom = min(height, math.ceil(y + box_height + pad_y))
    if right <= left or bottom <= top:
        raise ValueError(f"box is outside image bounds: {values}, {(width, height)}")
    return left, top, right, bottom


def intersection_area(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def matched_control_box(
    target: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Choose an equal-shape in-frame box with minimum target overlap."""

    left, top, right, bottom = target
    box_width, box_height = right - left, bottom - top
    candidates = []
    for candidate_left in (0, max(0, width - box_width), max(0, width // 2 - box_width // 2)):
        for candidate_top in (0, max(0, height - box_height), max(0, height // 2 - box_height // 2)):
            candidate = (
                candidate_left,
                candidate_top,
                candidate_left + box_width,
                candidate_top + box_height,
            )
            if candidate[2] <= width and candidate[3] <= height:
                candidates.append(candidate)
    if not candidates:
        raise ValueError("cannot place an equal-area control box")
    return min(
        candidates,
        key=lambda box: (
            intersection_area(target, box),
            abs((box[0] + box[2]) - (left + right)),
            abs((box[1] + box[3]) - (top + bottom)),
            box,
        ),
    )


def mean_fill(
    image: Image.Image, boxes: Iterable[tuple[int, int, int, int]]
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = np.zeros(array.shape[:2], dtype=bool)
    for left, top, right, bottom in boxes:
        mask[top:bottom, left:right] = True
    if not mask.any() or mask.all():
        raise ValueError("occlusion mask must cover a strict non-empty image subset")
    fill = np.round(array[~mask].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    array[mask] = fill
    return Image.fromarray(array, mode="RGB")


def slake_rows(
    root: Path, findings: Iterable[str], per_finding: int, seed: int, padding: float
) -> list[dict[str, Any]]:
    """Select globally image-disjoint positive and negative rows per finding.

    SLAKE's detection file is treated as an exhaustive screening annotation for
    its eight X-ray abnormalities.  This remains weaker than a clinical
    multi-reader negative and is recorded as such in the output.
    """

    finding_list = list(findings)
    inventory: list[dict[str, Any]] = []
    for detection_path in root.glob("*/detection.json"):
        image_path = detection_path.with_name("source.jpg")
        question_path = detection_path.with_name("question.json")
        if not image_path.is_file() or not question_path.is_file():
            continue
        questions = load_json(question_path)
        if not any(
            row.get("q_lang") == "en" and row.get("modality") == "X-Ray"
            for row in questions
        ):
            continue
        detections = load_json(detection_path)
        labels = {
            str(label)
            for item in detections
            for label in item
        }
        with Image.open(image_path) as image:
            width, height = image.size
        inventory.append(
            {
                "case_id": detection_path.parent.name,
                "image_path": str(image_path.resolve()),
                "image_size": [width, height],
                "detections": detections,
                "labels": labels,
            }
        )

    selected: list[dict[str, Any]] = []
    used_cases: set[str] = set()
    for finding in finding_list:
        candidates = []
        for source in inventory:
            detections = source["detections"]
            raw_boxes = [
                box
                for item in detections
                for label, box in item.items()
                if str(label) == finding
            ]
            if not raw_boxes:
                continue
            width, height = source["image_size"]
            boxes = [normalize_box(box, width, height, padding) for box in raw_boxes]
            target = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            control = matched_control_box(target, width, height)
            candidates.append(
                {
                    "case_id": source["case_id"],
                    "image_path": source["image_path"],
                    "finding_label": finding,
                    "finding": FINDING_TEXT.get(finding, finding.lower()),
                    "reference_polarity": "positive",
                    "reference_labels": sorted(source["labels"]),
                    "image_size": [width, height],
                    "target_box": list(target),
                    "control_box": list(control),
                    "target_control_overlap": intersection_area(target, control),
                }
            )
        candidates.sort(key=lambda row: stable_key(seed, finding, row["case_id"]))
        candidates = [row for row in candidates if row["case_id"] not in used_cases]
        if len(candidates) < per_finding:
            raise RuntimeError(
                f"finding {finding!r} has only {len(candidates)} eligible cases"
            )
        chosen = candidates[:per_finding]
        selected.extend(chosen)
        used_cases.update(str(row["case_id"]) for row in chosen)

    # Select negatives only after positives so no image contributes to two
    # claims or polarities.  Requiring a different annotated abnormality avoids
    # an empty/failed annotation file masquerading as a negative.
    for finding in finding_list:
        candidates = []
        for source in inventory:
            if source["case_id"] in used_cases:
                continue
            labels = set(source["labels"])
            if finding in labels or not labels:
                continue
            candidates.append(
                {
                    "case_id": source["case_id"],
                    "image_path": source["image_path"],
                    "finding_label": finding,
                    "finding": FINDING_TEXT.get(finding, finding.lower()),
                    "reference_polarity": "negative",
                    "reference_labels": sorted(labels),
                    "image_size": list(source["image_size"]),
                    "target_box": None,
                    "control_box": None,
                    "target_control_overlap": None,
                }
            )
        candidates.sort(key=lambda row: stable_key(seed, finding, "negative", row["case_id"]))
        if len(candidates) < per_finding:
            raise RuntimeError(
                f"finding {finding!r} has only {len(candidates)} image-disjoint negatives"
            )
        chosen = candidates[:per_finding]
        selected.extend(chosen)
        used_cases.update(str(row["case_id"]) for row in chosen)
    return selected


def half_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    midpoint = width // 2
    if midpoint <= 0 or width - midpoint <= 0:
        raise ValueError("image is too narrow for half-field occlusion")
    # One center column may differ for odd widths; both boxes otherwise cover
    # the complete left/right field without overlap.
    return {
        "left_half_occlusion": (0, 0, midpoint, height),
        "right_half_occlusion": (midpoint, 0, width, height),
    }


@torch.inference_mode()
def score_huatuo(bot: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, _ = prepared_embeddings(bot, prompt, tensor)
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    logits = layer_logits(bot, hidden, [len(hidden) - 1], label_ids(bot))[
        len(hidden) - 1
    ]
    return score_payload(logits)


@torch.inference_mode()
def score_hulu(runtime: HuluRuntime, image: Image.Image, prompt: str) -> dict[str, Any]:
    embeddings, attention, positions, _ = prepared_embeddings_hulu(
        runtime, prompt, image
    )
    hidden = hidden_trajectory(runtime, embeddings, attention, positions)
    logits = layer_logits(runtime, hidden, [len(hidden) - 1], label_ids(runtime))[
        len(hidden) - 1
    ]
    return score_payload(logits)


@torch.inference_mode()
def score_llava(runtime: LlavaRuntime, image: Image.Image, prompt: str) -> dict[str, Any]:
    embeddings, attention, positions, _ = prepared_embeddings_llava(
        runtime, prompt, image
    )
    hidden = hidden_trajectory(runtime, embeddings, attention, positions)
    logits = layer_logits(runtime, hidden, [len(hidden) - 1], label_ids(runtime))[
        len(hidden) - 1
    ]
    return score_payload(logits)


def score_payload(logits: Mapping[str, float]) -> dict[str, Any]:
    probabilities = softmax_states(logits)
    return {
        "logits": dict(logits),
        "probabilities": probabilities,
        "state": max(probabilities, key=probabilities.get),
        "coordinates": epistemic_coordinates(logits),
    }


def bootstrap_mean(values: list[float], seed: int, draws: int) -> dict[str, Any]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [array[rng.integers(0, len(array), len(array))].mean() for _ in range(draws)]
    )
    return {
        "n": len(values),
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def state_margin(score: Mapping[str, Any], state: str) -> float:
    return float(score["logits"][state]) - float(score["logits"]["undetermined"])


def analyze(records: list[dict[str, Any]], seed: int, draws: int) -> dict[str, Any]:
    rows = [row for row in records if row.get("status") == "ok"]
    if not rows:
        raise ValueError("no successful records")
    positives = [row for row in rows if row["reference_polarity"] == "positive"]
    negatives = [row for row in rows if row["reference_polarity"] == "negative"]
    if not positives or not negatives:
        raise ValueError("both positive witness and negative coverage rows are required")
    attenuation = []
    refutation_transfer = []
    uncertainty_over_refutation = []
    ordinary_false_negation = []
    explicit_false_negation = []
    explicit_uncertainty = []
    original_supported = []
    per_finding: dict[str, dict[str, Any]] = {}
    for row in positives:
        ordinary = row["scores"]["ordinary"]
        explicit = row["scores"]["coverage_explicit"]
        attenuation.append(
            (state_margin(ordinary["original"], "supported")
             - state_margin(ordinary["target_occlusion"], "supported"))
            - (state_margin(ordinary["original"], "supported")
               - state_margin(ordinary["control_occlusion"], "supported"))
        )
        refute_shift = (
            (float(ordinary["target_occlusion"]["logits"]["refuted"])
             - float(ordinary["target_occlusion"]["logits"]["supported"]))
            - (float(ordinary["control_occlusion"]["logits"]["refuted"])
               - float(ordinary["control_occlusion"]["logits"]["supported"]))
        )
        unknown_shift = (
            (float(ordinary["target_occlusion"]["logits"]["undetermined"])
             - float(ordinary["target_occlusion"]["logits"]["supported"]))
            - (float(ordinary["control_occlusion"]["logits"]["undetermined"])
               - float(ordinary["control_occlusion"]["logits"]["supported"]))
        )
        refutation_transfer.append(refute_shift)
        uncertainty_over_refutation.append(unknown_shift - refute_shift)
        ordinary_false_negation.append(
            float(ordinary["target_occlusion"]["state"] == "refuted")
        )
        explicit_false_negation.append(
            float(explicit["target_occlusion"]["state"] == "refuted")
        )
        explicit_uncertainty.append(
            float(explicit["target_occlusion"]["state"] == "undetermined")
        )
        original_supported.append(float(ordinary["original"]["state"] == "supported"))

    full_negative_correct = []
    partial_negative_persistence = []
    partial_undetermined = []
    ordinary_partial_negative_persistence = []
    for row in negatives:
        ordinary = row["scores"]["ordinary"]
        explicit = row["scores"]["coverage_explicit"]
        full_is_negative = explicit["original"]["state"] == "refuted"
        full_negative_correct.append(float(full_is_negative))
        for variant in ("left_half_occlusion", "right_half_occlusion"):
            if full_is_negative:
                partial_negative_persistence.append(
                    float(explicit[variant]["state"] == "refuted")
                )
                partial_undetermined.append(
                    float(explicit[variant]["state"] == "undetermined")
                )
            if ordinary["original"]["state"] == "refuted":
                ordinary_partial_negative_persistence.append(
                    float(ordinary[variant]["state"] == "refuted")
                )

    for finding in sorted({str(row["finding"]) for row in rows}):
        subset = [row for row in rows if row["finding"] == finding]
        positive_subset = [row for row in subset if row["reference_polarity"] == "positive"]
        negative_subset = [row for row in subset if row["reference_polarity"] == "negative"]
        eligible_negative_subset = [
            row
            for row in negative_subset
            if row["scores"]["coverage_explicit"]["original"]["state"] == "refuted"
        ]
        per_finding[finding] = {
            "n_positive": len(positive_subset),
            "n_negative": len(negative_subset),
            "ordinary_original_supported_rate": float(np.mean([
                row["scores"]["ordinary"]["original"]["state"] == "supported"
                for row in positive_subset
            ])),
            "coverage_explicit_target_state_rates": {
                state: float(np.mean([
                    row["scores"]["coverage_explicit"]["target_occlusion"]["state"]
                    == state
                    for row in positive_subset
                ]))
                for state in ("supported", "refuted", "undetermined")
            },
            "coverage_explicit_full_negative_rate": float(np.mean([
                row["scores"]["coverage_explicit"]["original"]["state"] == "refuted"
                for row in negative_subset
            ])),
            "eligible_full_negative_cases": len(eligible_negative_subset),
            "coverage_explicit_partial_negative_persistence": (
                float(np.mean([
                    row["scores"]["coverage_explicit"][variant]["state"] == "refuted"
                    for row in eligible_negative_subset
                    for variant in ("left_half_occlusion", "right_half_occlusion")
                ]))
                if eligible_negative_subset
                else None
            ),
        }

    manipulation = bootstrap_mean(attenuation, seed + 1, draws)
    refutation_shift_summary = bootstrap_mean(refutation_transfer, seed + 2, draws)
    uncertainty_selectivity = bootstrap_mean(
        uncertainty_over_refutation, seed + 3, draws
    )
    negative_eligible = len(partial_negative_persistence)
    coverage_failure_rate = (
        float(np.mean(partial_negative_persistence)) if negative_eligible else None
    )
    coverage_uncertain_rate = (
        float(np.mean(partial_undetermined)) if negative_eligible else None
    )
    return {
        "version": VERSION,
        "status": "complete",
        "n": len(rows),
        "n_errors": len(records) - len(rows),
        "positive_witness_manipulation": {
            "estimand": (
                "target-minus-control attenuation of the supported-undetermined margin"
            ),
            **manipulation,
            "passed": manipulation["ci_low"] > 0.0,
        },
        "positive_evidence_removal_decomposition": {
            "uncertainty_transfer": {
                "definition": (
                    "target-minus-control increase in undetermined-supported logit gap"
                ),
                **manipulation,
            },
            "refutation_transfer": {
                "definition": (
                    "target-minus-control increase in refuted-supported logit gap"
                ),
                **refutation_shift_summary,
            },
            "uncertainty_minus_refutation_transfer": {
                "definition": (
                    "positive means evidence removal favors the third state over "
                    "the opposite polarity"
                ),
                **uncertainty_selectivity,
            },
            "ordinary_target_undetermined_argmax_rate": float(np.mean([
                row["scores"]["ordinary"]["target_occlusion"]["state"]
                == "undetermined"
                for row in positives
            ])),
            "ordinary_target_definite_argmax_rate": float(np.mean([
                row["scores"]["ordinary"]["target_occlusion"]["state"]
                != "undetermined"
                for row in positives
            ])),
        },
        "state_rates": {
            "ordinary_original_supported": float(np.mean(original_supported)),
            "ordinary_target_definite_refutation": float(
                np.mean(ordinary_false_negation)
            ),
            "coverage_explicit_target_definite_refutation": float(
                np.mean(explicit_false_negation)
            ),
            "coverage_explicit_target_undetermined": float(
                np.mean(explicit_uncertainty)
            ),
        },
        "negative_coverage_test": {
            "reference": (
                "SLAKE target absent from exhaustive screening detection labels; "
                "another abnormality is annotated"
            ),
            "full_image_negative_rate_under_coverage_prompt": float(
                np.mean(full_negative_correct)
            ),
            "eligible_full_negative_cases": int(sum(full_negative_correct)),
            "eligible_partial_views": negative_eligible,
            "partial_definite_negative_persistence": coverage_failure_rate,
            "partial_undetermined_rate": coverage_uncertain_rate,
            "ordinary_prompt_partial_negative_persistence": (
                float(np.mean(ordinary_partial_negative_persistence))
                if ordinary_partial_negative_persistence
                else None
            ),
            "interpretation": (
                "A left or right half-field occlusion makes a global absence "
                "claim uncertifiable even when the complete image is negative."
            ),
        },
        "per_finding": per_finding,
        "screening_gate": {
            "witness_is_localized": manipulation["ci_low"] > 0.0,
            "uncertainty_transfer_ci_above_zero": manipulation["ci_low"] > 0.0,
            "uncertainty_preferred_over_refutation": (
                uncertainty_selectivity["estimate"] > 0.0
            ),
            "third_state_not_selected_on_majority": float(np.mean([
                row["scores"]["ordinary"]["target_occlusion"]["state"]
                == "undetermined"
                for row in positives
            ])) < 0.5,
            "latent_but_unselected_third_state_pattern": bool(
                manipulation["ci_low"] > 0.0
                and uncertainty_selectivity["estimate"] > 0.0
                and float(np.mean([
                    row["scores"]["ordinary"]["target_occlusion"]["state"]
                    == "undetermined"
                    for row in positives
                ])) < 0.5
            ),
            "negative_arm_has_minimum_eligible_cases": int(sum(full_negative_correct)) >= 8,
            "coverage_failure_observed": bool(
                manipulation["ci_low"] > 0.0
                and int(sum(full_negative_correct)) >= 8
                and coverage_failure_rate is not None
                and coverage_failure_rate >= 0.5
            ),
            "prompt_alone_repairs_majority": bool(
                coverage_uncertain_rate is not None
                and coverage_uncertain_rate >= 0.5
                and coverage_failure_rate is not None
                and coverage_failure_rate < (
                    float(np.mean(ordinary_partial_negative_persistence))
                    if ordinary_partial_negative_persistence
                    else 0.0
                )
            ),
        },
        "claim_ceiling": (
            "A joint positive result shows localized positive witnessing plus "
            "persistence of a full-field negative after half the relevant anatomy "
            "is hidden. It does not validate clinical negatives or a report decoder."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu", "llava_med"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slake-root", type=Path, default=DEFAULT_SLAKE)
    parser.add_argument(
        "--findings", nargs="+", default=["Effusion", "Pneumothorax"]
    )
    parser.add_argument("--per-finding", type=int, default=8)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.per_finding <= 0 or not 0.0 <= args.padding <= 1.0:
        raise ValueError("invalid sample count or padding")
    args.output_dir.mkdir(parents=True)
    rows = slake_rows(
        args.slake_root, args.findings, args.per_finding, args.seed, args.padding
    )
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "slake_root": str(args.slake_root.resolve()),
        "findings": args.findings,
        "per_finding": args.per_finding,
        "padding": args.padding,
        "selection": "stable hash over SLAKE X-ray cases with real lesion boxes",
        "prompts": PROMPTS,
        "variants": {
            "original": "unaltered image",
            "positive/target_occlusion": "mean-fill padded ground-truth lesion box",
            "positive/control_occlusion": "mean-fill equal-shape minimum-overlap box",
            "negative/left_half_occlusion": "mean-fill the complete left half-field",
            "negative/right_half_occlusion": "mean-fill the complete right half-field",
        },
        "evidence_grade": (
            "screening: SLAKE disease boxes and exhaustive detection-file negatives; "
            "deliberate partial-observation intervention"
        ),
        "seed": args.seed,
        "code_sha256": sha256_file(Path(__file__)),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.model == "huatuo":
        constructor = import_huatuo(Path("/home/dbw/HuatuoGPT-Vision"))
        runtime = constructor("/home/dbw/models/HuatuoGPT-Vision-7B", device="cuda:0")
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
        record = {"version": VERSION, **row, "status": "error"}
        try:
            with Image.open(row["image_path"]) as opened:
                original = opened.convert("RGB")
            if row["reference_polarity"] == "positive":
                variants = {
                    "original": original,
                    "target_occlusion": mean_fill(
                        original, [tuple(row["target_box"])]
                    ),
                    "control_occlusion": mean_fill(
                        original, [tuple(row["control_box"])]
                    ),
                }
            else:
                width, height = row["image_size"]
                variants = {
                    "original": original,
                    **{
                        name: mean_fill(original, [box])
                        for name, box in half_boxes(width, height).items()
                    },
                }
            record["scores"] = {
                prompt_name: {
                    variant_name: scorer(
                        runtime,
                        image,
                        template.format(finding=row["finding"]),
                    )
                    for variant_name, image in variants.items()
                }
                for prompt_name, template in PROMPTS.items()
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
    summary = analyze(records, args.seed, args.bootstrap_draws)
    summary["config"] = config
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
