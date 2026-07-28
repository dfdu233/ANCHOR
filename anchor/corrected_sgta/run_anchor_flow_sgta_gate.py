#!/usr/bin/env python3
"""ANCHOR-Flow-SGTA pilot runner.

This script tests a deliberately small question:

Do source-style image views create useful *generated sentence* candidates, and
can a source-success output-path energy select a better complete response?

It is intentionally separate from the older SGTA, Riemann, and ConfGen runners.
Selection never reads target labels and never uses canonical Yes/No logits.
References are carried only for the final gate analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.anchor_models import NULL_RGB, load_anchor_adapter
from corrected_sgta.anchor_transport import (
    DEFAULT_PROJECTION_SEED,
    DEFAULT_QUANTILES,
    FEATURE_NAMES,
    VERSION as TRANSPORT_VERSION,
    deterministic_directions,
    file_sha256,
    load_json_or_jsonl,
    model_artifact_fingerprint,
    nearest_source_indices,
    normalize_manifest_record,
    normalize_trajectory,
    resolve_image_path,
    stable_json_sha256,
)
from corrected_sgta.cache import repair_truncated_jsonl_tail
from corrected_sgta.evaluate_medheval_answers import evaluate_rows, rule_pope_prediction
from corrected_sgta.frequency_alignment_source_spectrum import source_spectrum_alignment
from corrected_sgta.frequency_alignment_v2 import feddg_frequency_interpolation_v2
from corrected_sgta.infer_ce import _structure_metrics, resize_image
from corrected_sgta.infer_oe import report_prompt
from corrected_sgta.methods import gamma_transform
from corrected_sgta.protocol_v2 import build_prompt, prediction_index, task_kind
from corrected_sgta.riemann_geometry import nearest_manifold_distance, zscore


ImageFile.LOAD_TRUNCATED_IMAGES = True

RUN_VERSION = "anchor-flow-sgta-gate-v1"
DEFAULT_BANK = Path("corrected_runs/final_anchor_riemann_gate_v1/source_bank.json")
DEFAULT_CENTER = Path("/root/autodl-tmp/multimodal_center_report/centers/pubmedvision_xray.npy")
DEFAULT_OUTPUT_DIR = Path("corrected_runs/final_anchor_flow_sgta_gate_v1")
REPORT_PROMPT = (
    "You are a professional radiologist. You are provided with a chest X-ray "
    "image. Write a concise report with Findings and Impression sections. "
    "Only describe findings visible in the image."
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def per_item_seed(seed: int, identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
    return int(digest[:8], 16)


def load_bank(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("method_version") != TRANSPORT_VERSION:
        raise RuntimeError("source bank method-version mismatch")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("source bank evidence-feature mismatch")
    if payload.get("source_answer_text_stored") is not False:
        raise RuntimeError("source bank must not store source answer text")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source bank contains no records")
    return payload


def normalize_target_record(
    row: dict[str, Any],
    *,
    task: str,
    default_domain: str,
    default_prompt: str,
    require_answer: bool = True,
) -> dict[str, Any]:
    if task == "ce":
        prompt = build_prompt(row).replace("<image>", "").strip()
        answer = str(row.get("answer", row.get("gt", row.get("gt_ans", ""))).strip())
        image = row.get("image", row.get("image_path", row.get("img_name")))
        if isinstance(image, list):
            image = image[0] if image else None
        identifier = row.get("id", row.get("qid", row.get("question_id")))
        patient = row.get("patient_id", row.get("subject_id", identifier))
        missing = [
            name
            for name, value in (
                ("id", identifier),
                ("image", image),
                ("answer", answer if require_answer else "ok"),
            )
            if value is None or not str(value).strip()
        ]
        if missing:
            raise ValueError(f"CE row missing required fields: {missing}")
        return {
            "id": str(identifier),
            "image": str(image),
            "prompt": prompt,
            "answer": answer,
            "domain": str(row.get("domain", row.get("dataset", default_domain))),
            "patient_id": str(patient),
            "question_type": "binary" if answer.lower().strip(" .") in {"yes", "no"} else str(row.get("question_type") or task_kind(row)),
            "raw": row,
        }
    try:
        normalized = normalize_manifest_record(
            row, require_answer=require_answer, default_domain=default_domain
        )
    except ValueError:
        image = row.get("image", row.get("image_path", row.get("img_name")))
        if isinstance(image, list):
            image = image[0] if image else None
        answer = row.get("answer", row.get("reference", row.get("report", row.get("gt"))))
        identifier = row.get("id", row.get("qid", row.get("study_id")))
        patient = row.get("patient_id", row.get("subject_id", identifier))
        missing = [
            name
            for name, value in (
                ("id", identifier),
                ("image", image),
                ("answer", answer if require_answer else "ok"),
            )
            if value is None or not str(value).strip()
        ]
        if missing:
            raise ValueError(f"OE row missing required fields: {missing}")
        normalized = {
            "id": str(identifier),
            "image": str(image),
            "prompt": str(row.get("prompt") or row.get("question") or default_prompt).replace("<image>", "").strip(),
            "answer": str(answer).strip() if require_answer else "",
            "domain": str(row.get("domain", row.get("dataset", default_domain))),
            "patient_id": str(patient),
        }
    normalized["prompt"] = report_prompt(
        {
            "question": normalized["prompt"],
            "dataset": normalized["domain"],
            "domain": normalized["domain"],
            "task": "report",
        },
        "structured",
    )
    normalized["question_type"] = "open"
    normalized["raw"] = row
    return normalized


def make_views(
    image: Image.Image,
    *,
    center: np.ndarray,
    max_side: int,
    spectrum_alpha: float,
    low_frequency_ratio: float,
    gamma: float,
) -> list[dict[str, Any]]:
    original = resize_image(image, max_side)
    specs: list[tuple[str, Image.Image, dict[str, Any]]] = [
        ("original", original, {"family": "original", "parameters": {}}),
        (
            "sgta_source_spectrum",
            source_spectrum_alignment(
                original, center, low_frequency_ratio=spectrum_alpha, source_ratio=0.0
            ),
            {
                "family": "sgta_source_spectrum",
                "parameters": {"spectral_alpha": spectrum_alpha},
            },
        ),
        (
            "sgta_low_frequency",
            feddg_frequency_interpolation_v2(
                original, center, low_frequency_ratio=low_frequency_ratio, source_ratio=0.0
            ),
            {
                "family": "sgta_low_frequency",
                "parameters": {"low_frequency_ratio": low_frequency_ratio},
            },
        ),
        (
            "gamma",
            gamma_transform(original, gamma),
            {"family": "gamma", "parameters": {"gamma": gamma}},
        ),
    ]
    output = []
    for index, (name, view, meta) in enumerate(specs):
        structure = (
            {"pixel_mse": 0.0, "psnr": None, "edge_correlation": 1.0}
            if index == 0
            else _structure_metrics(original, view)
        )
        output.append(
            {
                "name": name,
                "image": view,
                "metadata": {
                    **meta,
                    "domain_id": "xray_source_proxy" if index else "original",
                    "structure": structure,
                },
            }
        )
    return output


def normal_template(text: str) -> bool:
    value = " ".join(str(text).lower().split())
    return (
        "appears to be normal" in value
        or "no acute cardiopulmonary" in value
        or "clear lungs" in value and "no " in value
    )


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = str(prediction).lower().split()
    ref = str(reference).lower().split()
    if not pred or not ref:
        return 0.0
    dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i, token in enumerate(pred, start=1):
        row = dp[i]
        prev = dp[i - 1]
        for j, ref_token in enumerate(ref, start=1):
            row[j] = prev[j - 1] + 1 if token == ref_token else max(prev[j], row[j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_text(text: str, row: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "ce":
        sample = row["raw"]
        pred = prediction_index(text, sample)
        gt = prediction_index(row["answer"], sample)
        # Fall back to RULE/POPE first-sentence semantics for generated binary sentences.
        if pred is None and str(row.get("question_type", "")).lower() == "binary":
            pope_pred = rule_pope_prediction(text)
            pope_gt = rule_pope_prediction(row["answer"])
            if pope_pred is not None and pope_gt is not None:
                return {
                    "metric": "accuracy",
                    "score": float(pope_pred == pope_gt),
                    "correct": bool(pope_pred == pope_gt),
                    "parseable": True,
                    "parsed_answer": pope_pred,
                    "parser": "rule_pope_negative_word",
                }
        # Fall back to the robust generated-answer evaluator for irregular rows.
        if pred is None or gt is None:
            detail = evaluate_rows(
                [
                    {
                        "qid": row["id"],
                        "question": row["prompt"],
                        "ground_truth": row["answer"],
                        "text": text,
                        "question_type": row.get("question_type", "binary"),
                    }
                ]
            )["details"][0]
            return {
                "metric": "accuracy",
                "score": float(detail["correct"]),
                "correct": bool(detail["correct"]),
                "parseable": detail["prediction"] is not None,
                "parsed_answer": detail["prediction"],
            }
        return {
            "metric": "accuracy",
            "score": float(pred == gt),
            "correct": bool(pred == gt),
            "parseable": True,
            "parsed_answer": pred,
        }
    score = rouge_l_f1(text, row["answer"])
    return {
        "metric": "rouge_l",
        "score": score,
        "correct": None,
        "parseable": True,
        "parsed_answer": None,
        "normal_template": normal_template(text),
    }


def summarize_records(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    methods = ("greedy", "nll", "random", "anchor_flow", "oracle")
    summary: dict[str, Any] = {"n": n, "task": task}
    for method in methods:
        scores = [float(record[f"{method}_score"]) for record in records]
        summary[method] = {
            "mean_score": float(np.mean(scores)),
            "score_delta_vs_greedy": float(np.mean(scores) - np.mean([r["greedy_score"] for r in records])),
        }
        if task == "ce":
            summary[method]["accuracy"] = float(np.mean(scores))
    unique_rates = [record["unique_text_count"] / max(1, len(record["candidates"])) for record in records]
    summary["style_diversity"] = {
        "mean_unique_text_rate": float(np.mean(unique_rates)),
        "view_disagreement_rate": float(np.mean([record["view_disagreement"] for record in records])),
        "oracle_headroom": float(summary["oracle"]["mean_score"] - summary["greedy"]["mean_score"]),
    }
    rescue = sum(1 for r in records if r["anchor_flow_score"] > r["greedy_score"])
    harm = sum(1 for r in records if r["anchor_flow_score"] < r["greedy_score"])
    summary["anchor_flow_rescue_harm"] = {"rescue": rescue, "harm": harm, "net": rescue - harm}
    if task == "oe":
        summary["normal_template_rate"] = {
            "greedy": float(np.mean([normal_template(r["greedy_text"]) for r in records])),
            "anchor_flow": float(np.mean([normal_template(r["anchor_flow_text"]) for r in records])),
        }
    summary["gate_decision"] = {
        "candidate_oracle_pass": bool(summary["style_diversity"]["oracle_headroom"] >= (0.05 if task == "ce" else 0.01)),
        "selection_beats_nll": bool(summary["anchor_flow"]["mean_score"] > summary["nll"]["mean_score"]),
        "selection_beats_random": bool(summary["anchor_flow"]["mean_score"] > summary["random"]["mean_score"]),
        "note": "OE uses ROUGE-L sanity unless clinical metrics are restored.",
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--default-domain", default="unknown")
    parser.add_argument("--task", choices=("ce", "oe"), required=True)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--center", type=Path, default=DEFAULT_CENTER)
    parser.add_argument("--model", choices=("hulu", "llava"), default="llava")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--candidate-batch", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--projections", type=int, default=32)
    parser.add_argument("--quantiles", type=int, default=DEFAULT_QUANTILES)
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--spectrum-alpha", type=float, default=0.01)
    parser.add_argument("--low-frequency-ratio", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 0 or args.max_new_tokens <= 0 or args.lambda_value < 0:
        raise ValueError("invalid sample/generation arguments")
    bank = load_bank(args.bank)
    if bank["model"] != args.model:
        raise RuntimeError("source bank and target model differ")
    center = np.load(args.center)
    rows = [
        normalize_target_record(
            row,
            task=args.task,
            default_domain=args.default_domain,
            default_prompt=REPORT_PROMPT,
            require_answer=True,
        )
        for row in load_json_or_jsonl(args.manifest)
    ]
    if args.max_samples:
        rows = rows[: args.max_samples]

    model_path = args.model_path
    if model_path is None:
        from corrected_sgta.models import HULU_PATH, LLAVA_PATH

        model_path = HULU_PATH if args.model == "hulu" else LLAVA_PATH
    artifact = model_artifact_fingerprint(model_path)
    if artifact["fingerprint"] != bank["model_artifacts"]["fingerprint"]:
        raise RuntimeError("loaded model artifacts do not match source bank")

    bank_records = bank["records"]
    source_embeddings = np.asarray([r["embedding"] for r in bank_records], dtype=np.float64)
    source_trajectories = [np.asarray(r["trajectory"], dtype=np.float64) for r in bank_records]
    directions = deterministic_directions(
        dimension=len(FEATURE_NAMES),
        count=args.projections,
        seed=args.projection_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.task}_raw.jsonl"
    summary_path = args.output_dir / f"{args.task}_summary.json"
    if raw_path.exists():
        repair_truncated_jsonl_tail(raw_path)
    completed: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            record = json.loads(line)
            completed[record["id"]] = record
    if completed and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume")

    fingerprint_payload = {
        "version": RUN_VERSION,
        "task": args.task,
        "manifest_sha256": file_sha256(args.manifest),
        "bank_sha256": file_sha256(args.bank),
        "bank_fingerprint": bank["fingerprint"],
        "center_sha256": file_sha256(args.center),
        "model": args.model,
        "model_artifact_fingerprint": artifact["fingerprint"],
        "views": {
            "original": True,
            "source_spectrum_alpha": args.spectrum_alpha,
            "low_frequency_ratio": args.low_frequency_ratio,
            "gamma": args.gamma,
        },
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "lambda_value": args.lambda_value,
        "target_domain_used_for_selection": False,
        "target_labels_used_for_generation_or_selection": False,
        "uses_canonical_label_logits_for_prediction": False,
        "null_rgb": NULL_RGB,
    }
    fingerprint = stable_json_sha256(fingerprint_payload)

    adapter = load_anchor_adapter(args.model, model_path)
    try:
        with raw_path.open("a") as handle:
            for row in tqdm(rows, desc=f"ANCHOR-Flow-SGTA {args.task}/{args.model}"):
                if row["id"] in completed:
                    continue
                image_path = resolve_image_path(row["image"], args.image_root)
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                views = make_views(
                    image,
                    center=center,
                    max_side=args.max_image_side,
                    spectrum_alpha=args.spectrum_alpha,
                    low_frequency_ratio=args.low_frequency_ratio,
                    gamma=args.gamma,
                )
                item_seed = per_item_seed(args.seed, row["id"])
                candidates = []
                query_embedding = adapter.input_embedding(views[0]["image"], row["prompt"])
                neighbor_indices, similarities = nearest_source_indices(
                    query_embedding, source_embeddings, args.neighbors
                )
                retrieved = [source_trajectories[int(i)] for i in neighbor_indices]
                for view_index, view in enumerate(views):
                    generated = adapter.generate_candidates(
                        view["image"],
                        row["prompt"],
                        candidate_budget=2,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        seed=item_seed + view_index,
                        candidate_batch=args.candidate_batch,
                    )[0]
                    evidence = adapter.sequence_evidence(
                        view["image"],
                        row["prompt"],
                        generated["text"],
                        bank["max_sequence_tokens"],
                    )
                    normalized = normalize_trajectory(
                        evidence.trajectory, bank["feature_statistics"]
                    )
                    distance, neighbor_local_index, individual = nearest_manifold_distance(
                        normalized,
                        retrieved,
                        directions=directions,
                        quantiles=args.quantiles,
                    )
                    eval_result = evaluate_text(generated["text"], row, args.task)
                    candidate = {
                        **generated,
                        **evidence.to_json(),
                        "candidate_id": f"view-{view_index}",
                        "view_index": view_index,
                        "view_name": view["name"],
                        "view_metadata": view["metadata"],
                        "normalized_trajectory": normalized.astype(float).tolist(),
                        "source_path_distance": float(distance),
                        "source_neighbor_local_index": int(neighbor_local_index),
                        "source_neighbor_distances": [float(x) for x in individual],
                        "sequence_nll": float(-evidence.mean_image_log_probability),
                        "anchor_flow_energy": float(
                            -evidence.mean_image_log_probability
                            + args.lambda_value * distance
                        ),
                        "evaluation": eval_result,
                    }
                    candidates.append(candidate)

                nll_selected = int(np.argmin([c["sequence_nll"] for c in candidates]))
                flow_selected = int(np.argmin([c["anchor_flow_energy"] for c in candidates]))
                random_selected = per_item_seed(args.seed ^ 0xBAD5EED, row["id"]) % len(candidates)
                oracle_selected = int(np.argmax([c["evaluation"]["score"] for c in candidates]))
                original_score = float(candidates[0]["evaluation"]["score"])
                flow_score = float(candidates[flow_selected]["evaluation"]["score"])
                record = {
                    "version": RUN_VERSION,
                    "fingerprint": fingerprint,
                    "status": "ok",
                    "task": args.task,
                    "id": row["id"],
                    "patient_id": row["patient_id"],
                    "image": row["image"],
                    "prompt": row["prompt"],
                    "reference": row["answer"],
                    "evaluation_group": row["domain"],
                    "item_seed": item_seed,
                    "neighbors": [
                        {
                            "domain": bank_records[int(index)]["domain"],
                            "id": bank_records[int(index)]["id"],
                            "similarity": float(similarity),
                        }
                        for index, similarity in zip(neighbor_indices, similarities)
                    ],
                    "candidates": candidates,
                    "unique_text_count": len({c["text"] for c in candidates}),
                    "view_disagreement": len({c["text"] for c in candidates}) > 1,
                    "greedy_index": 0,
                    "greedy_text": candidates[0]["text"],
                    "greedy_score": original_score,
                    "nll_selected_index": nll_selected,
                    "nll_text": candidates[nll_selected]["text"],
                    "nll_score": float(candidates[nll_selected]["evaluation"]["score"]),
                    "random_selected_index": random_selected,
                    "random_text": candidates[random_selected]["text"],
                    "random_score": float(candidates[random_selected]["evaluation"]["score"]),
                    "anchor_flow_selected_index": flow_selected,
                    "anchor_flow_text": candidates[flow_selected]["text"],
                    "anchor_flow_score": flow_score,
                    "oracle_selected_index": oracle_selected,
                    "oracle_text": candidates[oracle_selected]["text"],
                    "oracle_score": float(candidates[oracle_selected]["evaluation"]["score"]),
                    "anchor_flow_outcome": (
                        "rescue"
                        if flow_score > original_score
                        else "harm" if flow_score < original_score else "unchanged"
                    ),
                    "target_domain_used_for_selection": False,
                    "target_labels_used_for_generation_or_selection": False,
                    "uses_canonical_label_logits_for_prediction": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[row["id"]] = record
    finally:
        adapter.close()

    ordered = [completed[row["id"]] for row in rows if row["id"] in completed]
    payload = {
        "version": RUN_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "status": "final",
        "raw": str(raw_path),
        "records": len(ordered),
        "summary": summarize_records(ordered, args.task),
    }
    atomic_json(summary_path, payload)
    print(json.dumps({"summary": str(summary_path), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
