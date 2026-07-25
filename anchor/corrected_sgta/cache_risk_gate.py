#!/usr/bin/env python3
"""Cache-only risk gates for feature-space SGTA candidates.

This analyzer reuses formal CE caches and existing selector reports. It does
not run VLM inference. For each slice, it reconstructs the baseline and selected
candidate predictions, extracts simple image/style/feature risk scores, fits a
single-threshold route-on-calibration gate, and evaluates the gate on the held
out test split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from corrected_sgta.analyze_feature_sgta import (
    build_feature_matrix,
    initial_probabilities,
    load_inputs,
    normalize_rows,
    softmax,
)
from corrected_sgta.protocol_v2 import deterministic_split
from corrected_sgta.scat_methods import fit_logit_scale, tim_probabilities


METHOD_RE = re.compile(r"^fsgta_a(?P<alpha>[0-9.]+)_t(?P<temperature>[0-9.]+)_(?P<kind>initial|tim|tim_kl)$")

SLICE_FILES = {
    "llava_cxr_n128": {
        "calibration_all": "llava_cxr_visualdep_n128.calibration_all.json",
        "alpha0_timkl": "llava_cxr_visualdep_n128.alpha0_timkl.json",
        "alpha_positive": "llava_cxr_visualdep_n128.alpha_positive.json",
    },
    "llava_cxr_n512": {
        "calibration_all": "llava_cxr_visualdep_n512.calibration-all.json",
        "alpha0_timkl": "llava_cxr_visualdep_n512.alpha0-tim-kl.json",
        "alpha_positive": "llava_cxr_visualdep_n512.alpha-positive-tim-kl.json",
    },
    "hulu_cxr_n512": {
        "calibration_all": "hulu_cxr_visualdep_n512.calibration-all.json",
        "alpha0_timkl": "hulu_cxr_visualdep_n512.alpha0-tim-kl.json",
        "alpha_positive": "hulu_cxr_visualdep_n512.alpha-positive-tim-kl.json",
    },
    "llava_mm_n512": {
        "calibration_all": "llava_mm_visualdep_n512.calibration-all.json",
        "alpha0_timkl": "llava_mm_visualdep_n512.alpha0-tim-kl.json",
        "alpha_positive": "llava_mm_visualdep_n512.alpha-positive-tim-kl.json",
    },
    "hulu_mm_n512": {
        "calibration_all": "hulu_mm_visualdep_n512.calibration-all.json",
        "alpha0_timkl": "hulu_mm_visualdep_n512.alpha0-tim-kl.json",
        "alpha_positive": "hulu_mm_visualdep_n512.alpha-positive-tim-kl.json",
    },
    "llava_knowledge_n512": {
        "calibration_all": "llava_knowledge_visualdep_n512.calibration-all.json",
        "alpha0_timkl": "llava_knowledge_visualdep_n512.alpha0-tim-kl.json",
        "alpha_positive": "llava_knowledge_visualdep_n512.alpha-positive-tim-kl.json",
    },
    "hulu_knowledge_n512": {
        "calibration_all": "hulu_knowledge_visualdep_n512.calibration-all.json",
        "alpha0_timkl": "hulu_knowledge_visualdep_n512.alpha0-tim-kl.json",
        "alpha_positive": "hulu_knowledge_visualdep_n512.alpha-positive-tim-kl.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--features-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-legacy-diagnostic", action="store_true")
    return parser.parse_args()


def entropy2(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(2.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    au = normalize_rows(np.asarray(a, dtype=np.float32).reshape(1, -1))[0]
    bu = normalize_rows(np.asarray(b, dtype=np.float32).reshape(1, -1))[0]
    return float(1.0 - np.dot(au, bu))


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def max_or_none(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def method_spec(method: str) -> tuple[float, float, str] | None:
    if method == "baseline_surface_logits":
        return None
    match = METHOD_RE.match(method)
    if not match:
        raise ValueError(f"unsupported method name: {method}")
    return float(match.group("alpha")), float(match.group("temperature")), match.group("kind")


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def reconstruct_context(report: dict, args: argparse.Namespace) -> dict:
    cache = Path(report["source_cache"])
    prototypes = Path(report["source_prototypes"])
    ns = SimpleNamespace(
        cache=cache,
        prototypes=prototypes,
        include_gamma=False,
        allow_legacy_diagnostic=args.allow_legacy_diagnostic,
        device=args.device,
    )
    records, cache_meta, proto_meta, prototypes_array = load_inputs(ns)
    qids = [row["qid"] for row in records]
    calibration_qids, test_qids = deterministic_split(qids, args.calibration_fraction, args.seed)
    by_qid = {row["qid"]: row for row in records}
    ordered_qids = calibration_qids + test_qids
    ordered_records = [by_qid[qid] for qid in ordered_qids]
    base_features = np.stack([row["features"][0] for row in ordered_records])
    base_logits = np.stack([row["base_logits"] for row in ordered_records])
    scale = fit_logit_scale(base_features, prototypes_array, base_logits)
    counts = np.bincount([by_qid[qid]["gt_index"] for qid in calibration_qids], minlength=2)
    return {
        "records": ordered_records,
        "by_qid": by_qid,
        "calibration_qids": calibration_qids,
        "test_qids": test_qids,
        "ordered_qids": ordered_qids,
        "prototypes": prototypes_array,
        "scale": scale,
        "counts": counts,
        "cache_meta": cache_meta,
        "proto_meta": proto_meta,
    }


def probabilities_for_method(method: str, context: dict, args: argparse.Namespace) -> dict[str, np.ndarray]:
    records = context["records"]
    qids = context["ordered_qids"]
    if method == "baseline_surface_logits":
        return {row["qid"]: softmax(row["base_logits"]) for row in records}
    spec = method_spec(method)
    if spec is None:
        raise ValueError(method)
    alpha, temperature, kind = spec
    features = build_feature_matrix(records, alpha, temperature)
    if kind == "initial":
        probs = initial_probabilities(features, context["prototypes"], context["scale"])
    else:
        probs = tim_probabilities(
            features,
            context["prototypes"],
            context["scale"],
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            observed_marginal=context["counts"] if kind == "tim_kl" else None,
            entropy_weight=1.0,
            device=args.device,
        )
    return {qid: probs[index] for index, qid in enumerate(qids)}


def predictions(probabilities: dict[str, np.ndarray]) -> dict[str, int]:
    return {qid: int(np.argmax(prob)) for qid, prob in probabilities.items()}


def accuracy(rows: list[dict], use_candidate: bool = True) -> float:
    if not rows:
        return float("nan")
    key = "candidate_correct" if use_candidate else "baseline_correct"
    return float(np.mean([row[key] for row in rows]))


def gated_accuracy(rows: list[dict], key: str, threshold: float, direction: str) -> tuple[float, int, int, int]:
    routed = []
    correct = []
    rescues = 0
    harmful = 0
    for row in rows:
        score = row["risk_scores"].get(key)
        use_candidate = False if score is None else (score >= threshold if direction == ">=" else score <= threshold)
        if use_candidate:
            routed.append(row)
            rescues += int(row["rescue"])
            harmful += int(row["harmful"])
        correct.append(row["candidate_correct"] if use_candidate else row["baseline_correct"])
    return float(np.mean(correct)) if correct else float("nan"), len(routed), rescues, harmful


def fit_threshold(rows: list[dict], key: str) -> dict | None:
    values = sorted({float(row["risk_scores"][key]) for row in rows if row["risk_scores"].get(key) is not None})
    if not values:
        return None
    thresholds = [values[0] - 1e-9] + [(a + b) / 2.0 for a, b in zip(values, values[1:])] + [values[-1] + 1e-9]
    best = None
    for direction in (">=", "<="):
        for threshold in thresholds:
            acc, routed, rescues, harmful = gated_accuracy(rows, key, threshold, direction)
            item = (acc, -routed, rescues - harmful, direction, float(threshold), routed, rescues, harmful)
            if best is None or item > best:
                best = item
    if best is None:
        return None
    return {
        "score_key": key,
        "direction": best[3],
        "threshold": best[4],
        "train_accuracy": best[0],
        "train_routed": best[5],
        "train_rescues": best[6],
        "train_harmful": best[7],
    }


def evaluate_gate(rows: list[dict], gate: dict) -> dict:
    acc, routed, rescues, harmful = gated_accuracy(rows, gate["score_key"], gate["threshold"], gate["direction"])
    return {
        "accuracy": acc,
        "routed": routed,
        "coverage": float(routed / len(rows)) if rows else None,
        "rescues": rescues,
        "harmful": harmful,
    }


def select_gate(train_rows: list[dict], test_rows: list[dict]) -> dict | None:
    keys = sorted(train_rows[0]["risk_scores"]) if train_rows else []
    candidates = []
    for key in keys:
        gate = fit_threshold(train_rows, key)
        if gate is None:
            continue
        train_eval = evaluate_gate(train_rows, gate)
        test_eval = evaluate_gate(test_rows, gate)
        gate["train"] = train_eval
        gate["test"] = test_eval
        candidates.append(gate)
    if not candidates:
        return None
    baseline_train = accuracy(train_rows, use_candidate=False)
    # Conservative selection: maximize calibration accuracy, then route fewer
    # samples, then prefer larger rescue-minus-harmful balance.
    candidates.append(
        {
            "score_key": "__never_route__",
            "direction": ">=",
            "threshold": float("inf"),
            "train": {
                "accuracy": baseline_train,
                "routed": 0,
                "coverage": 0.0,
                "rescues": 0,
                "harmful": 0,
            },
            "test": {
                "accuracy": accuracy(test_rows, use_candidate=False),
                "routed": 0,
                "coverage": 0.0,
                "rescues": 0,
                "harmful": 0,
            },
        }
    )
    return max(
        candidates,
        key=lambda gate: (
            gate["train"]["accuracy"],
            -gate["train"]["routed"],
            gate["train"]["rescues"] - gate["train"]["harmful"],
            gate["score_key"],
        ),
    )


def row_risk_scores(row: dict, prototypes: np.ndarray, scale: float) -> dict[str, float | None]:
    logits = np.asarray(row["style_logits"], dtype=np.float32)
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1)
    base_prob = probs[0]
    logit_shift = np.linalg.norm(logits[1:] - logits[0], axis=1) if len(logits) > 1 else np.asarray([])
    features = normalize_rows(row["features"])
    feature_shift = np.asarray([cosine_distance(features[0], features[i]) for i in range(1, len(features))])
    proto_prob = initial_probabilities(features[0:1], prototypes, scale)[0]
    center_distances = []
    psnr_values = []
    edge_values = []
    feddg_center = None
    feddg_feature_shift = None
    feddg_logit_shift = None
    feddg_flip = None
    for index, metadata in enumerate(row.get("style_metadata") or []):
        if index == 0:
            continue
        center = (metadata.get("center_distance") or {}).get("log_amplitude_cosine_distance")
        if center is not None:
            center_distances.append(float(center))
        structure = metadata.get("structure") or {}
        if structure.get("psnr") is not None:
            psnr_values.append(float(structure["psnr"]))
        if structure.get("edge_correlation") is not None:
            edge_values.append(float(structure["edge_correlation"]))
        if str(row.get("style_names", [""])[index]).startswith("feddg_"):
            feddg_center = None if center is None else float(center)
            feddg_feature_shift = cosine_distance(features[0], features[index])
            feddg_logit_shift = float(np.linalg.norm(logits[index] - logits[0]))
            feddg_flip = float(pred[index] != pred[0])
    scores = {
        "base_confidence": float(np.max(base_prob)),
        "base_entropy": entropy2(base_prob),
        "base_margin": float(abs(base_prob[0] - base_prob[1])),
        "proto_confidence": float(np.max(proto_prob)),
        "proto_entropy": entropy2(proto_prob),
        "proto_margin": float(abs(proto_prob[0] - proto_prob[1])),
        "base_proto_disagree": float(int(np.argmax(base_prob) != np.argmax(proto_prob))),
        "style_pred_disagreement": float(np.mean(pred[1:] != pred[0])) if len(pred) > 1 else 0.0,
        "style_logit_shift_mean": mean_or_none(logit_shift.tolist()),
        "style_logit_shift_max": max_or_none(logit_shift.tolist()),
        "style_feature_shift_mean": mean_or_none(feature_shift.tolist()),
        "style_feature_shift_max": max_or_none(feature_shift.tolist()),
        "center_distance_mean": mean_or_none(center_distances),
        "center_distance_max": max_or_none(center_distances),
        "feddg_center_distance": feddg_center,
        "feddg_feature_shift": feddg_feature_shift,
        "feddg_logit_shift": feddg_logit_shift,
        "feddg_prediction_flip": feddg_flip,
        "psnr_min": float(np.min(psnr_values)) if psnr_values else None,
        "edge_correlation_min": float(np.min(edge_values)) if edge_values else None,
    }
    seq = row.get("style_sequence_nll")
    if seq:
        seq_arr = np.asarray(seq, dtype=np.float32)
        scores["sequence_margin"] = float(abs(seq_arr[0, 0] - seq_arr[0, 1]))
        if len(seq_arr) > 1:
            scores["sequence_shift_mean"] = float(np.mean(np.linalg.norm(seq_arr[1:] - seq_arr[0], axis=1)))
    return scores


def build_candidate_rows(
    slice_name: str,
    selector_name: str,
    method: str,
    context: dict,
    args: argparse.Namespace,
) -> list[dict]:
    base_prob = probabilities_for_method("baseline_surface_logits", context, args)
    cand_prob = probabilities_for_method(method, context, args)
    base_pred = predictions(base_prob)
    cand_pred = predictions(cand_prob)
    rows = []
    split_map = {qid: "calibration" for qid in context["calibration_qids"]}
    split_map.update({qid: "test" for qid in context["test_qids"]})
    for row in context["records"]:
        qid = row["qid"]
        gt = int(row["gt_index"])
        baseline_correct = base_pred[qid] == gt
        candidate_correct = cand_pred[qid] == gt
        rows.append(
            {
                "slice": slice_name,
                "selector": selector_name,
                "candidate_method": method,
                "qid": qid,
                "split": split_map[qid],
                "gt_index": gt,
                "baseline_prediction": base_pred[qid],
                "candidate_prediction": cand_pred[qid],
                "baseline_correct": bool(baseline_correct),
                "candidate_correct": bool(candidate_correct),
                "rescue": bool((not baseline_correct) and candidate_correct),
                "harmful": bool(baseline_correct and (not candidate_correct)),
                "risk_scores": row_risk_scores(row, context["prototypes"], context["scale"]),
            }
        )
    return rows


def summarize_candidate(rows: list[dict], gate: dict | None) -> dict:
    train = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    out = {
        "n_calibration": len(train),
        "n_test": len(test),
        "baseline_calibration_accuracy": accuracy(train, use_candidate=False),
        "candidate_calibration_accuracy": accuracy(train, use_candidate=True),
        "baseline_test_accuracy": accuracy(test, use_candidate=False),
        "candidate_test_accuracy": accuracy(test, use_candidate=True),
        "candidate_test_delta": accuracy(test, use_candidate=True) - accuracy(test, use_candidate=False),
        "calibration_rescues": sum(row["rescue"] for row in train),
        "calibration_harmful": sum(row["harmful"] for row in train),
        "test_rescues": sum(row["rescue"] for row in test),
        "test_harmful": sum(row["harmful"] for row in test),
        "selected_gate": gate,
    }
    if gate is not None:
        out["gated_test_delta"] = gate["test"]["accuracy"] - out["baseline_test_accuracy"]
    return out


def main() -> None:
    args = parse_args()
    all_feature_rows = []
    eval_payload = {
        "version": "cache-risk-gate-v1",
        "run_dir": str(args.run_dir),
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "device": args.device,
        "slices": {},
        "aggregate": {},
    }

    for slice_name, files in SLICE_FILES.items():
        reports = {selector: load_report(args.run_dir / filename) for selector, filename in files.items()}
        context = reconstruct_context(reports["calibration_all"], args)
        methods = {selector: report["selected_by_calibration"] for selector, report in reports.items()}
        eval_payload["slices"][slice_name] = {"candidate_methods": methods, "candidates": {}}
        for selector, method in methods.items():
            rows = build_candidate_rows(slice_name, selector, method, context, args)
            train = [row for row in rows if row["split"] == "calibration"]
            test = [row for row in rows if row["split"] == "test"]
            gate = select_gate(train, test)
            eval_payload["slices"][slice_name]["candidates"][selector] = summarize_candidate(rows, gate)
            all_feature_rows.extend(rows)

    for selector in ("calibration_all", "alpha0_timkl", "alpha_positive"):
        items = [payload["candidates"][selector] for payload in eval_payload["slices"].values()]
        eval_payload["aggregate"][selector] = {
            "mean_candidate_test_delta": float(np.mean([item["candidate_test_delta"] for item in items])),
            "mean_gated_test_delta": float(np.mean([item.get("gated_test_delta", item["candidate_test_delta"]) for item in items])),
            "min_candidate_test_delta": float(np.min([item["candidate_test_delta"] for item in items])),
            "min_gated_test_delta": float(np.min([item.get("gated_test_delta", item["candidate_test_delta"]) for item in items])),
            "num_candidate_negative": int(sum(item["candidate_test_delta"] < 0 for item in items)),
            "num_gated_negative": int(sum(item.get("gated_test_delta", item["candidate_test_delta"]) < 0 for item in items)),
        }

    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    with args.features_output.open("w") as handle:
        for row in all_feature_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(eval_payload["aggregate"], indent=2))
    print(f"wrote {args.features_output}")
    print(f"wrote {args.eval_output}")


if __name__ == "__main__":
    main()
