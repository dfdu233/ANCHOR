#!/usr/bin/env python3
"""Feature-space SGTA + SCA-T analysis for fixed Yes/No CE rows.

This is a fast, cache-only diagnostic/formal analyzer. It tests whether the
matched-center style views help after moving from pixel logits to visual-feature
semantic prototypes, without regenerating VLM outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from corrected_sgta.analyze_ce import conformal_report, point_summary
from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.protocol_v2 import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION, deterministic_split


SUPPORTED_PROTOCOL_VERSIONS = {PROTOCOL_VERSION, "medheval-sgta-v5.2"}
from corrected_sgta.scat_methods import fit_logit_scale, tim_probabilities

METHOD_VERSION = "feature-space-sgta-scat-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prototypes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, nargs="*", default=(0.1, 0.05))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--feature-alpha-grid", type=float, nargs="*", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--temperature-grid", type=float, nargs="*", default=(0.05, 0.1, 0.2, 0.5))
    parser.add_argument("--include-gamma", action="store_true")
    parser.add_argument(
        "--selector",
        choices=("calibration-all", "tim-kl-only", "alpha0-tim-kl", "alpha-positive-tim-kl"),
        default="tim-kl-only",
        help="Claim-safe method family used for calibration selection.",
    )
    parser.add_argument("--allow-legacy-diagnostic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def style_indices(row: dict, include_gamma: bool) -> list[int]:
    out = [0]
    for index, name in enumerate(row.get("style_names", [])[1:], start=1):
        value = str(name)
        if value.startswith("feddg_") or (include_gamma and value.startswith("gamma_")):
            out.append(index)
    return out


def structure_ok(metadata: dict) -> bool:
    structure = metadata.get("structure") or {}
    psnr = structure.get("psnr")
    edge = structure.get("edge_correlation")
    return psnr is None or edge is None or (float(psnr) >= 15.0 and float(edge) >= 0.85)


def style_weight(row: dict, index: int, temperature: float) -> float:
    if index == 0:
        return 1.0
    metadata_list = row.get("style_metadata") or []
    metadata = metadata_list[index] if index < len(metadata_list) else {}
    if not structure_ok(metadata):
        return 0.0
    distance = (metadata.get("center_distance") or {}).get("log_amplitude_cosine_distance")
    distance = 0.0 if distance is None else float(distance)
    return math.exp(-distance / max(float(temperature), 1e-6))


def build_feature_matrix(records: list[dict], feature_alpha: float, temperature: float) -> np.ndarray:
    output = []
    for row in records:
        features = normalize_rows(row["features"])
        indices = row["style_indices"]
        weights = np.asarray([style_weight(row, index, temperature) for index in indices], dtype=np.float32)
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(len(indices), dtype=np.float32)
        weights = weights / weights.sum()
        style_mean = np.sum(features[indices] * weights[:, None], axis=0)
        mixed = (1.0 - feature_alpha) * features[0] + feature_alpha * style_mean
        output.append(mixed)
    return normalize_rows(np.stack(output))


def initial_probabilities(features: np.ndarray, prototypes: np.ndarray, scale: float) -> np.ndarray:
    return softmax(float(scale) * (normalize_rows(features) @ normalize_rows(prototypes).T))


def predictions_from_prob(probabilities: dict[str, np.ndarray], qids: list[str]) -> dict[str, int]:
    return {qid: int(np.argmax(probabilities[qid])) for qid in qids}


def accuracy_on(predictions: dict[str, int], by_qid: dict[str, dict], qids: list[str]) -> float | None:
    if not qids:
        return None
    return float(np.mean([predictions[qid] == int(by_qid[qid]["gt_index"]) for qid in qids]))


def parse_feature_alpha(method: str) -> float | None:
    if not method.startswith("fsgta_a"):
        return None
    rest = method.removeprefix("fsgta_a")
    alpha_token = rest.split("_t", 1)[0]
    try:
        return float(alpha_token)
    except ValueError:
        return None


def method_tie_break(method: str) -> tuple[int, float, int, str]:
    """Prefer simpler methods when calibration accuracy is tied.

    This is intentionally conservative for claim safety: a feature-space SGTA
    variant must beat the alpha=0/SCA-T or surface baseline on calibration, not
    merely win by lexicographic ordering.
    """

    if method == "baseline_surface_logits":
        return (0, 0.0, 0, method)
    alpha = parse_feature_alpha(method)
    if alpha is None:
        return (2, 999.0, 99, method)
    if method.endswith("_tim_kl"):
        adaptation_rank = 0
    elif method.endswith("_initial"):
        adaptation_rank = 1
    elif method.endswith("_tim"):
        adaptation_rank = 2
    else:
        adaptation_rank = 9
    return (1, float(alpha), adaptation_rank, method)


def selector_candidates(methods: list[str], selector: str) -> list[str]:
    if selector == "calibration-all":
        return list(methods)
    if selector == "tim-kl-only":
        return [method for method in methods if method == "baseline_surface_logits" or method.endswith("_tim_kl")]
    if selector == "alpha0-tim-kl":
        return [
            method
            for method in methods
            if method == "baseline_surface_logits" or (method.startswith("fsgta_a0_t") and method.endswith("_tim_kl"))
        ]
    if selector == "alpha-positive-tim-kl":
        return [
            method
            for method in methods
            if method.startswith("fsgta_a")
            and not method.startswith("fsgta_a0_")
            and not method.startswith("fsgta_a0_t")
            and method.endswith("_tim_kl")
        ]
    raise ValueError(f"unknown selector {selector}")


def load_inputs(args: argparse.Namespace) -> tuple[list[dict], dict, dict, np.ndarray]:
    cache_meta = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    proto_meta = json.loads(args.prototypes.with_suffix(args.prototypes.suffix + ".meta.json").read_text())
    cache_protocol = cache_meta.get("protocol_version")
    proto_protocol = proto_meta.get("protocol_version")
    if cache_protocol not in SUPPORTED_PROTOCOL_VERSIONS or proto_protocol not in SUPPORTED_PROTOCOL_VERSIONS:
        raise RuntimeError("unsupported protocol")
    if (cache_protocol != PROTOCOL_VERSION or proto_protocol != PROTOCOL_VERSION) and not args.allow_legacy_diagnostic:
        raise RuntimeError("legacy protocol requires --allow-legacy-diagnostic")
    if proto_meta.get("model") != cache_meta.get("config", {}).get("model"):
        raise RuntimeError("prototype/cache model mismatch")
    legacy = cache_meta.get("cache_schema_version") != CACHE_SCHEMA_VERSION
    if legacy and not args.allow_legacy_diagnostic:
        raise RuntimeError("feature SGTA requires v5.4 cache unless --allow-legacy-diagnostic is set")
    fingerprint = cache_meta["fingerprint"]
    rows = []
    for row in iter_successes(args.cache, fingerprint):
        if row.get("question_type") != "binary" or row.get("labels") != ["Yes", "No"]:
            continue
        row = dict(row)
        row["qid"] = str(row["qid"])
        row["features"] = decode_array(row["style_features"]).astype(np.float32)
        row["base_logits"] = np.asarray(row["style_logits"][0], dtype=np.float32)
        row["style_indices"] = style_indices(row, args.include_gamma)
        if len(row["style_indices"]) < 1:
            row["style_indices"] = [0]
        rows.append(row)
    if len(rows) < 4:
        raise RuntimeError("feature SGTA requires at least four fixed Yes/No rows")
    prototypes = np.load(args.prototypes, allow_pickle=False)["prototypes"].astype(np.float32)
    if prototypes.shape != (2, rows[0]["features"].shape[1]):
        raise RuntimeError(f"prototype shape {prototypes.shape} incompatible with features {rows[0]['features'].shape}")
    cache_meta["evidence_status"] = "legacy_diagnostic_only" if legacy else "formal_v5.4"
    return rows, cache_meta, proto_meta, prototypes


def main() -> None:
    args = parse_args()
    records, cache_meta, proto_meta, prototypes = load_inputs(args)
    qids = [row["qid"] for row in records]
    calibration_qids, test_qids = deterministic_split(qids, args.calibration_fraction, args.seed)
    by_qid = {row["qid"]: row for row in records}
    ordered_qids = calibration_qids + test_qids
    ordered_records = [by_qid[qid] for qid in ordered_qids]
    base_features = np.stack([row["features"][0] for row in ordered_records])
    base_logits = np.stack([row["base_logits"] for row in ordered_records])
    scale = fit_logit_scale(base_features, prototypes, base_logits)

    method_probs: dict[str, dict[str, np.ndarray]] = {
        "baseline_surface_logits": {
            row["qid"]: softmax(row["base_logits"]) for row in ordered_records
        }
    }
    diagnostics = {"scale": scale, "feature_grid": []}
    counts = np.bincount([by_qid[qid]["gt_index"] for qid in calibration_qids], minlength=2)

    for feature_alpha in args.feature_alpha_grid:
        for temperature in args.temperature_grid:
            features = build_feature_matrix(ordered_records, feature_alpha, temperature)
            name = f"fsgta_a{feature_alpha:g}_t{temperature:g}"
            init = initial_probabilities(features, prototypes, scale)
            method_probs[f"{name}_initial"] = {qid: init[i] for i, qid in enumerate(ordered_qids)}
            tim = tim_probabilities(
                features,
                prototypes,
                scale,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                observed_marginal=None,
                entropy_weight=1.0,
                device=args.device,
            )
            tim_kl = tim_probabilities(
                features,
                prototypes,
                scale,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                observed_marginal=counts,
                entropy_weight=1.0,
                device=args.device,
            )
            method_probs[f"{name}_tim"] = {qid: tim[i] for i, qid in enumerate(ordered_qids)}
            method_probs[f"{name}_tim_kl"] = {qid: tim_kl[i] for i, qid in enumerate(ordered_qids)}
            diagnostics["feature_grid"].append({"feature_alpha": feature_alpha, "temperature": temperature})

    predictions = {method: predictions_from_prob(values, ordered_qids) for method, values in method_probs.items()}
    cal_acc = {method: accuracy_on(values, by_qid, calibration_qids) for method, values in predictions.items()}
    candidates = selector_candidates(list(cal_acc), args.selector)
    if not candidates:
        raise RuntimeError(f"selector {args.selector} produced no candidate methods")
    selected_method = sorted(
        candidates,
        key=lambda method: (
            -(cal_acc[method] if cal_acc[method] is not None else -1.0),
            method_tie_break(method),
        ),
    )[0]
    oracle_rows = []
    for qid in test_qids:
        correct = [predictions[method][qid] == int(by_qid[qid]["gt_index"]) for method in predictions]
        oracle_rows.append(any(correct))
    point = {
        method: point_summary({qid: values[qid] for qid in test_qids}, records)
        for method, values in predictions.items()
    }
    pool_point = {method: point_summary(values, records) for method, values in predictions.items()}
    conformal = {
        method: conformal_report(method, values, by_qid, calibration_qids, test_qids, list(args.alpha), args.seed)
        for method, values in method_probs.items()
    }
    baseline = point["baseline_surface_logits"]["accuracy"]
    selected = point[selected_method]["accuracy"]
    best_test_method = max(point, key=lambda method: -1 if point[method]["accuracy"] is None else point[method]["accuracy"])
    prediction_audit = [
        {
            "qid": qid,
            "gt_index": int(by_qid[qid]["gt_index"]),
            "baseline_prediction": int(predictions["baseline_surface_logits"][qid]),
            "selected_prediction": int(predictions[selected_method][qid]),
            "selected_correct": bool(predictions[selected_method][qid] == int(by_qid[qid]["gt_index"])),
            "baseline_correct": bool(predictions["baseline_surface_logits"][qid] == int(by_qid[qid]["gt_index"])),
        }
        for qid in test_qids
    ]
    feature_alpha0_methods = [method for method in point if method.startswith("fsgta_a0_") or method.startswith("fsgta_a0_t")]
    feature_positive_methods = [method for method in point if method.startswith("fsgta_a") and not method.startswith("fsgta_a0_") and not method.startswith("fsgta_a0_t")]
    best_alpha0 = max(feature_alpha0_methods, key=lambda method: -1 if point[method]["accuracy"] is None else point[method]["accuracy"]) if feature_alpha0_methods else None
    best_alpha_positive = max(feature_positive_methods, key=lambda method: -1 if point[method]["accuracy"] is None else point[method]["accuracy"]) if feature_positive_methods else None

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION,
        "source_cache": str(args.cache),
        "source_prototypes": str(args.prototypes),
        "fingerprint": cache_meta["fingerprint"],
        "evidence_status": cache_meta["evidence_status"],
        "n_yes_no": len(records),
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "n_calibration": len(calibration_qids),
            "n_test": len(test_qids),
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
            "calibration_class_counts": counts.tolist(),
        },
        "adaptation": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "include_gamma": args.include_gamma,
            "feature_alpha_grid": list(args.feature_alpha_grid),
            "temperature_grid": list(args.temperature_grid),
            "scale": scale,
            "selector": args.selector,
        },
        "calibration_accuracy": cal_acc,
        "selected_by_calibration": selected_method,
        "selected_candidates": candidates,
        "selected_tie_break": "within selector: highest calibration accuracy, then baseline/alpha=0 before alpha>0, then TIM-KL before initial before TIM",
        "best_test_method_diagnostic_only": best_test_method,
        "best_alpha0_scat_method_diagnostic_only": best_alpha0,
        "best_alpha_positive_feature_method_diagnostic_only": best_alpha_positive,
        "feature_increment_over_alpha0_diagnostic_only": (
            None
            if best_alpha0 is None or best_alpha_positive is None
            else point[best_alpha_positive]["accuracy"] - point[best_alpha0]["accuracy"]
        ),
        "point_accuracy": point,
        "point_accuracy_transductive_pool": pool_point,
        "conformal": conformal,
        "gate": {
            "baseline_accuracy": baseline,
            "selected_accuracy": selected,
            "selected_delta": None if baseline is None or selected is None else selected - baseline,
            "best_test_accuracy_diagnostic_only": point[best_test_method]["accuracy"],
            "best_test_delta_diagnostic_only": None if baseline is None else point[best_test_method]["accuracy"] - baseline,
            "test_method_oracle_accuracy_diagnostic_only": float(np.mean(oracle_rows)) if oracle_rows else None,
            "selected_noninferior_margin_0_5pp": selected is not None and baseline is not None and selected >= baseline - 0.005,
            "selected_positive": selected is not None and baseline is not None and selected > baseline,
        },
        "method_scope": {
            "classes": "fixed Yes/No only; MC excluded because option semantics are question-specific",
            "claim_use": "selected_by_calibration is claim-safe; best_test is diagnostic only",
            "feature_space": "normalized last multimodal prompt hidden states, averaged over matched style views with domain-distance weights",
            "relationship_to_scat": "alpha=0 reproduces SCA-T on original visual features; alpha>0 tests SGTA feature-space style alignment",
        },
        "diagnostics": diagnostics,
        "prediction_audit_test": prediction_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gate"], indent=2))
    print(f"selected_by_calibration={selected_method}")
    print(f"best_test_diagnostic={best_test_method}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
