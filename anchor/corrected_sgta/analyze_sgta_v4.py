#!/usr/bin/env python3
"""Leakage-safe SGTA-v4 routing and selective-risk analysis on CE caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.methods import softmax_np
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    deterministic_split,
)
from corrected_sgta.scat_methods import fit_logit_scale

METHOD_VERSION = "matched-center-sgta-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--router-fraction", type=float, default=0.2)
    parser.add_argument("--conformal-fraction", type=float, default=0.2)
    parser.add_argument("--min-calibration-gain", type=float, default=0.01)
    parser.add_argument("--calibration-alpha", type=float, default=0.10)
    parser.add_argument("--noninferiority-margin", type=float, default=0.005)
    parser.add_argument("--fixed-coverage", type=float, default=0.8)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--max-feature-distance", type=float, default=0.25)
    parser.add_argument("--include-gamma", action="store_true")
    parser.add_argument(
        "--allow-legacy-diagnostic",
        action="store_true",
        help="read an old cache only as explicitly labeled diagnostic evidence",
    )
    parser.add_argument(
        "--prototypes",
        type=Path,
        default=None,
        help="optional SCA-T Yes/No semantic prototype npz",
    )
    return parser.parse_args()


def entropy(probability: np.ndarray) -> float:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(value * np.log(value)))


def margin(probability: np.ndarray) -> float:
    ordered = np.sort(np.asarray(probability, dtype=np.float64))
    return float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(np.asarray(left, dtype=np.float64), 1e-12, 1.0)
    right = np.clip(np.asarray(right, dtype=np.float64), 1e-12, 1.0)
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(left / middle))
        + 0.5 * np.sum(right * np.log(right / middle))
    )


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - float(left @ right) / denominator, 0.0, 2.0))


def selected_style_indices(row: dict, include_gamma: bool) -> list[int]:
    selected = [0]
    for index, name in enumerate(row["style_names"][1:], start=1):
        if str(name).startswith("feddg_") or (include_gamma and str(name).startswith("gamma_")):
            selected.append(index)
    return selected


def _metadata_value(row: dict, index: int, key: str, fallback: float) -> float:
    metadata = row.get("style_metadata") or []
    if index >= len(metadata):
        return fallback
    value = (metadata[index].get("structure") or {}).get(key)
    return fallback if value is None or not math.isfinite(float(value)) else float(value)


def _center_metadata_value(row: dict, index: int, key: str, fallback: float = 0.0) -> float:
    metadata = row.get("style_metadata") or []
    if index >= len(metadata):
        return fallback
    value = (metadata[index].get("center_distance") or {}).get(key)
    return fallback if value is None or not math.isfinite(float(value)) else float(value)


def _parameter_value(row: dict, index: int, key: str, fallback: float = 0.0) -> float:
    metadata = row.get("style_metadata") or []
    if index >= len(metadata):
        return fallback
    value = (metadata[index].get("parameters") or {}).get(key)
    return fallback if value is None or not math.isfinite(float(value)) else float(value)


def candidate_features(row: dict, probabilities: np.ndarray, index: int) -> np.ndarray:
    base = probabilities[0]
    candidate = probabilities[index]
    base_logits = row["logits"][0]
    candidate_logits = row["logits"][index]
    return np.asarray(
        [
            entropy(base),
            margin(base),
            float(np.max(base)),
            entropy(candidate),
            margin(candidate),
            float(np.max(candidate)),
            js_divergence(base, candidate),
            cosine_distance(row["features"][0], row["features"][index]),
            float(np.linalg.norm(candidate_logits - base_logits)),
            float(np.max(candidate) - np.max(base)),
            _metadata_value(row, index, "pixel_mse", 0.0),
            _metadata_value(row, index, "edge_correlation", 1.0),
            _center_metadata_value(
                row, index, "log_amplitude_cosine_distance", 0.0
            ),
            _center_metadata_value(
                row, index, "log_amplitude_relative_rmse", 0.0
            ),
            float(index == 0),
            _parameter_value(row, index, "low_frequency_ratio", 0.0),
            _parameter_value(row, index, "source_ratio", 0.0),
        ],
        dtype=np.float64,
    )


def row_risk_features(row: dict, probabilities: np.ndarray, indices: list[int]) -> np.ndarray:
    base = probabilities[0]
    candidate_rows = [candidate_features(row, probabilities, index) for index in indices]
    matrix = np.stack(candidate_rows)
    predictions = np.argmax(probabilities[indices], axis=1)
    return np.asarray(
        [
            entropy(base),
            margin(base),
            float(np.max(base)),
            float(matrix[:, 6].mean()),
            float(matrix[:, 6].max()),
            float(matrix[:, 7].mean()),
            float(matrix[:, 7].max()),
            float(np.mean(predictions != predictions[0])),
            float(matrix[:, 10].max()),
            float(matrix[:, 11].min()),
            float(matrix[:, 12].mean()),
            float(matrix[:, 13].mean()),
            float(matrix[:, 15].mean()),
            float(matrix[:, 16].mean()),
        ],
        dtype=np.float64,
    )


class BinaryProbabilityModel:
    """Logistic model with a deterministic constant fallback for one-class pilots."""

    def __init__(self) -> None:
        self.model = None
        self.constant = 0.5

    def fit(self, features: list[np.ndarray], labels: list[int]) -> None:
        if not features:
            raise ValueError("cannot fit an empty router")
        unique = sorted(set(labels))
        if len(unique) == 1:
            self.constant = float(unique[0])
            return
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=0),
        )
        self.model.fit(np.stack(features), np.asarray(labels, dtype=np.int64))

    def probability(self, features: np.ndarray) -> float:
        if self.model is None:
            return self.constant
        classes = list(self.model[-1].classes_)
        values = self.model.predict_proba(features[None, :])[0]
        return float(values[classes.index(1)]) if 1 in classes else 0.0


def three_way_split(
    qids: list[str], router_fraction: float, conformal_fraction: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    development_fraction = router_fraction + conformal_fraction
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("router + conformal fractions must be in (0, 1)")
    development, test = deterministic_split(qids, development_fraction, seed)
    relative = router_fraction / development_fraction
    router, calibration = deterministic_split(development, relative, seed + 1)
    return router, calibration, test


def load_records(
    cache: Path,
    include_gamma: bool,
    max_feature_distance: float,
    allow_legacy_diagnostic: bool,
) -> tuple[list[dict], dict]:
    metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    legacy = metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION
    if legacy and not allow_legacy_diagnostic:
        raise RuntimeError(
            "SGTA-v4 requires a v5.4 evidence cache; use --allow-legacy-diagnostic "
            "only for explicitly labeled old-cache diagnosis"
        )
    if not legacy and metadata.get("config", {}).get("center_policy") != "matched":
        raise RuntimeError("SGTA-v4 formal analysis requires center_policy=matched")
    fingerprint = metadata["fingerprint"]
    records = list(iter_successes(cache, fingerprint))
    if not legacy:
        for row in records:
            if row.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                raise RuntimeError(f"qid {row.get('qid')} has an incompatible row schema")
            if len(row.get("style_metadata") or []) != len(row.get("style_names") or []):
                raise RuntimeError(f"qid {row.get('qid')} has incomplete style metadata")
    metadata["evidence_status"] = "legacy_diagnostic_only" if legacy else "formal_v5.4"
    for row in records:
        row["qid"] = str(row["qid"])
        row["logits"] = np.asarray(row["style_logits"], dtype=np.float64)
        row["features"] = decode_array(row["style_features"]).astype(np.float64)
        raw_indices = selected_style_indices(row, include_gamma)
        row["style_indices"] = [
            index for index in raw_indices
            if index == 0 or cosine_distance(row["features"][0], row["features"][index]) <= max_feature_distance
        ]
        sequence_nll = row.get("style_sequence_nll")
        row["sequence_nll"] = (
            None if not sequence_nll or any(value is None for value in sequence_nll)
            else np.asarray(sequence_nll, dtype=np.float64)
        )
    return records, metadata


def surface_probabilities(row: dict) -> np.ndarray:
    return softmax_np(row["logits"], axis=-1)


def sequence_nll_probabilities(row: dict) -> np.ndarray:
    if row["sequence_nll"] is None:
        raise ValueError("cache does not contain style_sequence_nll")
    return softmax_np(-row["sequence_nll"], axis=-1)


def semantic_probability_factory(records: list[dict], prototypes_path: Path):
    prototypes = np.load(prototypes_path, allow_pickle=False)["prototypes"].astype(np.float64)
    if prototypes.shape[0] != 2 or any(row["logits"].shape[1] != 2 for row in records):
        raise ValueError("SCA-T compatibility currently requires fixed Yes/No rows")
    original_features = np.stack([row["features"][0] for row in records])
    original_logits = np.stack([row["logits"][0] for row in records])
    scale = fit_logit_scale(original_features, prototypes, original_logits)
    normalized_prototypes = prototypes / np.clip(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12, None
    )

    def probabilities(row: dict) -> np.ndarray:
        features = row["features"]
        normalized = features / np.clip(
            np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None
        )
        return softmax_np(scale * (normalized @ normalized_prototypes.T), axis=-1)

    return probabilities, float(scale)


def fit_routers(records: list[dict], probability_fn):
    error_features: list[np.ndarray] = []
    error_labels: list[int] = []
    candidate_rows: list[np.ndarray] = []
    candidate_labels: list[int] = []
    for row in records:
        probabilities = probability_fn(row)
        indices = row["style_indices"]
        gt = int(row["gt_index"])
        error_features.append(row_risk_features(row, probabilities, indices))
        error_labels.append(int(np.argmax(probabilities[0]) != gt))
        for index in indices:
            candidate_rows.append(candidate_features(row, probabilities, index))
            candidate_labels.append(int(np.argmax(probabilities[index]) == gt))
    error_model = BinaryProbabilityModel()
    candidate_model = BinaryProbabilityModel()
    error_model.fit(error_features, error_labels)
    candidate_model.fit(candidate_rows, candidate_labels)
    return error_model, candidate_model


def choose_index(
    row: dict,
    probabilities: np.ndarray,
    error_model: BinaryProbabilityModel,
    candidate_model: BinaryProbabilityModel,
    error_threshold: float,
    gain_threshold: float,
) -> tuple[int, float, dict]:
    indices = row["style_indices"]
    risk = error_model.probability(row_risk_features(row, probabilities, indices))
    scores = {
        index: candidate_model.probability(candidate_features(row, probabilities, index))
        for index in indices
    }
    best = max(indices, key=lambda index: (scores[index], -index))
    gain = scores[best] - scores[0]
    selected = best if best != 0 and risk >= error_threshold and gain >= gain_threshold else 0
    return selected, risk, {"predicted_gain": float(gain), "candidate_scores": scores}


def point_metrics(rows: list[dict], selected: dict[str, int], probability_fn) -> dict:
    correct, nll, used = [], [], {}
    for row in rows:
        index = selected[row["qid"]]
        probability = probability_fn(row)[index]
        gt = int(row["gt_index"])
        correct.append(int(np.argmax(probability)) == gt)
        nll.append(-math.log(max(float(probability[gt]), 1e-12)))
        style = row["style_names"][index]
        used[style] = used.get(style, 0) + 1
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if correct else None,
        "nll": float(np.mean(nll)) if nll else None,
        "used_styles": used,
    }


def selective_metrics(correct: list[bool], risk: list[float], fixed_coverage: float) -> dict:
    if not correct:
        return {"aurc": None, "fixed_coverage": fixed_coverage, "risk_at_coverage": None}
    order = np.argsort(np.asarray(risk, dtype=np.float64))
    errors = 1.0 - np.asarray(correct, dtype=np.float64)[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    answered = max(1, int(math.floor(len(errors) * fixed_coverage)))
    return {
        "aurc": float(np.mean(cumulative_risk)),
        "fixed_coverage": float(fixed_coverage),
        "answered": answered,
        "risk_at_coverage": float(cumulative_risk[answered - 1]),
        "abstention_rate": float(1.0 - answered / len(errors)),
    }


def evaluate_policy(
    rows: list[dict], probability_fn, error_model, candidate_model, error_threshold, gain_threshold, fixed_coverage
) -> dict:
    selected: dict[str, int] = {}
    risks: list[float] = []
    baseline_correct: list[bool] = []
    selected_correct: list[bool] = []
    audit = []
    for row in rows:
        probabilities = probability_fn(row)
        index, risk, details = choose_index(
            row, probabilities, error_model, candidate_model, error_threshold, gain_threshold
        )
        gt = int(row["gt_index"])
        base_ok = int(np.argmax(probabilities[0])) == gt
        chosen_ok = int(np.argmax(probabilities[index])) == gt
        selected[row["qid"]] = index
        risks.append(risk)
        baseline_correct.append(base_ok)
        selected_correct.append(chosen_ok)
        audit.append(
            {
                "qid": row["qid"],
                "selected_index": index,
                "selected_style": row["style_names"][index],
                "risk": risk,
                "baseline_correct_evaluation_only": base_ok,
                "selected_correct_evaluation_only": chosen_ok,
                **details,
            }
        )
    baseline = {row["qid"]: 0 for row in rows}
    baseline_point = point_metrics(rows, baseline, probability_fn)
    selected_point = point_metrics(rows, selected, probability_fn)
    oracle = []
    for row in rows:
        probabilities = probability_fn(row)
        gt = int(row["gt_index"])
        oracle.append(any(int(np.argmax(probabilities[index])) == gt for index in row["style_indices"]))
    harmful = sum(base and not chosen for base, chosen in zip(baseline_correct, selected_correct))
    rescues = sum((not base) and chosen for base, chosen in zip(baseline_correct, selected_correct))
    baseline_aurc = selective_metrics(baseline_correct, risks, fixed_coverage)
    selected_aurc = selective_metrics(selected_correct, risks, fixed_coverage)
    oracle_accuracy = float(np.mean(oracle)) if oracle else None
    headroom = oracle_accuracy - baseline_point["accuracy"]
    recovered = selected_point["accuracy"] - baseline_point["accuracy"]
    recovery = recovered / headroom if headroom > 1e-12 else None
    return {
        "baseline": baseline_point,
        "selected": selected_point,
        "delta_accuracy": recovered,
        "style_oracle_accuracy_evaluation_only": oracle_accuracy,
        "oracle_headroom": headroom,
        "oracle_headroom_recovery": recovery,
        "rescues": rescues,
        "harmful_flips": harmful,
        "baseline_selective": baseline_aurc,
        "selected_selective": selected_aurc,
        "audit": audit,
    }


def calibration_search(rows, probability_fn, error_model, candidate_model, args):
    baseline_accuracy = point_metrics(rows, {row["qid"]: 0 for row in rows}, probability_fn)["accuracy"]
    required = max(float(args.min_calibration_gain), 2.0 / max(1, len(rows)))
    search = []
    for error_threshold in (0.35, 0.5, 0.65):
        for gain_threshold in (0.0, 0.025, 0.05, 0.10):
            result = evaluate_policy(
                rows,
                probability_fn,
                error_model,
                candidate_model,
                error_threshold,
                gain_threshold,
                args.fixed_coverage,
            )
            discordant = result["rescues"] + result["harmful_flips"]
            safety_p = (
                float(
                    binomtest(
                        result["rescues"], discordant, 0.5, alternative="greater"
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            base_passed = (
                result["selected"]["accuracy"] >= baseline_accuracy + required
                and result["selected"]["nll"] <= result["baseline"]["nll"]
                and result["rescues"] >= result["harmful_flips"]
            )
            search.append(
                {
                    "error_threshold": error_threshold,
                    "gain_threshold": gain_threshold,
                    "base_passed": base_passed,
                    "passed": False,
                    "accuracy": result["selected"]["accuracy"],
                    "nll": result["selected"]["nll"],
                    "delta_accuracy": result["delta_accuracy"],
                    "rescues": result["rescues"],
                    "harmful_flips": result["harmful_flips"],
                    "one_sided_p": safety_p,
                }
            )
    multiplicity_alpha = args.calibration_alpha / max(1, len(search))
    for row in search:
        row["multiplicity_adjusted_alpha"] = multiplicity_alpha
        row["passed"] = row.pop("base_passed") and row["one_sided_p"] <= multiplicity_alpha
    valid = [row for row in search if row["passed"]]
    if not valid:
        return {
            "mode": "original",
            "error_threshold": 1.1,
            "gain_threshold": 1.1,
            "multiplicity_adjusted_alpha": multiplicity_alpha,
        }, search, required
    best = sorted(valid, key=lambda row: (-row["accuracy"], row["nll"], -row["error_threshold"], -row["gain_threshold"]))[0]
    return {
        "mode": "router",
        "error_threshold": best["error_threshold"],
        "gain_threshold": best["gain_threshold"],
        "multiplicity_adjusted_alpha": multiplicity_alpha,
    }, search, required


def structure_summary(records: list[dict], args) -> dict:
    values = []
    for row in records:
        for index in row["style_indices"]:
            if index == 0:
                continue
            psnr = _metadata_value(row, index, "psnr", float("nan"))
            edge = _metadata_value(row, index, "edge_correlation", float("nan"))
            if math.isfinite(psnr) and math.isfinite(edge):
                values.append((psnr, edge))
    if not values:
        return {"available": False, "pass_rate": None}
    passed = [psnr >= args.min_psnr and edge >= args.min_edge_correlation for psnr, edge in values]
    return {
        "available": True,
        "n_views": len(values),
        "mean_psnr": float(np.mean([value[0] for value in values])),
        "mean_edge_correlation": float(np.mean([value[1] for value in values])),
        "pass_rate": float(np.mean(passed)),
        "thresholds": {"min_psnr": args.min_psnr, "min_edge_correlation": args.min_edge_correlation},
    }


def run_channel(name, router, calibration, test, probability_fn, args, extra=None):
    error_model, candidate_model = fit_routers(router, probability_fn)
    selected, search, required = calibration_search(
        calibration, probability_fn, error_model, candidate_model, args
    )
    result = evaluate_policy(
        test,
        probability_fn,
        error_model,
        candidate_model,
        selected["error_threshold"],
        selected["gain_threshold"],
        args.fixed_coverage,
    )
    base_aurc = result["baseline_selective"]["aurc"]
    selected_aurc = result["selected_selective"]["aurc"]
    aurc_reduction = (
        (base_aurc - selected_aurc) / base_aurc if base_aurc and base_aurc > 0 else None
    )
    gate = {
        "oracle_headroom_at_least_2pp": result["oracle_headroom"] >= 0.02,
        "rescues_not_less_than_harmful": result["rescues"] >= result["harmful_flips"],
        "router_recovers_30pct_headroom": (
            result["oracle_headroom_recovery"] is not None
            and result["oracle_headroom_recovery"] >= 0.30
        ),
        "point_noninferiority": result["delta_accuracy"] >= -args.noninferiority_margin,
        "aurc_relative_reduction_10pct": aurc_reduction is not None and aurc_reduction >= 0.10,
    }
    return {
        "name": name,
        "selected_on_conformal_calibration": selected,
        "required_calibration_accuracy_gain": required,
        "calibration_search": search,
        "test": result,
        "aurc_relative_reduction": aurc_reduction,
        "gate": {**gate, "passed": all(gate.values())},
        **(extra or {}),
    }


def main() -> None:
    args = parse_args()
    records, metadata = load_records(
        args.cache,
        args.include_gamma,
        args.max_feature_distance,
        args.allow_legacy_diagnostic,
    )
    qids = [row["qid"] for row in records]
    router_qids, calibration_qids, test_qids = three_way_split(
        qids, args.router_fraction, args.conformal_fraction, args.seed
    )
    by_qid = {row["qid"]: row for row in records}
    router = [by_qid[qid] for qid in router_qids]
    calibration = [by_qid[qid] for qid in calibration_qids]
    test = [by_qid[qid] for qid in test_qids]
    channels = {
        "surface": run_channel(
            "surface", router, calibration, test, surface_probabilities, args
        )
    }
    if all(row["sequence_nll"] is not None for row in records):
        channels["label_sequence_nll"] = run_channel(
            "label_sequence_nll",
            router,
            calibration,
            test,
            sequence_nll_probabilities,
            args,
        )
    if args.prototypes is not None:
        semantic_records = [
            row for row in records
            if row.get("question_type") == "binary"
            and row.get("labels") == ["Yes", "No"]
            and row["logits"].shape[1] == 2
        ]
        if len(semantic_records) >= 5:
            semantic_router_qids, semantic_calibration_qids, semantic_test_qids = (
                three_way_split(
                    [row["qid"] for row in semantic_records],
                    args.router_fraction,
                    args.conformal_fraction,
                    args.seed,
                )
            )
            semantic_by_qid = {row["qid"]: row for row in semantic_records}
            semantic_router = [semantic_by_qid[qid] for qid in semantic_router_qids]
            semantic_calibration = [
                semantic_by_qid[qid] for qid in semantic_calibration_qids
            ]
            semantic_test = [semantic_by_qid[qid] for qid in semantic_test_qids]
            semantic_fn, scale = semantic_probability_factory(
                semantic_router + semantic_calibration, args.prototypes
            )
            channels["semantic_prototype"] = run_channel(
                "semantic_prototype",
                semantic_router,
                semantic_calibration,
                semantic_test,
                semantic_fn,
                args,
                {
                    "fitted_label_free_logit_scale": scale,
                    "prototypes": str(args.prototypes),
                    "scope": "fixed Yes/No rows only",
                    "split": {
                        "n_router_train": len(semantic_router),
                        "n_conformal_calibration": len(semantic_calibration),
                        "n_locked_test": len(semantic_test),
                        "router_qids": semantic_router_qids,
                        "conformal_calibration_qids": semantic_calibration_qids,
                        "locked_test_qids": semantic_test_qids,
                    },
                },
            )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION,
        "source_cache": str(args.cache),
        "fingerprint": metadata["fingerprint"],
        "cache_schema_version": metadata.get("cache_schema_version"),
        "evidence_status": metadata["evidence_status"],
        "split": {
            "seed": args.seed,
            "n_router_train": len(router),
            "n_conformal_calibration": len(calibration),
            "n_locked_test": len(test),
            "router_qids": router_qids,
            "conformal_calibration_qids": calibration_qids,
            "locked_test_qids": test_qids,
        },
        "filters": {
            "max_feature_cosine_distance": args.max_feature_distance,
            "calibration_safety_alpha": args.calibration_alpha,
        },
        "style_availability": {
            "rows_with_eligible_style": sum(len(row["style_indices"]) > 1 for row in records),
            "rows_original_only": sum(len(row["style_indices"]) == 1 for row in records),
        },
        "structure": structure_summary(records, args),
        "channels": channels,
        "scope": (
            "Router labels come only from router-train. Thresholds are selected only on the "
            "conformal-calibration split. Locked-test labels are evaluation-only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    compact = {
        name: {
            "selected": value["selected_on_conformal_calibration"],
            "test_delta": value["test"]["delta_accuracy"],
            "oracle_headroom": value["test"]["oracle_headroom"],
            "gate": value["gate"],
        }
        for name, value in channels.items()
    }
    print(json.dumps(compact, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
