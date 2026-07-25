#!/usr/bin/env python3
"""Offline CE baselines, SGTA, LAME, LATA, LAC and APS from one GPU cache."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.methods import (
    conformal_quantile,
    entropy_weighted_fusion,
    lame_rbf_affinity,
    laplacian_optimization,
    lata_refine,
    softmax_np,
)
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, nargs="*", default=(0.1, 0.05))
    parser.add_argument("--sgta-gamma", type=float, default=0.35)
    parser.add_argument("--sgta-iterations", type=int, default=8)
    parser.add_argument("--transductive-window", type=int, default=256)
    parser.add_argument("--lata-gamma", type=float, default=1.0)
    parser.add_argument("--lata-knn", type=int, default=5)
    return parser.parse_args()


def sgta_probabilities(
    logits: np.ndarray, features: np.ndarray, gamma: float, iterations: int
) -> np.ndarray:
    """The proposed intra-image style graph, distinct from LAME/LATA.

    This preserves the legacy SGTA update but uses correct class logits and
    normalized last-prompt features.  The graph is full RBF across styles.
    """

    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    feature_tensor = F.normalize(
        torch.tensor(features, dtype=torch.float32), p=2, dim=-1
    )
    distance = torch.cdist(feature_tensor, feature_tensor)
    nonzero = distance[distance > 0]
    sigma = nonzero.mean().clamp_min(1e-6) if len(nonzero) else torch.tensor(1.0)
    kernel = torch.exp(-(distance**2) / (2 * sigma**2))
    z = logit_tensor.softmax(-1)
    for _ in range(iterations):
        z = (logit_tensor + gamma * kernel.matmul(z)).softmax(-1)
    return z.mean(0).numpy()


def point_summary(predictions: dict[str, int], records: list[dict]) -> dict:
    rows = [row for row in records if str(row["qid"]) in predictions]
    correct = [predictions[str(row["qid"])] == row["gt_index"] for row in rows]
    by_kind = {}
    for kind in ("binary", "multichoice"):
        subset = [row for row in rows if row["question_type"] == kind]
        by_kind[kind] = {
            "n": len(subset),
            "accuracy": float(
                np.mean(
                    [predictions[str(row["qid"])] == row["gt_index"] for row in subset]
                )
            )
            if subset
            else None,
        }
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if rows else None,
        "by_kind": by_kind,
    }


def apply_in_windows(
    records: list[dict], window: int, method: str, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    """Apply fixed-class transduction in deterministic qid-hash windows."""

    binary = [
        row
        for row in records
        if row["question_type"] == "binary" and row["labels"] == ["Yes", "No"]
    ]
    # The split key is label-free and stable under cache row ordering.
    binary.sort(
        key=lambda row: (
            __import__("hashlib")
            .sha256(f"{args.seed}:{row['qid']}".encode())
            .hexdigest()
        )
    )
    result: dict[str, np.ndarray] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for start in range(0, len(binary), window):
        chunk = binary[start : start + window]
        logits = torch.tensor(
            np.stack([row["style_logits"][0] for row in chunk]), device=device
        )
        probabilities = logits.softmax(-1)
        features = torch.tensor(
            np.stack([row["features"][0] for row in chunk]), device=device
        )
        if method == "lame":
            kernel = lame_rbf_affinity(features, knn=5, force_symmetry=False)
            refined = laplacian_optimization(probabilities, kernel)
        elif method == "lata":
            refined = lata_refine(
                probabilities,
                features,
                gamma=args.lata_gamma,
                knn=args.lata_knn,
                iterations=10,
            )
        else:
            raise ValueError(method)
        for row, prob in zip(chunk, refined.float().cpu().numpy()):
            result[str(row["qid"])] = prob
        del logits, probabilities, features, refined
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def aps_all_scores(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(-probabilities)
    ordered = probabilities[order]
    cumulative = np.cumsum(ordered)
    scores_ordered = cumulative - ordered * rng.random(len(ordered))
    scores = np.empty_like(scores_ordered)
    scores[order] = scores_ordered
    return scores


def conformal_report(
    name: str,
    probabilities: dict[str, np.ndarray],
    records_by_qid: dict[str, dict],
    calibration_qids: list[str],
    test_qids: list[str],
    alpha_values: list[float],
    seed: int,
) -> dict:
    available_cal = [qid for qid in calibration_qids if qid in probabilities]
    available_test = [qid for qid in test_qids if qid in probabilities]
    output = {
        "name": name,
        "n_calibration": len(available_cal),
        "n_test": len(available_test),
        "lac": {},
        "aps": {},
    }
    lac_cal_scores = [
        1.0 - probabilities[qid][records_by_qid[qid]["gt_index"]]
        for qid in available_cal
    ]
    rng = np.random.default_rng(seed)
    aps_cal_scores = []
    for qid in available_cal:
        scores = aps_all_scores(probabilities[qid], rng)
        aps_cal_scores.append(scores[records_by_qid[qid]["gt_index"]])
    test_aps_scores = {
        qid: aps_all_scores(probabilities[qid], rng) for qid in available_test
    }

    for alpha in alpha_values:
        lac_threshold = conformal_quantile(lac_cal_scores, alpha)
        lac_sets = {
            qid: (1.0 - probabilities[qid]) <= lac_threshold for qid in available_test
        }
        aps_threshold = conformal_quantile(aps_cal_scores, alpha)
        aps_sets = {
            qid: test_aps_scores[qid] <= aps_threshold for qid in available_test
        }
        for label, threshold, sets in (
            ("lac", lac_threshold, lac_sets),
            ("aps", aps_threshold, aps_sets),
        ):
            coverage = [
                bool(sets[qid][records_by_qid[qid]["gt_index"]])
                for qid in available_test
            ]
            sizes = [int(sets[qid].sum()) for qid in available_test]
            answered = [qid for qid in available_test if int(sets[qid].sum()) == 1]
            singleton_accuracy = (
                np.mean(
                    [
                        int(np.flatnonzero(sets[qid])[0])
                        == records_by_qid[qid]["gt_index"]
                        for qid in answered
                    ]
                )
                if answered
                else math.nan
            )
            output[label][str(alpha)] = {
                "threshold": float(threshold),
                "coverage": float(np.mean(coverage)) if coverage else None,
                "average_set_size": float(np.mean(sizes)) if sizes else None,
                "singleton_rate": len(answered) / len(available_test)
                if available_test
                else None,
                "singleton_accuracy": None
                if math.isnan(singleton_accuracy)
                else float(singleton_accuracy),
            }
    return output


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    fingerprint = metadata["fingerprint"]
    records = list(iter_successes(args.cache, fingerprint))
    for row in records:
        row["style_logits"] = np.asarray(row["style_logits"], dtype=np.float32)
        row["features"] = decode_array(row["style_features"]).astype(np.float32)

    predictions: dict[str, dict[str, int]] = defaultdict(dict)
    probabilities: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in records:
        qid = str(row["qid"])
        logits = row["style_logits"]
        features = row["features"]
        feddg_indices = [
            index
            for index, style in enumerate(row["style_names"])
            if style == "feddg_center" or style.startswith("feddg_")
        ]
        inferred_feddg = logits[feddg_indices[0]] if feddg_indices else logits[0]
        domain_center_mean = (
            logits[feddg_indices].mean(0) if feddg_indices else logits[0]
        )
        method_logits = {
            "baseline": logits[0],
            "feddg_center": inferred_feddg,
            "domain_center_mean": domain_center_mean,
            "tta_mean": logits.mean(0),
            "tta_entropy": entropy_weighted_fusion(logits),
        }
        for method, value in method_logits.items():
            probabilities[method][qid] = softmax_np(value)
            predictions[method][qid] = int(np.argmax(value))
        sgta = sgta_probabilities(
            logits, features, args.sgta_gamma, args.sgta_iterations
        )
        probabilities["sgta"][qid] = sgta
        predictions["sgta"][qid] = int(np.argmax(sgta))

    for method in ("lame", "lata"):
        adapted = apply_in_windows(records, args.transductive_window, method, args)
        probabilities[method] = adapted
        predictions[method] = {
            qid: int(np.argmax(prob)) for qid, prob in adapted.items()
        }

    qids = [str(row["qid"]) for row in records]
    calibration_qids, test_qids = deterministic_split(
        qids, args.calibration_fraction, args.seed
    )
    by_qid = {str(row["qid"]): row for row in records}
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "source_cache": str(args.cache),
        "fingerprint": fingerprint,
        "n": len(records),
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "n_calibration": len(calibration_qids),
            "n_test": len(test_qids),
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
        },
        "point_accuracy": {
            method: point_summary(value, records)
            for method, value in predictions.items()
        },
        "point_accuracy_test_only": {
            method: point_summary(
                value, [by_qid[qid] for qid in test_qids if qid in by_qid]
            )
            for method, value in predictions.items()
        },
        "conformal": {
            method: conformal_report(
                method,
                value,
                by_qid,
                calibration_qids,
                test_qids,
                list(args.alpha),
                args.seed,
            )
            for method, value in probabilities.items()
        },
        "method_scope": {
            "baseline/feddg_center/domain_center_mean/tta/sgta": "all valid finite-label rows",
            "feddg_center": "first inferred-domain FedDG view; legacy name retained for compatibility",
            "domain_center_mean": "uniform logit mean over every FedDG domain-center view",
            "point_accuracy_test_only": "held-out qid-hash test split; preferred for paper tables",
            "lame/lata": "Yes/No rows only; fixed semantic class space; deterministic windows",
            "sgta": "per-image style graph; not reported as upstream LAME or LATA",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["point_accuracy"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
