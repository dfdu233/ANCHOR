#!/usr/bin/env python3
"""Leakage-controlled paired Original/DG SCA-T analysis on RULE Yes/No.

RULE contains multiple questions for the same image and, for MIMIC-CXR,
multiple images for the same patient.  Calibration and locked test therefore
split whole clinical groups rather than question ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_source_adapter_scat import (
    METHODS,
    PairedScatError,
    load_inputs,
    paired_metrics,
    run_arm_scat,
)
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, file_sha256


VERSION = "rule-dg-paired-scat-v1"


def clinical_group(row: dict[str, Any], dataset: str) -> str:
    """Return a patient group for MIMIC and an image/study group for IU-Xray."""

    parts = Path(str(row.get("image", row.get("group_id", "")))).parts
    if dataset == "mimic":
        for part in parts:
            if part.startswith("p") and part[1:].isdigit() and len(part) > 3:
                return part
    if dataset == "iuxray" and parts:
        return parts[0]
    group = str(row.get("group_id", row.get("image", "")))
    if not group:
        raise PairedScatError(f"missing clinical group for qid={row.get('qid')}")
    return group


def grouped_split(
    records: list[dict[str, Any]], dataset: str, fraction: float, seed: int
) -> tuple[list[str], list[str], dict[str, str]]:
    """Deterministically assign intact groups near the requested row fraction."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0,1)")
    group_by_qid = {
        str(row["qid"]): clinical_group(row, dataset) for row in records
    }
    qids_by_group: dict[str, list[str]] = {}
    for qid, group in group_by_qid.items():
        qids_by_group.setdefault(group, []).append(qid)
    if len(qids_by_group) < 2:
        raise PairedScatError("grouped split requires at least two clinical groups")
    groups = sorted(
        qids_by_group,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    target = fraction * len(records)
    calibration_groups: set[str] = set()
    count = 0
    for group in groups:
        size = len(qids_by_group[group])
        if not calibration_groups or abs((count + size) - target) <= abs(count - target):
            calibration_groups.add(group)
            count += size
    if len(calibration_groups) == len(groups):
        calibration_groups.remove(groups[-1])
    calibration = sorted(
        (
            qid
            for group in calibration_groups
            for qid in qids_by_group[group]
        ),
        key=str,
    )
    test = sorted(
        (
            qid
            for group in groups
            if group not in calibration_groups
            for qid in qids_by_group[group]
        ),
        key=str,
    )
    if not calibration or not test:
        raise PairedScatError("grouped split produced an empty partition")
    return calibration, test, group_by_qid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prototypes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, cache_meta, prototype_meta, prototypes = load_inputs(
        args.cache, args.prototypes
    )
    dataset = str(cache_meta.get("config", {}).get("rule_dataset", ""))
    if dataset not in {"mimic", "iuxray"}:
        raise PairedScatError(f"unsupported RULE dataset metadata: {dataset}")
    by_qid = {str(row["qid"]): row for row in records}
    calibration_qids, test_qids, group_by_qid = grouped_split(
        records, dataset, args.calibration_fraction, args.seed
    )
    ordered_qids = calibration_qids + test_qids
    ground_truth = np.asarray(
        [int(by_qid[qid]["gt_index"]) for qid in ordered_qids], dtype=np.int64
    )
    calibration_counts = np.bincount(
        ground_truth[: len(calibration_qids)], minlength=2
    )
    style_names = tuple(cache_meta["config"]["style_names"])
    adapted_style = style_names[1]
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    scales: dict[str, float] = {}
    logits_by_arm: dict[str, np.ndarray] = {}
    for arm_index, arm in enumerate(style_names):
        features = np.stack(
            [by_qid[qid]["_features"][arm_index] for qid in ordered_qids]
        )
        logits = np.stack(
            [by_qid[qid]["_logits"][arm_index] for qid in ordered_qids]
        )
        logits_by_arm[arm] = logits
        probabilities[arm], scales[arm] = run_arm_scat(
            features,
            logits,
            prototypes,
            calibration_counts,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            device=args.device,
        )

    test_slice = slice(len(calibration_qids), len(ordered_qids))
    test_gt = ground_truth[test_slice]
    comparisons = {
        "surface": paired_metrics(
            test_gt,
            logits_by_arm["original"][test_slice].argmax(1),
            logits_by_arm[adapted_style][test_slice].argmax(1),
        )
    }
    for method in METHODS:
        comparisons[method] = paired_metrics(
            test_gt,
            probabilities["original"][method][test_slice].argmax(1),
            probabilities[adapted_style][method][test_slice].argmax(1),
        )
    calibration_groups = {group_by_qid[qid] for qid in calibration_qids}
    test_groups = {group_by_qid[qid] for qid in test_qids}
    fingerprint_payload = {
        "version": VERSION,
        "cache_sha256": file_sha256(args.cache),
        "cache_fingerprint": cache_meta["fingerprint"],
        "prototypes_sha256": file_sha256(args.prototypes),
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "split_unit": "patient" if dataset == "mimic" else "image_or_study",
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    report = {
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "source_cache": str(args.cache.resolve()),
        "source_prototypes": str(args.prototypes.resolve()),
        "prototype_model": prototype_meta.get("model"),
        "rule_dataset": dataset,
        "n_yes_no": len(records),
        "split": {
            "seed": args.seed,
            "calibration_fraction_requested": args.calibration_fraction,
            "n_calibration": len(calibration_qids),
            "n_test": len(test_qids),
            "n_calibration_groups": len(calibration_groups),
            "n_test_groups": len(test_groups),
            "group_overlap": sorted(calibration_groups & test_groups),
            "calibration_class_counts": calibration_counts.tolist(),
            "split_unit": fingerprint_payload["split_unit"],
            "shared_across_arms": True,
        },
        "fitted_positive_logit_scale": scales,
        f"paired_{adapted_style}_vs_original": comparisons,
        "scope": {
            "classes": "fixed Yes/No only",
            "routing": "none",
            "transduction": "shared full unlabeled calibration+test pool",
            "label_use": "TIM-KL uses calibration marginal only",
            "claim_boundary": (
                "SCA-T absolute gain is not an SGTA gain; only the paired "
                "adapted-minus-original delta isolates the DG adapter."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
