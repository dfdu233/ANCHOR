#!/usr/bin/env python3
"""Bad-case mechanism audit for source-center interventions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.cache import decode_array
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split
from corrected_sgta.protocol import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--geometry-raw", action="append", default=[], type=Path)
    parser.add_argument(
        "--visual-dependency", action="append", default=[], metavar="NAME=JSONL"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.clip(values.sum(), 1e-12, None)


def entropy(prob: np.ndarray) -> float:
    clipped = np.clip(prob, 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def margin(prob: np.ndarray) -> float:
    if len(prob) < 2:
        return 0.0
    values = np.sort(prob)
    return float(values[-1] - values[-2])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    )


def mean(values: list[Any]) -> float | None:
    usable = [float(value) for value in values if value is not None and value != ""]
    return float(np.mean(usable)) if usable else None


def slot_name(item: dict[str, Any], index: int) -> str:
    if index == 0 or item.get("family") == "original":
        return "original"
    params = item.get("parameters") or {}
    source_ratio = params.get("source_ratio")
    if source_ratio is not None:
        return f"matched_sr{float(source_ratio):g}"
    return str(item.get("family") or f"style_{index}")


def outcome_name(row: dict[str, Any]) -> str:
    if row["rescue"]:
        return "rescue"
    if row["harm"]:
        return "harm"
    if row["changed"]:
        return "changed_neutral"
    return "unchanged"


def read_geometry(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    geometry: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                geometry[(row["run"], str(row["qid"]), row["slot"])] = row
    return geometry


def read_visual_dependency(paths: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in paths:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "ok":
                    rows[(name, str(row["qid"]))] = row
    return rows


def load_run(
    name: str,
    path: Path,
    geometry: dict[tuple[str, str, str], dict[str, Any]],
    visual_dependency: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"{path}: protocol mismatch")
    fingerprint = metadata["fingerprint"]
    records = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "ok":
                raise RuntimeError(f"{path}: non-ok row qid={row.get('qid')}")
            if row.get("fingerprint") != fingerprint:
                raise RuntimeError(f"{path}: fingerprint mismatch qid={row.get('qid')}")
            records.append(row)

    rows: list[dict[str, Any]] = []
    for record in records:
        qid = str(record["qid"])
        logits = np.asarray(record["style_logits"], dtype=np.float64)
        features = decode_array(record["style_features"]).astype(np.float64)
        metadata_items = record["style_metadata"]
        if len(logits) != len(features) or len(logits) != len(metadata_items):
            raise RuntimeError(f"{path}: style count mismatch at qid={qid}")
        base_prob = probabilities(logits[0])
        base_pred = int(np.argmax(base_prob))
        base_correct = int(base_pred == int(record["gt_index"]))
        base_norm = float(np.linalg.norm(features[0]))
        vd = visual_dependency.get((name, qid), {})
        blank_prob = None
        blank_pred = None
        if vd:
            blank_prob = np.asarray(vd["style_probabilities"][1], dtype=np.float64)
            blank_pred = int(vd["blank_pred"])
        for index, item in enumerate(metadata_items):
            slot = slot_name(item, index)
            if slot == "original":
                continue
            prob = probabilities(logits[index])
            pred = int(np.argmax(prob))
            correct = int(pred == int(record["gt_index"]))
            structure = item.get("structure") or {}
            psnr = structure.get("psnr")
            edge = structure.get("edge_correlation")
            geom = geometry.get((name, qid, slot), {})
            feature_l2 = float(np.linalg.norm(features[index] - features[0]))
            base_blank_js = (
                None if blank_prob is None else js_divergence(base_prob, blank_prob)
            )
            style_blank_js = (
                None if blank_prob is None else js_divergence(prob, blank_prob)
            )
            row = {
                "run": name,
                "qid": qid,
                "slot": slot,
                "domain_id": item.get("domain_id", "unknown"),
                "source_ratio": (item.get("parameters") or {}).get("source_ratio"),
                "gt_index": int(record["gt_index"]),
                "base_pred": base_pred,
                "style_pred": pred,
                "base_correct": base_correct,
                "style_correct": correct,
                "rescue": int((not base_correct) and bool(correct)),
                "harm": int(bool(base_correct) and not bool(correct)),
                "changed": int(pred != base_pred),
                "base_confidence": float(base_prob.max()),
                "style_confidence": float(prob.max()),
                "confidence_delta": float(prob.max() - base_prob.max()),
                "base_margin": margin(base_prob),
                "style_margin": margin(prob),
                "margin_delta": margin(prob) - margin(base_prob),
                "base_entropy": entropy(base_prob),
                "style_entropy": entropy(prob),
                "entropy_delta": entropy(prob) - entropy(base_prob),
                "prob_l1": float(np.abs(prob - base_prob).sum()),
                "logit_l2": float(np.linalg.norm(logits[index] - logits[0])),
                "logit_js_divergence": js_divergence(base_prob, prob),
                "pixel_mse": float(structure.get("pixel_mse", 0.0)),
                "psnr": None if psnr is None else float(psnr),
                "edge_correlation": None if edge is None else float(edge),
                "feature_cosine_distance": cosine_distance(features[0], features[index]),
                "feature_l2": feature_l2,
                "feature_l2_over_base_norm": feature_l2 / max(base_norm, 1e-12),
                "feature_norm_delta": float(np.linalg.norm(features[index]) - base_norm),
                "rrmse_before": none_float(geom.get("rrmse_before")),
                "rrmse_after": none_float(geom.get("rrmse_after")),
                "rrmse_relative_closure": none_float(geom.get("rrmse_relative_closure")),
                "cosine_relative_closure": none_float(geom.get("cosine_relative_closure")),
                "visual_dependent": none_int(vd.get("visual_dependent")) if vd else None,
                "image_helpful": none_int(
                    bool(vd.get("image_correct")) and not bool(vd.get("blank_correct"))
                )
                if vd
                else None,
                "image_harmful": none_int(
                    (not bool(vd.get("image_correct"))) and bool(vd.get("blank_correct"))
                )
                if vd
                else None,
                "blank_pred": blank_pred,
                "blank_correct": none_int(vd.get("blank_correct")) if vd else None,
                "max_prob_delta_image_blank": none_float(vd.get("max_prob_delta")) if vd else None,
                "max_logit_delta_image_blank": none_float(vd.get("max_logit_delta")) if vd else None,
                "base_agrees_blank": None
                if blank_pred is None
                else int(base_pred == blank_pred),
                "style_agrees_blank": None
                if blank_pred is None
                else int(pred == blank_pred),
                "base_blank_js": base_blank_js,
                "style_blank_js": style_blank_js,
                "style_minus_base_blank_js": None
                if base_blank_js is None or style_blank_js is None
                else float(style_blank_js - base_blank_js),
            }
            row["outcome"] = outcome_name(row)
            rows.append(row)

    info = {
        "name": name,
        "cache": str(path),
        "cache_sha256": file_sha256(path),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "fingerprint": fingerprint,
        "n_ok": len(records),
        "config": metadata.get("config"),
    }
    return info, rows


def none_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def none_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(bool(value))


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, items in sorted(groups.items()):
        base = {key: value for key, value in zip(keys, group)}
        n = len(items)
        baseline = np.asarray([item["base_correct"] for item in items], dtype=float)
        styled = np.asarray([item["style_correct"] for item in items], dtype=float)
        base.update(
            {
                "n": n,
                "baseline_accuracy": float(baseline.mean()) if n else None,
                "style_accuracy": float(styled.mean()) if n else None,
                "delta": float((styled - baseline).mean()) if n else None,
                "rescue_rate": mean([item["rescue"] for item in items]),
                "harm_rate": mean([item["harm"] for item in items]),
                "changed_rate": mean([item["changed"] for item in items]),
                "mean_psnr": mean([item["psnr"] for item in items]),
                "mean_edge_correlation": mean([item["edge_correlation"] for item in items]),
                "mean_feature_cosine_distance": mean([item["feature_cosine_distance"] for item in items]),
                "mean_logit_js_divergence": mean([item["logit_js_divergence"] for item in items]),
                "mean_rrmse_relative_closure": mean([item["rrmse_relative_closure"] for item in items]),
                "mean_style_minus_base_blank_js": mean(
                    [item["style_minus_base_blank_js"] for item in items]
                ),
                "style_agrees_blank_rate": mean([item["style_agrees_blank"] for item in items]),
            }
        )
        output.append(base)
    return output


def bin_summary(rows: list[dict[str, Any]], features: list[str], bins: int = 4) -> list[dict[str, Any]]:
    output = []
    for feature in features:
        values = np.asarray([row[feature] for row in rows if row.get(feature) is not None], dtype=float)
        if len(values) < bins:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) <= 2:
            continue
        for left, right in zip(edges[:-1], edges[1:]):
            bucket = [
                row
                for row in rows
                if row.get(feature) is not None
                and float(row[feature]) >= left
                and (float(row[feature]) <= right if right == edges[-1] else float(row[feature]) < right)
            ]
            if not bucket:
                continue
            output.append(
                {
                    "feature": feature,
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "n": len(bucket),
                    "rescue_rate": mean([row["rescue"] for row in bucket]),
                    "harm_rate": mean([row["harm"] for row in bucket]),
                    "changed_rate": mean([row["changed"] for row in bucket]),
                    "delta": mean([row["style_correct"] - row["base_correct"] for row in bucket]),
                }
            )
    return output


def evaluate_gate(rows: list[dict[str, Any]], feature: str, op: str, threshold: float) -> dict[str, float]:
    base = np.asarray([row["base_correct"] for row in rows], dtype=float)
    styled = []
    used = []
    for row in rows:
        value = row.get(feature)
        if value is None:
            accept = False
        elif op == ">=":
            accept = float(value) >= threshold
        elif op == "<=":
            accept = float(value) <= threshold
        else:
            raise ValueError(op)
        used.append(float(accept))
        styled.append(row["style_correct"] if accept else row["base_correct"])
    styled_array = np.asarray(styled, dtype=float)
    return {
        "accuracy": float(styled_array.mean()) if len(rows) else math.nan,
        "delta": float((styled_array - base).mean()) if len(rows) else math.nan,
        "use_style_rate": float(np.mean(used)) if len(rows) else math.nan,
    }


def gate_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["run"], row["slot"])].append(row)
    candidate_features = {
        "psnr": [">="],
        "edge_correlation": [">="],
        "feature_cosine_distance": ["<="],
        "feature_l2_over_base_norm": ["<="],
        "logit_js_divergence": ["<="],
        "prob_l1": ["<="],
        "confidence_delta": ["<=", ">="],
        "entropy_delta": ["<=", ">="],
        "rrmse_relative_closure": [">=", "<="],
        "style_minus_base_blank_js": ["<=", ">="],
        "base_blank_js": [">=", "<="],
    }
    output = []
    for (run, slot), items in sorted(by_group.items()):
        qids = sorted({row["qid"] for row in items})
        cal_qids, test_qids = deterministic_split(qids, args.calibration_fraction, args.seed)
        cal = [row for row in items if row["qid"] in set(cal_qids)]
        test = [row for row in items if row["qid"] in set(test_qids)]
        if not cal or not test:
            continue
        baseline_cal = evaluate_gate(cal, "psnr", ">=", float("inf"))
        baseline_test = evaluate_gate(test, "psnr", ">=", float("inf"))
        always_cal = {
            "accuracy": mean([row["style_correct"] for row in cal]),
            "delta": mean([row["style_correct"] - row["base_correct"] for row in cal]),
            "use_style_rate": 1.0,
        }
        always_test = {
            "accuracy": mean([row["style_correct"] for row in test]),
            "delta": mean([row["style_correct"] - row["base_correct"] for row in test]),
            "use_style_rate": 1.0,
        }
        output.append(candidate_row(run, slot, "always_style", "all", "", None, cal, test, always_cal, always_test))
        output.append(candidate_row(run, slot, "always_baseline", "none", "", None, cal, test, baseline_cal, baseline_test))
        for feature, ops in candidate_features.items():
            values = sorted({float(row[feature]) for row in cal if row.get(feature) is not None})
            if len(values) < 2:
                continue
            thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
            for op in ops:
                best = None
                for threshold in thresholds:
                    cal_eval = evaluate_gate(cal, feature, op, float(threshold))
                    test_eval = evaluate_gate(test, feature, op, float(threshold))
                    row = candidate_row(run, slot, "threshold", feature, op, float(threshold), cal, test, cal_eval, test_eval)
                    key = (row["cal_delta"], row["cal_accuracy"], -row["cal_use_style_rate"])
                    if best is None or key > best[0]:
                        best = (key, row)
                if best is not None:
                    output.append(best[1])
    return output


def candidate_row(
    run: str,
    slot: str,
    rule_type: str,
    feature: str,
    op: str,
    threshold: float | None,
    cal: list[dict[str, Any]],
    test: list[dict[str, Any]],
    cal_eval: dict[str, float],
    test_eval: dict[str, float],
) -> dict[str, Any]:
    return {
        "run": run,
        "slot": slot,
        "rule_type": rule_type,
        "feature": feature,
        "operator": op,
        "threshold": threshold,
        "n_cal": len(cal),
        "n_test": len(test),
        "cal_accuracy": cal_eval["accuracy"],
        "cal_delta": cal_eval["delta"],
        "cal_use_style_rate": cal_eval["use_style_rate"],
        "test_accuracy": test_eval["accuracy"],
        "test_delta": test_eval["delta"],
        "test_use_style_rate": test_eval["use_style_rate"],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = read_geometry(args.geometry_raw)
    visual_dependency = read_visual_dependency(args.visual_dependency)
    run_info = []
    rows: list[dict[str, Any]] = []
    for spec in args.run:
        name, raw_path = spec.split("=", 1)
        info, run_rows = load_run(name, Path(raw_path), geometry, visual_dependency)
        run_info.append(info)
        rows.extend(run_rows)

    slot_summary = summarize(rows, ["run", "slot"])
    domain_summary = summarize(rows, ["run", "domain_id", "slot"])
    outcome_summary = summarize(rows, ["run", "slot", "outcome"])
    bins = bin_summary(
        rows,
        [
            "psnr",
            "edge_correlation",
            "feature_cosine_distance",
            "feature_l2_over_base_norm",
            "logit_js_divergence",
            "prob_l1",
            "confidence_delta",
            "entropy_delta",
            "rrmse_relative_closure",
            "style_minus_base_blank_js",
            "base_blank_js",
            "style_blank_js",
        ],
    )
    candidates = gate_candidates(rows, args)
    best_candidates = []
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_group[(row["run"], row["slot"])].append(row)
    for (run, slot), group in sorted(by_group.items()):
        best = max(
            group,
            key=lambda row: (
                row["cal_delta"],
                row["cal_accuracy"],
                -row["cal_use_style_rate"],
                row["test_delta"],
            ),
        )
        best_candidates.append(best)

    files = {
        "raw": args.output_dir / "mechanism_rows_v1.tsv",
        "slot_summary": args.output_dir / "mechanism_slot_summary_v1.tsv",
        "domain_summary": args.output_dir / "mechanism_domain_summary_v1.tsv",
        "outcome_summary": args.output_dir / "mechanism_outcome_summary_v1.tsv",
        "bin_summary": args.output_dir / "mechanism_bin_summary_v1.tsv",
        "gate_candidates": args.output_dir / "mechanism_gate_candidates_v1.tsv",
        "best_gate_candidates": args.output_dir / "mechanism_best_gate_candidates_v1.tsv",
    }
    write_tsv(files["raw"], rows)
    write_tsv(files["slot_summary"], slot_summary)
    write_tsv(files["domain_summary"], domain_summary)
    write_tsv(files["outcome_summary"], outcome_summary)
    write_tsv(files["bin_summary"], bins)
    write_tsv(files["gate_candidates"], candidates)
    write_tsv(files["best_gate_candidates"], best_candidates)

    summary = {
        "version": "source-center-mechanism-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Analyze when matched source-center image interventions rescue or harm VLM predictions, using existing caches only.",
        "runs": run_info,
        "n_rows": len(rows),
        "slot_summary": slot_summary,
        "outcome_summary": outcome_summary,
        "best_gate_candidates": best_candidates,
        "interpretation_boundary": [
            "Gate candidates are diagnostic single-feature certificates selected on calibration splits; do not claim them as a final method without cross-model validation.",
            "Center-distance fields are after-transfer only when joined from geometry_raw_v1.tsv; cached center_distance_before alone is not used as proof of movement.",
        ],
        "files": {key: str(path) for key, path in files.items()},
    }
    summary_path = args.output_dir / "mechanism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": str(summary_path), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
