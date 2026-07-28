#!/usr/bin/env python3
"""Summarize ANCHOR-NBP pilot outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "anchor-nbp-pilot-analysis-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float | bool]) -> float | None:
    return float(np.mean(values)) if values else None


def binary_balanced_accuracy(rows: list[dict[str, Any]]) -> float | None:
    pos = [r for r in rows if r.get("positive_gt") is True]
    neg = [r for r in rows if r.get("positive_gt") is False]
    if not pos or not neg:
        return safe_mean([bool(r.get("correct")) for r in rows])
    tpr = np.mean([bool(r.get("correct")) for r in pos])
    tnr = np.mean([bool(r.get("correct")) for r in neg])
    return float((tpr + tnr) / 2)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    out: dict[str, Any] = {"n": len(rows), "ok": len(ok), "errors": len(errors)}
    if not ok:
        out["error_examples"] = errors[:3]
        return out
    task = ok[0].get("task")
    if task == "ce":
        out.update(
            {
                "accuracy": safe_mean([bool(r.get("correct")) for r in ok]),
                "balanced_accuracy": binary_balanced_accuracy(ok),
                "parse_rate": safe_mean([bool(r.get("parseable")) for r in ok]),
                "positive_rate": safe_mean([bool(r.get("positive_gt")) for r in ok]),
                "confusion": dict(
                    Counter(
                        f"{r.get('gt')}->{r.get('parsed') if r.get('parsed') is not None else 'invalid'}"
                        for r in ok
                    )
                ),
            }
        )
    else:
        out.update(
            {
                "rouge_l": safe_mean([float(r.get("rouge_l", 0.0)) for r in ok]),
                "avg_length_words": safe_mean([float(r.get("length_words", 0.0)) for r in ok]),
                "radgraph_f1": None,
                "ratescore": None,
                "chexbert_f1": None,
            }
        )
    geoms = [r.get("geometry") for r in ok if isinstance(r.get("geometry"), dict)]
    if geoms:
        out["geometry"] = {
            "mean_delta_norm": safe_mean([float(g.get("delta_norm", 0.0)) for g in geoms]),
            "mean_e_perp": safe_mean([float(g.get("e_perp", 0.0)) for g in geoms]),
            "mean_tangent_energy": safe_mean([float(g.get("tangent_energy", 0.0)) for g in geoms]),
        }
    return out


def rescue_break(rows: list[dict[str, Any]], baseline_method: str = "greedy") -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("task") != "ce" or "error" in row:
            continue
        key = (str(row.get("split")), str(row.get("shift")), str(row.get("id")), int(row.get("random_seed", 0)))
        grouped[key][str(row.get("method"))] = row
    out: dict[str, Any] = {}
    for method in sorted({r.get("method") for r in rows if r.get("method") != baseline_method}):
        if method is None:
            continue
        rescue = break_ = same_correct = same_wrong = n = 0
        for methods in grouped.values():
            if baseline_method not in methods or method not in methods:
                continue
            n += 1
            b = bool(methods[baseline_method].get("correct"))
            m = bool(methods[method].get("correct"))
            rescue += int((not b) and m)
            break_ += int(b and (not m))
            same_correct += int(b and m)
            same_wrong += int((not b) and (not m))
        out[str(method)] = {
            "n_pairs": n,
            "rescue": rescue,
            "break": break_,
            "same_correct": same_correct,
            "same_wrong": same_wrong,
            "net": rescue - break_,
        }
    return out


def eperp_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ce = [r for r in rows if r.get("task") == "ce" and "error" not in r and isinstance(r.get("geometry"), dict)]
    if not ce:
        return {"available": False}
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ce:
        by_method[str(row.get("method"))].append(row)
    out = {}
    for method, subset in by_method.items():
        if method == "greedy" or len(subset) < 8:
            continue
        e = np.asarray([float(r["geometry"].get("e_perp", 0.0)) for r in subset], dtype=np.float64)
        y = np.asarray([float(bool(r.get("correct"))) for r in subset], dtype=np.float64)
        if np.std(e) < 1e-12:
            corr = None
        else:
            corr = float(np.corrcoef(e, y)[0, 1])
        median = float(np.median(e))
        low = [r for r in subset if float(r["geometry"].get("e_perp", 0.0)) <= median]
        high = [r for r in subset if float(r["geometry"].get("e_perp", 0.0)) > median]
        out[method] = {
            "corr_eperp_correct": corr,
            "low_eperp_accuracy": safe_mean([bool(r.get("correct")) for r in low]),
            "high_eperp_accuracy": safe_mean([bool(r.get("correct")) for r in high]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("corrected_runs/final_anchor_nbp_pilot_v1"))
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.raw or args.output_dir / "raw_outputs.jsonl"
    output = args.output or args.output_dir / "summary.json"
    rows = load_jsonl(raw)
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("task")),
                str(row.get("split")),
                str(row.get("shift")),
                str(row.get("method")),
                str(row.get("weights_mode", "reliability")),
                int(row.get("random_seed", 0)),
            )
        ].append(row)
    summary = {
        "version": VERSION,
        "raw": str(raw),
        "groups": {
            "|".join(map(str, key)): summarize_group(value)
            for key, value in sorted(grouped.items(), key=lambda item: item[0])
        },
        "rescue_break_vs_greedy": rescue_break(rows),
        "eperp_analysis": eperp_interaction(rows),
        "notes": {
            "primary": "Shift-D1 CE patient-level deterministic parser balanced accuracy.",
            "clinical_oe_metrics": "RadGraph/RaTEScore/CheXbert are placeholders unless external metric caches are merged.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
