#!/usr/bin/env python3
"""CPU-only Pareto screen for simple frozen-specialist/VLM cooperation.

This is an idea-search diagnostic, not a proposed method.  It asks whether the
failure of a one-bit specialist veto was merely caused by ignoring the VLM's
own margin.  All rules are selected on image-disjoint development data and
then evaluated once on confirmation data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from anchor.corrected_sgta.screen_external_visual_increment_v1 import load_claims
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import FINDING_TARGETS, XRV_LABELS


VERSION = "expert-margin-pareto-fatal-v1"


def load_experts(nih_path: Path, domain_path: Path) -> dict[str, dict[str, np.ndarray]]:
    nih = np.load(nih_path, allow_pickle=False)
    domain = np.load(domain_path, allow_pickle=False)
    if [str(x) for x in nih["labels"]] != list(XRV_LABELS):
        raise ValueError("NIH label order drift")
    if [str(x) for x in domain["labels"]] != list(XRV_LABELS):
        raise ValueError("domain label order drift")
    output: dict[str, dict[str, np.ndarray]] = {}
    for image_id, logits in zip(nih["image_ids"], nih["logits"]):
        output.setdefault(str(image_id), {})["nih"] = np.asarray(logits, dtype=np.float64)
    domains = [str(x) for x in domain["domains"]]
    for d_idx, name in enumerate(domains):
        for image_id, logits in zip(domain["image_ids"], domain["logits"][d_idx]):
            output.setdefault(str(image_id), {})[name] = np.asarray(logits, dtype=np.float64)
    if any(set(v) != {"nih", *domains} for v in output.values()):
        raise ValueError("expert image coverage mismatch")
    return output


def attach(rows: list[dict[str, Any]], experts: dict[str, dict[str, np.ndarray]]) -> None:
    label_index = {name: i for i, name in enumerate(XRV_LABELS)}
    for row in rows:
        per_domain = []
        for domain in sorted(experts[row["image_id"]]):
            logits = experts[row["image_id"]][domain]
            per_domain.append(max(float(logits[label_index[x]]) for x in FINDING_TARGETS[row["finding"]]))
        row["expert_vector"] = np.asarray(per_domain, dtype=np.float64)


def percentile(values: np.ndarray, references: np.ndarray) -> np.ndarray:
    ordered = np.sort(references)
    return np.searchsorted(ordered, values, side="right") / max(len(ordered), 1)


def add_development_ranks(dev: list[dict[str, Any]], test: list[dict[str, Any]], agg: str) -> None:
    reducer = {"min": np.min, "median": np.median, "max": np.max}[agg]
    for row in dev + test:
        row[f"expert_{agg}"] = float(reducer(row["expert_vector"]))
    for finding in sorted({row["finding"] for row in dev}):
        d = [row for row in dev if row["finding"] == finding]
        t = [row for row in test if row["finding"] == finding]
        for key in ("margin", f"expert_{agg}"):
            ref = np.asarray([row[key] for row in d])
            for rows in (d, t):
                ranks = percentile(np.asarray([row[key] for row in rows]), ref)
                for row, rank in zip(rows, ranks):
                    row[f"rank_{key}_{agg}"] = float(rank)


def outcome(rows: list[dict[str, Any]], agg: str, qm: float, qe: float) -> dict[str, Any]:
    drafts = [row for row in rows if row["margin"] > 0]
    fired = [
        row for row in drafts
        if row[f"rank_margin_{agg}"] <= qm and row[f"rank_expert_{agg}_{agg}"] <= qe
    ]
    tp = [row for row in drafts if row["label"] == 1]
    fp = [row for row in drafts if row["label"] == 0]
    tp_fire = sum(row["label"] == 1 for row in fired)
    fp_fire = sum(row["label"] == 0 for row in fired)
    return {
        "positive_drafts": len(drafts),
        "tp": len(tp),
        "fp": len(fp),
        "triggered": len(fired),
        "triggered_tp": tp_fire,
        "triggered_fp": fp_fire,
        "tp_harm_rate": tp_fire / max(len(tp), 1),
        "fp_removal_rate": fp_fire / max(len(fp), 1),
        "net_errors_removed": fp_fire - tp_fire,
    }


def select_rule(rows: list[dict[str, Any]], max_tp_harm: float) -> dict[str, Any]:
    grid = np.linspace(0.025, 0.8, 32)
    candidates = []
    for agg in ("min", "median", "max"):
        for qm in grid:
            for qe in grid:
                result = outcome(rows, agg, float(qm), float(qe))
                if result["tp_harm_rate"] <= max_tp_harm:
                    candidates.append((result["fp_removal_rate"], result["net_errors_removed"], -qm * qe, agg, qm, qe, result))
    if not candidates:
        return {"agg": "median", "qm": 0.0, "qe": 0.0, "development": outcome(rows, "median", 0.0, 0.0)}
    _, _, _, agg, qm, qe, result = max(candidates)
    return {"agg": agg, "qm": float(qm), "qe": float(qe), "development": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--nih-logits", type=Path, required=True)
    parser.add_argument("--domain-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-development-tp-harm", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    experts = load_experts(args.nih_logits, args.domain_logits)
    paths = {
        "huatuo": (args.huatuo_dev, args.huatuo_confirmation),
        "hulu": (args.hulu_dev, args.hulu_confirmation),
    }
    result = {"version": VERSION, "models": {}}
    for model, (dev_path, test_path) in paths.items():
        dev = load_claims(dev_path, "development", model)
        test = load_claims(test_path, "confirmation", model)
        attach(dev, experts)
        attach(test, experts)
        for agg in ("min", "median", "max"):
            add_development_ranks(dev, test, agg)
        rule = select_rule(dev, args.max_development_tp_harm)
        rule["confirmation"] = outcome(test, rule["agg"], rule["qm"], rule["qe"])
        result["models"][model] = rule
    passes = [
        x["confirmation"]["fp_removal_rate"] >= 0.20
        and x["confirmation"]["tp_harm_rate"] <= 0.01
        and x["confirmation"]["net_errors_removed"] > 0
        for x in result["models"].values()
    ]
    result["decision"] = "PASS_NECESSARY_CONDITION" if all(passes) else "NO_GO"
    result["claim_boundary"] = "Optimistic CE deletion screen only; not an OE mitigation method."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
