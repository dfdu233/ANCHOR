#!/usr/bin/env python3
"""Preregistered surface screen for content--commitment view dissociation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.oe_metrics import token_f1


def uncertainty_rate(text: str, patterns: list[re.Pattern[str]]) -> float:
    sentences = [part.strip() for part in re.split(r"[.!?\n]+", text) if part.strip()]
    if not sentences:
        return 0.0
    uncertain = sum(any(pattern.search(sentence.lower()) for pattern in patterns) for sentence in sentences)
    return uncertain / len(sentences)


def analyze(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    patterns = [re.compile(pattern, re.I) for pattern in config["uncertainty_patterns"]]
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["id"]), str(row["conv_mode"]), str(row["prompt_mode"]))
        grouped[key][str(row["view"])] = row
    controls = [config["primary_control"], *config["secondary_controls"]]
    comparisons: dict[str, Any] = {}
    for control in controls:
        records = []
        for (item_id, conv_mode, prompt_mode), views in grouped.items():
            if "real" not in views or control not in views:
                continue
            real_text = str(views["real"].get("text", ""))
            control_text = str(views[control].get("text", ""))
            real_commitment = uncertainty_rate(real_text, patterns)
            control_commitment = uncertainty_rate(control_text, patterns)
            records.append({
                "id": item_id,
                "conv_mode": conv_mode,
                "prompt_mode": prompt_mode,
                "content_token_f1": token_f1(real_text, control_text),
                "content_change": 1.0 - token_f1(real_text, control_text),
                "real_uncertainty_rate": real_commitment,
                "control_uncertainty_rate": control_commitment,
                "commitment_shift": abs(real_commitment - control_commitment),
            })
        ids = sorted({record["id"] for record in records})
        by_id = {item_id: [row for row in records if row["id"] == item_id] for item_id in ids}

        def aggregate(selected_ids: list[str]) -> tuple[float, float, float]:
            selected = [row for item_id in selected_ids for row in by_id[item_id]]
            content = float(np.mean([row["content_change"] for row in selected]))
            commitment = float(np.mean([row["commitment_shift"] for row in selected]))
            return content, commitment, content - commitment

        observed = aggregate(ids) if ids else (0.0, 0.0, 0.0)
        rng = np.random.default_rng(int(config["bootstrap_seed"]))
        bootstrap = []
        if ids:
            for _ in range(int(config["bootstrap_replicates"])):
                sampled = [ids[index] for index in rng.integers(0, len(ids), len(ids))]
                bootstrap.append(aggregate(sampled)[2])
        ci = (
            [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
            if bootstrap else [None, None]
        )
        comparisons[control] = {
            "n_pairs": len(records),
            "n_image_clusters": len(ids),
            "mean_content_change": observed[0],
            "mean_content_token_f1": 1.0 - observed[0],
            "mean_commitment_shift": observed[1],
            "content_minus_commitment_dissociation": observed[2],
            "dissociation_cluster_bootstrap_ci95": ci,
            "mean_real_uncertainty_rate": (
                float(np.mean([row["real_uncertainty_rate"] for row in records]))
                if records else 0.0
            ),
            "mean_control_uncertainty_rate": (
                float(np.mean([row["control_uncertainty_rate"] for row in records]))
                if records else 0.0
            ),
        }
    primary = comparisons[config["primary_control"]]
    gate = bool(
        primary["n_pairs"] >= int(config["minimum_complete_primary_pairs"])
        and primary["mean_content_change"] >= float(config["minimum_mean_content_change"])
        and primary["mean_commitment_shift"] <= float(config["maximum_mean_commitment_shift"])
        and primary["dissociation_cluster_bootstrap_ci95"][0] is not None
        and primary["dissociation_cluster_bootstrap_ci95"][0] > 0
    )
    return {
        "protocol": config["protocol"],
        "interpretation": config["interpretation"],
        "comparisons": comparisons,
        "exploratory_dissociation_gate_pass": gate,
        "claim_grade_allowed": False,
        "notes": [
            "Uncertainty markers are surface proxies, not clinical certainty truth.",
            "Uniform definiteness can pass this screen and requires reader-vote follow-up.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    config = json.loads(args.config.read_text())
    result = analyze(rows, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
