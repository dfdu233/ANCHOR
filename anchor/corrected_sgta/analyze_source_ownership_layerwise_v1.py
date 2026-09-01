#!/usr/bin/env python3
"""Analyze polarity, source, and binding trajectories on discovery data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_source_ownership_factorial_v1 import atomic_json, sha256_file
from corrected_sgta.run_target_blind_dicom_tristate_margin_v2 import EXPECTED_ARMS, load_rows


VERSION = "source-ownership-layerwise-analysis-v1"


def cluster_summary(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "n_images": int(len(values)), "mean": float(values.mean()),
        "positive_fraction": float((values > 0).mean()),
        "image_cluster_bootstrap_ci95": [float(x) for x in np.quantile(means, [.025, .975])],
    }


def build_layer_blocks(manifest: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_qid = {str(row["qid"]): row for row in manifest}
    measured = {str(row["qid"]): row for row in rows}
    if set(manifest_by_qid) != set(measured) or len(measured) != len(rows):
        raise ValueError("manifest and unique layerwise qids must match exactly")
    grouped: dict[str, list[str]] = {}
    for qid, row in manifest_by_qid.items():
        grouped.setdefault(str(row["pair_id"]), []).append(qid)
    blocks = []
    layer_sets = {
        tuple(sorted(map(int, measured[qid]["layers"]))) for qid in measured
    }
    if len(layer_sets) != 1:
        raise ValueError("inconsistent layer sets")
    layers = next(iter(layer_sets))
    for pair_id, qids in grouped.items():
        arms = {str(manifest_by_qid[qid]["arm"]): qid for qid in qids}
        if set(arms) != EXPECTED_ARMS:
            raise ValueError(f"incomplete arms for {pair_id}")
        finding = str(manifest_by_qid[qids[0]]["finding"])
        image = str(manifest_by_qid[qids[0]]["img_name"])
        for layer in layers:
            score = {
                arm: float(measured[qid]["layers"][str(layer)]["polarity_yes_minus_no"])
                for arm, qid in arms.items()
            }
            pc = score["current_present"] - score["current_absent"]
            po = score["other_present"] - score["other_absent"]
            blocks.append({
                "pair_id": pair_id, "image": image, "finding": finding, "layer": layer,
                "polarity_plane": .5 * (pc + po),
                "source_plane": .5 * (
                    score["current_present"] + score["current_absent"]
                    - score["other_present"] - score["other_absent"]
                ),
                "binding_plane": .5 * (pc - po),
                "current_transport": pc, "other_transport": po,
                "uncertain_source_difference": score["current_uncertain"] - score["other_uncertain"],
                "unrelated_minus_plain": score["random_unrelated_state"] - score["plain"],
            })
    return blocks


def analyze(blocks: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    layers = sorted({int(row["layer"]) for row in blocks})
    findings = sorted({str(row["finding"]) for row in blocks})
    final = max(layers)
    candidates = [layer for layer in layers if 0 < layer < final]
    if not candidates or len(findings) < 2:
        raise ValueError("need intermediate layers and at least two findings")
    by_key = {(row["pair_id"], row["layer"]): row for row in blocks}
    pairs = sorted({row["pair_id"] for row in blocks})

    per_layer = {}
    for layer in layers:
        current = [row for row in blocks if row["layer"] == layer]
        per_layer[str(layer)] = {
            field: cluster_summary(
                np.asarray([row[field] for row in current]), draws, seed + 1000 * layer + offset
            )
            for offset, field in enumerate((
                "polarity_plane", "source_plane", "binding_plane",
                "current_transport", "other_transport", "uncertain_source_difference",
                "unrelated_minus_plain",
            ))
        }

    selected_by_finding = {}
    oof_binding = []
    oof_pair_ids = []
    for held in findings:
        train = [row for row in blocks if row["finding"] != held]
        means = {
            layer: float(np.mean([row["binding_plane"] for row in train if row["layer"] == layer]))
            for layer in candidates
        }
        selected = max(candidates, key=lambda layer: (means[layer], -layer))
        test = [row for row in blocks if row["finding"] == held and row["layer"] == selected]
        values = np.asarray([row["binding_plane"] for row in test])
        selected_by_finding[held] = {
            "selected_layer": selected, "train_mean_binding": means[selected],
            "test_mean_binding": float(values.mean()), "test_n": len(test),
        }
        oof_binding.extend(values.tolist())
        oof_pair_ids.extend(row["pair_id"] for row in test)
    if sorted(oof_pair_ids) != pairs:
        raise RuntimeError("LOFO did not produce exactly one OOF value per image")
    oof = np.asarray([value for _, value in sorted(zip(oof_pair_ids, oof_binding))])
    final_values = np.asarray([by_key[(pair_id, final)]["binding_plane"] for pair_id in pairs])
    improvement = oof - final_values
    final_polarity = np.asarray([by_key[(pair_id, final)]["polarity_plane"] for pair_id in pairs])
    oof_summary = cluster_summary(oof, draws, seed + 7)
    improvement_summary = cluster_summary(improvement, draws, seed + 8)
    final_polarity_summary = cluster_summary(final_polarity, draws, seed + 9)
    positive_findings = sum(row["test_mean_binding"] > 0 for row in selected_by_finding.values())
    checks = {
        "final_polarity_persists": final_polarity_summary["image_cluster_bootstrap_ci95"][0] > 0,
        "intermediate_binding_positive_out_of_finding": oof_summary["image_cluster_bootstrap_ci95"][0] > 0,
        "binding_exceeds_final_out_of_finding": improvement_summary["image_cluster_bootstrap_ci95"][0] > 0,
        "binding_positive_in_at_least_three_findings": positive_findings >= 3,
    }
    return {
        "status": "PASS_BINDING_ERASURE_DISCOVERY" if all(checks.values()) else "FAIL_BINDING_ERASURE_DISCOVERY",
        "n_images": len(pairs), "findings": findings, "layers": layers, "final_layer": final,
        "planes": {
            "P_polarity": "0.5 * [(current present-current absent)+(other present-other absent)]",
            "S_source": "0.5 * [(current present+current absent)-(other present+other absent)]",
            "B_binding": "0.5 * [(current present-current absent)-(other present-other absent)]",
        },
        "selection": {
            "protocol": "leave-one-finding-out; maximize training-findings mean B; earliest-layer tie break",
            "by_held_finding": selected_by_finding,
        },
        "oof_binding": oof_summary,
        "oof_binding_minus_final": improvement_summary,
        "final_polarity": final_polarity_summary,
        "per_layer_descriptive": per_layer,
        "decision": {
            "checks": checks,
            "positive_findings": positive_findings,
            "scope": (
                "A pass establishes a replicated discovery candidate only. It does not prove "
                "causal transport, mitigation, natural-report validity, or confirmation replication."
            ),
            "next_if_pass": "bidirectional binding-only/path patch on discovery; keep confirmation sealed",
            "next_if_fail": "kill ownership-erasure; retain final-output source-blindness result",
        },
    }


def run_self_test() -> None:
    findings = ["a", "b", "c", "d"]
    blocks = []
    for index in range(32):
        for layer in (0, 1, 2):
            blocks.append({
                "pair_id": f"p{index}", "image": f"i{index}", "finding": findings[index % 4],
                "layer": layer, "polarity_plane": 2., "source_plane": 0.,
                "binding_plane": 1. if layer == 1 else 0., "current_transport": 2.,
                "other_transport": 2., "uncertain_source_difference": 0.,
                "unrelated_minus_plain": 0.,
            })
    result = analyze(blocks, 1000, 7)
    assert result["status"] == "PASS_BINDING_ERASURE_DISCOVERY"
    assert result["oof_binding"]["mean"] == 1.
    print(json.dumps({"status": "passed", "tests": 2, "gpu_used": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--layerwise", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.manifest or not args.layerwise or not args.output:
        parser.error("--manifest, --layerwise, and --output are required")
    if args.bootstrap_draws < 1000:
        raise ValueError("at least 1000 bootstrap draws required")
    manifest = load_rows(args.manifest)  # refuses confirmation
    rows = [json.loads(line) for line in args.layerwise.read_text().splitlines() if line.strip()]
    result = analyze(build_layer_blocks(manifest, rows), args.bootstrap_draws, args.seed)
    result.update({
        "version": VERSION, "analysis_split": "discovery_only",
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "layerwise": str(args.layerwise), "layerwise_sha256": sha256_file(args.layerwise),
        "bootstrap": {"unit": "image/pair_id", "draws": args.bootstrap_draws, "seed": args.seed},
    })
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
