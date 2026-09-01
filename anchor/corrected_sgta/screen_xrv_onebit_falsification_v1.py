#!/usr/bin/env python3
"""CPU-only fatal screen for specialist *negative* evidence.

The specialist is not allowed to contribute a positive score, rank candidates,
or fuse a posterior with the VLM.  It emits only a bit saying that a claim is
strongly contradicted.  This script asks whether such a bit is sufficiently
one-sided to be useful: it should fire on VLM false positives while almost
never firing on VLM true positives, and it should outperform a VLM-margin-only
bit at the same development-set safety quantile.

This is a necessary-condition screen, not a mitigation result.  It uses cached
XRV logits and never touches a GPU or the baseline queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from anchor.corrected_sgta.screen_external_visual_increment_v1 import (
    FINDINGS,
    final_margin,
    read_jsonl,
    sha256_file,
)
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import (
    FINDING_TARGETS,
    XRV_LABELS,
)


VERSION = "xrv-onebit-falsification-fatal-v1"


def stable_image_split(image_id: str) -> str:
    value = int(hashlib.sha256(f"xrv-onebit-v1:{image_id}".encode()).hexdigest()[:8], 16)
    return "development" if value % 10 < 3 else "confirmation"


def load_logits(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    labels = [str(value) for value in payload["labels"]]
    if labels != list(XRV_LABELS):
        raise ValueError("XRV label order drift")
    return {
        str(image_id): np.asarray(logit, dtype=np.float64)
        for image_id, logit in zip(payload["image_ids"], payload["logits"])
    }


def load_rows(path: Path, model: str, logits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    target_index = {name: index for index, name in enumerate(XRV_LABELS)}
    rows = []
    for row in read_jsonl(path):
        finding = str(row["finding"])
        votes = int(row["positive_votes"])
        image_id = str(row["image_id"])
        if finding not in FINDINGS or votes not in (0, 3) or image_id not in logits:
            continue
        expert = max(logits[image_id][target_index[name]] for name in FINDING_TARGETS[finding])
        rows.append(
            {
                "image_id": image_id,
                "finding": finding,
                "label": int(votes == 3),
                "margin": float(final_margin(row)),
                "expert": float(expert),
                "model": model,
            }
        )
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = {}
    for row in rows:
        key = (row["image_id"], row["finding"])
        previous = found.get(key)
        if previous is not None and previous["label"] != row["label"]:
            raise ValueError(f"label disagreement for {key}")
        found[key] = row
    return list(found.values())


def lower_quantile_thresholds(
    rows: list[dict[str, Any]], key: str, quantile: float
) -> dict[str, float]:
    thresholds = {}
    for finding in FINDINGS:
        values = [row[key] for row in rows if row["finding"] == finding and row["label"] == 1]
        if not values:
            raise ValueError(f"no positive development examples for {finding}")
        # The strict '< threshold' rule makes the empirical lower-tail rate no
        # larger than the selected order statistic.
        rank = int(np.floor(quantile * len(values)))
        rank = min(max(rank, 0), len(values) - 1)
        thresholds[finding] = float(np.sort(np.asarray(values))[rank])
    return thresholds


def row_outcomes(
    rows: list[dict[str, Any]], expert_thresholds: dict[str, float], margin_thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    outcomes = []
    for row in rows:
        # The target action only exists for positive claims already emitted by
        # the VLM.  A negative margin is not turned into a new positive.
        if row["margin"] <= 0:
            continue
        outcomes.append(
            {
                **row,
                "expert_veto": int(row["expert"] < expert_thresholds[row["finding"]]),
                "margin_veto": int(row["margin"] < margin_thresholds[row["finding"]]),
                "native_zero_veto": int(row["expert"] < 0.0),
            }
        )
    return outcomes


def summarize_bit(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tp = [row for row in rows if row["label"] == 1]
    fp = [row for row in rows if row["label"] == 0]
    tp_fire = sum(row[key] for row in tp)
    fp_fire = sum(row[key] for row in fp)
    total_fire = tp_fire + fp_fire
    return {
        "draft_positive_claims": len(rows),
        "tp": len(tp),
        "fp": len(fp),
        "triggered": total_fire,
        "triggered_tp": tp_fire,
        "triggered_fp": fp_fire,
        "tp_veto_rate": tp_fire / max(len(tp), 1),
        "fp_veto_rate": fp_fire / max(len(fp), 1),
        "veto_precision_for_fp": fp_fire / max(total_fire, 1),
        "net_binary_errors_removed": fp_fire - tp_fire,
        "relative_fp_reduction_if_deleted": fp_fire / max(len(fp), 1),
    }


def bootstrap(
    rows: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        sample_ids = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample = [row for image_id in sample_ids for row in groups[image_id]]
        expert = summarize_bit(sample, "expert_veto")
        margin = summarize_bit(sample, "margin_veto")
        if expert["fp"] and expert["tp"]:
            values["expert_fp_veto_rate"].append(expert["fp_veto_rate"])
            values["expert_tp_veto_rate"].append(expert["tp_veto_rate"])
            values["expert_minus_margin_fp_veto_rate"].append(
                expert["fp_veto_rate"] - margin["fp_veto_rate"]
            )
            values["expert_net_error_rate"].append(
                expert["net_binary_errors_removed"] / len(sample)
            )

    return {
        key: {
            "mean": float(np.mean(current)),
            "ci95": [float(np.quantile(current, 0.025)), float(np.quantile(current, 0.975))],
        }
        for key, current in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-quantile", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    logits = load_logits(args.xrv_logits)
    sources = {
        "huatuo": (args.huatuo_dev, args.huatuo_confirmation),
        "hulu": (args.hulu_dev, args.hulu_confirmation),
    }
    loaded = {
        model: {
            "development": load_rows(dev, model, logits),
            "confirmation": load_rows(confirmation, model, logits),
        }
        for model, (dev, confirmation) in sources.items()
    }

    # Expert thresholds must be model-independent.  Both VLM files contain the
    # same VinDr labels, so deduplicate their union before setting the bit.
    expert_development = deduplicate(
        loaded["huatuo"]["development"] + loaded["hulu"]["development"]
    )
    expert_thresholds = lower_quantile_thresholds(
        expert_development, "expert", args.safety_quantile
    )

    analyses = {}
    passes = []
    for model, splits in loaded.items():
        margin_thresholds = lower_quantile_thresholds(
            splits["development"], "margin", args.safety_quantile
        )
        confirmation = row_outcomes(
            splits["confirmation"], expert_thresholds, margin_thresholds
        )
        boot = bootstrap(confirmation, args.bootstrap_draws, args.seed)
        expert = summarize_bit(confirmation, "expert_veto")
        analyses[model] = {
            "development_rows": len(splits["development"]),
            "confirmation_rows": len(splits["confirmation"]),
            "confirmation_positive_drafts": len(confirmation),
            "expert_safe_bit": expert,
            "margin_matched_bit": summarize_bit(confirmation, "margin_veto"),
            "expert_native_zero_bit": summarize_bit(confirmation, "native_zero_veto"),
            "margin_thresholds": margin_thresholds,
            "image_cluster_bootstrap": boot,
        }
        passes.append(
            expert["fp_veto_rate"] >= 0.20
            and expert["tp_veto_rate"] <= 0.02
            and boot["expert_fp_veto_rate"]["ci95"][0] > 0.10
            and boot["expert_tp_veto_rate"]["ci95"][1] < 0.05
            and boot["expert_minus_margin_fp_veto_rate"]["ci95"][0] > 0.0
            and boot["expert_net_error_rate"]["ci95"][0] > 0.0
        )

    config = {
        "version": VERSION,
        "safety_quantile": args.safety_quantile,
        "expert_bit": "1[xrv_finding_logit < finding-specific lower positive-tail threshold]",
        "expert_threshold_scope": "model-independent; VinDr development positives only",
        "action_scope": "only VLM claims with final supported-minus-refuted margin > 0",
        "control": "VLM-margin lower-tail bit at the same development safety quantile",
        "sources": {
            model: {
                "development": {"path": str(dev), "sha256": sha256_file(dev)},
                "confirmation": {"path": str(conf), "sha256": sha256_file(conf)},
            }
            for model, (dev, conf) in sources.items()
        },
        "xrv_logits": str(args.xrv_logits),
        "xrv_logits_sha256": sha256_file(args.xrv_logits),
        "expert_thresholds": expert_thresholds,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }
    result = {
        "status": "complete_cpu_onebit_necessary_condition",
        "decision": "PASS" if all(passes) else "NO_GO",
        "decision_rule": (
            "For both VLMs on confirmation: the model-independent expert negative bit must veto "
            ">=20% of VLM false-positive claims and <=2% of true-positive claims; image-bootstrap "
            "95% CI must have FP-veto lower bound >10%, TP-veto upper bound <5%, positive net error "
            "removal, and positive FP-veto advantage over a VLM-margin bit with the same dev safety quantile."
        ),
        "claim_boundary": (
            "Deletion numbers are an optimistic necessary condition only. The screen does not implement "
            "fixed-content replacement and cannot establish an OE/report mitigation result."
        ),
        "config": config,
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
