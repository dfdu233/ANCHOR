#!/usr/bin/env python3
"""Target-blind analysis of the controlled Source x Polarity discovery gate.

The analysis intentionally consumes no reader votes, target answers, or clinical
labels.  Each image is one cluster and contributes a complete eight-arm block.
Confirmation data must be analyzed separately after the discovery decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


VERSION = "source-ownership-factorial-analysis-v1"
ARMS = {
    "current_present", "current_absent", "other_present", "other_absent",
    "current_uncertain", "other_uncertain", "plain", "random_unrelated_state",
}
FORBIDDEN = {
    "answer", "answers", "groundtruth", "gtanswer", "label", "labels", "target",
    "targets", "reference", "gold", "readervotes", "readerlabels", "positivevotes",
    "readersupport", "votebin",
}


def normalized_key(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def reject_targets(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if normalized_key(key) in FORBIDDEN:
                raise ValueError(f"forbidden target field at {path}.{key}")
            reject_targets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_targets(child, f"{path}[{index}]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected nonempty row list: {path}")
    reject_targets(rows)
    return rows


def build_blocks(
    manifest: list[dict[str, Any]], margins: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    manifest_by_qid = {str(row["qid"]): row for row in manifest}
    margin_by_qid = {str(row["qid"]): row for row in margins}
    if len(manifest_by_qid) != len(manifest) or len(margin_by_qid) != len(margins):
        raise ValueError("duplicate qid")
    if set(manifest_by_qid) != set(margin_by_qid):
        raise ValueError("manifest and margin qids differ")
    grouped: dict[str, list[str]] = {}
    for qid, row in manifest_by_qid.items():
        grouped.setdefault(str(row["pair_id"]), []).append(qid)
    blocks = []
    for pair_id, qids in grouped.items():
        arms = {str(manifest_by_qid[qid]["arm"]): qid for qid in qids}
        if set(arms) != ARMS or len(qids) != len(ARMS):
            raise ValueError(f"incomplete eight-arm block: {pair_id}")
        findings = {str(manifest_by_qid[qid]["finding"]) for qid in qids}
        images = {str(manifest_by_qid[qid]["img_name"]) for qid in qids}
        if len(findings) != 1 or len(images) != 1:
            raise ValueError(f"block metadata drift: {pair_id}")
        score = {
            arm: float(margin_by_qid[qid]["polarity_yes_minus_no"])
            for arm, qid in arms.items()
        }
        if not all(math.isfinite(value) for value in score.values()):
            raise ValueError(f"nonfinite margin: {pair_id}")
        pc = score["current_present"] - score["current_absent"]
        po = score["other_present"] - score["other_absent"]
        blocks.append({
            "pair_id": pair_id,
            "image": next(iter(images)),
            "finding": next(iter(findings)),
            "current_polarity_transport": pc,
            "other_polarity_transport": po,
            "binding_interaction_half": 0.5 * (pc - po),
            "uncertain_source_difference": (
                score["current_uncertain"] - score["other_uncertain"]
            ),
            "unrelated_minus_plain": score["random_unrelated_state"] - score["plain"],
        })
    if len({row["image"] for row in blocks}) != len(blocks):
        raise ValueError("an image occurs in more than one independent block")
    return blocks


def summarize_vector(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("empty estimand")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "n_images": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive_fraction": float((values > 0).mean()),
        "image_cluster_bootstrap_ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
        "valid_draws": draws,
    }


def summarize(blocks: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    fields: dict[str, Callable[[dict[str, Any]], float]] = {
        "current_polarity_transport": lambda row: row["current_polarity_transport"],
        "other_polarity_transport": lambda row: row["other_polarity_transport"],
        "binding_interaction_half": lambda row: row["binding_interaction_half"],
        "uncertain_source_difference": lambda row: row["uncertain_source_difference"],
        "unrelated_minus_plain": lambda row: row["unrelated_minus_plain"],
    }
    result = {}
    for offset, (name, getter) in enumerate(fields.items()):
        values = np.asarray([getter(row) for row in blocks], dtype=np.float64)
        result[name] = summarize_vector(values, draws, seed + offset)
    pc = result["current_polarity_transport"]["mean"]
    po = result["other_polarity_transport"]["mean"]
    control = result["unrelated_minus_plain"]["mean"]
    result["derived"] = {
        "other_over_current_transport": None if abs(pc) < 1e-12 else float(po / pc),
        "absolute_unrelated_over_other_transport": (
            None if abs(po) < 1e-12 else float(abs(control) / abs(po))
        ),
    }
    return result


def decision(overall: dict[str, Any], by_finding: dict[str, Any]) -> dict[str, Any]:
    pc = overall["current_polarity_transport"]
    po = overall["other_polarity_transport"]
    ratio = overall["derived"]["other_over_current_transport"]
    specificity = overall["derived"]["absolute_unrelated_over_other_transport"]
    positive_findings = sum(
        values["other_polarity_transport"]["mean"] > 0 for values in by_finding.values()
    )
    checks = {
        "current_manipulation_detectable": pc["image_cluster_bootstrap_ci95"][0] > 0,
        "other_patient_state_transport_detectable": po["image_cluster_bootstrap_ci95"][0] > 0,
        "other_transport_at_least_quarter_of_current": ratio is not None and ratio >= 0.25,
        "other_transport_not_generic_unrelated_prompt_effect": (
            specificity is not None and specificity <= 0.50
        ),
        "other_transport_positive_in_at_least_three_findings": positive_findings >= 3,
    }
    return {
        "status": "PASS_BEHAVIORAL_DISCOVERY" if all(checks.values()) else "FAIL_BEHAVIORAL_DISCOVERY",
        "checks": checks,
        "positive_findings": positive_findings,
        "important_scope": (
            "A pass only opens layer/path tests. It does not establish ownership erasure, "
            "a clinical hallucination rate, mitigation efficacy, or novelty over Ghost Context."
        ),
        "kill_rule": (
            "The paper candidate requires this gate in both Huatuo and Hulu. Failure in either "
            "model, or transport confined to one finding, kills the claimed general mechanism."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--margins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000:
        raise ValueError("at least 1000 bootstrap draws required")
    manifest = load_json_or_jsonl(args.manifest)
    margins = load_json_or_jsonl(args.margins)
    blocks = build_blocks(manifest, margins)
    overall = summarize(blocks, args.bootstrap_draws, args.seed)
    findings = sorted({row["finding"] for row in blocks})
    by_finding = {
        finding: summarize(
            [row for row in blocks if row["finding"] == finding],
            args.bootstrap_draws,
            args.seed + 100 * (index + 1),
        )
        for index, finding in enumerate(findings)
    }
    payload = {
        "version": VERSION,
        "analysis_is_target_blind": True,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "margins": str(args.margins),
        "margins_sha256": sha256_file(args.margins),
        "bootstrap": {"unit": "image/pair_id", "draws": args.bootstrap_draws, "seed": args.seed},
        "n_images": len(blocks),
        "findings": findings,
        "overall": overall,
        "by_finding": by_finding,
        "decision": decision(overall, by_finding),
    }
    reject_targets(payload)
    atomic_json(args.output, payload)
    print(json.dumps({"n_images": len(blocks), "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
