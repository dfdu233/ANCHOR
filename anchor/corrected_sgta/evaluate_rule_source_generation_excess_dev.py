#!/usr/bin/env python3
"""Evaluate generation-aligned Source-DRO on the consumed development split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.build_rule_source_manifest import canonical_rgb_sha256, sha256_bytes
from corrected_sgta.evaluate_medheval_answers import parse_answer, rule_pope_prediction
from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.evaluate_rule_source_preference_barycenter import _module_from_state, _predict, summarize_predictions
from corrected_sgta.infer_rule_dg_adapter import decode
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_generation_excess_dro import VERSION as TRAIN_VERSION
from corrected_sgta.rule_source_preference import LinearLowRankResidual
from corrected_sgta.rule_source_preference import canonical_binary_answer, file_sha256, rule_mimic_prompt, stable_json_sha256
from corrected_sgta.train_rule_dg_adapter import canonical_answer

VERSION = "rule-source-generation-excess-development-eval-v1"
DOMAINS = ("rule_iuxray", "slake_xray", "vqa_rad_train")


def mcnemar_exact_p(rescues: int, harms: int) -> float:
    n = rescues + harms
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(rescues, harms) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(records: dict[str, list[dict[str, Any]]], variant: str, seed: int = 20260725) -> list[float]:
    delta = []
    for rows in records.values():
        for row in rows:
            gt = row["ground_truth"]
            delta.append(int(row["predictions"][variant] == gt) - int(row["predictions"]["identity"] == gt))
    values = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(10000)])
    return [float(x * 100.0) for x in np.quantile(means, [0.025, 0.975])]


def confirmation_gate(summary: dict[str, Any], n_required: int) -> dict[str, Any]:
    primary = summary["source_dro"]
    micro = primary["micro"]
    rescues, harms = int(micro["rescues"]), int(micro["harms"])
    p_value = mcnemar_exact_p(rescues, harms)
    per_domain = primary["per_domain"]
    checks = {
        "complete": int(micro["n"]) == n_required,
        "delta_at_least_3pp": float(micro["delta_pp"]) >= 3.0,
        "mcnemar_p_below_005": p_value < 0.05,
        "rescue_harm_ratio": rescues >= max(1, 2 * harms),
        "two_domains_nondeclining": sum(float(v["delta_pp"]) >= 0 for v in per_domain.values()) >= 2,
        "no_domain_worse_by_more_than_one": all(int(v["harms"]) - int(v["rescues"]) <= 1 for v in per_domain.values()),
    }
    passed = all(checks.values())
    return {"status": "passed" if passed else "failed", "target_pilot_allowed": passed, "primary_interface": "actual_greedy_rule_pope", "mcnemar_exact_p": p_value, "checks": checks}


def verify_confirm_manifest(base: dict[str, Any], confirm: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if confirm.get("version") != "rule-source-confirm-manifest-v1":
        raise ValueError("unsupported confirm manifest")
    if confirm.get("base_manifest_fingerprint") != base.get("fingerprint"):
        raise ValueError("confirm/base fingerprint mismatch")
    if checkpoint.get("manifest_contract", {}).get("manifest_fingerprint") != base.get("fingerprint"):
        raise ValueError("checkpoint/base fingerprint mismatch")
    if confirm.get("target_file_opened") is not False or checkpoint.get("target_labels_accessed") is not False:
        raise ValueError("target seal is absent")
    if int(confirm.get("images_per_domain", -1)) != 48:
        raise ValueError("confirmation protocol requires 48 images per domain")
    base_hashes = set()
    for split in ("train", "dev"):
        for row in json.loads(Path(base["outputs"][split]["json"]).read_text()):
            base_hashes.add(str(row["image_sha256"]))
    selected = {}
    for domain in DOMAINS:
        rows = json.loads(Path(confirm["outputs"][domain]["json"]).read_text())
        if len(rows) != 48:
            raise ValueError(f"domain {domain} does not contain 48 rows")
        hashes = [str(row["image_sha256"]) for row in rows]
        if len(set(hashes)) != 48 or set(hashes) & base_hashes:
            raise ValueError(f"domain {domain} is not independent")
        for row in rows:
            if canonical_rgb_sha256(Path(row["image"])) != row["image_sha256"]:
                raise ValueError(f"image hash mismatch: {row['image']}")
        selected[domain] = rows
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--confirm-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images-per-domain", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.max_images_per_domain <= 48:
        raise ValueError("max-images-per-domain must be in [1,48]")
    base = json.loads(args.base_manifest.read_text())
    confirm = json.loads(args.confirm_manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != TRAIN_VERSION:
        raise ValueError("unsupported checkpoint version")
    selected = verify_confirm_manifest(base, confirm, checkpoint)
    selected = {domain: rows[: args.max_images_per_domain] for domain, rows in selected.items()}

    constrained = {domain: [] for domain in DOMAINS}
    greedy = {domain: [] for domain in DOMAINS}
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    module = LinearLowRankResidual(
        int(checkpoint["width"]), int(checkpoint["rank"]),
        float(checkpoint["max_relative_update"]),
    ).to(adapter.model.device)
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    try:
        with torch.no_grad():
            for domain in DOMAINS:
                for row in tqdm(selected[domain], desc=f"source-confirm:{domain}"):
                    with Image.open(row["image"]) as handle:
                        image = handle.convert("RGB")
                    question = str(row["conversations"][0]["value"]).replace("<image>", "").strip()
                    prompt = rule_mimic_prompt(question)
                    gt = canonical_binary_answer(canonical_answer(row["conversations"][1]["value"]))
                    identity, identity_scores = _predict(adapter, image, prompt, None)
                    adapted, adapted_scores = _predict(adapter, image, prompt, module)
                    identity_text = decode(adapter, image, prompt, args.max_new_tokens, None, "post")
                    adapted_text = decode(adapter, image, prompt, args.max_new_tokens, module, "post")
                    constrained[domain].append({"id": row["id"], "image": row["image"], "ground_truth": gt, "predictions": {"identity": identity, "source_dro": adapted}, "sequence_log_probabilities": {"identity": identity_scores, "source_dro": adapted_scores}})
                    greedy[domain].append({"id": row["id"], "image": row["image"], "ground_truth": gt, "predictions": {"identity": canonical_binary_answer(rule_pope_prediction(identity_text)), "source_dro": canonical_binary_answer(rule_pope_prediction(adapted_text))}, "raw_text": {"identity": identity_text, "source_dro": adapted_text}, "strict_parse": {"identity": parse_answer(identity_text, answer_type="binary").status, "source_dro": parse_answer(adapted_text, answer_type="binary").status}})
    finally:
        adapter.close()
    constrained_summary = summarize_predictions(constrained, ["source_dro"])
    greedy_summary = summarize_predictions(greedy, ["source_dro"])
    n_required = 48 * len(DOMAINS)
    diagnostic_gate = confirmation_gate(greedy_summary, n_required) if args.max_images_per_domain == 48 else {"status": "smoke_only"}
    gate = {
        "status": "development_only",
        "target_pilot_allowed": False,
        "would_pass_numeric_screen": diagnostic_gate.get("status") == "passed",
        "diagnostic": diagnostic_gate,
    }
    payload = {
        "version": VERSION,
        "fingerprint": stable_json_sha256({"version": VERSION, "base_manifest_sha256": file_sha256(args.base_manifest), "confirm_manifest_sha256": file_sha256(args.confirm_manifest), "checkpoint_sha256": file_sha256(args.checkpoint), "max_images_per_domain": args.max_images_per_domain, "max_new_tokens": args.max_new_tokens}),
        "target_labels_used": False,
        "interfaces": {"constrained_complete_sequence": constrained_summary, "actual_greedy_rule_pope": greedy_summary},
        "statistics": {"constrained_ci95_pp": paired_bootstrap_ci(constrained, "source_dro"), "greedy_ci95_pp": paired_bootstrap_ci(greedy, "source_dro")},
        "confirmation_gate": gate,
        "records": {"constrained": constrained, "greedy": greedy},
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("fingerprint", "interfaces", "statistics", "confirmation_gate")}, indent=2))


if __name__ == "__main__":
    main()
