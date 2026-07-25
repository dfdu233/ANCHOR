"""Summarize native-view projection probes and candidate-level bad cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def softmax(logits):
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


def js_divergence(probabilities):
    probs = np.asarray(probabilities, dtype=np.float64)
    mean = probs.mean(axis=0)
    return float((probs * np.log(np.clip(probs, 1e-12, None) / np.clip(mean, 1e-12, None))).sum(axis=1).mean())


def summarize(path: Path):
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    summary = payload.get("summary", {})
    print(f"\n## {path}")
    print(json.dumps({k: summary.get(k) for k in ["version", "n", "support_n", "grid"]}, indent=2, ensure_ascii=False))
    for split in ["overall", "train", "test"]:
        block = summary.get(split, {})
        if not block:
            continue
        methods = [
            "original",
            "entropy_best",
            "margin_best",
            "nll_best",
            "entropy_fusion",
            "margin_fusion",
            "nll_fusion",
            "laplacian_fusion",
        ]
        print(f"{split}: n={block.get('n')} accepted={block.get('accepted_rate')} mean_accepted={block.get('mean_accepted')}")
        for method in methods:
            key = f"{method}_accuracy"
            if key in block:
                rescue = block.get(f"rescues_{method}", "-")
                harm = block.get(f"harmful_{method}", "-")
                print(f"  {method:18s} acc={block[key]} rescue={rescue} harmful={harm}")

    candidate_flips = 0
    candidate_rescues = []
    candidate_harmful = []
    deltas_by_family = {}
    js_values = []
    for row in rows:
        if not row.get("candidates"):
            continue
        gt = int(row["gt_index"])
        base = row["candidates"][0]
        base_logits = np.asarray(base["logits"], dtype=np.float64)
        base_pred = int(base["logit_prediction"])
        base_correct = base_pred == gt
        probs = [softmax(c["logits"]) for c in row["candidates"]]
        js_values.append(js_divergence(probs))
        for candidate in row["candidates"][1:]:
            logits = np.asarray(candidate["logits"], dtype=np.float64)
            delta = float(np.linalg.norm(logits - base_logits))
            deltas_by_family.setdefault(candidate["family"], []).append(delta)
            pred = int(candidate["logit_prediction"])
            if pred != base_pred:
                candidate_flips += 1
            if (not base_correct) and pred == gt:
                candidate_rescues.append((row["qid"], candidate["name"], candidate["family"]))
            if base_correct and pred != gt:
                candidate_harmful.append((row["qid"], candidate["name"], candidate["family"]))
    print(f"candidate_flips={candidate_flips} candidate_rescues={len(candidate_rescues)} candidate_harmful={len(candidate_harmful)}")
    if js_values:
        print(f"js_mean={float(np.mean(js_values)):.6g} js_max={float(np.max(js_values)):.6g}")
    for family, values in sorted(deltas_by_family.items()):
        if values:
            print(f"delta[{family}] n={len(values)} mean={float(np.mean(values)):.6g} max={float(np.max(values)):.6g}")
    if candidate_rescues:
        print("rescues:", candidate_rescues[:20])
    if candidate_harmful:
        print("harmful:", candidate_harmful[:20])


def main():
    for path in parse_args().paths:
        summarize(path)


if __name__ == "__main__":
    main()
