"""Explore label-free visual-edge risk scores from an existing raw artifact.

This is a retrospective analysis only: labels and natural-pair JS are used
strictly for evaluation. Candidate risk scores themselves use one image's
Full/visual-blocked logits and selected state only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STATES = ("supported", "refuted", "undetermined")


def auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    # Mann–Whitney U with average ties, avoiding a sklearn dependency.
    ranks = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1
    for value in np.unique(s):
        idx = np.flatnonzero(s == value)
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def softmax(logits: dict[str, float]) -> np.ndarray:
    x = np.asarray([float(logits[s]) for s in STATES], dtype=float)
    x -= x.max()
    p = np.exp(x)
    return p / p.sum()


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_features(score: dict[str, Any]) -> dict[str, float]:
    full = {k: float(v) for k, v in score["full_logits"].items()}
    blocked = {k: float(v) for k, v in score["visual_blocked_logits"].items()}
    selected = str(score["selected_state"])
    alternative = str(score["selected_alternative"])

    def margin(logits: dict[str, float], state: str) -> float:
        return logits[state] - max(v for k, v in logits.items() if k != state)

    candidate_support = {
        state: margin(full, state) - margin(blocked, state) for state in STATES
    }
    selected_support = candidate_support[selected]
    full_margin = margin(full, selected)
    blocked_margin = margin(blocked, selected)
    abs_span = max(candidate_support.values()) - min(candidate_support.values())
    p_full = softmax(full)
    p_blocked = softmax(blocked)
    # Every value below is computable before knowing the answer.
    return {
        "selected_support": float(selected_support),
        "negative_selected_support": float(-selected_support),
        "negative_max_abs_candidate_support": float(-max(abs(v) for v in candidate_support.values())),
        "negative_candidate_support_span": float(-abs_span),
        "top1_switch": float(max(full, key=full.get) != max(blocked, key=blocked.get)),
        "blocked_margin_persistence": float(blocked_margin / (abs(full_margin) + 1e-6)),
        "negative_blocked_margin_persistence": float(-blocked_margin / (abs(full_margin) + 1e-6)),
        "full_entropy": entropy(p_full),
        "negative_visual_entropy_change": float(-(entropy(p_blocked) - entropy(p_full))),
        "selected_state": selected,
        "selected_alternative": alternative,
    }


def bootstrap_pair_auc(labels: list[int], scores: list[float], draws: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(auc(y[idx], s[idx]))
    if not values:
        return {"valid_draws": 0, "ci_low": float("nan"), "ci_high": float("nan")}
    q = np.quantile(values, [0.025, 0.975])
    return {"valid_draws": len(values), "ci_low": float(q[0]), "ci_high": float(q[1])}


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = Path(args.raw)
    source_path = Path(args.source_raw)
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    source = {}
    for line in source_path.read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            source[value["pair_id"]] = value
    images: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        pair_images = []
        expected = {"positive": "supported", "negative": "refuted"}
        for role in ("positive", "negative"):
            score = row["scores"][role]["canonical"]
            feat = image_features(score)
            feat.update({"pair_id": row["pair_id"], "role": role, "image": row[f"{role}_image"]})
            feat["error"] = int(feat["selected_state"] != expected[role])
            images.append(feat)
            pair_images.append(feat)
        source_row = source[row["pair_id"]]
        p = []
        for role in ("positive", "negative"):
            p.extend(source_row["scores"][role]["canonical"]["probabilities"].values())
        pos = np.asarray(list(source_row["scores"]["positive"]["canonical"]["probabilities"].values()), dtype=float)
        neg = np.asarray(list(source_row["scores"]["negative"]["canonical"]["probabilities"].values()), dtype=float)
        pair = {
            "pair_id": row["pair_id"],
            "any_error": int(any(item["error"] for item in pair_images)),
            "oracle_between_image_js": float(0.5 * ((pos * np.log(np.maximum(pos / np.maximum(neg, 1e-12), 1e-12))).sum() + (neg * np.log(np.maximum(neg / np.maximum(pos, 1e-12), 1e-12))).sum())),
        }
        for name in (
            "negative_selected_support",
            "negative_max_abs_candidate_support",
            "negative_candidate_support_span",
            "top1_switch",
            "negative_blocked_margin_persistence",
            "full_entropy",
            "negative_visual_entropy_change",
        ):
            pair[name] = float(np.mean([item[name] for item in pair_images]))
        pairs.append(pair)

    labels_pair = [row["any_error"] for row in pairs]
    labels_image = [row["error"] for row in images]
    names = [
        "negative_selected_support",
        "negative_max_abs_candidate_support",
        "negative_candidate_support_span",
        "top1_switch",
        "negative_blocked_margin_persistence",
        "full_entropy",
        "negative_visual_entropy_change",
    ]
    pair_auc = {name: auc(labels_pair, [row[name] for row in pairs]) for name in names}
    image_auc = {name: auc(labels_image, [row[name] for row in images]) for name in names}
    bootstrap = {name: bootstrap_pair_auc(labels_pair, [row[name] for row in pairs], args.bootstrap_draws, args.seed) for name in names}
    return {
        "protocol": "vqa-rad-visual-edge-label-free-analysis-v1",
        "raw": str(raw_path.resolve()),
        "raw_sha256": sha256(raw_path),
        "source_raw": str(source_path.resolve()),
        "source_raw_sha256": sha256(source_path),
        "command": " ".join(args.command),
        "n_pairs": len(pairs),
        "n_images": len(images),
        "metrics": {"pair_error_auroc": pair_auc, "image_error_auroc": image_auc, "pair_error_bootstrap": bootstrap},
        "pairs": pairs,
        "images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--source-raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260815)
    args = parser.parse_args()
    args.command = ["python", str(Path(__file__).resolve()), *sum(([k, str(v)] for k, v in vars(args).items() if k != "command"), [])]
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
