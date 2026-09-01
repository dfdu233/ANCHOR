#!/usr/bin/env python3
"""Fit a tiny layer-logit calibrator with an optional selectivity objective.

The probe is intentionally small: it learns one scalar weight per inspected
decoder layer and one bias.  It is not a VLM fine-tune and cannot by itself
support a mitigation claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


VERSION = "selectivity-calibrator-probe-v1"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def auc(labels: list[int], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return None
    return sum(
        float(pos > neg) + 0.5 * float(pos == neg)
        for pos in positive
        for neg in negative
    ) / (len(positive) * len(negative))


def metrics(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    predictions = [int(score > 0.0) for score in scores]
    positive = sum(labels)
    negative = len(labels) - positive
    return {
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "balanced_accuracy": 0.5
        * (
            sum(a == b == 1 for a, b in zip(labels, predictions)) / positive
            + sum(a == b == 0 for a, b in zip(labels, predictions)) / negative
        ),
        "auroc": auc(labels, scores),
        "fabrication_rate": sum(a == 0 and b == 1 for a, b in zip(labels, predictions))
        / negative,
        "omission_rate": sum(a == 1 and b == 0 for a, b in zip(labels, predictions))
        / positive,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--selectivity-weight", type=float, default=0.25)
    parser.add_argument("--selectivity-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    contracts = {str(row["image_id"]): row for row in load_jsonl(args.manifest)}
    raw = [row for row in load_jsonl(args.raw) if row.get("status") == "ok"]
    if not raw:
        raise ValueError("no successful raw records")
    layers = sorted(int(key) for key in raw[0]["measurement"]["trajectory"])

    items: dict[tuple[str, str], tuple[torch.Tensor, int, str]] = {}
    roles: dict[tuple[str, str], tuple[str, str]] = {}
    for row in raw:
        contract = contracts[str(row["image_id"])]
        features = []
        for layer in layers:
            logits = row["measurement"]["trajectory"][str(layer)]["real_logits"]
            features.append(0.5 * (logits["supported"] - logits["refuted"]))
        key = (str(contract["finding"]), str(contract["image_path"]))
        value = (
            torch.tensor(features, dtype=torch.float32),
            int(contract["positive_votes"]) == 3,
            str(contract["experiment_split"]),
        )
        if key in items:
            previous = items[key]
            if previous[1:] != value[1:] or not torch.equal(previous[0], value[0]):
                raise ValueError(f"inconsistent duplicate image/claim: {key}")
        items[key] = value
        roles[(str(contract["triplet_id"]), str(contract["swap_role"]))] = key

    train_keys = sorted(key for key, value in items.items() if value[2] == "dev")
    test_keys = sorted(key for key, value in items.items() if value[2] == "test")
    if not train_keys or not test_keys:
        raise ValueError("both dev and test items are required")
    train_index = {key: index for index, key in enumerate(train_keys)}
    train_x = torch.stack([items[key][0] for key in train_keys])
    train_y = torch.tensor([items[key][1] for key in train_keys], dtype=torch.float32)
    test_x = torch.stack([items[key][0] for key in test_keys])
    test_y = [int(items[key][1]) for key in test_keys]

    triplets = []
    for triplet_id, role in sorted(roles):
        if role != "anchor":
            continue
        keys = [
            roles.get((triplet_id, member))
            for member in ("anchor", "same_state_swap", "opposite_state_swap")
        ]
        if any(key is None or key not in train_index for key in keys):
            continue
        anchor, same, opposite = (train_index[key] for key in keys)
        sign = 1.0 if items[keys[0]][1] == 1 else -1.0
        triplets.append((anchor, same, opposite, sign))

    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0).clamp_min(0.1)

    def fit(name: str, final_only: bool, selective: bool) -> dict[str, object]:
        torch.manual_seed(args.seed)
        normalized = (train_x - feature_mean) / feature_std
        if final_only:
            normalized = normalized[:, -1:]
        weights = torch.zeros(normalized.shape[1], requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias], lr=args.learning_rate)
        for _ in range(args.steps):
            logits = normalized @ weights + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y)
            loss = loss + args.l2 * weights.square().sum()
            if selective:
                if not triplets:
                    raise ValueError("selectivity fit requires complete dev triplets")
                pair_losses = []
                for anchor, same, opposite, sign in triplets:
                    invariance = (logits[anchor] - logits[same]).square()
                    sensitivity = torch.relu(
                        args.selectivity_margin
                        - sign * (logits[anchor] - logits[opposite])
                    ).square()
                    pair_losses.append(invariance + sensitivity)
                loss = loss + args.selectivity_weight * torch.stack(pair_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        normalized_test = (test_x - feature_mean) / feature_std
        if final_only:
            normalized_test = normalized_test[:, -1:]
        scores = (normalized_test @ weights.detach() + bias.detach()).tolist()
        return {
            "name": name,
            "weights": weights.detach().tolist(),
            "bias": float(bias.detach()),
            "metrics": metrics(test_y, scores),
        }

    result = {
        "version": VERSION,
        "evidence_grade": "C",
        "formal_reference": False,
        "layers": layers,
        "train_unique_claim_images": len(train_keys),
        "test_unique_claim_images": len(test_keys),
        "train_complete_triplets": len(triplets),
        "hyperparameters_frozen_without_test_tuning": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "selectivity_weight": args.selectivity_weight,
            "selectivity_margin": args.selectivity_margin,
            "seed": args.seed,
        },
        "methods": {
            "raw_final": {
                "metrics": metrics(test_y, test_x[:, -1].tolist()),
            },
            "calibrated_final": fit("calibrated_final", True, False),
            "supervised_layer_mixer": fit("supervised_layer_mixer", False, False),
            "selectivity_layer_mixer": fit("selectivity_layer_mixer", False, True),
        },
        "claim_ceiling": (
            "This probe can show feasibility of a scalar layer calibrator only; "
            "a selectivity-specific mitigation claim requires improvement over "
            "both calibrated-final and supervised-layer controls on formal data."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
