#!/usr/bin/env python3
"""Fit and audit a tiny reader-referenced clinical response aligner.

The aligner learns one scalar per inspected decoder layer and one bias.  Its
purpose is to test whether ordered reader support supplies information beyond
ordinary calibration and unsigned image sensitivity; it is not a VLM
fine-tune.  Every method is fit with the same optimizer and dev images.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from scipy.stats import spearmanr


VERSION = "clinical-response-aligner-v2"
ROLES = ("anchor", "same_state_swap", "opposite_state_swap")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def probability_metrics(
    positive_votes: list[int], reader_counts: list[int], scores: list[float]
) -> dict[str, float | None]:
    if not positive_votes or not (
        len(positive_votes) == len(reader_counts) == len(scores)
    ):
        raise ValueError("vote, reader-count, and score lists must be non-empty")
    supports = [votes / readers for votes, readers in zip(positive_votes, reader_counts)]
    probabilities = [1.0 / (1.0 + math.exp(-score)) for score in scores]
    states = [
        "refuted" if probability < 1 / 3 else
        "supported" if probability > 2 / 3 else
        "undetermined"
        for probability in probabilities
    ]
    references = [
        "refuted" if votes == 0 else
        "supported" if votes == readers else
        "undetermined"
        for votes, readers in zip(positive_votes, reader_counts)
    ]
    clear_labels = [
        int(votes == readers)
        for votes, readers in zip(positive_votes, reader_counts)
        if votes in {0, readers}
    ]
    clear_scores = [
        score
        for votes, readers, score in zip(positive_votes, reader_counts, scores)
        if votes in {0, readers}
    ]
    eps = 1e-7
    nll = mean(
        -support * math.log(max(eps, min(1 - eps, probability)))
        -(1 - support) * math.log(max(eps, min(1 - eps, 1 - probability)))
        for support, probability in zip(supports, probabilities)
    )
    relation = spearmanr(supports, scores)
    rho = float(relation.statistic)
    if not math.isfinite(rho):
        rho = None
    negatives = sum(votes == 0 for votes in positive_votes)
    positives = sum(votes == readers for votes, readers in zip(positive_votes, reader_counts))
    disagreements = sum(
        votes not in {0, readers}
        for votes, readers in zip(positive_votes, reader_counts)
    )
    return {
        "reader_support_brier": mean(
            (probability - support) ** 2
            for probability, support in zip(probabilities, supports)
        ),
        "reader_support_nll": nll,
        "reader_support_spearman": rho,
        "unanimous_claim_state_auroc": auc(clear_labels, clear_scores),
        "tristate_accuracy": mean(
            float(prediction == reference)
            for prediction, reference in zip(states, references)
        ),
        "unanimous_fabrication_rate": (
            sum(
                votes == 0 and state == "supported"
                for votes, state in zip(positive_votes, states)
            ) / negatives
            if negatives else None
        ),
        "unanimous_omission_rate": (
            sum(
                votes == readers and state == "refuted"
                for votes, readers, state in zip(positive_votes, reader_counts, states)
            ) / positives
            if positives else None
        ),
        "disagreement_overcommitment_rate": (
            sum(
                votes not in {0, readers} and state != "undetermined"
                for votes, readers, state in zip(positive_votes, reader_counts, states)
            ) / disagreements
            if disagreements else None
        ),
        "mean_predicted_support": mean(probabilities),
    }


def paired_metrics(
    score_by_key: dict[tuple[str, str], float],
    item_by_key: dict[tuple[str, str], dict[str, object]],
    roles: dict[tuple[str, str], tuple[str, str]],
) -> dict[str, float | None]:
    signed = []
    absolute = []
    nuisance = []
    responsive = []
    misdirected = []
    for triplet_id in sorted({triplet_id for triplet_id, _ in roles}):
        keys = [roles.get((triplet_id, role)) for role in ROLES]
        if any(key is None or key not in score_by_key for key in keys):
            continue
        anchor, same, opposite = keys
        anchor_score = score_by_key[anchor]
        same_score = score_by_key[same]
        opposite_score = score_by_key[opposite]
        anchor_support = float(item_by_key[anchor]["reader_support"])
        opposite_support = float(item_by_key[opposite]["reader_support"])
        direction = 1.0 if anchor_support > opposite_support else -1.0
        signed_change = direction * (anchor_score - opposite_score)
        absolute_change = abs(anchor_score - opposite_score)
        nuisance_change = abs(anchor_score - same_score)
        is_responsive = absolute_change > nuisance_change
        signed.append(signed_change)
        absolute.append(absolute_change)
        nuisance.append(nuisance_change)
        responsive.append(float(is_responsive))
        misdirected.append(float(is_responsive and signed_change < 0.0))
    if not signed:
        return {"complete_triplets": 0}
    responsive_count = sum(responsive)
    return {
        "complete_triplets": len(signed),
        "mean_signed_clinical_change": mean(signed),
        "mean_absolute_clinical_change": mean(absolute),
        "mean_absolute_nuisance_change": mean(nuisance),
        "clinical_selectivity_gap": mean(
            clinical - noise for clinical, noise in zip(signed, nuisance)
        ),
        "directional_pairwise_accuracy": mean(float(value > 0.0) for value in signed),
        "unsigned_responsive_rate": mean(responsive),
        "misdirected_responsive_rate": mean(misdirected),
        "misdirection_given_unsigned_responsive": (
            sum(misdirected) / responsive_count if responsive_count else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = {}
    for row in load_jsonl(args.manifest):
        key = (str(row["finding"]), str(row["image_id"]))
        if key in manifest:
            raise ValueError(f"duplicate finding/image contract: {key}")
        manifest[key] = row
    raw = [
        row
        for path in args.raw
        for row in load_jsonl(path)
        if row.get("status") == "ok"
    ]
    if not raw:
        raise ValueError("no successful raw records")
    layers = sorted(int(key) for key in raw[0]["measurement"]["trajectory"])

    items: dict[tuple[str, str], dict[str, object]] = {}
    roles: dict[tuple[str, str], tuple[str, str]] = {}
    for row in raw:
        key = (str(row["finding"]), str(row["image_id"]))
        contract = manifest.get(key)
        if contract is None:
            raise ValueError(f"raw record missing from manifest: {key}")
        features = []
        for layer in layers:
            logits = row["measurement"]["trajectory"][str(layer)]["real_logits"]
            features.append(0.5 * (logits["supported"] - logits["refuted"]))
        items[key] = {
            "features": torch.tensor(features, dtype=torch.float32),
            "positive_votes": int(contract["positive_votes"]),
            "reader_count": int(contract["reader_count"]),
            "reader_support": float(contract["reader_support"]),
            "split": str(contract["experiment_split"]),
        }
        roles[(str(contract["triplet_id"]), str(contract["swap_role"]))] = key

    train_keys = sorted(key for key, item in items.items() if item["split"] == "dev")
    test_keys = sorted(key for key, item in items.items() if item["split"] == "test")
    if not train_keys or not test_keys:
        raise ValueError("both dev and test items are required")
    train_index = {key: index for index, key in enumerate(train_keys)}
    train_x = torch.stack([items[key]["features"] for key in train_keys])
    train_support = torch.tensor(
        [items[key]["reader_support"] for key in train_keys], dtype=torch.float32
    )
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0).clamp_min(0.1)

    train_triplets = []
    for triplet_id in sorted({triplet_id for triplet_id, _ in roles}):
        keys = [roles.get((triplet_id, role)) for role in ROLES]
        if any(key is None or key not in train_index for key in keys):
            continue
        anchor, same, opposite = (train_index[key] for key in keys)
        anchor_support = float(items[keys[0]]["reader_support"])
        opposite_support = float(items[keys[2]]["reader_support"])
        direction = 1.0 if anchor_support > opposite_support else -1.0
        separation = abs(anchor_support - opposite_support)
        train_triplets.append((anchor, same, opposite, direction, separation))

    def evaluate(scores: list[float], keys: list[tuple[str, str]]) -> dict[str, object]:
        score_by_key = dict(zip(keys, scores))
        return {
            "claim_metrics": probability_metrics(
                [int(items[key]["positive_votes"]) for key in keys],
                [int(items[key]["reader_count"]) for key in keys],
                scores,
            ),
            "paired_metrics": paired_metrics(score_by_key, items, roles),
        }

    def fit(name: str, final_only: bool, pair_mode: str) -> dict[str, object]:
        torch.manual_seed(args.seed)
        normalized = (train_x - feature_mean) / feature_std
        if final_only:
            normalized = normalized[:, -1:]
        weights = (0.01 * torch.randn(normalized.shape[1])).requires_grad_()
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias], lr=args.learning_rate)
        for _ in range(args.steps):
            scores = normalized @ weights + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scores, train_support
            ) + args.l2 * weights.square().sum()
            if pair_mode != "none":
                if not train_triplets:
                    raise ValueError("paired fit requires complete dev triplets")
                pair_losses = []
                for anchor, same, opposite, direction, separation in train_triplets:
                    delta = scores[anchor] - scores[opposite]
                    margin = args.pair_margin * separation
                    nuisance = (scores[anchor] - scores[same]).abs()
                    if pair_mode == "invariance":
                        response = nuisance.square()
                    elif pair_mode == "unsigned":
                        response = torch.relu(margin - delta.abs()).square()
                    elif pair_mode == "unsigned_selective":
                        response = torch.relu(
                            margin + nuisance - delta.abs()
                        ).square()
                    elif pair_mode == "directional":
                        response = torch.relu(margin - direction * delta).square()
                    elif pair_mode == "aligned":
                        # A constant/uncertain solution cannot satisfy this margin:
                        # signed clinical response must exceed nuisance response.
                        response = torch.relu(
                            margin + nuisance - direction * delta
                        ).square()
                    else:
                        raise ValueError(f"unknown pair mode: {pair_mode}")
                    pair_losses.append(response)
                loss = loss + args.pair_weight * torch.stack(pair_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        def score(keys: list[tuple[str, str]]) -> list[float]:
            matrix = torch.stack([items[key]["features"] for key in keys])
            matrix = (matrix - feature_mean) / feature_std
            if final_only:
                matrix = matrix[:, -1:]
            return (matrix @ weights.detach() + bias.detach()).tolist()

        return {
            "name": name,
            "weights": weights.detach().tolist(),
            "bias": float(bias.detach()),
            "dev": evaluate(score(train_keys), train_keys),
            "test": evaluate(score(test_keys), test_keys),
        }

    raw_test_scores = [float(items[key]["features"][-1]) for key in test_keys]
    contracts = list(manifest.values())
    formal_reference = all(
        row.get("formal_reference") is True
        and row.get("reference_source") == "vindr_reader_votes"
        for row in contracts
    )
    result = {
        "version": VERSION,
        "formal_reference": formal_reference,
        "evidence_grades": sorted(
            {str(row.get("evidence_grade", "ungraded")) for row in contracts}
        ),
        "layers": layers,
        "train_unique_claim_images": len(train_keys),
        "test_unique_claim_images": len(test_keys),
        "train_complete_triplets": len(train_triplets),
        "hyperparameters_frozen_without_test_tuning": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "pair_weight": args.pair_weight,
            "pair_margin": args.pair_margin,
            "seed": args.seed,
        },
        "methods": {
            "raw_final": {"test": evaluate(raw_test_scores, test_keys)},
            "calibrated_final": fit("calibrated_final", True, "none"),
            "supervised_layer_mixer": fit("supervised_layer_mixer", False, "none"),
            "invariance_only_mixer": fit("invariance_only_mixer", False, "invariance"),
            "unsigned_response_mixer": fit("unsigned_response_mixer", False, "unsigned"),
            "unsigned_selectivity_mixer": fit(
                "unsigned_selectivity_mixer", False, "unsigned_selective"
            ),
            "directional_response_mixer": fit("directional_response_mixer", False, "directional"),
            "clinical_response_aligner": fit("clinical_response_aligner", False, "aligned"),
        },
        "claim_ceiling": (
            "A mitigation claim requires formal VinDr reader votes, improvement over "
            "calibrated-final and supervised-layer controls, a win over the unsigned "
            "response control, and OE gains at matched claim coverage."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
