#!/usr/bin/env python3
"""Estimate reader-adjusted claim support from independent VinDr votes.

Raw 0/3--3/3 bins remain the primary, directly auditable reference.  This
module supplies a sensitivity analysis for the possibility that a particular
three-reader panel is systematically liberal or conservative.  Reader and
finding effects are learned on the development split only.  Test-item support
is then inferred from its three observed votes with those nuisance effects
frozen; no VLM feature or output enters the reference model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch


VERSION = "reader-adjusted-support-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validated_votes(row: dict[str, object]) -> tuple[tuple[str, int], ...]:
    if int(row.get("reader_count", -1)) != 3:
        raise ValueError("formal VinDr rows require reader_count == 3")
    raw = row.get("reader_votes")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("formal VinDr rows require three reader_votes")
    pairs: list[tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("reader_votes entries must be objects")
        reader = str(item.get("rad_id", ""))
        vote = int(item.get("vote", -1))
        if not reader or vote not in {0, 1}:
            raise ValueError("reader vote requires a rad_id and binary vote")
        pairs.append((reader, vote))
    pairs.sort()
    if len({reader for reader, _ in pairs}) != 3:
        raise ValueError("reader_votes must contain three distinct rad_ID values")
    if sum(vote for _, vote in pairs) != int(row.get("positive_votes", -1)):
        raise ValueError("reader_votes disagree with positive_votes")
    expected = row.get("reader_ids")
    if expected is not None and tuple(str(value) for value in expected) != tuple(
        reader for reader, _ in pairs
    ):
        raise ValueError("reader_ids disagree with reader_votes")
    return tuple(pairs)


def infer_item_map(
    votes: tuple[tuple[str, int], ...],
    finding_intercept: float,
    reader_bias: dict[str, float],
    item_l2: float,
    max_steps: int = 50,
) -> tuple[float, float]:
    """Infer one test item's MAP residual and Laplace standard error."""

    if item_l2 <= 0:
        raise ValueError("item_l2 must be positive")
    missing = sorted({reader for reader, _ in votes} - set(reader_bias))
    if missing:
        raise ValueError(f"test row contains readers unseen on dev: {missing}")
    residual = 0.0
    hessian = item_l2
    for _ in range(max_steps):
        probabilities = [
            1.0
            / (1.0 + math.exp(-(finding_intercept + reader_bias[reader] + residual)))
            for reader, _ in votes
        ]
        gradient = sum(
            probability - vote
            for probability, (_, vote) in zip(probabilities, votes)
        ) + item_l2 * residual
        hessian = sum(
            probability * (1.0 - probability) for probability in probabilities
        ) + item_l2
        update = gradient / hessian
        residual -= update
        if abs(update) < 1e-8:
            break
    return residual, 1.0 / math.sqrt(hessian)


def fit_reader_effects(
    rows: Iterable[dict[str, object]],
    *,
    steps: int = 1500,
    learning_rate: float = 0.03,
    reader_l2: float = 0.2,
    item_l2: float = 1.0,
    finding_l2: float = 0.01,
    seed: int = 42,
) -> dict[str, object]:
    """Fit a penalized many-facet logistic model on development rows."""

    dev_rows = [row for row in rows if str(row.get("experiment_split")) == "dev"]
    if not dev_rows:
        raise ValueError("reader effects require non-empty development rows")
    if min(steps, reader_l2, item_l2, finding_l2) <= 0 or learning_rate <= 0:
        raise ValueError("optimization steps, learning rate, and penalties must be positive")

    votes_by_item = [validated_votes(row) for row in dev_rows]
    readers = sorted({reader for votes in votes_by_item for reader, _ in votes})
    findings = sorted({str(row["finding"]) for row in dev_rows})
    reader_index = {value: index for index, value in enumerate(readers)}
    finding_index = {value: index for index, value in enumerate(findings)}

    observation_item: list[int] = []
    observation_reader: list[int] = []
    observation_finding: list[int] = []
    observation_vote: list[float] = []
    for item_index, (row, votes) in enumerate(zip(dev_rows, votes_by_item)):
        finding = finding_index[str(row["finding"])]
        for reader, vote in votes:
            observation_item.append(item_index)
            observation_reader.append(reader_index[reader])
            observation_finding.append(finding)
            observation_vote.append(float(vote))

    item_ids = torch.tensor(observation_item, dtype=torch.long)
    reader_ids = torch.tensor(observation_reader, dtype=torch.long)
    finding_ids = torch.tensor(observation_finding, dtype=torch.long)
    targets = torch.tensor(observation_vote, dtype=torch.float32)

    torch.manual_seed(seed)
    finding_intercepts = torch.zeros(len(findings), requires_grad=True)
    raw_reader_biases = torch.zeros(len(readers), requires_grad=True)
    item_residuals = torch.zeros(len(dev_rows), requires_grad=True)
    optimizer = torch.optim.Adam(
        [finding_intercepts, raw_reader_biases, item_residuals],
        lr=learning_rate,
    )
    final_loss = None
    for _ in range(steps):
        centered_reader_biases = raw_reader_biases - raw_reader_biases.mean()
        logits = (
            finding_intercepts[finding_ids]
            + centered_reader_biases[reader_ids]
            + item_residuals[item_ids]
        )
        likelihood = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="sum"
        )
        penalty = (
            reader_l2 * centered_reader_biases.square().sum()
            + item_l2 * item_residuals.square().sum()
            + finding_l2 * finding_intercepts.square().sum()
        )
        loss = likelihood + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    centered = raw_reader_biases.detach() - raw_reader_biases.detach().mean()
    return {
        "version": VERSION,
        "fit_split": "dev",
        "readers": readers,
        "findings": findings,
        "reader_bias": {
            reader: float(centered[index]) for reader, index in reader_index.items()
        },
        "finding_intercept": {
            finding: float(finding_intercepts.detach()[index])
            for finding, index in finding_index.items()
        },
        "dev_item_residual": {
            f"{row['finding']}:{row['image_id']}": float(item_residuals.detach()[index])
            for index, row in enumerate(dev_rows)
        },
        "hyperparameters": {
            "steps": steps,
            "learning_rate": learning_rate,
            "reader_l2": reader_l2,
            "item_l2": item_l2,
            "finding_l2": finding_l2,
            "seed": seed,
        },
        "fit_counts": {
            "claim_rows": len(dev_rows),
            "votes": len(observation_vote),
            "readers": len(readers),
            "findings": len(findings),
        },
        "objective_final": final_loss,
        "identifiability_constraint": "reader biases sum to zero",
    }


def adjust_rows(
    rows: Iterable[dict[str, object]], model: dict[str, object]
) -> list[dict[str, object]]:
    reader_bias = {
        str(key): float(value)
        for key, value in dict(model["reader_bias"]).items()
    }
    finding_intercept = {
        str(key): float(value)
        for key, value in dict(model["finding_intercept"]).items()
    }
    dev_residual = {
        str(key): float(value)
        for key, value in dict(model["dev_item_residual"]).items()
    }
    item_l2 = float(dict(model["hyperparameters"])["item_l2"])
    output = []
    for source in rows:
        row = dict(source)
        votes = validated_votes(row)
        finding = str(row["finding"])
        if finding not in finding_intercept:
            raise ValueError(f"row contains finding unseen on dev: {finding}")
        key = f"{finding}:{row['image_id']}"
        if str(row.get("experiment_split")) == "dev":
            if key not in dev_residual:
                raise ValueError(f"development row absent from fitted item effects: {key}")
            residual = dev_residual[key]
            _, standard_error = infer_item_map(
                votes, finding_intercept[finding], reader_bias, item_l2
            )
            inference = "joint_dev_fit"
        elif str(row.get("experiment_split")) == "test":
            residual, standard_error = infer_item_map(
                votes, finding_intercept[finding], reader_bias, item_l2
            )
            inference = "test_vote_map_with_frozen_dev_effects"
        else:
            raise ValueError("experiment_split must be dev or test")
        latent_logit = finding_intercept[finding] + residual
        support = 1.0 / (1.0 + math.exp(-latent_logit))
        clarity = abs(2.0 * support - 1.0)
        probability = min(max(support, 1e-12), 1.0 - 1e-12)
        entropy = -(
            probability * math.log(probability)
            + (1.0 - probability) * math.log(1.0 - probability)
        ) / math.log(2.0)
        row.update(
            {
                "reader_adjustment_version": VERSION,
                "reader_adjusted_support": support,
                "reader_adjusted_clarity": clarity,
                "reader_adjusted_entropy_bits": entropy,
                "reader_adjusted_logit": latent_logit,
                "reader_adjusted_item_residual": residual,
                "reader_adjusted_item_se_laplace": standard_error,
                "reader_adjusted_inference": inference,
                "reader_adjusted_reference_role": "sensitivity_only",
            }
        )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--reader-l2", type=float, default=0.2)
    parser.add_argument("--item-l2", type=float, default=1.0)
    parser.add_argument("--finding-l2", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    model = fit_reader_effects(
        rows,
        steps=args.steps,
        learning_rate=args.learning_rate,
        reader_l2=args.reader_l2,
        item_l2=args.item_l2,
        finding_l2=args.finding_l2,
        seed=args.seed,
    )
    adjusted = adjust_rows(rows, model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reader_effect_model.json").write_text(
        json.dumps(
            {
                **model,
                "manifest_sha256": sha256_file(args.manifest),
                "reference_semantics": (
                    "reader-adjusted support is a sensitivity analysis; raw "
                    "0/3--3/3 bins remain primary"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "reader_adjusted_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in adjusted),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "version": VERSION,
                "rows": len(adjusted),
                "fit_counts": model["fit_counts"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
