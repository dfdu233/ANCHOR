#!/usr/bin/env python3
"""Probe whether multi-reader ambiguity survives into a VLM claim state.

This probe deliberately separates *what* the claim says (polarity) from how
strongly it is said (commitment).  It never adds, deletes, or flips a claim:
the learned gate may only retain a definite final-layer polarity or realize it
as ``undetermined``.  Formal use requires all four VinDr reader-vote bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from corrected_sgta.clinical_claims import epistemic_coordinates
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    intervention_coordinate_changes,
)


VERSION = "reader-agreement-gate-v7-directional-boundary"

CONTROL_NAMES = (
    "polarity_control",
    "confidence_control",
    "entropy_control",
    "temperature_control",
    "image_null_control",
    "norm_matched_null_control",
    "random_features_control",
)


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


def validated_reader_ids(row: dict[str, object]) -> tuple[str, ...] | None:
    """Validate reader-level provenance and return sorted pseudonymous IDs."""

    if int(row.get("reader_count", -1)) != 3:
        return None
    raw = row.get("reader_votes")
    if not isinstance(raw, list) or len(raw) != int(row["reader_count"]):
        return None
    pairs: list[tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        rad_id = str(item.get("rad_id", ""))
        vote = int(item.get("vote", -1))
        if not rad_id or vote not in {0, 1}:
            return None
        pairs.append((rad_id, vote))
    pairs.sort()
    if len({rad_id for rad_id, _ in pairs}) != int(row["reader_count"]):
        return None
    if sum(vote for _, vote in pairs) != int(row["positive_votes"]):
        return None
    expected_ids = row.get("reader_ids")
    ids = tuple(rad_id for rad_id, _ in pairs)
    if expected_ids is not None and tuple(expected_ids) != ids:
        return None
    return ids


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


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid for unbounded sequence scores."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def probability_features(logits: dict[str, float], temperature: float = 1.0) -> torch.Tensor:
    values = torch.tensor(
        [float(logits[name]) / temperature for name in ("supported", "refuted", "undetermined")],
        dtype=torch.float32,
    )
    probabilities = torch.softmax(values, dim=0)
    ordered = torch.sort(probabilities, descending=True).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return torch.stack((ordered[0], ordered[0] - ordered[1] + 0.0 * entropy))


def probability_map(logits: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    names = ("supported", "refuted", "undetermined")
    values = torch.tensor([float(logits[name]) / temperature for name in names])
    probabilities = torch.softmax(values, dim=0).tolist()
    return dict(zip(names, probabilities))


def stable_random_features(key: tuple[str, str], layer: int) -> torch.Tensor:
    digest = hashlib.sha256(f"{VERSION}:{key[0]}:{key[1]}:{layer}".encode()).digest()
    values = [int.from_bytes(digest[offset:offset + 8], "big") / 2**64 for offset in (0, 8)]
    return torch.tensor([2.0 * value - 1.0 for value in values], dtype=torch.float32)


def agreement_metrics(
    positive_votes: list[int],
    reader_counts: list[int],
    agreement_logits: list[float],
    polarity_scores: list[float],
) -> dict[str, float | None]:
    if not positive_votes or not (
        len(positive_votes)
        == len(reader_counts)
        == len(agreement_logits)
        == len(polarity_scores)
    ):
        raise ValueError("agreement metric inputs must be non-empty and aligned")
    labels = [
        int(votes in {0, readers})
        for votes, readers in zip(positive_votes, reader_counts)
    ]
    probabilities = [sigmoid(value) for value in agreement_logits]
    predicted_clear = [probability >= 0.5 for probability in probabilities]
    prediction_states = [
        ("supported" if polarity >= 0.0 else "refuted")
        if clear else "undetermined"
        for clear, polarity in zip(predicted_clear, polarity_scores)
    ]
    reference_states = [
        "supported" if votes == readers else
        "refuted" if votes == 0 else
        "undetermined"
        for votes, readers in zip(positive_votes, reader_counts)
    ]
    clear_indices = [index for index, label in enumerate(labels) if label]
    disagreement_indices = [index for index, label in enumerate(labels) if not label]
    negative_indices = [index for index, votes in enumerate(positive_votes) if votes == 0]
    positive_indices = [
        index
        for index, (votes, readers) in enumerate(zip(positive_votes, reader_counts))
        if votes == readers
    ]
    eps = 1e-7
    nll = mean(
        -label * math.log(max(eps, min(1 - eps, probability)))
        -(1 - label) * math.log(max(eps, min(1 - eps, 1 - probability)))
        for label, probability in zip(labels, probabilities)
    )
    return {
        "agreement_auroc": auc(labels, agreement_logits),
        "agreement_brier": mean(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        ),
        "agreement_nll": nll,
        "agreement_accuracy": mean(
            float(prediction == bool(label))
            for prediction, label in zip(predicted_clear, labels)
        ),
        "tri_state_accuracy": mean(
            float(prediction == reference)
            for prediction, reference in zip(prediction_states, reference_states)
        ),
        "clear_case_accuracy": (
            mean(
                float(prediction_states[index] == reference_states[index])
                for index in clear_indices
            ) if clear_indices else None
        ),
        "disagreement_overcommitment_rate": (
            mean(
                float(prediction_states[index] != "undetermined")
                for index in disagreement_indices
            ) if disagreement_indices else None
        ),
        "unanimous_fabrication_rate": (
            mean(
                float(prediction_states[index] == "supported")
                for index in negative_indices
            ) if negative_indices else None
        ),
        "unanimous_omission_rate": (
            mean(
                float(prediction_states[index] != "supported")
                for index in positive_indices
            ) if positive_indices else None
        ),
        "definite_rate": mean(float(value) for value in predicted_clear),
    }


def paired_cluster_bootstrap_increment(
    labels: list[int],
    baseline_logits: list[float],
    candidate_logits: list[float],
    clusters: list[str],
    draws: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    """Held-out gain from a genuinely new feature, clustered by image.

    AUROC gain is candidate minus baseline.  Brier and NLL gains are baseline
    minus candidate, so positive values consistently favor the candidate.
    """

    lengths = {len(labels), len(baseline_logits), len(candidate_logits), len(clusters)}
    if lengths != {len(labels)} or not labels:
        raise ValueError("bootstrap inputs must be non-empty and aligned")
    if draws <= 0:
        raise ValueError("draws must be positive")

    def metrics(indices: list[int]) -> dict[str, float | None]:
        selected_labels = [labels[index] for index in indices]
        baseline = [baseline_logits[index] for index in indices]
        candidate = [candidate_logits[index] for index in indices]
        baseline_probabilities = [sigmoid(value) for value in baseline]
        candidate_probabilities = [sigmoid(value) for value in candidate]
        eps = 1e-7

        def brier(probabilities: list[float]) -> float:
            return mean(
                (probability - label) ** 2
                for probability, label in zip(probabilities, selected_labels)
            )

        def nll(probabilities: list[float]) -> float:
            return mean(
                -label * math.log(max(eps, min(1 - eps, probability)))
                - (1 - label)
                * math.log(max(eps, min(1 - eps, 1 - probability)))
                for probability, label in zip(probabilities, selected_labels)
            )

        baseline_auc = auc(selected_labels, baseline)
        candidate_auc = auc(selected_labels, candidate)
        return {
            "auroc_gain": (
                None
                if baseline_auc is None or candidate_auc is None
                else candidate_auc - baseline_auc
            ),
            "brier_gain": brier(baseline_probabilities) - brier(candidate_probabilities),
            "nll_gain": nll(baseline_probabilities) - nll(candidate_probabilities),
        }

    by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        by_cluster.setdefault(cluster, []).append(index)
    cluster_ids = sorted(by_cluster)
    observed = metrics(list(range(len(labels))))
    samples: dict[str, list[float]] = {name: [] for name in observed}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        indices = [
            index
            for cluster in sampled
            for index in by_cluster[str(cluster)]
        ]
        values = metrics(indices)
        for name, value in values.items():
            if value is not None and math.isfinite(value):
                samples[name].append(value)

    output = {}
    for name, estimate in observed.items():
        values = samples[name]
        output[name] = {
            "estimate": estimate,
            "ci_low": float(np.quantile(values, 0.025)) if values else None,
            "ci_high": float(np.quantile(values, 0.975)) if values else None,
            "valid_draws": len(values),
        }
    return output


def paired_cluster_bootstrap_continuous(
    targets: list[float],
    baseline_probabilities: list[float],
    candidate_probabilities: list[float],
    clusters: list[str],
    draws: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    """Clustered held-out gain for a continuous clarity reference."""

    lengths = {
        len(targets),
        len(baseline_probabilities),
        len(candidate_probabilities),
        len(clusters),
    }
    if lengths != {len(targets)} or not targets:
        raise ValueError("continuous bootstrap inputs must be non-empty and aligned")
    if draws <= 0 or any(not 0.0 <= value <= 1.0 for value in targets):
        raise ValueError("continuous targets must lie in [0,1] and draws be positive")

    def metrics(indices: list[int]) -> dict[str, float]:
        def error(values: list[float], power: int) -> float:
            return mean(
                abs(values[index] - targets[index]) ** power for index in indices
            )

        return {
            "brier_gain": error(baseline_probabilities, 2)
            - error(candidate_probabilities, 2),
            "mae_gain": error(baseline_probabilities, 1)
            - error(candidate_probabilities, 1),
        }

    by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        by_cluster.setdefault(cluster, []).append(index)
    cluster_ids = sorted(by_cluster)
    observed = metrics(list(range(len(targets))))
    samples: dict[str, list[float]] = {name: [] for name in observed}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        indices = [
            index for cluster in sampled for index in by_cluster[str(cluster)]
        ]
        values = metrics(indices)
        for name, value in values.items():
            if math.isfinite(value):
                samples[name].append(value)
    return {
        name: {
            "estimate": estimate,
            "ci_low": float(np.quantile(samples[name], 0.025)),
            "ci_high": float(np.quantile(samples[name], 0.975)),
            "valid_draws": len(samples[name]),
        }
        for name, estimate in observed.items()
    }


def cluster_bootstrap_mean(
    values: list[float], clusters: list[str], draws: int, seed: int
) -> dict[str, float | int | None]:
    if not values or len(values) != len(clusters):
        return {"estimate": None, "ci_low": None, "ci_high": None, "valid_draws": 0}
    by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        by_cluster.setdefault(cluster, []).append(index)
    cluster_ids = sorted(by_cluster)
    observed = mean(values)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        indices = [index for cluster in selected for index in by_cluster[str(cluster)]]
        samples.append(mean(values[index] for index in indices))
    return {
        "estimate": observed,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "valid_draws": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--reader-adjusted-manifest",
        type=Path,
        help=(
            "optional sensitivity-only manifest from "
            "fit_reader_adjusted_support; required for formal authorization"
        ),
    )
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--boundary-output",
        type=Path,
        help="optional JSONL records for medeval.classify_layer_boundary",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument(
        "--power-max-ci-width",
        type=float,
        default=0.10,
        help="maximum AUROC 95%% CI width accepted as powered for a 0.05 boundary",
    )
    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=10,
        help=(
            "minimum test examples required in each of the 0/3, 1/3, 2/3, "
            "and 3/3 bins for a finding to enter the majority gate"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-id",
        help="stable model identifier required by the formal projection authorization",
    )
    args = parser.parse_args()
    if args.min_test_per_class <= 0:
        raise ValueError("min-test-per-class must be positive")
    if args.power_max_ci_width <= 0:
        raise ValueError("power-max-ci-width must be positive")
    if args.boundary_output is not None and not args.model_id:
        raise ValueError("--boundary-output requires --model-id")

    manifest = {}
    for row in load_jsonl(args.manifest):
        key = (str(row["finding"]), str(row["image_id"]))
        if key in manifest:
            raise ValueError(f"duplicate finding/image contract: {key}")
        manifest[key] = row
    adjusted_reference: dict[tuple[str, str], dict[str, object]] = {}
    if args.reader_adjusted_manifest is not None:
        for row in load_jsonl(args.reader_adjusted_manifest):
            key = (str(row["finding"]), str(row["image_id"]))
            if key in adjusted_reference:
                raise ValueError(f"duplicate adjusted finding/image reference: {key}")
            source = manifest.get(key)
            if source is None:
                raise ValueError(f"adjusted reference absent from raw manifest: {key}")
            if row.get("reader_adjusted_reference_role") != "sensitivity_only":
                raise ValueError("reader-adjusted references must be sensitivity_only")
            if (
                int(row["positive_votes"]) != int(source["positive_votes"])
                or row.get("reader_votes") != source.get("reader_votes")
                or str(row.get("experiment_split"))
                != str(source.get("experiment_split"))
            ):
                raise ValueError(f"adjusted reference changed raw vote provenance: {key}")
            clarity = float(row["reader_adjusted_clarity"])
            if not 0.0 <= clarity <= 1.0:
                raise ValueError(f"invalid reader-adjusted clarity for {key}: {clarity}")
            adjusted_reference[key] = row
        if set(adjusted_reference) != set(manifest):
            missing = sorted(set(manifest) - set(adjusted_reference))[:10]
            raise ValueError(f"adjusted reference does not cover raw manifest: {missing}")
    raw = [
        row
        for path in args.raw
        for row in load_jsonl(path)
        if row.get("status") == "ok"
    ]
    if not raw:
        raise ValueError("no successful model records")
    layers = sorted(int(key) for key in raw[0]["measurement"]["trajectory"])
    final_layer = layers[-1]

    items = {}
    for row in raw:
        key = (str(row["finding"]), str(row["image_id"]))
        if key in items:
            raise ValueError(f"duplicate model finding/image record: {key}")
        contract = manifest.get(key)
        if contract is None:
            raise ValueError(f"raw record missing from manifest: {key}")
        row_layers = sorted(int(value) for value in row["measurement"]["trajectory"])
        if row_layers != layers:
            raise ValueError(
                f"inconsistent layer trajectory for {key}: {row_layers} != {layers}"
            )
        polarities = []
        commitments = []
        baseline_states = []
        layer_controls: dict[str, list[torch.Tensor]] = {
            name: [] for name in CONTROL_NAMES
        }
        boundary_controls_complete = True
        for layer in layers:
            trajectory = row["measurement"]["trajectory"][str(layer)]
            coordinates = epistemic_coordinates(trajectory["real_logits"])
            polarities.append(float(coordinates["polarity"]))
            commitments.append(float(coordinates["commitment"]))
            baseline_states.append(str(trajectory["baseline_state"]))
            polarity = float(coordinates["polarity"])
            probabilities = trajectory.get("real_probabilities") or probability_map(
                trajectory["real_logits"]
            )
            entropy = float(
                trajectory.get(
                    "real_entropy",
                    -sum(float(value) * math.log(max(float(value), 1e-12)) for value in probabilities.values()),
                )
            )
            null_logits = trajectory.get("null_logits", trajectory["real_logits"])
            null_coordinates = epistemic_coordinates(null_logits)
            norm_null_logits = trajectory.get("norm_matched_null_logits")
            controls_complete = norm_null_logits is not None
            boundary_controls_complete = boundary_controls_complete and controls_complete
            norm_null_logits = norm_null_logits or null_logits
            norm_null_coordinates = epistemic_coordinates(norm_null_logits)
            layer_controls["polarity_control"].append(
                torch.tensor([abs(polarity), 0.0], dtype=torch.float32)
            )
            ordered = sorted((float(value) for value in probabilities.values()), reverse=True)
            layer_controls["confidence_control"].append(
                torch.tensor([ordered[0], ordered[0] - ordered[1]], dtype=torch.float32)
            )
            layer_controls["entropy_control"].append(
                torch.tensor([entropy, 0.0], dtype=torch.float32)
            )
            layer_controls["temperature_control"].append(
                probability_features(trajectory["real_logits"], temperature=1.2)
            )
            layer_controls["image_null_control"].append(
                torch.tensor(
                    [abs(float(null_coordinates["polarity"])), float(null_coordinates["commitment"])],
                    dtype=torch.float32,
                )
            )
            layer_controls["norm_matched_null_control"].append(
                torch.tensor(
                    [abs(float(norm_null_coordinates["polarity"])), float(norm_null_coordinates["commitment"])],
                    dtype=torch.float32,
                )
            )
            layer_controls["random_features_control"].append(
                stable_random_features(key, layer)
            )
        items[key] = {
            "finding": str(contract["finding"]),
            "image_id": str(contract["image_id"]),
            "positive_votes": int(contract["positive_votes"]),
            "reader_count": int(contract["reader_count"]),
            "reader_ids": validated_reader_ids(contract),
            "split": str(contract["experiment_split"]),
            "polarities": torch.tensor(polarities, dtype=torch.float32),
            "commitments": torch.tensor(commitments, dtype=torch.float32),
            "baseline_state": baseline_states[-1],
            "layer_controls": layer_controls,
            "raw_record": row,
            "boundary_controls_complete": boundary_controls_complete and "activation_intervention" in row["measurement"],
        }

    train_keys = sorted(key for key, item in items.items() if item["split"] == "dev")
    test_keys = sorted(key for key, item in items.items() if item["split"] == "test")
    if not train_keys or not test_keys:
        raise ValueError("both image-disjoint dev and test items are required")
    for split_name, keys in (("dev", train_keys), ("test", test_keys)):
        has_clear = any(
            int(items[key]["positive_votes"])
            in {0, int(items[key]["reader_count"])}
            for key in keys
        )
        has_disagreement = any(
            0
            < int(items[key]["positive_votes"])
            < int(items[key]["reader_count"])
            for key in keys
        )
        if not (has_clear and has_disagreement):
            raise ValueError(
                f"{split_name} lacks clear or disagreement reader-vote examples; "
                "grade-C binary smoke data cannot test agreement retention"
            )
        vote_bins = {
            int(items[key]["positive_votes"])
            for key in keys
            if int(items[key]["reader_count"]) == 3
        }
        if vote_bins != {0, 1, 2, 3}:
            raise ValueError(
                f"{split_name} lacks one or more 0/3--3/3 vote bins: "
                f"observed {sorted(vote_bins)}"
            )

    reader_vocab = sorted(
        {
            reader
            for key in train_keys
            for reader in (items[key]["reader_ids"] or ())
        }
    )
    finding_vocab = sorted({str(items[key]["finding"]) for key in train_keys})
    test_readers = sorted(
        {
            reader
            for key in test_keys
            for reader in (items[key]["reader_ids"] or ())
        }
    )
    test_findings = sorted({str(items[key]["finding"]) for key in test_keys})
    unseen_test_readers = sorted(set(test_readers) - set(reader_vocab))
    unseen_test_findings = sorted(set(test_findings) - set(finding_vocab))
    if unseen_test_readers or unseen_test_findings:
        raise ValueError(
            "reader/finding adjustment is undefined for held-out nuisance levels; "
            f"unseen readers={unseen_test_readers}, "
            f"unseen findings={unseen_test_findings}"
        )

    def nuisance_features(key: tuple[str, str]) -> torch.Tensor:
        readers = set(items[key]["reader_ids"] or ())
        finding = str(items[key]["finding"])
        return torch.tensor(
            [float(reader in readers) for reader in reader_vocab]
            + [float(value == finding) for value in finding_vocab],
            dtype=torch.float32,
        )

    def finding_features(key: tuple[str, str]) -> torch.Tensor:
        finding = str(items[key]["finding"])
        return torch.tensor(
            [float(value == finding) for value in finding_vocab],
            dtype=torch.float32,
        )

    def features(key: tuple[str, str], mode: str) -> torch.Tensor:
        polarity = items[key]["polarities"]
        commitment = items[key]["commitments"]
        reader_adjusted = mode.endswith("_reader_adjusted")
        finding_adjusted = mode.endswith("_finding_adjusted")
        if reader_adjusted and finding_adjusted:
            raise ValueError("feature mode cannot use two adjustment suffixes")
        if reader_adjusted:
            core_mode = mode.removesuffix("_reader_adjusted")
        elif finding_adjusted:
            core_mode = mode.removesuffix("_finding_adjusted")
        else:
            core_mode = mode
        if core_mode.startswith("layer_"):
            layer_text, coordinate = core_mode.removeprefix("layer_").split("_", 1)
            layer = int(layer_text)
            index = layers.index(layer)
            if coordinate == "polarity":
                core = polarity[index : index + 1].abs()
            elif coordinate == "plane":
                core = torch.stack((polarity[index].abs(), commitment[index]))
            elif coordinate in CONTROL_NAMES:
                core = items[key]["layer_controls"][coordinate][index]
            else:
                raise ValueError(f"unknown layer coordinate: {coordinate}")
        elif core_mode == "multilayer_abs_polarity":
            core = polarity.abs()
        elif core_mode == "multilayer_claim_plane":
            core = torch.cat((polarity.abs(), commitment))
        else:
            raise ValueError(f"unknown feature mode: {mode}")
        if reader_adjusted:
            return torch.cat((nuisance_features(key), core))
        if finding_adjusted:
            return torch.cat((finding_features(key), core))
        return core

    train_labels = torch.tensor(
        [
            float(
                int(items[key]["positive_votes"])
                in {0, int(items[key]["reader_count"])}
            )
            for key in train_keys
        ],
        dtype=torch.float32,
    )

    def fit(mode: str) -> tuple[dict[str, object], dict[tuple[str, str], float]]:
        torch.manual_seed(args.seed)
        train_x = torch.stack([features(key, mode) for key in train_keys])
        feature_mean = train_x.mean(dim=0)
        feature_std = train_x.std(dim=0).clamp_min(0.1)
        normalized = (train_x - feature_mean) / feature_std
        weights = torch.zeros(normalized.shape[1], requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias], lr=args.learning_rate)
        for _ in range(args.steps):
            logits = normalized @ weights + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, train_labels
            ) + args.l2 * weights.square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        def predict(keys: list[tuple[str, str]]) -> list[float]:
            matrix = torch.stack([features(key, mode) for key in keys])
            return (
                (matrix - feature_mean) / feature_std @ weights.detach()
                + bias.detach()
            ).tolist()

        def evaluate(keys: list[tuple[str, str]]) -> dict[str, float | None]:
            logits = predict(keys)
            return agreement_metrics(
                [int(items[key]["positive_votes"]) for key in keys],
                [int(items[key]["reader_count"]) for key in keys],
                logits,
                [float(items[key]["polarities"][-1]) for key in keys],
            )

        record = {
            "feature_mode": mode,
            "weights": weights.detach().tolist(),
            "bias": float(bias.detach()),
            "normalization": {
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "fit_split": "dev",
            },
            "dev": evaluate(train_keys),
            "test": evaluate(test_keys),
        }
        all_keys = sorted(items)
        return record, dict(zip(all_keys, predict(all_keys)))

    layer_modes = [
        mode
        for layer in layers
        for mode in (
            f"layer_{layer}_polarity",
            f"layer_{layer}_plane",
            f"layer_{layer}_polarity_reader_adjusted",
            f"layer_{layer}_plane_reader_adjusted",
        )
    ]
    control_modes = [
        f"layer_{layer}_{control}_reader_adjusted"
        for layer in layers
        for control in CONTROL_NAMES
    ]
    modes = (
        *layer_modes,
        *control_modes,
        "multilayer_abs_polarity",
        "multilayer_claim_plane",
        "multilayer_abs_polarity_reader_adjusted",
        "multilayer_claim_plane_reader_adjusted",
    )
    methods: dict[str, dict[str, object]] = {}
    test_logits: dict[str, dict[tuple[str, str], float]] = {}
    for mode in modes:
        methods[mode], test_logits[mode] = fit(mode)

    # The layer choice is frozen using dev only.  Test metrics never choose a layer.
    early_layers = layers[:-1]
    if not early_layers:
        raise ValueError("at least one intermediate and one final layer are required")
    selected_layer = max(
        early_layers,
        key=lambda layer: (
            methods[f"layer_{layer}_plane_reader_adjusted"]["dev"]["agreement_auroc"]
            if methods[f"layer_{layer}_plane_reader_adjusted"]["dev"]["agreement_auroc"] is not None
            else -math.inf,
            -float(
                methods[f"layer_{layer}_plane_reader_adjusted"]["dev"]
                ["agreement_brier"]
            ),
            -layer,
        ),
    )

    test_labels = [
        int(
            int(items[key]["positive_votes"])
            in {0, int(items[key]["reader_count"])}
        )
        for key in test_keys
    ]
    test_clusters = [key[1] for key in test_keys]

    def compare(baseline: str, candidate: str, seed_offset: int) -> dict[str, object]:
        result = paired_cluster_bootstrap_increment(
            test_labels,
            [test_logits[baseline][key] for key in test_keys],
            [test_logits[candidate][key] for key in test_keys],
            test_clusters,
            args.bootstrap_draws,
            args.seed + seed_offset,
        )
        return {"baseline": baseline, "candidate": candidate, **result}

    selected_polarity = f"layer_{selected_layer}_polarity"
    selected_plane = f"layer_{selected_layer}_plane"
    final_polarity = f"layer_{final_layer}_polarity"
    final_plane = f"layer_{final_layer}_plane"
    selected_polarity_adjusted = f"{selected_polarity}_reader_adjusted"
    selected_plane_adjusted = f"{selected_plane}_reader_adjusted"
    final_polarity_adjusted = f"{final_polarity}_reader_adjusted"
    final_plane_adjusted = f"{final_plane}_reader_adjusted"
    comparisons = {
        "selected_layer_commitment_increment": compare(
            selected_polarity_adjusted, selected_plane_adjusted, 101
        ),
        "final_layer_commitment_increment": compare(
            final_polarity_adjusted, final_plane_adjusted, 102
        ),
        "selected_early_vs_final_claim_plane": compare(
            final_plane_adjusted, selected_plane_adjusted, 103
        ),
        "multilayer_commitment_increment": compare(
            "multilayer_abs_polarity_reader_adjusted",
            "multilayer_claim_plane_reader_adjusted",
            104,
        ),
        "unadjusted_selected_layer_commitment_increment_diagnostic": compare(
            selected_polarity, selected_plane, 105
        ),
    }

    def positive_ci(metric: dict[str, object], minimum: float = 0.0) -> bool:
        return bool(
            metric["estimate"] is not None
            and metric["ci_low"] is not None
            and float(metric["estimate"]) >= minimum
            and float(metric["ci_low"]) > 0.0
        )

    reader_adjusted_sensitivity: dict[str, object] = {
        "provided": False,
        "required_for_formal_authorization": True,
        "passed": False,
    }
    if adjusted_reference:
        adjusted_targets = {
            key: float(adjusted_reference[key]["reader_adjusted_clarity"])
            for key in items
        }

        def fit_continuous(mode: str) -> tuple[dict[str, object], dict[tuple[str, str], float]]:
            torch.manual_seed(args.seed)
            train_x = torch.stack([features(key, mode) for key in train_keys])
            target = torch.tensor(
                [adjusted_targets[key] for key in train_keys], dtype=torch.float32
            )
            feature_mean = train_x.mean(dim=0)
            feature_std = train_x.std(dim=0).clamp_min(0.1)
            normalized = (train_x - feature_mean) / feature_std
            weights = torch.zeros(normalized.shape[1], requires_grad=True)
            bias = torch.zeros((), requires_grad=True)
            optimizer = torch.optim.Adam([weights, bias], lr=args.learning_rate)
            for _ in range(args.steps):
                prediction = torch.sigmoid(normalized @ weights + bias)
                loss = torch.nn.functional.mse_loss(prediction, target) + (
                    args.l2 * weights.square().sum()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            def predict(keys: list[tuple[str, str]]) -> list[float]:
                matrix = torch.stack([features(key, mode) for key in keys])
                logits = (
                    (matrix - feature_mean) / feature_std @ weights.detach()
                    + bias.detach()
                )
                return torch.sigmoid(logits).tolist()

            predictions = predict(test_keys)
            return (
                {
                    "feature_mode": mode,
                    "weights": weights.detach().tolist(),
                    "bias": float(bias.detach()),
                    "normalization": {
                        "feature_mean": feature_mean.tolist(),
                        "feature_std": feature_std.tolist(),
                        "fit_split": "dev",
                    },
                    "target": "reader_adjusted_clarity",
                    "test_brier": mean(
                        (prediction - adjusted_targets[key]) ** 2
                        for key, prediction in zip(test_keys, predictions)
                    ),
                },
                dict(zip(test_keys, predictions)),
            )

        sensitivity_modes = {
            "selected_polarity": f"{selected_polarity}_finding_adjusted",
            "selected_plane": f"{selected_plane}_finding_adjusted",
            "final_plane": f"{final_plane}_finding_adjusted",
        }
        sensitivity_models = {}
        sensitivity_predictions = {}
        for name, mode in sensitivity_modes.items():
            sensitivity_models[name], sensitivity_predictions[name] = fit_continuous(mode)

        target_values = [adjusted_targets[key] for key in test_keys]
        selected_increment = paired_cluster_bootstrap_continuous(
            target_values,
            [sensitivity_predictions["selected_polarity"][key] for key in test_keys],
            [sensitivity_predictions["selected_plane"][key] for key in test_keys],
            test_clusters,
            args.bootstrap_draws,
            args.seed + 201,
        )
        early_vs_final_adjusted = paired_cluster_bootstrap_continuous(
            target_values,
            [sensitivity_predictions["final_plane"][key] for key in test_keys],
            [sensitivity_predictions["selected_plane"][key] for key in test_keys],
            test_clusters,
            args.bootstrap_draws,
            args.seed + 202,
        )
        adjusted_passed = positive_ci(selected_increment["brier_gain"]) and positive_ci(
            early_vs_final_adjusted["brier_gain"]
        )
        reader_adjusted_sensitivity = {
            "provided": True,
            "required_for_formal_authorization": True,
            "manifest_sha256": sha256_file(args.reader_adjusted_manifest),
            "target": "reader_adjusted_clarity",
            "layer_reselection_forbidden": True,
            "selected_layer_inherited_from_raw_dev_gate": selected_layer,
            "nuisance_features": "finding identity only; panel bias is already adjusted in the reference",
            "models": sensitivity_models,
            "selected_layer_commitment_increment": selected_increment,
            "selected_early_vs_final_claim_plane": early_vs_final_adjusted,
            "pass_rule": (
                "cluster-bootstrap Brier-gain CI must be above zero for both "
                "same-layer commitment increment and selected-early versus final"
            ),
            "passed": adjusted_passed,
        }

    def test_label(key: tuple[str, str]) -> int:
        return int(
            int(items[key]["positive_votes"])
            in {0, int(items[key]["reader_count"])}
        )

    def directional_causal_patch(
        keys: list[tuple[str, str]], clear_vote: int, disagreement_vote: int, seed: int
    ) -> dict[str, object]:
        disagreement = [key for key in keys if int(items[key]["positive_votes"]) == disagreement_vote]
        clear = [key for key in keys if int(items[key]["positive_votes"]) == clear_vote]

        def state(key: tuple[str, str], condition: str) -> str:
            record = items[key]["raw_record"]
            if condition == "baseline":
                final = str(record["measurement"]["final_layer"])
                return str(record["measurement"]["trajectory"][final]["baseline_state"])
            return str(record["measurement"]["activation_intervention"][condition]["state"])

        target_baseline = [
            float(state(key, "targeted") != "undetermined")
            - float(state(key, "baseline") != "undetermined")
            for key in disagreement
        ]
        target_random = [
            float(state(key, "targeted") != "undetermined")
            - float(state(key, "random_orthogonal") != "undetermined")
            for key in disagreement
        ]
        target_baseline_ci = cluster_bootstrap_mean(
            target_baseline, [key[1] for key in disagreement], args.bootstrap_draws, seed
        )
        target_random_ci = cluster_bootstrap_mean(
            target_random, [key[1] for key in disagreement], args.bootstrap_draws, seed + 1
        )
        expected = "refuted" if clear_vote == 0 else "supported"
        baseline_clear = mean(float(state(key, "baseline") == expected) for key in clear) if clear else None
        targeted_clear = mean(float(state(key, "targeted") == expected) for key in clear) if clear else None
        changes = [intervention_coordinate_changes(items[key]["raw_record"]) for key in keys]
        sign_flip_rate = mean(value["targeted_polarity_sign_flip"] for value in changes) if changes else 1.0
        norm_errors, random_target_cosines, random_polarity_cosines, target_polarity_cosines = [], [], [], []
        for key in keys:
            intervention = items[key]["raw_record"]["measurement"]["activation_intervention"]
            for condition in ("targeted", "random_orthogonal"):
                audit = intervention[condition]["norm_audit"]
                norm_errors.append(
                    abs(float(audit["matched_l2"]) - float(audit["original_l2"]))
                    / max(float(audit["original_l2"]), 1e-12)
                )
            random_target_cosines.append(float(intervention["random_orthogonal"]["absolute_cosine_with_target"]))
            random_polarity_cosines.append(float(intervention["random_orthogonal"]["absolute_cosine_with_polarity"]))
            target_polarity_cosines.append(abs(float(intervention["null_claim_plane"]["target_preserve_cosine_after"])))
        clear_drop = (
            None if baseline_clear is None or targeted_clear is None
            else baseline_clear - targeted_clear
        )
        passed = bool(
            target_baseline_ci["ci_high"] is not None
            and float(target_baseline_ci["ci_high"]) < 0.0
            and target_random_ci["ci_high"] is not None
            and float(target_random_ci["ci_high"]) < 0.0
            and clear_drop is not None and clear_drop <= 0.01
            and sign_flip_rate <= 0.01
            and max(norm_errors, default=float("inf")) <= 1e-5
            and max(random_target_cosines, default=float("inf")) <= 1e-5
            and max(random_polarity_cosines, default=float("inf")) <= 1e-5
            and max(target_polarity_cosines, default=float("inf")) <= 1e-5
        )
        return {
            "targeted_minus_baseline_disagreement_overcommitment": target_baseline_ci,
            "targeted_minus_random_disagreement_overcommitment": target_random_ci,
            "baseline_clear_accuracy": baseline_clear,
            "targeted_clear_accuracy": targeted_clear,
            "clear_accuracy_drop": clear_drop,
            "targeted_polarity_sign_flip_rate": sign_flip_rate,
            "maximum_norm_relative_error": max(norm_errors, default=None),
            "maximum_random_target_absolute_cosine": max(random_target_cosines, default=None),
            "maximum_random_polarity_absolute_cosine": max(random_polarity_cosines, default=None),
            "maximum_targeted_polarity_absolute_cosine": max(target_polarity_cosines, default=None),
            "passed": passed,
        }

    boundary_records = []
    boundary_ready = all(bool(items[key]["boundary_controls_complete"]) for key in items)
    if args.boundary_output is not None and not boundary_ready:
        raise ValueError(
            "boundary-output requires v8 raw rows with norm-matched null and "
            "activation-intervention controls"
        )
    direction_contracts = {
        "negative": {"votes": (0, 1), "bins": ("0/3", "1/3")},
        "positive": {"votes": (3, 2), "bins": ("3/3", "2/3")},
    }
    boundary_findings = (
        sorted({str(items[key]["finding"]) for key in items}) if boundary_ready else []
    )
    for finding_index, finding in enumerate(boundary_findings):
        for direction_index, (direction, contract) in enumerate(direction_contracts.items()):
            clear_vote, disagreement_vote = contract["votes"]
            dev_direction = [
                key for key in train_keys
                if str(items[key]["finding"]) == finding
                and int(items[key]["positive_votes"]) in {clear_vote, disagreement_vote}
            ]
            test_direction = [
                key for key in test_keys
                if str(items[key]["finding"]) == finding
                and int(items[key]["positive_votes"]) in {clear_vote, disagreement_vote}
            ]
            dev_labels = [test_label(key) for key in dev_direction]
            test_direction_labels = [test_label(key) for key in test_direction]
            counts = {
                f"{vote}/3": sum(int(int(items[key]["positive_votes"]) == vote) for key in test_direction)
                for vote in (clear_vote, disagreement_vote)
            }
            if len(set(dev_labels)) < 2 or len(set(test_direction_labels)) < 2:
                continue
            def dev_plane_auc(layer: int) -> float:
                value = auc(
                    dev_labels,
                    [test_logits[f"layer_{layer}_plane_reader_adjusted"][key] for key in dev_direction],
                )
                return -math.inf if value is None else value

            selected_direction_layer = max(
                layers[:-1], key=lambda layer: (dev_plane_auc(layer), -layer)
            )
            final_direction_layer = final_layer

            def strongest_control(layer: int) -> tuple[str, float]:
                values = []
                for control in CONTROL_NAMES:
                    mode = f"layer_{layer}_{control}_reader_adjusted"
                    score = auc(dev_labels, [test_logits[mode][key] for key in dev_direction])
                    values.append((control, -math.inf if score is None else score))
                return max(values, key=lambda value: (value[1], value[0]))

            early_control, early_control_dev_auc = strongest_control(selected_direction_layer)
            final_control, final_control_dev_auc = strongest_control(final_direction_layer)
            early_mode = f"layer_{selected_direction_layer}_plane_reader_adjusted"
            final_mode = f"layer_{final_direction_layer}_plane_reader_adjusted"
            early_control_mode = f"layer_{selected_direction_layer}_{early_control}_reader_adjusted"
            final_control_mode = f"layer_{final_direction_layer}_{final_control}_reader_adjusted"
            seed_base = args.seed + 5000 + finding_index * 20 + direction_index * 5
            early_final = paired_cluster_bootstrap_increment(
                test_direction_labels,
                [test_logits[final_mode][key] for key in test_direction],
                [test_logits[early_mode][key] for key in test_direction],
                [key[1] for key in test_direction],
                args.bootstrap_draws,
                seed_base,
            )["auroc_gain"]
            early_increment = paired_cluster_bootstrap_increment(
                test_direction_labels,
                [test_logits[early_control_mode][key] for key in test_direction],
                [test_logits[early_mode][key] for key in test_direction],
                [key[1] for key in test_direction],
                args.bootstrap_draws,
                seed_base + 1,
            )["auroc_gain"]
            final_increment = paired_cluster_bootstrap_increment(
                test_direction_labels,
                [test_logits[final_control_mode][key] for key in test_direction],
                [test_logits[final_mode][key] for key in test_direction],
                [key[1] for key in test_direction],
                args.bootstrap_draws,
                seed_base + 2,
            )["auroc_gain"]
            causal = directional_causal_patch(
                test_direction, clear_vote, disagreement_vote, seed_base + 3
            )
            sufficiently_sampled = min(counts.values()) >= args.min_test_per_class
            increment_widths = [
                float(metric["ci_high"]) - float(metric["ci_low"])
                for metric in (early_increment, final_increment)
                if metric["ci_high"] is not None and metric["ci_low"] is not None
            ]
            powered = bool(
                sufficiently_sampled
                and early_increment["valid_draws"] >= int(0.95 * args.bootstrap_draws)
                and final_increment["valid_draws"] >= int(0.95 * args.bootstrap_draws)
                and len(increment_widths) == 2
                and max(increment_widths) <= args.power_max_ci_width
            )
            boundary_records.append({
                "model_id": args.model_id,
                "finding": finding,
                "direction": direction,
                "direction_bins": list(contract["bins"]),
                "test_vote_bin_counts": counts,
                "minimum_test_per_vote_bin": args.min_test_per_class,
                "selected_early_layer": selected_direction_layer,
                "final_layer": final_direction_layer,
                "layer_selection_split": "dev_only",
                "early_minus_final_auroc": early_final,
                "increment_over_strongest_control": {
                    "early": {**early_increment, "control": early_control, "control_dev_auroc": early_control_dev_auc},
                    "final": {**final_increment, "control": final_control, "control_dev_auroc": final_control_dev_auc},
                },
                "all_preregistered_controls_present": True,
                "powered_for_margin": powered,
                "power_rule": {
                    "maximum_increment_ci_width": args.power_max_ci_width,
                    "observed_increment_ci_widths": increment_widths,
                },
                "causal_patch": causal,
                "causal_patch_passed": causal["passed"],
            })

    finding_results = {}
    for finding in sorted({str(items[key]["finding"]) for key in test_keys}):
        keys = [key for key in test_keys if str(items[key]["finding"]) == finding]
        labels = [test_label(key) for key in keys]
        vote_bin_counts = {
            f"{votes}/3": sum(
                int(int(items[key]["positive_votes"]) == votes) for key in keys
            )
            for votes in range(4)
        }
        class_counts = {
            "disagreement": labels.count(0),
            "unanimous": labels.count(1),
        }
        qualified = min(vote_bin_counts.values()) >= args.min_test_per_class
        record: dict[str, object] = {
            "test_items": len(keys),
            "test_class_counts": class_counts,
            "test_vote_bin_counts": vote_bin_counts,
            "qualified": qualified,
            "minimum_test_per_class": args.min_test_per_class,
            "qualification_rule": (
                "each of 0/3, 1/3, 2/3, and 3/3 must meet the minimum; "
                "pooling into unanimous/disagreement is insufficient"
            ),
        }
        if qualified:
            conditional_result = paired_cluster_bootstrap_increment(
                labels,
                [test_logits[selected_polarity_adjusted][key] for key in keys],
                [test_logits[selected_plane_adjusted][key] for key in keys],
                [key[1] for key in keys],
                args.bootstrap_draws,
                args.seed + 1000 + len(finding_results) * 2,
            )
            early_result = paired_cluster_bootstrap_increment(
                labels,
                [test_logits[final_plane_adjusted][key] for key in keys],
                [test_logits[selected_plane_adjusted][key] for key in keys],
                [key[1] for key in keys],
                args.bootstrap_draws,
                args.seed + 1001 + len(finding_results) * 2,
            )
            conditional_pass = positive_ci(conditional_result["auroc_gain"])
            early_pass = positive_ci(
                early_result["auroc_gain"], minimum=0.05
            )
            record.update(
                {
                    "selected_layer_commitment_increment": conditional_result,
                    "selected_early_vs_final_claim_plane": early_result,
                    "commitment_increment_passed": conditional_pass,
                    "early_minus_final_passed": early_pass,
                    "mechanism_passed": conditional_pass and early_pass,
                }
            )
        finding_results[finding] = record

    qualified_findings = [
        finding for finding, record in finding_results.items() if record["qualified"]
    ]
    passed_findings = [
        finding
        for finding in qualified_findings
        if finding_results[finding].get("mechanism_passed") is True
    ]
    majority_passed = bool(qualified_findings) and (
        len(passed_findings) > len(qualified_findings) / 2
    )

    conditional = comparisons["selected_layer_commitment_increment"]
    early_vs_final = comparisons["selected_early_vs_final_claim_plane"]
    final_test = methods[final_plane_adjusted]["test"]
    selected_test = methods[selected_plane_adjusted]["test"]

    clear_drop = float(final_test["clear_case_accuracy"]) - float(
        selected_test["clear_case_accuracy"]
    )
    omission_increase = float(selected_test["unanimous_omission_rate"]) - float(
        final_test["unanimous_omission_rate"]
    )
    formal_reference = all(
        row.get("formal_reference") is True
        and row.get("reference_source") == "vindr_reader_votes"
        and validated_reader_ids(row) is not None
        for row in manifest.values()
    )
    gates = {
        "formal_reader_reference": formal_reference,
        "reader_adjusted_sensitivity_passed": bool(
            reader_adjusted_sensitivity["passed"]
        ),
        "commitment_adds_conditional_auroc_ci_above_zero": positive_ci(
            conditional["auroc_gain"]
        ),
        "commitment_adds_conditional_brier_ci_above_zero": positive_ci(
            conditional["brier_gain"]
        ),
        "early_minus_final_auroc_ge_0.05_ci_above_zero": positive_ci(
            early_vs_final["auroc_gain"], minimum=0.05
        ),
        "clear_case_drop_le_0.01": clear_drop <= 0.01,
        "unanimous_omission_increase_le_0.01": omission_increase <= 0.01,
        "majority_qualified_findings_pass": majority_passed,
    }
    gates["measurement_authorized"] = all(gates.values())

    result = {
        "version": VERSION,
        "model_id": args.model_id,
        "formal_reference": formal_reference,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "raw_sha256": [sha256_file(path) for path in args.raw],
            "fit_split": "dev",
            "test_used_for_selection_or_fitting": False,
        },
        "layers": layers,
        "final_layer": final_layer,
        "selected_early_layer": selected_layer,
        "layer_selection_rule": (
            "maximum dev agreement AUROC among non-final reader/finding-adjusted "
            "Claim-Plane probes; ties use lower dev Brier then earlier layer; "
            "test is never consulted"
        ),
        "train_items": len(train_keys),
        "test_items": len(test_keys),
        "bootstrap_unit": "image_id",
        "bootstrap_draws": args.bootstrap_draws,
        "reader_effect_control": {
            "reader_identity_preserved": all(
                items[key]["reader_ids"] is not None for key in items
            ),
            "reader_vocab_fit_on_dev": reader_vocab,
            "finding_vocab_fit_on_dev": finding_vocab,
            "test_readers": test_readers,
            "test_findings": test_findings,
            "unseen_test_readers": unseen_test_readers,
            "unseen_test_findings": unseen_test_findings,
            "nuisance_features": "multi-hot rad_ID composition plus finding identity",
            "primary_increment_tests_reader_adjusted": True,
        },
        "reader_adjusted_sensitivity": reader_adjusted_sensitivity,
        "methods": methods,
        "heldout_increment_tests": comparisons,
        "per_finding_heldout_tests": finding_results,
        "directional_boundary_records": boundary_records,
        "directional_boundary_contract": {
            "positive_label": "reader_unanimous_clarity",
            "negative_direction": "0/3 versus 1/3",
            "positive_direction": "3/3 versus 2/3",
            "layer_selection": "per finding and direction on dev only",
            "strongest_control_selection": "per layer on dev only",
            "controls": list(CONTROL_NAMES),
        },
        "finding_majority_gate": {
            "minimum_test_per_class": args.min_test_per_class,
            "qualified_findings": qualified_findings,
            "passed_findings": passed_findings,
            "strict_majority_required": True,
            "passed": majority_passed,
        },
        "selected_clarity_calibrator": {
            "feature_mode": selected_plane,
            "selected_layer": selected_layer,
            "weights": methods[selected_plane]["weights"],
            "bias": methods[selected_plane]["bias"],
            "normalization": methods[selected_plane]["normalization"],
            "target": "reader_unanimity",
            "fit_split": "dev",
            "deployment_features_exclude_reader_identity": True,
        },
        "locked_test_predictions": [
            {
                "finding": key[0],
                "image_id": key[1],
                "reader_unanimous": test_label(key),
                "selected_plane_logit": test_logits[selected_plane][key],
                "selected_plane_reader_adjusted_logit": test_logits[
                    selected_plane_adjusted
                ][key],
                "selected_plane_probability": sigmoid(test_logits[selected_plane][key]),
                "same_layer_abs_polarity_logit": test_logits[selected_polarity][key],
                "final_plane_logit": test_logits[final_plane][key],
            }
            for key in test_keys
        ],
        "safety_deltas": {
            "selected_early_minus_final_clear_case_accuracy": -clear_drop,
            "selected_early_minus_final_unanimous_omission_rate": omission_increase,
        },
        "mechanism_gates": gates,
        "no_free_lunch_note": (
            "Any strictly monotone recalibration of one final scalar preserves its "
            "AUROC.  Authorization therefore requires held-out information from "
            "Claim-Plane commitment beyond same-layer absolute polarity, plus an "
            "early-versus-final gain; threshold movement alone is insufficient."
        ),
        "intervention_scope": (
            "The gate may change only definite versus undetermined wording; "
            "it never adds, deletes, or flips a claim."
        ),
        "claim_ceiling": (
            "A mechanism claim requires formal VinDr 0/1/2/3 vote bins, a "
            "held-out advantage over absolute polarity in a strict majority of "
            "qualified findings, a direction-consistent reader-adjusted clarity "
            "sensitivity analysis, an early-over-final advantage, a second "
            "model, and causal image/null controls."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.boundary_output is not None:
        args.boundary_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.boundary_output.with_suffix(args.boundary_output.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in boundary_records),
            encoding="utf-8",
        )
        temporary.replace(args.boundary_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
