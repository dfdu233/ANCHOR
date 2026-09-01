#!/usr/bin/env python3
"""CPU-only, outcome-blind falsification for PPI v3.1.

This program uses only the official VinDr bounding-box annotations.  It never
loads a VLM, model score, activation, generated answer, or GPU runtime.  Its
purpose is deliberately narrower than a training launcher:

* freeze the exact R8/R9/R10 image panel and a source-label-only claim set;
* enumerate the complete balanced-sign randomization space before drawing r;
* use one optimizer family for +g, -g, and zero image-level assignments;
* audit multilabel and reader-bin nuisance balance; and
* preregister/simulate a reader-bin by fingerprint test that separates an
  additive cue intercept, an unconditional fingerprint trigger, a vote-margin
  artifact, and evidence-gated prior use.

The 5,501-image panel is odd.  Therefore a single audit-only image is removed
by a frozen hash rule *before* claim selection or fingerprint construction.
This is not an outcome-dependent deletion.  If any later feasibility gate
fails, the script writes a NO-GO decision and does not emit training authority.
Even a passing CPU result cannot authorize GPU work: PPI v3.1 additionally
requires a separately admitted natural-source bridge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shlex
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "ppi-v3.1-cpu-falsification-v1"
PANEL = ("R8", "R9", "R10")
NO_FINDING = "No finding"
SELECTION_SEED = 31082026
ASSIGNMENT_SEED = 73012026
POWER_SEED = 19052027
MIN_PER_READER_BIN = 100
MIN_MAJORITY_POSITIVE = 300
TARGET_G = 0.04
MAX_TARGET_ERROR = 0.008
MAX_NUISANCE_SMD = 0.025
MAX_GLOBAL_READER_BIN_RATE_GAP = 0.012
MAX_READER_POSITIVE_RATE_GAP = 0.012
OPTIMIZER_RESTARTS = 4
OPTIMIZER_STEPS = 80_000
POWER_DRAWS = 1_000
POWER_SEEDS = 5
POWER_IMAGE_SIGMA = 1.0
POWER_SEED_SIGMA = 0.045
POWER_EFFECT = 0.20
POWER_INTERACTION = 0.16
NATURAL_BRIDGE_STATUS = "failed_source_only_minimum_8_claim_gate_only_2_eligible"


class ContractError(RuntimeError):
    """A frozen pre-model contract cannot be satisfied."""


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    bins: tuple[int, ...]
    majority: tuple[int, ...]
    no_finding: int
    reader_positive_counts: tuple[int, ...]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hex(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def load_panel(path: Path) -> tuple[dict[str, dict[str, set[str]]], list[str]]:
    by_image: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "class_name", "rad_id"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ContractError(f"CSV missing {sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, start=2):
            image_id = str(row["image_id"]).strip()
            rad_id = str(row["rad_id"]).strip()
            claim = str(row["class_name"]).strip()
            if not image_id or not rad_id or not claim:
                raise ContractError(f"empty identifier on line {line}")
            by_image[image_id][rad_id].add(claim)
    selected = sorted(
        image_id for image_id, readers in by_image.items() if set(readers) == set(PANEL)
    )
    if len(selected) != 5501:
        raise ContractError(f"expected exact R8/R9/R10 panel of 5501, found {len(selected)}")
    for image_id in selected:
        for reader in PANEL:
            claims = by_image[image_id][reader]
            if NO_FINDING in claims and len(claims) != 1:
                raise ContractError(f"No finding co-occurs with positive claim: {image_id}/{reader}")
    return {image_id: by_image[image_id] for image_id in selected}, selected


def freeze_even_eligibility(
    panel: Mapping[str, Mapping[str, set[str]]], image_ids: Sequence[str]
) -> tuple[list[str], str]:
    """Freeze one audit-only unanimous-no-finding image by hash, before claims."""

    candidates = [
        image_id
        for image_id in image_ids
        if all(panel[image_id][reader] == {NO_FINDING} for reader in PANEL)
    ]
    if not candidates:
        raise ContractError("no unanimous-no-finding image available for odd-N audit holdout")
    holdout = min(
        candidates,
        key=lambda image_id: stable_hex(SELECTION_SEED, "odd-panel-audit-holdout", image_id),
    )
    eligible = [image_id for image_id in image_ids if image_id != holdout]
    if len(eligible) % 2:
        raise ContractError("hash-frozen eligibility remains odd")
    return eligible, holdout


def freeze_claims(
    panel: Mapping[str, Mapping[str, set[str]]], image_ids: Sequence[str]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    universe = sorted(
        {
            claim
            for image_id in image_ids
            for reader in PANEL
            for claim in panel[image_id][reader]
            if claim != NO_FINDING
        }
    )
    audit: dict[str, dict[str, Any]] = {}
    claims: list[str] = []
    for claim in universe:
        bins = Counter(
            sum(claim in panel[image_id][reader] for reader in PANEL)
            for image_id in image_ids
        )
        majority = bins[2] + bins[3]
        admitted = min(bins[value] for value in range(4)) >= MIN_PER_READER_BIN and majority >= MIN_MAJORITY_POSITIVE
        audit[claim] = {
            "reader_bin_counts": {str(value): bins[value] for value in range(4)},
            "majority_positive": majority,
            "admitted": admitted,
            "rule": f"all four bins >= {MIN_PER_READER_BIN} and majority positives >= {MIN_MAJORITY_POSITIVE}",
        }
        if admitted:
            claims.append(claim)
    if len(claims) < 8:
        raise ContractError(f"fewer than eight source-label-qualified claims: {claims}")
    # The fixed eight-claim dimension permits a complete order-8 Hadamard basis.
    claims = claims[:8]
    return claims, audit


def build_records(
    panel: Mapping[str, Mapping[str, set[str]]], image_ids: Sequence[str], claims: Sequence[str]
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for image_id in image_ids:
        bins = tuple(sum(claim in panel[image_id][reader] for reader in PANEL) for claim in claims)
        records.append(
            ImageRecord(
                image_id=image_id,
                bins=bins,
                majority=tuple(int(value >= 2) for value in bins),
                no_finding=int(all(panel[image_id][reader] == {NO_FINDING} for reader in PANEL)),
                reader_positive_counts=tuple(
                    len(panel[image_id][reader] - {NO_FINDING}) for reader in PANEL
                ),
            )
        )
    return records


def balanced_fingerprints(dimension: int) -> list[tuple[int, ...]]:
    if dimension % 2:
        raise ContractError("balanced-sign fingerprints require an even claim count")
    output = []
    for positive in itertools.combinations(range(dimension), dimension // 2):
        signs = [-1] * dimension
        for index in positive:
            signs[index] = 1
        output.append(tuple(signs))
    return output


def fingerprint_id(signs: Sequence[int]) -> str:
    return "".join("p" if value > 0 else "m" for value in signs)


def dot(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(left, right))


def feature_rows(records: Sequence[ImageRecord]) -> tuple[list[list[int]], list[str], int]:
    """Return target-first feature rows and names for a single optimizer family."""

    claim_count = len(records[0].majority)
    names = [f"target_majority::{index}" for index in range(claim_count)]
    names += [
        "nuisance::positive_claim_count",
        "nuisance::any_multilabel",
        "nuisance::three_plus_multilabel",
    ]
    names += [f"nuisance::reader_positive::{reader}" for reader in PANEL]
    names += [f"nuisance::aggregate_reader_bin::{value}" for value in range(4)]
    rows = []
    for record in records:
        majority_count = sum(record.majority)
        rows.append(
            list(record.majority)
            + [majority_count, int(majority_count >= 2), int(majority_count >= 3)]
            + list(record.reader_positive_counts)
            + [sum(value == reader_bin for value in record.bins) for reader_bin in range(4)]
        )
    return rows, names, claim_count


def assignment_deltas(signs: Sequence[int], features: Sequence[Sequence[int]]) -> list[int]:
    return [
        sum(sign * row[column] for sign, row in zip(signs, features))
        for column in range(len(features[0]))
    ]


def standardized_difference(delta: float, values: Sequence[int], n_half: int) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    if variance <= 1e-12:
        return 0.0 if abs(delta) <= 1e-12 else float("inf")
    return (delta / n_half) / math.sqrt(variance)


def optimize_assignment(
    records: Sequence[ImageRecord],
    target_signs: Sequence[int],
    *,
    seed: int,
    target_g: float = TARGET_G,
    restarts: int = OPTIMIZER_RESTARTS,
    steps: int = OPTIMIZER_STEPS,
) -> tuple[list[int], dict[str, Any]]:
    """One swap optimizer used unchanged for positive, negative, and zero arms."""

    features, names, claim_count = feature_rows(records)
    if len(target_signs) != claim_count or any(value not in {-1, 0, 1} for value in target_signs):
        raise ValueError("invalid target-sign vector")
    n = len(records)
    n_half = n // 2
    target_delta = [target_g * n_half * value for value in target_signs] + [0.0] * (len(names) - claim_count)
    columns = [[row[column] for row in features] for column in range(len(names))]
    # Target errors are expressed in probability-gap units.  Nuisances use SMD.
    scales = [n_half * MAX_TARGET_ERROR] * claim_count
    for column in range(claim_count, len(names)):
        values = columns[column]
        mean = sum(values) / n
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / max(1, n - 1))
        scales.append(max(1.0, n_half * MAX_NUISANCE_SMD * sd))

    def objective(deltas: Sequence[float]) -> float:
        return sum(((value - target) / scale) ** 2 for value, target, scale in zip(deltas, target_delta, scales))

    best: tuple[float, list[int], list[int], int] | None = None
    nf_indices = [index for index, record in enumerate(records) if record.no_finding]
    abn_indices = [index for index, record in enumerate(records) if not record.no_finding]
    if len(nf_indices) % 2 or len(abn_indices) % 2:
        raise ContractError("frozen odd-image removal did not make no-finding strata even")
    for restart in range(restarts):
        rng = random.Random(seed + 104729 * restart)
        assignment = [-1] * n
        for stratum in (nf_indices, abn_indices):
            shuffled = list(stratum)
            rng.shuffle(shuffled)
            for index in shuffled[: len(shuffled) // 2]:
                assignment[index] = 1
        deltas = assignment_deltas(assignment, features)
        score = objective(deltas)
        plus = [index for index, value in enumerate(assignment) if value == 1]
        minus = [index for index, value in enumerate(assignment) if value == -1]
        positions_plus = {index: position for position, index in enumerate(plus)}
        positions_minus = {index: position for position, index in enumerate(minus)}
        stalled = 0
        for step in range(steps):
            a = plus[rng.randrange(len(plus))]
            # Preserve exact no-finding balance by swapping within its stratum.
            candidates = nf_indices if records[a].no_finding else abn_indices
            while True:
                b = candidates[rng.randrange(len(candidates))]
                if assignment[b] == -1:
                    break
            proposal = [
                delta + 2 * (features[b][column] - features[a][column])
                for column, delta in enumerate(deltas)
            ]
            proposed_score = objective(proposal)
            temperature = max(0.002, 0.20 * (1.0 - step / steps))
            accept = proposed_score < score or rng.random() < math.exp(
                min(0.0, (score - proposed_score) / temperature)
            )
            if accept:
                pos_a = positions_plus.pop(a)
                pos_b = positions_minus.pop(b)
                plus[pos_a] = b
                minus[pos_b] = a
                positions_plus[b] = pos_a
                positions_minus[a] = pos_b
                assignment[a], assignment[b] = -1, 1
                deltas, score = proposal, proposed_score
                stalled = 0
            else:
                stalled += 1
            if score < 0.05:
                break
            if stalled > 20_000:
                break
        candidate = (score, list(assignment), list(deltas), step + 1)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    score, assignment, deltas, used_steps = best
    target_achieved = [deltas[index] / n_half for index in range(claim_count)]
    target_error = [target_achieved[index] - target_g * target_signs[index] for index in range(claim_count)]
    nuisance_smd = {
        names[index]: standardized_difference(deltas[index], columns[index], n_half)
        for index in range(claim_count, len(names))
    }
    audit = {
        "optimizer": "matched-stratum pair-swap simulated annealing",
        "objective": score,
        "steps": used_steps,
        "a_count": sum(value == 1 for value in assignment),
        "b_count": sum(value == -1 for value in assignment),
        "no_finding_a": sum(record.no_finding for record, value in zip(records, assignment) if value == 1),
        "no_finding_b": sum(record.no_finding for record, value in zip(records, assignment) if value == -1),
        "target_g": target_g,
        "achieved_g": target_achieved,
        "target_error": target_error,
        "max_abs_target_error": max(abs(value) for value in target_error),
        "nuisance_smd": nuisance_smd,
        "max_abs_nuisance_smd": max(abs(value) for value in nuisance_smd.values()),
    }
    return assignment, audit


def detailed_audit(
    records: Sequence[ImageRecord], assignment: Sequence[int], claims: Sequence[str]
) -> dict[str, Any]:
    n_half = len(records) // 2
    per_claim: dict[str, Any] = {}
    for claim_index, claim in enumerate(claims):
        bin_counts = {"A": Counter(), "B": Counter()}
        for record, sign in zip(records, assignment):
            bin_counts["A" if sign == 1 else "B"][record.bins[claim_index]] += 1
        per_claim[claim] = {
            "reader_bins_A": {str(value): bin_counts["A"][value] for value in range(4)},
            "reader_bins_B": {str(value): bin_counts["B"][value] for value in range(4)},
            "majority_prevalence_A": sum(bin_counts["A"][value] for value in (2, 3)) / n_half,
            "majority_prevalence_B": sum(bin_counts["B"][value] for value in (2, 3)) / n_half,
        }
    aggregate_bins = {"A": Counter(), "B": Counter()}
    reader_positive = {"A": Counter(), "B": Counter()}
    majority_count = {"A": Counter(), "B": Counter()}
    for record, sign in zip(records, assignment):
        group = "A" if sign == 1 else "B"
        aggregate_bins[group].update(record.bins)
        for reader, count in zip(PANEL, record.reader_positive_counts):
            reader_positive[group][reader] += count
        majority_count[group][sum(record.majority)] += 1
    denom_bins = n_half * len(claims)
    global_bin_gaps = {
        str(value): aggregate_bins["A"][value] / denom_bins - aggregate_bins["B"][value] / denom_bins
        for value in range(4)
    }
    total_reader = {
        group: sum(reader_positive[group].values()) for group in ("A", "B")
    }
    reader_rate_gaps = {
        reader: (reader_positive["A"][reader] - reader_positive["B"][reader]) / n_half
        for reader in PANEL
    }
    return {
        "per_claim": per_claim,
        "aggregate_reader_bins": {
            group: {str(value): aggregate_bins[group][value] for value in range(4)}
            for group in ("A", "B")
        },
        "global_reader_bin_rate_gap_A_minus_B": global_bin_gaps,
        "max_abs_global_reader_bin_rate_gap": max(abs(value) for value in global_bin_gaps.values()),
        "reader_positive_counts": {group: dict(reader_positive[group]) for group in ("A", "B")},
        "reader_positive_per_image_gap_A_minus_B": reader_rate_gaps,
        "max_abs_reader_positive_per_image_gap": max(abs(value) for value in reader_rate_gaps.values()),
        "total_positive_annotation_count": total_reader,
        "majority_positive_claim_count_histogram": {
            group: {str(key): value for key, value in sorted(majority_count[group].items())}
            for group in ("A", "B")
        },
    }


def cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(matrix)
    output = [[0.0] * n for _ in range(n)]
    for row in range(n):
        for column in range(row + 1):
            value = matrix[row][column] - sum(
                output[row][k] * output[column][k] for k in range(column)
            )
            if row == column:
                output[row][column] = math.sqrt(max(value, 1e-12))
            else:
                output[row][column] = value / output[column][column]
    return output


def cell_noise_cholesky(records: Sequence[ImageRecord], claim_count: int) -> list[list[float]]:
    cells = [(claim, reader_bin) for claim in range(claim_count) for reader_bin in range(4)]
    members = [
        {index for index, record in enumerate(records) if record.bins[claim] == reader_bin}
        for claim, reader_bin in cells
    ]
    covariance = []
    for left in members:
        row = []
        for right in members:
            row.append(POWER_IMAGE_SIGMA**2 * len(left & right) / (len(left) * len(right)))
        covariance.append(row)
    for index in range(len(covariance)):
        covariance[index][index] += 1e-10
    return cholesky(covariance)


def mean_and_t(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    se = math.sqrt(variance / len(values))
    return mean, mean / se if se > 1e-12 else (float("inf") if mean > 0 else float("-inf"))


def simulate_power(
    records: Sequence[ImageRecord], fingerprints: Sequence[Sequence[int]], draws: int = POWER_DRAWS
) -> dict[str, Any]:
    """Pre-outcome power for the frozen equal-cell reader-bin interaction."""

    claim_count = len(records[0].bins)
    lower = cell_noise_cholesky(records, claim_count)
    h = (1.0, 0.75, 0.45, 0.0)
    h_mean = sum(h) / 4
    h_centered = tuple(value - h_mean for value in h)
    h_scale = sum(value * value for value in h_centered) / 4
    mechanisms = {
        "additive_intercept": {"intercept": POWER_EFFECT, "main": 0.0, "interaction": 0.0, "margin": 0.0},
        "unconditional_fingerprint": {"intercept": 0.0, "main": POWER_EFFECT, "interaction": 0.0, "margin": 0.0},
        "vote_margin_artifact": {"intercept": 0.0, "main": 0.0, "interaction": 0.0, "margin": POWER_EFFECT},
        "evidence_gated_prior": {"intercept": 0.0, "main": POWER_EFFECT, "interaction": POWER_INTERACTION, "margin": 0.0},
        "strict_null": {"intercept": 0.0, "main": 0.0, "interaction": 0.0, "margin": 0.0},
    }
    for interaction in (0.04, 0.08, 0.12, 0.16, 0.20):
        mechanisms[f"interaction_curve_{interaction:.2f}"] = {
            "intercept": 0.0,
            "main": POWER_EFFECT,
            "interaction": interaction,
            "margin": 0.0,
        }
    critical = 2.262  # two-sided 0.05 t critical, df=9 (2 fingerprints x 5 runs).
    result: dict[str, Any] = {}
    for mechanism_index, (name, parameters) in enumerate(mechanisms.items()):
        rng = random.Random(POWER_SEED + 1009 * mechanism_index)
        counts = Counter()
        estimates: dict[str, list[float]] = defaultdict(list)
        for _ in range(draws):
            unit_main: list[float] = []
            unit_interaction: list[float] = []
            unit_intercept: list[float] = []
            unit_margin: list[float] = []
            for fingerprint in fingerprints:
                for _seed in range(POWER_SEEDS):
                    normals = [rng.gauss(0.0, 1.0) for _ in range(claim_count * 4)]
                    cell_noise = [
                        sum(lower[row][column] * normals[column] for column in range(row + 1))
                        for row in range(claim_count * 4)
                    ]
                    seed_intercept = rng.gauss(0.0, POWER_SEED_SIGMA)
                    seed_main = rng.gauss(0.0, POWER_SEED_SIGMA)
                    seed_interaction = rng.gauss(0.0, POWER_SEED_SIGMA)
                    values: list[tuple[int, int, float]] = []
                    for claim in range(claim_count):
                        sign = fingerprint[claim]
                        for reader_bin in range(4):
                            value = (
                                parameters["intercept"]
                                + seed_intercept
                                + (parameters["main"] + seed_main) * sign
                                + (parameters["interaction"] + seed_interaction) * sign * h_centered[reader_bin]
                                + parameters["margin"] * h_centered[reader_bin]
                                + cell_noise[4 * claim + reader_bin]
                            )
                            values.append((sign, reader_bin, value))
                    unit_intercept.append(sum(value for _, _, value in values) / len(values))
                    unit_main.append(sum(sign * value for sign, _, value in values) / len(values))
                    unit_interaction.append(
                        sum(sign * h_centered[reader_bin] * value for sign, reader_bin, value in values)
                        / (len(values) * h_scale)
                    )
                    unit_margin.append(
                        sum(h_centered[reader_bin] * value for _, reader_bin, value in values)
                        / (len(values) * h_scale)
                    )
            intercept, t_intercept = mean_and_t(unit_intercept)
            main, t_main = mean_and_t(unit_main)
            interaction, t_interaction = mean_and_t(unit_interaction)
            margin, t_margin = mean_and_t(unit_margin)
            detected = {
                "intercept": abs(t_intercept) > critical,
                "main": t_main > critical,
                "interaction": t_interaction > critical,
                "margin": abs(t_margin) > critical,
            }
            if detected["main"] and detected["interaction"] and not detected["margin"]:
                classification = "evidence_gated_prior"
            elif detected["main"] and not detected["interaction"]:
                classification = "unconditional_fingerprint"
            elif detected["margin"] and not detected["main"]:
                classification = "vote_margin_artifact"
            elif detected["intercept"] and not detected["main"] and not detected["interaction"]:
                classification = "additive_intercept"
            else:
                classification = "unresolved_or_null"
            counts[f"classified::{classification}"] += 1
            for diagnostic, flag in detected.items():
                counts[f"detected::{diagnostic}"] += int(flag)
            estimates["intercept"].append(intercept)
            estimates["main"].append(main)
            estimates["interaction"].append(interaction)
            estimates["margin"].append(margin)
        expected_class = (
            "evidence_gated_prior" if name.startswith("interaction_curve_") else name
        )
        result[name] = {
            "parameters": parameters,
            "draws": draws,
            "detection_rate": {
                key.split("::", 1)[1]: value / draws
                for key, value in counts.items()
                if key.startswith("detected::")
            },
            "classification_rate": {
                key.split("::", 1)[1]: value / draws
                for key, value in counts.items()
                if key.startswith("classified::")
            },
            "correct_classification_rate": (
                counts["classified::unresolved_or_null"] / draws
                if name == "strict_null"
                else counts[f"classified::{expected_class}"] / draws
            ),
            "mean_estimate": {key: sum(values) / len(values) for key, values in estimates.items()},
        }
    interaction_curve = [
        {
            "interaction_q_units": float(name.rsplit("_", 1)[1]),
            "evidence_gated_classification_power": result[name]["classification_rate"].get(
                "evidence_gated_prior", 0.0
            ),
            "interaction_detection_power": result[name]["detection_rate"].get(
                "interaction", 0.0
            ),
        }
        for name in result
        if name.startswith("interaction_curve_")
    ]
    interaction_curve.sort(key=lambda row: row["interaction_q_units"])
    mde = next(
        (
            row["interaction_q_units"]
            for row in interaction_curve
            if row["evidence_gated_classification_power"] >= 0.80
        ),
        None,
    )
    return {
        "preregistered_estimand": {
            "cell": "same-image A-minus-B q contrast, averaged within claim x R8/R9/R10 vote bin",
            "main": "equal-cell mean of r_c * d_c,b",
            "interaction": "OLS slope of r_c*d_c,b on centered h_b, h=(1,.75,.45,0)",
            "margin_negative_control": "OLS slope of d_c,b on centered h_b without fingerprint sign",
            "intercept_negative_control": "equal-cell unsigned mean d_c,b",
            "critical_value": critical,
            "experimental_unit": "fingerprint-specific matched training seed triplet",
            "claim_and_bin_weighting": "equal cell; image counts affect only cell-mean covariance",
        },
        "simulation": {
            "draws": draws,
            "training_seeds": POWER_SEEDS,
            "fingerprints": len(fingerprints),
            "image_cluster_sigma": POWER_IMAGE_SIGMA,
            "training_seed_random_effect_sigma": POWER_SEED_SIGMA,
            "main_effect_q_units": POWER_EFFECT,
            "interaction_q_units": POWER_INTERACTION,
            "results": result,
            "interaction_power_curve": interaction_curve,
            "grid_minimum_detectable_interaction_at_80pct_power": mde,
        },
    }


def choose_orthogonal_pair(
    fingerprints: Sequence[tuple[int, ...]], feasible: set[str]
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    pairs = [
        (left, right)
        for left in fingerprints
        for right in fingerprints
        if fingerprint_id(left) in feasible
        and fingerprint_id(right) in feasible
        and fingerprint_id(left) < fingerprint_id(right)
        and dot(left, right) == 0
    ]
    if not pairs:
        raise ContractError("R* contains no orthogonal fingerprint pair")
    pairs.sort(key=lambda pair: (fingerprint_id(pair[0]), fingerprint_id(pair[1])))
    rng = random.Random(SELECTION_SEED)
    index = rng.randrange(len(pairs))
    return pairs[index][0], pairs[index][1], len(pairs)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, all_ids = load_panel(args.labels_csv)
    eligible_ids, audit_holdout = freeze_even_eligibility(panel, all_ids)
    claims, claim_audit = freeze_claims(panel, eligible_ids)
    records = build_records(panel, eligible_ids, claims)
    fingerprints = balanced_fingerprints(len(claims))
    expected_r_star = math.comb(len(claims), len(claims) // 2)
    if len(fingerprints) != expected_r_star:
        raise ContractError("complete balanced-sign enumeration failed")

    contract = {
        "protocol_id": PROTOCOL_ID,
        "dataset": "VinDr-CXR train annotations, exact R8/R9/R10 panel",
        "model": "none (CPU-only source-label falsification)",
        "method": "PPI v3.1 complete-R* image-level matched pair-swap assignment",
        "input_csv": str(args.labels_csv.resolve()),
        "input_sha256": sha256_file(args.labels_csv),
        "seeds": {
            "selection": SELECTION_SEED,
            "assignment": ASSIGNMENT_SEED,
            "power": POWER_SEED,
        },
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "reader_panel": list(PANEL),
        "panel_images": len(all_ids),
        "odd_n_resolution": "one unanimous-no-finding image selected by frozen SHA-256 rule before claim selection",
        "audit_only_holdout": audit_holdout,
        "eligible_images": len(eligible_ids),
        "claims": claims,
        "claim_freeze_rule": {
            "min_each_reader_bin": MIN_PER_READER_BIN,
            "min_majority_positive": MIN_MAJORITY_POSITIVE,
            "take": "lexicographically first eight if more than eight qualify",
        },
        "fingerprint_law": "uniform over unordered orthogonal pairs in complete feasible balanced-sign R*",
        "fingerprint_space_size_before_optimizer": len(fingerprints),
        "target_g": TARGET_G,
        "assignment_unit": "unique image_id",
        "no_model_outputs_read": True,
        "gpu_authorized": False,
        "natural_bridge_status": NATURAL_BRIDGE_STATUS,
    }
    contract["fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "input_sha256": contract["input_sha256"],
                "panel": PANEL,
                "claims": claims,
                "seeds": contract["seeds"],
                "target_g": TARGET_G,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    atomic_json(args.output_dir / "frozen_contract.json", contract)
    atomic_json(args.output_dir / "claim_panel.json", {"claims": claims, "all_claim_audit": claim_audit})

    feasibility_rows: list[dict[str, Any]] = []
    cached: dict[str, tuple[list[int], dict[str, Any]]] = {}
    # Complements have identical feasibility.  The deterministic optimizer is
    # rerun for one canonical member; the other is audited as the exact -g arm.
    for signs in fingerprints:
        identifier = fingerprint_id(signs)
        complement = tuple(-value for value in signs)
        complement_id = fingerprint_id(complement)
        if complement_id in cached:
            assignment = [-value for value in cached[complement_id][0]]
            # The objective is sign-equivariant: complementing every image
            # assignment and target changes every delta's sign and preserves
            # the objective exactly.  Reuse this exact member of the same
            # optimizer family rather than introduce a second stochastic
            # search that could violate the mandated +/- complement.
            features, names, claim_count = feature_rows(records)
            deltas = assignment_deltas(assignment, features)
            achieved = [deltas[index] / (len(records) // 2) for index in range(claim_count)]
            columns = [[row[column] for row in features] for column in range(len(names))]
            nuisance = {
                names[index]: standardized_difference(
                    deltas[index], columns[index], len(records) // 2
                )
                for index in range(claim_count, len(names))
            }
            audit = {
                "optimizer": "sign-equivariant exact complement of matched-stratum pair-swap solution",
                "objective": None,
                "steps": 0,
                "a_count": sum(value == 1 for value in assignment),
                "b_count": sum(value == -1 for value in assignment),
                "no_finding_a": sum(record.no_finding for record, value in zip(records, assignment) if value == 1),
                "no_finding_b": sum(record.no_finding for record, value in zip(records, assignment) if value == -1),
                "target_g": TARGET_G,
                "achieved_g": achieved,
                "target_error": [achieved[index] - TARGET_G * signs[index] for index in range(claim_count)],
                "max_abs_target_error": max(abs(achieved[index] - TARGET_G * signs[index]) for index in range(claim_count)),
                "nuisance_smd": nuisance,
                "max_abs_nuisance_smd": max(abs(value) for value in nuisance.values()),
            }
        else:
            seed = ASSIGNMENT_SEED + int(identifier.replace("p", "1").replace("m", "0"), 2)
            assignment, audit = optimize_assignment(records, signs, seed=seed)
        detail = detailed_audit(records, assignment, claims)
        accepted = (
            audit["a_count"] == audit["b_count"] == len(records) // 2
            and audit["no_finding_a"] == audit["no_finding_b"]
            and audit["max_abs_target_error"] <= MAX_TARGET_ERROR
            and audit["max_abs_nuisance_smd"] <= MAX_NUISANCE_SMD
            and detail["max_abs_global_reader_bin_rate_gap"] <= MAX_GLOBAL_READER_BIN_RATE_GAP
            and detail["max_abs_reader_positive_per_image_gap"] <= MAX_READER_POSITIVE_RATE_GAP
        )
        cached[identifier] = (assignment, audit)
        feasibility_rows.append(
            {
                "fingerprint_id": identifier,
                "signs": list(signs),
                "accepted": accepted,
                "optimizer_audit": audit,
                "balance_audit": {
                    key: value for key, value in detail.items() if key != "per_claim"
                },
                "assignment_digest": hashlib.sha256(
                    "\n".join(
                        f"{record.image_id}:{sign}" for record, sign in zip(records, assignment)
                    ).encode()
                ).hexdigest(),
            }
        )
    feasible = {row["fingerprint_id"] for row in feasibility_rows if row["accepted"]}
    atomic_json(
        args.output_dir / "admissible_randomization_space.json",
        {
            "complete_enumeration": True,
            "balanced_sign_vectors": len(fingerprints),
            "accepted_R_star": len(feasible),
            "acceptance_thresholds": {
                "max_target_error": MAX_TARGET_ERROR,
                "max_nuisance_smd": MAX_NUISANCE_SMD,
                "max_global_reader_bin_rate_gap": MAX_GLOBAL_READER_BIN_RATE_GAP,
                "max_reader_positive_per_image_gap": MAX_READER_POSITIVE_RATE_GAP,
            },
            "rows": feasibility_rows,
        },
    )
    if len(feasible) < 2:
        decision = {
            "decision": "NO-GO",
            "reason": "exact precomputed R* has fewer than two feasible fingerprints",
            "gpu_authorized": False,
            "natural_bridge_status": NATURAL_BRIDGE_STATUS,
        }
        atomic_json(args.output_dir / "decision.json", decision)
        return decision

    first, second, orthogonal_pair_count = choose_orthogonal_pair(fingerprints, feasible)
    selected = [first, second]
    assignments_summary: dict[str, Any] = {}
    for fingerprint_index, signs in enumerate(selected, start=1):
        identifier = fingerprint_id(signs)
        plus_assignment, plus_optimizer = cached[identifier]
        minus_assignment = [-value for value in plus_assignment]
        zero_assignment, zero_optimizer = optimize_assignment(
            records,
            [0] * len(claims),
            seed=ASSIGNMENT_SEED + 10_000 + fingerprint_index,
        )
        arms = {
            "plus": (plus_assignment, plus_optimizer),
            "minus": (minus_assignment, cached[fingerprint_id(tuple(-value for value in signs))][1]),
            "zero": (zero_assignment, zero_optimizer),
        }
        assignments_summary[identifier] = {}
        for arm, (assignment, optimizer_audit) in arms.items():
            detail = detailed_audit(records, assignment, claims)
            path = args.output_dir / f"assignment_{fingerprint_index}_{identifier}_{arm}.jsonl"
            atomic_jsonl(
                path,
                (
                    {
                        "image_id": record.image_id,
                        "shell": "A" if sign == 1 else "B",
                        "reader_bins": {claim: record.bins[index] for index, claim in enumerate(claims)},
                    }
                    for record, sign in zip(records, assignment)
                ),
            )
            assignments_summary[identifier][arm] = {
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "optimizer_audit": optimizer_audit,
                "detailed_balance": detail,
            }
        if any(minus_assignment[index] != -plus_assignment[index] for index in range(len(records))):
            raise AssertionError("minus arm is not exact complement")
    atomic_json(
        args.output_dir / "selected_assignment_audit.json",
        {
            "fingerprints": [
                {"id": fingerprint_id(value), "signs": list(value)} for value in selected
            ],
            "dot_product": dot(first, second),
            "orthogonal_feasible_pair_count": orthogonal_pair_count,
            "selection_seed": SELECTION_SEED,
            "uniform_pair_index_law": True,
            "arms": assignments_summary,
        },
    )
    power = simulate_power(records, selected, draws=args.power_draws)
    atomic_json(args.output_dir / "reader_bin_interaction_power.json", power)
    cpu_pass = (
        len(feasible) == len(fingerprints)
        and power["simulation"]["results"]["evidence_gated_prior"]["correct_classification_rate"] >= 0.80
        and power["simulation"]["results"]["unconditional_fingerprint"]["classification_rate"].get("evidence_gated_prior", 0.0) <= 0.05
        and power["simulation"]["results"]["vote_margin_artifact"]["classification_rate"].get("evidence_gated_prior", 0.0) <= 0.05
    )
    decision = {
        "cpu_assignment_and_power_gate": "PASS" if cpu_pass else "NO-GO",
        "decision": "GPU-NO-GO",
        "reason": (
            "CPU model-organism design is feasible, but the independent natural-source bridge gate failed (2 eligible claims < 8)."
            if cpu_pass
            else "CPU exact-randomization or mechanism-discrimination gate failed; natural bridge also failed."
        ),
        "R_star_size": len(feasible),
        "complete_balanced_space_size": len(fingerprints),
        "natural_bridge_status": NATURAL_BRIDGE_STATUS,
        "gpu_authorized": False,
        "prohibited_inference": "These source-label-only assignments cannot explain any natural medical checkpoint or authorize training.",
    }
    atomic_json(args.output_dir / "decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("corrected_runs/ppi_v31"))
    parser.add_argument("--power-draws", type=int, default=POWER_DRAWS)
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
