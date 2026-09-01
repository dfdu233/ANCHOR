#!/usr/bin/env python3
"""No-leak CPU feasibility audit for randomized source two-plane assignments.

This program is deliberately upstream of images, models, VinDr, and GPUs.  It
uses only frozen PubMedVision source text labels and nuisance fields.  Its
output can establish assignment feasibility, never clinical validity or GPU
authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from anchor.corrected_sgta.build_pubmedvision_source_semantic_admission_v1 import (
    DEFAULT_ALIGNMENT,
    DEFAULT_SOURCE_INDEX,
    load_source_index,
    scan_stage,
    sha256_file,
)
from anchor.corrected_sgta.build_pubmedvision_source_semantic_admission_v3 import (
    PRIMARY_DOMAIN,
    response_domain,
)


VERSION = "ppi-source-two-plane-assignment-v1"
DEFAULT_SEMANTIC_DIR = Path(
    "/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr_v2/"
    "source_semantic_admission_v3_5"
)
DEFAULT_OUTPUT = Path("corrected_runs/ppi_source_assignment_v1")
BITS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
MIN_PER_STATE = 20
CROSS_BOUND = 0.05
TOKEN_FRACTION = 0.01
TIE_TOLERANCE = 0.01
ZERO_TOLERANCE = 0.01


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_create(path: Path, text: str) -> None:
    """Create a write-once artifact; identical reruns are harmless."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != text:
            raise
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SourceUnit:
    response_unit_id: str
    source_group: str
    response: str
    word_count: int
    modality: str
    length_bin: str
    archive: str
    labels: Mapping[str, str]


def source_labels(states: Mapping[str, str], claim: str) -> tuple[int, int]:
    """Return source polarity and linguistic-definiteness labels.

    Uncertain and unmentioned do not enter the polarity contrast.  Unmentioned
    does not enter the definiteness contrast.  This is source semantics only.
    """

    state = states.get(claim, "unmentioned")
    y_mu = 1 if state == "positive" else -1 if state == "negative" else 0
    y_kappa = 1 if state in {"positive", "negative"} else -1 if state == "uncertain" else 0
    return y_mu, y_kappa


def _length_bin(words: int) -> str:
    lower = min(words // 25, 8) * 25
    return f"{lower:03d}+" if lower == 200 else f"{lower:03d}-{lower + 24:03d}"


def load_units(
    semantic_dir: Path,
    source_index_path: Path,
    alignment_path: Path,
) -> tuple[list[SourceUnit], list[str], dict[str, Any]]:
    source_rows, source_stats = load_source_index(source_index_path)
    scanned, scan_stats = scan_stage(alignment_path, "alignment", source_rows, streaming=False)
    selected = [
        unit
        for unit in scanned
        if unit["source_split"] == "source_train" and response_domain(unit) == PRIMARY_DOMAIN
    ]
    ids = [str(unit["response_unit_id"]) for unit in selected]
    groups = [str(unit["source_group"]) for unit in selected]
    if len(ids) != len(set(ids)) or len(groups) != len(set(groups)):
        raise ValueError("assignment units must have unique response IDs and source groups")
    if len(selected) % 4:
        raise ValueError(f"unit count must be divisible by four, got {len(selected)}")

    eligible_payload = json.loads((semantic_dir / "eligible_claims.json").read_text())
    automatically_eligible = {
        str(row["claim_id"])
        for row in eligible_payload
        if bool(row.get("automatic_count_eligible"))
    }
    selected_ids = set(ids)
    states: dict[str, dict[str, str]] = defaultdict(dict)
    with (semantic_dir / "mention_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            unit_id = str(row["response_unit_id"])
            if unit_id in selected_ids:
                states[unit_id][str(row["finding"])] = str(row["assistant_state"])

    counts: dict[str, Counter[str]] = {}
    claims = []
    for claim in sorted(automatically_eligible):
        count = Counter(states[unit_id].get(claim, "unmentioned") for unit_id in ids)
        counts[claim] = count
        if all(count[state] >= MIN_PER_STATE for state in ("positive", "negative", "uncertain")):
            if count["positive"] + count["negative"] >= MIN_PER_STATE:
                claims.append(claim)
    if len(claims) != 2:
        raise ValueError(f"v1 requires exactly two strictly eligible claims, got {claims}")

    units = []
    for unit in selected:
        response = " ".join(str(unit["assistant_response"]).split())
        words = len(response.split())
        images = list(unit["matched_source_images"])
        archives = {
            Path(str(source_rows[image].get("archive", "unknown"))).name for image in images
        }
        archive = next(iter(archives)) if len(archives) == 1 else "multi_archive"
        units.append(
            SourceUnit(
                response_unit_id=str(unit["response_unit_id"]),
                source_group=str(unit["source_group"]),
                response=response,
                word_count=words,
                modality=str(unit.get("vqa_modality") or "unknown"),
                length_bin=_length_bin(words),
                archive=archive,
                labels={claim: states[str(unit["response_unit_id"])].get(claim, "unmentioned") for claim in claims},
            )
        )
    return units, claims, {
        "source_index_stats": source_stats,
        "alignment_scan_stats": scan_stats,
        "strict_state_counts": {claim: dict(sorted(counts[claim].items())) for claim in claims},
    }


class ConstraintBuilder:
    def __init__(self, size: int):
        self.size = size
        self.rows: list[dict[int, float]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, values: Mapping[int, float], lower: float, upper: float) -> None:
        self.rows.append(dict(values))
        self.lower.append(lower)
        self.upper.append(upper)

    def matrix(self) -> LinearConstraint:
        rr, cc, vv = [], [], []
        for row_index, values in enumerate(self.rows):
            for column, value in values.items():
                if value:
                    rr.append(row_index)
                    cc.append(column)
                    vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), self.size)).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def _x_index(unit: int, combo: int) -> int:
    return 4 * unit + combo


def _base_constraints(units: Sequence[SourceUnit], extra_variables: int = 1) -> ConstraintBuilder:
    n = len(units)
    builder = ConstraintBuilder(4 * n + extra_variables)
    for i in range(n):
        builder.add({_x_index(i, k): 1.0 for k in range(4)}, 1.0, 1.0)
    target = n // 4
    for k in range(4):
        builder.add({_x_index(i, k): 1.0 for i in range(n)}, target, target)

    nuisance_values = {
        "modality": [unit.modality for unit in units],
        "length_bin": [unit.length_bin for unit in units],
        "archive": [unit.archive for unit in units],
    }
    for values in nuisance_values.values():
        for category in sorted(set(values)):
            members = [i for i, value in enumerate(values) if value == category]
            lower, upper = math.floor(len(members) / 4), math.ceil(len(members) / 4)
            for k in range(4):
                builder.add({_x_index(i, k): 1.0 for i in members}, lower, upper)

    total_words = sum(unit.word_count for unit in units)
    token_target = total_words / 4
    token_tolerance = max(5.0, TOKEN_FRACTION * token_target)
    for k in range(4):
        builder.add(
            {_x_index(i, k): float(unit.word_count) for i, unit in enumerate(units)},
            token_target - token_tolerance,
            token_target + token_tolerance,
        )
    return builder


def _labels(units: Sequence[SourceUnit], claim: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [source_labels(unit.labels, claim) for unit in units]
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def _contrast_row(values: np.ndarray, bit_position: int, sign: int = 1) -> dict[int, float]:
    row: dict[int, float] = {}
    for i, value in enumerate(values):
        if not value:
            continue
        for k, bits in enumerate(BITS):
            row[_x_index(i, k)] = float(sign * bits[bit_position] * value)
    return row


def _solve(c: np.ndarray, constraints: ConstraintBuilder, variable_lower: np.ndarray, variable_upper: np.ndarray):
    result = milp(
        c=c,
        integrality=np.r_[np.ones(len(c) - 1, dtype=int), 0],
        bounds=Bounds(variable_lower, variable_upper),
        constraints=constraints.matrix(),
        options={"time_limit": 180, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status} message={result.message}")
    return result


def _decode(x: np.ndarray, n: int) -> list[int]:
    matrix = x[: 4 * n].reshape(n, 4)
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("non-one-hot assignment returned")
    return [int(value) for value in np.argmax(matrix, axis=1)]


def solve_plus(
    units: Sequence[SourceUnit],
    claims: Sequence[str],
    experiment: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    n = len(units)
    t_index = 4 * n
    pairing = {
        1: {"r_mu": (1, 1), "r_kappa": (1, -1)},
        2: {"r_mu": (1, -1), "r_kappa": (1, 1)},
    }[experiment]
    builder = _base_constraints(units)
    for claim_index, claim in enumerate(claims):
        y_mu, y_kappa = _labels(units, claim)
        n_mu, n_kappa = int(np.count_nonzero(y_mu)), int(np.count_nonzero(y_kappa))
        mean_row = _contrast_row(y_mu, 0, pairing["r_mu"][claim_index])
        mean_row[t_index] = -float(n_mu)
        builder.add(mean_row, 0.0, np.inf)
        precision_row = _contrast_row(y_kappa, 1, pairing["r_kappa"][claim_index])
        precision_row[t_index] = -float(n_kappa)
        builder.add(precision_row, 0.0, np.inf)
        builder.add(_contrast_row(y_mu, 1), -CROSS_BOUND * n_mu, CROSS_BOUND * n_mu)
        builder.add(_contrast_row(y_kappa, 0), -CROSS_BOUND * n_kappa, CROSS_BOUND * n_kappa)

    lower = np.r_[np.zeros(4 * n), -1.0]
    upper = np.ones(4 * n + 1)
    objective = np.zeros(4 * n + 1)
    objective[t_index] = -1.0
    optimum = _solve(objective, builder, lower, upper)
    optimum_t = float(optimum.x[t_index])

    # Freeze near-optimal scientific signal, then randomize among valid designs.
    # Reuse every existing constraint and add a near-optimal lower bound on t.
    tie_builder = builder
    tie_builder.add({t_index: 1.0}, optimum_t - TIE_TOLERANCE, np.inf)
    rng = np.random.default_rng(seed + 1009 * experiment)
    tie_objective = np.r_[rng.uniform(-1.0, 1.0, 4 * n), 0.0]
    tied = _solve(tie_objective, tie_builder, lower, upper)
    assignment = _decode(tied.x, n)
    return assignment, {
        "experiment": experiment,
        "pairing": pairing,
        "optimal_min_target_contrast": optimum_t,
        "realized_t_variable": float(tied.x[t_index]),
        "solver_status": int(tied.status),
        "solver_message": str(tied.message),
    }


def solve_zero(
    units: Sequence[SourceUnit], claims: Sequence[str], seed: int
) -> tuple[list[int], dict[str, Any]]:
    n = len(units)
    z_index = 4 * n
    builder = _base_constraints(units)
    for claim in claims:
        y_mu, y_kappa = _labels(units, claim)
        for values in (y_mu, y_kappa):
            denominator = int(np.count_nonzero(values))
            for bit_position in (0, 1):
                row = _contrast_row(values, bit_position)
                row[z_index] = -float(denominator)
                builder.add(row, -np.inf, 0.0)
                negative = {column: -value for column, value in row.items() if column != z_index}
                negative[z_index] = -float(denominator)
                builder.add(negative, -np.inf, 0.0)
    lower = np.zeros(4 * n + 1)
    upper = np.ones(4 * n + 1)
    objective = np.zeros(4 * n + 1)
    objective[z_index] = 1.0
    optimum = _solve(objective, builder, lower, upper)
    optimum_z = float(optimum.x[z_index])
    builder.add({z_index: 1.0}, -np.inf, optimum_z + ZERO_TOLERANCE)
    rng = np.random.default_rng(seed + 7919)
    tied = _solve(np.r_[rng.uniform(-1.0, 1.0, 4 * n), 0.0], builder, lower, upper)
    return _decode(tied.x, n), {
        "optimal_max_absolute_contrast": optimum_z,
        "realized_z_variable": float(tied.x[z_index]),
        "solver_status": int(tied.status),
        "solver_message": str(tied.message),
    }


def assignment_metrics(
    units: Sequence[SourceUnit], claims: Sequence[str], assignment: Sequence[int]
) -> dict[str, Any]:
    u = np.asarray([BITS[k][0] for k in assignment])
    v = np.asarray([BITS[k][1] for k in assignment])
    result: dict[str, Any] = {
        "combo_counts": dict(sorted(Counter(assignment).items())),
        "word_mass_by_combo": [
            int(sum(unit.word_count for unit, combo in zip(units, assignment) if combo == k))
            for k in range(4)
        ],
        "claims": {},
    }
    for claim in claims:
        y_mu, y_kappa = _labels(units, claim)
        n_mu, n_kappa = np.count_nonzero(y_mu), np.count_nonzero(y_kappa)
        result["claims"][claim] = {
            "u_mu": float(np.dot(u, y_mu) / n_mu),
            "v_mu": float(np.dot(v, y_mu) / n_mu),
            "u_kappa": float(np.dot(u, y_kappa) / n_kappa),
            "v_kappa": float(np.dot(v, y_kappa) / n_kappa),
        }
    return result


def build_artifact(
    units: Sequence[SourceUnit], claims: Sequence[str], seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for seed in seeds:
        zero, zero_solver = solve_zero(units, claims, seed)
        for experiment in (1, 2):
            plus, plus_solver = solve_plus(units, claims, experiment, seed)
            minus = [3 - combo for combo in plus]
            arms = {"plus": plus, "minus": minus, "zero": zero}
            for arm, assignment in arms.items():
                metrics = assignment_metrics(units, claims, assignment)
                run_rows.append(
                    {
                        "seed": seed,
                        "experiment": experiment,
                        "arm": arm,
                        "metrics": metrics,
                        "solver": plus_solver if arm in {"plus", "minus"} else zero_solver,
                    }
                )
                for unit, combo in zip(units, assignment):
                    assignment_rows.append(
                        {
                            "version": VERSION,
                            "seed": seed,
                            "experiment": experiment,
                            "arm": arm,
                            "response_unit_id": unit.response_unit_id,
                            "source_group": unit.source_group,
                            "combo": combo,
                            "s_mu": BITS[combo][0],
                            "s_kappa": BITS[combo][1],
                            "response_word_count": unit.word_count,
                            "response_sha256": _sha256_text(unit.response),
                        }
                    )
    return assignment_rows, run_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[83017, 83018, 83019])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    units, claims, load_stats = load_units(args.semantic_dir, args.source_index, args.alignment)
    assignments, runs = build_artifact(units, claims, args.seeds)
    assignments_text = "".join(_canonical_json(row) + "\n" for row in assignments)
    assignments_hash = _sha256_text(assignments_text)
    audit = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "docs/PROVENANCE_TWO_PLANE_BINDING_PROTOCOL_V3_2.md",
        "scope": "source-only automatic-label assignment feasibility",
        "unit_count": len(units),
        "eligible_claims": list(claims),
        "seeds": list(args.seeds),
        "runs": runs,
        "load_stats": load_stats,
        "source_hashes": {
            "source_index": sha256_file(args.source_index),
            "alignment": sha256_file(args.alignment),
            "eligible_claims": sha256_file(args.semantic_dir / "eligible_claims.json"),
            "mention_records": sha256_file(args.semantic_dir / "mention_records.jsonl"),
            "code": sha256_file(Path(__file__)),
        },
        "assignments_sha256": assignments_hash,
        "human_extractor_admitted": False,
        "clinical_null_admitted": False,
        "vin_dr_consumed": False,
        "model_consumed": False,
        "gpu_consumed": False,
        "gpu_authorized": False,
        "decision": "CPU_FEASIBLE_ONLY",
        "prohibited_inference": "does not establish clinical validity, model effect, or training authorization",
    }
    audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    complete = {
        "version": VERSION,
        "decision": audit["decision"],
        "gpu_authorized": False,
        "assignments_sha256": assignments_hash,
        "audit_sha256": _sha256_text(audit_text),
    }
    _atomic_create(args.output / "assignments.jsonl", assignments_text)
    _atomic_create(args.output / "audit.json", audit_text)
    _atomic_create(args.output / "_COMPLETE.json", json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
