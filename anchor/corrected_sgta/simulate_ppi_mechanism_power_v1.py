#!/usr/bin/env python3
"""CPU power gate for discriminating a gated prior from two artifacts.

The simulation is prospective and consumes no model outputs or clinical truth.
It asks whether the registered exact-parent logit assay can separate an
evidence-gated additive effect from (a) an unconditional cue trigger and (b) a
surface threshold/margin artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


VERSION = "ppi-mechanism-power-simulation-v1"
DEFAULT_ASSIGNMENT_AUDIT = Path("corrected_runs/ppi_source_assignment_v1/audit.json")
DEFAULT_OUTPUT = Path("corrected_runs/ppi_mechanism_power_v1")
MECHANISMS = ("evidence_gated", "unconditional_trigger", "margin_artifact")
CLARITY_LEVELS = np.asarray([0.15, 0.45, 0.75, 0.95])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_create(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def one_trial(
    mechanism: str,
    n_per_bucket: int,
    rng: np.random.Generator,
    *,
    beta: float = 0.30,
    logit_noise: float = 0.30,
) -> dict[str, Any]:
    clarity = np.repeat(CLARITY_LEVELS, n_per_bucket * 2 * 3)
    polarity = rng.choice((-1.0, 1.0), size=len(clarity))
    parent = polarity * (0.15 + 2.2 * clarity) + rng.normal(0, 0.35, len(clarity))
    plus_noise = rng.normal(0, logit_noise, len(clarity))
    minus_noise = rng.normal(0, logit_noise, len(clarity))
    zero_delta = rng.normal(0, logit_noise / np.sqrt(2), len(clarity))

    if mechanism == "evidence_gated":
        strength = beta * (1.0 - 0.70 * clarity)
        plus_delta = strength + plus_noise
        minus_delta = -strength + minus_noise
        plus_decision = parent + plus_delta > 0
        minus_decision = parent + minus_delta > 0
    elif mechanism == "unconditional_trigger":
        plus_delta = beta + plus_noise
        minus_delta = -beta + minus_noise
        plus_decision = parent + plus_delta > 0
        minus_decision = parent + minus_delta > 0
    elif mechanism == "margin_artifact":
        # The raw registered claim logit does not move.  Only an unregistered
        # downstream threshold is shifted, producing weak-margin answer flips.
        plus_delta = plus_noise
        minus_delta = minus_noise
        plus_decision = parent + beta > 0
        minus_decision = parent - beta > 0
    else:
        raise ValueError(mechanism)

    antisymmetric = (plus_delta - minus_delta) / 2
    regression = stats.linregress(clarity, antisymmetric)
    diagonal = stats.ttest_1samp(antisymmetric, popmean=0.0, alternative="greater")
    surface_flip = np.not_equal(plus_decision, minus_decision).astype(float)
    surface_regression = stats.linregress(clarity, surface_flip)

    # TOST-like conservative equivalence checks using 95% confidence intervals.
    symmetry_residual = plus_delta + minus_delta
    symmetry_sem = stats.sem(symmetry_residual)
    symmetry_ci = stats.t.interval(
        0.95, len(symmetry_residual) - 1, loc=float(np.mean(symmetry_residual)), scale=symmetry_sem
    )
    zero_sem = stats.sem(zero_delta)
    zero_ci = stats.t.interval(
        0.95, len(zero_delta) - 1, loc=float(np.mean(zero_delta)), scale=zero_sem
    )
    diagonal_pass = bool(np.mean(antisymmetric) >= 0.10 and diagonal.pvalue < 0.05)
    interaction_pass = bool(regression.slope <= -0.10 and regression.pvalue < 0.05)
    symmetry_pass = bool(symmetry_ci[0] > -0.05 and symmetry_ci[1] < 0.05)
    zero_pass = bool(zero_ci[0] > -0.05 and zero_ci[1] < 0.05)
    return {
        "diagonal_pass": diagonal_pass,
        "interaction_pass": interaction_pass,
        "symmetry_pass": symmetry_pass,
        "zero_pass": zero_pass,
        "mechanism_admitted": diagonal_pass and interaction_pass and symmetry_pass and zero_pass,
        "surface_weak_margin_pattern": bool(surface_regression.slope < 0 and surface_regression.pvalue < 0.05),
        "mean_antisymmetric_logit_effect": float(np.mean(antisymmetric)),
        "clarity_interaction_slope": float(regression.slope),
    }


def simulate(
    n_per_bucket_values: list[int], repetitions: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for n_per_bucket in n_per_bucket_values:
        for mechanism in MECHANISMS:
            trials = [one_trial(mechanism, n_per_bucket, rng) for _ in range(repetitions)]
            rows.append(
                {
                    "n_per_claim_reader_bucket_per_seed": n_per_bucket,
                    "mechanism": mechanism,
                    "repetitions": repetitions,
                    "mechanism_admission_rate": float(np.mean([row["mechanism_admitted"] for row in trials])),
                    "diagonal_detection_rate": float(np.mean([row["diagonal_pass"] for row in trials])),
                    "interaction_detection_rate": float(np.mean([row["interaction_pass"] for row in trials])),
                    "symmetry_equivalence_rate": float(np.mean([row["symmetry_pass"] for row in trials])),
                    "zero_equivalence_rate": float(np.mean([row["zero_pass"] for row in trials])),
                    "surface_weak_margin_pattern_rate": float(np.mean([row["surface_weak_margin_pattern"] for row in trials])),
                    "median_logit_effect": float(np.median([row["mean_antisymmetric_logit_effect"] for row in trials])),
                    "median_interaction_slope": float(np.median([row["clarity_interaction_slope"] for row in trials])),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-audit", type=Path, default=DEFAULT_ASSIGNMENT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-per-bucket", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=83021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment = json.loads(args.assignment_audit.read_text())
    if assignment.get("decision") != "CPU_FEASIBLE_ONLY" or assignment.get("gpu_authorized") is not False:
        raise ValueError("power simulation requires a completed, GPU-NO-GO assignment audit")
    rows = simulate(args.n_per_bucket, args.repetitions, args.seed)
    planned = next(row for row in rows if row["n_per_claim_reader_bucket_per_seed"] == 100 and row["mechanism"] == "evidence_gated")
    artifacts = [row for row in rows if row["n_per_claim_reader_bucket_per_seed"] == 100 and row["mechanism"] != "evidence_gated"]
    power_pass = planned["mechanism_admission_rate"] >= 0.80
    artifact_pass = all(row["mechanism_admission_rate"] <= 0.05 for row in artifacts)
    result = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "prospective synthetic assay power; no clinical or model evidence",
        "assignment_audit": str(args.assignment_audit.resolve()),
        "assignment_audit_sha256": _sha256(args.assignment_audit),
        "simulation_seed": args.seed,
        "repetitions": args.repetitions,
        "registered_admission_rule": {
            "mean_exact_parent_antisymmetric_logit_effect_min": 0.10,
            "clarity_interaction_slope_max": -0.10,
            "one_sided_alpha": 0.05,
            "plus_minus_residual_equivalence_margin": 0.05,
            "zero_equivalence_margin": 0.05,
        },
        "rows": rows,
        "planned_n_power_pass": power_pass,
        "artifact_false_admission_pass": artifact_pass,
        "decision": "POWER_GATE_PASS" if power_pass and artifact_pass else "POWER_GATE_FAIL",
        "gpu_authorized": False,
        "prohibited_inference": "simulation cannot establish a biological, clinical, or learned model mechanism",
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    _atomic_create(args.output / "power_audit.json", text)
    _atomic_create(
        args.output / "_COMPLETE.json",
        json.dumps(
            {
                "version": VERSION,
                "decision": result["decision"],
                "gpu_authorized": False,
                "power_audit_sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps({"decision": result["decision"], "gpu_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
