#!/usr/bin/env python3
"""Pre-registered cross-model cutoff for the n=8 calibration-state pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "metric-calibration-two-model-pilot-decision-v2"
MIN_OVERCOMMIT_RATE = 0.20
MAX_ENDPOINT_DRIFT = 0.01
MIN_DIRECT_OVERCOMMIT_RATE = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    cells = value["cells"]
    missing = cells["vision_coordinate:missing"]["patient_type_overcommitment_rate"]
    ambiguous = cells["vision_coordinate:header_unknown"]["patient_type_overcommitment_rate"]
    detector = cells["vision_coordinate:detector_only"]["patient_type_overcommitment_rate"]
    direct = value["direct"]["unidentifiable_unqualified_numeric_unit_rate"]
    oracle_drift = value["transformation"]["median_endpoint_rms_drift"].get("oracle_coordinate")
    oracle_slope = value["transformation"]["median_log_value_vs_log_scale_slope"].get("oracle_coordinate")
    neutral_signal = max(number for number in (missing, ambiguous) if number is not None)
    return {
        "analysis": str(path.resolve()),
        "analysis_sha256": sha256(path),
        "runtime_admissible": value["runtime_admissible"],
        "vision_missing_patient_type_overcommitment_rate": missing,
        "vision_header_unknown_patient_type_overcommitment_rate": ambiguous,
        "vision_detector_patient_type_overcommitment_rate": detector,
        "neutral_missing_or_ambiguous_signal": neutral_signal,
        "direct_unqualified_numeric_unit_rate": direct,
        "oracle_endpoint_drift": oracle_drift,
        "oracle_scale_slope": oracle_slope,
        "neutral_gate": neutral_signal >= MIN_OVERCOMMIT_RATE,
        "direct_gate": direct is not None and direct >= MIN_DIRECT_OVERCOMMIT_RATE,
        "factorization_gate": oracle_drift is not None and oracle_drift <= MAX_ENDPOINT_DRIFT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--huatuo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {"qwen_parent": summarize(args.qwen), "huatuo_medical": summarize(args.huatuo)}
    runtime = all(row["runtime_admissible"] for row in models.values())
    neutral = all(row["neutral_gate"] for row in models.values())
    direct = all(row["direct_gate"] for row in models.values())
    factorization = all(row["factorization_gate"] for row in models.values())
    expand = runtime and neutral and direct and factorization
    result = {
        "version": VERSION,
        "models": models,
        "thresholds": {
            "minimum_missing_or_header_unknown_patient_type_overcommitment_rate": MIN_OVERCOMMIT_RATE,
            "minimum_direct_unqualified_numeric_unit_rate": MIN_DIRECT_OVERCOMMIT_RATE,
            "maximum_oracle_endpoint_rms_drift": MAX_ENDPOINT_DRIFT,
        },
        "joint_gates": {
            "runtime": runtime,
            "neutral_overcommitment_cross_model": neutral,
            "direct_overcommitment_cross_model": direct,
            "dimensionless_endpoint_factorization": factorization,
        },
        "decision": "EXPAND_TO_N97_DIAGNOSTIC_ONLY" if expand else "STOP_AFTER_N8",
        "n97_authorized": expand,
        "oral_mainline_authorized": False,
        "gpu_authorized": expand,
        "collision_lock": "scale arithmetic/equivariance alone is covered by MedVision and FactCheXcker and cannot promote this branch",
        "broader_mechanism_missing": "a second metadata-defined quantity/modality and shared causal layer mechanism",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
