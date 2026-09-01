#!/usr/bin/env python3
"""Fail-closed authorization for Reader-Calibrated Commitment Projection.

This is a scientific admission gate, not a result aggregator.  It binds one
model to three locked artifacts: directional visual admission, observational
clarity erasure, and the reader-adjusted clarity measurement gate.  Outcome
gates (causal controls, matched coverage, and omission safety) remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


VERSION = "reader-grounded-projection-authorization-v3-boundary-gated"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def nested_bool(row: Mapping[str, object], group: str, key: str) -> bool:
    values = row.get(group)
    return bool(isinstance(values, Mapping) and values.get(key) is True)


def authorization_fingerprint(payload: Mapping[str, object]) -> str:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"fingerprint", "command"}
    }
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_authorization(
    model_id: str,
    directional: Mapping[str, object],
    tetrad: Mapping[str, object],
    clarity: Mapping[str, object],
    boundary: Mapping[str, object],
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    model_gate = boundary.get("model_gate")
    current_model_gate = (
        model_gate.get(model_id, {}) if isinstance(model_gate, Mapping) else {}
    )
    checks = {
        "directional_model_matches": directional.get("model_id") == model_id,
        "tetrad_model_matches": tetrad.get("model_id") == model_id,
        "clarity_model_matches": clarity.get("model_id") == model_id,
        "directional_is_locked_test": (
            directional.get("experiment_split") == "test"
            and directional.get("test_layer_selected_without_test_labels") is True
        ),
        "directional_formal_reader_reference": directional.get("formal_reference")
        is True,
        "directional_admission_authorized": nested_bool(
            directional, "mechanism_gates", "directional_admission_authorized"
        ),
        "tetrad_formal_reader_reference": tetrad.get("formal_reference") is True,
        "observational_erasure_authorized": nested_bool(
            tetrad, "mechanism_gates", "observational_erasure_authorized"
        ),
        "clarity_formal_reader_reference": clarity.get("formal_reference") is True,
        "clarity_measurement_authorized": nested_bool(
            clarity, "mechanism_gates", "measurement_authorized"
        ),
        "global_two_model_method_branch_authorized": boundary.get(
            "method_branch_authorized"
        ) is True,
        "current_model_strict_finding_majority": (
            isinstance(current_model_gate, Mapping)
            and current_model_gate.get("strict_majority") is True
        ),
    }
    authorized = all(checks.values())
    payload: dict[str, object] = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "reader_grounded_projection_authorized": authorized,
        "checks": checks,
        "artifacts": dict(artifact_hashes),
        "scope": (
            "Permission to implement and evaluate the conditional two-coordinate "
            "projection only after the preregistered two-model Early-erasure "
            "boundary gate. It does not authorize claim exchange or efficacy claims."
        ),
        "remaining_outcome_gates": [
            "polarity-preserving causal activation control",
            "disagreement overcommitment relative reduction >= 20%",
            "clear-case loss <= 0.01",
            "reader-distribution Brier relative improvement >= 5%",
            "matched coverage and omission safety",
            "replication in a second admitted model",
        ],
    }
    payload["fingerprint"] = authorization_fingerprint(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--directional-admission", type=Path, required=True)
    parser.add_argument("--tetrad-erasure", type=Path, required=True)
    parser.add_argument("--clarity-gate", type=Path, required=True)
    parser.add_argument("--boundary-classification", type=Path, required=True)
    parser.add_argument("--support-calibrator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_paths = {
        "directional_admission": args.directional_admission,
        "tetrad_erasure": args.tetrad_erasure,
        "clarity_gate": args.clarity_gate,
        "boundary_classification": args.boundary_classification,
    }
    paths = {**report_paths, "support_calibrator": args.support_calibrator}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"authorization artifacts do not exist: {missing}")
    reports = {name: load_json(path) for name, path in report_paths.items()}
    artifact_hashes = {
        name: sha256_file(path) for name, path in paths.items()
    }
    result = build_authorization(
        args.model_id,
        reports["directional_admission"],
        reports["tetrad_erasure"],
        reports["clarity_gate"],
        reports["boundary_classification"],
        artifact_hashes,
    )
    result["command"] = shlex.join(
        [str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    # The command is audit metadata, not part of the scientific fingerprint.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["reader_grounded_projection_authorized"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
