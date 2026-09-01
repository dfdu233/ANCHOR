#!/usr/bin/env python3
"""Strict, deterministic evaluator for metric-calibration probe generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "metric-calibration-analysis-v1"
NUMBER_UNIT = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(mm|cm)\b", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    start = text.find("{")
    if start < 0:
        return None, "no_json_object"
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as error:
        return None, f"json_decode:{error.msg}"
    if not isinstance(value, dict):
        return None, "json_not_object"
    required = {"visible", "endpoints_normalized", "measurement_type", "physical_value", "unit"}
    if set(value) != required:
        return None, "json_schema_keys"
    if value["measurement_type"] not in {"patient-mm", "detector-mm", "pixel-only/unknown"}:
        return None, "measurement_type"
    if value["unit"] not in {"mm", "cm", None}:
        return None, "unit"
    if value["physical_value"] is not None:
        try:
            value["physical_value"] = float(value["physical_value"])
        except (TypeError, ValueError):
            return None, "physical_value"
    return value, None


def endpoint_distance(left: Any, right: Any) -> float | None:
    try:
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if a.shape != (2, 2) or b.shape != (2, 2):
            return None
        direct = float(np.sqrt(np.mean((a - b) ** 2)))
        reversed_value = float(np.sqrt(np.mean((a - b[::-1]) ** 2)))
        return min(direct, reversed_value)
    except (TypeError, ValueError):
        return None


def analyze(answers: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()]
    parsed_rows = []
    errors = Counter()
    for row in rows:
        parsed, error = extract_json(str(row["raw_text"]))
        if error:
            errors[error] += 1
        parsed_rows.append((row, parsed, error))

    valid = [(row, value) for row, value, error in parsed_rows if not error and value is not None]
    type_correct = [value["measurement_type"] == row["expected_measurement_type"] for row, value in valid]
    patient_type_overcommitment = []
    value_policy_violations = []
    numeric_errors = []
    endpoint_by_key: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row, value in valid:
        identifiable = bool(row["patient_value_identifiable"])
        textual_number = bool(NUMBER_UNIT.search(str(row["raw_text"])))
        if not identifiable:
            patient_type_overcommitment.append(value["measurement_type"] == "patient-mm")
            value_policy_violations.append(
                value["physical_value"] is not None or value["unit"] is not None or textual_number
            )
        expected = row.get("expected_physical_value")
        if expected is not None and value["physical_value"] is not None and value["unit"] == row.get("expected_unit"):
            numeric_errors.append(abs(math.log(max(value["physical_value"], 1e-12) / expected)))
        endpoint_by_key[(row["image_id"], row["arm"])].append(value["endpoints_normalized"])

    endpoint_distances = []
    for endpoints in endpoint_by_key.values():
        if len(endpoints) < 2:
            continue
        for other in endpoints[1:]:
            distance = endpoint_distance(endpoints[0], other)
            if distance is not None:
                endpoint_distances.append(distance)
    count = len(rows)
    result = {
        "version": VERSION,
        "answers": str(answers.resolve()),
        "answers_sha256": sha256_file(answers),
        "rows": count,
        "json_valid_rate": len(valid) / count if count else 0.0,
        "json_errors": dict(sorted(errors.items())),
        "nonempty_rate": sum(bool(str(row["raw_text"]).strip()) for row in rows) / count if count else 0.0,
        "cap_hit_rate": sum(row.get("stop_reason") == "max_new_tokens" for row in rows) / count if count else 0.0,
        "measurement_type_accuracy": float(np.mean(type_correct)) if type_correct else None,
        "unidentifiable_patient_type_overcommitment_rate": (
            float(np.mean(patient_type_overcommitment)) if patient_type_overcommitment else None
        ),
        "unidentifiable_value_policy_violation_rate": (
            float(np.mean(value_policy_violations)) if value_policy_violations else None
        ),
        "certified_value_median_absolute_log_ratio_error": float(np.median(numeric_errors)) if numeric_errors else None,
        "endpoint_condition_median_rms_drift": float(np.median(endpoint_distances)) if endpoint_distances else None,
        "runtime_admissible": bool(
            count
            and len(valid) / count >= 0.95
            and sum(bool(str(row["raw_text"]).strip()) for row in rows) / count >= 0.95
            and sum(row.get("stop_reason") == "max_new_tokens" for row in rows) / count <= 0.05
        ),
        "gpu_authorized_downstream": False,
        "interpretation_lock": "runtime qualification and paired behavior only; no patient-mm accuracy claim",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.answers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
