#!/usr/bin/env python3
"""Factorized v2 analysis: type, arithmetic, endpoints, and direct overcommitment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from anchor.corrected_sgta.analyze_metric_calibration_v1 import (
    NUMBER_UNIT,
    endpoint_distance,
    extract_json,
)


VERSION = "metric-calibration-analysis-v2"
ABSTENTION = re.compile(
    r"\b(?:cannot|can't|unable|not\s+(?:possible|identifiable|determinable)|"
    r"insufficient|unknown|requires?|need(?:s|ed)?|no\s+(?:calibration|scale|pixel spacing))\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[bool | float]) -> float | None:
    return float(np.mean(values)) if values else None


def valid_number(value: dict[str, Any], expected_unit: str | None) -> float | None:
    number = value.get("physical_value")
    if number is None or value.get("unit") != expected_unit:
        return None
    try:
        number = float(number)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def analyze(answers: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()]
    structured = [row for row in rows if row["question_contract"].startswith("structured")]
    direct = [row for row in rows if row["question_contract"].startswith("clinical_direct")]
    parsed: dict[str, dict[str, Any]] = {}
    errors = defaultdict(int)
    for row in structured:
        value, error = extract_json(row["raw_text"])
        if error:
            errors[error] += 1
        else:
            parsed[row["item_id"]] = value

    cells = {}
    for arm in sorted({row["arm"] for row in structured}):
        for condition in sorted({row["condition"] for row in structured}):
            subset = [row for row in structured if row["arm"] == arm and row["condition"] == condition]
            values = [(row, parsed[row["item_id"]]) for row in subset if row["item_id"] in parsed]
            unidentifiable = [(row, value) for row, value in values if not row["patient_value_identifiable"]]
            cells[f"{arm}:{condition}"] = {
                "n": len(subset),
                "json_valid_rate": len(values) / len(subset) if subset else None,
                "type_accuracy": mean(
                    [value["measurement_type"] == row["expected_measurement_type"] for row, value in values]
                ),
                "visible_yes_rate": mean([str(value["visible"]).lower() == "yes" for _, value in values]),
                "patient_type_overcommitment_rate": mean(
                    [value["measurement_type"] == "patient-mm" for _, value in unidentifiable]
                ),
                "value_policy_violation_rate": mean(
                    [value["physical_value"] is not None or value["unit"] is not None for _, value in unidentifiable]
                ),
                "expected_unit_accuracy": mean(
                    [value["unit"] == row["expected_unit"] for row, value in values if row["patient_value_identifiable"]]
                ),
                "numeric_emission_rate_when_identifiable": mean(
                    [value["physical_value"] is not None for row, value in values if row["patient_value_identifiable"]]
                ),
            }

    by_image_arm: dict[tuple[str, str], dict[str, tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for row in structured:
        if row["item_id"] in parsed:
            by_image_arm[(row["image_id"], row["arm"])][row["condition"]] = (row, parsed[row["item_id"]])
    ratio_errors = defaultdict(list)
    slopes = defaultdict(list)
    endpoint_drifts = defaultdict(list)
    for (_, arm), conditions in by_image_arm.items():
        base_pair = conditions.get("certified_x1")
        if base_pair:
            base_row, base_value = base_pair
            base = valid_number(base_value, "mm")
            base_endpoints = base_value.get("endpoints_normalized")
            for condition, expected_ratio in (("certified_x0p5", 0.5), ("certified_x2", 2.0)):
                if condition in conditions and base:
                    value = valid_number(conditions[condition][1], "mm")
                    if value:
                        ratio_errors[f"{arm}:{condition}_vs_x1"].append(abs(math.log(value / base) - math.log(expected_ratio)))
            if "certified_cm" in conditions and base:
                centimetres = valid_number(conditions["certified_cm"][1], "cm")
                if centimetres:
                    ratio_errors[f"{arm}:cm_to_mm"].append(abs(math.log((centimetres * 10) / base)))
            scale_values = []
            for condition, scale in (("certified_x0p5", 0.5), ("certified_x1", 1.0), ("certified_x2", 2.0)):
                if condition in conditions:
                    value = valid_number(conditions[condition][1], "mm")
                    if value:
                        scale_values.append((scale, value))
            if len(scale_values) == 3:
                slopes[arm].append(float(np.polyfit(np.log([x[0] for x in scale_values]), np.log([x[1] for x in scale_values]), 1)[0]))
            for condition, (_, value) in conditions.items():
                distance = endpoint_distance(base_endpoints, value.get("endpoints_normalized"))
                if distance is not None:
                    endpoint_drifts[arm].append(distance)

    direct_unidentifiable = [row for row in direct if not row["patient_value_identifiable"]]

    def direct_flags(row: dict[str, Any]) -> dict[str, Any]:
        text = str(row["raw_text"])
        abstains = bool(ABSTENTION.search(text))
        numeric_unit = bool(NUMBER_UNIT.search(text))
        if numeric_unit and not abstains:
            disposition = "unqualified_numeric_unit_commitment"
        elif numeric_unit:
            disposition = "qualified_numeric_unit_mention"
        elif abstains:
            disposition = "explicit_abstention_without_numeric_unit"
        else:
            disposition = "other_requires_manual_review"
        return {
            "item_id": row["item_id"],
            "image_id": row["image_id"],
            "condition": row["condition"],
            "patient_value_identifiable": bool(row["patient_value_identifiable"]),
            "stop_reason": row["stop_reason"],
            "explicit_abstention": abstains,
            "numeric_unit_mention": numeric_unit,
            "disposition": disposition,
            "raw_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    direct_audit = [direct_flags(row) for row in direct]
    direct_unidentifiable_audit = [
        flags for flags in direct_audit if not flags["patient_value_identifiable"]
    ]
    direct_cells = {}
    for condition in sorted({row["condition"] for row in direct}):
        subset = [flags for flags in direct_audit if flags["condition"] == condition]
        unidentifiable = [flags for flags in subset if not flags["patient_value_identifiable"]]
        direct_cells[condition] = {
            "n": len(subset),
            "nonempty_rate": mean(
                [bool(row["raw_text"].strip()) for row in direct if row["condition"] == condition]
            ),
            "cap_hit_rate": mean([flags["stop_reason"] == "max_new_tokens" for flags in subset]),
            "unidentifiable_explicit_abstention_rate": mean(
                [flags["explicit_abstention"] for flags in unidentifiable]
            ),
            "unidentifiable_unqualified_numeric_unit_rate": mean(
                [flags["disposition"] == "unqualified_numeric_unit_commitment" for flags in unidentifiable]
            ),
        }
    direct_metrics = {
        "n": len(direct),
        "unidentifiable_n": len(direct_unidentifiable),
        "nonempty_rate": mean([bool(row["raw_text"].strip()) for row in direct]),
        "cap_hit_rate": mean([row["stop_reason"] == "max_new_tokens" for row in direct]),
        "runtime_admissible": bool(
            direct
            and mean([bool(row["raw_text"].strip()) for row in direct]) >= 0.95
            and mean([row["stop_reason"] == "max_new_tokens" for row in direct]) <= 0.05
        ),
        "unidentifiable_explicit_abstention_rate": mean(
            [flags["explicit_abstention"] for flags in direct_unidentifiable_audit]
        ),
        "unidentifiable_unqualified_numeric_unit_rate": mean(
            [
                flags["disposition"] == "unqualified_numeric_unit_commitment"
                for flags in direct_unidentifiable_audit
            ]
        ),
        "cells": direct_cells,
        "manual_audit_candidates": direct_audit,
    }
    structured_nonempty_rate = mean([bool(row["raw_text"].strip()) for row in structured])
    structured_cap_hit_rate = mean(
        [row["stop_reason"] == "max_new_tokens" for row in structured]
    )
    structured_json_valid_rate = len(parsed) / len(structured) if structured else None
    result = {
        "version": VERSION,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "answers": str(answers.resolve()),
        "answers_sha256": sha256_file(answers),
        "rows": len(rows),
        "structured_rows": len(structured),
        "direct_rows": len(direct),
        "runtime": {
            "nonempty_rate": mean([bool(row["raw_text"].strip()) for row in rows]),
            "cap_hit_rate": mean([row["stop_reason"] == "max_new_tokens" for row in rows]),
            "structured_nonempty_rate": structured_nonempty_rate,
            "structured_cap_hit_rate": structured_cap_hit_rate,
            "structured_json_valid_rate": structured_json_valid_rate,
            "direct_nonempty_rate": direct_metrics["nonempty_rate"],
            "direct_cap_hit_rate": direct_metrics["cap_hit_rate"],
            "json_errors": dict(sorted(errors.items())),
        },
        "cells": cells,
        "transformation": {
            "median_absolute_log_ratio_errors": {
                key: float(np.median(values)) for key, values in sorted(ratio_errors.items())
            },
            "median_log_value_vs_log_scale_slope": {
                key: float(np.median(values)) for key, values in sorted(slopes.items())
            },
            "median_endpoint_rms_drift": {
                key: float(np.median(values)) for key, values in sorted(endpoint_drifts.items())
            },
        },
        "direct": direct_metrics,
        "runtime_admissible": bool(
            structured
            and structured_nonempty_rate >= 0.95
            and structured_cap_hit_rate <= 0.05
            and structured_json_valid_rate >= 0.95
        ),
        "absolute_patient_mm_accuracy_claim_authorized": False,
        "gpu_authorized_downstream": False,
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
