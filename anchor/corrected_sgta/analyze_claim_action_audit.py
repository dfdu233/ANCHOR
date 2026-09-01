#!/usr/bin/env python3
"""Audit whether hedging changes a report-method ranking without fixing content.

The input claims are prediction-side RadGraph proposals.  Reference reports are
used only when they explicitly mention a finding as present or absent; an
unmentioned reference finding is never silently treated as negative.  This
makes the audit conservative, but it remains Grade-C evidence because a single
report and an automatic extractor are not clinical truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VERSION = "claim-action-rank-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reports(path: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"{path} contains no RadGraph reports")
    indexed: dict[str, dict[str, object]] = {}
    for row in reports:
        identifier = str(row["id"])
        if identifier in indexed:
            raise ValueError(f"duplicate report id in {path}: {identifier}")
        indexed[identifier] = row
    return dict(payload.get("config", {})), indexed


def finding_axes(row: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Collapse entity duplicates to finding-level polarity and commitment.

    A positive mention dominates a negative mention because the report still
    introduces positive clinical content.  Mixed certainty is retained.
    """

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for claim in row.get("claims", []):
        if claim.get("provenance", "image_grounded") != "image_grounded":
            continue
        grouped[str(claim["finding"])].append(claim)
    output: dict[str, dict[str, object]] = {}
    for finding, claims in grouped.items():
        polarities = {str(claim["polarity"]) for claim in claims}
        uncertainties = {str(claim["uncertainty"]) for claim in claims}
        output[finding] = {
            "polarity": "present" if "present" in polarities else "absent",
            "uncertainty": (
                "uncertain"
                if "uncertain" in uncertainties
                else "definite"
            ),
            "contradictory": polarities == {"present", "absent"},
            "mention_count": len(claims),
        }
    return output


def report_counts(
    prediction: Mapping[str, object], reference: Mapping[str, object]
) -> dict[str, float]:
    predicted = finding_axes(prediction)
    truth = finding_axes(reference)
    reference_positive = {
        finding for finding, axes in truth.items() if axes["polarity"] == "present"
    }
    reference_negative = {
        finding for finding, axes in truth.items() if axes["polarity"] == "absent"
    }
    predicted_positive = {
        finding
        for finding, axes in predicted.items()
        if axes["polarity"] == "present"
    }
    predicted_definite_positive = {
        finding
        for finding in predicted_positive
        if predicted[finding]["uncertainty"] == "definite"
    }
    predicted_hedged_positive = predicted_positive - predicted_definite_positive
    explicit_reference = reference_positive | reference_negative
    resolved_positive = predicted_positive & explicit_reference
    resolved_definite = predicted_definite_positive & explicit_reference
    false_positive = predicted_positive & reference_negative
    false_definite = predicted_definite_positive & reference_negative
    false_hedged = predicted_hedged_positive & reference_negative
    recovered = reference_positive & predicted_positive
    overcommitted = {
        finding
        for finding in predicted_definite_positive & reference_positive
        if truth[finding]["uncertainty"] == "uncertain"
    }
    undercommitted = {
        finding
        for finding in predicted_hedged_positive & reference_positive
        if truth[finding]["uncertainty"] == "definite"
    }
    audit = prediction.get("audit", {})
    words = len(str(prediction.get("report", "")).split())
    return {
        "axis_false_positive": float(len(false_positive)),
        "axis_resolved_positive": float(len(resolved_positive)),
        "legacy_false_positive": float(len(false_definite)),
        "legacy_resolved_positive": float(len(resolved_definite)),
        "masked_hedged_false_positive": float(len(false_hedged)),
        "reference_positive": float(len(reference_positive)),
        "recovered_positive": float(len(recovered)),
        "predicted_positive": float(len(predicted_positive)),
        "predicted_definite_positive": float(len(predicted_definite_positive)),
        "predicted_hedged_positive": float(len(predicted_hedged_positive)),
        "overcommitted_reference_uncertain": float(len(overcommitted)),
        "undercommitted_reference_definite": float(len(undercommitted)),
        "prediction_claims": float(len(predicted)),
        "prediction_words": float(words),
        "unmatched_observations": float(len(audit.get("unmatched_observations", []))),
        "observation_roots": float(audit.get("n_observation_roots", 0)),
    }


def ratio(rows: Sequence[Mapping[str, float]], numerator: str, denominator: str) -> float | None:
    den = sum(row[denominator] for row in rows)
    return sum(row[numerator] for row in rows) / den if den else None


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("cannot aggregate an empty cohort")
    n = len(rows)
    return {
        "n_reports": n,
        "axis_aware_explicit_negative_false_positive_rate": ratio(
            rows, "axis_false_positive", "axis_resolved_positive"
        ),
        "legacy_collapsed_third_state_false_positive_rate": ratio(
            rows, "legacy_false_positive", "legacy_resolved_positive"
        ),
        "positive_finding_recall_against_reference": ratio(
            rows, "recovered_positive", "reference_positive"
        ),
        "hedged_positive_rate": ratio(
            rows, "predicted_hedged_positive", "predicted_positive"
        ),
        "masked_hedged_false_positive_count": int(
            sum(row["masked_hedged_false_positive"] for row in rows)
        ),
        "overcommitted_reference_uncertain_count": int(
            sum(row["overcommitted_reference_uncertain"] for row in rows)
        ),
        "undercommitted_reference_definite_count": int(
            sum(row["undercommitted_reference_definite"] for row in rows)
        ),
        "mean_positive_findings": sum(row["predicted_positive"] for row in rows) / n,
        "mean_extracted_findings": sum(row["prediction_claims"] for row in rows) / n,
        "mean_words": sum(row["prediction_words"] for row in rows) / n,
        "radgraph_ontology_match_rate": (
            1.0
            - ratio(rows, "unmatched_observations", "observation_roots")
            if sum(row["observation_roots"] for row in rows)
            else None
        ),
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_metric(
    rows: Sequence[Mapping[str, float]],
    metric: str,
    draws: int,
    rng: random.Random,
) -> dict[str, float | int | None]:
    estimate = aggregate(rows)[metric]
    if estimate is None:
        return {"estimate": None, "ci95": None, "valid_draws": 0}
    values: list[float] = []
    for _ in range(draws):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = aggregate(sample)[metric]
        if value is not None:
            values.append(float(value))
    return {
        "estimate": float(estimate),
        "ci95": [percentile(values, 0.025), percentile(values, 0.975)] if values else None,
        "valid_draws": len(values),
    }


def ranking(
    summaries: Mapping[str, Mapping[str, object]], metric: str
) -> list[str]:
    eligible = [
        (float(summary[metric]), method)
        for method, summary in summaries.items()
        if summary.get(metric) is not None
    ]
    return [method for _, method in sorted(eligible)]


def parse_methods(values: Iterable[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--method must have NAME=PATH form")
        name, path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"invalid or duplicate method name: {name!r}")
        output[name] = Path(path)
    if len(output) < 2:
        raise ValueError("at least two methods are required")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    method_paths = parse_methods(args.method)
    reference_config, references = load_reports(args.reference)
    method_payloads = {
        method: load_reports(path) for method, path in method_paths.items()
    }
    common_ids = set(references)
    for _, reports in method_payloads.values():
        common_ids &= set(reports)
    ordered_ids = sorted(common_ids)
    if not ordered_ids:
        raise ValueError("methods and reference have no common report ids")

    per_method_rows: dict[str, list[dict[str, float]]] = {}
    for method, (_, reports) in method_payloads.items():
        per_method_rows[method] = [
            report_counts(reports[identifier], references[identifier])
            for identifier in ordered_ids
        ]
    summaries = {
        method: aggregate(rows) for method, rows in per_method_rows.items()
    }
    rng = random.Random(args.seed)
    interval_metrics = (
        "axis_aware_explicit_negative_false_positive_rate",
        "legacy_collapsed_third_state_false_positive_rate",
        "positive_finding_recall_against_reference",
        "hedged_positive_rate",
    )
    intervals = {
        method: {
            metric: bootstrap_metric(rows, metric, args.bootstrap_draws, rng)
            for metric in interval_metrics
        }
        for method, rows in per_method_rows.items()
    }
    axis_ranking = ranking(
        summaries, "axis_aware_explicit_negative_false_positive_rate"
    )
    legacy_ranking = ranking(
        summaries, "legacy_collapsed_third_state_false_positive_rate"
    )
    payload = {
        "config": {
            "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reference": str(args.reference.resolve()),
            "reference_sha256": sha256_file(args.reference),
            "reference_radgraph_fingerprint": reference_config.get("fingerprint"),
            "methods": {
                method: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "radgraph_fingerprint": method_payloads[method][0].get("fingerprint"),
                }
                for method, path in method_paths.items()
            },
            "common_report_count": len(ordered_ids),
            "common_id_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
            "code_sha256": sha256_file(Path(__file__)),
            "evidence_grade": "C",
        },
        "summaries": summaries,
        "bootstrap": intervals,
        "rankings": {
            "axis_aware_false_positive_rate_lower_is_better": axis_ranking,
            "legacy_false_positive_rate_lower_is_better": legacy_ranking,
            "ranking_changed": axis_ranking != legacy_ranking,
        },
        "interpretation_contract": {
            "positive_content": "definite and hedged positive findings both count",
            "legacy_error": "hedged positive findings are erased from the positive denominator",
            "reference_policy": "only explicitly positive/negative reference findings are scored; unmentioned is not negative",
            "claim_ceiling": "single reference report plus automatic RadGraph extraction; not clinical truth",
        },
    }
    payload["config"]["fingerprint"] = hashlib.sha256(
        json.dumps(payload["config"], sort_keys=True).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
