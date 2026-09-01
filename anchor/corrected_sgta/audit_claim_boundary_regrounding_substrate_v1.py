#!/usr/bin/env python3
"""Outcome-blind substrate audit for claim-boundary re-grounding.

The audit never compares a generated claim with a reference claim and never
computes a position--error association.  A single reference report plus an
automatic RadGraph extraction is not independent visual truth.  Likewise,
sequence-mean likelihoods under a changed image are not claim-level causal
dependence.  If either requirement is absent, the formal mechanism analysis is
fail-closed before any outcome is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "clinical-claim-boundary-regrounding-substrate-audit-v1"
RADGRAPH_VERSION = "missing-third-state-radgraph-claims-v2"


class AuditError(RuntimeError):
    """An input violates the frozen, outcome-blind substrate contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise AuditError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def load_claim_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid claim artifact {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise AuditError(f"claim artifact must be an object: {path}")
    config = payload.get("config")
    reports = payload.get("reports")
    if not isinstance(config, Mapping) or not isinstance(reports, list):
        raise AuditError(f"claim artifact lacks config/reports: {path}")
    identifiers = [str(row.get("id", "")) for row in reports if isinstance(row, Mapping)]
    if len(identifiers) != len(reports) or any(not value for value in identifiers):
        raise AuditError(f"claim artifact has malformed report rows: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise AuditError(f"claim artifact has duplicate report IDs: {path}")
    return dict(config), [dict(row) for row in reports]


def record_start(record: Mapping[str, Any], entities: Mapping[str, Any]) -> int:
    component_ids = record.get("component_entity_ids")
    if not isinstance(component_ids, list) or not component_ids:
        raise AuditError("claim record lacks component_entity_ids")
    starts: list[int] = []
    for entity_id in component_ids:
        entity = entities.get(str(entity_id))
        if not isinstance(entity, Mapping) or "start_ix" not in entity:
            raise AuditError(f"claim component {entity_id!r} lacks RadGraph start_ix")
        try:
            starts.append(int(entity["start_ix"]))
        except (TypeError, ValueError) as error:
            raise AuditError(f"invalid RadGraph start_ix for {entity_id!r}") from error
    return min(starts)


def summarize_prediction(path: Path) -> tuple[dict[str, Any], set[str]]:
    config, reports = load_claim_artifact(path)
    total_records = 0
    position_recoverable = 0
    explicit_ordinal = 0
    position_collisions = 0
    claim_counts: list[int] = []
    distinct_image_finding_counts: list[int] = []
    multi_claim_reports: list[str] = []
    multi_claim_sequences: list[tuple[tuple[str, str], ...]] = []

    for report in reports:
        audit = report.get("audit")
        if not isinstance(audit, Mapping):
            raise AuditError(f"{path}: report {report['id']} lacks audit")
        records = audit.get("records")
        entities = audit.get("radgraph_entities")
        if not isinstance(records, list) or not isinstance(entities, Mapping):
            raise AuditError(f"{path}: report {report['id']} lacks records/entities")
        ordered: list[tuple[int, str, str]] = []
        positions: list[int] = []
        image_findings: set[str] = set()
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                raise AuditError(f"{path}: malformed record in {report['id']}")
            total_records += 1
            explicit_ordinal += int(
                "claim_ordinal" in raw_record or "ordinal" in raw_record
            )
            start = record_start(raw_record, entities)
            position_recoverable += 1
            positions.append(start)
            claim = raw_record.get("claim")
            if not isinstance(claim, Mapping):
                raise AuditError(f"{path}: record in {report['id']} lacks claim")
            finding = str(claim.get("finding", ""))
            polarity = str(claim.get("polarity", ""))
            if not finding or polarity not in {"present", "absent"}:
                raise AuditError(f"{path}: malformed claim in {report['id']}")
            ordered.append((start, finding, polarity))
            if claim.get("provenance") == "image_grounded":
                image_findings.add(finding)
        position_collisions += len(positions) - len(set(positions))
        claim_counts.append(len(records))
        distinct_image_finding_counts.append(len(image_findings))
        if len(records) >= 2:
            multi_claim_reports.append(str(report.get("report", "")))
            multi_claim_sequences.append(
                tuple((finding, polarity) for _, finding, polarity in sorted(ordered))
            )

    report_frequency = Counter(multi_claim_reports)
    largest_template_share = (
        max(report_frequency.values()) / len(multi_claim_reports)
        if multi_claim_reports
        else None
    )
    return (
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "radgraph_version": config.get("version"),
            "radgraph_fingerprint": config.get("fingerprint"),
            "reports": len(reports),
            "extracted_claim_records": total_records,
            "explicit_native_ordinal_records": explicit_ordinal,
            "text_position_recoverable_records": position_recoverable,
            "text_position_collisions": position_collisions,
            "reports_with_at_least_two_extracted_claims": sum(
                value >= 2 for value in claim_counts
            ),
            "reports_with_at_least_three_extracted_claims": sum(
                value >= 3 for value in claim_counts
            ),
            "reports_with_at_least_two_distinct_image_grounded_findings": sum(
                value >= 2 for value in distinct_image_finding_counts
            ),
            "unique_exact_reports_among_multi_claim_reports": len(report_frequency),
            "largest_exact_template_share_among_multi_claim_reports": (
                largest_template_share
            ),
            "unique_ordered_claim_sequences_among_multi_claim_reports": len(
                set(multi_claim_sequences)
            ),
        },
        {str(report["id"]) for report in reports},
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise AuditError(f"{path}:{number}: row must be an object")
            rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSONL {path}: {error}") from error
    if not rows:
        raise AuditError(f"counterfactual artifact is empty: {path}")
    return rows


def summarize_counterfactual_artifact(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    top_level_keys = sorted(set().union(*(row.keys() for row in rows)))
    generated_counterfactual_fields = {
        "counterfactual_answer",
        "counterfactual_text",
        "shuffled_answer",
        "shuffled_text",
        "swap_answer",
        "swap_text",
        "zero_answer",
        "zero_text",
    }
    claimwise_fields = {
        "claim_counterfactuals",
        "claim_scores",
        "per_claim_counterfactual",
        "per_claim_image_dependence",
    }
    rows_with_generated_counterfactual = sum(
        any(field in row for field in generated_counterfactual_fields) for row in rows
    )
    rows_with_claimwise_counterfactual = sum(
        any(field in row for field in claimwise_fields) for row in rows
    )
    observed_interventions = set()
    for row in rows:
        if "zero_visual_generated_evidence" in row:
            observed_interventions.add("zero_visual_sequence_aggregate")
        if "shuffled_image" in row:
            observed_interventions.add("shuffled_image_sequence_aggregate")
        scores = row.get("scores")
        if isinstance(scores, Mapping):
            for condition in scores:
                normalized = str(condition).casefold()
                if normalized in {"zero", "shuffled"}:
                    observed_interventions.add(f"{normalized}_sequence_aggregate")
                nested = scores[condition]
                if isinstance(nested, Mapping) and "shuffled_image" in nested:
                    observed_interventions.add("shuffled_image_sequence_aggregate")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "top_level_keys": top_level_keys,
        "observed_image_interventions": sorted(observed_interventions),
        "rows_with_generated_counterfactual_answer": rows_with_generated_counterfactual,
        "rows_with_per_claim_counterfactual": rows_with_claimwise_counterfactual,
        "claim_boundary_counterfactual_eligible": bool(
            rows_with_generated_counterfactual == len(rows)
            and rows_with_claimwise_counterfactual == len(rows)
        ),
        "interpretation": (
            "Sequence-aggregate teacher-forced likelihood is not a per-claim "
            "counterfactual answer or a boundary-local causal dependence measure."
        ),
    }


def audit(
    reference_path: Path,
    summary_path: Path,
    prediction_paths: Sequence[tuple[str, Path]],
    counterfactual_paths: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    if not prediction_paths:
        raise AuditError("at least one prediction claim artifact is required")
    names = [name for name, _ in prediction_paths]
    if len(set(names)) != len(names):
        raise AuditError("prediction names must be unique")
    cf_names = [name for name, _ in counterfactual_paths]
    if len(set(cf_names)) != len(cf_names):
        raise AuditError("counterfactual names must be unique")

    reference_config, reference_reports = load_claim_artifact(reference_path)
    reference_ids = {str(report["id"]) for report in reference_reports}
    predictions: dict[str, Any] = {}
    aligned = True
    for name, path in prediction_paths:
        summary, identifiers = summarize_prediction(path)
        predictions[name] = summary
        aligned = aligned and identifiers == reference_ids

    try:
        claim_action_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid claim-action summary {summary_path}: {error}") from error
    evidence_grade = claim_action_summary.get("config", {}).get("evidence_grade")
    claim_ceiling = claim_action_summary.get("interpretation_contract", {}).get(
        "claim_ceiling"
    )
    automatic_reference = reference_config.get("version") == RADGRAPH_VERSION
    independent_visual_truth = False

    counterfactuals = {
        name: summarize_counterfactual_artifact(path)
        for name, path in counterfactual_paths
    }
    per_claim_counterfactual = bool(counterfactuals) and all(
        row["claim_boundary_counterfactual_eligible"]
        for row in counterfactuals.values()
    )
    sufficient_multiclaim = all(
        row["reports_with_at_least_two_distinct_image_grounded_findings"] >= 100
        for row in predictions.values()
    )
    template_diverse = all(
        row["largest_exact_template_share_among_multi_claim_reports"] is not None
        and row["largest_exact_template_share_among_multi_claim_reports"] <= 0.20
        for row in predictions.values()
    )
    formal_gate = (
        aligned
        and independent_visual_truth
        and per_claim_counterfactual
        and sufficient_multiclaim
        and template_diverse
    )

    source_path = Path(__file__).resolve()
    source_inputs = {
        "reference": reference_path,
        "claim_action_summary": summary_path,
        **{f"prediction:{name}": path for name, path in prediction_paths},
        **{f"counterfactual:{name}": path for name, path in counterfactual_paths},
    }
    source_rows = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(source_inputs.items())
    }
    fingerprint_payload = {
        "protocol_id": PROTOCOL_ID,
        "auditor_sha256": sha256_file(source_path),
        "inputs": {name: row["sha256"] for name, row in source_rows.items()},
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "protocol_id": PROTOCOL_ID,
        "status": "go_cpu_mechanism_probe" if formal_gate else "no_go_current_substrate",
        "dataset": "local MIMIC/OE generated artifacts",
        "model": "artifact-dependent; no model execution",
        "method": "outcome-blind substrate audit",
        "seed": "not_applicable_deterministic_audit",
        "command": [sys.executable, *sys.argv],
        "fingerprint": fingerprint,
        "provenance": {
            "auditor_path": str(source_path),
            "auditor_sha256": sha256_file(source_path),
            "inputs": source_rows,
        },
        "outcome_blind_contract": {
            "generated_vs_reference_claim_matching_performed": False,
            "position_error_association_computed": False,
            "sealed_confirmation_opened": False,
            "gpu_used": False,
            "reference_states_used_as_visual_truth": False,
        },
        "reference_truth_audit": {
            "path": str(reference_path.resolve()),
            "sha256": sha256_file(reference_path),
            "reports": len(reference_reports),
            "radgraph_version": reference_config.get("version"),
            "automatic_radgraph_reference": automatic_reference,
            "declared_evidence_grade": evidence_grade,
            "declared_claim_ceiling": claim_ceiling,
            "independent_per_claim_visual_truth": independent_visual_truth,
            "reason": (
                "The source is one reference report processed by an automatic "
                "RadGraph/ontology converter. It is neither multi-reader image "
                "truth nor completed physician per-claim review."
            ),
        },
        "prediction_ordinal_audit": predictions,
        "counterfactual_audit": counterfactuals,
        "gates": {
            "exact_case_alignment": aligned,
            "independent_per_claim_visual_truth": independent_visual_truth,
            "per_claim_image_counterfactual": per_claim_counterfactual,
            "minimum_100_multiclaim_cases_per_model": sufficient_multiclaim,
            "largest_exact_template_share_at_most_0_20": template_diverse,
            "formal_mechanism_analysis_authorized": formal_gate,
            "gpu_authorized": False,
        },
        "failure_axes": {
            "truth_failure": not independent_visual_truth,
            "counterfactual_granularity_failure": not per_claim_counterfactual,
            "multiclaim_prevalence_failure": not sufficient_multiclaim,
            "template_concentration_failure": not template_diverse,
        },
        "scientific_naming_gate": {
            "claim_boundary_regrounding_failure_authorized": False,
            "permitted_current_description": (
                "automatic-report positional substrate audit with severe template "
                "concentration"
            ),
            "reason": (
                "A worsening error curve would be non-identifiable without "
                "independent claim truth and boundary-local image counterfactuals."
            ),
        },
    }


def atomic_write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--claim-action-summary", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", type=parse_named_path, required=True
    )
    parser.add_argument(
        "--counterfactual", action="append", type=parse_named_path, default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.reference,
        args.claim_action_summary,
        args.prediction,
        args.counterfactual,
    )
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    atomic_write_new(args.output, encoded)
    print(json.dumps({
        "status": result["status"],
        "fingerprint": result["fingerprint"],
        "output": str(args.output),
        "gates": result["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
