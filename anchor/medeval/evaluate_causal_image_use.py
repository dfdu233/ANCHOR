#!/usr/bin/env python3
"""Backend-neutral causal image-use triad for medical VLM evaluation.

The point estimands and behavioral taxonomy follow the official implementation
of arXiv:2606.17710v2 (commit 6acd5639...), while uncertainty is adapted to
patient/episode-cluster bootstrap.  Passing this gate proves only stable image
use for a model/finding cell.  It cannot authorize a PCEM mechanism claim,
representation capture, image download, or GPU experiment by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .hashing import sha256_file, sha256_json


PROTOCOL_ID = "anchor-causal-image-use-triad-v1"
PARSER_CONTRACT = "upstream-normalized-explicit-binary-v1"
OFFICIAL_PAPER = "https://arxiv.org/abs/2606.17710v2"
OFFICIAL_REPOSITORY = "https://github.com/mahshadlotfinia/causal"
OFFICIAL_COMMIT = "6acd5639f06c7ac89c890f67a7e1eef335726d47"
CONDITIONS = ("original", "swap", "target_mask", "irrelevant_mask")
DECISIONS = {"yes", "no", "invalid"}
GROUND_TRUTH = {"yes", "no"}
VIEWS = {"AP", "PA", "OTHER"}
GROUND_TRUTH_PROVENANCE = {
    "independent_clinical",
    "expert_image_annotation",
    "report_derived",
    "model_derived",
}
TRUSTED_GROUND_TRUTH_PROVENANCE = {
    "independent_clinical",
    "expert_image_annotation",
}
REGION_PROVENANCE = {
    "expert_box",
    "clinician_validated_segmentation",
    "automatic_unvalidated",
}
TRUSTED_REGION_PROVENANCE = {
    "expert_box",
    "clinician_validated_segmentation",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class CausalConditionRecord:
    case_id: str
    model_id: str
    finding: str
    cluster_id: str
    view: str
    condition: str
    ground_truth: str
    ground_truth_provenance: str
    decision: str
    raw_text_sha256: str
    parser_version: str
    question_sha256: str
    prompt_sha256: str
    reference_contract_sha256: str
    swap_manifest_sha256: str
    source_image_sha256: str
    condition_image_sha256: str
    source_subject_hash: str
    condition_subject_hash: str
    swap_label_preserved: bool
    target_region_defined: bool
    irrelevant_region_defined: bool
    region_provenance: str
    mask_sha256: str | None = None
    mask_area_pixels: int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CausalConditionRecord":
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected.difference(row))
        extra = sorted(set(row).difference(expected))
        if missing or extra:
            raise ContractError(f"record fields missing={missing} extra={extra}")
        record = cls(**row)
        record.validate()
        return record

    def validate(self) -> None:
        for name in ("case_id", "model_id", "finding", "cluster_id", "parser_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ContractError(f"{name} must be a nonempty string")
        if self.view not in VIEWS:
            raise ContractError(f"view must be one of {sorted(VIEWS)}")
        if self.condition not in CONDITIONS:
            raise ContractError(f"unknown condition: {self.condition}")
        if self.ground_truth not in GROUND_TRUTH:
            raise ContractError("ground_truth must be yes or no")
        if self.ground_truth_provenance not in GROUND_TRUTH_PROVENANCE:
            raise ContractError(
                f"unknown ground_truth_provenance: {self.ground_truth_provenance}"
            )
        if self.decision not in DECISIONS:
            raise ContractError(f"decision must be one of {sorted(DECISIONS)}")
        for name in (
            "raw_text_sha256",
            "question_sha256",
            "prompt_sha256",
            "reference_contract_sha256",
            "swap_manifest_sha256",
            "source_image_sha256",
            "condition_image_sha256",
            "source_subject_hash",
            "condition_subject_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or HEX64.fullmatch(value) is None:
                raise ContractError(f"{name} must be lowercase sha256 hex")
        for name in (
            "swap_label_preserved",
            "target_region_defined",
            "irrelevant_region_defined",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(f"{name} must be boolean")
        if self.region_provenance not in REGION_PROVENANCE:
            raise ContractError(f"unknown region_provenance: {self.region_provenance}")
        if self.mask_sha256 is not None and HEX64.fullmatch(self.mask_sha256) is None:
            raise ContractError("mask_sha256 must be null or lowercase sha256 hex")
        if self.mask_area_pixels is not None and (
            not isinstance(self.mask_area_pixels, int) or self.mask_area_pixels <= 0
        ):
            raise ContractError("mask_area_pixels must be null or a positive integer")


def _require_equal(records: Sequence[CausalConditionRecord], field: str) -> Any:
    values = {getattr(record, field) for record in records}
    if len(values) != 1:
        raise ContractError(f"case conditions disagree on {field}: {sorted(values)}")
    return next(iter(values))


def validate_case_conditions(
    records: Sequence[CausalConditionRecord], *, require_all_conditions: bool = True
) -> dict[str, CausalConditionRecord]:
    if not records:
        raise ContractError("empty case condition group")
    by_condition: dict[str, CausalConditionRecord] = {}
    for record in records:
        if record.condition in by_condition:
            raise ContractError(
                f"duplicate case/model condition: {record.case_id}/{record.model_id}/{record.condition}"
            )
        by_condition[record.condition] = record
    required = set(CONDITIONS) if require_all_conditions else {"original", "swap"}
    missing = sorted(required.difference(by_condition))
    if missing:
        raise ContractError(
            f"case/model {records[0].case_id}/{records[0].model_id} missing conditions {missing}"
        )
    ordered = [by_condition[name] for name in by_condition]
    for field in (
        "case_id",
        "model_id",
        "finding",
        "cluster_id",
        "view",
        "ground_truth",
        "ground_truth_provenance",
        "parser_version",
        "question_sha256",
        "prompt_sha256",
        "reference_contract_sha256",
        "swap_manifest_sha256",
        "source_image_sha256",
        "source_subject_hash",
        "swap_label_preserved",
        "target_region_defined",
        "irrelevant_region_defined",
        "region_provenance",
    ):
        _require_equal(ordered, field)

    original = by_condition["original"]
    swap = by_condition["swap"]
    if original.condition_image_sha256 != original.source_image_sha256:
        raise ContractError("original condition must use the source image bytes")
    if original.condition_subject_hash != original.source_subject_hash:
        raise ContractError("original condition must retain the source subject")
    if not original.swap_label_preserved:
        raise ContractError("swap_label_preserved must be true for the frozen case")
    if swap.condition_image_sha256 == swap.source_image_sha256:
        raise ContractError("swap must use a different image")
    if swap.condition_subject_hash == swap.source_subject_hash:
        raise ContractError("swap must use a different patient")
    if swap.mask_sha256 is not None or swap.mask_area_pixels is not None:
        raise ContractError("swap condition cannot carry a mask")

    for condition in ("original",):
        row = by_condition[condition]
        if row.mask_sha256 is not None or row.mask_area_pixels is not None:
            raise ContractError(f"{condition} condition cannot carry a mask")

    if require_all_conditions:
        target = by_condition["target_mask"]
        irrelevant = by_condition["irrelevant_mask"]
        if not target.target_region_defined or not irrelevant.irrelevant_region_defined:
            raise ContractError("strict triad requires defined target and irrelevant regions")
        for row in (target, irrelevant):
            if row.condition_subject_hash != row.source_subject_hash:
                raise ContractError("mask conditions must retain the source patient")
            if row.condition_image_sha256 == row.source_image_sha256:
                raise ContractError("mask condition image must differ from source image")
            if row.mask_sha256 is None or row.mask_area_pixels is None:
                raise ContractError("mask conditions require mask hash and area")
        if target.mask_sha256 == irrelevant.mask_sha256:
            raise ContractError("target and irrelevant masks must differ")
        if target.condition_image_sha256 == irrelevant.condition_image_sha256:
            raise ContractError("target and irrelevant masked images must differ")
        if target.mask_area_pixels != irrelevant.mask_area_pixels:
            raise ContractError("target and irrelevant masks must have equal pixel area")
    return by_condition


def read_records(path: Path) -> list[CausalConditionRecord]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"invalid JSON at line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ContractError(f"line {line_number} is not a JSON object")
            try:
                records.append(CausalConditionRecord.from_dict(row))
            except (ContractError, TypeError) as error:
                raise ContractError(f"invalid record at line {line_number}: {error}") from error
    if not records:
        raise ContractError("causal image-use input is empty")
    return records


def _mean(outcomes: Sequence[tuple[str, float]]) -> float | None:
    return float(np.mean([value for _, value in outcomes])) if outcomes else None


def cluster_bootstrap_proportion(
    outcomes: Sequence[tuple[str, float]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("cluster bootstrap requires at least 100 replicates")
    if not outcomes:
        return {
            "point": None,
            "ci_lower": None,
            "ci_upper": None,
            "valid_replicates": 0,
            "n_cases": 0,
            "n_clusters": 0,
        }
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cluster, value in outcomes:
        by_cluster[cluster].append(float(value))
    clusters = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [value for cluster in selected for value in by_cluster[str(cluster)]]
        if sample:
            values.append(float(np.mean(sample)))
    if len(values) < math.ceil(0.95 * replicates):
        raise ContractError("too few valid patient-cluster bootstrap replicates")
    return {
        "point": _mean(outcomes),
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
        "valid_replicates": len(values),
        "n_cases": len(outcomes),
        "n_clusters": len(clusters),
    }


def official_behavior_category(
    *, cgr: dict[str, Any], uar: dict[str, Any], stability: dict[str, Any]
) -> str:
    """Reproduce the released Figure-2 category rule on proportions."""
    if any(metric["point"] is None for metric in (cgr, uar, stability)):
        return "not_evaluable"
    cgr_value = float(cgr["point"])
    cgr_lower = float(cgr["ci_lower"])
    uar_value = float(uar["point"])
    stability_value = float(stability["point"])
    if stability_value < 0.70:
        return "unstable"
    if (
        math.isclose(cgr_value, 0.0, abs_tol=1e-12)
        and math.isclose(uar_value, 1.0, abs_tol=1e-12)
        and math.isclose(stability_value, 1.0, abs_tol=1e-12)
    ):
        return "ignores_image"
    if cgr_value > 0.0 and cgr_lower > 0.0 and stability_value >= 0.90:
        return "uses_image"
    return "other"


def _case_outcomes(
    groups: Sequence[dict[str, CausalConditionRecord]],
) -> dict[str, Any]:
    cgr: list[tuple[str, float]] = []
    uar: list[tuple[str, float]] = []
    stability: list[tuple[str, float]] = []
    accuracy: list[tuple[str, float]] = []
    decisions_by_condition: dict[str, list[str]] = defaultdict(list)
    ground_truth_provenance: Counter[str] = Counter()
    region_provenance: Counter[str] = Counter()
    for group in groups:
        original = group["original"]
        ground_truth_provenance[original.ground_truth_provenance] += 1
        region_provenance[original.region_provenance] += 1
        for condition, row in group.items():
            decisions_by_condition[condition].append(row.decision)
        original_valid = original.decision in {"yes", "no"}
        correct = original_valid and original.decision == original.ground_truth
        if original_valid:
            accuracy.append((original.cluster_id, float(correct)))
        swap = group.get("swap")
        if correct and swap is not None and swap.decision in {"yes", "no"}:
            uar.append((original.cluster_id, float(swap.decision == original.decision)))
        target = group.get("target_mask")
        if (
            original.ground_truth == "yes"
            and original.decision == "yes"
            and original.target_region_defined
            and target is not None
            and target.decision in {"yes", "no"}
        ):
            cgr.append((original.cluster_id, float(target.decision != original.decision)))
        irrelevant = group.get("irrelevant_mask")
        if (
            correct
            and original.irrelevant_region_defined
            and irrelevant is not None
            and irrelevant.decision in {"yes", "no"}
        ):
            stability.append(
                (original.cluster_id, float(irrelevant.decision == original.decision))
            )
    parse_rates = {
        condition: (
            sum(value in {"yes", "no"} for value in decisions) / len(decisions)
            if decisions
            else 0.0
        )
        for condition, decisions in sorted(decisions_by_condition.items())
    }
    return {
        "cgr": cgr,
        "uar": uar,
        "stability": stability,
        "accuracy": accuracy,
        "parse_rates": parse_rates,
        "ground_truth_provenance_counts": dict(sorted(ground_truth_provenance.items())),
        "region_provenance_counts": dict(sorted(region_provenance.items())),
        "n_cases": len(groups),
    }


def summarize_cell(
    groups: Sequence[dict[str, CausalConditionRecord]],
    *,
    bootstrap_replicates: int,
    seed: int,
    minimum_eligible_cases: int,
    minimum_parse_rate: float,
) -> dict[str, Any]:
    outcomes = _case_outcomes(groups)
    cgr = cluster_bootstrap_proportion(
        outcomes["cgr"], replicates=bootstrap_replicates, seed=seed + 11
    )
    uar = cluster_bootstrap_proportion(
        outcomes["uar"], replicates=bootstrap_replicates, seed=seed + 23
    )
    stability = cluster_bootstrap_proportion(
        outcomes["stability"], replicates=bootstrap_replicates, seed=seed + 37
    )
    accuracy = cluster_bootstrap_proportion(
        outcomes["accuracy"], replicates=bootstrap_replicates, seed=seed + 53
    )
    category = official_behavior_category(cgr=cgr, uar=uar, stability=stability)
    count_gate = all(
        metric["n_cases"] >= minimum_eligible_cases for metric in (cgr, uar, stability)
    )
    parse_gate = all(
        outcomes["parse_rates"].get(condition, 0.0) >= minimum_parse_rate
        for condition in CONDITIONS
    )
    truth_provenance_gate = bool(outcomes["ground_truth_provenance_counts"]) and set(
        outcomes["ground_truth_provenance_counts"]
    ) <= TRUSTED_GROUND_TRUTH_PROVENANCE
    region_provenance_gate = bool(outcomes["region_provenance_counts"]) and set(
        outcomes["region_provenance_counts"]
    ) <= TRUSTED_REGION_PROVENANCE
    return {
        "n_cases": outcomes["n_cases"],
        "parse_rates": outcomes["parse_rates"],
        "ground_truth_provenance_counts": outcomes["ground_truth_provenance_counts"],
        "region_provenance_counts": outcomes["region_provenance_counts"],
        "accuracy": accuracy,
        "causal_grounding_rate": cgr,
        "unrelated_image_answer_rate": uar,
        "irrelevant_mask_stability": stability,
        "grounding_specificity_premium": (
            cgr["point"] - (1.0 - stability["point"])
            if cgr["point"] is not None and stability["point"] is not None
            else None
        ),
        "official_behavior_category": category,
        "minimum_eligible_case_gate_passed": count_gate,
        "parse_gate_passed": parse_gate,
        "ground_truth_provenance_gate_passed": truth_provenance_gate,
        "region_provenance_gate_passed": region_provenance_gate,
        "pcem_stable_image_user_cell": (
            category == "uses_image"
            and count_gate
            and parse_gate
            and truth_provenance_gate
            and region_provenance_gate
        ),
    }


def analyze_records(
    records: Sequence[CausalConditionRecord],
    *,
    target_models: Sequence[str],
    target_findings: Sequence[str],
    bootstrap_replicates: int = 10_000,
    seed: int = 260617,
    minimum_eligible_cases: int = 30,
    minimum_parse_rate: float = 0.95,
) -> dict[str, Any]:
    if not records:
        raise ContractError("causal image-use records are empty")
    for record in records:
        record.validate()
    if not target_models or len(set(target_models)) != len(target_models):
        raise ContractError("target_models must be a nonempty unique list")
    if not target_findings or len(set(target_findings)) != len(target_findings):
        raise ContractError("target_findings must be a nonempty unique list")
    if minimum_eligible_cases < 30:
        raise ContractError("minimum eligible cases cannot be below 30")
    if not 0.95 <= minimum_parse_rate <= 1.0:
        raise ContractError("minimum parse rate must be in [0.95, 1]")
    parser_versions = {record.parser_version for record in records}
    if len(parser_versions) != 1:
        raise ContractError(f"mixed parser versions: {sorted(parser_versions)}")
    if parser_versions != {PARSER_CONTRACT}:
        raise ContractError(
            f"parser version must be the frozen {PARSER_CONTRACT}, got {sorted(parser_versions)}"
        )
    reference_contracts = {record.reference_contract_sha256 for record in records}
    swap_manifests = {record.swap_manifest_sha256 for record in records}
    if len(reference_contracts) != 1 or len(swap_manifests) != 1:
        raise ContractError(
            "one run must bind exactly one reference contract and one frozen swap manifest"
        )

    raw_groups: dict[tuple[str, str, str], list[CausalConditionRecord]] = defaultdict(list)
    for record in records:
        raw_groups[(record.model_id, record.case_id, record.finding)].append(record)
    groups = [validate_case_conditions(rows) for _, rows in sorted(raw_groups.items())]
    present_models = {group["original"].model_id for group in groups}
    present_findings = {group["original"].finding for group in groups}
    missing_models = sorted(set(target_models).difference(present_models))
    missing_findings = sorted(set(target_findings).difference(present_findings))
    if missing_models or missing_findings:
        raise ContractError(
            f"target closure missing models={missing_models} findings={missing_findings}"
        )

    cells: dict[str, Any] = {}
    target_passes = []
    for model in target_models:
        for finding in target_findings:
            selected = [
                group
                for group in groups
                if group["original"].model_id == model
                and group["original"].finding == finding
            ]
            key = f"{model}::{finding}"
            summary = summarize_cell(
                selected,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16),
                minimum_eligible_cases=minimum_eligible_cases,
                minimum_parse_rate=minimum_parse_rate,
            )
            cells[key] = summary
            target_passes.append(bool(summary["pcem_stable_image_user_cell"]))

    by_view: dict[str, Any] = {}
    for model in target_models:
        for finding in target_findings:
            for view in ("AP", "PA"):
                selected = [
                    group
                    for group in groups
                    if group["original"].model_id == model
                    and group["original"].finding == finding
                    and group["original"].view == view
                ]
                if not selected:
                    continue
                key = f"{model}::{finding}::{view}"
                by_view[key] = summarize_cell(
                    selected,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16),
                    minimum_eligible_cases=minimum_eligible_cases,
                    minimum_parse_rate=minimum_parse_rate,
                )

    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "source_qualification": {
            "paper": OFFICIAL_PAPER,
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "adaptation": "same point estimands and released Figure-2 taxonomy; patient/episode-cluster bootstrap replaces iid case bootstrap",
        },
        "contract": {
            "conditions": list(CONDITIONS),
            "parser_version": next(iter(parser_versions)),
            "reference_contract_sha256": next(iter(reference_contracts)),
            "swap_manifest_sha256": next(iter(swap_manifests)),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_unit": "cluster_id",
            "seed": seed,
            "minimum_eligible_cases_per_metric": minimum_eligible_cases,
            "minimum_parse_rate_per_condition": minimum_parse_rate,
            "target_models": list(target_models),
            "target_findings": list(target_findings),
        },
        "model_finding_cells": cells,
        "model_finding_view_diagnostics": by_view,
        "admission": {
            "all_target_cells_stable_image_users": bool(target_passes)
            and all(target_passes),
            "cross_model_floor_passed": len(set(target_models)) >= 2,
            "pcem_image_use_gate_passed": bool(target_passes)
            and all(target_passes)
            and len(set(target_models)) >= 2,
            "echo_construct_gate_required": True,
            "geometry_by_view_behavior_gate_required": True,
            "representation_capture_authorized": False,
            "image_download_authorized": False,
            "gpu_authorized": False,
            "paper_claim_authorized": False,
        },
        "patient_identifiers_written": False,
    }
    result["fingerprint"] = sha256_json(result)
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-model", action="append", required=True)
    parser.add_argument("--target-finding", action="append", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=260617)
    parser.add_argument("--minimum-eligible-cases", type=int, default=30)
    parser.add_argument("--minimum-parse-rate", type=float, default=0.95)
    args = parser.parse_args()
    if args.bootstrap_replicates != 10_000:
        raise ContractError("CLI scientific runs require exactly 10,000 bootstrap replicates")
    records = read_records(args.input)
    result = analyze_records(
        records,
        target_models=args.target_model,
        target_findings=args.target_finding,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        minimum_eligible_cases=args.minimum_eligible_cases,
        minimum_parse_rate=args.minimum_parse_rate,
    )
    result["input"] = {
        "path": str(args.input.resolve()),
        "sha256": sha256_file(args.input),
        "records": len(records),
    }
    result["fingerprint"] = sha256_json(
        {key: value for key, value in result.items() if key != "fingerprint"}
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "fingerprint": result["fingerprint"],
                "pcem_image_use_gate_passed": result["admission"][
                    "pcem_image_use_gate_passed"
                ],
                "representation_capture_authorized": False,
                "gpu_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
