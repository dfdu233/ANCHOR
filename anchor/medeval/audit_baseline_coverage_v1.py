#!/usr/bin/env python3
"""Audit baseline/RAG coverage without promoting execution into efficacy.

The audit closes the current method configuration against the T0 and evidence
ladders, then reports dataset/model/task coverage separately.  Missing cells,
failed report controls, T2-only activations, and paper-native exclusions remain
explicit; none can be averaged into a generic "baselines complete" claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "baseline-coverage-audit-v1"
RAG_SCOPE = re.compile(
    r"^common_protocol visual CE-G; (?P<dataset>[^;]+); "
    r"(?P<model>[^;]+); (?P<arm>no_context|rag); (?P<stage>T2_n32|T3_n200)$"
)
OE_FULL_SCOPE = re.compile(
    r"^qualified raw OE generation; (?P<dataset>[^;]+); "
    r"(?P<model>[^;]+); (?P<arm>[^;]+); clinical claim evaluation pending$"
)
MITIGATION_SCOPE = re.compile(
    r"^canonical OE-VQA mitigation smoke; (?P<dataset>[^;]+); "
    r"(?P<model>[^;]+); (?P<method>[^;]+); T2_n32$"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw = value.split("=", 1)
    if not name or not raw:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw)


def _unique_names(rows: Iterable[dict[str, Any]], *, label: str) -> list[str]:
    names = [str(row.get("name", "")) for row in rows]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{label} method names are empty or duplicated")
    return names


def _method_cells(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for method in evidence["methods"]:
        for stage_name, stage in method["stages"].items():
            for item in stage.get("evidence", []):
                scope = str(item.get("evidence_scope", ""))
                match = RAG_SCOPE.fullmatch(scope)
                if match:
                    cells.append(
                        {
                            "dataset": match.group("dataset"),
                            "task": "ce_generation",
                            "model": match.group("model"),
                            "method": (
                                "greedy"
                                if match.group("arm") == "no_context"
                                else "shared_medical_rag"
                            ),
                            "stage": match.group("stage"),
                            "artifact_sha256": item.get("artifact_sha256"),
                        }
                    )
                    continue
                match = OE_FULL_SCOPE.fullmatch(scope)
                if match:
                    cells.append(
                        {
                            "dataset": match.group("dataset"),
                            "task": "oe_vqa",
                            "model": match.group("model"),
                            "method": "greedy",
                            "stage": "T3_n200",
                            "artifact_sha256": item.get("artifact_sha256"),
                        }
                    )
                    continue
                match = MITIGATION_SCOPE.fullmatch(scope)
                if match:
                    cells.append(
                        {
                            "dataset": match.group("dataset"),
                            "task": "oe_vqa",
                            "model": match.group("model"),
                            "method": match.group("method"),
                            "stage": "T2_n32",
                            "artifact_sha256": item.get("artifact_sha256"),
                        }
                    )
    unique = {
        (
            row["dataset"],
            row["task"],
            row["model"],
            row["method"],
            row["stage"],
            row["artifact_sha256"],
        ): row
        for row in cells
    }
    return sorted(
        unique.values(),
        key=lambda row: (
            row["dataset"],
            row["task"],
            row["model"],
            row["method"],
            row["stage"],
        ),
    )


def audit(
    *,
    config_path: Path,
    t0_path: Path,
    evidence_path: Path,
    registry_path: Path,
    native_acceptance_path: Path,
    rag_causal_path: Path,
    report_audits: list[tuple[str, Path]],
    physician_analysis_path: Path | None = None,
) -> dict[str, Any]:
    config = _load(config_path)
    t0 = _load(t0_path)
    evidence = _load(evidence_path)
    native = _load(native_acceptance_path)
    rag = _load(rag_causal_path)

    config_names = _unique_names(config["methods"], label="config")
    t0_names = _unique_names(t0["methods"], label="T0")
    evidence_names = _unique_names(evidence["methods"], label="evidence")
    if config_names != t0_names or config_names != evidence_names:
        raise ValueError("method order/identity differs across config, T0, and evidence")
    if t0.get("config_sha256") != sha256_file(config_path):
        raise ValueError("T0 does not bind the current method configuration")
    if evidence.get("t0_audit_sha256") != sha256_file(t0_path):
        raise ValueError("evidence ladder does not bind the current T0 audit")
    if evidence.get("artifact_registry_sha256") != sha256_file(registry_path):
        raise ValueError("evidence ladder does not bind the current registry")
    if evidence.get("summary", {}).get("stale_registry_events") != 0:
        raise ValueError("evidence ladder contains stale registry events")

    methods: list[dict[str, Any]] = []
    for config_row, evidence_row in zip(config["methods"], evidence["methods"]):
        stages = {
            name: stage["status"] for name, stage in evidence_row["stages"].items()
        }
        methods.append(
            {
                "name": config_row["name"],
                "family": config_row["family"],
                "tracks": config_row["tracks"],
                "tasks": config_row["tasks"],
                "stages": stages,
                "cutoff": config_row["cutoff"],
            }
        )

    t0_pass = [row["name"] for row in methods if row["stages"]["T0"] == "pass"]
    t0_excluded = [
        row["name"] for row in methods if row["stages"]["T0"] != "pass"
    ]
    t1_missing = [
        row["name"]
        for row in methods
        if row["stages"]["T0"] == "pass" and row["stages"]["T1"] != "pass"
    ]
    t2_missing = [
        row["name"]
        for row in methods
        if row["stages"]["T0"] == "pass" and row["stages"]["T2"] != "pass"
    ]
    full_pass = [row["name"] for row in methods if row["stages"]["full"] == "pass"]

    cells = _method_cells(evidence)
    cell_index = {
        (row["dataset"], row["task"], row["model"], row["method"], row["stage"])
        for row in cells
    }
    primary_models = {"huatuo", "hulu", "llava"}
    visual_ce_rows = []
    for dataset in ("iuxray", "mimic"):
        for method in ("greedy", "shared_medical_rag"):
            models = sorted(
                model
                for model in primary_models
                if (dataset, "ce_generation", model, method, "T3_n200")
                in cell_index
            )
            visual_ce_rows.append(
                {
                    "dataset": dataset,
                    "task": "ce_generation",
                    "method": method,
                    "models": models,
                    "three_model_T3_complete": set(models) == primary_models,
                }
            )

    native_models = {str(row["model"]) for row in native.get("models", [])}
    oe_models = {
        row["model"]
        for row in cells
        if row["dataset"] == "vqa-rad"
        and row["task"] == "oe_vqa"
        and row["method"] == "greedy"
        and row["stage"] == "T3_n200"
    }
    if native.get("passed") is not True or not native_models.issubset(oe_models):
        raise ValueError("native OE acceptance is inconsistent with the evidence ladder")

    report_rows = []
    for model, path in sorted(report_audits):
        payload = _load(path)
        report_rows.append(
            {
                "dataset": "mimic",
                "task": "report_generation",
                "model": model,
                "audit_path": str(path.resolve()),
                "audit_sha256": sha256_file(path),
                "n": payload.get("n_rows"),
                "admissible": payload.get("admissible_for_report_generation_claim")
                is True,
                "invalid_reasons": payload.get("invalid_reasons", []),
            }
        )

    rag_supported = rag.get("supported", [])
    clinical_analysis_present = bool(
        physician_analysis_path is not None and physician_analysis_path.is_file()
    )
    dataset_matrix = [
        {
            "dataset": "vqa-rad",
            "task": "oe_vqa",
            "status": "generation_qualified_clinical_scoring_pending",
            "qualified_greedy_models": sorted(oe_models),
            "required_models": sorted(primary_models),
            "clinical_analysis_present": clinical_analysis_present,
        },
        {
            "dataset": "slake",
            "task": "oe_vqa",
            "status": "conditional_not_run",
            "reason": "deferred until a mechanism or mitigation passes its primary gate",
        },
        {
            "dataset": "pathvqa",
            "task": "oe_vqa",
            "status": "conditional_not_run",
            "reason": "deferred until a mechanism or mitigation passes its primary gate",
        },
        *visual_ce_rows,
        *report_rows,
    ]

    gates = {
        "configuration_closure": True,
        "registry_fresh": True,
        "source_qualification_complete": len(t0_pass) + len(t0_excluded)
        == len(methods),
        "t1_identity_complete_for_t0_pass": not t1_missing,
        "t2_functional_complete_for_t0_pass": not t2_missing,
        "clinical_claim_evaluation_complete": clinical_analysis_present,
        "common_rag_causal_grounding_passed": bool(rag_supported),
        "report_generation_controls_passed": bool(report_rows)
        and all(row["admissible"] for row in report_rows),
        "generic_vlm_control_present": False,
        "paper_main_table_authorized": False,
    }
    provenance = {
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "t0": {"path": str(t0_path.resolve()), "sha256": sha256_file(t0_path)},
        "evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": sha256_file(evidence_path),
        },
        "registry": {
            "path": str(registry_path.resolve()),
            "sha256": sha256_file(registry_path),
        },
        "native_acceptance": {
            "path": str(native_acceptance_path.resolve()),
            "sha256": sha256_file(native_acceptance_path),
        },
        "rag_causal": {
            "path": str(rag_causal_path.resolve()),
            "sha256": sha256_file(rag_causal_path),
        },
    }
    for row in report_rows:
        provenance[f"report_audit:{row['model']}"] = {
            "path": row["audit_path"],
            "sha256": row["audit_sha256"],
        }
    fingerprint_payload = {
        "version": VERSION,
        "inputs": {name: row["sha256"] for name, row in provenance.items()},
        "method_names": config_names,
        "gates": gates,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "version": VERSION,
        "status": "partial_no_efficacy_table",
        "paper_baseline_claim_authorized": False,
        "methods": methods,
        "summary": {
            "method_count": len(methods),
            "t0_pass": t0_pass,
            "t0_not_admissible": t0_excluded,
            "t1_missing_after_t0_pass": t1_missing,
            "t2_missing_after_t0_pass": t2_missing,
            "full_pass": full_pass,
            "registered_dataset_model_method_cells": len(cells),
            "rag_supported_cells": rag_supported,
        },
        "dataset_matrix": dataset_matrix,
        "gates": gates,
        "immediate_actions": [
            "complete the frozen blinded physician OE analysis before any T3 mitigation promotion",
            "retain shared RAG as a failed causal-grounding cutoff, not a positive baseline",
            "qualify temperature/length, self-consistency, and calibrated-abstention T2 controls only if the surviving paper branch requires them",
            "regenerate report baselines with real/null/shuffled controls only after a report-generation branch is admitted",
            "add SLAKE, PathVQA, and a generic VLM only after the primary mechanism gate passes",
        ],
        "provenance": provenance,
        "fingerprint": fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t0", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--native-acceptance", type=Path, required=True)
    parser.add_argument("--rag-causal", type=Path, required=True)
    parser.add_argument("--report-audit", action="append", type=_named_path, default=[])
    parser.add_argument("--physician-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        config_path=args.config,
        t0_path=args.t0,
        evidence_path=args.evidence,
        registry_path=args.registry,
        native_acceptance_path=args.native_acceptance,
        rag_causal_path=args.rag_causal,
        report_audits=args.report_audit,
        physician_analysis_path=args.physician_analysis,
    )
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
