#!/usr/bin/env python3
"""Finalize the visual-claim common RAG ladder without pooling datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_registry import append_qualification, qualification_for
from .store import atomic_write_json


VERSION = "common-rag-visual-ce-ladder-summary-v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path, datasets: list[str], models: list[str]) -> dict:
    records = []
    for dataset in datasets:
        for model in models:
            record = {"dataset": dataset, "model": model, "levels": {}}
            for level in ("T2_n32", "T3_n200"):
                arms = {}
                for arm in ("no_context", "rag"):
                    directory = root / dataset / "visual_ce_v2" / "ladder_v3" / level / model / arm
                    qualification = directory / "qualification.json"
                    evaluation = directory / "evaluation.json"
                    arms[arm] = {
                        "present": qualification.is_file() and evaluation.is_file(),
                        "qualification": read(qualification) if qualification.is_file() else None,
                        "evaluation": read(evaluation) if evaluation.is_file() else None,
                    }
                record["levels"][level] = arms
            comparison = root / dataset / "visual_ce_v2" / "ladder_v3" / "T3_n200" / model / "comparison.json"
            record["comparison"] = read(comparison) if comparison.is_file() else None
            context_audit = root / dataset / "visual_ce_v2" / "ladder_v3" / "T3_n200" / model / "context_budget_audit.json"
            record["context_budget_audit"] = (
                read(context_audit) if context_audit.is_file() else None
            )
            record["t2_passed"] = all(
                arm["present"] and arm["qualification"].get("passed") is True
                for arm in record["levels"]["T2_n32"].values()
            )
            context_passed = bool(
                model != "hulu"
                or (
                    record["context_budget_audit"] is not None
                    and record["context_budget_audit"].get("passed") is True
                )
            )
            record["t3_passed"] = context_passed and all(
                arm["present"]
                and arm["qualification"].get("passed") is True
                and arm["evaluation"].get("invalid_ground_truth") == 0
                for arm in record["levels"]["T3_n200"].values()
            ) and record["comparison"] is not None
            record["full_authorized"] = bool(
                record["t3_passed"]
                and record["comparison"].get("full_run_authorized") is True
            )
            records.append(record)
    return {
        "protocol_version": VERSION,
        "dataset_pooling_forbidden": True,
        "records": records,
        "full_authorized": [
            {"dataset": row["dataset"], "model": row["model"]}
            for row in records if row["full_authorized"]
        ],
    }


def register(summary: dict, root: Path, registry: Path) -> list[str]:
    events = []
    for record in summary["records"]:
        for level, arms in record["levels"].items():
            for arm, values in arms.items():
                if not values["present"]:
                    continue
                directory = root / record["dataset"] / "visual_ce_v2" / "ladder_v3" / level / record["model"] / arm
                qualification = directory / "qualification.json"
                answer = directory / "answers.jsonl"
                evaluation = values["evaluation"]
                passed = bool(
                    values["qualification"].get("passed") is True
                    and evaluation.get("invalid_ground_truth") == 0
                )
                value = qualification_for(
                    answer,
                    status="admissible" if passed else "regenerate",
                    evaluator_version=str(evaluation.get("protocol_version", "missing")),
                    evidence_scope=(
                        f"common_protocol visual CE-G; {record['dataset']}; "
                        f"{record['model']}; {arm}; {level}"
                    ),
                    reason=(
                        "leading-label, observability, truncation, and reference gates passed"
                        if passed else "one or more frozen visual CE qualification gates failed"
                    ),
                    qualification=qualification,
                )
                events.append(append_qualification(registry, value)["event_id"])
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["iuxray", "mimic"])
    parser.add_argument("--models", nargs="+", default=["huatuo", "hulu", "llava"])
    args = parser.parse_args()
    result = summarize(args.root, args.datasets, args.models)
    if args.registry is not None:
        result["registry_event_ids"] = register(result, args.root, args.registry)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
