#!/usr/bin/env python3
"""Aggregate a completed mitigation matrix under the strict OE-VQA contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.evaluate_oe_vqa import (
    PROTOCOL_ID,
    _load_json,
    _load_jsonl,
    align_and_score,
    paired_summary,
    summarize,
)
from anchor.medeval.hashing import sha256_file
from anchor.medeval.legacy import audit_legacy_answers


def answer_files(method_dir: Path) -> list[Path]:
    return sorted(method_dir.glob("chunk_*.answers.jsonl"))


def evaluate_matrix(
    *,
    manifest_path: Path,
    run_root: Path,
    method_names: list[str],
    output_root: Path,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    expected_ids = [
        str(row.get("qid", row.get("question_id", row.get("id"))))
        for row in manifest
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "interpretation": "lexical proxies only; not clinical hallucination correctness",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "run_root": str(run_root.resolve()),
        "methods": {},
    }
    baseline_rows = None
    baseline_paths = answer_files(run_root / "greedy")
    try:
        if not baseline_paths:
            raise ValueError("no greedy answer chunks")
        baseline_rows = align_and_score(manifest, _load_jsonl(baseline_paths))
    except (OSError, ValueError) as error:
        payload["baseline_failure"] = str(error)

    for method in method_names:
        paths = answer_files(run_root / method)
        record: dict[str, Any] = {
            "answer_files": [str(path.resolve()) for path in paths],
            "answer_sha256": [sha256_file(path) for path in paths],
        }
        try:
            if not paths:
                raise ValueError("no answer chunks")
            # Once a method has entered the frozen full run, dominance and
            # brevity are outcomes, not infrastructure failures. Record these
            # diagnostics globally (never per an arbitrary chunk) and score the
            # method unchanged; otherwise evaluation would censor poor methods.
            combined_audit_path = output_root / f"{method}.answers.audit.jsonl"
            combined_audit_path.write_text(
                "".join(path.read_text() for path in paths)
            )
            output_audit = audit_legacy_answers(
                combined_audit_path,
                expected_ids,
                allow_short_answers=True,
                enforce_behavioral_quality=False,
            )
            record["output_audit"] = output_audit
            if not output_audit["aligned"] or output_audit["empty_predictions"]:
                raise ValueError("full answers failed structural output audit")
            rows = align_and_score(manifest, _load_jsonl(paths))
            record["status"] = "done"
            record["absolute"] = summarize(rows, replicates=replicates, seed=seed)
            if method != "greedy":
                if baseline_rows is None:
                    raise ValueError("greedy baseline unavailable for paired comparison")
                record["paired_vs_greedy"] = paired_summary(
                    rows,
                    baseline_rows,
                    replicates=replicates,
                    seed=seed,
                )
        except (OSError, ValueError) as error:
            record["status"] = "failed"
            record["failure_reason"] = str(error)
        payload["methods"][method] = record
        (output_root / f"{method}.json").write_text(json.dumps(record, indent=2) + "\n")
    payload["completed_methods"] = sorted(
        method for method, row in payload["methods"].items() if row["status"] == "done"
    )
    payload["failed_methods"] = {
        method: row.get("failure_reason")
        for method, row in payload["methods"].items()
        if row["status"] != "done"
    }
    greedy_record = payload["methods"].get("greedy", {})
    greedy_warnings = set(
        greedy_record.get("output_audit", {}).get("behavioral_warnings", [])
    )
    greedy_semantic_empty_rate = (
        greedy_record.get("absolute", {})
        .get("output_diagnostics", {})
        .get("empty_rate")
    )
    validity_reasons: list[str] = []
    if greedy_record.get("status") != "done":
        validity_reasons.append("greedy_common_baseline_unavailable")
    if "function_word_only_predictions_at_least_50_percent" in greedy_warnings:
        validity_reasons.append("greedy_common_port_is_function_word_fragmented")
    if (
        greedy_semantic_empty_rate is not None
        and float(greedy_semantic_empty_rate) >= 0.50
    ):
        validity_reasons.append("greedy_semantic_empty_rate_at_least_50_percent")
    payload["common_plumbing_valid"] = not validity_reasons
    payload["validity_reasons"] = validity_reasons
    payload["scientifically_comparable_methods"] = (
        payload["completed_methods"] if not validity_reasons else []
    )
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    methods = [str(method) for method in selection.get("methods", [])]
    if not methods:
        args.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol_id": PROTOCOL_ID,
            "status": "skipped",
            "reason": "no smoke-qualified methods",
            "selection": str(args.selection.resolve()),
        }
        (args.output_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return
    payload = evaluate_matrix(
        manifest_path=args.manifest,
        run_root=args.run_root,
        method_names=methods,
        output_root=args.output_root,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps({
        "output": str(args.output_root / "summary.json"),
        "completed_methods": payload["completed_methods"],
        "failed_methods": payload["failed_methods"],
    }, indent=2))


if __name__ == "__main__":
    main()
