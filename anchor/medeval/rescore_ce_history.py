#!/usr/bin/env python3
"""Non-destructively rescore historical CE-G answer artifacts.

Only files whose rows contain an explicit binary reference and a generated
prediction are admitted. New versioned reports are written under a separate
root; source answers and historical metrics are never modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file
from anchor.medeval.store import atomic_write_json
from corrected_sgta.evaluate_medheval_answers import (
    PROTOCOL_VERSION,
    _legacy_semantic_binary,
    binary_inconsistency,
    normalize_binary_reference,
    parse_answer,
)
from corrected_sgta.evaluate_rule_vqa import rule_normalized_prediction


VERSION = "ce-g-leading-history-rescore-v2"
PREDICTION_KEYS = ("text", "prediction", "output")
REFERENCE_KEYS = ("gt_ans", "gt_answer", "reference", "ground_truth")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object row at {path}:{number}")
        rows.append(value)
    return rows


def first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def prediction(row: dict[str, Any]) -> Any:
    value = first(row, PREDICTION_KEYS)
    if value is None and "gt_answer" in row:
        value = row.get("answer")
    return value


def qid(row: dict[str, Any], index: int) -> str:
    return str(
        row.get("question_id", row.get("qid", row.get("id", f"row-{index}")))
    )


def is_binary_answer_artifact(rows: list[dict[str, Any]]) -> bool:
    return any(
        prediction(row) is not None
        and normalize_binary_reference(first(row, REFERENCE_KEYS)) is not None
        for row in rows
    )


def rescore(path: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_jsonl(path)
    if not is_binary_answer_artifact(rows):
        raise ValueError("not an all-binary generated-answer artifact")
    source_row_count = len(rows)
    rows = [
        row
        for row in rows
        if prediction(row) is not None
        and normalize_binary_reference(first(row, REFERENCE_KEYS)) is not None
    ]
    records: list[dict[str, Any]] = []
    correct = valid = inconsistent = 0
    legacy_mismatch = legacy_invalid = legacy_ambiguous = rule_flip = 0
    for index, row in enumerate(rows):
        raw_prediction = str(prediction(row) or "")
        truth = normalize_binary_reference(first(row, REFERENCE_KEYS))
        assert truth is not None
        parsed = parse_answer(raw_prediction, answer_type="binary")
        decision = parsed.labels[0] if parsed.labels else None
        legacy = _legacy_semantic_binary(raw_prediction)
        legacy_decision = legacy.labels[0] if legacy.labels else None
        rule_decision = rule_normalized_prediction(raw_prediction)
        row_inconsistent = binary_inconsistency(raw_prediction, decision)
        row_correct = decision == truth
        valid += int(decision is not None)
        correct += int(row_correct)
        inconsistent += int(row_inconsistent)
        legacy_mismatch += int(
            decision is not None and legacy_decision != decision
        )
        legacy_invalid += int(legacy_decision is None)
        legacy_ambiguous += int(legacy.status == "ambiguous")
        rule_flip += int(decision is not None and rule_decision != decision)
        records.append(
            {
                "question_id": qid(row, index),
                "reference": truth,
                "prediction_text": raw_prediction,
                "leading_decision": decision,
                "valid": decision is not None,
                "correct": row_correct,
                "answer_inconsistency": row_inconsistent,
                "legacy_semantic_decision": legacy_decision,
                "legacy_semantic_status": legacy.status,
                "rule_normalized_decision": rule_decision,
            }
        )
    n = len(records)
    source_sha = sha256_file(path)
    metric = {
        "protocol_version": VERSION,
        "ce_evaluator_version": PROTOCOL_VERSION,
        "primary_metric": "ce_g_leading.accuracy_invalid_as_error",
        "source": str(path.resolve()),
        "source_relative": str(path.resolve().relative_to(root.resolve())),
        "source_sha256": source_sha,
        "n": n,
        "source_row_count": source_row_count,
        "valid": valid,
        "valid_parse_rate": valid / n,
        "correct": correct,
        "accuracy_invalid_as_error": correct / n,
        "answer_inconsistency_count": inconsistent,
        "legacy_semantic_mismatch_count": legacy_mismatch,
        "legacy_semantic_invalid_count": legacy_invalid,
        "legacy_semantic_ambiguous_count": legacy_ambiguous,
        "rule_normalized_flip_count": rule_flip,
        "artifact_status": "rescore_only" if valid / n >= 0.95 else "regenerate",
        "historical_metrics_modified": False,
    }
    return metric, records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = args.output_root.resolve()
    entries = []
    skipped = 0
    candidates = sorted(
        path
        for path in root.rglob("*.jsonl")
        if "answer" in path.name.lower() and output_root not in path.resolve().parents
    )
    for path in candidates:
        try:
            metric, records = rescore(path, root)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue
        artifact_id = f"{path.stem}-{metric['source_sha256'][:16]}"
        destination = output_root / artifact_id
        metrics_path = destination / "metrics.json"
        records_path = destination / "records.jsonl"
        if not metrics_path.exists():
            atomic_write_json(metrics_path, metric)
            write_jsonl(records_path, records)
        entries.append(
            {
                **metric,
                "metrics": str(metrics_path),
                "records": str(records_path),
            }
        )
    summary = {
        "protocol_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "output_root": str(output_root),
        "candidate_files": len(candidates),
        "admitted_files": len(entries),
        "skipped_files": skipped,
        "n": sum(entry["n"] for entry in entries),
        "legacy_semantic_mismatch_count": sum(
            entry["legacy_semantic_mismatch_count"] for entry in entries
        ),
        "legacy_semantic_invalid_count": sum(
            entry["legacy_semantic_invalid_count"] for entry in entries
        ),
        "legacy_semantic_ambiguous_count": sum(
            entry["legacy_semantic_ambiguous_count"] for entry in entries
        ),
        "rule_normalized_flip_count": sum(
            entry["rule_normalized_flip_count"] for entry in entries
        ),
        "status_counts": {
            status: sum(entry["artifact_status"] == status for entry in entries)
            for status in ("rescore_only", "regenerate")
        },
        "entries": entries,
    }
    atomic_write_json(output_root / "index.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
