"""Build MedHEval subsets from blank-image visual dependency audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


SPLITS = (
    "visual_dependent",
    "visual_independent",
    "prediction_changed",
    "image_helpful",
    "image_harmful",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        action="append",
        nargs=3,
        metavar=("NAME", "DATASET", "AUDIT_JSONL"),
        required=True,
        help="Dataset name, original JSON path, and visual dependency audit JSONL.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_json_compact(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))


def accuracy(rows: Iterable[dict], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if key in row]
    return mean(values) if values else None


def split_predicates(row: dict) -> dict[str, bool]:
    image_correct = bool(row.get("image_correct"))
    blank_correct = bool(row.get("blank_correct"))
    return {
        "visual_dependent": bool(row.get("visual_dependent")),
        "visual_independent": not bool(row.get("visual_dependent")),
        "prediction_changed": bool(row.get("prediction_changed")),
        "image_helpful": image_correct and not blank_correct,
        "image_harmful": (not image_correct) and blank_correct,
    }


def summarize_audit(name: str, audit_rows: list[dict], split_counts: dict[str, int]) -> dict:
    ok = [row for row in audit_rows if row.get("status") == "ok"]
    return {
        "name": name,
        "n": len(ok),
        "image_accuracy": accuracy(ok, "image_correct"),
        "blank_accuracy": accuracy(ok, "blank_correct"),
        "prediction_change_rate": mean(bool(row.get("prediction_changed")) for row in ok) if ok else None,
        "visual_dependent_rate": mean(bool(row.get("visual_dependent")) for row in ok) if ok else None,
        "mean_max_prob_delta": mean(float(row.get("max_prob_delta", 0.0)) for row in ok) if ok else None,
        "mean_max_logit_delta": mean(float(row.get("max_logit_delta", 0.0)) for row in ok) if ok else None,
        "split_counts": split_counts,
    }


def main() -> None:
    args = parse_args()
    aggregate = {
        "version": "visual-dependency-subsets-v1",
        "subset_rule": (
            "visual_dependent iff normal-vs-blank prediction changes or confidence/logit "
            "movement crosses the audit thresholds"
        ),
        "datasets": [],
    }
    for name, dataset_path_raw, audit_path_raw in args.spec:
        dataset_path = Path(dataset_path_raw)
        audit_path = Path(audit_path_raw)
        dataset_rows = json.loads(dataset_path.read_text())
        by_qid = {str(row["qid"]): row for row in dataset_rows}
        audit_rows = [row for row in read_jsonl(audit_path) if row.get("status") == "ok"]
        predicates_by_qid = {str(row["qid"]): split_predicates(row) for row in audit_rows}
        split_counts = {}
        qid_lists = {}
        for split in SPLITS:
            qids = [
                qid
                for qid, predicates in predicates_by_qid.items()
                if predicates[split] and qid in by_qid
            ]
            qid_lists[split] = qids
            split_counts[split] = len(qids)
            split_rows = [by_qid[qid] for qid in qids]
            write_json_compact(args.output_dir / f"{name}.{split}.json", split_rows)
            write_json(args.output_dir / f"{name}.{split}.qids.json", qids)
        write_json(
            args.output_dir / f"{name}.visual_dependency_labels.json",
            {
                "name": name,
                "dataset": str(dataset_path.resolve()),
                "audit": str(audit_path.resolve()),
                "labels_by_qid": predicates_by_qid,
                "qids": qid_lists,
            },
        )
        aggregate["datasets"].append(
            {
                **summarize_audit(name, audit_rows, split_counts),
                "dataset": str(dataset_path.resolve()),
                "audit": str(audit_path.resolve()),
            }
        )
    write_json(args.summary_output, aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
