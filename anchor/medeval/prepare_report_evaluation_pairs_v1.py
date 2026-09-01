#!/usr/bin/env python3
"""Join generated reports to frozen references without losing failed outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hashing import sha256_file

VERSION = "report-evaluation-pairs-v1"


def qid(row: dict, index: int) -> str:
    for key in ("qid", "question_id", "id", "sample_id"):
        if row.get(key) is not None:
            return str(row[key])
    return str(index)


def prediction(row: dict) -> str:
    for key in ("text", "model_answer", "prediction", "output"):
        if key in row:
            return str(row[key] or "").strip()
    raise ValueError("generated row has no prediction field")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--answers", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--dataset",
        choices=("iuxray", "mimic", "visual_mimic"),
        required=True,
    )
    p.add_argument("--method", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    generated = []
    for path in args.answers:
        generated.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    expected = {qid(row, index): row for index, row in enumerate(manifest)}
    observed = {qid(row, index): row for index, row in enumerate(generated)}
    if len(expected) != len(manifest) or len(observed) != len(generated):
        raise ValueError("duplicate qid in manifest or answers")
    if set(expected) != set(observed):
        raise ValueError(f"qid mismatch: missing={len(set(expected)-set(observed))} extra={len(set(observed)-set(expected))}")
    rows = []
    for item_id, reference in expected.items():
        image = str(reference.get("img_name", reference.get("image", "")))
        parts = Path(image).parts
        # MIMIC paths contain both a two-digit shard (for example ``p15``)
        # and the actual patient directory (``p15518538``).  The latter is
        # the longest numeric p-component; taking the first component would
        # collapse hundreds of patients into ten invalid bootstrap clusters.
        patient_parts = [part[1:] for part in parts if part.startswith("p") and part[1:].isdigit()]
        study_parts = [part[1:] for part in parts if part.startswith("s") and part[1:].isdigit()]
        patient = max(patient_parts, key=len, default=None)
        study = max(study_parts, key=len, default=None)
        rows.append({
            "item_id": item_id,
            "qid": item_id,
            "patient_id": patient or study or item_id,
            "study_id": study or item_id,
            "dataset": args.dataset,
            "task": "report_generation",
            "modality": "chest_radiograph",
            "clinical_metric_family": "chest_radiograph",
            "method": args.method,
            "model": args.model,
            "image": image,
            "ground_truth": str(reference["answer"]).strip(),
            "model_answer": prediction(observed[item_id]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    patient_clusters = len({row["patient_id"] for row in rows})
    study_clusters = len({row["study_id"] for row in rows})
    if args.dataset == "mimic" and len(rows) == 694:
        if patient_clusters != 218 or study_clusters != 647:
            raise ValueError(
                "frozen MIMIC report manifest must resolve to 218 patients and "
                f"647 studies, got {patient_clusters} and {study_clusters}"
            )
    if args.dataset == "visual_mimic" and len(rows) == 490:
        if patient_clusters != 193 or study_clusters != 483:
            raise ValueError(
                "frozen Visual-MIMIC report manifest must resolve to 193 patients "
                f"and 483 studies, got {patient_clusters} and {study_clusters}"
            )
    audit = {"version": VERSION, "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest), "answers": [str(path.resolve()) for path in args.answers], "answer_sha256": [sha256_file(path) for path in args.answers], "output": str(args.output.resolve()), "output_sha256": sha256_file(args.output), "rows": len(rows), "patient_clusters": patient_clusters, "study_clusters": study_clusters, "empty_predictions": sum(not row["model_answer"] for row in rows), "qid_exact": True, "reference_source": "frozen manifest only"}
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
