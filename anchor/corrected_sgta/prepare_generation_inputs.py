#!/usr/bin/env python3
"""Prepare source-separated OE/report inputs for the MedHEval LLaVA runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from corrected_sgta.audit_experiment_matrix import (
    REPO_ROOT,
    DatasetSpec,
    dataset_specs,
    prompt_value,
    reference_value,
    resolve_image,
    load_rows,
)


REPORT_PROMPT = (
    "You are a professional radiologist. You are provided with a medical image. "
    "Generate a concise report describing the key findings. Only output the report."
)


def convert_row(spec: DatasetSpec, row: dict[str, Any], index: int, image_root: Path) -> dict[str, Any] | None:
    image = resolve_image(spec, row)
    if image is None:
        return None
    try:
        relative = image.relative_to(image_root)
    except ValueError:
        return None
    prompt = prompt_value(row, spec.task)
    if spec.task == "report_generation" and not prompt:
        prompt = REPORT_PROMPT
    answer = reference_value(row)
    if not prompt or not answer:
        return None
    return {
        "id": str(row.get("id", row.get("qid", row.get("question_id", index)))),
        "qid": str(row.get("qid", row.get("question_id", row.get("id", index)))),
        "img_name": str(relative),
        "question": prompt.replace("<image>", "").strip(),
        "answer": answer,
        "source": spec.source,
        "dataset": spec.dataset,
        "task": spec.task,
    }


def infer_image_root(spec: DatasetSpec, rows: list[dict[str, Any]]) -> Path | None:
    if spec.image_root is not None and spec.image_root.exists():
        return spec.image_root
    resolved = [resolve_image(spec, row) for row in rows[:50]]
    resolved = [path for path in resolved if path is not None]
    if not resolved:
        return None
    # Pick the deepest common parent that keeps relative paths stable enough.
    return Path(os.path.commonpath([str(path.parent) for path in resolved]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "corrected_runs/high_efficiency/inputs")
    parser.add_argument("--tasks", nargs="*", default=["open_vqa", "report_generation"])
    parser.add_argument("--sources", nargs="*", default=["mmedrag", "medheval", "chexpert_report"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in dataset_specs():
        if spec.task not in args.tasks or spec.source not in args.sources:
            continue
        if args.datasets is not None and spec.dataset not in args.datasets:
            continue
        rows = load_rows(spec.path)
        image_root = infer_image_root(spec, rows)
        if image_root is None:
            manifest.append({"source": spec.source, "dataset": spec.dataset, "task": spec.task, "status": "blocked", "reason": "no image root"})
            continue
        converted = []
        for index, row in enumerate(rows):
            item = convert_row(spec, row, index, image_root)
            if item is not None:
                converted.append(item)
            if args.limit and len(converted) >= args.limit:
                break
        if not converted:
            manifest.append({"source": spec.source, "dataset": spec.dataset, "task": spec.task, "status": "blocked", "reason": "no convertible rows"})
            continue
        name = f"{spec.source}.{spec.dataset}.{spec.task}.json"
        output = args.out / name
        output.write_text(json.dumps(converted, indent=2))
        manifest.append(
            {
                "source": spec.source,
                "dataset": spec.dataset,
                "task": spec.task,
                "status": "ready",
                "n": len(converted),
                "image_root": str(image_root),
                "question_file": str(output),
            }
        )
    manifest_path = args.out / "generation_inputs_manifest.json"
    manifest_path.write_text(json.dumps({"inputs": manifest}, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "ready": sum(1 for row in manifest if row["status"] == "ready"), "total": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
