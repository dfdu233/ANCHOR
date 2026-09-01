#!/usr/bin/env python3
"""Freeze the image-available eight-modality OmniMedVQA evaluation manifest.

The released archive mixes open- and restricted-access source datasets.  This
preparer never treats a QA row as evaluable unless its referenced image exists
locally and its ground truth identifies exactly one of the four released
options.  It preserves source IDs and emits a deterministic modality-balanced
smoke manifest alongside the full manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file


VERSION = "omnimedvqa-eight-modality-v1"
REFERENCE_MODALITIES = (
    "MRI",
    "Ultrasound",
    "OCT",
    "Dermoscopy",
    "CT",
    "X-Ray",
    "Microscopy",
    "Fundus",
)


def normalize(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def reference_modality(value: object) -> str | None:
    text = normalize(value)
    # The official release contains the typo ``Mag-netic Resonance Imaging``.
    if "resonance imaging" in text or text in {"mr", "mri"}:
        return "MRI"
    if "ultrasound" in text or "sonograph" in text:
        return "Ultrasound"
    if "optical coherence tomography" in text or text == "oct":
        return "OCT"
    if "dermoscop" in text:
        return "Dermoscopy"
    if "computed tomography" in text or text in {"ct", "ct scan"}:
        return "CT"
    if "x ray" in text or "radiograph" in text or "mammograph" in text:
        return "X-Ray"
    if "microscop" in text or "histopatholog" in text:
        return "Microscopy"
    if "fundus" in text or "retinal photograph" in text:
        return "Fundus"
    return None


def option_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = []
    for label in "ABCDEF":
        key = f"option_{label}"
        if key not in row:
            break
        text = str(row[key]).strip()
        if not text:
            raise ValueError(f"empty {key}")
        pairs.append((label, text))
    if len(pairs) < 2:
        raise ValueError("fewer than two options")
    return pairs


def answer_label(answer: object, options: list[tuple[str, str]]) -> str:
    value = str(answer).strip()
    explicit = re.match(r"^\s*(?:answer\s*(?:is|:)?\s*)?([A-F])(?:\b|[.):])", value, re.I)
    labels = {label for label, _ in options}
    if explicit and explicit.group(1).upper() in labels:
        return explicit.group(1).upper()
    target = normalize(value)
    matches = [label for label, text in options if normalize(text) == target]
    if len(matches) != 1:
        raise ValueError("ground truth does not exactly identify one option")
    return matches[0]


def stable_order(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["modality"]), str(row["source_dataset"]), str(row["qid"]))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke-output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--smoke-per-modality", type=int, default=8)
    args = parser.parse_args()
    if args.smoke_per_modality <= 0:
        raise ValueError("--smoke-per-modality must be positive")

    qa_root = args.root / "QA_information"
    if not qa_root.is_dir():
        raise FileNotFoundError(f"missing QA_information: {qa_root}")
    qa_files = sorted(qa_root.rglob("*.json"))
    if not qa_files:
        raise FileNotFoundError(f"no QA JSON files under {qa_root}")

    rows: list[dict[str, Any]] = []
    excluded = Counter()
    source_counts = Counter()
    seen: set[str] = set()
    for qa_path in qa_files:
        payload = json.loads(qa_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            excluded["invalid_json_root"] += 1
            continue
        access = qa_path.parent.name
        for source_index, source in enumerate(payload):
            try:
                modality = reference_modality(source.get("modality_type"))
                if modality is None:
                    excluded["outside_reference_modalities"] += 1
                    continue
                image_relative = Path(str(source["image_path"]))
                image_path = args.root / image_relative
                if not image_path.is_file():
                    excluded["missing_image"] += 1
                    continue
                options = option_pairs(source)
                gt_label = answer_label(source.get("gt_answer"), options)
                source_qid = str(source.get("question_id") or f"{qa_path.stem}_{source_index}")
                qid = f"omnimedvqa::{source_qid}"
                if qid in seen:
                    raise ValueError(f"duplicate qid: {qid}")
                seen.add(qid)
                image_identity = hashlib.sha256(str(image_relative).encode()).hexdigest()
                rows.append(
                    {
                        "id": qid,
                        "qid": qid,
                        "question_id": qid,
                        "img_name": str(image_path.resolve()),
                        "image_relative": str(image_relative),
                        "image_identity": image_identity,
                        "question": str(source["question"]).strip(),
                        "choices": ", ".join(f"{label}. {text}" for label, text in options),
                        "answer": gt_label,
                        "gt_answer_text": str(source.get("gt_answer", "")).strip(),
                        "question_type": "multi-choice",
                        "source_question_type": "multi-choice",
                        "task": "multi_choice",
                        "dataset": "omnimedvqa",
                        "source": "OmniMedVQA",
                        "source_dataset": str(source.get("dataset", qa_path.stem)),
                        "source_access": access,
                        "source_qa_file": str(qa_path.relative_to(args.root)),
                        "source_row": source_index,
                        "source_modality_type": str(source.get("modality_type", "")),
                        "modality": modality,
                    }
                )
                source_counts[str(source.get("dataset", qa_path.stem))] += 1
            except (KeyError, TypeError, ValueError):
                excluded["invalid_row"] += 1

    rows.sort(key=stable_order)
    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_modality[row["modality"]].append(row)
    missing_modalities = [name for name in REFERENCE_MODALITIES if not by_modality[name]]
    if missing_modalities:
        raise RuntimeError(f"no evaluable rows for modalities: {missing_modalities}")

    smoke = []
    for modality in REFERENCE_MODALITIES:
        ordered = sorted(
            by_modality[modality],
            key=lambda row: hashlib.sha256(f"{VERSION}:{row['qid']}".encode()).hexdigest(),
        )
        smoke.extend(ordered[: args.smoke_per_modality])
    smoke.sort(key=stable_order)

    write_json(args.output, rows)
    write_json(args.smoke_output, smoke)
    manifest = {
        "version": VERSION,
        "source_root": str(args.root.resolve()),
        "qa_files": len(qa_files),
        "evaluable_rows": len(rows),
        "unique_image_paths": len({row["img_name"] for row in rows}),
        "modality_counts": {name: len(by_modality[name]) for name in REFERENCE_MODALITIES},
        "source_dataset_counts": dict(sorted(source_counts.items())),
        "excluded": dict(sorted(excluded.items())),
        "ground_truth_contract": "exact option label or exact unique option text",
        "prediction_contract": "leading option label or exact unique option text; invalid counts as error",
        "smoke_per_modality": args.smoke_per_modality,
        "smoke_rows": len(smoke),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "smoke_output": str(args.smoke_output.resolve()),
        "smoke_output_sha256": sha256_file(args.smoke_output),
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
