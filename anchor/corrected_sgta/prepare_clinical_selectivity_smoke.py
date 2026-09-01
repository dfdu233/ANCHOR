#!/usr/bin/env python3
"""Build a small same-state/opposite-state image-swap diagnostic manifest.

This is deliberately a grade-C mechanism probe.  MedHEval's report-derived
answers are used to ask whether raw response sensitivity distinguishes a
clinical evidence change from an irrelevant change of patient/image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


VERSION = "clinical-selectivity-smoke-v3"
DEFAULT_SOURCE = Path(
    "data/medheval/benchmark_data/Visual_Misinterpretation_Hallucination/"
    "close-ended/fine-grained/mimic_cxr_closed_pairs.json"
)
DEFAULT_IMAGE_ROOT = Path("data/medheval/images")

ONTOLOGY = {
    "pleural_effusion": {
        "include": r"pleural effusion",
        "exclude": r"large|small|moderate|minimal|trace|bilateral|unilateral|right|left",
        "question": (
            "Does this chest X-ray show pleural effusion? "
            "Answer with exactly one word: Yes, No, or Maybe."
        ),
    },
    "pulmonary_edema": {
        "include": r"pulmonary edema",
        "exclude": r"mild|moderate|severe",
        "question": (
            "Does this chest X-ray show pulmonary edema? "
            "Answer with exactly one word: Yes, No, or Maybe."
        ),
    },
}


def stable_order(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def collect_labels(source: Path, image_root: Path) -> dict[str, dict[str, list[Path]]]:
    raw = json.loads(source.read_text(encoding="utf-8"))
    answers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in raw:
        if row.get("question_type") != "binary":
            continue
        answer = str(row.get("answer", "")).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        question = str(row.get("question", "")).lower()
        for finding, contract in ONTOLOGY.items():
            if not re.search(str(contract["include"]), question):
                continue
            if re.search(str(contract["exclude"]), question):
                continue
            answers[(finding, str(row["img_name"]))].add(answer)

    grouped: dict[str, dict[str, list[Path]]] = {
        finding: {"yes": [], "no": []} for finding in ONTOLOGY
    }
    for (finding, image_name), values in answers.items():
        if len(values) != 1:
            continue
        image_path = (image_root / image_name).resolve()
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
        except (OSError, ValueError):
            continue
        grouped[finding][next(iter(values))].append(image_path)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors-per-state", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.anchors_per_state <= 0:
        raise ValueError("anchors-per-state must be positive")

    grouped = collect_labels(args.source, args.image_root)
    rows: list[dict[str, object]] = []
    for finding, states in grouped.items():
        ordered = {
            state: sorted(
                paths,
                key=lambda path: stable_order(args.seed, finding, state, str(path)),
            )
            for state, paths in states.items()
        }
        required = 2 * args.anchors_per_state
        if any(len(paths) < required for paths in ordered.values()):
            counts = {state: len(paths) for state, paths in ordered.items()}
            raise RuntimeError(f"insufficient unique images for {finding}: {counts}")
        for state in ("yes", "no"):
            opposite = "no" if state == "yes" else "yes"
            for index in range(args.anchors_per_state):
                triplet_id = f"{finding}:{state}:{index}"
                experiment_split = "dev" if index % 2 == 0 else "test"
                members = (
                    ("anchor", ordered[state][index], state),
                    (
                        "same_state_swap",
                        ordered[state][index + args.anchors_per_state],
                        state,
                    ),
                    ("opposite_state_swap", ordered[opposite][index], opposite),
                )
                for role, image_path, member_state in members:
                    positive = 3 if member_state == "yes" else 0
                    rows.append(
                        {
                            "version": VERSION,
                            "triplet_id": triplet_id,
                            "pair_index": f"{finding}:{index}",
                            "swap_role": role,
                            "image_id": f"{triplet_id}:{role}",
                            "finding": finding,
                            "question": ONTOLOGY[finding]["question"],
                            "image_path": str(image_path),
                            "positive_votes": positive,
                            "reader_count": 3,
                            "reader_support": positive / 3,
                            "reader_state": "supported" if positive else "refuted",
                            "reference_source": "medheval_report_derived_binary_qa",
                            "evidence_grade": "C",
                            "formal_reference": False,
                            "experiment_split": experiment_split,
                        }
                    )
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "version": VERSION,
                "records": len(rows),
                "triplets": len(rows) // 3,
                "findings": sorted(grouped),
                "formal_reference": False,
                "evidence_grade": "C",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
