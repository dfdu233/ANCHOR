#!/usr/bin/env python3
"""Cache-only screen for systematic laterality-frame errors in natural outputs.

The parser is deliberately conservative.  It evaluates only positive findings
for which both the generated answer and reference contain exactly one
unambiguous left/right side.  This is a benchmark-proxy screen, not physician
adjudication and not a method efficacy result.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "cached-natural-laterality-screen-v1"

ALIASES: dict[str, tuple[str, ...]] = {
    "atelectasis": ("atelectasis", "subsegmental collapse"),
    "consolidation": ("consolidation", "airspace disease", "infiltrate"),
    "lung_opacity": ("lung opacity", "airspace opacity", "pulmonary opacity"),
    "nodule_mass": (
        "pulmonary nodule",
        "lung nodule",
        "lung mass",
        "pulmonary mass",
        "mass lesion",
    ),
    "pleural_effusion": ("pleural effusion", "pleural fluid"),
    "pleural_thickening": ("pleural thickening",),
    "pneumonia": ("pneumonia",),
    "pneumothorax": ("pneumothorax",),
    "fracture": ("fracture", "rib fracture"),
    "hilar_abnormality": ("hilar enlargement", "hilar prominence", "hilar mass"),
    "hemidiaphragm": ("hemidiaphragm", "diaphragm"),
}

NEGATION = re.compile(
    r"\b(no|not|without|absent|absence of|negative for|free of|clear of|"
    r"resolved|resolution of)\b"
)
SIDE = re.compile(r"\b(left|right)\b")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _positive_at(sentence: str, start: int) -> bool:
    return NEGATION.search(sentence[max(0, start - 55) : start]) is None


def _nearest_side(sentence: str, start: int, end: int) -> str | None:
    local = sentence[max(0, start - 55) : min(len(sentence), end + 55)]
    if re.search(r"\b(bilateral|left\s+and\s+right|right\s+and\s+left)\b", local):
        return None
    candidates: list[tuple[int, str]] = []
    for match in SIDE.finditer(sentence):
        distance = min(abs(match.start() - start), abs(match.start() - end))
        if distance <= 70:
            candidates.append((distance, match.group(1)))
    if not candidates:
        return None
    candidates.sort()
    minimum = candidates[0][0]
    tied = {side for distance, side in candidates if distance == minimum}
    if len(tied) != 1:
        # In coordinated radiology phrases ("left X and right Y"), an
        # adjective immediately before the finding binds more reliably than
        # the equally distant adjective introducing the next finding.
        preceding = [
            (start - match.start(), match.group(1))
            for match in SIDE.finditer(sentence)
            if match.start() < start and start - match.start() <= minimum
        ]
        if not preceding:
            return None
        preceding.sort()
        return preceding[0][1]
    nearest = candidates[0][1]
    return nearest


def parse_lateralized_positive_findings(text: str) -> dict[str, str]:
    observations: dict[str, list[str]] = defaultdict(list)
    lowered = str(text).lower()
    for sentence in SENTENCE_SPLIT.split(lowered):
        if not sentence.strip():
            continue
        for finding, aliases in ALIASES.items():
            hits = []
            for alias in aliases:
                hits.extend(re.finditer(rf"\b{re.escape(alias)}s?\b", sentence))
            for hit in hits:
                if not _positive_at(sentence, hit.start()):
                    continue
                side = _nearest_side(sentence, hit.start(), hit.end())
                if side is not None:
                    observations[finding].append(side)
    output = {}
    for finding, sides in observations.items():
        unique = set(sides)
        if len(unique) == 1:
            output[finding] = next(iter(unique))
    return output


def load_rows(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def infer_cell(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "native" in parts:
        index = parts.index("native")
        if len(parts) > index + 2:
            return parts[index + 1], parts[index + 2]
    return "unknown", path.parent.name


def bootstrap_delta(records: list[dict], draws: int, seed: int) -> list[float] | None:
    if not records:
        return None
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [float(record["swapped_correct"] - record["native_correct"]) for record in records]
    )
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sampled.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(path: Path, draws: int, seed: int) -> dict:
    model, dataset = infer_cell(path)
    records = []
    examples_swap_helps = []
    examples_swap_harms = []
    rows = 0
    generated_lateralized = 0
    reference_lateralized = 0
    for row in load_rows(path):
        rows += 1
        predicted = parse_lateralized_positive_findings(row.get("text", ""))
        reference = parse_lateralized_positive_findings(row.get("gt_ans", ""))
        generated_lateralized += len(predicted)
        reference_lateralized += len(reference)
        for finding in sorted(set(predicted) & set(reference)):
            native_correct = predicted[finding] == reference[finding]
            swapped_correct = predicted[finding] != reference[finding]
            record = {
                "question_id": str(row.get("question_id", "")),
                "finding": finding,
                "predicted_side": predicted[finding],
                "reference_side": reference[finding],
                "native_correct": int(native_correct),
                "swapped_correct": int(swapped_correct),
            }
            records.append(record)
            example = {
                **record,
                "text": str(row.get("text", ""))[:500],
                "gt_ans": str(row.get("gt_ans", ""))[:500],
            }
            if swapped_correct and len(examples_swap_helps) < 5:
                examples_swap_helps.append(example)
            if native_correct and len(examples_swap_harms) < 5:
                examples_swap_harms.append(example)

    n = len(records)
    native = sum(record["native_correct"] for record in records)
    swapped = sum(record["swapped_correct"] for record in records)
    by_finding = {}
    for finding in sorted({record["finding"] for record in records}):
        subset = [record for record in records if record["finding"] == finding]
        by_finding[finding] = {
            "n": len(subset),
            "native_accuracy": float(np.mean([x["native_correct"] for x in subset])),
            "swapped_accuracy": float(np.mean([x["swapped_correct"] for x in subset])),
        }
    return {
        "path": str(path),
        "model": model,
        "dataset": dataset,
        "rows": rows,
        "generated_lateralized_positive_claims": generated_lateralized,
        "reference_lateralized_positive_claims": reference_lateralized,
        "matched_unambiguous_claims": n,
        "native_accuracy": native / n if n else None,
        "swapped_accuracy": swapped / n if n else None,
        "swapped_minus_native": (swapped - native) / n if n else None,
        "bootstrap_ci95": bootstrap_delta(records, draws, seed),
        "by_finding": by_finding,
        "examples_swap_helps": examples_swap_helps,
        "examples_swap_harms": examples_swap_harms,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    cells = [
        evaluate(path, args.bootstrap_draws, args.seed + index)
        for index, path in enumerate(args.inputs)
    ]
    pooled_records = [record for cell in cells for record in cell["records"]]
    n = len(pooled_records)
    native = sum(record["native_correct"] for record in pooled_records)
    swapped = sum(record["swapped_correct"] for record in pooled_records)
    for cell in cells:
        cell.pop("records")
    payload = {
        "version": VERSION,
        "warning": (
            "Lexical matching against reference reports is a benchmark proxy, not "
            "clinical truth. Only unambiguous positive finding-side matches are scored."
        ),
        "cells": cells,
        "pooled": {
            "matched_unambiguous_claims": n,
            "native_accuracy": native / n if n else None,
            "swapped_accuracy": swapped / n if n else None,
            "swapped_minus_native": (swapped - native) / n if n else None,
            "bootstrap_ci95": bootstrap_delta(
                pooled_records, args.bootstrap_draws, args.seed + 1000
            ),
        },
        "decision_rule": (
            "A cache-only natural-output GO requires at least two medical VLMs with "
            ">=30 matched claims each, swapped-minus-native >=0.20, and bootstrap "
            "CI lower bound >0. It does not replace the certified screen-frame canary."
        ),
    }
    qualifying = [
        cell
        for cell in cells
        if cell["matched_unambiguous_claims"] >= 30
        and cell["swapped_minus_native"] is not None
        and cell["swapped_minus_native"] >= 0.20
        and cell["bootstrap_ci95"] is not None
        and cell["bootstrap_ci95"][0] > 0
    ]
    payload["decision"] = (
        "CACHE_NATURAL_GO" if len({cell["model"] for cell in qualifying}) >= 2 else "NO_GO"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
