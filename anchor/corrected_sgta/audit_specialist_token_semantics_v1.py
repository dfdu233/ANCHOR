#!/usr/bin/env python3
"""Audit whether radiology disease words identify claim polarity.

CCD-style expert guidance attaches a disease score to lexical tokens.  This
CPU-only audit asks a prerequisite question: when a disease word occurs in a
report, does that word identify a positive clinical assertion?  It deliberately
uses a frozen, transparent surface rule rather than assigning clinical truth.
The output is a mechanism screen, not a hallucination metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

PROTOCOL = "specialist-token-semantics-v1"

ALIASES = {
    "atelectasis": ("atelectasis", "atelectatic"),
    "consolidation": ("consolidation",),
    "infiltration": ("infiltrate", "infiltration"),
    "pneumothorax": ("pneumothorax",),
    "edema": ("edema", "oedema"),
    "emphysema": ("emphysema",),
    "fibrosis": ("fibrosis", "fibrotic"),
    "effusion": ("pleural effusion", "effusion"),
    "pneumonia": ("pneumonia",),
    "pleural_thickening": ("pleural thickening",),
    "cardiomegaly": ("cardiomegaly", "enlarged cardiac silhouette"),
    "nodule": ("nodule", "nodular opacity"),
    "mass": ("mass",),
    "fracture": ("fracture", "fractures"),
    "lung_opacity": ("lung opacity", "pulmonary opacity", "airspace opacity"),
    "enlarged_cardiomediastinum": (
        "enlarged cardiomediastinal silhouette",
        "cardiomediastinal enlargement",
    ),
}

NEGATION = re.compile(
    r"\b(no|not|without|absent|absence of|negative for|free of|neither|nor|resolved|clear of)\b",
    re.IGNORECASE,
)
UNCERTAINTY = re.compile(
    r"\b(may|might|could|possible|possibly|probable|cannot exclude|can not exclude|"
    r"questionable|suspicious for|suggesting)\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polarity(text: str, start: int) -> str:
    context = text[max(0, start - 90) : start]
    context = re.split(r"[.;\n]", context)[-1]
    if UNCERTAINTY.search(context):
        return "uncertain"
    if NEGATION.search(context):
        return "negative"
    return "positive"


def audit(texts: Iterable[str]) -> dict:
    by_finding: dict[str, Counter] = defaultdict(Counter)
    examples = []
    report_count = 0
    for raw in texts:
        report_count += 1
        text = str(raw).lower()
        for finding, aliases in ALIASES.items():
            occupied: list[tuple[int, int]] = []
            for alias in sorted(aliases, key=len, reverse=True):
                for match in re.finditer(r"\b" + re.escape(alias) + r"\b", text):
                    if any(not (match.end() <= left or match.start() >= right) for left, right in occupied):
                        continue
                    occupied.append(match.span())
                    state = polarity(text, match.start())
                    by_finding[finding][state] += 1
                    if len(examples) < 24 and state != "positive":
                        examples.append(
                            {
                                "finding": finding,
                                "surface_polarity": state,
                                "context": text[max(0, match.start() - 55) : match.end() + 55],
                            }
                        )
    totals = Counter()
    for counts in by_finding.values():
        totals.update(counts)
    mentions = sum(totals.values())
    return {
        "reports": report_count,
        "mentions": mentions,
        "totals": dict(totals),
        "fractions": {
            key: (totals[key] / mentions if mentions else None)
            for key in ("positive", "negative", "uncertain")
        },
        "by_finding": {key: dict(value) for key, value in sorted(by_finding.items())},
        "examples": examples,
    }


def load_manifest(path: Path) -> list[str]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list: {path}")
    return [str(row["answer"]) for row in rows]


def load_answers(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = ("answer", "prediction", "text", "output")
    output = []
    for row in rows:
        value = next((row.get(key) for key in keys if row.get(key) is not None), None)
        if value is None:
            raise KeyError(f"No generated-text field in {path}: {sorted(row)}")
        output.append(str(value))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--answers", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analyses = {}
    inputs = []
    for path in args.manifest:
        analyses[str(path)] = audit(load_manifest(path))
        inputs.append({"kind": "reference_manifest", "path": str(path), "sha256": sha256(path)})
    for path in args.answers:
        analyses[str(path)] = audit(load_answers(path))
        inputs.append({"kind": "generated_answers", "path": str(path), "sha256": sha256(path)})

    result = {
        "protocol": PROTOCOL,
        "status": "complete_surface_mechanism_audit",
        "question": "Does a radiology disease word determine the polarity of the clinical claim containing it?",
        "decision_rule": (
            "If at least 25% of disease-word mentions are negative or uncertain, a polarity-blind "
            "token bias is not a semantically identified clinical-evidence interface."
        ),
        "scope": (
            "Negation/uncertainty is assigned by a frozen lexical window. This audit establishes "
            "word-to-claim ambiguity only; it does not establish clinical truth or mitigation efficacy."
        ),
        "inputs": inputs,
        "analyses": analyses,
    }
    nonpositive = [
        value["fractions"]["negative"] + value["fractions"]["uncertain"]
        for value in analyses.values()
        if value["mentions"]
    ]
    result["decision"] = "TOKEN_INTERFACE_NOT_IDENTIFIED" if nonpositive and max(nonpositive) >= 0.25 else "NO_SIGNAL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
