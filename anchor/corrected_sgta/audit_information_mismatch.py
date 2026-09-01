#!/usr/bin/env python3
"""Audit a high-precision lower bound on report/input information mismatch.

This is deliberately not a clinical claim labeler.  It detects explicit textual
evidence that a target sentence relies on information beyond a single current
image.  The output keeps logically unavailable content (Tier A) separate from
knowledge/action content and weaker contextual cues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Corpus:
    name: str
    path: Path


DEFAULT_CORPORA = (
    Corpus("rule_mimic", REPO_ROOT / "data/rule/test/mimic_test.jsonl"),
    Corpus("rule_iuxray", REPO_ROOT / "data/rule/test/iuxray_test.jsonl"),
    Corpus(
        "chexpert_subset_report",
        REPO_ROOT / "data/chexpert_subset_report/processed-v1/rule_report.jsonl",
    ),
)


# Tier A requires information that is absent when the input is exactly one
# current radiograph.  Patterns are intentionally explicit and conservative.
TIER_A_PATTERNS: dict[str, tuple[str, ...]] = {
    "prior_image": (
        r"\bcompar(?:ed|ison)\s+(?:with|to)\s+(?:the\s+)?(?:(?:prior|previous|earlier)(?:\s+(?:exam(?:ination)?|study|radiograph|film|image))?|most\s+recent(?:\s+(?:exam(?:ination)?|study|radiograph|film|image))?|exam(?:ination)?|study|radiograph|film)\b",
        r"\b(?:since|from)\s+(?:the\s+)?(?:prior|previous|earlier)\s+(?:exam(?:ination)?|study|radiograph|film|image)\b",
        r"\binterval\s+(?:change|increase|decrease|improvement|worsening|progression|resolution)\b",
        r"\b(?:unchanged|stable)\s+(?:from|since|compared)\b",
        r"\b(?:new|resolved|improved|worsened)\s+since\b",
        r"\bagain\s+(?:seen|noted|demonstrated|identified)\b",
    ),
    "clinical_history": (
        r"\b(?:the\s+)?patient(?:'s|s')?\s+(?:known\s+)?history\s+of\b",
        r"\bclinical\s+history\s+(?:of|is|includes?)\b",
        r"\bgiven\s+(?:the\s+)?history\s+of\b",
        r"\bknown\s+history\s+of\b",
    ),
    "other_test": (
        r"\b(?:as\s+)?(?:seen|shown|demonstrated|identified|noted)\s+on\s+(?:the\s+)?(?:prior\s+)?(?:ct|mri?|ultrasound|sonogram|pet|pathology)\b",
        r"\b(?:ct|mri?|ultrasound|sonogram|pet|pathology)\s+(?:showed|shows|demonstrated|revealed|confirmed)\b",
        r"\b(?:laboratory|lab|pathology)\s+(?:results?|findings?)\s+(?:show|shows|showed|demonstrate|demonstrates|demonstrated|reveal|reveals|revealed|confirm|confirms|confirmed)\b",
    ),
}


# These sentences may be clinically useful, but they are not direct visual
# observations.  They are reported separately and never counted as Tier A.
NON_VISUAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "management_or_followup": (
        r"\b(?:recommend|recommended|recommendation|follow[- ]?up|correlate clinically|clinical correlation)\b",
        r"\b(?:should|could)\s+(?:be\s+)?(?:considered|obtained|performed)\b",
    ),
    "etiologic_or_diagnostic_inference": (
        r"\b(?:consistent with|compatible with|suggestive of|likely represents?|may represent|due to|secondary to)\b",
    ),
}


# Weaker cues are useful for sensitivity analysis but are excluded from the
# headline lower bound because they can sometimes describe current appearance.
TIER_B_PATTERNS: dict[str, tuple[str, ...]] = {
    "implicit_temporal": (
        r"\b(?:unchanged|stable|persistent|chronic|longstanding)\b",
        r"\b(?:improved|worsened|resolved|newly developed)\b",
    ),
    "procedure_or_status_history": (
        r"\b(?:status post|postoperative|postsurgical|postprocedural)\b",
    ),
}


MULTIVIEW_PATTERNS = (
    r"\b(?:pa|frontal)\s+and\s+lateral\s+(?:chest\s+)?(?:radiographs?|views?|images?)\b",
    r"\btwo\s+views?\s+of\s+the\s+chest\b",
)


def compile_patterns(groups: dict[str, tuple[str, ...]]) -> dict[str, tuple[re.Pattern[str], ...]]:
    return {
        label: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        for label, patterns in groups.items()
    }


TIER_A = compile_patterns(TIER_A_PATTERNS)
TIER_B = compile_patterns(TIER_B_PATTERNS)
NON_VISUAL = compile_patterns(NON_VISUAL_PATTERNS)
MULTIVIEW = tuple(re.compile(pattern, re.IGNORECASE) for pattern in MULTIVIEW_PATTERNS)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    for key in ("data", "records", "samples"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"unsupported corpus shape: {path}")


def report_value(row: dict[str, Any]) -> str:
    for key in ("report", "answer", "gt_ans", "reference"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def image_values(row: dict[str, Any]) -> tuple[str, ...]:
    value = row.get("images", row.get("image", row.get("img_name", row.get("image_path", ""))))
    if isinstance(value, list):
        return tuple(str(item) for item in value if item)
    return (str(value),) if value else ()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(report: str) -> list[str]:
    # Headings are left attached to the following sentence; this audit only
    # needs conservative lexical spans, not syntactic sentence boundaries.
    chunks = re.split(r"(?<=[.!?])\s+|[\r\n]+", normalize_space(report))
    return [chunk.strip(" .") for chunk in chunks if chunk.strip(" .")]


def labels_for(sentence: str, groups: dict[str, tuple[re.Pattern[str], ...]]) -> list[str]:
    return [label for label, patterns in groups.items() if any(p.search(sentence) for p in patterns)]


def stable_id(report: str, images: tuple[str, ...]) -> str:
    payload = normalize_space(report).lower() + "\n" + "\n".join(images)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def deduplicate(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    by_instance: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    unique_reports: set[str] = set()
    raw = 0
    for row in rows:
        raw += 1
        report = report_value(row)
        if not report:
            continue
        images = image_values(row)
        report_norm = normalize_space(report).lower()
        by_instance.setdefault((report_norm, images), {"report": report, "images": images})
        unique_reports.add(report_norm)
    return list(by_instance.values()), raw, len(unique_reports)


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def audit_corpus(corpus: Corpus, max_examples: int) -> dict[str, Any]:
    rows = load_rows(corpus.path)
    instances, raw_rows, unique_report_count = deduplicate(rows)
    sentence_count = 0
    tier_a_sentence_count = 0
    tier_b_sentence_count = 0
    non_visual_sentence_count = 0
    tier_a_reports = 0
    tier_b_reports = 0
    non_visual_reports = 0
    multiview_single_input_reports = 0
    label_counts: dict[str, Counter[str]] = {
        "tier_a": Counter(),
        "tier_b": Counter(),
        "non_visual": Counter(),
    }
    examples: dict[str, dict[str, list[dict[str, str]]]] = {
        level: defaultdict(list) for level in label_counts
    }
    multiview_examples: list[dict[str, str]] = []

    for instance in instances:
        report = instance["report"]
        images = instance["images"]
        item_id = stable_id(report, images)
        sentences = split_sentences(report)
        sentence_count += len(sentences)
        report_levels = {"tier_a": False, "tier_b": False, "non_visual": False}
        for sentence in sentences:
            matches = {
                "tier_a": labels_for(sentence, TIER_A),
                "tier_b": labels_for(sentence, TIER_B),
                "non_visual": labels_for(sentence, NON_VISUAL),
            }
            for level, labels in matches.items():
                if labels:
                    report_levels[level] = True
                    label_counts[level].update(labels)
                    for label in labels:
                        if len(examples[level][label]) < max_examples:
                            examples[level][label].append({"id": item_id, "sentence": sentence})
            tier_a_sentence_count += bool(matches["tier_a"])
            tier_b_sentence_count += bool(matches["tier_b"])
            non_visual_sentence_count += bool(matches["non_visual"])
        tier_a_reports += report_levels["tier_a"]
        tier_b_reports += report_levels["tier_b"]
        non_visual_reports += report_levels["non_visual"]

        if len(images) == 1 and any(pattern.search(report) for pattern in MULTIVIEW):
            multiview_single_input_reports += 1
            if len(multiview_examples) < max_examples:
                multiview_examples.append({"id": item_id, "sentence": normalize_space(report)[:500]})

    n = len(instances)
    return {
        "corpus": corpus.name,
        "path": str(corpus.path),
        "raw_rows": raw_rows,
        "unique_report_image_instances": n,
        "unique_report_texts": unique_report_count,
        "sentences": sentence_count,
        "tier_a_unavailable_lower_bound": {
            "reports": tier_a_reports,
            "report_rate": rate(tier_a_reports, n),
            "sentences": tier_a_sentence_count,
            "sentence_rate": rate(tier_a_sentence_count, sentence_count),
            "label_occurrences": dict(label_counts["tier_a"]),
            "examples": dict(examples["tier_a"]),
        },
        "tier_b_contextual_sensitivity_only": {
            "reports": tier_b_reports,
            "report_rate": rate(tier_b_reports, n),
            "sentences": tier_b_sentence_count,
            "sentence_rate": rate(tier_b_sentence_count, sentence_count),
            "label_occurrences": dict(label_counts["tier_b"]),
            "examples": dict(examples["tier_b"]),
        },
        "non_visual_but_not_unavailable": {
            "reports": non_visual_reports,
            "report_rate": rate(non_visual_reports, n),
            "sentences": non_visual_sentence_count,
            "sentence_rate": rate(non_visual_sentence_count, sentence_count),
            "label_occurrences": dict(label_counts["non_visual"]),
            "examples": dict(examples["non_visual"]),
        },
        "single_input_multiview_target_candidate": {
            "reports": multiview_single_input_reports,
            "report_rate": rate(multiview_single_input_reports, n),
            "examples": multiview_examples,
            "warning": "Candidate mismatch only; one file can be a composite image or metadata may be incomplete.",
        },
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    instances = sum(row["unique_report_image_instances"] for row in results)
    sentences = sum(row["sentences"] for row in results)
    tier_a_reports = sum(row["tier_a_unavailable_lower_bound"]["reports"] for row in results)
    tier_a_sentences = sum(row["tier_a_unavailable_lower_bound"]["sentences"] for row in results)
    return {
        "unique_report_image_instances": instances,
        "sentences": sentences,
        "tier_a_reports": tier_a_reports,
        "tier_a_report_rate": rate(tier_a_reports, instances),
        "tier_a_sentences": tier_a_sentences,
        "tier_a_sentence_rate": rate(tier_a_sentences, sentences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "corrected_runs/information_mismatch/audit_v1.json",
    )
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--corpus", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    corpora = list(DEFAULT_CORPORA)
    if args.corpus:
        corpora = []
        for value in args.corpus:
            name, separator, raw_path = value.partition("=")
            if not separator:
                raise ValueError("--corpus must be NAME=PATH")
            corpora.append(Corpus(name, Path(raw_path)))
    missing = [str(corpus.path) for corpus in corpora if not corpus.path.exists()]
    if missing:
        raise FileNotFoundError(f"missing corpora: {missing}")

    results = [audit_corpus(corpus, args.max_examples) for corpus in corpora]
    payload = {
        "definition": {
            "input_assumption": "one current radiograph per report-image instance",
            "tier_a": "explicit target content logically requiring an unavailable prior, history, or other test",
            "tier_b": "context-sensitive wording excluded from the lower bound",
            "non_visual": "knowledge/action content; not itself proof of unavailable information",
            "unit_warning": "regex spans are auditable textual indicators, not clinical ground-truth claims",
        },
        "aggregate": aggregate(results),
        "corpora": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "aggregate": payload["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
