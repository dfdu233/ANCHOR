#!/usr/bin/env python3
"""Outcome-blind CPU admission audit for Shared-Scope Evidence Pooling.

This module deliberately inspects only source/native text geometry.  It never
loads images, clinical labels, model scores, or shared evaluation code.  Regex
matches are parser candidates, not human scope or clinical truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


PROTOCOL_VERSION = "ssep-scope-admission-v1"

# Frozen before the census.  Reference reports do not count as a model.
ADMISSION_RULES = {
    "minimum_total_natural_scope_cases": 100,
    "minimum_models": 2,
    "minimum_cases_per_model": 30,
    "minimum_findings": 3,
    "minimum_cases_per_finding": 20,
    "minimum_parser_human_agreement": 0.90,
    "human_review_required": True,
    "formal_model_sources": ["hulu_mimic_report", "llava_mimic_report"],
    "formal_task": "mimic_single_image_report_generation",
    "minimal_pair_operator": "negated_coordination",
    "minimal_pair_sibling_count": 2,
    "minimal_pair_invariants": [
        "same_ordered_atomic_claims",
        "same_claim_count",
        "same_whitespace_word_count",
        "same_polarity",
        "human_naturalness_and_equivalence_required",
    ],
}


SOURCES = (
    {
        "source_id": "hulu_mimic_report",
        "source_kind": "model",
        "model_id": "hulu",
        "task": "mimic_single_image_report_generation",
        "path": "corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/predictions.jsonl",
        "text_field": "greedy.text",
    },
    {
        "source_id": "llava_mimic_report",
        "source_kind": "model",
        "model_id": "llava_med",
        "task": "mimic_single_image_report_generation",
        "path": "corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/predictions.jsonl",
        "text_field": "greedy.text",
    },
    {
        "source_id": "mimic_reference_report",
        "source_kind": "reference",
        "model_id": None,
        "task": "mimic_study_report_reference",
        "path": "corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/predictions.jsonl",
        "text_field": "answer",
    },
)


# Conservative chest-radiograph ontology.  Long forms precede short aliases.
FINDING_PATTERNS = {
    "pleural_effusion": (r"pleural effusions?", r"effusions?"),
    "pneumothorax": (r"pneumothoraces", r"pneumothorax"),
    "consolidation": (r"focal consolidations?", r"airspace consolidations?", r"consolidations?"),
    "pulmonary_edema": (r"pulmonary edema", r"interstitial edema", r"edema"),
    "atelectasis": (r"subsegmental atelectasis", r"bibasilar atelectasis", r"atelectasis"),
    "cardiomegaly": (r"cardiomegaly", r"enlarg(?:ed|ement of the) (?:cardiac silhouette|heart)"),
    "lung_opacity": (r"airspace opacit(?:y|ies)", r"pulmonary opacit(?:y|ies)", r"opacit(?:y|ies)"),
    "nodule_mass": (r"pulmonary nodules?", r"lung nodules?", r"mass(?:es)?"),
    "pneumonia": (r"pneumonia", r"infection"),
    "fracture": (r"acute osseous abnormalities", r"osseous abnormalities", r"fractures?"),
    "fibrosis": (r"pulmonary fibrosis", r"fibrotic changes?", r"fibrosis"),
    "emphysema": (r"emphysema",),
    "vascular_congestion": (r"pulmonary vascular congestion", r"vascular congestion"),
    "mediastinal_widening": (r"mediastinal widening", r"widened mediastinum"),
    "focal_airspace_disease": (r"focal airspace disease", r"focal airspace process"),
    "acute_cardiopulmonary_process": (r"acute cardiopulmonary (?:process|abnormality)",),
}

_FINDING_REGEX = {
    name: re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)
    for name, patterns in FINDING_PATTERNS.items()
}
_COORD = re.compile(r"\b(?:and|or|nor)\b|,", re.IGNORECASE)
_NEGATOR = re.compile(r"\b(?:no|without|neither)\b", re.IGNORECASE)
_HEDGE = re.compile(r"\b(?:possible|possibly|may|might|could|cannot exclude|can(?:not|'t) rule out)\b", re.IGNORECASE)
_CONTRAST = re.compile(r"\b(?:but|whereas|although|however)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Mention:
    finding: str
    surface: str
    start: int
    end: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dotted_get(row: dict, dotted: str):
    value = row
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_no"] = line_no
                yield row


def sentence_spans(text: str) -> Iterable[tuple[int, int, str]]:
    # Semicolon is kept inside a sentence because radiology coordination often
    # crosses it.  Newlines and terminal punctuation end a candidate span.
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]|$)", text):
        sentence = match.group(0).strip()
        if sentence:
            yield match.start(), match.end(), sentence


def finding_mentions(sentence: str) -> list[Mention]:
    mentions: list[Mention] = []
    occupied: list[tuple[int, int]] = []
    for finding, regex in _FINDING_REGEX.items():
        for match in regex.finditer(sentence):
            if any(not (match.end() <= a or match.start() >= b) for a, b in occupied):
                continue
            mentions.append(Mention(finding, match.group(0), match.start(), match.end()))
            occupied.append((match.start(), match.end()))
    mentions.sort(key=lambda m: (m.start, -(m.end - m.start), m.finding))
    # A sentence repeating the same atom is not a multi-sibling scope.
    deduped: list[Mention] = []
    seen: set[str] = set()
    for mention in mentions:
        if mention.finding not in seen:
            deduped.append(mention)
            seen.add(mention.finding)
    return deduped


def classify_operator(sentence: str, mentions: list[Mention]) -> str | None:
    if len(mentions) < 2:
        return None
    first, last = mentions[0], mentions[-1]
    between = sentence[first.end:last.start]
    contrast_between = _CONTRAST.search(between)
    if not _COORD.search(between) and not contrast_between:
        return None
    prefix = sentence[: first.start]
    full_scope = sentence[: last.end]
    contrast = _CONTRAST.search(full_scope)
    negators = list(_NEGATOR.finditer(full_scope))
    hedge = _HEDGE.search(prefix)
    if contrast and negators:
        return "contrastive_scope"
    if negators and negators[0].start() < first.start:
        # If a second explicit negator occurs before the last atom this is
        # already distributive realization, not shared scope.
        if len([m for m in negators if m.start() < last.start]) == 1:
            return "negated_coordination"
    if hedge and re.search(r"\bor\b", between, re.IGNORECASE):
        return "hedged_alternative"
    return None


def whitespace_words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


def build_minimal_pair(mentions: list[Mention], operator: str) -> dict | None:
    if operator != "negated_coordination" or len(mentions) != 2:
        return None
    a, b = mentions
    shared = f"No {a.surface} or {b.surface}."
    distributive = f"No {a.surface}; no {b.surface}."
    shared_count = len(whitespace_words(shared))
    distributive_count = len(whitespace_words(distributive))
    return {
        "shared": shared,
        "distributive": distributive,
        "ordered_claims": [a.finding, b.finding],
        "claim_count_equal": True,
        "ordered_claims_equal": True,
        "polarity_equal_by_de_morgan": True,
        "whitespace_word_count_shared": shared_count,
        "whitespace_word_count_distributive": distributive_count,
        "whitespace_word_count_equal": shared_count == distributive_count,
        "parser_constructible": shared_count == distributive_count,
        "human_naturalness": None,
        "human_semantic_equivalence": None,
    }


def extract_candidates(text: str) -> list[dict]:
    candidates: list[dict] = []
    for start, end, sentence in sentence_spans(text):
        mentions = finding_mentions(sentence)
        operator = classify_operator(sentence, mentions)
        if operator is None:
            continue
        minimal_pair = build_minimal_pair(mentions, operator)
        candidates.append(
            {
                "char_start": start,
                "char_end": end,
                "sentence": sentence,
                "operator": operator,
                "siblings": [m.__dict__ for m in mentions],
                "minimal_pair_candidate": minimal_pair,
                "parser_status": "candidate_only_not_human_truth",
            }
        )
    return candidates


def source_rows(repo: Path, source: dict) -> tuple[list[dict], dict]:
    path = repo / source["path"]
    results: list[dict] = []
    text_hashes: set[str] = set()
    n_rows = 0
    n_nonempty = 0
    n_sentences = 0
    for row in iter_jsonl(path):
        n_rows += 1
        text = dotted_get(row, source["text_field"])
        if not isinstance(text, str) or not text.strip():
            continue
        n_nonempty += 1
        n_sentences += sum(1 for _ in sentence_spans(text))
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        text_hashes.add(text_hash)
        for local_index, candidate in enumerate(extract_candidates(text)):
            candidate_id = hashlib.sha256(
                f'{source["source_id"]}\0{row["_line_no"]}\0{local_index}\0{candidate["sentence"]}'.encode("utf-8")
            ).hexdigest()[:24]
            results.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "candidate_id": candidate_id,
                    "source_id": source["source_id"],
                    "source_kind": source["source_kind"],
                    "model_id": source["model_id"],
                    "task": source["task"],
                    "source_path": source["path"],
                    "source_line": row["_line_no"],
                    "item_id": row.get("qid") or row.get("question_id") or row.get("id") or str(row["_line_no"]),
                    "text_sha256": text_hash,
                    **candidate,
                }
            )
    stats = {
        "source_id": source["source_id"],
        "source_kind": source["source_kind"],
        "model_id": source["model_id"],
        "task": source["task"],
        "path": source["path"],
        "path_sha256": sha256_file(path),
        "rows": n_rows,
        "nonempty_rows": n_nonempty,
        "sentences": n_sentences,
        "unique_texts": len(text_hashes),
        "parser_candidate_cases": len(results),
    }
    return results, stats


def summarize(candidates: list[dict], source_stats: list[dict]) -> dict:
    by_source = Counter(row["source_id"] for row in candidates)
    by_operator = Counter(row["operator"] for row in candidates)
    sibling_counts = Counter(len(row["siblings"]) for row in candidates)
    finding_case_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for row in candidates:
        findings = [s["finding"] for s in row["siblings"]]
        finding_case_counts.update(set(findings))
        pair_counts[" | ".join(findings)] += 1

    formal = [
        row
        for row in candidates
        if row["source_kind"] == "model"
        and row["source_id"] in ADMISSION_RULES["formal_model_sources"]
        and row["task"] == ADMISSION_RULES["formal_task"]
    ]
    formal_by_model = Counter(row["model_id"] for row in formal)
    formal_findings: Counter[str] = Counter()
    for row in formal:
        formal_findings.update({s["finding"] for s in row["siblings"]})
    qualifying_models = {
        model: count
        for model, count in formal_by_model.items()
        if count >= ADMISSION_RULES["minimum_cases_per_model"]
    }
    qualifying_findings = {
        finding: count
        for finding, count in formal_findings.items()
        if count >= ADMISSION_RULES["minimum_cases_per_finding"]
    }
    constructible = [
        row for row in formal
        if row["minimal_pair_candidate"]
        and row["minimal_pair_candidate"]["parser_constructible"]
    ]
    constructible_templates = {
        (
            row["minimal_pair_candidate"]["shared"].casefold(),
            row["minimal_pair_candidate"]["distributive"].casefold(),
        )
        for row in constructible
    }

    per_source_candidate_stats = {}
    for source_id in sorted({row["source_id"] for row in candidates}):
        rows = [row for row in candidates if row["source_id"] == source_id]
        source_ops = Counter(row["operator"] for row in rows)
        source_findings: Counter[str] = Counter()
        source_pairs: Counter[str] = Counter()
        sentence_counts = Counter(row["sentence"].casefold().strip() for row in rows)
        for row in rows:
            findings = [s["finding"] for s in row["siblings"]]
            source_findings.update(set(findings))
            source_pairs[" | ".join(findings)] += 1
        per_source_candidate_stats[source_id] = {
            "cases": len(rows),
            "unique_candidate_sentences": len(sentence_counts),
            "largest_exact_sentence_cluster": max(sentence_counts.values(), default=0),
            "largest_exact_sentence_cluster_fraction": (
                max(sentence_counts.values(), default=0) / len(rows) if rows else 0.0
            ),
            "by_operator": dict(sorted(source_ops.items())),
            "by_finding": dict(sorted(source_findings.items(), key=lambda x: (-x[1], x[0]))),
            "top_ordered_sibling_sets": dict(source_pairs.most_common(15)),
            "top_exact_candidate_sentences": dict(sentence_counts.most_common(10)),
        }

    mechanical_gates = {
        "total_cases": len(formal) >= ADMISSION_RULES["minimum_total_natural_scope_cases"],
        "two_models_with_minimum_cases": len(qualifying_models) >= ADMISSION_RULES["minimum_models"],
        "three_findings_with_minimum_cases": len(qualifying_findings) >= ADMISSION_RULES["minimum_findings"],
        "minimal_pair_candidates_nonempty": bool(constructible),
    }
    # Human agreement/naturalness is deliberately not inferred by code.
    final_go = all(mechanical_gates.values()) and not ADMISSION_RULES["human_review_required"]
    fatal_reasons = [name for name, passed in mechanical_gates.items() if not passed]
    if ADMISSION_RULES["human_review_required"]:
        fatal_reasons.append("independent_human_scope_naturalness_equivalence_review_absent")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "outcome_blind": True,
        "gpu_used": False,
        "clinical_labels_opened": False,
        "model_scores_opened": False,
        "parser_candidates_are_truth": False,
        "admission_rules": ADMISSION_RULES,
        "source_stats": source_stats,
        "source_candidate_stats": per_source_candidate_stats,
        "all_candidate_counts": {
            "total": len(candidates),
            "by_source": dict(sorted(by_source.items())),
            "by_operator": dict(sorted(by_operator.items())),
            "by_sibling_count": {str(k): v for k, v in sorted(sibling_counts.items())},
            "by_finding": dict(sorted(finding_case_counts.items(), key=lambda x: (-x[1], x[0]))),
            "top_ordered_sibling_sets": dict(pair_counts.most_common(30)),
        },
        "formal_model_census": {
            "cases": len(formal),
            "by_model": dict(sorted(formal_by_model.items())),
            "qualifying_models": qualifying_models,
            "by_finding": dict(sorted(formal_findings.items(), key=lambda x: (-x[1], x[0]))),
            "qualifying_findings": qualifying_findings,
            "parser_constructible_minimal_pairs": len(constructible),
            "unique_parser_constructible_pair_templates": len(constructible_templates),
        },
        "mechanical_gates": mechanical_gates,
        "human_gate": "not_run_blank_template_only",
        "decision": "GO" if final_go else "NO_GO",
        "fatal_reasons": fatal_reasons,
        "scope_claim_authorized": False,
        "minimal_pair_model_run_authorized": False,
    }


def write_review_template(path: Path, candidates: list[dict]) -> None:
    fields = [
        "candidate_id", "source_id", "model_id", "sentence", "operator",
        "siblings_json", "parser_scope_correct", "natural_radiology_language",
        "minimal_pair_semantically_equivalent", "reviewer_id", "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in candidates:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "source_id": row["source_id"],
                    "model_id": row["model_id"] or "",
                    "sentence": row["sentence"],
                    "operator": row["operator"],
                    "siblings_json": json.dumps(row["siblings"], ensure_ascii=False, separators=(",", ":")),
                    "parser_scope_correct": "",
                    "natural_radiology_language": "",
                    "minimal_pair_semantically_equivalent": "",
                    "reviewer_id": "",
                    "rationale": "",
                }
            )


def run(repo: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_candidates: list[dict] = []
    all_stats: list[dict] = []
    for source in SOURCES:
        candidates, stats = source_rows(repo, source)
        all_candidates.extend(candidates)
        all_stats.append(stats)
    all_candidates.sort(key=lambda row: (row["source_id"], row["source_line"], row["char_start"]))
    summary = summarize(all_candidates, all_stats)
    candidates_path = out_dir / "scope_candidate_spans.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in all_candidates:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_review_template(out_dir / "scope_human_review_template.csv", all_candidates)
    script_path = Path(__file__).resolve()
    provenance_core = {
        "dataset": "local MIMIC report-generation native/reference text only",
        "models": ["hulu", "llava_med"],
        "method": "outcome-blind SSEP scope admission census",
        "seed": None,
        "command": (
            "PYTHONPATH=. python -m anchor.corrected_sgta.ssep_scope_admission "
            "--out-dir corrected_runs/ssep_scope_gate_v1"
        ),
        "script_path": str(script_path.relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "candidate_spans_sha256": sha256_file(candidates_path),
        "review_template_sha256": sha256_file(out_dir / "scope_human_review_template.csv"),
    }
    provenance_core["fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                **provenance_core,
                "admission_rules": ADMISSION_RULES,
                "source_hashes": {row["source_id"]: row["path_sha256"] for row in all_stats},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary["provenance"] = provenance_core
    (out_dir / "scope_admission.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.repo.resolve(), args.out_dir.resolve())
    print(json.dumps({"decision": summary["decision"], "fatal_reasons": summary["fatal_reasons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
