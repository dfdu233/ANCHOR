#!/usr/bin/env python3
"""Strict paired evaluation for resumable RULE DG-adapter JSONL outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from corrected_sgta.evaluate_medheval_answers import (
    parse_answer,
    rule_pope_prediction,
)

EVALUATOR_VERSION = "rule-dg-adapter-paired-evaluator-v1"
SUCCESS_FIELDS = (
    "question_id",
    "image",
    "gt_answer",
    "fingerprint",
    "prompt",
    "base_text",
    "adapted_text",
)


class PairedEvaluationError(ValueError):
    """Raised when raw records cannot produce an unambiguous paired view."""


def stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def qid(row: dict[str, Any]) -> str:
    value = row.get("question_id", row.get("qid", row.get("id")))
    if value is None:
        raise PairedEvaluationError("record has no question id")
    return str(value)


def load_jsonl_with_provenance(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise PairedEvaluationError(
                f"{path}:{line_number}: expected JSON object"
            )
        row = dict(row)
        row["_source_line"] = line_number
        row["_source_line_sha256"] = sha256_bytes(line)
        rows.append(row)
    return rows, sha256_bytes(raw)


def _success_signature(row: dict[str, Any]) -> str:
    return stable_json({key: row.get(key) for key in SUCCESS_FIELDS})


def canonicalize_raw_records(
    questions: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    *,
    raw_source: str | None = None,
    raw_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one non-conflicting success per qid and retain attempt lineage."""
    question_ids = [qid(row) for row in questions]
    if len(question_ids) != len(set(question_ids)):
        raise PairedEvaluationError("duplicate qids in questions")
    expected = set(question_ids)
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(raw_records, 1):
        record_qid = qid(row)
        if record_qid not in expected:
            raise PairedEvaluationError(
                f"raw record has unexpected qid={record_qid}"
            )
        source_line = int(row.get("_source_line", index))
        status = row.get("status")
        if status not in {"ok", "error"}:
            raise PairedEvaluationError(
                f"qid={record_qid} line={source_line}: invalid status={status!r}"
            )
        attempts[record_qid].append(row)

    canonical: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    missing = []
    duplicate_identical = 0
    for question_id in question_ids:
        rows = attempts.get(question_id, [])
        successes = [row for row in rows if row["status"] == "ok"]
        signatures = {_success_signature(row) for row in successes}
        if len(signatures) > 1:
            lines = [row.get("_source_line") for row in successes]
            raise PairedEvaluationError(
                f"qid={question_id}: conflicting successful retries at lines={lines}"
            )
        if not successes:
            missing.append(question_id)
        else:
            duplicate_identical += max(0, len(successes) - 1)
            selected = dict(successes[-1])
            selected.pop("_source_line", None)
            selected.pop("_source_line_sha256", None)
            selected["_raw_provenance"] = {
                "source": raw_source,
                "raw_sha256": raw_sha256,
                "selected_line": successes[-1].get("_source_line"),
                "attempt_lines": [
                    row.get("_source_line") for row in rows
                ],
                "attempt_statuses": [row["status"] for row in rows],
                "attempt_line_sha256": [
                    row.get("_source_line_sha256") for row in rows
                ],
            }
            canonical.append(selected)
        provenance.append(
            {
                "question_id": question_id,
                "attempts": len(rows),
                "error_attempts": sum(
                    row["status"] == "error" for row in rows
                ),
                "success_attempts": len(successes),
                "lines": [row.get("_source_line") for row in rows],
                "selected_line": (
                    successes[-1].get("_source_line") if successes else None
                ),
            }
        )
    audit = {
        "raw_source": raw_source,
        "raw_sha256": raw_sha256,
        "raw_records": len(raw_records),
        "expected_questions": len(questions),
        "canonical_successes": len(canonical),
        "missing_success_qids": missing,
        "identical_duplicate_successes": duplicate_identical,
        "per_qid": provenance,
    }
    return canonical, audit


def exact_mcnemar(rescues: int, harms: int) -> dict[str, Any]:
    discordant = rescues + harms
    if discordant == 0:
        p_value = 1.0
    else:
        tail_numerator = sum(
            math.comb(discordant, index)
            for index in range(min(rescues, harms) + 1)
        )
        tail = Fraction(tail_numerator, 2**discordant)
        p_value = min(1.0, float(2 * tail))
    return {
        "base_wrong_adapted_right": rescues,
        "base_right_adapted_wrong": harms,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def patient_cluster(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("patient_id", "subject_id", "patient"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value), key
    image = str(row.get("image", ""))
    mimic = re.findall(r"(?:^|/)(p\d{5,})(?:/|$)", image, flags=re.I)
    if mimic:
        return mimic[-1].lower(), "image_mimic_patient"
    iu = re.search(r"(CXR\d+)_IM-", image, flags=re.I)
    if iu:
        return iu.group(1).upper(), "image_iuxray_patient"
    parent = Path(image).parent.as_posix()
    if parent not in {"", "."}:
        return parent, "image_parent_fallback"
    if image:
        return Path(image).stem, "image_stem_fallback"
    raise PairedEvaluationError(f"qid={qid(row)}: cannot derive patient cluster")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    pairs: list[dict[str, Any]],
    *,
    base_key: str,
    adapted_key: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise PairedEvaluationError("bootstrap replicates must be positive")
    if not pairs:
        return {
            "available": False,
            "reason": "no canonical successful pairs",
            "unit": "patient_id",
            "clusters": 0,
            "observations": 0,
            "replicates": replicates,
            "seed": seed,
            "percentile_95_ci": None,
        }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    for row in pairs:
        cluster, source = patient_cluster(row)
        groups[cluster].append(row)
        source_counts[source] += 1
    cluster_ids = sorted(groups)
    if not cluster_ids:
        raise PairedEvaluationError("cannot bootstrap empty canonical view")
    generator = random.Random(seed)
    samples = {"base": [], "adapted": [], "delta": []}
    for _ in range(replicates):
        selected = [
            generator.choice(cluster_ids) for _ in range(len(cluster_ids))
        ]
        drawn = [row for cluster in selected for row in groups[cluster]]
        base = sum(row[base_key] for row in drawn) / len(drawn)
        adapted = sum(row[adapted_key] for row in drawn) / len(drawn)
        samples["base"].append(base)
        samples["adapted"].append(adapted)
        samples["delta"].append(adapted - base)
    return {
        "unit": "patient_id",
        "clusters": len(cluster_ids),
        "observations": len(pairs),
        "cluster_id_sources": dict(sorted(source_counts.items())),
        "replicates": replicates,
        "seed": seed,
        "percentile_95_ci": {
            key: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for key, values in samples.items()
        },
    }


def _metric_summary(
    pairs: list[dict[str, Any]],
    *,
    prediction_key: str,
    correctness_key: str,
    parse_status_key: str | None,
) -> dict[str, Any]:
    n = len(pairs)
    base_correct = sum(row[f"base_{correctness_key}"] for row in pairs)
    adapted_correct = sum(
        row[f"adapted_{correctness_key}"] for row in pairs
    )
    rescues = sum(
        (not row[f"base_{correctness_key}"])
        and row[f"adapted_{correctness_key}"]
        for row in pairs
    )
    harms = sum(
        row[f"base_{correctness_key}"]
        and (not row[f"adapted_{correctness_key}"])
        for row in pairs
    )
    result = {
        "base": {
            "correct": base_correct,
            "accuracy": base_correct / n if n else 0.0,
        },
        "adapted": {
            "correct": adapted_correct,
            "accuracy": adapted_correct / n if n else 0.0,
        },
        "delta": (
            (adapted_correct - base_correct) / n if n else 0.0
        ),
        "delta_pp": (
            100.0 * (adapted_correct - base_correct) / n if n else 0.0
        ),
        "label_flips": sum(
            row[f"base_{prediction_key}"] != row[f"adapted_{prediction_key}"]
            for row in pairs
        ),
        "rescues": rescues,
        "harms": harms,
        "net_rescues": rescues - harms,
        "mcnemar": exact_mcnemar(rescues, harms),
    }
    if parse_status_key:
        for side in ("base", "adapted"):
            statuses = Counter(
                row[f"{side}_{parse_status_key}"] for row in pairs
            )
            parseable = statuses["parsed"]
            result[side].update(
                {
                    "parseable": parseable,
                    "parse_rate": parseable / n if n else 0.0,
                    "accuracy_parseable_only": (
                        result[side]["correct"] / parseable
                        if parseable
                        else 0.0
                    ),
                    "parse_status": dict(sorted(statuses.items())),
                }
            )
    return result


def evaluate_canonical_pairs(
    questions: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    question_map = {qid(row): row for row in questions}
    evaluated = []
    for row in canonical:
        question = question_map[qid(row)]
        gt_raw = question.get("answer", question.get("gt_answer"))
        if gt_raw is None:
            raise PairedEvaluationError(f"qid={qid(row)}: missing ground truth")
        if str(row.get("gt_answer")) != str(gt_raw):
            raise PairedEvaluationError(
                f"qid={qid(row)}: record/question ground truth mismatch"
            )
        if str(row.get("image")) != str(question.get("image")):
            raise PairedEvaluationError(
                f"qid={qid(row)}: record/question image mismatch"
            )
        gt = rule_pope_prediction(gt_raw)
        base_text = str(row.get("base_text", "")).strip()
        adapted_text = str(row.get("adapted_text", "")).strip()
        base_strict = parse_answer(base_text, answer_type="binary")
        adapted_strict = parse_answer(adapted_text, answer_type="binary")
        base_strict_label = (
            base_strict.labels[0] if base_strict.labels else None
        )
        adapted_strict_label = (
            adapted_strict.labels[0] if adapted_strict.labels else None
        )
        base_pope = rule_pope_prediction(base_text)
        adapted_pope = rule_pope_prediction(adapted_text)
        combined = dict(row)
        for patient_key in ("patient_id", "subject_id", "patient"):
            if patient_key not in combined and patient_key in question:
                combined[patient_key] = question[patient_key]
        combined.update(
            {
                "ground_truth_pope": gt,
                "base_pope_prediction": base_pope,
                "adapted_pope_prediction": adapted_pope,
                "base_pope_correct": base_pope == gt,
                "adapted_pope_correct": adapted_pope == gt,
                "base_strict_prediction": base_strict_label,
                "adapted_strict_prediction": adapted_strict_label,
                "base_strict_status": base_strict.status,
                "adapted_strict_status": adapted_strict.status,
                "base_strict_correct": base_strict_label == gt,
                "adapted_strict_correct": adapted_strict_label == gt,
                "text_exact_flip": base_text != adapted_text,
                "text_normalized_flip": " ".join(base_text.lower().split())
                != " ".join(adapted_text.lower().split()),
            }
        )
        evaluated.append(combined)
    pope = _metric_summary(
        evaluated,
        prediction_key="pope_prediction",
        correctness_key="pope_correct",
        parse_status_key=None,
    )
    strict = _metric_summary(
        evaluated,
        prediction_key="strict_prediction",
        correctness_key="strict_correct",
        parse_status_key="strict_status",
    )
    pope["patient_cluster_bootstrap"] = cluster_bootstrap(
        evaluated,
        base_key="base_pope_correct",
        adapted_key="adapted_pope_correct",
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    strict["patient_cluster_bootstrap"] = cluster_bootstrap(
        evaluated,
        base_key="base_strict_correct",
        adapted_key="adapted_strict_correct",
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "n": len(evaluated),
        "text_flips": {
            "exact": sum(row["text_exact_flip"] for row in evaluated),
            "normalized": sum(
                row["text_normalized_flip"] for row in evaluated
            ),
        },
        "pope_compatible": pope,
        "strict_explicit": strict,
    }, evaluated


def evaluate_paired_output(
    *,
    questions: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    meta: dict[str, Any] | None,
    expected_n: int | None,
    questions_sha256: str,
    raw_source: str | None = None,
    raw_sha256: str | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical, raw_audit = canonicalize_raw_records(
        questions,
        raw_records,
        raw_source=raw_source,
        raw_sha256=raw_sha256,
    )
    metrics, canonical = evaluate_canonical_pairs(
        questions,
        canonical,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    meta_fingerprint = meta.get("fingerprint") if meta else None
    record_fingerprints = {
        row.get("fingerprint") for row in raw_records
    }
    question_ids = [qid(row) for row in questions]
    canonical_ids = [qid(row) for row in canonical]
    gates = {
        "expected_n_declared": expected_n is not None,
        "questions_match_expected_n": (
            expected_n is not None and len(questions) == expected_n
        ),
        "canonical_successes_match_expected_n": (
            expected_n is not None and len(canonical) == expected_n
        ),
        "questions_fully_covered_in_order": canonical_ids == question_ids,
        "meta_present": meta is not None,
        "meta_fingerprint_present": (
            isinstance(meta_fingerprint, str) and bool(meta_fingerprint)
        ),
        "all_raw_records_match_meta_fingerprint": (
            meta_fingerprint is not None
            and record_fingerprints == {meta_fingerprint}
        ),
        "meta_questions_sha256_matches": (
            meta is not None
            and meta.get("questions_sha256") == questions_sha256
        ),
        "meta_n_requested_matches_expected": (
            meta is not None
            and expected_n is not None
            and meta.get("n_requested") == expected_n
        ),
        "meta_n_complete_matches_canonical": (
            meta is not None
            and meta.get("n_complete") == len(canonical)
        ),
    }
    final = all(gates.values())
    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "status": "final" if final else "interim",
        "final": final,
        "finalization_gates": gates,
        "interim_reasons": [
            key for key, passed in gates.items() if not passed
        ],
        "meta_fingerprint": meta_fingerprint,
        "questions_sha256": questions_sha256,
        "raw_provenance": raw_audit,
        "metrics": metrics,
        "provenance_limitations": [
            (
                "Current inference-v2 fingerprints omit inference-code, base-"
                "model-tree, and image-manifest hashes; this evaluator does not "
                "retroactively attest those identities."
            )
        ],
    }
    return report, canonical


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False)
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def compact_cli_summary(
    report: dict[str, Any],
    *,
    output: Path,
    canonical: Path,
) -> dict[str, Any]:
    """Return monitoring-safe stdout without per-record provenance."""
    metrics = report["metrics"]

    def paired(metric: dict[str, Any]) -> dict[str, Any]:
        bootstrap = metric["patient_cluster_bootstrap"]
        return {
            "base_accuracy": metric["base"]["accuracy"],
            "adapted_accuracy": metric["adapted"]["accuracy"],
            "delta_pp": metric["delta_pp"],
            "label_flips": metric["label_flips"],
            "rescues": metric["rescues"],
            "harms": metric["harms"],
            "mcnemar_exact_p": metric["mcnemar"][
                "two_sided_exact_p"
            ],
            "patient_cluster_95_ci": bootstrap.get(
                "percentile_95_ci"
            ),
        }

    return {
        "status": report["status"],
        "final": report["final"],
        "failed_gates": report["interim_reasons"],
        "n": metrics["n"],
        "text_flips": metrics["text_flips"],
        "pope_compatible": paired(metrics["pope_compatible"]),
        "strict_explicit": paired(metrics["strict_explicit"]),
        "outputs": {
            "metrics": str(output.resolve()),
            "canonical": str(canonical.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--expected-n", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    question_bytes = args.questions.read_bytes()
    questions = [
        json.loads(line)
        for line in question_bytes.splitlines()
        if line.strip()
    ]
    raw_records, raw_sha256 = load_jsonl_with_provenance(args.input)
    meta = (
        json.loads(args.meta.read_text())
        if args.meta and args.meta.is_file()
        else None
    )
    report, canonical = evaluate_paired_output(
        questions=questions,
        raw_records=raw_records,
        meta=meta,
        expected_n=args.expected_n,
        questions_sha256=sha256_bytes(question_bytes),
        raw_source=str(args.input.resolve()),
        raw_sha256=raw_sha256,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    atomic_json(args.output, report)
    write_jsonl(args.canonical, canonical)
    print(
        json.dumps(
            compact_cli_summary(
                report, output=args.output, canonical=args.canonical
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
