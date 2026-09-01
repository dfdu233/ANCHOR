#!/usr/bin/env python3
"""Strict, image-clustered evaluation for short-answer open medical VQA.

These lexical scores are transparent benchmark proxies.  They are not a
clinical hallucination judge and are deliberately separated from report and
atomic-claim evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from anchor.medeval.hashing import sha256_file
from anchor.medeval.stats import cluster_bootstrap_metric
from corrected_sgta.evaluate_oe_reports import score_text_pair
from corrected_sgta.oe_metrics import rouge_l


PROTOCOL_ID = "anchor-oe-vqa-lexical-v4-official-generation-metrics"
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
ARTICLES = {"a", "an", "the"}


def answer_tokens(text: str) -> list[str]:
    """Deterministic VQA-style normalization, without semantic rewriting."""

    import re

    raw = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())
    return [NUMBER_WORDS.get(token, token) for token in raw if token not in ARTICLES]


def normalize_answer(text: str) -> str:
    return " ".join(answer_tokens(text))


def token_f1(candidate: str, reference: str) -> float:
    candidate_counts = Counter(answer_tokens(candidate))
    reference_counts = Counter(answer_tokens(reference))
    if not candidate_counts or not reference_counts:
        return 0.0
    overlap = sum((candidate_counts & reference_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2 * precision * recall / (precision + recall)


def answer_token_recall(candidate: str, reference: str) -> float:
    """Reference-token coverage without penalizing a verbose answer's length."""

    candidate_counts = Counter(answer_tokens(candidate))
    reference_counts = Counter(answer_tokens(reference))
    if not reference_counts:
        return 0.0
    return sum((candidate_counts & reference_counts).values()) / sum(reference_counts.values())


def meteor(candidate: str, reference: str) -> float:
    """Official NLTK METEOR over the same frozen VQA-normalized tokens."""

    import nltk
    from nltk.translate.meteor_score import meteor_score

    local_data = Path("/home/dbw/nltk_data")
    if str(local_data) not in nltk.data.path:
        nltk.data.path.insert(0, str(local_data))
    return float(meteor_score([answer_tokens(reference)], answer_tokens(candidate)))


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return payload


def _load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{number}")
            rows.append(row)
    return rows


def _row_id(row: dict[str, Any]) -> str:
    for key in ("question_id", "qid", "id", "sample_id"):
        if key in row:
            return str(row[key])
    raise ValueError("answer row has no question identifier")


def _prediction(row: dict[str, Any]) -> str:
    for key in ("text", "model_answer", "prediction", "output"):
        if key in row:
            return str(row[key] or "").strip()
    raise ValueError(f"answer {_row_id(row)!r} has no prediction field")


def align_and_score(
    manifest: list[dict[str, Any]], answer_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(manifest):
        qid = str(row.get("qid", row.get("id", index)))
        if qid in expected:
            raise ValueError(f"duplicate manifest question id {qid!r}")
        expected[qid] = row
    observed: dict[str, dict[str, Any]] = {}
    for row in answer_rows:
        qid = _row_id(row)
        if qid in observed:
            raise ValueError(f"duplicate answer question id {qid!r}")
        observed[qid] = row
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(
            f"answer alignment failure: missing={missing[:10]} extra={extra[:10]}"
        )

    scored = []
    for qid, reference_row in expected.items():
        answer_row = observed[qid]
        reference = str(reference_row.get("answer", "")).strip()
        if not reference:
            raise ValueError(f"empty manifest reference for {qid!r}")
        embedded_reference = answer_row.get("gt_ans", answer_row.get("answer"))
        if embedded_reference is not None and normalize_answer(embedded_reference) != normalize_answer(reference):
            raise ValueError(f"embedded reference mismatch for {qid!r}")
        prediction = _prediction(answer_row)
        prediction_token_count = len(answer_tokens(prediction))
        reference_token_count = len(answer_tokens(reference))
        normalized_prediction = normalize_answer(prediction)
        normalized_reference = normalize_answer(reference)
        metadata = answer_row.get("metadata")
        generated_token_count = None
        if isinstance(metadata, dict) and metadata.get("generated_token_count") is not None:
            try:
                generated_token_count = int(metadata["generated_token_count"])
            except (TypeError, ValueError):
                generated_token_count = None
        reference_phrase_covered = bool(
            normalized_reference
            and f" {normalized_reference} " in f" {normalized_prediction} "
        )
        image_identity = str(
            reference_row.get("img_name") or reference_row.get("image") or qid
        )
        image_parts = Path(image_identity).parts
        patient_parts = [
            part[1:]
            for part in image_parts
            if part.startswith("p") and part[1:].isdigit()
        ]
        patient_identity = max(patient_parts, key=len, default=None)
        cluster_id = str(
            patient_identity
            or reference_row.get("patient_id")
            or reference_row.get("image_sha256")
            or image_identity
        )
        generation_metrics = score_text_pair(prediction, reference)
        scored.append({
            "question_id": qid,
            "cluster_id": cluster_id,
            "reference": reference,
            "prediction": prediction,
            "normalized_reference": normalized_reference,
            "normalized_prediction": normalized_prediction,
            "normalized_exact": float(normalized_prediction == normalized_reference),
            "token_f1": token_f1(prediction, reference),
            "answer_token_recall": answer_token_recall(prediction, reference),
            "rouge_l": rouge_l(prediction, reference),
            "meteor": meteor(prediction, reference),
            **{
                key: generation_metrics[key]
                for key in (
                    "bleu_1", "bleu_2", "bleu_3", "bleu_4",
                    "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
                    "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
                    "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
                )
            },
            "prediction_tokens": prediction_token_count,
            "reference_tokens": reference_token_count,
            "prediction_reference_token_ratio": (
                prediction_token_count / max(reference_token_count, 1)
            ),
            "reference_phrase_covered": reference_phrase_covered,
            "ends_terminal_punctuation": prediction.rstrip().endswith((".", "!", "?")),
            "generated_token_count": generated_token_count,
        })
    return scored


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize(
    rows: list[dict[str, Any]], *, replicates: int, seed: int,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    metrics = {}
    metric_names = (
        "normalized_exact", "token_f1", "answer_token_recall",
        "reference_phrase_covered", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
        "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
        "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
        "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
        "rouge_l", "meteor",
    )
    for offset, key in enumerate(metric_names):
        metrics[key] = cluster_bootstrap_metric(
            rows,
            lambda sample, metric=key: _mean(sample, metric),
            replicates=replicates,
            seed=seed + offset,
        )
    lengths = [int(row["prediction_tokens"]) for row in rows]
    reference_lengths = [int(row["reference_tokens"]) for row in rows]
    expansion = [float(row["prediction_reference_token_ratio"]) for row in rows]
    generated_lengths = [
        int(row["generated_token_count"])
        for row in rows
        if row.get("generated_token_count") is not None
    ]
    normalized = [row["normalized_prediction"] for row in rows]
    return {
        "n_questions": len(rows),
        "n_images": len({row["cluster_id"] for row in rows}),
        "metrics": metrics,
        "output_diagnostics": {
            "empty_rate": sum(not text for text in normalized) / len(rows),
            "unique_prediction_rate": len(set(normalized)) / len(rows),
            "median_prediction_tokens": statistics.median(lengths),
            "mean_prediction_tokens": sum(lengths) / len(lengths),
            "median_reference_tokens": statistics.median(reference_lengths),
            "mean_reference_tokens": sum(reference_lengths) / len(reference_lengths),
            "median_prediction_reference_token_ratio": statistics.median(expansion),
            "mean_prediction_reference_token_ratio": sum(expansion) / len(expansion),
            "reference_phrase_coverage_rate": (
                sum(bool(row["reference_phrase_covered"]) for row in rows) / len(rows)
            ),
            "terminal_punctuation_rate": (
                sum(bool(row["ends_terminal_punctuation"]) for row in rows) / len(rows)
            ),
            "generated_token_count_coverage": len(generated_lengths) / len(rows),
            "token_budget_hit_rate": (
                sum(length >= max_new_tokens for length in generated_lengths)
                / len(generated_lengths)
                if generated_lengths and max_new_tokens
                else None
            ),
            "max_new_tokens_for_budget_diagnostic": max_new_tokens,
            "interpretation": (
                "length, lexical reference coverage, and truncation diagnostics only; "
                "none is a clinical hallucination score"
            ),
        },
    }


def paired_summary(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_id = {row["question_id"]: row for row in baseline}
    if set(baseline_by_id) != {row["question_id"] for row in candidate}:
        raise ValueError("candidate and baseline question sets differ")
    paired = []
    for row in candidate:
        other = baseline_by_id[row["question_id"]]
        if row["cluster_id"] != other["cluster_id"]:
            raise ValueError(f"candidate/baseline cluster mismatch for {row['question_id']!r}")
        paired.append({
            "question_id": row["question_id"],
            "cluster_id": row["cluster_id"],
            **{
                f"delta_{key}": float(row[key]) - float(other[key])
                for key in (
                    "normalized_exact", "token_f1", "answer_token_recall",
                    "reference_phrase_covered", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
                    "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
                    "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
                    "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
                    "rouge_l", "meteor",
                )
            },
        })
    output = {}
    for offset, key in enumerate((
        "normalized_exact", "token_f1", "answer_token_recall",
        "reference_phrase_covered", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
        "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
        "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
        "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
        "rouge_l", "meteor",
    )):
        delta = f"delta_{key}"
        output[key] = cluster_bootstrap_metric(
            paired,
            lambda sample, metric=delta: _mean(sample, metric),
            replicates=replicates,
            seed=seed + offset,
        )
    return {"direction": "candidate_minus_baseline", "metrics": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--answers", required=True, nargs="+", type=Path)
    parser.add_argument("--baseline-answers", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.limit:
        manifest = manifest[: args.limit]
    candidate = align_and_score(manifest, _load_jsonl(args.answers))
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "command": [sys.executable, *sys.argv],
        "evaluator_source": str(Path(__file__).resolve()),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "interpretation": (
            "Open-generation lexical evaluation. BLEU-1/2/3/4, ROUGE-1/2/L and "
            "METEOR match the released MedHEval metric family; normalized exact is "
            "a strict short-answer diagnostic, not the sole OE conclusion. None is "
            "a clinical hallucination or semantic correctness judge."
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_prefix_limit": args.limit,
        "answers": [str(path.resolve()) for path in args.answers],
        "answer_sha256": [sha256_file(path) for path in args.answers],
        "absolute": summarize(
            candidate,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        ),
        "records": candidate,
    }
    if args.baseline_answers:
        baseline = align_and_score(manifest, _load_jsonl(args.baseline_answers))
        payload["baseline_answers"] = [str(path.resolve()) for path in args.baseline_answers]
        payload["baseline_answer_sha256"] = [sha256_file(path) for path in args.baseline_answers]
        payload["paired"] = paired_summary(
            candidate,
            baseline,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **payload["absolute"]}, indent=2))


if __name__ == "__main__":
    main()
