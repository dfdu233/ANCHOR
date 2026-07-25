"""Offline-reproducible subset of MedHEval open-ended metrics."""

from __future__ import annotations

from collections import Counter

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .oe_metrics import lcs_length, word_tokens


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)
    )


def rouge_n(candidate: str, reference: str, n: int) -> float:
    candidate_tokens = word_tokens(candidate)
    reference_tokens = word_tokens(reference)
    candidate_counts = _ngram_counts(candidate_tokens, n)
    reference_counts = _ngram_counts(reference_tokens, n)
    if not candidate_counts or not reference_counts:
        return 0.0
    overlap = sum((candidate_counts & reference_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2.0 * precision * recall / (precision + recall)


def rouge_l(candidate: str, reference: str) -> float:
    candidate_tokens = word_tokens(candidate)
    reference_tokens = word_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = lcs_length(candidate_tokens, reference_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def token_f1(candidate: str, reference: str) -> float:
    candidate_counts = Counter(word_tokens(candidate))
    reference_counts = Counter(word_tokens(reference))
    if not candidate_counts or not reference_counts:
        return 0.0
    overlap = sum((candidate_counts & reference_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2.0 * precision * recall / (precision + recall)


def bleu4(candidate: str, reference: str) -> float:
    candidate_tokens = word_tokens(candidate)
    reference_tokens = word_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=SmoothingFunction().method4,
        )
    )


def lexical_metrics(candidate: str, reference: str) -> dict[str, float]:
    return {
        "bleu4": bleu4(candidate, reference),
        "rouge_1": rouge_n(candidate, reference, 1),
        "rouge_2": rouge_n(candidate, reference, 2),
        "rouge_l": rouge_l(candidate, reference),
        "token_f1": token_f1(candidate, reference),
    }


def lexical_admissible(
    candidate: str, reference: str, rouge_threshold: float = 0.30
) -> bool:
    return rouge_l(candidate, reference) >= rouge_threshold
