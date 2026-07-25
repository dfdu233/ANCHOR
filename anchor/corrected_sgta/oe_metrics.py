"""Dependency-light, deterministic metrics for open-ended calibration.

MedHEval's public Knowledge hallucination judge imports a proprietary Bedrock
client.  Consequently, conformal calibration uses an explicitly named lexical
admissibility proxy.  These functions never inspect a test answer to select a
candidate; references are used only for calibration and final evaluation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def word_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(str(text))]


def lcs_length(left: Iterable[str], right: Iterable[str]) -> int:
    """Return LCS length using O(min(n,m)) memory."""

    a, b = list(left), list(right)
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for index, other in enumerate(b, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """Word-level ROUGE-L F1, matching the public benchmark's score family."""

    candidate_tokens = word_tokens(candidate)
    reference_tokens = word_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    common = lcs_length(candidate_tokens, reference_tokens)
    if common == 0:
        return 0.0
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
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


def lexical_metrics(candidate: str, reference: str) -> dict[str, float]:
    return {
        "rouge_l": rouge_l(candidate, reference),
        "token_f1": token_f1(candidate, reference),
    }


def lexical_admissible(
    candidate: str, reference: str, rouge_threshold: float = 0.30
) -> bool:
    """Fixed proxy used by ConfGen; not a clinical correctness judgment."""

    return rouge_l(candidate, reference) >= rouge_threshold
