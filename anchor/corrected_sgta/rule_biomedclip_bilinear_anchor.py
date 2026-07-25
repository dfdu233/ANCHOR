"""Frozen-BiomedCLIP bilinear source anchor for RULE Yes/No evaluation.

This module is deliberately small: a rank-r residual metric is learned on
external source domains, then its log-odds are added to the VLM's complete
``Yes.``/``No.`` sequence margin (an odds product).  It never reads target
labels during source-only validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


VERSION = "rule-biomedclip-bilinear-anchor-v1"
ANCHOR_TEMPLATE = "Medical image question: {question} Answer: {answer}."
DOMAIN_NAMES = ("rule_iuxray", "slake_xray", "vqa_rad_train")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_question(value: str) -> str:
    """Apply the only label-free text canonicalization used by the method."""
    lines = [line.strip() for line in str(value).splitlines()]
    kept = [line for line in lines if line and line != "<image>"]
    return " ".join(" ".join(kept).split()).rstrip("?").strip() + "?"


def anchor_texts(question: str) -> tuple[str, str]:
    question = canonical_question(question)
    return (
        ANCHOR_TEMPLATE.format(question=question, answer="yes"),
        ANCHOR_TEMPLATE.format(question=question, answer="no"),
    )


def answer_index(value: str) -> int:
    normalized = str(value).strip().lower().rstrip(".")
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    raise ValueError(f"Expected Yes./No. answer, got {value!r}")


def assert_disjoint_by_image(
    train_rows: Iterable[dict[str, Any]],
    dev_rows: Iterable[dict[str, Any]],
) -> None:
    """Fail closed when either decoded-image or source-blob hashes overlap."""
    dev_rows = list(dev_rows)
    image_hashes = {str(row["image_sha256"]) for row in dev_rows}
    blob_hashes = {str(row["image_blob_sha256"]) for row in dev_rows}
    conflicts = [
        str(row["id"])
        for row in train_rows
        if str(row["image_sha256"]) in image_hashes
        or str(row["image_blob_sha256"]) in blob_hashes
    ]
    if conflicts:
        raise ValueError(
            f"Source train overlaps frozen dev by image content: {conflicts[:5]}"
        )


class ResidualBilinearAnchor(nn.Module):
    """Rank-r residual metric ``u^T (I + A B^T) d`` with no free bias."""

    def __init__(self, dimension: int, rank: int, *, seed: int = 42) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.left = nn.Parameter(
            torch.randn(dimension, rank, generator=generator) * 1e-3
        )
        self.right = nn.Parameter(torch.zeros(dimension, rank))

    def forward(
        self,
        image_features: torch.Tensor,
        text_directions: torch.Tensor,
        *,
        logit_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        identity = (image_features * text_directions).sum(dim=-1)
        residual = (
            (image_features @ self.left) * (text_directions @ self.right)
        ).sum(dim=-1)
        return torch.as_tensor(
            logit_scale, device=identity.device, dtype=identity.dtype
        ) * (identity + residual)


def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(logits.dtype)
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Both Yes and No labels are required")
    weights = torch.where(
        labels > 0.5,
        labels.new_tensor(labels.numel() / (2.0 * float(positives))),
        labels.new_tensor(labels.numel() / (2.0 * float(negatives))),
    )
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=weights
    )


@dataclass(frozen=True)
class FoldMetrics:
    n: int
    baseline_accuracy: float
    fused_accuracy: float
    anchor_accuracy: float
    image_shuffle_accuracy: float
    text_shuffle_accuracy: float
    bias_only_accuracy: float
    rescues: int
    harms: int


def accuracy(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    return float((prediction.bool() == labels.bool()).float().mean())


def deterministic_permutation(length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)


def fold_metrics(
    *,
    labels: torch.Tensor,
    vlm_margin: torch.Tensor,
    anchor_margin: torch.Tensor,
    image_shuffle_margin: torch.Tensor,
    text_shuffle_margin: torch.Tensor,
    bias_log_odds: float,
) -> FoldMetrics:
    baseline = vlm_margin > 0
    fused = vlm_margin + anchor_margin > 0
    image_shuffle = vlm_margin + image_shuffle_margin > 0
    text_shuffle = vlm_margin + text_shuffle_margin > 0
    bias_only = vlm_margin + bias_log_odds > 0
    correct = labels.bool()
    baseline_correct = baseline == correct
    fused_correct = fused == correct
    return FoldMetrics(
        n=labels.numel(),
        baseline_accuracy=accuracy(baseline, labels),
        fused_accuracy=accuracy(fused, labels),
        anchor_accuracy=accuracy(anchor_margin > 0, labels),
        image_shuffle_accuracy=accuracy(image_shuffle, labels),
        text_shuffle_accuracy=accuracy(text_shuffle, labels),
        bias_only_accuracy=accuracy(bias_only, labels),
        rescues=int((~baseline_correct & fused_correct).sum()),
        harms=int((baseline_correct & ~fused_correct).sum()),
    )

