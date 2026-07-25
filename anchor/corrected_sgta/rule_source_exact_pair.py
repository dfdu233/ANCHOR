"""Core contracts for exact-question visual counterfactual source training.

The canonical question function in this module is the *only* definition used
by pair construction, training, tests, and fingerprints.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from corrected_sgta.rule_source_preference import stable_json_sha256


VERSION = "rule-source-exact-pair-v1"
PAIR_MANIFEST_VERSION = "rule-source-exact-pair-manifest-v1"
DEV_EVAL_VERSION = "rule-source-exact-pair-dev-eval-v1"
RANK = 16
MAX_RELATIVE_UPDATE = 0.02
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
PAIR_WEIGHT = 1.0
STEPS = 256
SEED = 42
SOURCE_DOMAINS = 3
DEV_IMAGES_TOTAL = 85


def canonical_question(value: object) -> str:
    """Canonicalize a question without semantic rewriting.

    Only the explicit image placeholder, case, punctuation, and whitespace are
    removed.  No synonym or negation normalization is permitted.
    """
    text = str(value).replace("<image>", " ").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    canonical = " ".join(text.split())
    if not canonical:
        raise ValueError("empty canonical question")
    return canonical


def canonical_label(value: object) -> str:
    text = str(value).strip().casefold().rstrip(".")
    if text not in {"yes", "no"}:
        raise ValueError(f"expected Yes/No label, got {value!r}")
    return text


def study_id(row: Mapping[str, Any]) -> str | None:
    """Return a conservative study identifier when the source exposes one."""
    domain = str(row.get("source_domain", "")).strip()
    source_id = str(row.get("source_id", "")).strip().replace("\\", "/")
    if not source_id:
        return None
    if domain in {"rule_iuxray", "slake_xray"}:
        return source_id.split("/", 1)[0] or None
    if domain == "vqa_rad_train":
        # VQA-RAD source_id is its original image identifier.
        return source_id
    return None


def signed_margin(
    yes_log_probability: torch.Tensor,
    no_log_probability: torch.Tensor,
    target_sign: int,
) -> torch.Tensor:
    if target_sign not in {-1, 1}:
        raise ValueError("target_sign must be -1 or +1")
    value = (yes_log_probability - no_log_probability) * target_sign
    if value.numel() != 1 or not bool(torch.isfinite(value)):
        raise FloatingPointError("invalid signed margin")
    return value


def reference_relative_pair_loss(
    positive_margin: torch.Tensor,
    negative_margin: torch.Tensor,
    reference_gap: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Increase the Yes-image minus No-image margin beyond the frozen model."""
    gap = positive_margin - negative_margin
    reference = torch.as_tensor(
        reference_gap, dtype=gap.dtype, device=gap.device
    ).detach()
    improvement = gap - reference
    loss = F.softplus(-improvement)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(improvement)):
        raise FloatingPointError("non-finite exact-pair loss")
    return loss, improvement


def exact_pair_experiment_fingerprint(
    *,
    manifest_contract: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
    code_sha256: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    payload = {
        "version": VERSION,
        "manifest_contract": dict(manifest_contract),
        "pair_manifest": dict(pair_manifest),
        "config": dict(config),
        "selected": dict(selected),
        "code_sha256": dict(sorted(code_sha256.items())),
    }
    return stable_json_sha256(payload), payload


def exact_pair_dev_gate(
    summary: Mapping[str, Any],
    margin_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    primary = summary.get("source_exact_pair")
    if not isinstance(primary, Mapping):
        return {
            "status": "failed",
            "target_evaluation_allowed": False,
            "reason": "source_exact_pair results are absent",
        }
    micro = primary.get("micro", {})
    per_domain = primary.get("per_domain", {})
    rescues = int(micro.get("rescues", 0))
    harms = int(micro.get("harms", 0))
    positive = int(margin_diagnostic.get("positive_delta_count", 0))
    negative = int(margin_diagnostic.get("negative_delta_count", 0))
    checks = {
        "complete_source_dev": {
            "value": int(micro.get("n", -1)),
            "required": DEV_IMAGES_TOTAL,
            "passed": int(micro.get("n", -1)) == DEV_IMAGES_TOTAL,
        },
        "net_rescues": {
            "value": rescues - harms,
            "required": f">=3/{DEV_IMAGES_TOTAL}",
            "passed": rescues - harms >= 3,
        },
        "nondeclining_domains": {
            "value": sum(
                float(values.get("delta_pp", -float("inf"))) >= 0
                for values in per_domain.values()
            ),
            "required": ">=2/3",
            "passed": (
                len(per_domain) == SOURCE_DOMAINS
                and sum(
                    float(values.get("delta_pp", -float("inf"))) >= 0
                    for values in per_domain.values()
                )
                >= 2
            ),
        },
        "harms_not_greater_than_rescues": {
            "value": {"rescues": rescues, "harms": harms},
            "required": "harms<=rescues",
            "passed": harms <= rescues,
        },
        "not_global_margin_bias": {
            "value": {
                "positive_delta_count": positive,
                "negative_delta_count": negative,
            },
            "required": "both positive and negative margin deltas",
            "passed": positive > 0 and negative > 0,
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "passed" if passed else "failed",
        "target_evaluation_allowed": passed,
        "primary_variant": "source_exact_pair",
        "checks": checks,
        "selection_note": (
            "Frozen source-dev gate. Failure terminates the pilot without "
            "target-label access or hyperparameter changes."
        ),
    }


def pair_manifest_identity(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "version": payload.get("version"),
        "fingerprint": payload.get("fingerprint"),
        "pairs": len(payload.get("pairs", [])),
    }
