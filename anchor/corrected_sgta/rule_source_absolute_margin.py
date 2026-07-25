"""Mathematical core for the preregistered source-DRO margin pilot.

For a binary answer, define the complete-sequence margin

    m_theta(x) = log p_theta("Yes." | x) - log p_theta("No." | x)

and encode the ground truth as ``y in {-1, +1}``.  The shared adapter minimizes
the worst-source logistic risk ``max_d E_d softplus(-y m_theta(x))``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from corrected_sgta.rule_source_preference import stable_json_sha256


VERSION = "rule-source-absolute-margin-dro-v1"
RANK = 16
MAX_RELATIVE_UPDATE = 0.02
TRAIN_IMAGES_PER_SOURCE = 95
DEV_IMAGES_TOTAL = 85
SOURCE_DOMAINS = 3


def experiment_fingerprint(
    *,
    manifest_contract: Mapping[str, Any],
    config: Mapping[str, Any],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    code_sha256: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    """Bind the absolute-margin protocol, inputs, and implementation bytes."""
    payload = {
        "version": VERSION,
        "manifest_contract": dict(manifest_contract),
        "config": dict(config),
        "selected": {
            name: [dict(row) for row in rows]
            for name, rows in sorted(selected.items())
        },
        "code_sha256": dict(sorted(code_sha256.items())),
    }
    return stable_json_sha256(payload), payload


def binary_sign(answer: object) -> int:
    """Map a canonical Yes/No answer to its signed-margin target."""
    value = str(answer).strip().lower().rstrip(".")
    if value == "yes":
        return 1
    if value == "no":
        return -1
    raise ValueError(f"expected a Yes/No answer, got {answer!r}")


def absolute_margin_loss(
    yes_log_probability: torch.Tensor,
    no_log_probability: torch.Tensor,
    target_sign: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return calibrated binary logistic loss on the absolute sequence margin."""
    if target_sign not in {-1, 1}:
        raise ValueError("target_sign must be -1 or +1")
    margin = yes_log_probability - no_log_probability
    signed_margin = margin * target_sign
    loss = F.softplus(-signed_margin)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(margin)):
        raise FloatingPointError("non-finite absolute-margin loss")
    return loss, margin


def select_worst_domain(losses: Mapping[str, torch.Tensor]) -> str:
    """Select the exact empirical worst source, with a deterministic tie break."""
    if not losses:
        raise ValueError("source-DRO requires at least one domain loss")
    values: dict[str, float] = {}
    for name, loss in losses.items():
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise ValueError(f"invalid scalar loss for domain {name!r}")
        values[name] = float(loss.detach())
    maximum = max(values.values())
    return min(name for name, value in values.items() if value == maximum)


def source_dro_dev_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the frozen 85-example source-dev gate is passed."""
    primary = summary.get("source_dro")
    if not isinstance(primary, Mapping):
        return {
            "status": "failed",
            "target_evaluation_allowed": False,
            "reason": "source_dro results are absent",
        }
    micro = primary.get("micro", {})
    per_domain = primary.get("per_domain", {})
    rescues = int(micro.get("rescues", 0))
    harms = int(micro.get("harms", 0))
    net_rescues = rescues - harms
    nondeclining = sum(
        float(values["delta_pp"]) >= 0.0 for values in per_domain.values()
    )
    checks = {
        "complete_source_dev": {
            "value": int(micro.get("n", -1)),
            "required": DEV_IMAGES_TOTAL,
            "passed": int(micro.get("n", -1)) == DEV_IMAGES_TOTAL,
        },
        "net_rescues": {
            "value": net_rescues,
            "required": f">=3/{DEV_IMAGES_TOTAL}",
            "passed": net_rescues >= 3,
        },
        "nondeclining_domains": {
            "value": nondeclining,
            "domains": len(per_domain),
            "required": ">=2/3",
            "passed": (
                len(per_domain) == SOURCE_DOMAINS and nondeclining >= 2
            ),
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "passed" if passed else "failed",
        "target_evaluation_allowed": passed,
        "primary_variant": "source_dro",
        "checks": checks,
        "selection_note": (
            "This one-shot gate is frozen before target evaluation; failure "
            "terminates the absolute-margin pilot without target-label access."
        ),
    }
