"""Core objects for a source-preference post-projector adapter.

The learned update is deliberately linear:

    T_d(h) = B_d A_d h.

For inference, source updates are averaged before applying one relative trust
region.  Averaging the functions ``B_d A_d`` rather than the factors makes the
source barycenter invariant to low-rank gauge reparameterizations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


VERSION = "rule-source-preference-barycenter-v1"
IGNORE_INDEX = -100
RULE_MIMIC_NO_REFERENCE_SUFFIX = (
    "Please answer the question based on the image and report and choose from "
    "the following two options: [yes, no]."
)


def stable_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_mimic_prompt(question: object) -> str:
    """Return the exact no-reference prompt used by RULE MIMIC inference."""
    value = str(question).replace("<image>", "").strip()
    if not value:
        raise ValueError("empty RULE question")
    if RULE_MIMIC_NO_REFERENCE_SUFFIX.lower() in value.lower():
        return value
    return f"{value} {RULE_MIMIC_NO_REFERENCE_SUFFIX}"


def canonical_binary_answer(value: object) -> str:
    direct = str(value).strip().lower().rstrip(".")
    if direct not in {"yes", "no"}:
        raise ValueError(f"expected canonical binary answer, got {value!r}")
    return direct.capitalize() + "."


def opposite_binary_answer(value: object) -> str:
    canonical = canonical_binary_answer(value)
    return "No." if canonical == "Yes." else "Yes."


def target_ids_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """Extract causal answer-token targets corresponding to answer logits."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    targets = labels[:, 1:]
    targets = targets[targets.ne(IGNORE_INDEX)]
    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError("labels contain no answer-token targets")
    return targets


def sequence_log_probability(
    answer_logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Sum token log probabilities for one complete candidate sequence."""
    if answer_logits.ndim != 2:
        raise ValueError("answer logits must have shape [tokens, vocabulary]")
    targets = target_ids.reshape(-1).to(answer_logits.device)
    if answer_logits.shape[0] != targets.numel():
        raise ValueError(
            "answer-logit/target length mismatch: "
            f"{answer_logits.shape[0]} != {targets.numel()}"
        )
    if targets.min() < 0 or targets.max() >= answer_logits.shape[1]:
        raise ValueError("target token id is outside the answer vocabulary")
    selected = F.log_softmax(answer_logits.float(), dim=-1).gather(
        1, targets[:, None]
    )
    value = selected.sum()
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("non-finite sequence log probability")
    return value


def preference_improvement_loss(
    adapted_positive_logp: torch.Tensor,
    adapted_negative_logp: torch.Tensor,
    reference_positive_logp: torch.Tensor,
    reference_negative_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO loss on improvement of the full-sequence positive/negative margin."""
    if beta <= 0:
        raise ValueError("preference beta must be positive")
    adapted_margin = adapted_positive_logp - adapted_negative_logp
    reference_margin = (
        reference_positive_logp.detach() - reference_negative_logp.detach()
    )
    improvement = adapted_margin - reference_margin
    loss = -F.logsigmoid(beta * improvement)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(improvement)):
        raise FloatingPointError("non-finite preference loss/margin")
    return loss, adapted_margin, improvement


def pooled_preference_objective(
    losses: Sequence[torch.Tensor],
    aggregation: str,
) -> torch.Tensor:
    """Exact mean or worst-domain objective for a balanced source step."""
    if not losses:
        raise ValueError("pooled preference objective requires domain losses")
    stacked = torch.stack(list(losses))
    if aggregation == "mean":
        return stacked.mean()
    if aggregation == "worst":
        return stacked.max()
    raise ValueError(f"unknown pooled aggregation: {aggregation}")


def apply_relative_cap(
    reference: torch.Tensor, raw_update: torch.Tensor, maximum: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap a raw update once per token relative to the reference feature norm."""
    if reference.shape != raw_update.shape:
        raise ValueError("reference/update shapes differ")
    if maximum <= 0:
        raise ValueError("maximum relative update must be positive")
    reference_norm = reference.detach().float().norm(dim=-1, keepdim=True)
    update_norm = raw_update.float().norm(dim=-1, keepdim=True)
    permitted = maximum * reference_norm
    scale = torch.where(
        update_norm > permitted,
        permitted / update_norm.clamp_min(1e-12),
        torch.ones_like(update_norm),
    )
    return raw_update.float() * scale, scale


class LinearLowRankResidual(nn.Module):
    """Linear ``BA`` update with zero initialization and one relative cap."""

    def __init__(self, width: int, rank: int, max_relative_update: float):
        super().__init__()
        if width <= 0 or rank <= 0 or max_relative_update <= 0:
            raise ValueError("width, rank, and max_relative_update must be positive")
        self.width = int(width)
        self.rank = int(rank)
        self.max_relative_update = float(max_relative_update)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.last_mean_relative_norm = 0.0
        self.last_max_relative_norm = 0.0

    def raw_update(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.width:
            raise ValueError("input width does not match low-rank module")
        hidden = F.linear(value.float(), self.down.weight.float())
        return F.linear(hidden, self.up.weight.float())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update, _ = apply_relative_cap(
            value, self.raw_update(value), self.max_relative_update
        )
        relative = update.norm(dim=-1) / value.detach().float().norm(
            dim=-1
        ).clamp_min(1e-12)
        self.last_mean_relative_norm = float(relative.mean().detach())
        self.last_max_relative_norm = float(relative.max().detach())
        return value + update.to(value.dtype)

    def dense_update_matrix(self) -> torch.Tensor:
        return self.up.weight.float() @ self.down.weight.float()


def _factor_pair(
    state: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = set(state)
    required = {"down.weight", "up.weight"}
    if not required <= keys:
        raise ValueError(f"low-rank state lacks keys {sorted(required - keys)}")
    down = state["down.weight"].detach().clone()
    up = state["up.weight"].detach().clone()
    if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
        raise ValueError("invalid low-rank factor shapes")
    return down, up


def dense_barycenter_matrix(
    states: Mapping[str, Mapping[str, torch.Tensor]],
    included_domains: Sequence[str] | None = None,
) -> torch.Tensor:
    """Return the exact uniform mean of the gauge-invariant dense ``BA`` maps."""
    names = sorted(states) if included_domains is None else sorted(included_domains)
    if not names:
        raise ValueError("source barycenter requires at least one domain")
    if len(names) != len(set(names)) or any(name not in states for name in names):
        raise ValueError("invalid or duplicate included source domain")
    matrices = []
    shape = None
    for name in names:
        down, up = _factor_pair(states[name])
        matrix = up.float() @ down.float()
        shape = matrix.shape if shape is None else shape
        if matrix.shape != shape:
            raise ValueError("source low-rank updates have incompatible widths")
        matrices.append(matrix)
    return torch.stack(matrices).mean(dim=0)


class SourceBarycenterResidual(nn.Module):
    """Average source ``BA`` functions, then apply exactly one relative cap."""

    def __init__(
        self,
        states: Mapping[str, Mapping[str, torch.Tensor]],
        max_relative_update: float,
        included_domains: Sequence[str] | None = None,
    ):
        super().__init__()
        names = sorted(states) if included_domains is None else sorted(included_domains)
        if not names:
            raise ValueError("source barycenter requires at least one domain")
        if len(names) != len(set(names)) or any(name not in states for name in names):
            raise ValueError("invalid or duplicate included source domain")
        self.domain_names = tuple(names)
        self.max_relative_update = float(max_relative_update)
        first_width = None
        self.factors = nn.ModuleDict()
        for name in names:
            down, up = _factor_pair(states[name])
            width = int(down.shape[1])
            if up.shape[0] != width:
                raise ValueError("source update is not width preserving")
            first_width = width if first_width is None else first_width
            if width != first_width:
                raise ValueError("source low-rank updates have incompatible widths")
            module = LinearLowRankResidual(
                width, int(down.shape[0]), max_relative_update
            )
            with torch.no_grad():
                module.down.weight.copy_(down)
                module.up.weight.copy_(up)
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            self.factors[name] = module
        self.width = int(first_width)
        self.last_mean_relative_norm = 0.0
        self.last_max_relative_norm = 0.0

    def raw_update(self, value: torch.Tensor) -> torch.Tensor:
        updates = [self.factors[name].raw_update(value) for name in self.domain_names]
        return torch.stack(updates, dim=0).mean(dim=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update, _ = apply_relative_cap(
            value, self.raw_update(value), self.max_relative_update
        )
        relative = update.norm(dim=-1) / value.detach().float().norm(
            dim=-1
        ).clamp_min(1e-12)
        self.last_mean_relative_norm = float(relative.mean().detach())
        self.last_max_relative_norm = float(relative.max().detach())
        return value + update.to(value.dtype)

    def dense_update_matrix(self) -> torch.Tensor:
        return dense_barycenter_matrix(
            {
                name: self.factors[name].state_dict()
                for name in self.domain_names
            }
        )


def validate_source_manifest(
    manifest_path: Path,
    source_jsons: Mapping[str, Path],
    locked_test: Path,
) -> dict[str, Any]:
    """Fail closed unless source files and locked target match one manifest."""
    manifest = json.loads(manifest_path.read_text())
    fingerprint = manifest.get("fingerprint")
    if manifest.get("version") != "rule-source-manifest-v2":
        raise ValueError("unsupported source-manifest version")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("source manifest lacks a SHA-256 fingerprint")
    locked = manifest.get("locked_test")
    if not isinstance(locked, dict):
        raise ValueError("source manifest lacks locked-test audit")
    if locked.get("source_overlap") != 0:
        raise ValueError("source manifest reports source/target image leakage")
    if locked.get("labels_read_for_selection") is not False:
        raise ValueError("source manifest does not seal target labels")
    configured_target = Path(manifest["config"]["locked_test"]).resolve()
    if configured_target != locked_test.resolve():
        raise ValueError("locked target does not match source manifest")
    declared = manifest.get("outputs", {}).get("by_domain", {})
    if set(source_jsons) != set(declared):
        raise ValueError("source domains do not match source manifest")
    source_sha256: dict[str, str] = {}
    for domain, path in sorted(source_jsons.items()):
        actual = file_sha256(path)
        expected = declared[domain]["train"]["json_sha256"]
        if actual != expected:
            raise ValueError(f"source JSON hash mismatch for {domain}")
        source_sha256[domain] = actual
    return {
        "manifest_version": manifest["version"],
        "manifest_fingerprint": fingerprint,
        "manifest_sha256": file_sha256(manifest_path),
        "source_json_sha256": source_sha256,
        "locked_test_sha256": file_sha256(locked_test),
        "source_target_image_overlap": 0,
        "target_labels_read_for_selection": False,
    }


def experiment_fingerprint(
    *,
    manifest_contract: Mapping[str, Any],
    config: Mapping[str, Any],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    code_sha256: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
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
