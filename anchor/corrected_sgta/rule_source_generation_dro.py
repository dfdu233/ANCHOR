"""Class-balanced generation-aligned source GroupDRO residual."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from corrected_sgta.rule_source_absolute_margin_batch_dro import mean_domain_losses, select_worst_domain
from corrected_sgta.rule_source_preference import LinearLowRankResidual, stable_json_sha256

VERSION = "rule-source-generation-centered-dro-v1"
RANK = 16
MAX_RELATIVE_UPDATE = 0.02
TRAIN_IMAGES_PER_SOURCE = 95
DOMAIN_BATCH_SIZE = 4
SOURCE_DOMAINS = 3


class TokenCenteredLowRankResidual(LinearLowRankResidual):
    """Low-rank residual with exactly zero mean over visual tokens."""
    def raw_update(self, value: torch.Tensor) -> torch.Tensor:
        update = super().raw_update(value)
        if update.ndim < 2:
            raise ValueError("visual token tensor must have at least two dimensions")
        return update - update.mean(dim=-2, keepdim=True)


def experiment_fingerprint(*, manifest_contract: Mapping[str, Any], config: Mapping[str, Any], selected: Mapping[str, Sequence[Mapping[str, Any]]], code_sha256: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    payload = {"version": VERSION, "manifest_contract": dict(manifest_contract), "config": dict(config), "selected": {name: [dict(row) for row in rows] for name, rows in sorted(selected.items())}, "code_sha256": dict(sorted(code_sha256.items()))}
    return stable_json_sha256(payload), payload
