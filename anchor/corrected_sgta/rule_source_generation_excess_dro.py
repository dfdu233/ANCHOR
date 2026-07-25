"""Class-balanced generation-aligned excess-risk Source-DRO."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from corrected_sgta.rule_source_absolute_margin_batch_dro import mean_domain_losses, select_worst_domain
from corrected_sgta.rule_source_preference import LinearLowRankResidual, stable_json_sha256
VERSION="rule-source-generation-excess-dro-v1"
RANK=16
MAX_RELATIVE_UPDATE=0.02
TRAIN_IMAGES_PER_SOURCE=95
DOMAIN_BATCH_SIZE=4
SOURCE_DOMAINS=3

def experiment_fingerprint(*, manifest_contract: Mapping[str, Any], config: Mapping[str, Any], selected: Mapping[str, Sequence[Mapping[str, Any]]], code_sha256: Mapping[str, str]):
    payload={"version":VERSION,"manifest_contract":dict(manifest_contract),"config":dict(config),"selected":{n:[dict(r) for r in rows] for n,rows in sorted(selected.items())},"code_sha256":dict(sorted(code_sha256.items()))}
    return stable_json_sha256(payload),payload
