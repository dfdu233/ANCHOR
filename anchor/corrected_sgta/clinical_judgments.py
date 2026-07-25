"""Stable OE judgment IDs, blinded export helpers, and clinical admissibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def candidate_key(
    qid: object, field: str, index: int, output: dict, fingerprint: str
) -> str:
    text_hash = hashlib.sha256(
        str(output.get("text", "")).strip().encode()
    ).hexdigest()
    payload = json.dumps(
        {
            "fingerprint": fingerprint,
            "qid": str(qid),
            "field": field,
            "index": int(index),
            "style": str(output.get("style", "unknown")),
            "text_sha256": text_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def load_judgments(path: Path) -> dict[str, dict]:
    judgments: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row.get("item_id", ""))
        if not item_id:
            raise ValueError(f"missing item_id at {path}:{line_number}")
        if item_id in judgments:
            raise ValueError(f"duplicate item_id {item_id} in {path}")
        judgments[item_id] = row
    return judgments


def clinical_admissible(judgment: dict, task: str) -> bool:
    explicit = judgment.get("clinically_admissible")
    if explicit is not None and type(explicit) is not bool:
        raise ValueError("clinically_admissible must be a JSON boolean")
    if task == "knowledge":
        score = judgment.get("hallucination_score")
        if score is None:
            raise ValueError("knowledge judgment requires hallucination_score")
        score = float(score)
        if not 0.0 <= score <= 5.0:
            raise ValueError("hallucination_score must be in [0, 5]")
        derived = score <= 2.0
    elif task == "report":
        precision = judgment.get("clinical_entity_precision")
        recall = judgment.get("clinical_fact_recall")
        contradiction = judgment.get("critical_contradiction")
        if precision is None or recall is None or type(contradiction) is not bool:
            raise ValueError(
                "report judgment requires numeric clinical_entity_precision/clinical_fact_recall "
                "and boolean critical_contradiction"
            )
        derived = (
            float(precision) >= 0.80
            and float(recall) >= 0.50
            and not contradiction
        )
    else:
        raise ValueError(f"unknown OE task: {task}")
    if explicit is not None and explicit != derived:
        raise ValueError("clinically_admissible contradicts the registered task rule")
    return derived


def validate_judgment(judgment: dict, task: str) -> None:
    clinical_admissible(judgment, task)
    for key in ("clinical_entity_precision", "clinical_fact_recall", "radgraph_f1", "ratescore"):
        value = judgment.get(key)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
