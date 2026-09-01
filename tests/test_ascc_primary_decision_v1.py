from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anchor.corrected_sgta.run_huatuo_ascc_interaction_v1 import (
    PRIMARY_EDGE,
    PROMPTS,
    record_key,
)


ROOT = Path(__file__).resolve().parents[1]
SCORE_DIR = ROOT / "corrected_runs/ascc/huatuo_score_v3"
SUBSTRATE = ROOT / "corrected_runs/ascc/confirmatory_substrate_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ascc_v1_is_complete_but_superseded_by_construct_invalidation() -> None:
    decision = json.loads(
        (SCORE_DIR / "primary_progression_decision_v1.json").read_text()
    )
    analysis = json.loads((SCORE_DIR / "primary_analysis_v1.json").read_text())
    rows = [
        json.loads(line)
        for line in (SUBSTRATE / "selected_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["edge_id"] == PRIMARY_EDGE]
    expected = {
        record_key(row["item_id"], prompt["name"]) + ".json"
        for row in rows
        for prompt in PROMPTS
    }
    actual = {path.name for path in (SCORE_DIR / "shards").glob("*.json")}

    assert actual == expected
    assert decision["completion_audit"]["expected_shards"] == len(expected) == 1552
    assert decision["provenance"]["analysis_sha256"] == _sha256(
        SCORE_DIR / "primary_analysis_v1.json"
    )
    assert decision["frozen_gate"] == analysis["analysis"]["gate"]
    invalidation = json.loads(
        (SCORE_DIR / "INVALIDATED_BEFORE_OUTCOME_INSPECTION.json").read_text()
    )
    assert decision["decision"] == "no_go"
    assert invalidation["status"] == "construct_invalidated_before_outcome_inspection"
    assert invalidation["formal_analysis_authorized"] is False
    assert invalidation["hidden_state_patch_authorized"] is False
    assert invalidation["score_config_sha256"] == _sha256(
        SCORE_DIR / "score_config.json"
    )
