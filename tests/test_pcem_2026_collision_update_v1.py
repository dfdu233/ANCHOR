from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results_reference/pcem_2026_collision_update_v1.json"


def _load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_collision_artifact_fingerprint_is_exact() -> None:
    row = _load()
    expected = row.pop("fingerprint")
    canonical = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == expected


def test_new_2026_neighbors_narrow_instead_of_promote_pcem() -> None:
    row = _load()
    causal = row["sources"]["causal_image_use_triad"]
    assert causal["official_head"] == "6acd5639f06c7ac89c890f67a7e1eef335726d47"
    assert causal["license"] == "MIT"
    assert causal["direct_collision"]["view_dependent_grounding"] == "occupied"
    assert causal["direct_collision"]["cardiomegaly_grounding"] == "occupied"
    assert "original/swap/target-mask/irrelevant-mask" in causal["mandatory_pcem_gate"]

    medfocus = row["sources"]["medfocus"]
    assert medfocus["official_head"] == "4f11fafcd6d53e8338a32c7b5a4c14f7f26db73d"
    assert medfocus["license"] == "MIT"
    assert medfocus["direct_collision"]["targeted_intervention_attribution"] == "occupied"

    decision = row["updated_collision_boundary"]
    assert decision["decision"] == "CONDITIONALLY_OPEN_BUT_STRONGLY_NARROWED"
    assert decision["image_download_authorized"] is False
    assert decision["gpu_authorized"] is False


def test_hallucxr_without_official_code_is_not_impersonated() -> None:
    hallu = _load()["sources"]["hallucxr"]
    assert hallu["official_repository"] is None
    assert hallu["official_code_found"] is False
    assert hallu["paper_native_t0"] == "not_admissible_without_official_code"
    assert "omission" in hallu["mandatory_pcem_gate"].lower()
