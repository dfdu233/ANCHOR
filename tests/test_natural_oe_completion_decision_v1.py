import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "corrected_runs"
    / "specificity_ratchet"
    / "natural_oe_diagnostic_completion_huatuo_v1"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_decision_binds_analysis_and_stops_progression():
    decision = json.loads((RUN / "decision_v1.json").read_text())
    assert decision["decision"] == "no_go_diagnostic_completion_mechanism"
    assert _sha(ROOT / decision["analysis"]) == decision["analysis_sha256"]
    assert _sha(RUN / "generations.jsonl") == decision["generations_sha256"]
    assert decision["strict_events"]["total"] == 2
    assert decision["generation"]["cap_hits"] == 10
    assert all(
        decision["gates"][key] is False
        for key in (
            "response_geometry_passed",
            "repeated_extreme_event_substrate_passed",
            "physician_construct_review_authorized",
            "second_model_generation_authorized",
            "hidden_state_replay_authorized",
            "larger_generation_authorized",
        )
    )
