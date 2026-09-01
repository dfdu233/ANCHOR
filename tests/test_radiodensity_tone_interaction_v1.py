import json
from pathlib import Path

from anchor.corrected_sgta import audit_radiodensity_tone_interaction_v1 as screen


def _row(index: int, family: str, truth: str, signed_slope: float) -> dict:
    question = (
        "Is pleural effusion present?"
        if family == "opacity"
        else "Is pneumothorax present?"
    )
    return {
        "question_id": str(index),
        "patient_id": f"p-{index}",
        "question": question,
        "ground_truth": truth,
        "scores": {
            "original": {"yes_minus_no": 0.2},
            "gamma_0.9": {"yes_minus_no": signed_slope / 2},
            "gamma_1.1": {"yes_minus_no": -signed_slope / 2},
        },
        "style_guards": {
            "gamma_0.9": {"passed": True},
            "gamma_1.1": {"passed": True},
        },
    }


def test_strong_balanced_signed_interaction_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "PERMUTATIONS", 500)
    monkeypatch.setattr(screen, "BOOTSTRAPS", 500)
    rows = []
    index = 0
    for family, slope in (("opacity", 0.2), ("lucency", -0.2)):
        for truth in ("yes", "no"):
            for _ in range(20):
                rows.append(_row(index, family, truth, slope))
                index += 1
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = screen.audit(path)
    assert result["status"] == "candidate_passed_cpu_screen"
    assert result["frozen_gates"]["all_passed"] is True


def test_ambiguous_mixed_claim_is_excluded():
    assert screen.classify_claim("Is there pleural effusion or pneumothorax?") is None


def test_frozen_local_cache_rejects_candidate(monkeypatch):
    monkeypatch.setattr(screen, "PERMUTATIONS", 1_000)
    monkeypatch.setattr(screen, "BOOTSTRAPS", 1_000)
    result = screen.audit(
        Path("corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/raw.jsonl")
    )
    assert result["status"] == "candidate_rejected_cpu_screen"
    assert result["family_counts"] == {"lucency": 16, "opacity": 49}
    assert result["truth_cell_counts"]["lucency_yes"] == 3
    assert result["frozen_gates"]["prevalence_gate"] is False
    assert result["frozen_gates"]["truth_cell_gate"] is False
    assert result["frozen_gates"]["direction_and_effect_gate"] is False
