import json

from anchor.medeval.analyze_commitment_dissociation import analyze


def test_content_change_with_stable_commitment_is_only_exploratory():
    rows = []
    for index in range(4):
        for prompt in ("a", "b", "c"):
            rows.extend([
                {"id": str(index), "conv_mode": "native", "prompt_mode": prompt,
                 "view": "real", "text": f"Left pleural effusion {index}."},
                {"id": str(index), "conv_mode": "native", "prompt_mode": prompt,
                 "view": "shuffled", "text": f"Right upper lobe opacity {index}."},
            ])
    config = {
        "protocol": "test", "primary_control": "shuffled", "secondary_controls": [],
        "uncertainty_patterns": ["\\bmay\\b"], "minimum_complete_primary_pairs": 12,
        "minimum_mean_content_change": 0.3, "maximum_mean_commitment_shift": 0.1,
        "bootstrap_replicates": 100, "bootstrap_seed": 1,
        "interpretation": "lead only",
    }
    result = analyze(rows, config)
    assert result["exploratory_dissociation_gate_pass"]
    assert result["claim_grade_allowed"] is False
