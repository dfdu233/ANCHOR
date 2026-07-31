import json
from pathlib import Path

import numpy as np

from anchor.corrected_sgta.analyze_style_prior_template_probe import (
    frozen_reference_style_margin,
    held_patient_template_style_identification,
    load_template_evidence,
    paired_half_retention,
)


def test_held_patient_template_id_recovers_shared_style() -> None:
    rng = np.random.default_rng(101)
    signature = rng.normal(size=(4, 5))
    first = np.stack(
        [signature + rng.normal(scale=0.01, size=signature.shape) for _ in range(12)]
    )
    second = np.stack(
        [signature + rng.normal(scale=0.01, size=signature.shape) for _ in range(12)]
    )
    result = held_patient_template_style_identification(
        first,
        second,
        [f"p-{index}" for index in range(12)],
        repeats=30,
        seed=103,
    )
    assert result["symmetric_accuracy"] > 0.95
    assert result["symmetric_accuracy"] > result["chance"]


def test_template_loader_forms_complete_sentence_contrast(tmp_path: Path) -> None:
    path = tmp_path / "probe.jsonl"
    rows = []
    for case_index in range(8):
        for view in ("real", "null", "style_0", "style_1", "style_2"):
            for disease_index, disease in enumerate(("a", "b")):
                for polarity, nll in (("positive", 0.2), ("negative", 0.7)):
                    rows.append(
                        {
                            "case_id": f"c-{case_index}",
                            "image_relative": (
                                f"p00/p{10000000 + case_index}/s1/x.jpg"
                            ),
                            "view": view,
                            "disease": disease,
                            "polarity": polarity,
                            "sequence_nll": nll + disease_index,
                            "answer_token_count": 4,
                            "model": "m",
                            "template_id": "t",
                        }
                    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    evidence, cases, patients, diseases, styles, model, metadata = (
        load_template_evidence(path, "t")
    )
    assert evidence.shape == (8, 5, 2)
    assert np.allclose(evidence, 0.5)
    assert len(cases) == len(patients) == 8
    assert diseases == ["a", "b"]
    assert styles == ["style_0", "style_1", "style_2"]
    assert model == "m"
    assert metadata["template_id"] == "t"


def test_frozen_reference_margin_prefers_matched_style() -> None:
    rng = np.random.default_rng(107)
    signature = rng.normal(size=(4, 5))
    reference = np.stack(
        [signature + rng.normal(scale=0.01, size=signature.shape) for _ in range(20)]
    )
    target = np.stack(
        [signature + rng.normal(scale=0.01, size=signature.shape) for _ in range(10)]
    )
    result = frozen_reference_style_margin(
        reference,
        [f"r-{index}" for index in range(20)],
        target,
        [f"t-{index}" for index in range(10)],
        repeats=30,
        seed=109,
    )
    assert result["patient_balanced_matched_minus_mismatched_cosine"] > 0.9
    assert result["patient_cluster_bootstrap_95pct"][0] > 0.9


def test_half_retention_accepts_equal_paired_margins() -> None:
    result = paired_half_retention(
        [0.2] * 10,
        [0.2] * 10,
        [f"p-{index}" for index in range(10)],
        repeats=30,
        seed=113,
    )
    assert result["alternative_minus_half_reference"] > 0
    assert result["patient_cluster_bootstrap_95pct"][0] > 0
