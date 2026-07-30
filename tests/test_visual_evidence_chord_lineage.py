import numpy as np

from anchor.corrected_sgta.compare_visual_evidence_chord_models import (
    patient_id_by_case,
    paired_lineage_comparison,
    variance_fractions,
)


def test_variance_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    values = variance_fractions(rng.normal(size=(8, 3, 2)))
    assert np.isclose(values.sum(), 1.0)
    assert np.all(values >= 0)


def test_paired_comparison_detects_medical_style_component():
    rng = np.random.default_rng(1)
    base = rng.normal(scale=0.05, size=(40, 2, 2))
    medical = base.copy()
    medical[:, 0] += np.asarray([1.0, 0.0])
    medical[:, 1] += np.asarray([-1.0, 0.0])
    result = paired_lineage_comparison(
        medical,
        base,
        ["style_0", "style_1"],
        cluster_ids=[f"p{index // 2}" for index in range(40)],
        first_visual_scale=np.ones(40),
        second_visual_scale=np.ones(40),
        draws=500,
    )
    assert result["first_minus_second"]["style"]["point"] > 0
    assert (
        result["first_minus_second"]["style"][
            "paired_patient_cluster_bootstrap_ci95"
        ][0]
        > 0
    )
    assert (
        result["normalized_style_susceptibility"][
            "first_minus_second"
        ]
        > 0
    )


def test_patient_id_by_case_uses_mimic_path():
    rows = [
        {
            "case_id": "mimic-000",
            "image_relative": (
                "p15/p15518538/s53078789/example.jpg"
            ),
        },
        {
            "case_id": "mimic-001",
            "image_relative": (
                "p15/p15518538/s99999999/example.jpg"
            ),
        },
    ]
    assert patient_id_by_case(rows) == {
        "mimic-000": "p15518538",
        "mimic-001": "p15518538",
    }
