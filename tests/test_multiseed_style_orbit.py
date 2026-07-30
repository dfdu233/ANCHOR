from __future__ import annotations

import numpy as np

from anchor.corrected_sgta.analyze_multiseed_style_orbit import (
    crossed_seed_patient_bootstrap,
    parse_run,
    patient_effects,
)


def test_parse_run_requires_features_and_metadata(tmp_path):
    features = tmp_path / "seed7.npz"
    np.savez(features, layer=np.zeros((1, 1)))
    features.with_suffix(".json").write_text("{}")
    assert parse_run(f"seed7={features}") == (
        "seed7",
        features.resolve(),
        "matched",
        "permuted",
    )
    assert parse_run(f"seed7,m7,p7={features}") == (
        "seed7",
        features.resolve(),
        "m7",
        "p7",
    )


def test_patient_effects_average_repeated_images():
    matched = np.asarray([0.8, 0.6, 0.5])
    permuted = np.asarray([1.0, 1.0, 1.0])
    patients, effects = patient_effects(
        matched, permuted, ["patient-a", "patient-a", "patient-b"]
    )
    assert patients == ["patient-a", "patient-b"]
    np.testing.assert_allclose(effects, [-0.3, -0.5])


def test_crossed_bootstrap_detects_consistent_contraction():
    effects = np.asarray(
        [
            [-0.10, -0.08, -0.12, -0.09],
            [-0.05, -0.07, -0.06, -0.08],
            [-0.12, -0.11, -0.09, -0.10],
        ]
    )
    point, interval, samples = crossed_seed_patient_bootstrap(
        effects, draws=2_000, seed=7
    )
    assert point < 0
    assert interval[1] < 0
    assert samples.shape == (2_000,)
