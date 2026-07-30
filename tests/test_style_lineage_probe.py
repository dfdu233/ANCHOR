import numpy as np

from anchor.corrected_sgta.analyze_style_lineage_probe import dispersion
from anchor.corrected_sgta.analyze_style_lineage_report_probe import (
    cluster_effect,
    holm_adjust,
    positive_mention,
)
from anchor.corrected_sgta.analyze_prompt_style_factorial import (
    paired_prompt_range,
)


def test_dispersion_uses_cluster_affirmative_rate_range():
    rows = []
    for cluster, values in enumerate([["no", "no"], ["yes", "no"]]):
        for replicate, value in enumerate(values):
            rows.append(
                {
                    "id": f"{cluster}-{replicate}",
                    "disease": "effusion",
                    "cluster": cluster,
                    "explicit_prediction": value,
                }
            )
    value, details = dispersion(rows)
    assert value == 0.5
    assert details["effusion"]["range"] == 0.5


def test_report_probe_handles_negation_and_cluster_effect():
    assert positive_mention("No pleural effusion is present.", r"effusion") == 0
    assert positive_mention("Pleural effusion is absent.", r"effusion") == 0
    assert positive_mention("It does not show pleural effusion.", r"effusion") == 0
    assert positive_mention("Possible pleural effusion.", r"effusion") == 0
    assert positive_mention("A pleural effusion is present.", r"effusion") == 1
    labels = np.asarray([0, 0, 1, 1], dtype=float)
    clusters = np.asarray([0, 0, 1, 1])
    assert cluster_effect(labels, clusters) == 1.0
    assert holm_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}


def test_prompt_range_is_paired_by_prototype_and_disease():
    rows = []
    for frame, value in [("positive", "yes"), ("neutral", "no"), ("negative", "no")]:
        rows.append(
            {
                "prototype_id": "p0",
                "disease": "effusion",
                "prompt_frame": frame,
                "explicit_prediction": value,
                "text": (
                    "Pleural effusion is present."
                    if value == "yes"
                    else "No pleural effusion is present."
                ),
            }
        )
    value, details = paired_prompt_range(rows)
    assert value == 1.0
    assert details["p0::effusion"] == 1.0
