import numpy as np

from anchor.corrected_sgta.analyze_style_lineage_probe import dispersion
from anchor.corrected_sgta.analyze_style_lineage_report_probe import (
    cluster_effect,
    positive_mention,
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
    assert positive_mention("A pleural effusion is present.", r"effusion") == 1
    labels = np.asarray([0, 0, 1, 1], dtype=float)
    clusters = np.asarray([0, 0, 1, 1])
    assert cluster_effect(labels, clusters) == 1.0
