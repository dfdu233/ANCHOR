import json
from pathlib import Path

import numpy as np

from anchor.corrected_sgta.analyze_layerwise_attractor import (
    analyze_files,
    centroid_distance_controls,
)


def test_layerwise_analysis_detects_late_null_alignment(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    rng = np.random.default_rng(7)
    arrays = {}
    for index in (0, 7, 14, 21, 27):
        for token_type in ("image", "prompt"):
            features = rng.normal(size=(1, 6, 5, 8))
            features[:, :, 1] = 0.0
            strength = 0.05 if index == 0 else 0.6
            if token_type == "image":
                strength *= 0.5
            real = features[:, :, [0]]
            features[:, :, 2:] = (1 - strength) * real
            arrays[f"llm_{index}_{token_type}"] = features.astype(np.float32)
    np.savez_compressed(path, **arrays)
    metadata = {
        "fingerprint": "test",
        "variants": [{"name": "base"}],
        "cases": 6,
        "rows": [
            {
                "case_id": f"case-{index}",
                "patient_id": f"patient-{index}",
            }
            for index in range(6)
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    result = analyze_files([path], permutation_repeats=8)
    assert result["decision"]["final_prompt_exceeds_initial_prompt_every_lineage"]
    assert result["decision"][
        "prompt_exceeds_image_at_every_sampled_layer_and_lineage"
    ]


def test_centroid_distance_distinguishes_alignment_from_contraction() -> None:
    real = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    features = np.zeros((3, 3, 2), dtype=np.float32)
    features[:, 0] = real
    features[:, 1] = 0.0
    # Case 0 moves partly toward its LOO centroid (3, 0), but a large
    # orthogonal component makes it farther overall.
    features[0, 2] = [1.0, 4.0]
    # Cases 1 and 2 move directly toward their respective LOO centroids.
    features[1, 2] = [1.5, 0.0]
    features[2, 2] = [2.5, 0.0]
    metrics = centroid_distance_controls(
        features, ["patient-0", "patient-1", "patient-2"]
    )
    assert metrics["mean_centroid_projection_coefficient"] > 0
    assert 0 < metrics["fraction_closer_to_centroid"] < 1
