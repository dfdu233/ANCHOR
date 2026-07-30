import io

import numpy as np
from PIL import Image

from anchor.corrected_sgta.analyze_pubmed_style_prior import (
    clinical_labels,
    question_family,
    style_features,
)


def test_style_features_are_finite_and_fixed_size():
    array = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    result = style_features(buffer.getvalue())
    assert result.shape == (54,)
    assert np.isfinite(result).all()


def test_question_and_label_rules():
    assert question_family("What device is visible?") == "device"
    labels = clinical_labels(
        "There is a pleural effusion and pneumothorax without cardiomegaly."
    )
    # Regex labels describe concepts mentioned in the answer, not polarity.
    assert labels[0] == 1
    assert labels[1] == 1
    assert labels[3] == 1
