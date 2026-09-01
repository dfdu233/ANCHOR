import numpy as np
import pytest

from anchor.medeval.prepare_agla_augmented import agla_binary_mask, question_text


def test_agla_mask_ratio_and_order():
    attention = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask, ratio = agla_binary_mask(attention, similarity=1.0)
    assert ratio == pytest.approx(0.5)
    assert mask[9, 9] == 1 and mask[0, 0] == 0
    assert 0.49 <= mask.mean() <= 0.51


def test_agla_question_normalization():
    assert question_text("What is this? (Image #1)") == "what is this? image 1"
