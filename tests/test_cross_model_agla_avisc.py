import numpy as np
import pytest
import torch
from PIL import Image

from anchor.corrected_sgta.cross_model_agla import agla_logits, prompt_match_augment
from anchor.corrected_sgta.cross_model_avisc import avisc_logits, blind_region_image


def test_agla_equation_and_gate():
    original = torch.tensor([[4.0, 3.0, 0.0]])
    augmented = torch.tensor([[1.0, 5.0, 9.0]])
    out = agla_logits(original, augmented, alpha=2.0, beta=0.5)
    assert torch.isfinite(out[0, 0])
    assert torch.isneginf(out[0, 1]) and torch.isneginf(out[0, 2])
    assert out.argmax().item() == 0


def test_avisc_equation_and_validation():
    original = torch.tensor([[4.0, 3.0]])
    masked = torch.tensor([[3.0, 4.0]])
    out = avisc_logits(original, masked, alpha=2.5, beta=0.1)
    assert out.argmax().item() == 0
    with pytest.raises(ValueError):
        avisc_logits(original, masked[:, :1])


def test_image_space_policies_are_deterministic():
    image = Image.new("RGB", (8, 8), (100, 100, 100))
    image.putpixel((4, 4), (255, 255, 255))
    importance = np.zeros((2, 2), dtype=np.float32)
    importance[1, 1] = 1.0
    assert np.array_equal(np.asarray(prompt_match_augment(image, importance)), np.asarray(prompt_match_augment(image, importance)))
    out = np.asarray(blind_region_image(image, importance, lamb=0.0))
    assert out.shape == (8, 8, 3)
