import numpy as np
import pytest
import torch
from PIL import Image

from anchor.corrected_sgta.cross_model_cve import cve_logits, rebalance_image


def test_cve_equation_seven() -> None:
    clean = torch.tensor([[1.0, 2.0, 3.0]])
    enhanced = torch.tensor([[0.5, 1.0, 1.5]])
    actual = cve_logits(clean, enhanced, alpha=2.0, beta=0.2)
    assert torch.allclose(actual, torch.tensor([[1.6, 3.8, 6.0]]))


def test_cve_zero_coefficients_are_identity() -> None:
    clean = torch.randn(2, 7)
    assert torch.equal(cve_logits(clean, torch.randn_like(clean), alpha=0, beta=0), clean)


def test_cve_rebalance_preserves_image_contract() -> None:
    image = Image.new("RGB", (32, 24), (100, 100, 100))
    importance = np.arange(12, dtype=np.float32).reshape(3, 4) / 11
    output = rebalance_image(image, importance)
    assert output.mode == "RGB"
    assert output.size == image.size


def test_cve_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError):
        cve_logits(torch.zeros(1, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        rebalance_image(Image.new("RGB", (4, 4)), np.zeros((2, 2)), mid_quantile=.9, high_quantile=.5)
