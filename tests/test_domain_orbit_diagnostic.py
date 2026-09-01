import torch

from corrected_sgta.domain_orbit_diagnostic import (
    canonicalize,
    degeneration_ratio,
    fit_feature_basis,
    heldout_attenuation,
    random_basis,
    token_stability_gate,
)
from corrected_sgta.run_huatuo_domain_orbit_diagnostic_v1 import bbox_instability_summary


def test_full_basis_reduces_to_orbit_mean():
    orbit = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ]
    )
    center, basis, _ = fit_feature_basis(orbit, rank=2)
    candidate = canonicalize(orbit[0], center, basis)
    assert torch.allclose(candidate, center, atol=1e-5)
    assert degeneration_ratio(candidate, orbit[0], center) < 1e-5


def test_known_direction_attenuates_heldout_orbit():
    content = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    direction = torch.tensor([0.0, 1.0, 0.0])
    orbit = torch.stack([content - direction, content, content + direction])
    center, basis, explained = fit_feature_basis(orbit, rank=1)
    attenuation = heldout_attenuation(content, (content + 2.0 * direction).unsqueeze(0), basis)
    assert torch.allclose(center, content)
    assert explained[0] > 0.99
    assert attenuation > 0.99


def test_random_basis_is_orthonormal():
    basis = random_basis(8, 3, seed=7, device=torch.device("cpu"))
    assert torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-5)


def test_bbox_instability_uses_padded_image_coordinates():
    values = torch.ones(16)
    # A tall 100x50 image is horizontally padded to 100x100. This box covers
    # one upper-left original-image patch center.
    values[1] = 5.0
    result = bbox_instability_summary(
        values,
        [{"x_min": 0, "x_max": 25, "y_min": 0, "y_max": 25}],
        image_height=100,
        image_width=50,
    )
    assert result is not None
    assert result["inside_outside_ratio"] > 1.0


def test_token_stability_gate_selects_requested_tokens_and_restores_norm():
    original = torch.ones(4, 3)
    instability = torch.tensor([0.1, 4.0, 2.0, 0.2])
    candidate, mask = token_stability_gate(
        original, instability, fraction=0.5, alpha=1.0, mode="unstable"
    )
    assert mask.tolist() == [False, True, True, False]
    assert torch.allclose(candidate.norm(), original.norm())
    assert torch.all(candidate[mask] == 0)
