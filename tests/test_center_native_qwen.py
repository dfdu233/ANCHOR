from __future__ import annotations

import torch

from anchor.corrected_sgta.center_native_qwen import (
    FeatureCenter,
    StreamingAmplitudeCenter,
    calibrate_patch_tokens,
)
from anchor.corrected_sgta.run_center_native_qwen import early_vision_parameters


def _center(tokens: torch.Tensor, grid=(1, 4, 4), kind="log") -> FeatureCenter:
    builder = StreamingAmplitudeCenter(kind=kind, grid_thw=grid, channels=tokens.shape[-1])
    builder.update(tokens)
    return builder.finalize()


def test_tau_zero_is_exact_identity() -> None:
    tokens = torch.randn(16, 8)
    center = _center(tokens)
    output = calibrate_patch_tokens(tokens, torch.tensor([[1, 4, 4]]), center, tau=0.0)
    assert output.data_ptr() == tokens.data_ptr()
    assert torch.equal(output, tokens)


def test_identical_log_center_is_identity_up_to_fft_error() -> None:
    tokens = torch.randn(16, 8)
    center = _center(tokens, kind="log")
    output = calibrate_patch_tokens(tokens, torch.tensor([[1, 4, 4]]), center, tau=0.5)
    assert torch.allclose(output, tokens, atol=2e-5, rtol=2e-5)


def test_batch_images_are_calibrated_independently() -> None:
    first = torch.randn(16, 6)
    second = torch.randn(16, 6)
    center = _center(first)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    joint = calibrate_patch_tokens(
        torch.cat([first, second]),
        grid,
        center,
        tau=0.5,
        apply_mask=torch.tensor([False, True]),
    )
    single = calibrate_patch_tokens(second, grid[:1], center, tau=0.5)
    assert torch.equal(joint[:16], first)
    assert torch.allclose(joint[16:], single, atol=1e-6, rtol=1e-6)


def test_spatial_phase_is_preserved() -> None:
    tokens = torch.randn(16, 5)
    source = torch.randn(16, 5)
    center = _center(source)
    output = calibrate_patch_tokens(tokens, torch.tensor([[1, 4, 4]]), center, tau=0.4)
    before = torch.fft.fft2(tokens.reshape(1, 4, 4, 5).permute(0, 3, 1, 2))
    after = torch.fft.fft2(output.reshape(1, 4, 4, 5).permute(0, 3, 1, 2))
    mask = (before.abs() > 1e-4) & (after.abs() > 1e-4)
    phase_delta = torch.angle(after[mask] * before[mask].conj())
    assert phase_delta.abs().max().item() < 2e-4


def test_grid_mismatch_is_rejected() -> None:
    tokens = torch.randn(16, 4)
    center = _center(tokens)
    wrong = torch.randn(24, 4)
    try:
        calibrate_patch_tokens(wrong, torch.tensor([[1, 4, 6]]), center, tau=0.5)
    except ValueError as exc:
        assert "does not match center grid" in str(exc)
    else:
        raise AssertionError("grid mismatch must raise")


def test_early_vision_parameter_filter_is_bounded() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.patch_embed = torch.nn.Linear(2, 2)
            self.visual.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(3)]
            )
            self.language_model = torch.nn.Linear(2, 2)

    names = {
        name for name, _ in early_vision_parameters(Toy(), blocks=2)
    }
    assert names
    assert all(
        "visual.patch_embed." in name
        or "visual.blocks.0." in name
        or "visual.blocks.1." in name
        for name in names
    )
    assert not any("visual.blocks.2." in name for name in names)
    assert not any("language_model" in name for name in names)
