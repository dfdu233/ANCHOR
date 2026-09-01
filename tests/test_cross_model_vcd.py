import torch

from anchor.corrected_sgta.cross_model_vcd import noise_visual_input


def test_anyres_visual_list_noise_is_deterministic_and_shape_preserving():
    inputs = [torch.zeros(1, 3, 4, 4), torch.ones(2, 3, 4, 4)]
    first = noise_visual_input(inputs, noise_step=500, generator=torch.Generator().manual_seed(7))
    second = noise_visual_input(inputs, noise_step=500, generator=torch.Generator().manual_seed(7))
    assert isinstance(first, list) and len(first) == 2
    assert [x.shape for x in first] == [x.shape for x in inputs]
    assert all(torch.equal(x, y) for x, y in zip(first, second))
    assert not torch.equal(first[0], inputs[0])
