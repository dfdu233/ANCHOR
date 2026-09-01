import pytest
import torch

from corrected_sgta.visual_token_ranges import (
    absolute_visual_indices,
    expanded_visual_range,
    remove_visual_embeddings,
)


def test_expanded_range_follows_prompt_specific_placeholder():
    short = torch.tensor([[1, -200, 2, 3]])
    long = torch.tensor([[1, 2, 3, 4, -200, 5]])
    assert expanded_visual_range(short, image_token_index=-200, num_image_tokens=4)["image_start"] == 1
    assert expanded_visual_range(long, image_token_index=-200, num_image_tokens=4)["image_start"] == 4


def test_remove_visual_embeddings_uses_dynamic_range():
    embeddings = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    kept = remove_visual_embeddings(embeddings, image_start=2, num_image_tokens=3)
    assert kept.flatten().tolist() == [0, 1, 5, 6, 7]


def test_absolute_visual_indices_fail_closed():
    assert absolute_visual_indices(torch.tensor([0, 2]), image_start=3, sequence_length=8).tolist() == [3, 5]
    with pytest.raises(ValueError, match="outside"):
        absolute_visual_indices(torch.tensor([5]), image_start=3, sequence_length=8)

