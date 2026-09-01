"""Dynamic visual-token range utilities for LLaVA decoding baselines."""

from __future__ import annotations

import torch


def expanded_visual_range(input_ids, *, image_token_index: int, num_image_tokens: int) -> dict[str, int]:
    if not isinstance(num_image_tokens, int) or num_image_tokens <= 0:
        raise ValueError("num_image_tokens must be positive")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("one unpadded input sequence is required")
    positions = input_ids[0].eq(image_token_index).nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError(f"expected one image placeholder, received {positions.numel()}")
    image_start = int(positions.item())
    image_end = image_start + num_image_tokens - 1
    response_start = int(input_ids.shape[1]) + num_image_tokens - 1
    if not (0 <= image_start <= image_end < response_start):
        raise ValueError("invalid expanded visual-token range")
    return {
        "image_start": image_start,
        "image_end": image_end,
        "response_start": response_start,
        "num_image_tokens": num_image_tokens,
    }


def absolute_visual_indices(relative_indices, *, image_start: int, sequence_length: int):
    if not isinstance(image_start, int) or image_start < 0:
        raise ValueError("image_start must be a non-negative integer")
    absolute = relative_indices.to(dtype=torch.long) + image_start
    if absolute.numel() and (
        int(absolute.min().item()) < 0 or int(absolute.max().item()) >= sequence_length
    ):
        raise ValueError("visual mask falls outside the expanded input sequence")
    return absolute


def remove_visual_embeddings(inputs_embeds, *, image_start: int, num_image_tokens: int):
    if (
        not isinstance(image_start, int)
        or not isinstance(num_image_tokens, int)
        or image_start < 0
        or num_image_tokens <= 0
        or image_start + num_image_tokens > inputs_embeds.shape[1]
    ):
        raise ValueError("invalid visual-token range for embeddings")
    return torch.cat(
        [
            inputs_embeds[:, :image_start, :],
            inputs_embeds[:, image_start + num_image_tokens:, :],
        ],
        dim=1,
    )

