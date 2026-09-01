from __future__ import annotations

import torch

from anchor.corrected_sgta.run_vqa_rad_visual_edge_constraint import (
    LastQueryImageEdgeSession,
)


def test_last_query_image_edge_session_only_blocks_final_prompt_row() -> None:
    weights = torch.softmax(torch.randn(1, 2, 5, 5), dim=-1)
    original = weights.clone()
    session = LastQueryImageEdgeSession(prefix_length=5, image_span=(1, 3))

    patched = session.apply_chunk(weights, query_start=0, total_query_length=5)

    assert torch.equal(patched[..., :4, :], original[..., :4, :])
    assert torch.count_nonzero(patched[..., 4, 1:3]) == 0
    assert torch.allclose(
        patched[..., 4, :].sum(dim=-1),
        torch.ones_like(patched[..., 4, :].sum(dim=-1)),
        atol=1e-6,
    )
    assert session.patched_rows == 2
    assert max(session.source_mass_after) == 0.0


def test_suffix_scope_blocks_all_text_queries_after_image() -> None:
    weights = torch.softmax(torch.randn(1, 2, 6, 6), dim=-1)
    original = weights.clone()
    session = LastQueryImageEdgeSession(
        prefix_length=6,
        image_span=(1, 3),
        query_scope="suffix_after_image",
    )

    patched = session.apply_chunk(weights, query_start=0, total_query_length=6)

    assert torch.equal(patched[..., :3, :], original[..., :3, :])
    assert torch.count_nonzero(patched[..., 3:, 1:3]) == 0
    assert torch.allclose(
        patched[..., 3:, :].sum(dim=-1),
        torch.ones_like(patched[..., 3:, :].sum(dim=-1)),
        atol=1e-6,
    )
    assert session.patched_rows == 6
