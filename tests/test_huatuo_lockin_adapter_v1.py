from __future__ import annotations

import pytest

from corrected_sgta.clinical_autoregressive_lockin_probe_v1 import RowExclusion
from corrected_sgta.huatuo_lockin_adapter_v1 import partition_answer_tokens


def test_partition_clips_leading_space_token_into_continuation() -> None:
    result = partition_answer_tokens(
        answer_text="The chest shows opacities \n",
        prefix="The chest shows ",
        continuation="opacities",
        token_ids=[1, 2, 3, 4, 5],
        offsets=[(0, 3), (3, 9), (9, 15), (15, 18), (18, 26)],
    )
    assert result["prefix_token_ids"] == [1, 2, 3]
    assert result["continuation_token_ids"] == [4, 5]
    assert result["continuation_token_offsets"] == [(0, 2), (2, 9)]
    assert result["continuation_sequence_indices"] == [3, 4]


def test_partition_preserves_standalone_prefix_whitespace_token() -> None:
    result = partition_answer_tokens(
        answer_text="A x \n",
        prefix="A ",
        continuation="x",
        token_ids=[1, 2, 3, 4],
        offsets=[(0, 1), (1, 2), (2, 3), (3, 5)],
    )
    assert result["prefix_token_ids"] == [1, 2]
    assert result["prefix_token_offsets"] == [(0, 1), (1, 2)]
    assert result["continuation_token_ids"] == [3]


def test_partition_rejects_non_whitespace_cross_boundary() -> None:
    with pytest.raises(RowExclusion, match="crosses"):
        partition_answer_tokens(
            answer_text="abcd \n",
            prefix="ab",
            continuation="cd",
            token_ids=[1, 2],
            offsets=[(0, 4), (4, 6)],
        )
