from __future__ import annotations

import pytest

from anchor.corrected_sgta.hulu_specificity_ratchet_adapter_v1 import (
    build_adapter,
    partition_hulu_assistant_target,
)
from anchor.corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    ContractError,
    RowExclusion,
)


def test_partition_excludes_native_im_end_and_newline_tokens():
    prompt = "<|im_start|>assistant\n"
    target = "A left lesion."
    rendered = prompt + target + "<|im_end|>\n"
    tokens = [1, 2, 3, 4, 5, 6, 7]
    offsets = [
        (0, len(prompt)),
        (len(prompt), len(prompt) + 1),
        (len(prompt) + 1, len(prompt) + 6),
        (len(prompt) + 6, len(prompt) + 13),
        (len(prompt) + 13, len(prompt) + len(target)),
        (len(prompt) + len(target), len(rendered) - 1),
        (len(rendered) - 1, len(rendered)),
    ]
    result = partition_hulu_assistant_target(
        rendered_assistant=rendered,
        generation_prompt=prompt,
        target=target,
        token_ids=tokens,
        offsets=offsets,
    )
    assert result["target_token_ids"] == [2, 3, 4, 5]
    assert result["assistant_token_indices"] == [1, 2, 3, 4]
    assert result["target_token_offsets"] == [(0, 1), (1, 6), (6, 13), (13, 14)]


def test_partition_allows_whitespace_spill_but_clips_to_raw_target():
    prompt = "P\n"
    target = "left lesion"
    rendered = prompt + target + " END"
    result = partition_hulu_assistant_target(
        rendered_assistant=rendered,
        generation_prompt=prompt,
        target=target,
        token_ids=[1, 2, 3, 4],
        offsets=[(0, 2), (1, 6), (6, 13), (13, 17)],
    )
    assert result["target_token_ids"] == [2, 3]
    assert result["target_token_offsets"] == [(0, 4), (4, 11)]


def test_partition_rejects_nonwhitespace_template_boundary_merge():
    prompt = "P:"
    target = "left"
    rendered = prompt + target + "<END>"
    with pytest.raises(RowExclusion, match="template boundary"):
        partition_hulu_assistant_target(
            rendered_assistant=rendered,
            generation_prompt=prompt,
            target=target,
            token_ids=[1, 2],
            offsets=[(0, 3), (3, len(rendered))],
        )


def test_partition_rejects_rendering_drift_and_incomplete_offsets():
    with pytest.raises(ContractError, match="raw target interval"):
        partition_hulu_assistant_target(
            rendered_assistant="P:right END",
            generation_prompt="P:",
            target="left",
            token_ids=[1, 2, 3],
            offsets=[(0, 2), (2, 7), (7, 11)],
        )


def test_current_hulu_factory_refuses_huatuo_sourced_scientific_use():
    with pytest.raises(ContractError, match="separate Hulu full-visible-answer substrate"):
        build_adapter({})
    with pytest.raises(ContractError, match="do not cover"):
        partition_hulu_assistant_target(
            rendered_assistant="P:left END",
            generation_prompt="P:",
            target="left",
            token_ids=[1, 2, 3],
            offsets=[(0, 2), (2, 4), (7, 10)],
        )
