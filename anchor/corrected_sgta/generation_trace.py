"""Backend-neutral classification of generated-token termination."""

from __future__ import annotations

from typing import Iterable


def classify_generated_tokens(
    token_ids: Iterable[int],
    *,
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_new_tokens: int | None,
) -> dict:
    """Normalize a generation-only sequence and retain why it stopped.

    The caller must remove any prompt prefix first. Boundary BOS/EOS/PAD tokens
    are excluded from decoded clinical content but retained in trace metadata.
    """

    raw = [int(value) for value in token_ids]
    content = list(raw)
    leading_bos_removed = bool(
        content and bos_token_id is not None and content[0] == bos_token_id
    )
    if leading_bos_removed:
        content = content[1:]

    terminal_ids: list[int] = []
    terminal_set = {
        int(value) for value in (eos_token_id, pad_token_id) if value is not None
    }
    while content and content[-1] in terminal_set:
        terminal_ids.append(content.pop())
    terminal_ids.reverse()

    if eos_token_id is not None and int(eos_token_id) in terminal_ids:
        stop_reason = "eos"
    elif terminal_ids:
        stop_reason = "pad"
    elif max_new_tokens is not None and len(content) >= int(max_new_tokens):
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "unknown"

    return {
        "generated_token_ids": content,
        "raw_generated_token_ids": raw,
        "raw_generated_token_count": len(raw),
        "terminal_token_ids": terminal_ids,
        "leading_bos_removed": leading_bos_removed,
        "stop_reason": stop_reason,
    }
