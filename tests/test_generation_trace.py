from anchor.corrected_sgta.generation_trace import classify_generated_tokens


def test_eos_trace_excludes_boundaries_but_retains_reason() -> None:
    trace = classify_generated_tokens(
        [1, 10, 11, 2, 0],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=8,
    )
    assert trace["generated_token_ids"] == [10, 11]
    assert trace["raw_generated_token_ids"] == [1, 10, 11, 2, 0]
    assert trace["terminal_token_ids"] == [2, 0]
    assert trace["stop_reason"] == "eos"


def test_budget_trace_is_not_misclassified_as_eos() -> None:
    trace = classify_generated_tokens(
        [10, 11, 12],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=3,
    )
    assert trace["generated_token_ids"] == [10, 11, 12]
    assert trace["terminal_token_ids"] == []
    assert trace["stop_reason"] == "max_new_tokens"


def test_short_unmarked_trace_fails_closed_as_unknown() -> None:
    trace = classify_generated_tokens(
        [10],
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        max_new_tokens=8,
    )
    assert trace["stop_reason"] == "unknown"
