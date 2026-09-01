from anchor.medeval.audit_hulu_context import summarize


def test_context_audit_uses_model_window_not_tokenizer_warning():
    rows = [
        {
            "qid": "a",
            "image": "a.jpg",
            "baseline": {"input_tokens": 16326, "image_tokens": 16240},
            "candidate": {"input_tokens": 16775, "image_tokens": 16240},
        }
    ]
    result = summarize(
        rows,
        max_position_embeddings=262144,
        tokenizer_model_max_length=16384,
        max_new_tokens=128,
    )
    assert result["passed"] is True
    assert result["tokenizer_metadata_warning_count"] == 1
    assert result["model_context_overflow_count"] == 0


def test_context_audit_rejects_true_model_overflow():
    rows = [
        {
            "qid": "a",
            "image": "a.jpg",
            "baseline": {"input_tokens": 1900, "image_tokens": 100},
            "candidate": {"input_tokens": 1950, "image_tokens": 100},
        }
    ]
    result = summarize(
        rows,
        max_position_embeddings=2048,
        tokenizer_model_max_length=2048,
        max_new_tokens=128,
    )
    assert result["passed"] is False
    assert result["model_context_overflow_count"] == 1
