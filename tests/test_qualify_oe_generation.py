from anchor.medeval.qualify_oe_generation import (
    generated_token_count,
    has_repetition_loop,
    qualify,
    terminal_required,
)


def test_terminal_policy_uses_explicit_prompt_wording_only():
    assert terminal_required("Describe the pathology", "explicit_sentence_instruction")
    assert terminal_required("Why is this abnormal?", "explicit_sentence_instruction")
    assert not terminal_required("What is the pathology?", "explicit_sentence_instruction")
    assert not terminal_required("Where is the lesion?", "explicit_sentence_instruction")


def test_generated_token_count_accepts_certified_native_and_port_schemas():
    assert generated_token_count({"metadata": {"generated_token_count": 7}}) == 7
    assert generated_token_count({"metadata": {"decoded_sequence_token_count": 8}}) == 8
    assert generated_token_count({"metadata": {"generated_token_ids": [1, 2, 3]}}) == 3
    assert generated_token_count({"metadata": {}}) is None


def _manifest(n=4):
    return [{"qid": f"q{i}"} for i in range(n)]


def _answers(values):
    return [{"qid": f"q{i}", "text": value} for i, value in enumerate(values)]


def _answers_with_counts(values, counts):
    return [
        {
            "qid": f"q{i}",
            "text": value,
            "metadata": {"generated_token_count": counts[i]},
        }
        for i, value in enumerate(values)
    ]


def test_qualify_accepts_aligned_noncollapsed_outputs():
    result = qualify(_manifest(), _answers(["left", "right", "upper", "lower"]), limit=4)
    assert result["passed"]


def test_qualify_rejects_reordered_qids():
    answers = _answers(["left", "right", "upper", "lower"])
    answers[0]["qid"], answers[1]["qid"] = answers[1]["qid"], answers[0]["qid"]
    result = qualify(_manifest(), answers, limit=4)
    assert not result["passed"]
    assert not result["exact_qid_alignment"]


def test_qualify_rejects_collapsed_or_sentinel_outputs():
    result = qualify(_manifest(), _answers(["skipped"] * 4), limit=4)
    assert not result["passed"]
    assert result["sentinel_count"] == 4


def test_qualify_rejects_diverse_function_word_fragments():
    values = ["The", "This", "In", "On"] * 5
    result = qualify(_manifest(20), _answers(values), limit=20)
    assert not result["passed"]
    assert result["unique_prediction_rate"] >= 0.10
    assert result["function_word_only_rate"] == 1.0


def test_qualify_treats_token_budget_exhaustion_as_diagnostic():
    answers = _answers_with_counts(
        ["Complete sentence.", "Another sentence.", "Third sentence.", "Fourth sentence."],
        [256, 256, 20, 30],
    )
    result = qualify(
        _manifest(),
        answers,
        limit=4,
        max_new_tokens=256,
        require_terminal_completeness=True,
    )
    assert result["passed"]
    assert result["cap_hit_rate"] == 0.5
    assert result["cap_hit_is_diagnostic_only"]
    assert result["artifact_status"] == "admissible"


def test_qualify_rejects_obvious_repetition_loops():
    loop = "no focal opacity in either lung " * 3
    assert has_repetition_loop(loop)
    answers = _answers([loop, "left base", "right apex", "normal heart"])
    result = qualify(
        _manifest(), answers, limit=4, max_repetition_loop_rate=0.0
    )
    assert not result["passed"]
    assert result["repetition_loop_count"] == 1


def test_repeated_medical_words_without_span_loop_are_allowed():
    text = (
        "No pleural effusion. The pleural surfaces are smooth. "
        "No focal opacity and no pneumothorax are present."
    )
    assert not has_repetition_loop(text)


def test_qualify_accepts_complete_uncapped_sentences():
    answers = _answers_with_counts(
        ["Complete sentence.", "Another sentence.", "Third sentence.", "Fourth sentence."],
        [20, 21, 22, 23],
    )
    result = qualify(
        _manifest(),
        answers,
        limit=4,
        max_new_tokens=256,
        require_terminal_completeness=True,
    )
    assert result["passed"]
    assert result["artifact_status"] == "admissible"
