from anchor.medeval.prepare_vqa_rad_oe import is_open_answer, normalized_answer


def test_binary_answers_are_excluded_after_punctuation_normalization() -> None:
    assert not is_open_answer("Yes.")
    assert not is_open_answer(" NO ")


def test_short_clinical_answers_remain_open() -> None:
    assert is_open_answer("left lower lobe")
    assert is_open_answer("2")
    assert normalized_answer("Left-lower lobe") == "left lower lobe"
