from anchor.corrected_sgta.run_huatuo_information_dependence import (
    choose_sentence_pair,
    contains_finding,
)


ALIASES = ("pleural effusion", "cardiomegaly", "edema")


def test_choose_within_report_unavailable_and_visible_pair() -> None:
    report = (
        "Compared with the prior study, pulmonary edema has improved. "
        "There is a small pleural effusion. Recommend clinical follow-up."
    )
    pair = choose_sentence_pair(report, ALIASES)
    assert pair is not None
    unavailable, visible, labels = pair
    assert "prior" in unavailable
    assert visible == "There is a small pleural effusion"
    assert labels == ["prior_image"]


def test_alias_matching_uses_word_boundaries() -> None:
    assert contains_finding("There is cardiomegaly.", ALIASES)
    assert not contains_finding("The cardiomegalys token is malformed.", ALIASES)
