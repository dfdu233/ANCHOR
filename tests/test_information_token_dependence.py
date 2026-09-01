import numpy as np

from anchor.corrected_sgta.run_huatuo_token_dependence import (
    finding_spans,
    source_spans,
    token_masks,
)


def test_source_and_finding_spans_are_separate() -> None:
    sentence = "Interval decrease in pulmonary edema compared with the prior study."
    aliases = ("pulmonary edema", "edema")
    source = [sentence[a:b] for a, b in source_spans(sentence)]
    finding = [sentence[a:b] for a, b in finding_spans(sentence, aliases)]
    assert "Interval decrease" in source
    assert "compared with the prior study" in source
    assert finding[0] == "pulmonary edema"


def test_token_masks_follow_character_offsets_without_overlap() -> None:
    sentence = "Edema is again noted."
    offsets = [(0, 5), (5, 8), (8, 14), (14, 20), (20, 21)]
    source, finding = token_masks(sentence, offsets, ("edema",))
    assert np.flatnonzero(source).tolist() == [2, 3]
    assert np.flatnonzero(finding).tolist() == [0]
    assert not np.any(source & finding)
