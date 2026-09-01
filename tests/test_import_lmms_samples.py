import pytest

from anchor.medeval.import_lmms_samples import convert_samples


def test_convert_lmms_sample_to_unified_answer():
    samples = [{
        "doc": {
            "qid": "q1",
            "question": "Where?",
            "answer": "right",
            "image_sha256": "abc",
        },
        "filtered_resps": [["right lung"]],
    }]
    rows = convert_samples(samples)
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["text"] == "right lung"
    assert rows[0]["image_sha256"] == "abc"


def test_convert_rejects_duplicate_qids():
    sample = {"doc": {"qid": "q1"}, "filtered_resps": [["x"]]}
    with pytest.raises(ValueError, match="duplicate qid"):
        convert_samples([sample, sample])


def test_convert_rejects_ambiguous_prediction_shape():
    sample = {"doc": {"qid": "q1"}, "filtered_resps": [["x", "y"]]}
    with pytest.raises(ValueError, match="one string prediction"):
        convert_samples([sample])
