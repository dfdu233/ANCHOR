from anchor.medeval.compare_ce_arms import compare
from anchor.medeval.qualify_ce_generation import qualify


def answer(qid, text, tokens=3):
    return {"question_id": qid, "text": text, "metadata": {"generated_token_count": tokens}}


def test_ce_qualification_and_paired_cutoff():
    manifest = [{"qid":"a", "img_name":"x", "answer":"Yes."}, {"qid":"b", "img_name":"y", "answer":"No."}]
    baseline = [answer("a", "No, absent."), answer("b", "No, absent.")]
    candidate = [answer("a", "Yes, present."), answer("b", "No, absent.")]
    assert qualify(manifest, candidate, 64)["passed"] is True
    result = compare(manifest, baseline, candidate, draws=100, seed=1)
    assert result["accuracy_delta"] == .5
    assert result["candidate_accuracy"] == 1.0


def test_ce_qualification_parses_choice_and_label_only_options():
    manifest = [
        {
            "qid": "a", "answer": "B", "source_question_type": "multi-choice",
            "question_type": "choice", "choices": "A, B, C, D",
        },
        {
            "qid": "b", "answer": "Clear", "source_question_type": "multi-choice",
            "question_type": "choice", "choices": "A. Congested, B. Clear, C. Inflamed",
        },
    ]
    candidate = [answer("a", "B"), answer("b", "The answer is B.")]
    result = qualify(manifest, candidate, 64)
    assert result["passed"] is True
    assert result["invalid_ground_truth_count"] == 0
    assert result["strict_parse_rate"] == 1.0


def test_ce_qualification_rejects_unjudgeable_outputs_not_low_accuracy():
    manifest = [
        {"qid": "a", "answer": "Yes", "source_question_type": "binary"},
        {"qid": "b", "answer": "No", "source_question_type": "binary"},
    ]
    wrong_but_parseable = [answer("a", "No"), answer("b", "Yes")]
    assert qualify(manifest, wrong_but_parseable, 64)["passed"] is True
    unparseable = [answer("a", "There is edema"), answer("b", "No")]
    assert qualify(
        manifest, unparseable, 64, minimum_parse_rate=1.0
    )["passed"] is False
