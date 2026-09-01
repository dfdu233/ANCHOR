from anchor.corrected_sgta.build_cecd_admission_pack_v1 import stratified_review_rows
from anchor.corrected_sgta.run_cecd_factorial_v1 import FROZEN_FINDINGS, FROZEN_VOTES


def test_review_selection_is_balanced_deterministic_and_vote_blinded_later():
    rows = [
        {"image_id": f"{finding}-{vote}-{index}", "finding": finding, "positive_votes": vote}
        for finding in FROZEN_FINDINGS
        for vote in FROZEN_VOTES
        for index in range(10)
    ]
    first = stratified_review_rows(rows, seed=7)
    second = stratified_review_rows(list(reversed(rows)), seed=7)
    assert first == second
    assert len(first) == 60
    for finding in FROZEN_FINDINGS:
        assert sum(row["finding"] == finding for row in first) == 15
