import copy

import pytest

from anchor.medeval.qualify_clinical_presupposition_generation import qualify


def _rows(items=2, lengths=(40, 41, 41)):
    rows = []
    for item in range(items):
        for condition, length in zip(
            ("neutral", "existential", "negative_obligation"), lengths
        ):
            rows.append(
                {
                    "item_id": f"image-{item}",
                    "prompt_condition": condition,
                    "text": "word " * min(length, 30),
                    "generated_token_count": length,
                    "generated_token_ids": list(range(length)),
                    "hit_max_new_tokens": False,
                    "surface_refusal_match": False,
                    "fingerprint": "frozen",
                    "claim_universe_sha256": (str(item) * 64)[:64],
                    "clinical_claim_evaluation_status": "pending_shared_audit",
                    "ground_truth_used_for_generation_or_selection": False,
                    "automatic_labeler_used": False,
                }
            )
    return rows


def test_qualification_authorizes_only_sufficient_strict_pairs():
    passed = qualify(_rows(), minimum_pairs=2)
    assert passed["passed"] is True
    assert passed["human_claim_audit_authorized"] is True
    failed = qualify(_rows(lengths=(40, 80, 80)), minimum_pairs=2)
    assert failed["passed"] is False
    assert failed["second_model_generation_authorized_from_this_model"] is False


def test_qualification_rejects_incomplete_or_truth_assigning_generation():
    rows = _rows()
    with pytest.raises(ValueError, match="incomplete prompt triplet"):
        qualify(rows[:-1], minimum_pairs=1)
    changed = copy.deepcopy(rows)
    changed[0]["clinical_claim_evaluation_status"] = "supported"
    with pytest.raises(ValueError, match="assigned clinical truth"):
        qualify(changed, minimum_pairs=1)


def test_qualification_rejects_cross_prompt_claim_universe_change():
    rows = _rows()
    rows[0]["claim_universe_sha256"] = "a" * 64
    rows[1]["claim_universe_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="claim universe differs"):
        qualify(rows, minimum_pairs=1)
