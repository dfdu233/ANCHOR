from __future__ import annotations

from anchor.medeval.prepare_physician_oe_review import make_bundle


def test_physician_bundle_is_image_disjoint_and_model_blinded() -> None:
    manifest = [
        {
            "qid": f"q{i}",
            "question": f"what is finding {i}?",
            "answer": "right" if i % 2 else "left lower lobe",
            "img_name": f"image-{i}.jpg",
            "image_sha256": f"sha-{i}",
        }
        for i in range(6)
    ]
    answer_sets = {
        model: [
            {"question_id": row["qid"], "text": f"{model} answer {row['qid']}"}
            for row in manifest
        ]
        for model in ("a", "b", "c")
    }
    bundle, mapping, metadata = make_bundle(
        manifest, answer_sets, n_qids=4, seed=7
    )
    assert len(bundle) == 4
    assert len(mapping) == 12
    assert len({row["image"]["sha256"] for row in bundle}) == 4
    assert metadata["n_answer_units"] == 12
    assert all("source_model" not in candidate for row in bundle for candidate in row["candidate_answers"])
    assert {row["source_model"] for row in mapping} == {"a", "b", "c"}
    assert all(len(row["candidate_answers"]) == 3 for row in bundle)
    assert all(
        candidate["annotation"]["no_clinical_claims"] is None
        for row in bundle
        for candidate in row["candidate_answers"]
    )
    assert metadata["protocol_id"] == "anchor-physician-oe-review-v2"
    assert metadata["review_contract"]["atomic_claim_schema"]["normalized_claim"]["polarity"] == [
        "present",
        "absent",
    ]


def test_exact_duplicate_answers_can_be_reviewed_once_without_losing_method_mapping() -> None:
    manifest = [
        {
            "qid": "q1",
            "question": "what is shown?",
            "answer": "opacity",
            "img_name": "image.jpg",
            "image_sha256": "sha",
        }
    ]
    answer_sets = {
        "greedy": [{"question_id": "q1", "text": "An opacity."}],
        "method_off": [{"question_id": "q1", "text": "An opacity."}],
        "method_on": [{"question_id": "q1", "text": "No opacity."}],
    }
    bundle, mapping, metadata = make_bundle(
        manifest,
        answer_sets,
        n_qids=1,
        seed=7,
        deduplicate_exact_answers=True,
    )
    assert len(bundle[0]["candidate_answers"]) == 2
    assert len(mapping) == 3
    assert metadata["n_answer_units"] == 2
    assert metadata["n_model_assignments"] == 3
    assert metadata["exact_duplicate_model_assignments_collapsed"] == 1
    duplicate_ids = [
        row["answer_id"]
        for row in mapping
        if row["source_model"] in {"greedy", "method_off"}
    ]
    assert len(set(duplicate_ids)) == 1
