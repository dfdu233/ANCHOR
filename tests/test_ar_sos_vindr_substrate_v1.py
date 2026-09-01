from __future__ import annotations

from anchor.corrected_sgta.audit_ar_sos_vindr_substrate_v1 import (
    association_rows,
    confirmation_availability,
    fisher_greater,
    freeze_ab_pairs_on_dev,
    vectorize_rows,
)


FINDINGS = ("a", "b", "c", "d")


def _vectors(dev_repeats: int = 8, confirmation_repeats: int = 45):
    vectors = {}
    splits = {}
    # A and C have a strong dev association, while target cases have A/B=3
    # and C=0.  D provides an unused unanimous-negative A' substrate.
    for index in range(20):
        image = f"dev-joint-{index}"
        vectors[image] = {"a": 3, "b": 0, "c": 3, "d": 0}
        splits[image] = "dev"
    for index in range(80):
        image = f"dev-null-{index}"
        vectors[image] = {"a": 0, "b": 0, "c": 0, "d": 0}
        splits[image] = "dev"
    for index in range(dev_repeats):
        image = f"dev-target-{index}"
        vectors[image] = {"a": 3, "b": 3, "c": 0, "d": 0}
        splits[image] = "dev"
    for index in range(confirmation_repeats):
        image = f"confirmation-target-{index}"
        vectors[image] = {"a": 3, "b": 3, "c": 0, "d": 0}
        splits[image] = "confirmation"
    return vectors, splits


def test_fisher_greater_detects_positive_association():
    assert fisher_greater(20, 12, 0, 80) < 0.001
    assert fisher_greater(0, 80, 20, 12) > 0.9


def test_confirmation_cannot_change_dev_frozen_identity():
    vectors, splits = _vectors()
    associations = association_rows(vectors, splits, findings=FINDINGS)
    frozen = freeze_ab_pairs_on_dev(
        vectors,
        splits,
        associations,
        required_pairs=1,
        min_dev_targets=8,
        findings=FINDINGS,
    )
    assert [(row["finding_a"], row["finding_b"], row["finding_c"]) for row in frozen] == [
        ("a", "b", "c")
    ]
    before = confirmation_availability(frozen, vectors, splits)[0]
    for image in [key for key in vectors if splits[key] == "confirmation"]:
        vectors[image]["b"] = 0
    after = confirmation_availability(frozen, vectors, splits)[0]
    assert before["confirmation_target_count"] == 45
    assert after["confirmation_target_count"] == 0
    assert (before["finding_a"], before["finding_b"], before["finding_c"]) == (
        after["finding_a"], after["finding_b"], after["finding_c"]
    )


def test_vectorizer_rejects_incomplete_truth_vectors():
    rows = [
        {
            "image_id": "x",
            "finding": finding,
            "positive_votes": 3,
            "reader_panel": ["R8", "R9", "R10"],
            "experiment_split": "dev",
        }
        for finding in FINDINGS[:-1]
    ]
    try:
        vectorize_rows(rows, FINDINGS)
    except RuntimeError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete vector was admitted")


def test_association_fit_reads_dev_only():
    vectors, splits = _vectors()
    first = association_rows(vectors, splits, findings=FINDINGS)
    for image in [key for key in vectors if splits[key] == "confirmation"]:
        vectors[image] = {"a": 0, "b": 0, "c": 3, "d": 3}
    second = association_rows(vectors, splits, findings=FINDINGS)
    assert first == second
