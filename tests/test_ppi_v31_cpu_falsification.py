from __future__ import annotations

import random

from corrected_sgta.ppi_v31_cpu_falsification import (
    ImageRecord,
    assignment_deltas,
    balanced_fingerprints,
    detailed_audit,
    dot,
    feature_rows,
    fingerprint_id,
    optimize_assignment,
)


def synthetic_records(n: int = 400) -> list[ImageRecord]:
    rng = random.Random(7)
    records = []
    for index in range(n):
        bins = tuple((index // (claim + 1) + claim + rng.randrange(2)) % 4 for claim in range(8))
        records.append(
            ImageRecord(
                image_id=f"image-{index:04d}",
                bins=bins,
                majority=tuple(int(value >= 2) for value in bins),
                no_finding=int(index < n // 4),
                reader_positive_counts=tuple(sum(value > reader for value in bins) for reader in range(3)),
            )
        )
    return records


def test_complete_balanced_sign_space_and_orthogonality() -> None:
    fingerprints = balanced_fingerprints(8)
    assert len(fingerprints) == 70
    assert len({fingerprint_id(value) for value in fingerprints}) == 70
    assert all(sum(value) == 0 for value in fingerprints)
    assert any(dot(left, right) == 0 for left in fingerprints for right in fingerprints)


def test_assignment_optimizer_is_image_level_balanced_and_complementable() -> None:
    records = synthetic_records()
    target = (1, 1, 1, 1, -1, -1, -1, -1)
    assignment, audit = optimize_assignment(
        records,
        target,
        seed=11,
        target_g=0.02,
        restarts=2,
        steps=20_000,
    )
    assert len(assignment) == len({record.image_id for record in records})
    assert assignment.count(1) == assignment.count(-1) == len(records) // 2
    assert audit["no_finding_a"] == audit["no_finding_b"]
    features, _, _ = feature_rows(records)
    plus = assignment_deltas(assignment, features)
    minus = assignment_deltas([-value for value in assignment], features)
    assert minus == [-value for value in plus]


def test_detailed_audit_preserves_counts() -> None:
    records = synthetic_records()
    assignment = [1 if index % 2 == 0 else -1 for index in range(len(records))]
    claims = [f"claim-{index}" for index in range(8)]
    audit = detailed_audit(records, assignment, claims)
    assert set(audit["per_claim"]) == set(claims)
    for group in ("A", "B"):
        assert sum(audit["aggregate_reader_bins"][group].values()) == len(records) // 2 * 8
