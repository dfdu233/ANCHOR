from anchor.corrected_sgta.prepare_alignment_contraction_control import (
    image_permuted_rows,
    select_strict_cxr,
)


def records():
    return [
        {
            "id": f"id-{index}",
            "group_id": f"group-{index}",
            "is_strict_cxr": index != 7,
            "source_parquet": "/tmp/source.parquet",
            "parquet_row_index": index,
            "image_sha256": f"sha-{index}",
            "dhash64": f"{index:016x}",
        }
        for index in range(8)
    ]


def test_selection_is_deterministic_and_strict():
    first = select_strict_cxr(records(), 6, 42)
    second = select_strict_cxr(records(), 6, 42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert all(row["is_strict_cxr"] for row in first)


def test_image_permutation_preserves_marginal_and_deranges_groups():
    selected = select_strict_cxr(records(), 6, 42)
    permuted, permutation = image_permuted_rows(selected)
    assert sorted(permutation) == list(range(len(selected)))
    assert sorted(row["image_sha256"] for row in selected) == sorted(
        row["image_sha256"] for row in permuted
    )
    assert all(
        target["id"] != donor["image_record_id"]
        for target, donor in zip(selected, permuted, strict=True)
    )
    assert all(
        target["group_id"] != donor["image_group_id"]
        for target, donor in zip(selected, permuted, strict=True)
    )


def test_derangement_seed_changes_pairing_without_changing_sources():
    selected = select_strict_cxr(
        [
            {
                "id": f"id-{index}",
                "group_id": f"group-{index // 2}",
                "is_strict_cxr": True,
                "source_parquet": "/tmp/source.parquet",
                "parquet_row_index": index,
                "image_sha256": f"sha-{index}",
                "dhash64": f"{index:016x}",
            }
            for index in range(12)
        ],
        12,
        42,
    )
    first, first_permutation = image_permuted_rows(selected, seed=7)
    second, second_permutation = image_permuted_rows(selected, seed=19)
    assert first_permutation != second_permutation
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert all(
        target["id"] != donor["image_record_id"]
        for target, donor in zip(selected, first, strict=True)
    )
    assert all(
        target["id"] != donor["image_record_id"]
        for target, donor in zip(selected, second, strict=True)
    )
