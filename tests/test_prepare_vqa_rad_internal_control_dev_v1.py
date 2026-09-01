import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import prepare


def _write_parquet(path: Path, rows: list[dict]) -> None:
    schema = pa.schema(
        [
            pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
            pa.field("question", pa.string()),
            pa.field("answer", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _row(image: bytes, question: str, answer: str, name: str = "image.jpg") -> dict:
    return {"image": {"bytes": image, "path": name}, "question": question, "answer": answer}


def test_prepare_excludes_all_test_images_without_reading_test_labels(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    shared = b"shared-image"
    kept = b"kept-image"
    binary = b"binary-image"
    _write_parquet(
        train,
        [
            _row(shared, "What shared finding?", "mass"),
            _row(kept, "What finding?", "left opacity"),
            _row(kept, "What finding?", "left opacity"),
            _row(binary, "Is it normal?", "No."),
        ],
    )
    # Poison strings demonstrate that test question/answer content is irrelevant.
    _write_parquet(test, [_row(shared, "DO NOT READ", "DO NOT READ")])
    output = tmp_path / "dev.json"
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare(
        train_parquet=train,
        test_parquet=test,
        image_dir=tmp_path / "images",
        output=output,
        manifest_path=manifest_path,
        expected_train_sha256=None,
        expected_test_sha256=None,
    )

    rows = json.loads(output.read_text())
    assert [row["qid"] for row in rows] == ["vqa-rad-train-0001"]
    assert rows[0]["split_role"] == "development_only"
    assert rows[0]["image_sha256"] == hashlib.sha256(kept).hexdigest()
    assert manifest["selection_contract"]["test_columns_read"] == ["image"]
    assert manifest["selection_contract"]["test_answers_read"] is False
    assert manifest["selection_contract"]["test_labels_used_for_selection"] is False
    assert manifest["counts"] == {
        "train_unique_images": 3,
        "test_unique_images": 1,
        "overlap_unique_images": 1,
        "overlap_train_rows": 1,
        "eligible_rows_after_image_exclusion": 3,
        "eligible_unique_images_after_image_exclusion": 2,
        "open_rows": 1,
        "open_unique_images": 1,
        "excluded_binary": {"no": 1},
        "duplicate_open_triplets_removed": 1,
        "max_open_questions_per_image": 1,
        "test_image_overlap_after_filter": 0,
    }
    assert manifest_path.exists()
    assert (tmp_path / "images" / f"{hashlib.sha256(kept).hexdigest()}.jpg").read_bytes() == kept


def test_prepare_rejects_wrong_source_hash(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    _write_parquet(train, [_row(b"a", "What?", "finding")])
    _write_parquet(test, [_row(b"b", "What?", "finding")])

    try:
        prepare(
            train_parquet=train,
            test_parquet=test,
            image_dir=tmp_path / "images",
            output=tmp_path / "dev.json",
            manifest_path=tmp_path / "manifest.json",
            expected_train_sha256="0" * 64,
            expected_test_sha256=None,
        )
    except ValueError as exc:
        assert "unexpected train parquet SHA-256" in str(exc)
    else:
        raise AssertionError("wrong source hash should fail closed")
