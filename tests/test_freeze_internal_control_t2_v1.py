import json
from pathlib import Path

from anchor.medeval.freeze_internal_control_t2_v1 import freeze, select_one_per_image


def test_select_one_per_image_is_deterministic_and_label_blind() -> None:
    rows = [
        {"qid": "a1", "image_sha256": "a" * 64, "answer": "first"},
        {"qid": "a2", "image_sha256": "a" * 64, "answer": "second"},
        {"qid": "b1", "image_sha256": "b" * 64, "answer": "third"},
    ]
    first = select_one_per_image(rows, 2, "seed")
    changed = [dict(row, answer="poison") for row in rows]
    second = select_one_per_image(changed, 2, "seed")
    assert [row["qid"] for row in first] == [row["qid"] for row in second]
    assert len({row["image_sha256"] for row in first}) == 2


def test_freeze_hashes_held_out_without_parsing_it(tmp_path: Path) -> None:
    dev = tmp_path / "dev.json"
    audit = tmp_path / "audit.json"
    held_out = tmp_path / "held-out.secret"
    contract = tmp_path / "contract.json"
    output = tmp_path / "pilot.json"
    provenance = tmp_path / "provenance.json"
    dev.write_text(
        json.dumps(
            [
                {"qid": f"q{i}", "image_sha256": f"{i:064x}", "answer": "label"}
                for i in range(3)
            ]
        )
    )
    audit.write_text(json.dumps({"counts": {"test_image_overlap_after_filter": 0}}))
    held_out.write_bytes(b"not JSON and must not be parsed")
    contract.write_text(
        json.dumps({"development": {"pilot_size": 2, "selection_seed": "fixed"}})
    )
    result = freeze(
        development_manifest=dev,
        development_audit=audit,
        held_out_manifest=held_out,
        execution_contract=contract,
        output=output,
        provenance_path=provenance,
    )
    assert result["held_out_manifest_content_parsed"] is False
    assert result["held_out_answers_read_for_selection"] is False
    assert result["pilot_rows"] == result["pilot_unique_images"] == 2
    assert len(json.loads(output.read_text())) == 2
