"""Integrity and reviewer-blinding audit for the CECD admission pack."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "cecd-admission-pack-integrity-v1"
CLINICAL_DECISION_FIELDS = {
    "support_state_same_supported_refuted_undetermined",
    "lesion_visibility",
    "clinically_interchangeable",
    "unable_to_judge",
    "comments",
}
LANGUAGE_DECISION_FIELDS = {
    "same_clinical_proposition",
    "same_speech_act",
    "same_certainty_demand",
    "same_answer_space",
    "comments",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(pack_dir: Path) -> dict[str, Any]:
    manifest_path = pack_dir / "manifest.json"
    mapping_path = pack_dir / "sealed_mapping.json"
    manifest = json.loads(manifest_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    _require(manifest["version"] == "cecd-blinded-human-admission-pack-v2", "wrong pack version")
    _require(mapping["version"] == manifest["version"], "mapping/manifest version mismatch")

    for relative, expected in manifest["artifact_sha256"].items():
        path = pack_dir / relative
        _require(path.is_file() and sha256_file(path) == expected, f"stale artifact: {relative}")

    pairs = mapping["clinical_pairs"]
    pair_ids = [row["pair_id"] for row in pairs]
    _require(len(pairs) == 252 and len(set(pair_ids)) == 252, "clinical pair count/IDs invalid")
    transforms = Counter(row["transform"] for row in pairs)
    _require(transforms.pop("identity_lossless_duplicate", 0) == 12, "identity controls invalid")
    _require(set(transforms.values()) == {60} and len(transforms) == 4, "primary transform balance invalid")
    _require(all(row["transform_guard"]["clinical_guard_pass"] is True for row in pairs), "failed render guard included")

    expected_pngs: set[Path] = set()
    for row in pairs:
        left = pack_dir / "images" / f"{row['pair_id']}_A.png"
        right = pack_dir / "images" / f"{row['pair_id']}_B.png"
        _require(sha256_file(left) == row["left_png_sha256"], f"left PNG mismatch: {row['pair_id']}")
        _require(sha256_file(right) == row["right_png_sha256"], f"right PNG mismatch: {row['pair_id']}")
        if row["transform"] == "identity_lossless_duplicate":
            _require(row["left_png_sha256"] == row["right_png_sha256"], "identity pair is not lossless")
        expected_pngs.update((left.resolve(), right.resolve()))
    actual_pngs = {path.resolve() for path in (pack_dir / "images").glob("*.png")}
    _require(actual_pngs == expected_pngs, "image directory has missing or extra PNGs")

    for reviewer in (1, 2):
        rows = _csv(pack_dir / f"clinical_reviewer_{reviewer}.csv")
        _require({row["pair_id"] for row in rows} == set(pair_ids) and len(rows) == 252, "clinical sheet IDs invalid")
        for row in rows:
            pair_id = row["pair_id"]
            _require(row["image_A"] == f"images/{pair_id}_A.png", "unblinded A image path")
            _require(row["image_B"] == f"images/{pair_id}_B.png", "unblinded B image path")
            _require(all(not row[field].strip() for field in CLINICAL_DECISION_FIELDS), "review sheet is not blank")
            hidden = next(item for item in pairs if item["pair_id"] == pair_id)
            visible = json.dumps(row, sort_keys=True)
            _require(hidden["image_id"] not in visible, "image ID leaked to reviewer")
            _require(hidden["transform"] not in visible, "transform name leaked to reviewer")

    language_ids = {row["item_id"] for row in mapping["language_items"]}
    _require(len(language_ids) == 8, "language item count invalid")
    for filename in ("clinical_template_reviewer.csv", "language_annotator.csv"):
        rows = _csv(pack_dir / filename)
        _require({row["item_id"] for row in rows} == language_ids and len(rows) == 8, "language sheet IDs invalid")
        _require(all(not row[field].strip() for row in rows for field in LANGUAGE_DECISION_FIELDS), "language sheet is not blank")

    return {
        "protocol_version": VERSION,
        "passed": True,
        "pack": str(pack_dir.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "sealed_mapping_sha256": sha256_file(mapping_path),
        "clinical_pairs": len(pairs),
        "primary_render_pairs": 240,
        "identity_control_pairs": 12,
        "pngs": len(actual_pngs),
        "language_pairs": len(language_ids),
        "review_sheets_blank": True,
        "reviewer_visible_leakage_checks_passed": True,
        "status": "awaiting_independent_human_reviews",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.pack_dir)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
