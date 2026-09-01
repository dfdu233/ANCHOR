#!/usr/bin/env python3
"""Build the blinded human-admission pack required before CECD model scoring.

Derived VinDr images stay outside the repository.  Reviewer sheets never
expose reader votes, transform names, baseline side, image identifiers, or
model outputs.  The sealed mapping is for analysis only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from corrected_sgta.run_cecd_factorial_v1 import (
    BASELINE_VIEW,
    DEFAULT_BBOXES,
    DEFAULT_IMAGE_ROOT,
    FROZEN_FINDINGS,
    FROZEN_VOTES,
    IDENTITY_RENDER_NAME,
    PROMPT_TEMPLATES,
    SCIENCE_RENDER_NAMES,
    build_render_views,
    load_jsonl,
    prompts_for,
    read_dicom_pixels,
    resolve_image,
    selection,
    sha256_file,
)
from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import canonical_json_sha256
from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json


VERSION = "cecd-blinded-human-admission-pack-v2"


def _rank(seed: int, *values: object) -> str:
    return hashlib.sha256(":".join(map(str, (VERSION, seed, *values))).encode()).hexdigest()


def stratified_review_rows(
    rows: Sequence[dict[str, Any]], *, total: int = 60, seed: int = 20260802
) -> list[dict[str, Any]]:
    """Select 15 claims/finding with one rotating 3-case vote stratum."""

    if total != 60:
        raise ValueError("the frozen CECD clinical review uses exactly 60 claims")
    selected: list[dict[str, Any]] = []
    for finding_index, finding in enumerate(FROZEN_FINDINGS):
        reduced_vote = FROZEN_VOTES[finding_index % len(FROZEN_VOTES)]
        for vote in FROZEN_VOTES:
            count = 3 if vote == reduced_vote else 4
            group = [
                row
                for row in rows
                if str(row["finding"]) == finding and int(row["positive_votes"]) == vote
            ]
            group.sort(key=lambda row: _rank(seed, finding, vote, row["image_id"]))
            if len(group) < count:
                raise ValueError(f"insufficient {finding} vote={vote}: {len(group)} < {count}")
            selected.extend(group[:count])
    if len(selected) != total:
        raise RuntimeError(f"stratified selection mismatch: {len(selected)} != {total}")
    keys = [(str(row["image_id"]), str(row["finding"])) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("review selection contains duplicate image-claim rows")
    return sorted(selected, key=lambda row: _rank(seed, row["image_id"], row["finding"]))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_png(image: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lossless, low-compression PNG keeps the blinded pixel content exact while
    # avoiding tens of minutes of CPU-only optimizer search per review pack.
    image.save(path, format="PNG", optimize=False, compress_level=1)
    return sha256_file(path)


def build_pack(output_dir: Path, seed: int) -> dict[str, Any]:
    frozen = selection()
    selected = stratified_review_rows(frozen, seed=seed)
    bbox_rows = load_jsonl(DEFAULT_BBOXES)
    boxes_by_claim = {
        (str(row["image_id"]), str(row["finding"])): list(row.get("boxes", []))
        for row in bbox_rows
    }
    boxes_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in bbox_rows:
        boxes_by_image.setdefault(str(row["image_id"]), []).extend(row.get("boxes", []))

    clinical_rows: list[dict[str, Any]] = []
    sealed_pairs: list[dict[str, Any]] = []
    render_names = [name for name in SCIENCE_RENDER_NAMES if name != BASELINE_VIEW]
    # Twelve exact identity pairs estimate reviewer false-alarm and task fidelity.
    identity_keys = {
        (str(row["image_id"]), str(row["finding"]))
        for row in sorted(
            selected, key=lambda row: _rank(seed, "identity", row["image_id"], row["finding"])
        )[:12]
    }
    cached_id = None
    cached_pixels = None
    for row in selected:
        image_id, finding = str(row["image_id"]), str(row["finding"])
        path = resolve_image(row, DEFAULT_IMAGE_ROOT)
        if image_id != cached_id:
            cached_pixels = read_dicom_pixels(path)
            cached_id = image_id
        views = build_render_views(
            cached_pixels,
            boxes_by_claim.get((image_id, finding), []),
            boxes_by_image.get(image_id, []),
        )
        by_name = {str(view["name"]): view for view in views}
        comparison_names = list(render_names)
        if (image_id, finding) in identity_keys:
            comparison_names.append(IDENTITY_RENDER_NAME)
        for transform in comparison_names:
            pair_id = _rank(seed, "pair", image_id, finding, transform)[:16]
            baseline_left = int(_rank(seed, "side", pair_id), 16) % 2 == 0
            left_name, right_name = (
                (BASELINE_VIEW, transform) if baseline_left else (transform, BASELINE_VIEW)
            )
            left_rel = Path("images") / f"{pair_id}_A.png"
            right_rel = Path("images") / f"{pair_id}_B.png"
            left_hash = _save_png(by_name[left_name]["image"], output_dir / left_rel)
            right_hash = _save_png(by_name[right_name]["image"], output_dir / right_rel)
            clinical_rows.append(
                {
                    "pair_id": pair_id,
                    "image_A": str(left_rel),
                    "image_B": str(right_rel),
                    "finding": finding.replace("_", " "),
                    "support_state_same_supported_refuted_undetermined": "",
                    "lesion_visibility": "",
                    "clinically_interchangeable": "",
                    "unable_to_judge": "",
                    "comments": "",
                }
            )
            sealed_pairs.append(
                {
                    "pair_id": pair_id,
                    "image_id": image_id,
                    "finding": finding,
                    "positive_votes": int(row["positive_votes"]),
                    "transform": transform,
                    "baseline_side": "A" if baseline_left else "B",
                    "left_render": left_name,
                    "right_render": right_name,
                    "left_png_sha256": left_hash,
                    "right_png_sha256": right_hash,
                    "transform_guard": by_name[transform]["audit"],
                }
            )

    clinical_rows.sort(key=lambda row: _rank(seed, "clinical-order", row["pair_id"]))
    clinical_fields = list(clinical_rows[0])
    for reviewer in (1, 2):
        _write_csv(output_dir / f"clinical_reviewer_{reviewer}.csv", clinical_fields, clinical_rows)

    language_rows: list[dict[str, Any]] = []
    sealed_language: list[dict[str, Any]] = []
    for finding in FROZEN_FINDINGS:
        prompts = {item["name"]: item["text"] for item in prompts_for(finding)}
        reference_name = PROMPT_TEMPLATES[0][0]
        for candidate_name, _ in PROMPT_TEMPLATES[1:]:
            item_id = _rank(seed, "language", finding, candidate_name)[:16]
            reference_left = int(_rank(seed, "language-side", item_id), 16) % 2 == 0
            left_name, right_name = (
                (reference_name, candidate_name)
                if reference_left
                else (candidate_name, reference_name)
            )
            language_rows.append(
                {
                    "item_id": item_id,
                    "wording_A": prompts[left_name],
                    "wording_B": prompts[right_name],
                    "same_clinical_proposition": "",
                    "same_speech_act": "",
                    "same_certainty_demand": "",
                    "same_answer_space": "",
                    "comments": "",
                }
            )
            sealed_language.append(
                {
                    "item_id": item_id,
                    "finding": finding,
                    "reference_side": "A" if reference_left else "B",
                    "candidate_prompt": candidate_name,
                }
            )
    language_rows.sort(key=lambda row: _rank(seed, "language-order", row["item_id"]))
    _write_csv(
        output_dir / "clinical_template_reviewer.csv",
        list(language_rows[0]),
        language_rows,
    )
    language_annotator_rows = sorted(
        language_rows, key=lambda row: _rank(seed, "language-annotator-order", row["item_id"])
    )
    _write_csv(
        output_dir / "language_annotator.csv",
        list(language_annotator_rows[0]),
        language_annotator_rows,
    )
    instructions = """# CECD blinded admission review\n\nDo not open `sealed_mapping.json`. Review independently.\n\nClinical image pairs:\n- `support_state_same_supported_refuted_undetermined`: `yes`, `no`, or `unable`.\n- `lesion_visibility`: `unchanged`, `A_clearer`, `B_clearer`, or `unable`.\n- `clinically_interchangeable`: `yes`, `no`, or `unable`.\n- `unable_to_judge`: `yes` or `no`; use `yes` if any primary field is `unable`.\n\nLanguage pairs:\n- Fill each of the four decision fields with `yes`, `no`, or `unable`.\n- Judge proposition, speech act, certainty demand, and answer space separately.\n\nDo not infer a finding from filenames or discuss decisions with the other reviewer.\n"""
    (output_dir / "REVIEW_INSTRUCTIONS.md").write_text(instructions)

    atomic_json(
        output_dir / "sealed_mapping.json",
        {
            "version": VERSION,
            "warning": "Do not provide this file to blinded reviewers.",
            "clinical_pairs": sealed_pairs,
            "language_items": sealed_language,
        },
    )
    selected_path = output_dir / "selected_claims.sealed.jsonl"
    selected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    files = [
        output_dir / "clinical_reviewer_1.csv",
        output_dir / "clinical_reviewer_2.csv",
        output_dir / "clinical_template_reviewer.csv",
        output_dir / "language_annotator.csv",
        output_dir / "REVIEW_INSTRUCTIONS.md",
        output_dir / "sealed_mapping.json",
        selected_path,
    ]
    manifest = {
        "version": VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "status": "awaiting_independent_human_reviews",
        "seed": seed,
        "source_selection_claims": len(frozen),
        "clinical_review_claims": len(selected),
        "primary_render_pairs": len(selected) * len(render_names),
        "identity_control_pairs": len(identity_keys),
        "clinical_pairs_total": len(clinical_rows),
        "language_pairs_total": len(language_rows),
        "reviewers_required": {"clinical": 2, "language": 1, "clinical_template": 1},
        "blinding": {
            "reviewer_hidden": [
                "reader votes", "image id", "transform name", "baseline side", "model outputs"
            ],
            "sealed_mapping": "sealed_mapping.json",
        },
        "decision_rule": (
            "each primary render family is removed if either support-state disagreement or "
            "systematic visibility change exceeds 5%; language wording requires unanimous "
            "same proposition and speech act from the clinical and language admission"
        ),
        "artifact_sha256": {str(path.relative_to(output_dir)): sha256_file(path) for path in files},
        "selection_sha256": canonical_json_sha256(
            [(str(row["image_id"]), str(row["finding"])) for row in selected]
        ),
        "derived_images_location": str((output_dir / "images").resolve()),
        "repository_admission": "derived restricted images must remain outside Git and Git LFS",
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("admission pack output must be a new empty directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build_pack(args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
